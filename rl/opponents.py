"""Opponent policies and league management for self-play training.

The opponent pool intentionally mixes three sources:
- frozen checkpoints from self-play,
- the same checkpoints with small action-logit biases,
- lightweight scripted sparring policies.

None of these change the environment reward. They only broaden the experience
distribution so the learner sees long-horizon and combo-heavy positions.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl.config import (
    LEAGUE_CONFIG,
    LEAGUE_DIR,
    NUM_NTP_REACTIONS,
    NUM_THUMB_OPTIONS,
    NUM_TP_ACTIONS,
    NUM_TP_SKILLS,
    OPPONENT_POOL_CONFIG,
    TOTAL_ACTIONS,
)
from rl.model_utils import load_maskable_ppo
from rl.actions import decode_action


# TP skill indices in TP_SKILL_OPTIONS.  Keeping these index based avoids
# brittle string matching against skill names in saved checkpoints.
SKILL_NUMBER = set(range(5))
SKILL_FLASH = 5
SKILL_CEMENT = 6
SKILL_GUARD = 7
SKILL_CHARGE = 8
SKILL_QUICK = 9
SKILL_SKIP = 10
SKILL_FEINT = 12
SKILL_LOCK = 13
SKILL_COPY = 14
SKILL_STOCK = 15
SKILL_ALL = 16
SKILL_DROP = 17
SKILL_BOOST = 18
SKILL_TIME = 19
SKILL_CHOICE_START = 20

REACTION_NONE = 0
REACTION_COUNTER = 1
REACTION_BLOCK = 2
REACTION_MIRROR = 3


@dataclass(frozen=True)
class OpponentSpec:
    kind: str
    path: str | None = None
    step: int | None = None
    preset: str | None = None
    temperature: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "step": self.step,
            "preset": self.preset,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, spec: dict[str, Any] | "OpponentSpec") -> "OpponentSpec":
        if isinstance(spec, OpponentSpec):
            return spec
        return cls(
            kind=str(spec.get("kind", "model")),
            path=spec.get("path"),
            step=spec.get("step"),
            preset=spec.get("preset"),
            temperature=float(spec.get("temperature", 1.0)),
        )


class _FrozenPolicy:
    """Fixed MaskablePPO policy used inside env subprocesses."""

    def __init__(self, model):
        self.model = model
        self.observation_dim = int(model.observation_space.shape[0])

    def predict(self, obs, action_masks=None, deterministic=False):
        import torch

        with torch.no_grad():
            device = next(self.model.policy.parameters()).device
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            if action_masks is not None:
                mask_tensor = torch.BoolTensor(action_masks).unsqueeze(0).to(device)
                dist = self.model.policy.get_distribution(
                    obs_tensor, action_masks=mask_tensor
                )
                action = dist.get_actions(deterministic=deterministic)
                return action.cpu().numpy().flatten(), None

            action, _ = self.model.predict(obs, deterministic=deterministic)
            return np.array([int(action)]), None


class _BiasedFrozenPolicy(_FrozenPolicy):
    """Frozen policy with mild strategy-specific logit biases."""

    def __init__(self, model, preset: str, temperature: float = 1.0):
        super().__init__(model)
        self.preset = preset
        self.temperature = max(0.05, float(temperature))

    def predict(self, obs, action_masks=None, deterministic=False):
        import torch

        with torch.no_grad():
            device = next(self.model.policy.parameters()).device
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask = np.ones(TOTAL_ACTIONS, dtype=bool)
            if action_masks is not None:
                mask = np.asarray(action_masks, dtype=bool)
                mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(device)
                dist = self.model.policy.get_distribution(
                    obs_tensor, action_masks=mask_tensor
                )
            else:
                dist = self.model.policy.get_distribution(obs_tensor)

            probs = dist.distribution.probs.squeeze(0).cpu().numpy()
            logits = np.log(np.clip(probs, 1e-12, 1.0)) / self.temperature
            logits += _action_bias_vector(self.preset)
            action = _sample_from_logits(logits, mask, deterministic)
            return np.array([action], dtype=np.int64), None


class RuleStrategyPolicy:
    """Lightweight scripted sparring policy.

    These policies are not treated as teachers. They simply generate positions
    that pure self-play may discover too slowly.
    """

    observation_dim = None

    def __init__(self, preset: str):
        self.preset = preset

    def predict_action(self, game_state, actor_key, action_mask, rng=None):
        del game_state, actor_key
        rng = rng or np.random.default_rng()
        mask = np.asarray(action_mask, dtype=bool)
        logits = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        logits += _action_bias_vector(self.preset)
        logits += rng.normal(0.0, 0.03, size=TOTAL_ACTIONS).astype(np.float32)
        return _sample_from_logits(logits, mask, deterministic=False, rng=rng)

    def predict(self, obs, action_masks=None, deterministic=False):
        del obs
        mask = np.asarray(action_masks, dtype=bool)
        logits = _action_bias_vector(self.preset)
        action = _sample_from_logits(logits, mask, deterministic)
        return np.array([action], dtype=np.int64), None


def _sample_from_logits(logits, mask, deterministic=False, rng=None) -> int:
    rng = rng or np.random.default_rng()
    mask = np.asarray(mask, dtype=bool)
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    masked = np.asarray(logits, dtype=np.float64).copy()
    masked[~mask] = -np.inf
    if deterministic:
        return int(np.argmax(masked))
    finite = masked[valid]
    finite = finite - np.max(finite)
    probs = np.exp(finite)
    probs_sum = probs.sum()
    if not np.isfinite(probs_sum) or probs_sum <= 0:
        return int(rng.choice(valid))
    probs = probs / probs_sum
    return int(rng.choice(valid, p=probs))


def _action_bias_vector(preset: str | None) -> np.ndarray:
    bias = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    preset = preset or "balanced"

    def add_skill(skill_idx: int, value: float):
        start = skill_idx * NUM_THUMB_OPTIONS
        bias[start:start + NUM_THUMB_OPTIONS] += value

    def add_skill_range(start_idx: int, end_idx: int, value: float):
        for skill_idx in range(start_idx, end_idx):
            add_skill(skill_idx, value)

    def add_reaction(reaction_idx: int, value: float):
        start = NUM_TP_ACTIONS + reaction_idx * NUM_THUMB_OPTIONS
        bias[start:start + NUM_THUMB_OPTIONS] += value

    if preset == "fast_finish":
        for idx in SKILL_NUMBER:
            add_skill(idx, 0.55)
        add_skill(SKILL_FLASH, 0.65)
        add_skill(SKILL_FEINT, 0.45)
        add_skill(SKILL_CHARGE, 0.30)
        add_reaction(REACTION_COUNTER, 0.35)
    elif preset == "cement_lock_flash":
        add_skill(SKILL_CEMENT, 0.95)
        add_skill(SKILL_LOCK, 0.75)
        add_skill(SKILL_FLASH, 0.85)
        add_skill(SKILL_TIME, 0.35)
        add_reaction(REACTION_COUNTER, 0.20)
    elif preset == "stock_choice":
        add_skill(SKILL_GUARD, 0.45)
        add_skill(SKILL_STOCK, 1.05)
        add_skill_range(SKILL_CHOICE_START, NUM_TP_SKILLS, 0.95)
        add_skill(SKILL_DROP, 0.55)
        add_skill(SKILL_ALL, 0.35)
        add_reaction(REACTION_BLOCK, 0.25)
    elif preset == "counter_block":
        add_skill(SKILL_LOCK, 0.35)
        add_skill(SKILL_FEINT, 0.35)
        add_reaction(REACTION_COUNTER, 1.05)
        add_reaction(REACTION_BLOCK, 0.85)
        add_reaction(REACTION_NONE, -0.25)
    elif preset == "tempo":
        add_skill(SKILL_SKIP, 0.85)
        add_skill(SKILL_GUARD, 0.50)
        add_skill(SKILL_BOOST, 0.55)
        add_skill(SKILL_QUICK, 0.45)
        add_skill(SKILL_COPY, 0.35)
        add_reaction(REACTION_BLOCK, 0.35)
    return bias


def create_opponent_policy(spec: dict[str, Any] | OpponentSpec, device: str = "cpu"):
    spec = OpponentSpec.from_dict(spec)
    if spec.kind == "rule":
        return RuleStrategyPolicy(spec.preset or "balanced")
    if spec.path is None:
        return None
    model = load_maskable_ppo(spec.path, device=device)
    if int(model.action_space.n) != TOTAL_ACTIONS:
        raise ValueError(
            f"Unsupported legacy action space: {int(model.action_space.n)} != {TOTAL_ACTIONS}"
        )
    if spec.kind == "biased":
        return _BiasedFrozenPolicy(
            model,
            preset=spec.preset or "balanced",
            temperature=spec.temperature,
        )
    return _FrozenPolicy(model)


def _create_frozen_policy(model):
    return _FrozenPolicy(model)


class LeagueManager:
    """Manage frozen checkpoints and sample opponent specs."""

    def __init__(self, model_class=None):
        self.model_class = model_class
        self.opponents: list[tuple[str, int]] = []
        self.win_rates = {}

        os.makedirs(LEAGUE_DIR, exist_ok=True)
        self._load_existing_opponents()

    def _load_existing_opponents(self):
        pattern = os.path.join(LEAGUE_DIR, "step_*.zip")
        paths = sorted(glob.glob(pattern))
        self.opponents = []
        for p in paths:
            basename = os.path.basename(p)
            try:
                step = int(basename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            self.opponents.append((p[:-4], step))

    def save_checkpoint(self, model, timestep):
        base_path = os.path.join(LEAGUE_DIR, f"step_{timestep:010d}")
        model.save(base_path)
        self.opponents.append((base_path, timestep))

        max_opps = LEAGUE_CONFIG["max_opponents"]
        if len(self.opponents) > max_opps:
            keep_first = 3
            keep_last = 5
            if len(self.opponents) > keep_first + keep_last:
                middle = self.opponents[keep_first:-keep_last]
                keep_middle = max_opps - keep_first - keep_last
                if len(middle) > keep_middle:
                    kept = sorted(random.sample(middle, keep_middle),
                                  key=lambda x: x[1])
                    to_remove = set(m[0] for m in middle) - set(m[0] for m in kept)
                    for p in to_remove:
                        zip_p = p + ".zip"
                        if os.path.exists(zip_p):
                            os.remove(zip_p)
                    self.opponents = (
                        self.opponents[:keep_first] + kept +
                        self.opponents[-keep_last:]
                    )

    def select_checkpoint(self):
        if not self.opponents:
            return None, None

        config = LEAGUE_CONFIG
        r = random.random()

        if r < config["recent_weight"] and len(self.opponents) >= 1:
            path, step = self.opponents[-1]
        elif (r < config["recent_weight"] + config["initial_weight"]
              and len(self.opponents) >= 2):
            path, step = self.opponents[0]
        else:
            path, step = random.choice(self.opponents)

        return path, step

    def select_opponent(self):
        """Return an opponent spec dict for env subprocesses."""
        pool = OPPONENT_POOL_CONFIG
        has_model = bool(self.opponents)
        weights = [
            pool["model_weight"] if has_model else 0.0,
            pool["biased_weight"] if has_model else 0.0,
            pool["rule_weight"],
        ]
        total = sum(weights)
        if total <= 0:
            return None

        r = random.random() * total
        if r < weights[0]:
            path, step = self.select_checkpoint()
            return OpponentSpec("model", path=path, step=step).as_dict()
        if r < weights[0] + weights[1]:
            path, step = self.select_checkpoint()
            preset = random.choice(pool["biased_presets"])
            return OpponentSpec(
                "biased",
                path=path,
                step=step,
                preset=preset,
                temperature=pool["biased_temperature"],
            ).as_dict()

        preset = random.choice(pool["rule_presets"])
        return OpponentSpec("rule", preset=preset).as_dict()

    def get_opponent_policy(self):
        spec = self.select_opponent()
        if spec is None:
            return None, None
        try:
            policy = create_opponent_policy(spec)
            return policy, spec.get("step")
        except Exception as e:
            print(f"[League] opponent load error: {e}")
            return None, None

    def update_win_rate(self, timestep, wins, total):
        if total > 0:
            self.win_rates[timestep] = wins / total

    @property
    def num_opponents(self):
        return len(self.opponents)

    def get_stats(self):
        return {
            "total_opponents": len(self.opponents),
            "timesteps": [s for _, s in self.opponents],
            "win_rates": dict(self.win_rates),
        }

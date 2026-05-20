"""Gymnasium environment for Complete Yubisuma.

The agent always acts as the **current turn player** (State.me).
NTP reactions are provided by *opponent_policy* (default: random).

When the turn switches, the new state already uses the opponent's perspective
as "me", so the same agent model naturally plays both sides in self-play.

Action space
~~~~~~~~~~~~
Discrete — index into a fixed canonical list of all possible TPActions for
the current config.  Illegal actions are indicated by ``action_masks()``.

Observation space
~~~~~~~~~~~~~~~~~
Box(OBS_SIZE,) of float32 values in [0, 1] — see ``complete_rl.obs``.

Reward
~~~~~~
+1.0 on win, -1.0 on loss, 0.0 at every intermediate step.

Usage
-----
>>> from complete_rl.env import CompleteEnv
>>> from complete_solver import RulesConfig
>>> env = CompleteEnv(config=RulesConfig(enable_mirror=True))
>>> obs, info = env.reset()
>>> mask = env.action_masks()
>>> action = int(np.argmax(mask))          # first legal action
>>> obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required: pip install gymnasium>=0.29"
    ) from exc

from complete_solver import RulesConfig, State, NTPAction, TPAction
from complete_solver import legal_ntp_actions, legal_tp_actions
from complete_solver.constants import (
    ALL,
    ANTI_COUNTER_SKILLS,
    BLOCK,
    BOOST,
    CHOICE,
    COUNTER,
    COPY,
    DROP,
    MIRROR_MAIN,
    MIRROR_PREP,
    NONE,
    NORMAL_SKILLS,
    PASS,
    REFERENCEABLE_SKILLS,
    REVERSI,
    STOCK,
    TIME,
    ULTIMATE_TP_SKILLS,
)
from complete_solver.transition import transition
from complete_rl.obs import OBS_SIZE, encode_state

OpponentPolicy = Callable[[State, RulesConfig], NTPAction]
REWARD_MODES: tuple[str, ...] = ("terminal", "material")
MIXED_NTP_POLICIES: tuple[str, ...] = ("mixed_basic", "weighted_none_counter")


# ── Opponent policies ────────────────────────────────────────────────────

def random_ntp_policy(state: State, config: RulesConfig) -> NTPAction:
    """Uniformly random legal NTP reaction."""
    return random.choice(legal_ntp_actions(state, config))


def none_ntp_policy(state: State, config: RulesConfig) -> NTPAction:
    """Always choose the no-reaction action with the lowest legal thumb."""
    return _first_legal_reaction(state, config, (NONE,))


def counter_first_ntp_policy(state: State, config: RulesConfig) -> NTPAction:
    """Prefer Counter, otherwise fall back to no reaction."""
    return _first_legal_reaction(state, config, (COUNTER, NONE))


def block_first_ntp_policy(state: State, config: RulesConfig) -> NTPAction:
    """Prefer Block, then Counter, then no reaction."""
    return _first_legal_reaction(state, config, (BLOCK, COUNTER, NONE))


def mirror_first_ntp_policy(state: State, config: RulesConfig) -> NTPAction:
    """Prefer Mirror when legal, then Counter, Block, and no reaction."""
    return _first_legal_reaction(state, config, (MIRROR_MAIN, COUNTER, BLOCK, NONE))


NAMED_NTP_POLICIES: dict[str, OpponentPolicy] = {
    "none": none_ntp_policy,
    "counter_first": counter_first_ntp_policy,
    "block_first": block_first_ntp_policy,
    "mirror_first": mirror_first_ntp_policy,
}


def _first_legal_reaction(
    state: State,
    config: RulesConfig,
    reactions: tuple[str, ...],
) -> NTPAction:
    legal = legal_ntp_actions(state, config)
    for reaction in reactions:
        for action in legal:
            if action.reaction == reaction:
                return action
    return legal[0]


# ── Canonical action enumeration ─────────────────────────────────────────

def build_canonical_tp_actions(config: RulesConfig) -> tuple[TPAction, ...]:
    """Return a fixed ordered tuple of all *possible* TP actions for *config*.

    The list is deterministic and independent of game state.  The ALL action
    uses ``all_order=()`` as a placeholder; ``resolve_canonical_action`` fills
    in the actual order at step time.
    """
    actions: list[TPAction] = []

    # PASS (only when skip_phases > 0, but enumerate always)
    actions.append(TPAction(PASS, 0))

    # Number declarations: total 0-4, thumb 0-2
    for total in range(5):
        for thumb in range(3):
            actions.append(TPAction(total, thumb))

    # Normal skills (MIRROR_PREP only when mirror enabled)
    normal_skills = sorted(
        NORMAL_SKILLS if config.enable_mirror else NORMAL_SKILLS - {MIRROR_PREP}
    )
    for skill in normal_skills:
        for thumb in range(3):
            actions.append(TPAction(skill, thumb))

    # Anti-counter skills
    for skill in sorted(ANTI_COUNTER_SKILLS):
        for thumb in range(3):
            actions.append(TPAction(skill, thumb))

    # Reference basics (COPY, DROP, STOCK — not CHOICE or ALL)
    for skill in sorted({COPY, DROP, STOCK}):
        for thumb in range(3):
            actions.append(TPAction(skill, thumb))

    # CHOICE: one entry per referenceable choice target × thumb
    choice_pool = sorted(
        REFERENCEABLE_SKILLS if config.enable_mirror
        else REFERENCEABLE_SKILLS - {MIRROR_PREP}
    )
    for choice in choice_pool:
        for thumb in range(3):
            actions.append(TPAction(CHOICE, thumb, choice=choice))

    # ALL: thumb only (all_order filled dynamically)
    for thumb in range(3):
        actions.append(TPAction(ALL, thumb, all_order=()))

    # Ultimate skills (REVERSI only when reversi enabled)
    ultimate_skills = sorted(
        ULTIMATE_TP_SKILLS if config.enable_reversi
        else ULTIMATE_TP_SKILLS - {REVERSI}
    )
    for skill in ultimate_skills:
        for thumb in range(3):
            actions.append(TPAction(skill, thumb))

    return tuple(actions)


def resolve_canonical_action(canonical: TPAction, state: State) -> TPAction:
    """Convert a canonical (possibly placeholder) action to an executable one.

    Currently only ALL needs dynamic filling (its ``all_order`` depends on
    the current stock).
    """
    if canonical.skill == ALL:
        return TPAction(ALL, canonical.thumb, all_order=tuple(sorted(state.me.stock)))
    return canonical


def build_action_mask(
    canonical: tuple[TPAction, ...],
    state: State,
    config: RulesConfig,
) -> np.ndarray:
    """Return a boolean mask of shape ``(len(canonical),)``.

    True at index *i* iff ``canonical[i]`` corresponds to a legal action in
    *state*.  ALL actions are matched by (skill, thumb) only, ignoring order.
    """
    legal = set(legal_tp_actions(state, config))
    mask = np.zeros(len(canonical), dtype=bool)
    for i, action in enumerate(canonical):
        if action.skill == ALL:
            mask[i] = any(a.skill == ALL and a.thumb == action.thumb for a in legal)
        else:
            mask[i] = action in legal
    return mask


# ── Gymnasium environment ────────────────────────────────────────────────

class CompleteEnv(gym.Env):
    """Gymnasium environment for Complete Yubisuma.

    Parameters
    ----------
    config:
        Which Mirror/Reversi rule variant to play.
    opponent_policy:
        Callable ``(state, config) -> NTPAction`` used to sample NTP
        reactions, or one of the named strings: ``"random"``, ``"none"``,
        ``"counter_first"``, ``"block_first"``, ``"mirror_first"``,
        ``"mixed_basic"``, ``"weighted_none_counter"``.
    max_steps:
        Episode truncation limit (prevents infinite loops in degenerate play).
    reward_mode:
        ``"terminal"`` keeps the original sparse ±1 terminal reward.
        ``"material"`` adds a small shaped reward for hand-count progress.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        config: RulesConfig = RulesConfig(),
        opponent_policy: str | OpponentPolicy = "random",
        max_steps: int = 500,
        reward_mode: str = "terminal",
    ) -> None:
        super().__init__()
        self.config = config
        self.max_steps = max_steps
        if reward_mode not in REWARD_MODES:
            valid_modes = ", ".join(REWARD_MODES)
            raise ValueError(f"reward_mode must be one of {valid_modes}; got {reward_mode!r}")
        self.reward_mode = reward_mode

        if opponent_policy == "random":
            self._ntp_policy: OpponentPolicy = self._seeded_random_ntp_policy
        elif opponent_policy == "mixed_basic":
            self._ntp_policy = self._seeded_mixed_basic_ntp_policy
        elif opponent_policy == "weighted_none_counter":
            self._ntp_policy = self._seeded_weighted_none_counter_ntp_policy
        elif isinstance(opponent_policy, str) and opponent_policy in NAMED_NTP_POLICIES:
            self._ntp_policy = NAMED_NTP_POLICIES[opponent_policy]
        elif callable(opponent_policy):
            self._ntp_policy = opponent_policy
        else:
            valid = ", ".join(["random", *MIXED_NTP_POLICIES, *sorted(NAMED_NTP_POLICIES)])
            raise ValueError(
                "opponent_policy must be a callable or one of "
                f"{valid}; got {opponent_policy!r}"
            )
        self._rng = random.Random()

        self._canonical: tuple[TPAction, ...] = build_canonical_tp_actions(config)
        n_actions = len(self._canonical)

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        self._state: State = State()
        self._steps: int = 0

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._state = State()
        self._steps = 0
        return encode_state(self._state), {}

    def step(
        self, action_idx: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        canonical = self._canonical[int(action_idx)]
        tp_action = resolve_canonical_action(canonical, self._state)
        ntp_action = self._ntp_policy(self._state, self.config)
        before_state = self._state

        result = transition(self._state, tp_action, ntp_action, self.config)
        self._steps += 1
        truncated = self._steps >= self.max_steps

        if result.terminal_reward is not None:
            obs = encode_state(self._state)   # return obs of terminal state
            return obs, float(result.terminal_reward), True, truncated, {
                "events": result.events,
                "reward_mode": self.reward_mode,
            }

        assert result.next_state is not None
        self._state = result.next_state
        obs = encode_state(self._state)
        shaped_reward = self._intermediate_reward(
            before_state,
            self._state,
            result.same_turn_player,
        )
        return obs, shaped_reward, False, truncated, {
            "same_turn_player": result.same_turn_player,
            "events": result.events,
            "reward_mode": self.reward_mode,
        }

    def action_masks(self) -> np.ndarray:
        """Boolean action mask compatible with ``sb3_contrib.MaskablePPO``.

        Returns a 1-D bool array of length ``n_actions``; True at index *i*
        means action *i* is legal in the current state.
        """
        return build_action_mask(self._canonical, self._state, self.config)

    def _seeded_random_ntp_policy(self, state: State, config: RulesConfig) -> NTPAction:
        """Uniform random NTP reaction tied to Gymnasium reset(seed=...)."""
        return self._rng.choice(legal_ntp_actions(state, config))

    def _seeded_mixed_basic_ntp_policy(self, state: State, config: RulesConfig) -> NTPAction:
        """Randomly choose among simple deterministic NTP policies each reaction."""
        policy_names = ["random", "none", "counter_first", "block_first"]
        if config.enable_mirror:
            policy_names.append("mirror_first")
        return self._choose_named_ntp_policy(policy_names, state, config)

    def _seeded_weighted_none_counter_ntp_policy(
        self,
        state: State,
        config: RulesConfig,
    ) -> NTPAction:
        """Weighted mix that stresses no-reaction and counter-heavy evaluation."""
        policy_names = [
            "none",
            "none",
            "counter_first",
            "counter_first",
            "block_first",
            "random",
        ]
        if config.enable_mirror:
            policy_names.append("mirror_first")
        return self._choose_named_ntp_policy(policy_names, state, config)

    def _choose_named_ntp_policy(
        self,
        policy_names: list[str],
        state: State,
        config: RulesConfig,
    ) -> NTPAction:
        chosen = self._rng.choice(policy_names)
        if chosen == "random":
            return self._seeded_random_ntp_policy(state, config)
        return NAMED_NTP_POLICIES[chosen](state, config)

    def _intermediate_reward(
        self,
        before: State,
        after: State,
        same_turn_player: bool,
    ) -> float:
        if self.reward_mode == "terminal":
            return 0.0

        after_actor = after.me if same_turn_player else after.opp
        after_opponent = after.opp if same_turn_player else after.me
        before_advantage = before.opp.hands - before.me.hands
        after_advantage = after_opponent.hands - after_actor.hands
        reward = 0.05 * (after_advantage - before_advantage)
        if after_actor.hands < before.me.hands:
            reward -= 0.01
        if after_opponent.hands < before.opp.hands:
            reward += 0.01
        return float(np.clip(reward, -0.1, 0.1))

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        """Current game state (read-only by convention)."""
        return self._state

    @property
    def n_actions(self) -> int:
        """Size of the action space for this config."""
        return len(self._canonical)

    def canonical_action(self, idx: int) -> TPAction:
        """Return the canonical TPAction at position *idx* (before order fill-in)."""
        return self._canonical[idx]

    def __repr__(self) -> str:
        mirror = "ON" if self.config.enable_mirror else "OFF"
        reversi = "ON" if self.config.enable_reversi else "OFF"
        return (
            f"CompleteEnv(mirror={mirror}, reversi={reversi}, "
            f"reward_mode={self.reward_mode}, n_actions={self.n_actions}, obs={OBS_SIZE})"
        )

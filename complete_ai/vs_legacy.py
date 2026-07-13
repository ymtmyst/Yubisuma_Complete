"""Head-to-head: new search agent vs legacy MaskablePPO models (N5b).

The legacy architecture only ever made TP declarations (the training env
supplied NTP reactions from a fixed policy), so a legacy "seat" here is:
TP = the PPO policy (masked, stochastic), NTP = a scripted reaction style.
We run several reaction styles and report each — quoting the legacy side's
BEST style is deliberately generous to the old model.

The legacy obs includes a 4-slot reaction history; we feed it every NTP
reaction that occurs in the game (both seats'), matching the training env's
bookkeeping.

Run:  python -m complete_ai.vs_legacy
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from complete_rl.env import (
    build_action_mask,
    build_canonical_tp_actions,
    resolve_canonical_action,
)
from complete_rl.obs import N_REACTION_HISTORY, encode_state
from complete_solver.actions import RulesConfig
from complete_solver.packed_engine import (
    code_to_ntp_action,
    ntp_action_to_code,
    pack_state,
    step,
    tp_action_to_code,
    unpack_state,
)
from complete_solver.state import initial_state

from .agents import ScriptedAgent, SearchAgent
from .batched_search import BatchedSearcher, _FULL_MASK
from .generation_loop import load_model

CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)

LEGACY_MODELS = {
    "episode_mixed(旧best1)":
        "results/_legacy/ppo_runs/maskable_ppo_bc_standard_episode_mixed/maskable_ppo_complete.zip",
    "depth4_proper(旧best2)":
        "results/_legacy/ppo_runs/maskable_ppo_bc_depth4_standard_proper/maskable_ppo_complete.zip",
}
REACTION_STYLES = ("random", "counter", "block")


class LegacySeat:
    """Legacy PPO declarations + scripted reactions, with obs history."""

    def __init__(self, ppo_model, reaction_style: str, rng: np.random.Generator):
        self.ppo = ppo_model
        self.reactor = ScriptedAgent(reaction_style, rng)
        self.canonical = build_canonical_tp_actions(CONFIG)
        self.history: list[str] = []

    def record_reaction(self, ntp_code: int) -> None:
        reaction = code_to_ntp_action(ntp_code).reaction
        self.history.insert(0, reaction)
        if len(self.history) > N_REACTION_HISTORY:
            self.history.pop()

    def reset(self) -> None:
        self.history = []

    def tp_action(self, lane0: int, lane1: int) -> int:
        state = unpack_state(lane0, lane1)
        obs = encode_state(state, tuple(self.history))
        mask = build_action_mask(self.canonical, state, CONFIG)
        action_idx, _ = self.ppo.predict(
            obs, action_masks=mask, deterministic=False
        )
        tp = resolve_canonical_action(self.canonical[int(action_idx)], state)
        return tp_action_to_code(tp)

    def ntp_action(self, lane0: int, lane1: int) -> int:
        return self.reactor.ntp_action(lane0, lane1)


def play_vs_legacy(agent, legacy: LegacySeat, n_games: int, max_plies: int = 120,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    init0, init1 = pack_state(initial_state())
    wins = [0, 0]  # [agent, legacy]
    truncations = 0
    for game in range(n_games):
        legacy.reset()
        mover = game % 2  # 0 = agent moves first
        lane0, lane1 = np.int64(init0), np.int64(init1)
        for _ in range(max_plies):
            if mover == 0:
                tp_code = agent.tp_action(int(lane0), int(lane1))
                ntp_code = legacy.ntp_action(int(lane0), int(lane1))
            else:
                tp_code = legacy.tp_action(int(lane0), int(lane1))
                ntp_code = agent.ntp_action(int(lane0), int(lane1))
            legacy.record_reaction(ntp_code)
            child0, child1, status, reward = step(
                lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
            )
            if status == 2:
                winner = mover if reward > 0 else 1 - mover
                wins[winner] += 1
                break
            if status == 0:
                mover = 1 - mover
            lane0, lane1 = child0, child1
        else:
            truncations += 1
    decided = wins[0] + wins[1]
    return {
        "games": n_games,
        "agent_wins": wins[0],
        "legacy_wins": wins[1],
        "truncations": truncations,
        "agent_winrate": wins[0] / decided if decided else 0.0,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path("models/value_latest.pt"), device)
    searcher = BatchedSearcher(model, device, prune_stock=True)
    agent = SearchAgent(searcher, np.random.default_rng(11))

    try:
        from sb3_contrib import MaskablePPO
        loader = MaskablePPO
    except ImportError as exc:
        raise SystemExit(f"sb3_contrib unavailable: {exc}")

    results: dict = {}
    for name, path in LEGACY_MODELS.items():
        if not Path(path).exists():
            print(f"SKIP {name}: {path} not found", flush=True)
            continue
        ppo = loader.load(path, device="cpu")
        results[name] = {}
        for style in REACTION_STYLES:
            legacy = LegacySeat(ppo, style, np.random.default_rng(77))
            t0 = time.perf_counter()
            outcome = play_vs_legacy(agent, legacy, n_games=200, seed=5)
            outcome["seconds"] = time.perf_counter() - t0
            results[name][style] = outcome
            print(f"{name} / reactions={style}: {json.dumps(outcome)}", flush=True)

    out = Path("data/n5_vs_legacy.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

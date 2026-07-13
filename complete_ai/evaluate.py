"""Fixed-baseline evaluation harness (N5).

Plays the search agent against clearly-specified scripted opponents, both
seatings, and reports win rates. Turn accounting follows the engine: the
terminal reward is from the current mover's perspective.

Run:  python -m complete_ai.evaluate --model models/value_latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from complete_solver.packed_engine import pack_state, step
from complete_solver.state import initial_state

from .agents import ScriptedAgent, SearchAgent
from .batched_search import BatchedSearcher, _FULL_MASK
from .generation_loop import load_model

BASELINES = ("random", "none", "counter", "block")


def play_games(agent, opponent, n_games: int, max_plies: int = 120,
               seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    init0, init1 = pack_state(initial_state())
    players = (agent, opponent)
    wins = [0, 0]
    truncations = 0
    plies_total = 0

    for game in range(n_games):
        mover = game % 2
        lane0, lane1 = np.int64(init0), np.int64(init1)
        for _ in range(max_plies):
            tp_code = players[mover].tp_action(int(lane0), int(lane1))
            ntp_code = players[1 - mover].ntp_action(int(lane0), int(lane1))
            child0, child1, status, reward = step(
                lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
            )
            plies_total += 1
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
        "opponent_wins": wins[1],
        "truncations": truncations,
        "agent_winrate": wins[0] / decided if decided else 0.0,
        "mean_plies": plies_total / n_games,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/value_latest.pt")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--out", default="data/n5_baseline_eval.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)
    searcher = BatchedSearcher(model, device, prune_stock=True)
    rng = np.random.default_rng(7)
    agent = SearchAgent(searcher, rng)

    results = {}
    for style in BASELINES:
        opponent = ScriptedAgent(style, np.random.default_rng(1000 + hash(style) % 1000))
        t0 = time.perf_counter()
        outcome = play_games(agent, opponent, n_games=args.games, seed=42)
        outcome["seconds"] = time.perf_counter() - t0
        results[style] = outcome
        print(f"vs {style}: {json.dumps(outcome)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "results": results}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

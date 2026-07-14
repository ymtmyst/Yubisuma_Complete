"""Litmus comparison: early-cement > late-cement across two models.

Designer litmus (2026-07-13): cement is STRONGER early (its payoff is ~10
plies away). If the opening-cement rate exceeds the endgame-cement rate, long-
horizon credit assignment has started working. Prints per-phase cement (and the
other watch skills) for two models side by side.

Run:  python -m complete_ai.n7a_litmus --a models/value_latest.pt --b models/value_gvi_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .agents import SearchAgent
from .batched_search import BatchedSearcher
from .generation_loop import load_model
from .policy_report import PHASES, _WATCH_SKILLS, collect_play_stats


def phase_skill_pct(stats):
    out = {}
    for p, counter in enumerate(stats["tp_counts"]):
        total = sum(counter.values()) or 1
        out[p] = {s: 100.0 * counter.get(s, 0) / total for s in _WATCH_SKILLS}
    return out


def run(model_path: str, device: str, games: int):
    model = load_model(Path(model_path), device)
    agent = SearchAgent(BatchedSearcher(model, device, prune_stock=True),
                        np.random.default_rng(3))
    stats = collect_play_stats(agent, n_games=games)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="models/value_latest.pt")
    parser.add_argument("--b", default="models/value_gvi_latest.pt")
    parser.add_argument("--a-name", default="depth")
    parser.add_argument("--b-name", default="graph-vi")
    parser.add_argument("--models", default=None,
                        help="override a/b with N models: 'path:name,path:name,...'")
    parser.add_argument("--games", type=int, default=600)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.models:
        specs = [(s.split(":")[0], s.split(":")[1]) for s in args.models.split(",")]
    else:
        specs = [(args.a, args.a_name), (args.b, args.b_name)]

    results = []
    for path, name in specs:
        stats = run(path, device, args.games)
        results.append((name, stats, phase_skill_pct(stats)))

    for name, stats, pct in results:
        print(f"\n════ {name}  (mean_plies={stats['mean_plies']:.1f}) ════")
        print(f"{'skill':>8} | " + " | ".join(f"{PHASES[p]:>12}" for p in range(3)))
        for s in _WATCH_SKILLS:
            print(f"{s:>8} | " + " | ".join(f"{pct[p][s]:>11.2f}%" for p in range(3)))
        cem = pct[0]["セメント"], pct[2]["セメント"]
        verdict = "✓ 序盤>終盤 (長期学習の兆候)" if cem[0] > cem[1] else "✗ 未逆転"
        print(f"  litmus セメント: 序盤 {cem[0]:.2f}% vs 終盤 {cem[1]:.2f}%  → {verdict}")

    print("\n──── watch-skill SELECTED/LEGAL (深いほど長期スキルを活用) ────")
    print(f"{'skill':>8} | " + " | ".join(f"{n:>14}" for n, _, _ in results))
    for s in _WATCH_SKILLS:
        cells = []
        for _, stats, _ in results:
            lg = stats["watch_legal"].get(s, 0)
            r = 100.0 * stats["watch_selected"].get(s, 0) / lg if lg else 0.0
            cells.append(f"{r:>13.2f}%")
        print(f"{s:>8} | " + " | ".join(cells))


if __name__ == "__main__":
    main()

"""N7-C feasibility: does selective deepening buy effective depth cheaply?

For a set of real self-play states, compares deepened-cell counts and values
for uniform depth-d (tau<0) vs support-restricted selective deepening at a few
tau. The thesis: selective reaches a larger effective depth for far fewer
deepened cells, while its value stays close to the uniform value of the SAME
depth (support pruning barely moves the equilibrium value).

Run:  python -m complete_ai.n7c_feasibility [--states 40] [--depth 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .batched_search import BatchedSearcher
from .generation_loop import load_model
from .selfplay import run_selfplay
from .selective_search import SelectiveSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=40)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--model", default="models/value_v0.pt")
    parser.add_argument("--taus", default="0.0,0.02,0.05")
    parser.add_argument("--skip-uniform", action="store_true",
                        help="skip the (expensive) uniform depth-d baseline")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)
    searcher = BatchedSearcher(model, device, prune_stock=False)
    sel = SelectiveSearcher(model, device, prune=False)

    # Sample real decision states from a little self-play.
    result = run_selfplay(searcher, n_games=60, epsilon=0.15, seed=3)
    k0, k1 = result["keys0"], result["keys1"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(k0), size=min(args.states, len(k0)), replace=False)
    states = [(int(k0[i]), int(k1[i])) for i in idx]
    taus = [float(t) for t in args.taus.split(",")]

    def run(tau: float):
        cells, verr, t0 = 0, [], time.perf_counter()
        vals = []
        for s0, s1 in states:
            sel.reset_stats()
            v = sel.value(s0, s1, depth=args.depth, tau=tau)
            cells += sel.stats["deep_cells"]
            vals.append(v)
        return np.array(vals), cells, time.perf_counter() - t0

    print(f"device={device}  states={len(states)}  depth={args.depth}", flush=True)
    if args.skip_uniform:
        uni_vals, uni_cells = None, 0
        print("(uniform baseline skipped)")
        print(f"{'tau':>6}{'deep_cells':>12}{'sec':>8}")
        for tau in taus:
            vals, cells, t = run(tau)
            print(f"{tau:>6.3f}{cells:>12,}{t:>8.1f}")
        return
    uni_vals, uni_cells, uni_t = run(-1.0)
    print(f"\nuniform depth-{args.depth}: deep_cells={uni_cells:,}  {uni_t:.1f}s")
    print(f"{'tau':>6}{'deep_cells':>12}{'cells/uni':>10}"
          f"{'mean|Δv|':>10}{'max|Δv|':>10}{'sec':>8}")
    for tau in taus:
        vals, cells, t = run(tau)
        dv = np.abs(vals - uni_vals)
        print(f"{tau:>6.3f}{cells:>12,}{cells/max(uni_cells,1):>10.3f}"
              f"{dv.mean():>10.4f}{dv.max():>10.4f}{t:>8.1f}")


if __name__ == "__main__":
    main()

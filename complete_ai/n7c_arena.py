"""N7-C: does selective deeper acting beat the depth-2 agent (same net)?

Both sides share ONE value net, so any win margin isolates the value of reading
the critical lines deeper (selective depth-d) over the current depth-2 search.
A positive margin = more accurate play for free (no new training) — directly on
the "closer to the true optimum" axis the project now prioritises.

Run:  python -m complete_ai.n7c_arena --model models/value_gvi_latest.pt --depth 4 --games 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from .arena import play_match
from .batched_search import BatchedSearcher
from .generation_loop import load_model
from .selective_search import SelectiveSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/value_gvi_latest.pt")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--games", type=int, default=60)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)

    depth2 = BatchedSearcher(model, device, prune_stock=True)
    sel = SelectiveSearcher(model, device, prune=True,
                            depth=args.depth, tau=args.tau)

    print(f"model={args.model}  selective depth={args.depth} tau={args.tau}  "
          f"games={args.games}", flush=True)
    t0 = time.perf_counter()
    # selective as A, depth-2 as B.
    match = play_match(sel, depth2, n_games=args.games, seed=1)
    dt = time.perf_counter() - t0
    wr = match["winrate_a"]
    print(f"\nselective(A) vs depth-2(B): {match}")
    print(f"selective win-rate: {wr:.3f}   ({dt:.0f}s, "
          f"{len(sel._solve_cache)} states cached)")
    verdict = ("selective reads more accurately (stronger)" if wr > 0.52 else
               "≈ parity" if wr >= 0.48 else "selective WORSE — investigate")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()

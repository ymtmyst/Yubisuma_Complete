"""Run graph-BR exact exploitability on a frozen model. Sanity vs PPO.

Run:  python -m complete_ai.n7_graph_br --model models/value_gvi_latest.pt --cap 30000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from .batched_search import BatchedSearcher
from .generation_loop import load_model
from .graph_br import enumerate_br, solve_br
from .graph_br_fast import solve_br_njit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/value_gvi_latest.pt")
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--support-eps", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--enum-only", action="store_true",
                    help="measure reachable-set size only (skip the VI)")
    ap.add_argument("--no-endgame", action="store_true",
                    help="disable the exact endgame cap (old un-closing behavior)")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="depth-limited BR: cap non-endgame states past this BFS "
                         "depth with the frozen agent's own value (under-counts "
                         "the attacker; raise until the estimate plateaus).")
    ap.add_argument("--engine", choices=("python", "njit"), default="njit",
                    help="VI backend: 'python' = reference graph_br.solve_br, "
                         "'njit' = compiled graph_br_fast.solve_br_njit (default).")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)

    endgame = None
    if not args.no_endgame:
        from .endgame_table import load_endgame_tablebase
        endgame = load_endgame_tablebase()

    # Freeze the ACTUAL deployed agent: pincer search (exact endgame leaves) when
    # the tablebase is present, so the measured exploitability is of the agent we
    # ship, and the A0 boundary is played optimally (hence exactly capped).
    if endgame is not None:
        from .endgame_table import PincerSearcher
        searcher = PincerSearcher(model, device, endgame, prune_stock=True)
        print(f"endgame cap ON ({len(endgame):,} exact states) + pincer frozen "
              f"agent", flush=True)
    else:
        searcher = BatchedSearcher(model, device, prune_stock=True)

    print(f"model={args.model} cap={args.cap} eps={args.support_eps}", flush=True)
    t0 = time.perf_counter()
    data = enumerate_br(searcher, cap=args.cap, support_eps=args.support_eps,
                        endgame=endgame, max_depth=args.max_depth)
    t1 = time.perf_counter()
    if args.enum_only:
        print(f"\nstates={data['n']}  hit_cap={data['n'] >= args.cap}  "
              f"enum={t1-t0:.0f}s (VI skipped)")
        return
    vi = solve_br_njit if args.engine == "njit" else solve_br
    print(f"VI engine: {args.engine}", flush=True)
    res = vi(data, gamma=args.gamma)
    t2 = time.perf_counter()
    hit_cap = data["n"] >= args.cap
    print(f"\nstates={data['n']}  hit_cap={hit_cap}  "
          f"enum={t1-t0:.0f}s vi={t2-t1:.0f}s")
    print(f"attacker_value seat0={res['attacker_value_seat0']:.4f} "
          f"seat1={res['attacker_value_seat1']:.4f} "
          f"mean={res['attacker_value']:.4f}")
    print(f"EXACT attacker win-rate ≈ {res['attacker_winrate']:.4f}  "
          f"(sweeps {res['sweeps']}, converged {res['converged']})")
    print("sanity: graph-BR (exact) should be ≥ PPO lower bound "
          "(gvi 0.44 / depth 0.54). hit_cap=True ⇒ under-estimate.")


if __name__ == "__main__":
    main()

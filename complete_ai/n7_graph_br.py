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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/value_gvi_latest.pt")
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--support-eps", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--enum-only", action="store_true",
                    help="measure reachable-set size only (skip the VI)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)
    searcher = BatchedSearcher(model, device, prune_stock=True)

    print(f"model={args.model} cap={args.cap} eps={args.support_eps}", flush=True)
    t0 = time.perf_counter()
    data = enumerate_br(searcher, cap=args.cap, support_eps=args.support_eps)
    t1 = time.perf_counter()
    if args.enum_only:
        print(f"\nstates={data['n']}  hit_cap={data['n'] >= args.cap}  "
              f"enum={t1-t0:.0f}s (VI skipped)")
        return
    res = solve_br(data, gamma=args.gamma)
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

"""N7-A A6: small-scale feasibility measurement of the graph-VI teacher.

Seeds a sub-graph with a random SUBSET of the exact A0 universe (so every seed
has known ground truth), closes the frontier with the value net, and reports:

- build cost, frontier fraction, VI sweeps/time, convergence;
- accuracy vs A0 exact truth for BOTH the raw net and the graph-VI teacher.

The key question: does propagating structure across the sampled sub-graph make
the interior teacher values *closer to exact* than the raw net that seeds the
boundary? If yes, graph-VI adds long-horizon signal the net alone lacks.

Run:  python -m complete_ai.n7a_feasibility [--seeds 5000] [--model models/value_v0.pt]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from complete_solver.packed_vi import PackedEndgameDB, alphabet_to_mask

from .graph_teacher import graph_vi_teacher, net_values
from .generation_loop import load_model

A0_DB = Path("data/endgame_h11_A0.npz")


def _corr_stats(pred: np.ndarray, exact: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    diff = np.abs(pred - exact)
    return {
        "spearman": float(spearmanr(pred, exact).statistic),
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "p99": float(np.quantile(diff, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5000)
    parser.add_argument("--model", default="models/value_v0.pt")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}", flush=True)
    model = load_model(Path(args.model), device)

    db = PackedEndgameDB.load(A0_DB)
    # A0 = stockless universe: alphabet ∅ ⇒ mask 0, solve_universe used max_stock 99.
    mask = alphabet_to_mask(frozenset())
    max_stock = 99
    assert db.alphabet_mask == mask, (db.alphabet_mask, mask)

    rng = np.random.default_rng(args.seed)
    n = len(db)
    k = min(args.seeds, n)
    sel = rng.choice(n, size=k, replace=False)
    seed0 = np.ascontiguousarray(db.keys0[sel], dtype=np.int64)
    seed1 = np.ascontiguousarray(db.keys1[sel], dtype=np.int64)
    exact = db.values[sel].astype(np.float64)

    # Warm up numba on a tiny call so the timed run measures steady state.
    graph_vi_teacher(model, device, seed0[:64], seed1[:64],
                     alphabet_mask=mask, max_stock=max_stock, gamma=args.gamma,
                     omega=args.omega)

    print(f"omega={args.omega}", flush=True)
    t0 = time.perf_counter()
    values, tab, info = graph_vi_teacher(
        model, device, seed0, seed1,
        alphabet_mask=mask, max_stock=max_stock, gamma=args.gamma,
        omega=args.omega, verbose=True,
    )
    wall = time.perf_counter() - t0

    raw = net_values(model, device, seed0, seed1)

    print("\n──────── sub-graph ────────")
    print(f"seeds (interior)   : {tab.n_seed:,}")
    print(f"frontier           : {tab.n_front:,}  "
          f"({tab.frontier_fraction*100:.1f}% of total)")
    print(f"cells              : {len(tab.child_idx):,}")
    print(f"VI sweeps          : {info['iterations']}  "
          f"(repairs {info['repairs']}, converged {info['converged']}, "
          f"stalled {info['stalled']}, max_delta {info['max_delta']:.2e})")
    print(f"VI time            : {info['vi_seconds']:.2f}s   wall {wall:.2f}s")

    print("\n──────── accuracy vs A0 exact (interior seeds) ────────")
    raw_s = _corr_stats(raw, exact)
    vi_s = _corr_stats(values, exact)
    print(f"{'':14s}{'spearman':>10s}{'mean_abs':>10s}{'p99':>10s}{'max_abs':>10s}")
    print(f"{'raw net':14s}{raw_s['spearman']:>10.4f}{raw_s['mean_abs']:>10.4f}"
          f"{raw_s['p99']:>10.4f}{raw_s['max_abs']:>10.4f}")
    print(f"{'graph-VI':14s}{vi_s['spearman']:>10.4f}{vi_s['mean_abs']:>10.4f}"
          f"{vi_s['p99']:>10.4f}{vi_s['max_abs']:>10.4f}")

    d_sp = vi_s["spearman"] - raw_s["spearman"]
    d_mae = raw_s["mean_abs"] - vi_s["mean_abs"]
    print(f"\nΔ spearman (VI−raw): {d_sp:+.4f}   "
          f"Δ mean_abs (raw−VI, +=better): {d_mae:+.4f}")
    verdict = "graph-VI improves on the raw net" if (d_sp > 0 and d_mae > 0) \
        else "no clear improvement — inspect boundary/scale"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()

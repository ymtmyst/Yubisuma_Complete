"""Regime-sliced accuracy: is the net WORSE in タイム/ロック regimes?

Designer hypothesis (2026-07-14): タイム (and lock/cement/stock) discretely
change the game's structure — values of other skills flip conditionally. If
true, the net (trained almost exclusively on non-タイム self-play data: タイム
usage is 0.00%) should be markedly less accurate on タイム-active states.

Test: the A0 exact DB ((1,1) stockless universe from ``h11_root``, ultimates
NOT pre-spent) contains タイム-active states with exact values. Split the
slice by the packed time/lock bits and compare net-vs-exact rank correlation
per regime. A large gap = the regime-blindness hypothesis, quantified.

Packed player-word layout (complete_solver/packed_engine.py):
  lock_active: bit 9,  used_ultimate: bit 14,  time_active: bit 15.

Run:  python -m complete_ai.n7_regime_slices --model models/value_gvi_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from complete_solver.packed_vi import PackedEndgameDB

from .features import features_from_lanes
from .generation_loop import load_model

A0_DB = Path("data/endgame_h11_A0.npz")

_TIME_BIT = 1 << 15
_LOCK_BIT = 1 << 9
_ULT_BIT = 1 << 14


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def report(name: str, mask: np.ndarray, pred: np.ndarray, exact: np.ndarray):
    n = int(mask.sum())
    if n < 50:
        print(f"{name:>28}: n={n} (too few)")
        return
    s = spearman(pred[mask], exact[mask])
    mae = float(np.abs(pred[mask] - exact[mask]).mean())
    print(f"{name:>28}: n={n:>7,}  spearman={s:.4f}  mean|err|={mae:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/value_gvi_latest.pt")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)
    db = PackedEndgameDB.load(A0_DB)
    k0 = np.ascontiguousarray(db.keys0)
    k1 = np.ascontiguousarray(db.keys1)
    exact = db.values

    feats = features_from_lanes(k0, k1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(feats), 131072):
            chunk = torch.from_numpy(feats[i:i + 131072]).to(device)
            preds.append(model(chunk).float().cpu().numpy().ravel())
    pred = np.concatenate(preds)

    time_any = ((k0 & _TIME_BIT) != 0) | ((k1 & _TIME_BIT) != 0)
    lock_any = ((k0 & _LOCK_BIT) != 0) | ((k1 & _LOCK_BIT) != 0)
    ult_avail = ((k0 & _ULT_BIT) == 0) | ((k1 & _ULT_BIT) == 0)

    print(f"model={args.model}  A0 slice: {len(exact):,} states "
          f"(time-active {int(time_any.sum()):,}, "
          f"lock-active {int(lock_any.sum()):,})\n")
    report("ALL", np.ones(len(exact), bool), pred, exact)
    report("time INACTIVE", ~time_any, pred, exact)
    report("time ACTIVE (regime)", time_any, pred, exact)
    report("lock ACTIVE", lock_any, pred, exact)
    report("time+lock ACTIVE", time_any & lock_any, pred, exact)
    report("ultimate still available", ult_avail, pred, exact)
    print("\nInterpretation: a large spearman/|err| gap between 'time ACTIVE'"
          "\nand 'time INACTIVE' confirms the regime-blindness hypothesis —"
          "\nthe net is systematically less reliable inside the タイム regime.")


if __name__ == "__main__":
    main()

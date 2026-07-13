"""Evaluate the trained v0 value net against ground-truth yardsticks (N3).

Runs as its own process (isolated from training) with staged, flushed logs so
any native crash is attributable to a specific stage.

Yardstick 1 — A0 endgame slice: values are EXACT for the stockless variant
(not the full game — stock options add value), so this is a proxy truth; the
rank correlation column is the most meaningful.
Yardstick 2 — depth-3 full-rules LP-backup values on held-out states (deeper
than the depth-2 training horizon).

Run:  python -m complete_ai.eval_v0  →  models/value_v0_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from complete_solver.packed_vi import PackedEndgameDB

from .features import features_from_lanes
from .packed_eval import depth3_values, material_leaf_bits
from .train_v0 import DATASET, MODEL_OUT, ValueNet, material_values

A0_DB = Path("data/endgame_h11_A0.npz")
EVAL_OUT = Path("models/value_v0_eval.json")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    payload = torch.load(MODEL_OUT, map_location=device, weights_only=True)
    model = ValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print("model loaded", flush=True)

    data = np.load(DATASET)
    keys0 = data["keys0"]
    keys1 = data["keys1"]
    gamma = float(data["gamma"][0])
    rng = np.random.default_rng(0)
    # Same validation split as training (same seed and order).
    order = rng.permutation(len(keys0))
    n_val = max(20_000, len(keys0) // 20)
    val_idx = order[:n_val]

    metrics: dict = {}

    def net_values(k0: np.ndarray, k1: np.ndarray) -> np.ndarray:
        feats = features_from_lanes(k0, k1)
        with torch.no_grad():
            out = []
            for i in range(0, len(feats), 65536):
                chunk = torch.from_numpy(feats[i:i + 65536]).to(device)
                out.append(model(chunk).cpu().numpy())
        return np.concatenate(out)

    from scipy.stats import spearmanr
    print("scipy loaded", flush=True)

    # ── Yardstick 1: A0 slice ────────────────────────────────────────────
    if A0_DB.exists():
        print("yardstick 1: A0 slice ...", flush=True)
        db = PackedEndgameDB.load(A0_DB)
        sample = rng.choice(len(db), size=min(100_000, len(db)), replace=False)
        k0 = np.ascontiguousarray(db.keys0[sample])
        k1 = np.ascontiguousarray(db.keys1[sample])
        exact = db.values[sample].astype(np.float32)
        print("  computing net values", flush=True)
        net_vals = net_values(k0, k1)
        print("  computing material values", flush=True)
        mat_vals = material_values(k0, k1)
        metrics["a0_slice_proxy"] = {
            "n": int(len(sample)),
            "mse_net": float(np.mean((net_vals - exact) ** 2)),
            "mse_material": float(np.mean((mat_vals - exact) ** 2)),
            "mae_net": float(np.mean(np.abs(net_vals - exact))),
            "mae_material": float(np.mean(np.abs(mat_vals - exact))),
            "spearman_net": float(spearmanr(net_vals, exact).statistic),
            "spearman_material": float(spearmanr(mat_vals, exact).statistic),
        }
        print(json.dumps(metrics["a0_slice_proxy"], indent=2), flush=True)

    # ── Yardstick 2: depth-3 spot set ────────────────────────────────────
    print("yardstick 2: depth-3 spot ...", flush=True)
    spot = val_idx[:4000]
    k0 = np.ascontiguousarray(keys0[spot])
    k1 = np.ascontiguousarray(keys1[spot])
    print("  computing depth-3 values", flush=True)
    deep = depth3_values(k0, k1, gamma)
    print("  computing net values", flush=True)
    net_vals = net_values(k0, k1)
    mat_vals = material_values(k0, k1)
    metrics["depth3_spot"] = {
        "n": int(len(spot)),
        "mse_net": float(np.mean((net_vals - deep) ** 2)),
        "mse_material": float(np.mean((mat_vals - deep) ** 2)),
        "spearman_net": float(spearmanr(net_vals, deep).statistic),
        "spearman_material": float(spearmanr(mat_vals, deep).statistic),
    }
    print(json.dumps(metrics["depth3_spot"], indent=2), flush=True)

    EVAL_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {EVAL_OUT}", flush=True)


if __name__ == "__main__":
    main()

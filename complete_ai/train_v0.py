"""Train the v0 value network and evaluate it against ground truths (N3).

Evaluations:
1. Held-out depth-2 target MSE — net vs the material_leaf baseline.
2. A0 exact endgame slice — net vs material_leaf against EXACT game values
   (stockless (1,1) universe solved in N2b). This is the honest yardstick:
   the exact values are independent of the training targets.

Run:  python -m complete_ai.train_v0  →  models/value_v0.pt + JSON metrics
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from complete_solver.packed_vi import PackedEndgameDB

from .features import FEATURE_SIZE, features_from_lanes
from .packed_eval import material_leaf_bits

DATASET = Path("data/value_v0_dataset.npz")
A0_DB = Path("data/endgame_h11_A0.npz")
MODEL_OUT = Path("models/value_v0.pt")
METRICS_OUT = Path("models/value_v0_metrics.json")


class ValueNet(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(FEATURE_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.body(x).squeeze(-1)


def material_values(keys0: np.ndarray, keys1: np.ndarray) -> np.ndarray:
    out = np.empty(len(keys0), dtype=np.float32)
    for i in range(len(keys0)):
        out[i] = material_leaf_bits(np.int64(keys0[i]), np.int64(keys1[i]))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    data = np.load(DATASET)
    feats = data["features"]
    targets = data["targets"]
    keys0 = data["keys0"]
    keys1 = data["keys1"]
    n = len(targets)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_val = max(20_000, n // 20)
    val_idx, train_idx = order[:n_val], order[n_val:]
    print(f"dataset: {n} states (train {len(train_idx)}, val {n_val})", flush=True)

    x_train = torch.from_numpy(feats[train_idx]).to(device)
    y_train = torch.from_numpy(targets[train_idx]).to(device)
    x_val = torch.from_numpy(feats[val_idx]).to(device)
    y_val = torch.from_numpy(targets[val_idx]).to(device)

    model = ValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch = 8192
    best_val = float("inf")
    best_state = None
    patience, bad_epochs = 6, 0

    t0 = time.perf_counter()
    for epoch in range(1, 61):
        model.train()
        perm = torch.randperm(len(x_train), device=device)
        total = 0.0
        for i in range(0, len(x_train), batch):
            idx = perm[i:i + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(x_val), y_val))
        print(
            f"epoch {epoch}: train {total / len(x_train):.6f} val {val_mse:.6f}",
            flush=True,
        )
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    # Save the model FIRST — evaluation must never be able to lose it.
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "feature_size": FEATURE_SIZE},
        MODEL_OUT,
    )
    print(f"model saved to {MODEL_OUT}", flush=True)

    # Baseline on the same validation split.
    material_val = material_values(keys0[val_idx], keys1[val_idx])
    material_mse = float(np.mean((material_val - targets[val_idx]) ** 2))

    metrics = {
        "train_seconds": train_seconds,
        "val_mse_net": best_val,
        "val_mse_material": material_mse,
        "n_train": int(len(train_idx)),
        "n_val": int(n_val),
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)

    # Ground-truth yardsticks run in a SEPARATE process (eval_v0.py): a
    # native crash there must not be able to take the trained model with it.


if __name__ == "__main__":
    main()

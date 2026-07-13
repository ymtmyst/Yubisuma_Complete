"""Fitted Nash-VI generation loop (N4 driver).

Each generation: self-play with the current net → retrain on the union of
recent generations' data → arena vs the previous net → A0-slice tracking.
Metrics stream to data/n4_generations.jsonl; models to models/value_gen_*.pt.

Run:  python -m complete_ai.generation_loop --generations 10 --games 2500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from complete_solver.packed_vi import PackedEndgameDB

from .arena import play_match
from .batched_search import BatchedSearcher
from .features import FEATURE_SIZE, features_from_lanes
from .selfplay import run_selfplay, save_generation
from .train_v0 import ValueNet

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
LOG_PATH = DATA_DIR / "n4_generations.jsonl"
A0_DB_PATH = DATA_DIR / "endgame_h11_A0.npz"
REPLAY_GENERATIONS = 3


def load_model(path: Path, device: str) -> ValueNet:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def train_on(model: ValueNet, datasets: list[Path], device: str,
             epochs: int = 25, batch: int = 8192) -> dict:
    feats = []
    targets = []
    for path in datasets:
        data = np.load(path)
        feats.append(data["features"])
        targets.append(data["targets"])
    x = torch.from_numpy(np.concatenate(feats)).to(device)
    y = torch.from_numpy(np.concatenate(targets)).to(device)
    n = len(x)
    n_val = min(max(2000, n // 20), n // 2)
    perm = torch.randperm(n, device=device)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(x[val_idx]), y[val_idx]))
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"val_mse": best_val, "n_samples": int(n)}


def a0_spearman(model: ValueNet, device: str, sample_size: int = 30000) -> float:
    if not A0_DB_PATH.exists():
        return float("nan")
    from scipy.stats import spearmanr

    db = PackedEndgameDB.load(A0_DB_PATH)
    rng = np.random.default_rng(1)
    sample = rng.choice(len(db), size=min(sample_size, len(db)), replace=False)
    k0 = np.ascontiguousarray(db.keys0[sample])
    k1 = np.ascontiguousarray(db.keys1[sample])
    exact = db.values[sample]
    feats = features_from_lanes(k0, k1)
    with torch.no_grad():
        vals = model(torch.from_numpy(feats).to(device)).cpu().numpy()
    return float(spearmanr(vals, exact).statistic)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--games", type=int, default=2500)
    parser.add_argument("--arena-games", type=int, default=120)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--start-model", default="models/value_v0.pt")
    parser.add_argument("--start-generation", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    model = load_model(Path(args.start_model), device)

    for gen in range(args.start_generation, args.start_generation + args.generations):
        print(f"=== generation {gen} ===", flush=True)
        t0 = time.perf_counter()

        searcher = BatchedSearcher(model, device, prune_stock=True)
        result = run_selfplay(
            searcher, n_games=args.games, epsilon=args.epsilon, seed=gen
        )
        # Deeper training targets (N4b core): act at depth 2, learn from
        # depth-3 net-leaf backups so slow skills' payoffs fall inside the
        # credit-assignment horizon (designer diagnosis 2026-07-13).
        keys0, keys1 = result["keys0"], result["keys1"]
        t_targets = time.perf_counter()
        deep = np.empty(len(keys0), dtype=np.float32)
        for i in range(len(keys0)):
            deep[i] = searcher.value_depth3(int(keys0[i]), int(keys1[i]))
        result["targets"] = deep
        print(
            f"depth-3 targets: {len(deep)} states "
            f"({time.perf_counter() - t_targets:.0f}s)",
            flush=True,
        )
        gen_path = DATA_DIR / f"selfplay_gen{gen:03d}.npz"
        save_generation(result, gen_path)
        print(
            f"selfplay: {len(result['targets'])} states, "
            f"outcomes={result['outcomes']}, mean_plies={result['mean_plies']:.1f}, "
            f"{result['seconds']:.0f}s",
            flush=True,
        )

        # Train a fresh copy so a bad generation can be rolled back.
        new_model = ValueNet().to(device)
        new_model.load_state_dict(model.state_dict())
        recent = sorted(DATA_DIR.glob("selfplay_gen*.npz"))[-REPLAY_GENERATIONS:]
        # Anchor replay (N4b): keep the broad v0 dataset in every training
        # set so on-policy fine-tuning cannot silently forget rare regions.
        anchor = DATA_DIR / "value_v0_dataset.npz"
        datasets = ([anchor] if anchor.exists() else []) + list(recent)
        train_info = train_on(new_model, datasets, device)
        print(f"train: {train_info}", flush=True)

        old_searcher = BatchedSearcher(model, device, prune_stock=True)
        new_searcher = BatchedSearcher(new_model, device, prune_stock=True)
        match = play_match(
            new_searcher, old_searcher, n_games=args.arena_games, seed=gen
        )
        print(f"arena (new as A): {match}", flush=True)

        spearman = a0_spearman(new_model, device)
        print(f"a0 spearman: {spearman:.4f}", flush=True)

        model_path = MODEL_DIR / f"value_gen{gen:03d}.pt"
        torch.save(
            {"state_dict": new_model.state_dict(), "feature_size": FEATURE_SIZE},
            model_path,
        )
        record = {
            "generation": gen,
            "selfplay_states": int(len(result["targets"])),
            "selfplay_outcomes": result["outcomes"],
            "mean_plies": result["mean_plies"],
            "selfplay_seconds": result["seconds"],
            "train": train_info,
            "arena_new_winrate": match["winrate_a"],
            "arena": match,
            "a0_spearman": spearman,
            "gen_seconds": time.perf_counter() - t0,
            "model": str(model_path),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        # Accept unless clearly worse (fitted VI: value accuracy is the goal,
        # the arena is a regression alarm, not a hill-climbing criterion).
        if match["winrate_a"] < 0.35:
            print("REGRESSION ALARM: keeping previous model", flush=True)
        else:
            model = new_model

    torch.save(
        {"state_dict": model.state_dict(), "feature_size": FEATURE_SIZE},
        MODEL_DIR / "value_latest.pt",
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

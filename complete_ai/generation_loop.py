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
from .batched_search import BatchedSearcher, parallel_depth3_values
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
    parser.add_argument("--threads", type=int, default=8,
                        help="threads for depth target generation")
    parser.add_argument("--target-depth", type=int, default=3, choices=(3, 4),
                        help="LP-backup depth for training targets (acting "
                             "stays depth 2)")
    parser.add_argument("--teacher", default="depth",
                        choices=("depth", "graph-vi", "selective"),
                        help="training-target source: fixed-depth LP backup "
                             "(default), N7-A sub-graph Nash-VI (long horizon), "
                             "or N7-C selective deepening (deep on support)")
    parser.add_argument("--sel-depth", type=int, default=5,
                        help="selective teacher: deepening depth")
    parser.add_argument("--sel-tau", type=float, default=0.05,
                        help="selective teacher: support threshold")
    parser.add_argument("--graph-max-stock", type=int, default=3,
                        help="graph-vi teacher: max stock in the transition "
                             "model (domain: 4+ never worth considering)")
    parser.add_argument("--graph-omega", type=float, default=0.5,
                        help="graph-vi teacher: Jacobi under-relaxation factor")
    parser.add_argument("--graph-coverage", type=int, default=0,
                        help="graph-vi teacher: add this many random-walk "
                             "reachable states to the interior seeds (neutral "
                             "coverage of long-horizon-skill lines that "
                             "on-policy self-play never reaches). 0 = off.")
    parser.add_argument("--tag", default="",
                        help="output namespace: files become "
                             "selfplay_<tag>_gen*.npz / value_<tag>_gen*.pt / "
                             "n4_<tag>_generations.jsonl / value_<tag>_latest.pt. "
                             "Empty = legacy names. Use to run an experiment "
                             "(e.g. --tag gvi) without clobbering another run.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    model = load_model(Path(args.start_model), device)

    # Output namespace: keep parallel experiments (e.g. graph-vi vs depth) in
    # separate files so neither run overwrites the other's generations/log.
    pfx = f"{args.tag}_" if args.tag else ""
    log_path = DATA_DIR / f"n4_{pfx}generations.jsonl"

    for gen in range(args.start_generation, args.start_generation + args.generations):
        print(f"=== generation {gen} ===", flush=True)
        t0 = time.perf_counter()

        searcher = BatchedSearcher(model, device, prune_stock=True)
        result = run_selfplay(
            searcher, n_games=args.games, epsilon=args.epsilon, seed=gen
        )
        # Deeper training targets (N4b core): act at depth 2, learn from
        # depth-3/4 net-leaf backups so slow skills' payoffs fall inside the
        # credit-assignment horizon (designer diagnosis 2026-07-13). --target-
        # depth 4 extends the horizon one more ply (N7).
        keys0, keys1 = result["keys0"], result["keys1"]
        t_targets = time.perf_counter()
        if args.teacher == "graph-vi":
            # N7-A: seed a sub-graph with the self-play states (connected
            # trajectories ⇒ denser interior), close the frontier with the net,
            # and solve Nash-VI so long-horizon value propagates across the
            # whole sampled region — reaching delayed-payoff skills that a fixed
            # depth-2/3 backup cannot see. Values return aligned to keys0/keys1.
            from .graph_teacher import graph_vi_teacher, random_walk_seeds
            if args.graph_coverage > 0:
                # Augment interior with neutral random-walk coverage so long-
                # horizon-skill lines (unreached by on-policy self-play) enter
                # the sub-graph; train on the FULL augmented set.
                rw0, rw1 = random_walk_seeds(args.graph_coverage, seed=gen)
                stacked = np.vstack([
                    np.stack([keys0, keys1], axis=1),
                    np.stack([rw0, rw1], axis=1),
                ])
                uniq = np.unique(stacked, axis=0)
                keys0 = np.ascontiguousarray(uniq[:, 0])
                keys1 = np.ascontiguousarray(uniq[:, 1])
                result["keys0"], result["keys1"] = keys0, keys1
            deep, tab, gvi = graph_vi_teacher(
                model, device, keys0, keys1,
                max_stock=args.graph_max_stock, gamma=0.999,
                omega=args.graph_omega,
            )
            # Sub-graph VI works in float64; training targets/features are
            # float32 (match the depth teacher, else loss.backward dtype-clashes).
            deep = deep.astype(np.float32)
            print(
                f"graph-vi targets: {tab.n_seed} interior + {tab.n_front} "
                f"frontier ({tab.frontier_fraction*100:.1f}%), "
                f"{gvi['iterations']} sweeps {gvi['vi_seconds']:.0f}s, "
                f"converged={gvi['converged']} stalled={gvi['stalled']} "
                f"max_delta={gvi['max_delta']:.1e} "
                f"({time.perf_counter() - t_targets:.0f}s)",
                flush=True,
            )
        elif args.teacher == "selective":
            # N7-C: deep-on-support selective values as targets. Reads the
            # equilibrium support to sel_depth (far deeper than depth-3) at a
            # fraction of a uniform deep search's cost — a MORE ACCURATE teacher
            # (the compounding lever: net fits its teacher, so a better teacher
            # → a better net). Pure-Python for now (slow ~0.3s/state at d5);
            # njit compilation is the scale unlock if this proves out.
            from .selective_search import SelectiveSearcher
            ss = SelectiveSearcher(model, device, prune=True,
                                   depth=args.sel_depth, tau=args.sel_tau)
            deep = np.array(
                [ss.value(int(k0), int(k1), args.sel_depth, args.sel_tau)
                 for k0, k1 in zip(keys0, keys1)],
                dtype=np.float32,
            )
            print(
                f"selective-d{args.sel_depth} targets: {len(deep)} states "
                f"({time.perf_counter() - t_targets:.0f}s)",
                flush=True,
            )
        else:
            # Deeper training targets (N4b core): act at depth 2, learn from
            # depth-3/4 net-leaf backups so slow skills' payoffs fall inside the
            # credit-assignment horizon (designer diagnosis 2026-07-13).
            deep = parallel_depth3_values(
                model, device, keys0, keys1, prune_stock=True,
                n_threads=args.threads, depth=args.target_depth,
            )
            print(
                f"depth-{args.target_depth} targets: {len(deep)} states "
                f"({time.perf_counter() - t_targets:.0f}s)",
                flush=True,
            )
        result["targets"] = deep
        gen_path = DATA_DIR / f"selfplay_{pfx}gen{gen:03d}.npz"
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
        recent = sorted(DATA_DIR.glob(f"selfplay_{pfx}gen*.npz"))[-REPLAY_GENERATIONS:]
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

        model_path = MODEL_DIR / f"value_{pfx}gen{gen:03d}.pt"
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
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        # Accept unless clearly worse (fitted VI: value accuracy is the goal,
        # the arena is a regression alarm, not a hill-climbing criterion).
        if match["winrate_a"] < 0.35:
            print("REGRESSION ALARM: keeping previous model", flush=True)
        else:
            model = new_model

    torch.save(
        {"state_dict": model.state_dict(), "feature_size": FEATURE_SIZE},
        MODEL_DIR / f"value_{pfx}latest.pt",
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

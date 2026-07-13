"""Dataset generation for the v0 value network (N3).

States are collected from skill-biased random playouts on the compiled
engine (from both the opening and the (1,1) endgame root), deduplicated,
then labelled with exact depth-2 LP-backup targets (material leaves).

Run:  python -m complete_ai.dataset  →  data/value_v0_dataset.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from complete_solver.endgame_abstraction import h11_root
from complete_solver.packed_engine import (
    legal_ntp_codes,
    legal_tp_codes,
    pack_state,
    step,
)
from complete_solver.state import initial_state

from .features import features_from_lanes
from .packed_eval import depth3_values

_FULL_MASK = np.int64(255)
_NO_CAP = np.int64(99)


@njit(cache=True)
def _collect_playouts(root0, root1, seed, n_games, max_steps, out0, out1, start):
    """Biased random playouts; stores every visited non-terminal state."""
    np.random.seed(seed)
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    skill_codes = np.zeros(96, dtype=np.int64)
    count = start
    cap = out0.shape[0]
    for _ in range(n_games):
        lane0 = root0
        lane1 = root1
        for _ in range(max_steps):
            if count >= cap:
                return count
            out0[count] = lane0
            out1[count] = lane1
            count += 1

            n_tp = legal_tp_codes(lane0, lane1, _FULL_MASK, _NO_CAP, tp_buf)
            n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)

            # Bias toward skill declarations to reach rare mechanics.
            n_skill = 0
            for a in range(n_tp):
                if tp_buf[a] >= 64:
                    skill_codes[n_skill] = tp_buf[a]
                    n_skill += 1
            if n_skill > 0 and np.random.random() < 0.65:
                tp_code = skill_codes[np.random.randint(0, n_skill)]
            else:
                tp_code = tp_buf[np.random.randint(0, n_tp)]
            ntp_code = ntp_buf[np.random.randint(0, n_ntp)]

            child0, child1, status, _ = step(
                lane0, lane1, tp_code, ntp_code, _FULL_MASK
            )
            if status == 2:
                break
            lane0 = child0
            lane1 = child1
    return count


def generate_dataset(
    n_opening_games: int = 60_000,
    n_endgame_games: int = 40_000,
    max_steps: int = 60,
    gamma: float = 0.999,
    seed: int = 0,
    out_path: str | Path = "data/value_v0_dataset.npz",
) -> dict:
    cap = (n_opening_games + n_endgame_games) * max_steps
    keys0 = np.empty(cap, dtype=np.int64)
    keys1 = np.empty(cap, dtype=np.int64)

    t0 = time.perf_counter()
    init0, init1 = pack_state(initial_state())
    end0, end1 = pack_state(h11_root())
    count = _collect_playouts(
        np.int64(init0), np.int64(init1), seed, n_opening_games, max_steps,
        keys0, keys1, 0,
    )
    count = _collect_playouts(
        np.int64(end0), np.int64(end1), seed + 1, n_endgame_games, max_steps,
        keys0, keys1, count,
    )
    playout_seconds = time.perf_counter() - t0

    stacked = np.stack([keys0[:count], keys1[:count]], axis=1)
    unique = np.unique(stacked, axis=0)
    k0 = np.ascontiguousarray(unique[:, 0])
    k1 = np.ascontiguousarray(unique[:, 1])
    print(
        f"playouts: {count} visits → {len(k0)} unique states "
        f"({playout_seconds:.1f}s)",
        flush=True,
    )

    t0 = time.perf_counter()
    targets = depth3_values(k0, k1, gamma)
    target_seconds = time.perf_counter() - t0
    print(f"depth-3 targets: {target_seconds:.1f}s", flush=True)

    t0 = time.perf_counter()
    feats = features_from_lanes(k0, k1)
    feature_seconds = time.perf_counter() - t0
    print(f"features: {feats.shape} ({feature_seconds:.1f}s)", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keys0=k0, keys1=k1, features=feats, targets=targets,
        gamma=np.array([gamma]),
    )
    info = {
        "unique_states": int(len(k0)),
        "playout_seconds": playout_seconds,
        "target_seconds": target_seconds,
        "feature_seconds": feature_seconds,
        "path": str(out_path),
    }
    print(f"saved {info}", flush=True)
    return info


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    generate_dataset()

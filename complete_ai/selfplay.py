"""Self-play generation for fitted Nash-VI (N4).

Both sides act from the same depth-2 net-leaf search, sampling from the root
LP mixture with ε-uniform exploration. The search's root value at every
visited state is recorded as that state's training target (acting and target
generation are the same computation).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from complete_solver.packed_engine import pack_state, step
from complete_solver.state import initial_state

from .batched_search import BatchedSearcher, _FULL_MASK
from .features import features_from_lanes


def _sample(rng: np.random.Generator, policy: np.ndarray, epsilon: float) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(policy)))
    p = np.asarray(policy, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0:
        return int(rng.integers(0, len(policy)))
    return int(rng.choice(len(p), p=p / total))


def run_selfplay(
    searcher: BatchedSearcher,
    n_games: int,
    epsilon: float = 0.12,
    max_plies: int = 100,
    seed: int = 0,
    log_every: int = 200,
    endgame_fraction: float = 0.25,
    max_random_opening_plies: int = 2,
) -> dict:
    """Play games, returning dedup'd (keys, targets) and play statistics.

    N4b diversity (neutral means only — no skill forcing): a fraction of
    games starts from the (1,1) endgame root, and each game opens with 0-2
    uniformly random plies so self-play does not funnel into one trunk.
    """
    from complete_solver.endgame_abstraction import h11_root

    rng = np.random.default_rng(seed)
    init0, init1 = pack_state(initial_state())
    end0, end1 = pack_state(h11_root())

    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    outcomes = {"terminal": 0, "truncated": 0}
    plies_total = 0
    t0 = time.perf_counter()

    # Memoize solve() by state: the model is fixed for the whole generation,
    # so a state's value/policy is deterministic. Self-play revisits states
    # heavily (measured ~6x visits per unique state), so caching turns ~6x
    # redundant solves into one solve per unique state — same targets, far
    # less work. The cached value equals the old sums/counts mean exactly.
    solve_cache: dict[tuple[int, int], tuple] = {}

    def cached_solve(key: tuple[int, int]):
        cached = solve_cache.get(key)
        if cached is None:
            cached = searcher.solve(key[0], key[1])
            solve_cache[key] = cached
        return cached

    for game in range(n_games):
        if rng.random() < endgame_fraction:
            lane0, lane1 = np.int64(end0), np.int64(end1)
        else:
            lane0, lane1 = np.int64(init0), np.int64(init1)
        random_opening = int(rng.integers(0, max_random_opening_plies + 1))
        for ply in range(max_plies):
            key = (int(lane0), int(lane1))
            value, tp_codes, ntp_codes, tp_policy, ntp_policy = cached_solve(key)
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
            plies_total += 1

            if ply < random_opening:
                tp_code = tp_codes[int(rng.integers(0, len(tp_codes)))]
                ntp_code = ntp_codes[int(rng.integers(0, len(ntp_codes)))]
            else:
                tp_code = tp_codes[_sample(rng, tp_policy, epsilon)]
                ntp_code = ntp_codes[_sample(rng, ntp_policy, epsilon)]
            child0, child1, status, _ = step(
                lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
            )
            if status == 2:
                outcomes["terminal"] += 1
                break
            lane0, lane1 = child0, child1
        else:
            outcomes["truncated"] += 1
        if log_every and (game + 1) % log_every == 0:
            rate = (game + 1) / (time.perf_counter() - t0)
            print(
                f"selfplay {game + 1}/{n_games} games "
                f"({rate:.1f} games/s, {len(sums)} unique states)",
                flush=True,
            )

    keys = np.array(sorted(sums.keys()), dtype=np.int64)
    keys0 = np.ascontiguousarray(keys[:, 0])
    keys1 = np.ascontiguousarray(keys[:, 1])
    targets = np.array(
        [sums[(int(k0), int(k1))] / counts[(int(k0), int(k1))]
         for k0, k1 in keys],
        dtype=np.float32,
    )
    return {
        "keys0": keys0,
        "keys1": keys1,
        "targets": targets,
        "games": n_games,
        "outcomes": outcomes,
        "mean_plies": plies_total / max(n_games, 1),
        "seconds": time.perf_counter() - t0,
    }


def save_generation(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = features_from_lanes(result["keys0"], result["keys1"])
    np.savez_compressed(
        path,
        keys0=result["keys0"],
        keys1=result["keys1"],
        features=feats,
        targets=result["targets"],
    )

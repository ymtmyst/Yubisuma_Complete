"""Head-to-head matches between two search agents (N4 generation gate)."""

from __future__ import annotations

import numpy as np

from complete_solver.packed_engine import pack_state, step
from complete_solver.state import initial_state

from .batched_search import BatchedSearcher, _FULL_MASK
from .selfplay import _sample


def play_match(
    searcher_a: BatchedSearcher,
    searcher_b: BatchedSearcher,
    n_games: int = 100,
    epsilon: float = 0.02,
    max_plies: int = 120,
    seed: int = 0,
) -> dict:
    """Alternating-first-mover match. Returns win/loss/truncation counts.

    The engine's terminal reward is from the CURRENT mover's perspective
    (+1 = the mover just won), so we track which agent holds the turn:
    status 1 keeps the mover, status 0 hands the turn to the other agent.
    """
    rng = np.random.default_rng(seed)
    init0, init1 = pack_state(initial_state())
    searchers = (searcher_a, searcher_b)
    wins = [0, 0]
    truncations = 0

    # Each searcher's model is fixed for the whole match, so solve() is
    # deterministic per state; cache it (the match revisits states heavily).
    caches: tuple[dict, dict] = ({}, {})

    def cached_solve(side: int, key: tuple[int, int]):
        cached = caches[side].get(key)
        if cached is None:
            cached = searchers[side].solve(key[0], key[1])
            caches[side][key] = cached
        return cached

    for game in range(n_games):
        mover = game % 2  # alternate who moves first
        lane0, lane1 = np.int64(init0), np.int64(init1)
        for _ in range(max_plies):
            key = (int(lane0), int(lane1))
            value, tp_codes, ntp_codes, tp_policy, ntp_policy = cached_solve(
                mover, key
            )
            # The NTP reaction belongs to the *other* agent: use their policy.
            _, _, _, _, opp_ntp_policy = cached_solve(1 - mover, key)
            tp_code = tp_codes[_sample(rng, tp_policy, epsilon)]
            ntp_code = ntp_codes[_sample(rng, opp_ntp_policy, epsilon)]
            # CHOICE fix: resolve a collapsed CHOICE declaration to the
            # concrete stocked skill now that the opponent's realized
            # reaction is known (post-reaction / second-mover pick). Resolved
            # against the MOVER's own searcher (resolve_tp_code re-solves
            # internally if its single-slot cache doesn't match this state —
            # e.g. because the opponent's cached_solve call above used a
            # different searcher instance, or this state's mover-side result
            # was itself a cache hit).
            tp_code = searchers[mover].resolve_tp_code(
                lane0, lane1, int(tp_code), int(ntp_code)
            )
            child0, child1, status, reward = step(
                lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
            )
            if status == 2:
                winner = mover if reward > 0 else 1 - mover
                wins[winner] += 1
                break
            if status == 0:
                mover = 1 - mover
            lane0, lane1 = child0, child1
        else:
            truncations += 1

    return {
        "games": n_games,
        "wins_a": wins[0],
        "wins_b": wins[1],
        "truncations": truncations,
        "winrate_a": wins[0] / max(wins[0] + wins[1], 1),
    }

"""Compiled leaf evaluation and shallow LP-backup targets (N3).

``material_leaf_bits`` is a bit-level port of
``complete_solver.finite_horizon.material_leaf_evaluator`` (differentially
tested). ``depth2_values`` computes, for a batch of packed states, the exact
depth-2 LP-backup value with material leaves — the v0 training target for the
value network. Semantics match
``FastHorizonSolver(gamma=γ, leaf_evaluator=material).value(s, 2)`` exactly.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes, step
from complete_solver.packed_vi import _matrix_value

_FULL_MASK = np.int64(255)
_NO_CAP = np.int64(99)


@njit(cache=True)
def material_leaf_bits(lane0, lane1):
    player_mask = (np.int64(1) << 42) - 1
    me = lane0 & player_mask
    opp = lane1 & player_mask
    me_hands = me & 3
    opp_hands = opp & 3
    me_declared = me >> 16 & 1
    opp_declared = opp >> 16 & 1

    if me_hands <= 0 and opp_hands <= 0:
        return 0.0
    if me_hands <= 0:
        return 1.0 if (me_declared and opp_declared) else 0.0
    if opp_hands <= 0:
        return -1.0 if (me_declared and opp_declared) else 0.0

    value = (opp_hands - me_hands) / 2.0
    if me >> 4 & 1:
        value += 0.03
    if opp >> 4 & 1:
        value -= 0.03
    if me >> 5 & 1:
        value += 0.02
    if opp >> 5 & 1:
        value -= 0.02
    value += 0.015 * ((me >> 6 & 3) - (opp >> 6 & 3))
    me_stock = 0
    opp_stock = 0
    for i in range(8):
        me_stock += me >> (18 + i) & 1
        opp_stock += opp >> (18 + i) & 1
    value += 0.01 * (me_stock - opp_stock)
    if me >> 14 & 1:
        value -= 0.04
    if opp >> 14 & 1:
        value += 0.04
    if value > 1.0:
        value = 1.0
    elif value < -1.0:
        value = -1.0
    return value


@njit(cache=True, nogil=True)
def material_leaf_batch(keys0, keys1):
    """Vectorised material leaf over a batch — one nogil njit call (so a
    material searcher's leaf eval stays GIL-free and threads scale)."""
    n = keys0.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = material_leaf_bits(keys0[i], keys1[i])
    return out


@njit(cache=True)
def _depth1_value(lane0, lane1, gamma, tp_buf, ntp_buf, matrix, tableau, basis):
    n_tp = legal_tp_codes(lane0, lane1, _FULL_MASK, _NO_CAP, tp_buf)
    n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
    for a in range(n_tp):
        for b in range(n_ntp):
            child0, child1, status, reward = step(
                lane0, lane1, tp_buf[a], ntp_buf[b], _FULL_MASK
            )
            if status == 2:
                matrix[a, b] = reward
            else:
                sign = 1.0 if status == 1 else -1.0
                matrix[a, b] = sign * gamma * material_leaf_bits(child0, child1)
    value, _ = _matrix_value(matrix, n_tp, n_ntp, tableau, basis)
    return value


@njit(cache=True)
def _depth2_value(lane0, lane1, gamma,
                  tp_buf, ntp_buf, matrix, tableau, basis,
                  tp_buf2, ntp_buf2, matrix2):
    n_tp = legal_tp_codes(lane0, lane1, _FULL_MASK, _NO_CAP, tp_buf)
    n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
    for a in range(n_tp):
        for b in range(n_ntp):
            child0, child1, status, reward = step(
                lane0, lane1, tp_buf[a], ntp_buf[b], _FULL_MASK
            )
            if status == 2:
                matrix[a, b] = reward
            else:
                sign = 1.0 if status == 1 else -1.0
                child_value = _depth1_value(
                    child0, child1, gamma, tp_buf2, ntp_buf2,
                    matrix2, tableau, basis,
                )
                matrix[a, b] = sign * gamma * child_value
    value, _ = _matrix_value(matrix, n_tp, n_ntp, tableau, basis)
    return value


@njit(cache=True)
def _depth3_value(lane0, lane1, gamma,
                  tp_buf, ntp_buf, matrix, tableau, basis,
                  tp_buf2, ntp_buf2, matrix2,
                  tp_buf3, ntp_buf3, matrix3):
    n_tp = legal_tp_codes(lane0, lane1, _FULL_MASK, _NO_CAP, tp_buf)
    n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
    for a in range(n_tp):
        for b in range(n_ntp):
            child0, child1, status, reward = step(
                lane0, lane1, tp_buf[a], ntp_buf[b], _FULL_MASK
            )
            if status == 2:
                matrix[a, b] = reward
            else:
                sign = 1.0 if status == 1 else -1.0
                child_value = _depth2_value(
                    child0, child1, gamma,
                    tp_buf2, ntp_buf2, matrix2, tableau, basis,
                    tp_buf3, ntp_buf3, matrix3,
                )
                matrix[a, b] = sign * gamma * child_value
    value, _ = _matrix_value(matrix, n_tp, n_ntp, tableau, basis)
    return value


@njit(cache=True)
def depth3_values(keys0, keys1, gamma):
    """Batch: exact depth-3 LP-backup values with material leaves."""
    n = keys0.shape[0]
    out = np.empty(n, dtype=np.float32)
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    tp_buf2 = np.zeros(96, dtype=np.int64)
    ntp_buf2 = np.zeros(16, dtype=np.int64)
    tp_buf3 = np.zeros(96, dtype=np.int64)
    ntp_buf3 = np.zeros(16, dtype=np.int64)
    matrix = np.zeros((96, 16), dtype=np.float64)
    matrix2 = np.zeros((96, 16), dtype=np.float64)
    matrix3 = np.zeros((96, 16), dtype=np.float64)
    tableau = np.zeros((97, 120), dtype=np.float64)
    basis = np.zeros(96, dtype=np.int64)
    for i in range(n):
        out[i] = _depth3_value(
            keys0[i], keys1[i], gamma,
            tp_buf, ntp_buf, matrix, tableau, basis,
            tp_buf2, ntp_buf2, matrix2,
            tp_buf3, ntp_buf3, matrix3,
        )
    return out


class MaterialLeafSearcher:
    """A BatchedSearcher whose 'network' is the material leaf — gives exact
    depth-2/3 LP-backup values with material leaves via the deduped compiled
    path, with NO neural model and purely on the CPU. Used to label the v0
    anchor dataset far faster than the naive full-width recursion (which has
    no transposition dedup)."""

    def __new__(cls, prune_stock: bool = False, gamma: float = 0.999):
        # Import here to avoid a module-load cycle (batched_search imports
        # nothing from packed_eval; packed_eval imports it lazily).
        from .batched_search import BatchedSearcher

        class _Impl(BatchedSearcher):
            def _net_values(self, keys0, keys1):
                return material_leaf_batch(
                    np.ascontiguousarray(keys0), np.ascontiguousarray(keys1)
                )

        return _Impl(model=None, device="cpu", gamma=gamma,
                     prune_stock=prune_stock)


def material_depth3_values(keys0, keys1, gamma: float = 0.999,
                           n_threads: int = 8, prune_stock: bool = False):
    """Threaded exact depth-3 material targets (v0 anchor). Bit-equivalent to
    depth3_values (the naive recursion) but ~100x faster per state via
    transposition dedup, and threaded on top. Falls back to depth 2 only on
    expansion-buffer overflow, which does not occur for reachable states
    (measured max leaves ~19k vs the 400k cap)."""
    from .batched_search import parallel_depth3_values

    def factory():
        return MaterialLeafSearcher(prune_stock=prune_stock, gamma=gamma)

    return parallel_depth3_values(
        None, "cpu", keys0, keys1, gamma=gamma, prune_stock=prune_stock,
        n_threads=n_threads, searcher_factory=factory,
    )


@njit(cache=True)
def depth2_values(keys0, keys1, gamma):
    """Batch: exact depth-2 LP-backup values with material leaves."""
    n = keys0.shape[0]
    out = np.empty(n, dtype=np.float32)
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    tp_buf2 = np.zeros(96, dtype=np.int64)
    ntp_buf2 = np.zeros(16, dtype=np.int64)
    matrix = np.zeros((96, 16), dtype=np.float64)
    matrix2 = np.zeros((96, 16), dtype=np.float64)
    tableau = np.zeros((97, 120), dtype=np.float64)
    basis = np.zeros(96, dtype=np.int64)
    for i in range(n):
        out[i] = _depth2_value(
            keys0[i], keys1[i], gamma,
            tp_buf, ntp_buf, matrix, tableau, basis,
            tp_buf2, ntp_buf2, matrix2,
        )
    return out

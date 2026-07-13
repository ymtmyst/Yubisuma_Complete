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

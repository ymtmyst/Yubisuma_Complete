"""Compiled exact value iteration over packed states (endgame tablebase).

Pipeline (all hot loops are Numba-compiled):
1. ``enumerate_universe`` — BFS closure over packed states.
2. ``build_tables``      — flat CSR-style transition tables
                           (child index, or terminal reward / sign byte).
3. ``gauss_seidel``      — Shapley value iteration; each state's small matrix
                           game is solved by an inlined saddle-point check,
                           2×2 closed form, or dense tableau simplex.
4. ``PackedEndgameDB``   — sorted-array value store with binary-search probe
                           and npz persistence.

The values are EXACT full-game values for every state in a closed set
(hands never increase ⇒ reachable sets are closed; see endgame_db.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit
from numba.typed import Dict as NumbaDict
from numba.core import types

from .actions import RulesConfig
from .packed_engine import (
    FULL_ALPHABET_MASK,
    SKILL_ID,
    legal_ntp_codes,
    legal_tp_codes,
    pack_state,
    step,
    unpack_state,
)
from .state import State

_KEY_TYPE = types.UniTuple(types.int64, 2)


# ── enumeration ────────────────────────────────────────────────────────────


@njit(cache=True)
def _enumerate(root0, root1, alphabet_mask, max_stock, cap):
    index = NumbaDict.empty(_KEY_TYPE, types.int64)
    keys0 = np.empty(cap, dtype=np.int64)
    keys1 = np.empty(cap, dtype=np.int64)
    index[(root0, root1)] = 0
    keys0[0] = root0
    keys1[0] = root1
    count = 1
    head = 0
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)

    while head < count:
        lane0 = keys0[head]
        lane1 = keys1[head]
        head += 1
        n_tp = legal_tp_codes(lane0, lane1, alphabet_mask, max_stock, tp_buf)
        n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
        for a in range(n_tp):
            for b in range(n_ntp):
                child0, child1, status, _ = step(
                    lane0, lane1, tp_buf[a], ntp_buf[b], alphabet_mask
                )
                if status == 2:
                    continue
                key = (child0, child1)
                if key not in index:
                    if count >= cap:
                        return keys0, keys1, index, -1  # over budget
                    index[key] = count
                    keys0[count] = child0
                    keys1[count] = child1
                    count += 1
    return keys0, keys1, index, count


# ── transition tables ──────────────────────────────────────────────────────


@njit(cache=True)
def _count_cells(keys0, keys1, count, alphabet_mask, max_stock):
    tp_counts = np.zeros(count, dtype=np.int16)
    ntp_counts = np.zeros(count, dtype=np.int16)
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    total = np.int64(0)
    for i in range(count):
        n_tp = legal_tp_codes(keys0[i], keys1[i], alphabet_mask, max_stock, tp_buf)
        n_ntp = legal_ntp_codes(keys0[i], keys1[i], ntp_buf)
        tp_counts[i] = n_tp
        ntp_counts[i] = n_ntp
        total += n_tp * n_ntp
    return tp_counts, ntp_counts, total


@njit(cache=True)
def _fill_tables(keys0, keys1, count, index, alphabet_mask, max_stock,
                 offsets, child_idx, cell_val):
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    for i in range(count):
        lane0 = keys0[i]
        lane1 = keys1[i]
        n_tp = legal_tp_codes(lane0, lane1, alphabet_mask, max_stock, tp_buf)
        n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
        base = offsets[i]
        pos = base
        for a in range(n_tp):
            for b in range(n_ntp):
                child0, child1, status, reward = step(
                    lane0, lane1, tp_buf[a], ntp_buf[b], alphabet_mask
                )
                if status == 2:
                    child_idx[pos] = -1
                    cell_val[pos] = reward
                else:
                    child_idx[pos] = index[(child0, child1)]
                    cell_val[pos] = 1 if status == 1 else -1
                pos += 1


# ── inlined small-matrix solver (value only) ──────────────────────────────


@njit(cache=True)
def _matrix_value(matrix, n_rows, n_cols, tableau, basis):
    """Exact value of a zero-sum matrix game (rows maximize)."""
    # 1×N / N×1
    if n_rows == 1:
        best = matrix[0, 0]
        for j in range(1, n_cols):
            if matrix[0, j] < best:
                best = matrix[0, j]
        return best, True
    if n_cols == 1:
        best = matrix[0, 0]
        for i in range(1, n_rows):
            if matrix[i, 0] > best:
                best = matrix[i, 0]
        return best, True

    # saddle point
    maximin = -1e18
    for i in range(n_rows):
        row_min = 1e18
        for j in range(n_cols):
            if matrix[i, j] < row_min:
                row_min = matrix[i, j]
        if row_min > maximin:
            maximin = row_min
    minimax = 1e18
    for j in range(n_cols):
        col_max = -1e18
        for i in range(n_rows):
            if matrix[i, j] > col_max:
                col_max = matrix[i, j]
        if col_max < minimax:
            minimax = col_max
    if maximin + 1e-12 >= minimax:
        return maximin, True

    # 2×2 closed form (no saddle ⇒ fully mixed)
    if n_rows == 2 and n_cols == 2:
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]
        denom = a - b - c + d
        if abs(denom) > 1e-12:
            return (a * d - b * c) / denom, True

    # Tableau simplex with Bland's rule, hardened against degenerate
    # cycling: iteration exhaustion is DETECTED (not silently used), a
    # perturbed retry follows, and any result is verified against the
    # mathematically necessary bound maximin ≤ v ≤ minimax before use.
    # (A silent fall-through here once produced |v|>10 from a ±1 matrix and
    # made the whole value iteration diverge — 2026-07-13.)
    value = _simplex_value(matrix, n_rows, n_cols, tableau, basis, 0.0)
    if value != value or value < maximin - 1e-6 or value > minimax + 1e-6:
        value = _simplex_value(matrix, n_rows, n_cols, tableau, basis, 1e-7)
    if value != value or value < maximin - 1e-6 or value > minimax + 1e-6:
        # Could not certify an equilibrium: return the midpoint placeholder
        # and report failure so the caller can repair it exactly.
        return 0.5 * (maximin + minimax), False
    if value < maximin:
        value = maximin
    elif value > minimax:
        value = minimax
    return value, True


@njit(cache=True)
def _simplex_value(matrix, n_rows, n_cols, tableau, basis, perturb):
    """One simplex pass; returns NaN when it cannot certify optimality."""
    low = matrix[0, 0]
    for i in range(n_rows):
        for j in range(n_cols):
            if matrix[i, j] < low:
                low = matrix[i, j]
    shift = 1.0 - low

    n_total = n_cols + n_rows
    for i in range(n_rows):
        for j in range(n_cols):
            tableau[i, j] = matrix[i, j] + shift
        for j in range(n_rows):
            tableau[i, n_cols + j] = 1.0 if j == i else 0.0
        tableau[i, n_total] = 1.0 + perturb * (i + 1)
    for j in range(n_cols):
        tableau[n_rows, j] = -1.0
    for j in range(n_cols, n_total + 1):
        tableau[n_rows, j] = 0.0
    for i in range(n_rows):
        basis[i] = n_cols + i

    optimal = False
    for _ in range(600):
        entering = -1
        for col in range(n_total):
            if tableau[n_rows, col] < -1e-9:
                entering = col
                break
        if entering < 0:
            optimal = True
            break
        best_ratio = 1e18
        leaving = -1
        for row in range(n_rows):
            coeff = tableau[row, entering]
            if coeff > 1e-9:
                ratio = tableau[row, n_total] / coeff
                if ratio < best_ratio - 1e-9 or (
                    ratio < best_ratio + 1e-9
                    and (leaving < 0 or basis[row] < basis[leaving])
                ):
                    best_ratio = ratio
                    leaving = row
        if leaving < 0:
            return np.nan  # unbounded: cannot happen for shifted matrices
        pivot = tableau[leaving, entering]
        for j in range(n_total + 1):
            tableau[leaving, j] /= pivot
        for row in range(n_rows + 1):
            if row != leaving:
                factor = tableau[row, entering]
                if factor != 0.0:
                    for j in range(n_total + 1):
                        tableau[row, j] -= factor * tableau[leaving, j]
        basis[leaving] = entering

    if not optimal:
        return np.nan
    w_sum = 0.0
    for row in range(n_rows):
        if basis[row] < n_cols:
            w_sum += tableau[row, n_total]
    if w_sum <= 1e-12:
        return np.nan
    shifted_value = 1.0 / w_sum

    # Verify the claimed equilibrium directly (a numerically stalled simplex
    # can report "optimal" with a wrong objective — 2026-07-13 regression).
    # Column strategy y from the primal w; row strategy x from the slack
    # reduced costs; both must certify the value against the SHIFTED matrix.
    u_sum = 0.0
    for i in range(n_rows):
        u_sum += tableau[n_rows, n_cols + i]
    if u_sum <= 1e-12:
        return np.nan
    for j in range(n_cols):
        acc = 0.0
        for i in range(n_rows):
            acc += tableau[n_rows, n_cols + i] * (matrix[i, j] + shift)
        if acc / u_sum < shifted_value - 1e-6:
            return np.nan
    for i in range(n_rows):
        acc = 0.0
        for row in range(n_rows):
            var = basis[row]
            if var < n_cols:
                acc += tableau[row, n_total] * (matrix[i, var] + shift)
        if acc / w_sum > shifted_value + 1e-6:
            return np.nan
    return shifted_value - shift


# ── value iteration ────────────────────────────────────────────────────────


@njit(cache=True)
def _jacobi_sweep(tp_counts, ntp_counts, offsets, child_idx, cell_val,
                  v_old, v_new, gamma, fail_idx):
    """One double-buffered Jacobi sweep of the Shapley operator.

    Jacobi (not Gauss-Seidel): the in-place GS variant let mid-sweep midpoint
    placeholders poison parents faster than post-sweep repairs could clean
    them, sustaining a limit cycle (2026-07-13). Jacobi applies one fixed
    operator per sweep — a gamma-contraction with guaranteed convergence.
    States whose matrix could not be certified go into *fail_idx* for exact
    scipy repair; returns (max_delta, n_failed)."""
    count = tp_counts.shape[0]
    matrix = np.zeros((96, 16), dtype=np.float64)
    tableau = np.zeros((97, 120), dtype=np.float64)
    basis = np.zeros(96, dtype=np.int64)
    max_delta = 0.0
    n_failed = 0
    for i in range(count):
        n_tp = tp_counts[i]
        n_ntp = ntp_counts[i]
        pos = offsets[i]
        for a in range(n_tp):
            for b in range(n_ntp):
                ci = child_idx[pos]
                if ci < 0:
                    matrix[a, b] = cell_val[pos]
                else:
                    matrix[a, b] = cell_val[pos] * gamma * v_old[ci]
                pos += 1
        new_value, ok = _matrix_value(matrix, n_tp, n_ntp, tableau, basis)
        if not ok and n_failed < fail_idx.shape[0]:
            fail_idx[n_failed] = i
            n_failed += 1
        v_new[i] = new_value
        delta = new_value - v_old[i]
        if delta < 0.0:
            delta = -delta
        if delta > max_delta:
            max_delta = delta
    return max_delta, n_failed


def _repair_state_value(i, tp_counts, ntp_counts, offsets, child_idx,
                        cell_val, values, gamma):
    """Exact scipy re-solve of one state's matrix game (repair path)."""
    from .small_matrix import solve_small_zero_sum

    n_tp = int(tp_counts[i])
    n_ntp = int(ntp_counts[i])
    pos = int(offsets[i])
    matrix = np.empty((n_tp, n_ntp))
    for a in range(n_tp):
        for b in range(n_ntp):
            ci = child_idx[pos]
            if ci < 0:
                matrix[a, b] = cell_val[pos]
            else:
                matrix[a, b] = cell_val[pos] * gamma * values[ci]
            pos += 1
    value, _, _ = solve_small_zero_sum(matrix)
    return float(value)


# ── python-side driver and store ───────────────────────────────────────────


class PackedEndgameDB:
    """Sorted-key exact value store with O(log n) probe."""

    def __init__(self, keys0, keys1, values, gamma, alphabet_mask):
        order = np.lexsort((keys1, keys0))
        self.keys0 = keys0[order]
        self.keys1 = keys1[order]
        self.values = values[order]
        self.gamma = gamma
        self.alphabet_mask = int(alphabet_mask)

    def __len__(self) -> int:
        return len(self.values)

    def get(self, state: State, default=None):
        try:
            lane0, lane1 = pack_state(state)
        except AssertionError:
            return default
        lo = np.searchsorted(self.keys0, lane0, side="left")
        hi = np.searchsorted(self.keys0, lane0, side="right")
        if lo == hi:
            return default
        pos = lo + np.searchsorted(self.keys1[lo:hi], lane1, side="left")
        if pos < hi and self.keys1[pos] == lane1:
            return float(self.values[pos])
        return default

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keys0=self.keys0,
            keys1=self.keys1,
            values=self.values,
            gamma=np.array([self.gamma]),
            alphabet_mask=np.array([self.alphabet_mask]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PackedEndgameDB":
        data = np.load(Path(path))
        db = cls.__new__(cls)
        db.keys0 = data["keys0"]
        db.keys1 = data["keys1"]
        db.values = data["values"]
        db.gamma = float(data["gamma"][0])
        db.alphabet_mask = int(data["alphabet_mask"][0])
        return db


def alphabet_to_mask(alphabet: frozenset[str] | None) -> int:
    if alphabet is None:
        return FULL_ALPHABET_MASK
    return sum(1 << SKILL_ID[skill] for skill in alphabet)


def solve_universe(
    root: State,
    alphabet: frozenset[str] | None = None,
    max_stock_size: int | None = None,
    gamma: float = 0.999,
    epsilon: float = 1e-9,
    max_states: int = 20_000_000,
    max_iterations: int = 3000,
    verbose: bool = True,
) -> tuple[PackedEndgameDB, dict]:
    """Enumerate and exactly solve the closed universe reachable from *root*."""
    mask = alphabet_to_mask(alphabet)
    max_stock = 99 if max_stock_size is None else int(max_stock_size)
    root0, root1 = pack_state(root)

    t0 = time.perf_counter()
    keys0, keys1, index, count = _enumerate(
        np.int64(root0), np.int64(root1), np.int64(mask), np.int64(max_stock),
        np.int64(max_states)
    )
    enum_seconds = time.perf_counter() - t0
    if count < 0:
        raise RuntimeError(f"universe exceeds {max_states} states")
    keys0 = keys0[:count].copy()
    keys1 = keys1[:count].copy()
    if verbose:
        print(f"enumerated {count} states in {enum_seconds:.1f}s", flush=True)

    t0 = time.perf_counter()
    tp_counts, ntp_counts, total_cells = _count_cells(
        keys0, keys1, count, np.int64(mask), np.int64(max_stock)
    )
    offsets = np.zeros(count, dtype=np.int64)
    np.cumsum(
        (tp_counts.astype(np.int64) * ntp_counts.astype(np.int64))[:-1],
        out=offsets[1:],
    )
    child_idx = np.empty(int(total_cells), dtype=np.int32)
    cell_val = np.empty(int(total_cells), dtype=np.int8)
    _fill_tables(keys0, keys1, count, index, np.int64(mask), np.int64(max_stock),
                 offsets, child_idx, cell_val)
    table_seconds = time.perf_counter() - t0
    if verbose:
        print(
            f"tables: {int(total_cells)} cells ({table_seconds:.1f}s, "
            f"{child_idx.nbytes / 1e6:.0f}MB)",
            flush=True,
        )

    t0 = time.perf_counter()
    values = np.zeros(count, dtype=np.float64)
    v_next = np.zeros(count, dtype=np.float64)
    fail_idx = np.zeros(4096, dtype=np.int64)
    iterations = 0
    max_delta = np.inf
    total_repairs = 0
    for iterations in range(1, max_iterations + 1):
        max_delta, n_failed = _jacobi_sweep(
            tp_counts, ntp_counts, offsets, child_idx, cell_val,
            values, v_next, gamma, fail_idx,
        )
        # Exact repair of uncertified solves (scipy) before publishing.
        for k in range(n_failed):
            i = int(fail_idx[k])
            repaired = _repair_state_value(
                i, tp_counts, ntp_counts, offsets, child_idx, cell_val,
                values, gamma,
            )
            delta = abs(repaired - values[i])
            if delta > max_delta:
                max_delta = delta
            v_next[i] = repaired
        total_repairs += n_failed
        values, v_next = v_next, values
        if verbose and (iterations % 10 == 0 or max_delta < epsilon):
            print(f"sweep {iterations} max_delta {max_delta:.3e} "
                  f"repairs {n_failed}", flush=True)
        if max_delta < epsilon:
            break
    if verbose and total_repairs:
        print(f"total exact repairs: {total_repairs}", flush=True)
    vi_seconds = time.perf_counter() - t0
    converged = max_delta < epsilon
    if verbose:
        print(
            f"VI: {iterations} sweeps, delta={max_delta:.2e}, "
            f"{vi_seconds:.1f}s, converged={converged}",
            flush=True,
        )
    if not converged:
        raise RuntimeError(
            f"VI did not converge: {iterations} sweeps, delta={max_delta:.2e}"
        )

    info = {
        "states": int(count),
        "cells": int(total_cells),
        "enum_seconds": enum_seconds,
        "table_seconds": table_seconds,
        "vi_seconds": vi_seconds,
        "iterations": int(iterations),
        "max_delta": float(max_delta),
        "gamma": gamma,
        "alphabet_mask": mask,
        "max_stock_size": max_stock,
    }
    return PackedEndgameDB(keys0, keys1, values, gamma, mask), info

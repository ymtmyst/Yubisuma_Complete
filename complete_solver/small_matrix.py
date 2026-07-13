"""Fast exact zero-sum matrix solving for small matrices.

scipy's ``linprog`` costs ~5 ms per call regardless of size, which dominates
search time when node matrices are tiny (double-oracle subgames are typically
2×2 … 6×5). This module solves such games exactly in tens of microseconds:

1. pure saddle point / 1×N / N×1 shortcuts,
2. closed-form 2×2 mixed solution,
3. a compact dense tableau simplex on the classic game↔LP transformation
   (shift the matrix positive, solve ``max 1ᵀw s.t. M̃w ≤ 1, w ≥ 0``; the
   optimal ``w`` yields the column strategy and the slack reduced costs yield
   the row strategy). Bland's rule guarantees termination on the highly
   degenerate matrices this game produces.
4. scipy LP fallback if the simplex result fails verification (paranoia; it
   also keeps behaviour for pathological inputs identical to the reference).
"""

from __future__ import annotations

import numpy as np

from .matrix_game import solve_zero_sum_matrix

_EPS = 1e-9


def solve_small_zero_sum(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Exactly solve a zero-sum matrix game. Returns (value, x, y)."""
    n_rows, n_cols = matrix.shape

    # Degenerate shapes: plain maximin / minimax.
    if n_rows == 1:
        j = int(np.argmin(matrix[0]))
        y = np.zeros(n_cols)
        y[j] = 1.0
        return float(matrix[0, j]), np.array([1.0]), y
    if n_cols == 1:
        i = int(np.argmax(matrix[:, 0]))
        x = np.zeros(n_rows)
        x[i] = 1.0
        return float(matrix[i, 0]), x, np.array([1.0])

    # Pure saddle point.
    row_mins = matrix.min(axis=1)
    maximin = row_mins.max()
    col_maxes = matrix.max(axis=0)
    minimax = col_maxes.min()
    if maximin + _EPS >= minimax:
        i = int(np.argmax(row_mins))
        j = int(np.argmin(col_maxes))
        x = np.zeros(n_rows)
        x[i] = 1.0
        y = np.zeros(n_cols)
        y[j] = 1.0
        return float(maximin), x, y

    # Closed-form 2×2 (no saddle point ⇒ fully mixed).
    if n_rows == 2 and n_cols == 2:
        solved = _solve_2x2(matrix)
        if solved is not None:
            return solved

    solved = _solve_by_simplex(matrix)
    if solved is not None:
        return solved

    solution = solve_zero_sum_matrix(matrix)
    return (
        float(solution.value),
        np.asarray(solution.row_policy, dtype=float),
        np.asarray(solution.col_policy, dtype=float),
    )


def _solve_2x2(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    denom = a - b - c + d
    if abs(denom) < 1e-12:
        return None
    p = (d - c) / denom
    q = (d - b) / denom
    if not (-_EPS <= p <= 1 + _EPS and -_EPS <= q <= 1 + _EPS):
        return None
    p = min(max(p, 0.0), 1.0)
    q = min(max(q, 0.0), 1.0)
    value = (a * d - b * c) / denom
    return float(value), np.array([p, 1.0 - p]), np.array([q, 1.0 - q])


def _solve_by_simplex(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Dense tableau simplex on ``max 1ᵀw s.t. M̃w ≤ 1, w ≥ 0`` (M̃ > 0)."""
    n_rows, n_cols = matrix.shape
    shift = 1.0 - float(matrix.min())
    shifted = matrix + shift  # all entries ≥ 1 → game value ≥ 1 → 1ᵀw bounded

    # Tableau layout: [w vars | slack vars | rhs], one extra objective row.
    n_total = n_cols + n_rows
    tableau = np.zeros((n_rows + 1, n_total + 1))
    tableau[:n_rows, :n_cols] = shifted
    tableau[:n_rows, n_cols:n_total] = np.eye(n_rows)
    tableau[:n_rows, -1] = 1.0
    tableau[-1, :n_cols] = -1.0  # maximise 1ᵀw  →  minimise -1ᵀw

    basis = list(range(n_cols, n_total))
    objective = tableau[-1]

    for _ in range(200):
        # Bland's rule: first improving column.
        entering = -1
        for col in range(n_total):
            if objective[col] < -_EPS:
                entering = col
                break
        if entering < 0:
            break  # optimal

        column = tableau[:n_rows, entering]
        rhs = tableau[:n_rows, -1]
        best_ratio = np.inf
        leaving = -1
        for row in range(n_rows):
            coeff = column[row]
            if coeff > _EPS:
                ratio = rhs[row] / coeff
                if ratio < best_ratio - _EPS or (
                    ratio < best_ratio + _EPS
                    and (leaving < 0 or basis[row] < basis[leaving])
                ):
                    best_ratio = ratio
                    leaving = row
        if leaving < 0:
            return None  # unbounded — cannot happen for M̃ > 0

        pivot_row = tableau[leaving]
        pivot_row /= pivot_row[entering]
        for row in range(n_rows + 1):
            if row != leaving:
                factor = tableau[row, entering]
                if factor != 0.0:
                    tableau[row] -= factor * pivot_row
        basis[leaving] = entering
    else:
        return None  # iteration cap — fall back to scipy

    w = np.zeros(n_cols)
    for row, var in enumerate(basis):
        if var < n_cols:
            w[var] = tableau[row, -1]
    w_sum = w.sum()
    if w_sum <= _EPS:
        return None
    shifted_value = 1.0 / w_sum

    # Dual solution: reduced costs on the slack columns.
    u = objective[n_cols:n_total]
    u_sum = u.sum()
    if u_sum <= _EPS:
        return None

    y = w / w_sum
    x = np.clip(u / u_sum, 0.0, None)
    x /= x.sum()

    # Verify equilibrium on the shifted matrix (cheap and makes this exact-or-fallback).
    if (x @ shifted).min() < shifted_value - 1e-7:
        return None
    if (shifted @ y).max() > shifted_value + 1e-7:
        return None
    return float(shifted_value - shift), x, y

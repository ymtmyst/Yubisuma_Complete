"""Zero-sum matrix game solver for Complete subgames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class MatrixGameSolution:
    value: float
    row_policy: np.ndarray
    col_policy: np.ndarray


def solve_zero_sum_matrix(matrix: np.ndarray, tolerance: float = 1e-10) -> MatrixGameSolution:
    """Solve a finite zero-sum matrix game for the row maximizer."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise ValueError("matrix must be non-empty")

    row_policy, row_value = _solve_row_player(matrix)
    col_policy, col_value = _solve_col_player(matrix)
    value = 0.5 * (row_value + col_value)

    return MatrixGameSolution(
        value=float(value),
        row_policy=_clean_probability(row_policy, tolerance),
        col_policy=_clean_probability(col_policy, tolerance),
    )


def _solve_row_player(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    rows, cols = matrix.shape
    objective = np.zeros(rows + 1)
    objective[-1] = -1.0

    a_ub = []
    b_ub = []
    for col in range(cols):
        constraint = np.zeros(rows + 1)
        constraint[:rows] = -matrix[:, col]
        constraint[-1] = 1.0
        a_ub.append(constraint)
        b_ub.append(0.0)

    a_eq = np.zeros((1, rows + 1))
    a_eq[0, :rows] = 1.0
    b_eq = np.array([1.0])

    result = linprog(
        c=objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * rows + [(-1.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"row LP failed: {result.message}")
    return result.x[:rows], float(result.x[-1])


def _solve_col_player(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    rows, cols = matrix.shape
    objective = np.zeros(cols + 1)
    objective[-1] = 1.0

    a_ub = []
    b_ub = []
    for row in range(rows):
        constraint = np.zeros(cols + 1)
        constraint[:cols] = matrix[row, :]
        constraint[-1] = -1.0
        a_ub.append(constraint)
        b_ub.append(0.0)

    a_eq = np.zeros((1, cols + 1))
    a_eq[0, :cols] = 1.0
    b_eq = np.array([1.0])

    result = linprog(
        c=objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * cols + [(-1.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"column LP failed: {result.message}")
    return result.x[:cols], float(result.x[-1])


def _clean_probability(policy: np.ndarray, tolerance: float) -> np.ndarray:
    cleaned = np.asarray(policy, dtype=float).copy()
    cleaned[np.abs(cleaned) < tolerance] = 0.0
    total = cleaned.sum()
    if total <= 0.0:
        cleaned[:] = 1.0 / len(cleaned)
    else:
        cleaned /= total
    return cleaned

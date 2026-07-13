"""solve_small_zero_sum must agree with the scipy LP reference solver."""

from __future__ import annotations

import unittest

import numpy as np

from complete_solver.matrix_game import solve_zero_sum_matrix
from complete_solver.small_matrix import solve_small_zero_sum

TOLERANCE = 1e-7


def assert_equilibrium(test: unittest.TestCase, matrix: np.ndarray) -> None:
    value, x, y = solve_small_zero_sum(matrix)
    reference = solve_zero_sum_matrix(matrix)

    test.assertAlmostEqual(value, float(reference.value), delta=TOLERANCE, msg=f"{matrix}")
    # x, y must be distributions.
    test.assertAlmostEqual(x.sum(), 1.0, delta=1e-9)
    test.assertAlmostEqual(y.sum(), 1.0, delta=1e-9)
    test.assertGreaterEqual(x.min(), -1e-12)
    test.assertGreaterEqual(y.min(), -1e-12)
    # x guarantees at least the value; y concedes at most the value.
    test.assertGreaterEqual((x @ matrix).min(), value - TOLERANCE, msg=f"{matrix}")
    test.assertLessEqual((matrix @ y).max(), value + TOLERANCE, msg=f"{matrix}")


class TestSmallMatrixSolver(unittest.TestCase):
    def test_known_games(self):
        # Matching pennies: value 0, uniform mixing.
        value, x, y = solve_small_zero_sum(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        self.assertAlmostEqual(value, 0.0, delta=TOLERANCE)
        self.assertAlmostEqual(x[0], 0.5, delta=TOLERANCE)
        self.assertAlmostEqual(y[0], 0.5, delta=TOLERANCE)

        # Rock-paper-scissors: value 0, uniform mixing.
        rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        value, x, y = solve_small_zero_sum(rps)
        self.assertAlmostEqual(value, 0.0, delta=TOLERANCE)
        np.testing.assert_allclose(x, [1 / 3] * 3, atol=1e-6)
        np.testing.assert_allclose(y, [1 / 3] * 3, atol=1e-6)

        # Saddle point.
        saddle = np.array([[3.0, 5.0], [2.0, 1.0]])
        value, x, y = solve_small_zero_sum(saddle)
        self.assertAlmostEqual(value, 3.0, delta=TOLERANCE)

    def test_random_continuous_matrices(self):
        # Payoffs stay within [-1, 1]: the scipy reference solver bounds the
        # game value to that interval (true for all real game matrices here).
        rng = np.random.default_rng(0)
        for _ in range(300):
            m = rng.integers(1, 9)
            n = rng.integers(1, 9)
            matrix = rng.uniform(-1.0, 1.0, size=(m, n))
            assert_equilibrium(self, matrix)

    def test_random_degenerate_integer_matrices(self):
        # Integer-valued matrices with few distinct values are highly
        # degenerate — exactly what game-tree payoff matrices look like.
        rng = np.random.default_rng(1)
        for _ in range(300):
            m = rng.integers(2, 9)
            n = rng.integers(2, 9)
            matrix = rng.integers(-1, 2, size=(m, n)).astype(float)
            assert_equilibrium(self, matrix)

    def test_constant_and_duplicate_rows(self):
        assert_equilibrium(self, np.zeros((4, 4)))
        assert_equilibrium(self, np.ones((3, 5)) * -0.5)
        matrix = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        assert_equilibrium(self, matrix)

    def test_larger_matrices_match_reference(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            matrix = rng.uniform(-1.0, 1.0, size=(45, 9))
            assert_equilibrium(self, matrix)


if __name__ == "__main__":
    unittest.main()

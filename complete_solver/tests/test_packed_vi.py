"""Packed VI: matrix solver equivalence, end-to-end mini universe, store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from complete_solver.actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from complete_solver.packed_vi import (
    PackedEndgameDB,
    _matrix_value,
    solve_universe,
)
from complete_solver.small_matrix import solve_small_zero_sum
from complete_solver.state import PlayerState, State
from complete_solver.transition import transition

GAMMA = 0.999


def matrix_value_reference_check(test, matrix):
    tableau = np.zeros((65, 96), dtype=np.float64)
    basis = np.zeros(64, dtype=np.int64)
    padded = np.zeros((64, 16), dtype=np.float64)
    rows, cols = matrix.shape
    padded[:rows, :cols] = matrix
    fast, _ = _matrix_value(padded, rows, cols, tableau, basis)
    expected, _, _ = solve_small_zero_sum(matrix)
    test.assertAlmostEqual(fast, expected, delta=1e-7, msg=f"{matrix}")


class TestMatrixValue(unittest.TestCase):
    def test_random_matrices(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            rows = int(rng.integers(1, 13))
            cols = int(rng.integers(1, 7))
            matrix = rng.uniform(-1.0, 1.0, size=(rows, cols))
            matrix_value_reference_check(self, matrix)

    def test_degenerate_integer_matrices(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            rows = int(rng.integers(2, 13))
            cols = int(rng.integers(2, 7))
            matrix = rng.integers(-1, 2, size=(rows, cols)).astype(float)
            matrix_value_reference_check(self, matrix)

    def test_highly_degenerate_matrices_stay_within_game_bounds(self):
        """Regression (2026-07-13): a degenerate 28×6 matrix cycled the
        simplex past its iteration cap and the stale tableau yielded |v|>10,
        making VI diverge. The value of any matrix game MUST satisfy
        maximin ≤ v ≤ minimax; agreement with scipy must hold for almost all
        samples (rare hardened-fallback midpoints are allowed)."""
        rng = np.random.default_rng(7)
        tableau = np.zeros((97, 120), dtype=np.float64)
        basis = np.zeros(96, dtype=np.int64)
        padded = np.zeros((96, 16), dtype=np.float64)
        disagreements = 0
        n_samples = 2000
        for _ in range(n_samples):
            rows = int(rng.integers(2, 30))
            cols = int(rng.integers(2, 7))
            # Game-tree-like degeneracy: few distinct values, duplicated rows.
            base = rng.choice([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
                              size=(rows, cols))
            for _ in range(rows // 3):
                base[int(rng.integers(0, rows))] = base[int(rng.integers(0, rows))]
            padded[:rows, :cols] = base
            fast, _ = _matrix_value(padded, rows, cols, tableau, basis)
            maximin = base.min(axis=1).max()
            minimax = base.max(axis=0).min()
            self.assertGreaterEqual(fast, maximin - 1e-6, msg=f"{base}")
            self.assertLessEqual(fast, minimax + 1e-6, msg=f"{base}")
            expected, _, _ = solve_small_zero_sum(base)
            if abs(fast - expected) > 1e-6:
                disagreements += 1
        self.assertLess(disagreements, n_samples * 0.005)


def no_stock_root() -> State:
    return State(
        me=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
        opp=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
    )


class TestSolveUniverseEndToEnd(unittest.TestCase):
    """Solve the stockless (alphabet=∅) (1,1) universe and verify it.

    With stocking forbidden the (1,1) universe is small enough for a test.
    Verification: for sampled states, rebuild the payoff matrix with the
    REFERENCE python engine and the DB's child values; the reference solver's
    value must equal the DB value (Bellman residual ≈ 0 through independent
    code paths for actions, transition and matrix solving).
    """

    @classmethod
    def setUpClass(cls):
        cls.config = RulesConfig(False, False, stock_alphabet=frozenset())
        cls.db, cls.info = solve_universe(
            no_stock_root(),
            alphabet=frozenset(),
            gamma=GAMMA,
            max_states=500_000,
            verbose=False,
        )

    def test_universe_is_nontrivial_and_converged(self):
        self.assertGreater(self.info["states"], 100)
        self.assertLess(self.info["max_delta"], 1e-9)

    def test_bellman_residual_via_reference_engine(self):
        rng = np.random.default_rng(2)
        n = len(self.db)
        sample = rng.choice(n, size=min(200, n), replace=False)
        from complete_solver.packed_engine import unpack_state

        for pos in sample:
            state = unpack_state(int(self.db.keys0[pos]), int(self.db.keys1[pos]))
            tp_actions = legal_tp_actions(state, self.config)
            ntp_actions = legal_ntp_actions(state, self.config)
            matrix = np.empty((len(tp_actions), len(ntp_actions)))
            for r, tp_action in enumerate(tp_actions):
                for c, ntp_action in enumerate(ntp_actions):
                    result = transition(state, tp_action, ntp_action, self.config)
                    if result.terminal_reward is not None:
                        matrix[r, c] = result.terminal_reward
                    else:
                        child_value = self.db.get(result.next_state)
                        self.assertIsNotNone(
                            child_value, f"closure leak: {result.next_state}"
                        )
                        sign = 1.0 if result.same_turn_player else -1.0
                        matrix[r, c] = sign * GAMMA * child_value
            expected, _, _ = solve_small_zero_sum(matrix)
            actual = self.db.get(state)
            self.assertAlmostEqual(
                expected, actual, delta=1e-6, msg=f"state={state}"
            )

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.npz"
            self.db.save(path)
            loaded = PackedEndgameDB.load(path)
            self.assertEqual(len(loaded), len(self.db))
            self.assertEqual(loaded.gamma, self.db.gamma)
            state = no_stock_root()
            self.assertEqual(loaded.get(state), self.db.get(state))


if __name__ == "__main__":
    unittest.main()

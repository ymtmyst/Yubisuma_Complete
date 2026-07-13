"""FastHorizonSolver must reproduce FiniteHorizonSolver values exactly."""

from __future__ import annotations

import random
import unittest

from complete_solver.actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from complete_solver.fast_solver import FastHorizonSolver
from complete_solver.finite_horizon import FiniteHorizonSolver
from complete_solver.state import State, initial_state
from complete_solver.transition import transition

CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)
TOLERANCE = 1e-6


def random_states(seed: int, count: int, max_walk: int = 25) -> list[State]:
    """Collect diverse non-terminal states via random playouts."""
    rng = random.Random(seed)
    states: list[State] = [initial_state()]
    while len(states) < count:
        state = initial_state()
        for _ in range(rng.randrange(1, max_walk)):
            tp = rng.choice(legal_tp_actions(state, CONFIG))
            ntp = rng.choice(legal_ntp_actions(state, CONFIG))
            result = transition(state, tp, ntp, CONFIG)
            if result.next_state is None:
                break
            state = result.next_state
            if len(states) < count and rng.random() < 0.5:
                states.append(state)
    return states[:count]


class TestFastSolverMatchesReference(unittest.TestCase):
    def assert_values_match(self, states: list[State], depth: int, gamma: float) -> None:
        reference = FiniteHorizonSolver(CONFIG, gamma=gamma)
        fast = FastHorizonSolver(CONFIG, gamma=gamma)
        for state in states:
            expected = reference.value(state, depth)
            actual = fast.value(state, depth)
            self.assertAlmostEqual(
                expected,
                actual,
                delta=TOLERANCE,
                msg=f"depth={depth} gamma={gamma} state={state}",
            )

    def test_depth1_random_states(self):
        self.assert_values_match(random_states(seed=1, count=40), depth=1, gamma=0.999)

    def test_depth2_random_states(self):
        self.assert_values_match(random_states(seed=2, count=15), depth=2, gamma=0.999)

    def test_depth3_initial_state(self):
        self.assert_values_match([initial_state()], depth=3, gamma=0.999)

    def test_gamma_one_and_low_gamma(self):
        states = random_states(seed=3, count=8)
        self.assert_values_match(states, depth=2, gamma=1.0)
        self.assert_values_match(states, depth=2, gamma=0.9)

    def test_full_matrix_mode_matches_double_oracle(self):
        states = random_states(seed=4, count=10)
        do_solver = FastHorizonSolver(CONFIG, gamma=0.999, use_double_oracle=True)
        full_solver = FastHorizonSolver(CONFIG, gamma=0.999, use_double_oracle=False)
        for state in states:
            self.assertAlmostEqual(
                do_solver.value(state, 2),
                full_solver.value(state, 2),
                delta=TOLERANCE,
            )


class TestFastSolverBehaviour(unittest.TestCase):
    def test_repeated_solves_are_stable_and_cached(self):
        solver = FastHorizonSolver(CONFIG, gamma=0.999)
        first = solver.value(initial_state(), 2)
        hits_before = solver.stats.tt_hits
        second = solver.value(initial_state(), 2)
        self.assertEqual(first, second)
        self.assertGreater(solver.stats.tt_hits, hits_before)

    def test_root_policy_is_valid_distribution(self):
        solver = FastHorizonSolver(CONFIG, gamma=0.999)
        policy = solver.solve_state(initial_state(), 2)
        self.assertEqual(len(policy.tp_policy), len(policy.tp_actions))
        self.assertEqual(len(policy.ntp_policy), len(policy.ntp_actions))
        self.assertAlmostEqual(sum(policy.tp_policy), 1.0, delta=1e-6)
        self.assertAlmostEqual(sum(policy.ntp_policy), 1.0, delta=1e-6)
        self.assertGreaterEqual(min(policy.tp_policy), 0.0)
        self.assertGreaterEqual(min(policy.ntp_policy), 0.0)

    def test_root_policy_value_matches_reference(self):
        reference = FiniteHorizonSolver(CONFIG, gamma=0.999)
        solver = FastHorizonSolver(CONFIG, gamma=0.999)
        expected = reference.solve_state(initial_state(), 2)
        actual = solver.solve_state(initial_state(), 2)
        self.assertAlmostEqual(expected.value, actual.value, delta=TOLERANCE)

    def test_depth_zero_returns_leaf(self):
        solver = FastHorizonSolver(CONFIG, gamma=0.999, leaf_evaluator=lambda s: 0.25)
        self.assertEqual(solver.value(initial_state(), 0), 0.25)

    def test_clear_caches(self):
        solver = FastHorizonSolver(CONFIG, gamma=0.999)
        solver.value(initial_state(), 2)
        self.assertGreater(solver.cache_sizes()["transposition"], 0)
        solver.clear_caches()
        self.assertEqual(solver.cache_sizes()["transposition"], 0)


if __name__ == "__main__":
    unittest.main()

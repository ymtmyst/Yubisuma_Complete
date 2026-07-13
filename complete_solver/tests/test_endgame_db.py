"""Endgame DB: closure enumeration, exact VI, persistence, solver probe."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complete_solver.actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from complete_solver.endgame_db import EndgameDB, compute_closure, solve_closure
from complete_solver.fast_solver import FastHorizonSolver
from complete_solver.finite_horizon import material_leaf_evaluator
from complete_solver.state import PlayerState, State, initial_state
from complete_solver.state_space import enumerate_reachable_states, value_iteration
from complete_solver.transition import transition

CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)
GAMMA = 0.999


def endgame_root() -> State:
    return State(
        me=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
        opp=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
    )


class TestComputeClosure(unittest.TestCase):
    def test_budget_rejection(self):
        result = compute_closure(initial_state(), CONFIG, max_states=50)
        self.assertIsNone(result.states)
        self.assertEqual(result.visited, 50)

    def test_partial_bfs_set_is_root_connected(self):
        # Sanity on the BFS front: root present, all states reachable.
        result = compute_closure(endgame_root(), CONFIG, max_states=500)
        self.assertIsNone(result.states)  # endgame closures exceed 500


class TestSolveClosureAgainstReference(unittest.TestCase):
    """solve_closure must reproduce the reference value_iteration fixed point.

    True closures are too large for a unit test, so we take a 250-state BFS
    prefix and pin its boundary with the material leaf: both solvers then
    compute the same well-defined fixed point on the same set.
    """

    def test_matches_reference_value_iteration(self):
        states = enumerate_reachable_states(initial_state(), CONFIG, max_states=250)

        # Pin every out-of-set successor with the material leaf value.
        boundary: dict[State, float] = {}
        for state in states:
            for tp_action in legal_tp_actions(state, CONFIG):
                for ntp_action in legal_ntp_actions(state, CONFIG):
                    result = transition(state, tp_action, ntp_action, CONFIG)
                    child = result.next_state
                    if child is not None and child not in states:
                        boundary[child] = material_leaf_evaluator(child)

        fast = solve_closure(
            states, CONFIG, gamma=GAMMA, epsilon=1e-9, known_values=boundary
        )
        self.assertTrue(fast.converged)
        self.assertLess(fast.max_bellman_residual, 1e-6)

        reference = value_iteration(
            states,
            CONFIG,
            gamma=GAMMA,
            epsilon=1e-9,
            max_iterations=5000,
            leaf_evaluator=material_leaf_evaluator,
        )
        self.assertTrue(reference.converged)

        for state in states:
            self.assertAlmostEqual(
                fast.values[state],
                reference.values[state],
                delta=1e-5,
                msg=f"state={state}",
            )

    def test_known_values_are_pinned_not_recomputed(self):
        states = enumerate_reachable_states(initial_state(), CONFIG, max_states=60)
        boundary: dict[State, float] = {}
        for state in states:
            for tp_action in legal_tp_actions(state, CONFIG):
                for ntp_action in legal_ntp_actions(state, CONFIG):
                    result = transition(state, tp_action, ntp_action, CONFIG)
                    child = result.next_state
                    if child is not None and child not in states:
                        boundary[child] = 0.5
        pinned_state = next(iter(states))
        boundary[pinned_state] = 0.123
        solution = solve_closure(
            states, CONFIG, gamma=GAMMA, known_values=boundary
        )
        self.assertNotIn(pinned_state, solution.values)

    def test_unclosed_set_without_known_values_raises(self):
        states = enumerate_reachable_states(initial_state(), CONFIG, max_states=60)
        with self.assertRaises(ValueError):
            solve_closure(states, CONFIG, gamma=GAMMA)


class TestEndgameDB(unittest.TestCase):
    def test_add_state_rejects_over_budget(self):
        db = EndgameDB(CONFIG, gamma=GAMMA, max_closure_states=100)
        self.assertFalse(db.add_state(endgame_root()))
        self.assertEqual(db.rejected_roots, 1)
        self.assertEqual(len(db), 0)

    def test_save_load_roundtrip(self):
        db = EndgameDB(CONFIG, gamma=GAMMA)
        state = endgame_root()
        db.values[state] = 0.25
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.pkl"
            db.save(path)
            loaded = EndgameDB.load(path, CONFIG)
            self.assertEqual(loaded.gamma, GAMMA)
            self.assertEqual(loaded.probe(state), 0.25)
            self.assertIsNone(loaded.probe(initial_state()))

    def test_load_rejects_wrong_config(self):
        db = EndgameDB(CONFIG, gamma=GAMMA)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.pkl"
            db.save(path)
            with self.assertRaises(ValueError):
                EndgameDB.load(path, RulesConfig(enable_mirror=True))


class TestSolverProbe(unittest.TestCase):
    def test_exact_value_short_circuits_search(self):
        state = endgame_root()
        solver = FastHorizonSolver(
            CONFIG, gamma=GAMMA, exact_values={state: 0.777}
        )
        for depth in (0, 1, 3, 10):
            self.assertEqual(solver.value(state, depth), 0.777)
        self.assertGreaterEqual(solver.stats.exact_hits, 4)

    def test_exact_child_values_feed_parent_search(self):
        # Parent search must consume exact child values through payoff cells.
        solver_plain = FastHorizonSolver(CONFIG, gamma=GAMMA)
        state = initial_state()
        plain = solver_plain.value(state, 1)

        # Pin every child of the initial state to +1 for the mover: with a
        # turn switch the parent sees -gamma; same-turn children see +gamma.
        exact: dict[State, float] = {}
        for tp_action in legal_tp_actions(state, CONFIG):
            for ntp_action in legal_ntp_actions(state, CONFIG):
                result = transition(state, tp_action, ntp_action, CONFIG)
                if result.next_state is not None:
                    exact[result.next_state] = 1.0
        solver = FastHorizonSolver(CONFIG, gamma=GAMMA, exact_values=exact)
        pinned = solver.value(state, 1)
        self.assertNotAlmostEqual(plain, pinned, delta=1e-9)


if __name__ == "__main__":
    unittest.main()

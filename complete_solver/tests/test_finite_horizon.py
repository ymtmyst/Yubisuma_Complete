from __future__ import annotations

import unittest

from complete_solver import FiniteHorizonSolver, PlayerState, State
from complete_solver.actions import RulesConfig
from complete_solver.constants import FLASH, MIRROR_PREP
from complete_solver.policy import policy_mass_by_skill, sample_tp_action


class FiniteHorizonTests(unittest.TestCase):
    def test_initial_depth_one_subgame_solves(self) -> None:
        solver = FiniteHorizonSolver()
        policy = solver.solve_state(State(), depth=1)

        self.assertGreater(len(policy.tp_actions), 0)
        self.assertGreater(len(policy.ntp_actions), 0)
        self.assertEqual(len(policy.matrix), len(policy.tp_actions))
        self.assertAlmostEqual(sum(policy.tp_policy), 1.0)
        self.assertAlmostEqual(sum(policy.ntp_policy), 1.0)
        self.assertGreaterEqual(policy.value, -1.0)
        self.assertLessEqual(policy.value, 1.0)

    def test_terminal_tactical_state_values_flash_highly(self) -> None:
        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True, cement=2, lock_pending=True),
        )
        solver = FiniteHorizonSolver()
        policy = solver.solve_state(state, depth=1)
        mass = policy_mass_by_skill(policy)

        self.assertIn(FLASH, mass)
        self.assertGreater(policy.value, 0.2)

    def test_config_changes_legal_policy_surface(self) -> None:
        off_policy = FiniteHorizonSolver().solve_state(State(), depth=1)
        on_policy = FiniteHorizonSolver(RulesConfig(enable_mirror=True)).solve_state(
            State(), depth=1
        )

        off_skills = {action.skill for action in off_policy.tp_actions}
        on_skills = {action.skill for action in on_policy.tp_actions}
        self.assertNotIn(MIRROR_PREP, off_skills)
        self.assertIn(MIRROR_PREP, on_skills)

    def test_policy_sampling_returns_legal_action(self) -> None:
        policy = FiniteHorizonSolver().solve_state(State(), depth=1)
        action = sample_tp_action(policy)

        self.assertIn(action, policy.tp_actions)


if __name__ == "__main__":
    unittest.main()

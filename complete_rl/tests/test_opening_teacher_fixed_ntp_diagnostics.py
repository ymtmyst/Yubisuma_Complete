"""Tests for the opening teacher vs fixed-NTP diagnostic."""

from __future__ import annotations

import unittest

from complete_solver import TPAction
from complete_solver.constants import FEINT
from complete_rl.opening_teacher_fixed_ntp_diagnostics import (
    fixed_scenarios,
    one_step_shape,
)


class OpeningTeacherFixedNtpDiagnosticsTests(unittest.TestCase):
    def test_feint_shape_differs_for_none_and_counter_fixed_ntp(self) -> None:
        scenarios = {scenario.policy_name: scenario for scenario in fixed_scenarios()}
        none_shape = one_step_shape(TPAction(FEINT, 1), scenarios["none_uniform"])
        counter_shape = one_step_shape(TPAction(FEINT, 1), scenarios["counter_uniform"])

        self.assertIn("feint_no_counter", none_shape["events"])
        self.assertIn("最初の手番交代", none_shape["outcome"])
        self.assertIn("feint_success", counter_shape["events"])
        self.assertIn("追加ターン", counter_shape["outcome"])


if __name__ == "__main__":
    unittest.main()

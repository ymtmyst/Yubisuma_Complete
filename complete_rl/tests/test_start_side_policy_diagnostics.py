"""Tests for start-side opening-chain diagnostics."""

from __future__ import annotations

import unittest
from collections import Counter

from complete_rl.start_side_policy_diagnostics import expectation_warnings


class StartSidePolicyDiagnosticsTests(unittest.TestCase):
    def test_none_policy_warns_on_anti_counter_opening(self) -> None:
        warnings = expectation_warnings(
            {
                "policy": "none_lowest",
                "first_skills": Counter({"フェイント": 4}),
            }
        )
        self.assertTrue(any("0%カウンター" in item for item in warnings))

    def test_counter_policy_warns_when_anti_counter_opening_is_too_low(self) -> None:
        warnings = expectation_warnings(
            {
                "policy": "counter_uniform",
                "first_skills": Counter({"フェイント": 3, "ガード": 2}),
            }
        )
        self.assertTrue(any("100%カウンター" in item for item in warnings))

    def test_counter_policy_accepts_anti_counter_opening(self) -> None:
        warnings = expectation_warnings(
            {
                "policy": "counter_lowest",
                "first_skills": Counter({"フェイント": 4, "ロック": 1}),
            }
        )
        self.assertFalse(warnings)


if __name__ == "__main__":
    unittest.main()

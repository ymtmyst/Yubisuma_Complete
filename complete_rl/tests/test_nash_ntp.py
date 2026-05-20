"""Tests for the Nash NTP policy computation."""

from __future__ import annotations

import unittest

from complete_solver import RulesConfig, State
from complete_rl.nash_ntp import compute_nash_ntp_strategies


class TestComputeNashNTPStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.strategies = compute_nash_ntp_strategies(
            RulesConfig(), max_states=30, vi_epsilon=1e-2
        )

    def test_returns_non_empty_dict(self) -> None:
        self.assertGreater(len(self.strategies), 0)

    def test_initial_state_has_entry(self) -> None:
        self.assertIn(State(), self.strategies)

    def test_probs_sum_to_one(self) -> None:
        for state, (acts, probs) in self.strategies.items():
            with self.subTest(state=state):
                self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_probs_non_negative(self) -> None:
        for state, (acts, probs) in self.strategies.items():
            for p in probs:
                self.assertGreaterEqual(p, 0.0)

    def test_acts_and_probs_same_length(self) -> None:
        for state, (acts, probs) in self.strategies.items():
            self.assertEqual(len(acts), len(probs))

    def test_mirror_config_works(self) -> None:
        strats = compute_nash_ntp_strategies(
            RulesConfig(enable_mirror=True), max_states=20, vi_epsilon=1e-2
        )
        self.assertGreater(len(strats), 0)


if __name__ == "__main__":
    unittest.main()

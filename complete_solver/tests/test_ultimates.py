from __future__ import annotations

import unittest

from complete_solver import NTPAction, PlayerState, State, TPAction, transition
from complete_solver.constants import BLOCK, BOOST, NONE, SKIP, TIME


class UltimateGoldenTests(unittest.TestCase):
    def test_boost_consumes_ultimate_and_grants_three_turn_chain(self) -> None:
        result = transition(State(), TPAction(BOOST, thumb=0), NTPAction(NONE, thumb=0))

        self.assertTrue(result.same_turn_player)
        self.assertTrue(result.next_state.me.used_ultimate)
        self.assertEqual(result.next_state.me_extra_turns, 2)
        self.assertIn("boost", result.events)

    def test_block_consumes_ntp_ultimate_and_stops_boost_but_not_ultimate_use(self) -> None:
        result = transition(State(), TPAction(BOOST, thumb=0), NTPAction(BLOCK, thumb=0))

        self.assertFalse(result.same_turn_player)
        self.assertTrue(result.next_state.me.used_ultimate)
        self.assertTrue(result.next_state.opp.used_ultimate)
        self.assertEqual(result.next_state.opp_extra_turns, 0)
        self.assertIn("blocked", result.events)

    def test_block_does_not_stop_skip_effect(self) -> None:
        result = transition(State(), TPAction(SKIP, thumb=0), NTPAction(BLOCK, thumb=0))

        # True skip (2026-07-13): the blocked opponent's next phase is
        # consumed at the turn switch, so the mover continues immediately.
        self.assertTrue(result.same_turn_player)
        self.assertTrue(result.next_state.opp.used_ultimate)
        self.assertEqual(result.next_state.opp.skip_phases, 0)
        self.assertIn("block_failed_against_skip", result.events)
        self.assertIn("skip_applied", result.events)
        self.assertIn("phase_skipped", result.events)

    def test_time_sets_field_effect_and_interrupts_opponent_extra_turn(self) -> None:
        time_result = transition(State(), TPAction(TIME, thumb=0), NTPAction(NONE, thumb=0))

        self.assertTrue(time_result.same_turn_player)
        self.assertTrue(time_result.next_state.me.used_ultimate)
        self.assertTrue(time_result.next_state.me.time_active)
        self.assertIn("time_set", time_result.events)

        interrupted = transition(
            State(opp=PlayerState(time_active=True)),
            TPAction(BOOST, thumb=0),
            NTPAction(NONE, thumb=0),
        )

        self.assertFalse(interrupted.same_turn_player)
        self.assertFalse(interrupted.next_state.me.time_active)
        self.assertIn("time_interrupted_extra_turn", interrupted.events)


if __name__ == "__main__":
    unittest.main()

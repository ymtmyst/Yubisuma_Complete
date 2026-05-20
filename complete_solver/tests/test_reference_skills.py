from __future__ import annotations

import unittest

from complete_solver import NTPAction, PlayerState, State, TPAction, transition
from complete_solver.constants import (
    ALL,
    CHARGE,
    CHOICE,
    DROP,
    FEINT,
    FLASH,
    GUARD,
    NONE,
)


class ReferenceSkillGoldenTests(unittest.TestCase):
    def test_choice_executes_selected_stock_without_consuming_stock(self) -> None:
        state = State(me=PlayerState(stock=frozenset({CHARGE, FLASH})))

        result = transition(
            state,
            TPAction(CHOICE, thumb=0, choice=CHARGE),
            NTPAction(NONE, thumb=0),
        )

        self.assertFalse(result.same_turn_player)
        self.assertEqual(result.next_state.opp.stock, frozenset({CHARGE, FLASH}))
        self.assertEqual(result.next_state.opp.choice_used_this_phase, frozenset({CHARGE}))
        self.assertTrue(result.next_state.opp.stock_alpha_used_this_phase)
        self.assertTrue(result.next_state.opp.charge_active)
        self.assertIn("choice_used", result.events)

    def test_all_executes_every_stocked_skill_and_clears_stock(self) -> None:
        state = State(me=PlayerState(stock=frozenset({CHARGE, GUARD})))

        result = transition(
            state,
            TPAction(ALL, thumb=0, all_order=(CHARGE,)),
            NTPAction(NONE, thumb=0),
        )

        self.assertTrue(result.same_turn_player)
        self.assertEqual(result.next_state.me.stock, frozenset())
        self.assertTrue(result.next_state.me.charge_active)
        self.assertTrue(result.next_state.me.guard_active)
        self.assertTrue(result.next_state.me.stock_alpha_used_this_phase)
        self.assertIn("all_used", result.events)

    def test_drop_blocks_current_stock_for_opponent_and_grants_extra_turn(self) -> None:
        stock = frozenset({FEINT, FLASH})
        state = State(me=PlayerState(stock=stock))

        result = transition(state, TPAction(DROP, thumb=0), NTPAction(NONE, thumb=0))

        self.assertTrue(result.same_turn_player)
        self.assertEqual(result.next_state.opp.drop_blocked_skills, stock)
        self.assertEqual(result.next_state.me.stock, stock)
        self.assertTrue(result.next_state.me.stock_alpha_used_this_phase)
        self.assertIn("drop_applied", result.events)


if __name__ == "__main__":
    unittest.main()

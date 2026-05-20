from __future__ import annotations

import unittest

from complete_solver import (
    NTPAction,
    PlayerState,
    RulesConfig,
    State,
    TPAction,
    legal_ntp_actions,
    legal_tp_actions,
    transition,
)
from complete_solver.constants import (
    ALL,
    CHARGE,
    COUNTER,
    FEINT,
    FLASH,
    GUARD,
    MIRROR_MAIN,
    MIRROR_PREP,
    NONE,
    REVERSI,
    STOCK,
)


class CompleteTransitionTests(unittest.TestCase):
    def test_number_hit_lowers_turn_player_and_switches_perspective(self) -> None:
        result = transition(State(), TPAction(2, thumb=1), NTPAction(NONE, thumb=1))

        self.assertIsNone(result.terminal_reward)
        self.assertFalse(result.same_turn_player)
        self.assertEqual(result.next_state.opp.hands, 1)
        self.assertTrue(result.next_state.opp.has_declared_skill)

    def test_countered_number_hit_lowers_non_turn_player(self) -> None:
        result = transition(State(), TPAction(2, thumb=1), NTPAction(COUNTER, thumb=1))

        self.assertIsNone(result.terminal_reward)
        self.assertEqual(result.next_state.me.hands, 1)
        self.assertIn("counter_number_hit_opp_scores", result.events)

    def test_flash_can_win_when_both_players_have_declared(self) -> None:
        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
        )

        result = transition(state, TPAction(FLASH, thumb=1), NTPAction(NONE, thumb=1))

        self.assertEqual(result.terminal_reward, 1.0)
        self.assertIsNone(result.next_state)

    def test_stock_is_a_finite_set_without_duplicates(self) -> None:
        state = State(previous_skill=FLASH, me=PlayerState(stock=frozenset({FLASH})))

        result = transition(state, TPAction(STOCK, thumb=0), NTPAction(NONE, thumb=0))

        self.assertEqual(result.next_state.opp.stock, frozenset({FLASH}))

    def test_all_forces_every_stocked_skill_even_if_order_is_partial(self) -> None:
        state = State(me=PlayerState(stock=frozenset({CHARGE, GUARD})))

        result = transition(state, TPAction(ALL, thumb=0, all_order=(CHARGE,)), NTPAction(NONE, thumb=0))

        self.assertTrue(result.same_turn_player)
        self.assertEqual(result.next_state.me.stock, frozenset())
        self.assertTrue(result.next_state.me.charge_active)
        self.assertTrue(result.next_state.me.guard_active)

    def test_mirror_reaction_reflects_number_effect(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))
        config = RulesConfig(enable_mirror=True)

        result = transition(
            state,
            TPAction(2, thumb=1),
            NTPAction(MIRROR_MAIN, thumb=1),
            config,
        )

        self.assertEqual(result.next_state.me.hands, 1)
        self.assertFalse(result.next_state.me.mirror_ready)
        self.assertIn("mirror_number", result.events)

    def test_reversi_swaps_hands_and_buffs_but_not_stock_skip_or_time(self) -> None:
        state = State(
            me=PlayerState(
                hands=1,
                guard_active=True,
                skip_phases=2,
                stock=frozenset({FLASH}),
                time_active=True,
            ),
            opp=PlayerState(
                hands=2,
                cement=1,
                charge_active=True,
                stock=frozenset({FEINT}),
            ),
        )

        result = transition(
            state,
            TPAction(REVERSI, thumb=0),
            NTPAction(NONE, thumb=0),
            RulesConfig(enable_reversi=True),
        )

        self.assertEqual(result.next_state.me.hands, 1)
        self.assertEqual(result.next_state.opp.hands, 2)
        self.assertEqual(result.next_state.me.stock, frozenset({FEINT}))
        self.assertEqual(result.next_state.opp.stock, frozenset({FLASH}))
        self.assertEqual(result.next_state.opp.skip_phases, 1)
        self.assertTrue(result.next_state.opp.time_active)

    def test_mirror_legality_follows_config(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))

        off_reactions = {action.reaction for action in legal_ntp_actions(state)}
        on_reactions = {
            action.reaction
            for action in legal_ntp_actions(state, RulesConfig(enable_mirror=True))
        }

        self.assertNotIn(MIRROR_MAIN, off_reactions)
        self.assertIn(MIRROR_MAIN, on_reactions)

        off_skills = {action.skill for action in legal_tp_actions(State())}
        on_skills = {
            action.skill
            for action in legal_tp_actions(State(), RulesConfig(enable_mirror=True))
        }

        self.assertNotIn(MIRROR_PREP, off_skills)
        self.assertIn(MIRROR_PREP, on_skills)


if __name__ == "__main__":
    unittest.main()

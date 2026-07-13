from __future__ import annotations

import unittest

from complete_solver import NTPAction, PlayerState, RulesConfig, State, TPAction, transition
from complete_solver.constants import (
    ALL,
    BOOST,
    CHARGE,
    CHOICE,
    CEMENT,
    DROP,
    FEINT,
    FLASH,
    GUARD,
    LOCK,
    MIRROR_MAIN,
    MIRROR_PREP,
    NONE,
    QUICK,
    REVERSI,
    SKIP,
    TIME,
)


class MirrorReversiGoldenTests(unittest.TestCase):
    def test_mirror_flash_reflects_two_hand_drop_to_turn_player(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))

        result = transition(
            state,
            TPAction(FLASH, thumb=1),
            NTPAction(MIRROR_MAIN, thumb=1),
            RulesConfig(enable_mirror=True),
        )

        self.assertEqual(result.next_state.me.hands, 0)
        self.assertFalse(result.next_state.me.mirror_ready)
        self.assertIn("mirror_flash_two_hands", result.events)
        self.assertIn("mirror_used", result.events)

    def test_mirror_skip_is_not_reflectable_and_original_skip_applies(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))

        result = transition(
            state,
            TPAction(SKIP, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )

        # True skip (2026-07-13): the mirror user's skipped phase is consumed
        # at the turn switch, so the skipper keeps the turn.
        self.assertTrue(result.same_turn_player)
        self.assertEqual(result.next_state.opp.skip_phases, 0)
        self.assertFalse(result.next_state.opp.mirror_ready)
        self.assertIn("skip_applied", result.events)
        self.assertIn("mirror_not_reflectable", result.events)
        self.assertIn("mirror_used", result.events)
        self.assertIn("phase_skipped", result.events)

    def test_mirror_does_not_reflect_buffs_or_mirror_preparation(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))

        guard_result = transition(
            state,
            TPAction(GUARD, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertTrue(guard_result.same_turn_player)
        self.assertTrue(guard_result.next_state.me.guard_active)
        self.assertFalse(guard_result.next_state.opp.mirror_ready)
        self.assertIn("guard_extra_turn", guard_result.events)
        self.assertIn("mirror_not_reflectable", guard_result.events)

        charge_result = transition(
            state,
            TPAction(CHARGE, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertTrue(charge_result.next_state.opp.charge_active)
        self.assertFalse(charge_result.next_state.me.mirror_ready)
        self.assertIn("charge_set", charge_result.events)
        self.assertIn("mirror_not_reflectable", charge_result.events)

        quick_result = transition(
            state,
            TPAction(QUICK, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertEqual(quick_result.next_state.opp.quick_level, 2)
        self.assertFalse(quick_result.next_state.me.mirror_ready)
        self.assertIn("quick", quick_result.events)
        self.assertIn("mirror_not_reflectable", quick_result.events)

        mirror_prep_result = transition(
            state,
            TPAction(MIRROR_PREP, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertTrue(mirror_prep_result.next_state.opp.mirror_ready)
        self.assertFalse(mirror_prep_result.next_state.me.mirror_ready)
        self.assertIn("mirror_ready", mirror_prep_result.events)
        self.assertIn("mirror_not_reflectable", mirror_prep_result.events)

    def test_reversi_swaps_only_hands_buffs_and_debuffs(self) -> None:
        state = State(
            me=PlayerState(
                hands=1,
                guard_active=True,
                skip_phases=2,
                stock=frozenset({FLASH}),
                time_active=True,
                used_ultimate=True,
            ),
            opp=PlayerState(
                hands=2,
                cement=1,
                charge_active=True,
                quick_level=2,
                mirror_ready=True,
                stock=frozenset({CHARGE}),
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
        self.assertFalse(result.next_state.me.guard_active)
        self.assertEqual(result.next_state.opp.cement, 1)
        self.assertTrue(result.next_state.opp.charge_active)
        self.assertEqual(result.next_state.opp.quick_level, 2)
        self.assertTrue(result.next_state.opp.mirror_ready)
        self.assertEqual(result.next_state.me.stock, frozenset({CHARGE}))
        self.assertEqual(result.next_state.opp.stock, frozenset({FLASH}))
        # True skip (2026-07-13): skip is consumed when the turn passes
        # TO the skipped player, not when their own turn ends.
        self.assertEqual(result.next_state.opp.skip_phases, 2)
        self.assertTrue(result.next_state.opp.time_active)
        self.assertTrue(result.next_state.opp.used_ultimate)
        self.assertIn("reversi", result.events)

    def test_mirror_reflects_cement_and_drop(self) -> None:
        state = State(
            me=PlayerState(stock=frozenset({FLASH})),
            opp=PlayerState(mirror_ready=True, stock=frozenset({CHARGE})),
        )

        cement_result = transition(
            state,
            TPAction(CEMENT, thumb=2),
            NTPAction(MIRROR_MAIN, thumb=1),
            RulesConfig(enable_mirror=True),
        )
        self.assertEqual(cement_result.next_state.opp.cement, 2)
        self.assertEqual(cement_result.next_state.me.cement, 1)
        self.assertIn("mirror_cement", cement_result.events)

        drop_result = transition(
            state,
            TPAction(DROP, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertEqual(drop_result.next_state.opp.drop_blocked_skills, frozenset({CHARGE}))
        self.assertEqual(drop_result.next_state.me_extra_turns, 1)
        self.assertIn("mirror_drop", drop_result.events)

    def test_mirror_does_not_trigger_anti_counter_or_reflect_ultimates(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))

        feint_result = transition(
            state,
            TPAction(FEINT, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertEqual(feint_result.next_state.opp.hands, 2)
        self.assertIn("feint_no_counter", feint_result.events)
        self.assertIn("mirror_not_reflectable", feint_result.events)

        lock_result = transition(
            state,
            TPAction(LOCK, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertFalse(lock_result.next_state.me.lock_pending)
        self.assertIn("lock_no_counter", lock_result.events)
        self.assertIn("mirror_not_reflectable", lock_result.events)

        boost_result = transition(
            state,
            TPAction(BOOST, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertTrue(boost_result.same_turn_player)
        self.assertEqual(boost_result.next_state.me_extra_turns, 2)
        self.assertIn("boost", boost_result.events)
        self.assertIn("mirror_not_reflectable", boost_result.events)

        time_result = transition(
            state,
            TPAction(TIME, thumb=0),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertTrue(time_result.same_turn_player)
        self.assertTrue(time_result.next_state.me.time_active)
        self.assertIn("time_set", time_result.events)
        self.assertIn("mirror_not_reflectable", time_result.events)

    def test_mirror_resolves_reference_skills_by_referenced_skill(self) -> None:
        state = State(
            me=PlayerState(stock=frozenset({FLASH, CHARGE})),
            opp=PlayerState(mirror_ready=True),
        )

        reflected = transition(
            state,
            TPAction(CHOICE, thumb=1, choice=FLASH),
            NTPAction(MIRROR_MAIN, thumb=1),
            RulesConfig(enable_mirror=True),
        )
        self.assertEqual(reflected.next_state.me.hands, 0)
        self.assertEqual(reflected.next_state.opp.choice_used_this_phase, frozenset({FLASH}))
        self.assertIn("mirror_flash_two_hands", reflected.events)
        self.assertIn("choice_used", reflected.events)

        not_reflected = transition(
            state,
            TPAction(CHOICE, thumb=0, choice=CHARGE),
            NTPAction(MIRROR_MAIN, thumb=0),
            RulesConfig(enable_mirror=True),
        )
        self.assertTrue(not_reflected.next_state.opp.charge_active)
        self.assertIn("mirror_reference_not_reflectable", not_reflected.events)
        self.assertIn("choice_used", not_reflected.events)

    def test_mirror_reflects_every_reflectable_skill_in_all(self) -> None:
        state = State(
            me=PlayerState(stock=frozenset({FLASH, DROP, CEMENT})),
            opp=PlayerState(mirror_ready=True, stock=frozenset({CHARGE})),
        )

        result = transition(
            state,
            TPAction(ALL, thumb=1, all_order=(FLASH, DROP, CEMENT)),
            NTPAction(MIRROR_MAIN, thumb=1),
            RulesConfig(enable_mirror=True),
        )

        self.assertEqual(result.next_state.me.hands, 0)
        self.assertEqual(result.next_state.opp.drop_blocked_skills, frozenset({CHARGE}))
        self.assertEqual(result.next_state.opp.cement, 1)
        self.assertEqual(result.next_state.opp.stock, frozenset())
        self.assertEqual(result.next_state.me_extra_turns, 1)
        self.assertIn("mirror_flash_two_hands", result.events)
        self.assertIn("mirror_drop", result.events)
        self.assertIn("mirror_cement", result.events)
        self.assertIn("all_used", result.events)


if __name__ == "__main__":
    unittest.main()

"""Golden tests for opening restrictions and compound skill interactions."""

from __future__ import annotations

import unittest

from complete_solver import NTPAction, PlayerState, State, TPAction, legal_ntp_actions, transition
from complete_solver.constants import BLOCK, COPY, COUNTER, FLASH, LOCK, NONE


class OpeningRestrictionTests(unittest.TestCase):
    def test_flash_cannot_win_before_both_players_have_declared(self) -> None:
        """Flash can drop hands to 0 but cannot end the game on the very first turn."""
        state = State()  # initial: opp.has_declared_skill=False
        result = transition(state, TPAction(FLASH, thumb=1), NTPAction(NONE, thumb=1))

        self.assertIsNone(result.terminal_reward)
        self.assertIn("flash_two_hands", result.events)
        # TP's hands dropped to 0; after perspective switch TP appears as opp
        self.assertEqual(result.next_state.opp.hands, 0)

    def test_flash_wins_after_both_players_have_declared(self) -> None:
        """Flash terminates the game once both players have declared at least once."""
        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
        )
        result = transition(state, TPAction(FLASH, thumb=1), NTPAction(NONE, thumb=1))

        self.assertEqual(result.terminal_reward, 1.0)
        self.assertIsNone(result.next_state)


class GuardTests(unittest.TestCase):
    def test_ntp_guard_absorbs_tp_flash_and_prevents_two_hand_drop(self) -> None:
        """NTP's guard blocks TP's Flash, consuming the guard without a hand drop."""
        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True, guard_active=True),
        )
        result = transition(state, TPAction(FLASH, thumb=1), NTPAction(NONE, thumb=1))

        self.assertIsNone(result.terminal_reward)
        self.assertIn("flash_guarded", result.events)
        # NTP's guard consumed (former NTP is now me after perspective switch)
        self.assertFalse(result.next_state.me.guard_active)
        # TP's hands unchanged (former TP is now opp after perspective switch)
        self.assertEqual(result.next_state.opp.hands, 2)

    def test_tp_guard_absorbs_counter_flash_and_prevents_ntp_drop(self) -> None:
        """TP's guard blocks the two-hand drop caused by a countered Flash."""
        state = State(
            me=PlayerState(has_declared_skill=True, guard_active=True),
            opp=PlayerState(has_declared_skill=True),
        )
        result = transition(state, TPAction(FLASH, thumb=1), NTPAction(COUNTER, thumb=1))

        self.assertIsNone(result.terminal_reward)
        self.assertIn("counter_flash_guarded", result.events)
        # TP's guard consumed; NTP's hands unchanged
        self.assertFalse(result.next_state.opp.guard_active)   # former TP (now opp)
        self.assertEqual(result.next_state.me.hands, 2)        # former NTP (now me)


class ChargeTests(unittest.TestCase):
    def test_charge_causes_number_to_fire_twice_on_hit(self) -> None:
        """Charge is consumed when a number is declared and causes it to fire twice."""
        state = State(
            me=PlayerState(charge_active=True),
            opp=PlayerState(has_declared_skill=True),
        )
        # TP declares total=2 (thumb=1), NTP shows thumb=1 → total=2 → hit; fires twice
        result = transition(state, TPAction(2, thumb=1), NTPAction(NONE, thumb=1))

        self.assertIn("charge_consumed", result.events)
        self.assertEqual(result.events.count("number_hit"), 2)
        # TP had 2 hands, lost 2 (double hit) → 0 hands → terminal win (both declared)
        self.assertEqual(result.terminal_reward, 1.0)

    def test_charge_does_not_fire_twice_on_miss(self) -> None:
        """Charge fires the number twice only if the first hit lands; a miss stops the chain."""
        state = State(
            me=PlayerState(charge_active=True),
            opp=PlayerState(has_declared_skill=True),
        )
        # TP declares total=2 (thumb=1), NTP shows thumb=0 → total=1 ≠ 2 → miss; no second fire
        result = transition(state, TPAction(2, thumb=1), NTPAction(NONE, thumb=0))

        self.assertIn("charge_consumed", result.events)
        self.assertEqual(result.events.count("number_miss"), 1)
        self.assertNotIn("number_hit", result.events)
        self.assertIsNone(result.terminal_reward)


class CopyTests(unittest.TestCase):
    def test_copy_executes_previous_number_twice_and_can_win(self) -> None:
        """Copy re-executes the previous number declaration twice."""
        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
            previous_skill=3,
        )
        # TP declares Copy (thumb=2), NTP shows thumb=1 → total=3=previous → referenced hit × 2
        result = transition(state, TPAction(COPY, thumb=2), NTPAction(NONE, thumb=1))

        self.assertIn("copy_used", result.events)
        # TP had 2 hands → 2 hits → 0 hands → terminal win
        self.assertEqual(result.terminal_reward, 1.0)

    def test_copy_fails_when_no_previous_skill(self) -> None:
        """Copy does nothing if there is no previous skill to reference."""
        result = transition(State(), TPAction(COPY, thumb=0), NTPAction(NONE, thumb=0))

        self.assertIn("copy_failed", result.events)
        self.assertIsNone(result.terminal_reward)


class TrueSkipTests(unittest.TestCase):
    """True skip (designer confirmation 2026-07-13: PASS is not a real
    action): when the turn would pass to a skipped player, their phase is
    consumed instantly — time-based effects still tick — and either their
    Time fires (they take the turn, fresh phase) or the mover continues
    with a fresh phase of their own."""

    def _state_with_skipped_opponent(self, time_active: bool) -> State:
        from complete_solver.constants import CHARGE

        return State(
            me=PlayerState(has_declared_skill=True, guard_active=True,
                           stock_alpha_used_this_phase=True),
            opp=PlayerState(
                has_declared_skill=True,
                skip_phases=1,
                time_active=time_active,
                quick_level=2,                   # decays during skipped phase
                guard_active=True,               # shield expires at own phase
                stock_alpha_used_this_phase=True,
                stock=frozenset({FLASH}),
            ),
            me_guard_extra_used_this_phase=True,
        )

    def test_skipped_phase_consumed_and_mover_continues(self) -> None:
        from complete_solver.constants import CHARGE

        result = transition(
            self._state_with_skipped_opponent(time_active=False),
            TPAction(CHARGE, thumb=0), NTPAction(NONE, thumb=0),
        )
        self.assertIn("phase_skipped", result.events)
        self.assertTrue(result.same_turn_player)      # mover acts again
        state = result.next_state
        skipped = state.opp
        self.assertEqual(skipped.skip_phases, 0)      # phase consumed
        self.assertEqual(skipped.quick_level, 1)      # time still passes
        self.assertFalse(skipped.guard_active)        # shield expired
        self.assertFalse(skipped.stock_alpha_used_this_phase)
        # Mover starts a FRESH phase: shield/phase flags reset.
        self.assertFalse(state.me.guard_active)
        self.assertFalse(state.me.stock_alpha_used_this_phase)
        self.assertFalse(state.me_guard_extra_used_this_phase)
        self.assertTrue(state.me.charge_active)       # the declaration held

    def test_skipped_player_with_time_takes_the_turn(self) -> None:
        from complete_solver.constants import CHARGE

        result = transition(
            self._state_with_skipped_opponent(time_active=True),
            TPAction(CHARGE, thumb=0), NTPAction(NONE, thumb=0),
        )
        self.assertIn("time_skip_interrupt", result.events)
        self.assertFalse(result.same_turn_player)     # the OTHER player moves
        state = result.next_state
        me = state.me                                  # = the skipped player
        self.assertFalse(me.time_active)               # time consumed
        self.assertEqual(me.skip_phases, 0)
        self.assertEqual(me.quick_level, 1)
        self.assertFalse(me.guard_active)
        self.assertFalse(state.me_guard_extra_used_this_phase)
        # The mover who was cut off sits as opp with their declaration held.
        self.assertTrue(state.opp.charge_active)

    def test_pass_is_not_a_legal_action_anywhere(self) -> None:
        from complete_solver.actions import legal_tp_actions
        from complete_solver.constants import PASS

        state = State(me=PlayerState(skip_phases=1))
        skills = {a.skill for a in legal_tp_actions(state)}
        self.assertNotIn(PASS, skills)


class GuardOncePerPhaseTests(unittest.TestCase):
    """New rules 2026-07-13: the WHOLE guard effect (shield + extra turn)
    fires once per phase; later guards in the same phase are fully inert."""

    def test_second_guard_in_same_phase_is_fully_inert(self) -> None:
        from complete_solver.constants import GUARD

        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
            me_guard_extra_used_this_phase=True,  # guard already fired
        )
        result = transition(state, TPAction(GUARD, thumb=0), NTPAction(NONE, thumb=0))

        self.assertIn("guard_inert_this_phase", result.events)
        self.assertFalse(result.same_turn_player)  # no extra turn
        # Shield NOT re-armed (former me is opp after the switch).
        self.assertFalse(result.next_state.opp.guard_active)

    def test_copied_guard_after_guard_fired_is_inert(self) -> None:
        from complete_solver.constants import GUARD

        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
            previous_skill=GUARD,
            me_guard_extra_used_this_phase=True,
        )
        result = transition(state, TPAction(COPY, thumb=0), NTPAction(NONE, thumb=0))
        self.assertFalse(result.same_turn_player)
        self.assertFalse(result.next_state.opp.guard_active)

    def test_first_guard_fires_shield_and_extra_turn(self) -> None:
        from complete_solver.constants import GUARD

        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
        )
        result = transition(state, TPAction(GUARD, thumb=0), NTPAction(NONE, thumb=0))
        self.assertIn("guard_extra_turn", result.events)
        self.assertTrue(result.same_turn_player)
        self.assertTrue(result.next_state.me.guard_active)


class ReferencedAntiCounterTests(unittest.TestCase):
    """Anti-counter skills referenced via Copy/Choice/All (designer ruling
    2026-07-13): inert without a counter, active (per reference multiplier)
    when countered."""

    def test_copied_feint_without_counter_does_nothing(self) -> None:
        from complete_solver.constants import FEINT

        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
            previous_skill=FEINT,
        )
        result = transition(state, TPAction(COPY, thumb=0), NTPAction(NONE, thumb=0))

        self.assertIsNone(result.terminal_reward)
        self.assertFalse(result.same_turn_player)  # no extra turns granted
        self.assertIn("referenced_anti_counter_inert", result.events)
        # TP's hands unchanged (former TP is opp after the switch).
        self.assertEqual(result.next_state.opp.hands, 2)

    def test_copied_feint_with_counter_fires_twice_and_wins(self) -> None:
        from complete_solver.constants import FEINT

        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
            previous_skill=FEINT,
        )
        result = transition(
            state, TPAction(COPY, thumb=0), NTPAction(COUNTER, thumb=0)
        )
        # Feint fires twice: 2 hands down from 2 → terminal win for TP.
        self.assertEqual(result.terminal_reward, 1.0)

    def test_choice_feint_without_counter_does_nothing(self) -> None:
        from complete_solver.constants import CHOICE, FEINT

        state = State(
            me=PlayerState(has_declared_skill=True, stock=frozenset({FEINT})),
            opp=PlayerState(has_declared_skill=True),
        )
        result = transition(
            state,
            TPAction(CHOICE, thumb=0, choice=FEINT),
            NTPAction(NONE, thumb=0),
        )
        self.assertIsNone(result.terminal_reward)
        self.assertFalse(result.same_turn_player)
        self.assertEqual(result.next_state.opp.hands, 2)

    def test_copied_lock_without_counter_does_not_lock(self) -> None:
        state = State(
            me=PlayerState(has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
            previous_skill=LOCK,
        )
        result = transition(state, TPAction(COPY, thumb=0), NTPAction(NONE, thumb=0))
        self.assertFalse(result.next_state.me.lock_pending)
        self.assertFalse(result.next_state.opp.lock_pending)


class LockTests(unittest.TestCase):
    def test_lock_success_sets_lock_pending_on_ntp(self) -> None:
        """Lock skill against NTP who counters → lock_pending set on NTP (former NTP is now me)."""
        result = transition(State(), TPAction(LOCK, thumb=0), NTPAction(COUNTER, thumb=0))

        self.assertIn("lock_success", result.events)
        # After perspective switch, former NTP (locked) becomes me
        self.assertTrue(result.next_state.me.lock_pending)

    def test_lock_pending_removes_counter_and_block_from_ntp_reactions(self) -> None:
        """When opp has lock_pending, only NONE is legal as NTP reaction."""
        state = State(opp=PlayerState(lock_pending=True))
        reactions = {action.reaction for action in legal_ntp_actions(state)}

        self.assertNotIn(COUNTER, reactions)
        self.assertNotIn(BLOCK, reactions)
        self.assertIn(NONE, reactions)

    def test_lock_no_counter_when_ntp_does_not_counter(self) -> None:
        """Lock with no counter reaction leaves NTP unlocked."""
        result = transition(State(), TPAction(LOCK, thumb=0), NTPAction(NONE, thumb=0))

        self.assertIn("lock_no_counter", result.events)
        self.assertFalse(result.next_state.me.lock_pending)
        self.assertFalse(result.next_state.opp.lock_pending)


if __name__ == "__main__":
    unittest.main()

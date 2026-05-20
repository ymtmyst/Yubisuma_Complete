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

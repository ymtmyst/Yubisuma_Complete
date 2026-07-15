"""Differential test: the packed engine must EXACTLY match the reference.

Random playouts (biased toward skill usage so that copy/stock/choice/all/
drop/charge/quick/lock/boost/time branches are all exercised) compare, at
every visited state:
  1. pack → unpack round trip,
  2. the legal TP/NTP action sets,
  3. every legal joint action's transition result (child state, terminal
     reward, same-turn flag).
Any mismatch prints the offending state/action for reproduction.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from complete_solver.actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from complete_solver.constants import FEINT, FLASH, GUARD, LOCK, SKIP
from complete_solver.packed_engine import (
    FULL_ALPHABET_MASK,
    PASS_CODE,
    SKILL_ID,
    legal_ntp_codes,
    legal_tp_codes,
    ntp_action_to_code,
    pack_state,
    step,
    tp_action_to_code,
    unpack_state,
)
from complete_solver.state import State, initial_state
from complete_solver.transition import transition

FULL_CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)
A5 = frozenset({FEINT, LOCK, FLASH, GUARD, SKIP})
A5_MASK = sum(1 << SKILL_ID[s] for s in A5)
RESTRICTED_CONFIG = RulesConfig(False, False, stock_alphabet=A5)


def assert_state_matches(test, config, alphabet_mask, state, depth_events, max_stock=99):
    lane0, lane1 = pack_state(state)

    # 1. round trip
    test.assertEqual(unpack_state(lane0, lane1), state, "pack/unpack mismatch")

    # 2. legal actions
    ref_tp = legal_tp_actions(state, config)
    ref_ntp = legal_ntp_actions(state, config)
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    n_tp = legal_tp_codes(lane0, lane1, alphabet_mask, max_stock, tp_buf)
    n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)

    ref_tp_codes = sorted(tp_action_to_code(a) for a in ref_tp)
    ref_ntp_codes = sorted(ntp_action_to_code(a) for a in ref_ntp)
    test.assertEqual(
        sorted(int(c) for c in tp_buf[:n_tp]),
        ref_tp_codes,
        f"TP action set mismatch at {state}",
    )
    test.assertEqual(
        sorted(int(c) for c in ntp_buf[:n_ntp]),
        ref_ntp_codes,
        f"NTP action set mismatch at {state}",
    )

    # 3. every joint transition
    for tp_action in ref_tp:
        tp_code = tp_action_to_code(tp_action)
        for ntp_action in ref_ntp:
            ntp_code = ntp_action_to_code(ntp_action)
            ref = transition(state, tp_action, ntp_action, config)
            for event in ref.events:
                depth_events.add(event)
            child0, child1, status, reward = step(
                lane0, lane1, tp_code, ntp_code, alphabet_mask
            )
            context = f"state={state} tp={tp_action} ntp={ntp_action}"
            if ref.terminal_reward is not None:
                test.assertEqual(int(status), 2, f"terminal missed: {context}")
                test.assertEqual(
                    float(reward), ref.terminal_reward, f"reward: {context}"
                )
            else:
                test.assertNotEqual(int(status), 2, f"false terminal: {context}")
                test.assertEqual(
                    bool(status == 1),
                    ref.same_turn_player,
                    f"same-turn flag: {context}",
                )
                test.assertEqual(
                    unpack_state(int(child0), int(child1)),
                    ref.next_state,
                    f"child state: {context}",
                )


def run_playouts(test, config, alphabet_mask, seed, games, max_steps=40, max_stock=99):
    rng = random.Random(seed)
    events: set = set()
    for _ in range(games):
        state = initial_state()
        for _ in range(max_steps):
            assert_state_matches(test, config, alphabet_mask, state, events, max_stock)
            tp_actions = legal_tp_actions(state, config)
            ntp_actions = legal_ntp_actions(state, config)
            # Bias toward skills (non-number actions) to exercise rare branches.
            skills_only = [a for a in tp_actions if not isinstance(a.skill, int)]
            pool = skills_only if skills_only and rng.random() < 0.7 else tp_actions
            tp_action = rng.choice(pool)
            ntp_action = rng.choice(ntp_actions)
            result = transition(state, tp_action, ntp_action, config)
            if result.next_state is None:
                break
            state = result.next_state
    return events


class TestPackedEngineDifferential(unittest.TestCase):
    def test_full_game_playouts(self):
        events = run_playouts(
            self, FULL_CONFIG, FULL_ALPHABET_MASK, seed=1, games=60
        )
        # Branch coverage: the playouts must actually have exercised the
        # interesting rules, otherwise this differential test proves little.
        for expected in (
            "number_hit", "number_miss", "counter_number_miss_me_scores",
            "flash", "cement_applied", "guard_extra_turn", "charge_set",
            "charge_consumed", "quick", "skip_applied", "phase_skipped",
            "feint_success", "feint_no_counter", "lock_success", "blocked",
            "copy_used", "stock_added", "choice_used", "all_used",
            "drop_applied", "boost", "time_set", "counter_no_effect",
        ):
            self.assertTrue(
                any(event.startswith(expected) for event in events),
                f"playouts never exercised: {expected}",
            )

    def test_restricted_alphabet_playouts(self):
        run_playouts(self, RESTRICTED_CONFIG, A5_MASK, seed=2, games=25)

    def test_directed_special_states(self):
        """States that random playouts rarely reach: skip+time pass turns,
        guard-already-fired phases, anti-counter references."""
        from complete_solver.constants import CEMENT, FEINT as F, GUARD, LOCK as L
        from complete_solver.state import PlayerState

        events: set = set()
        specials = [
            # Opening combo may not reduce the mover to zero before the
            # opponent has made their first declaration.
            State(
                me=PlayerState(hands=1, quick_level=1),
                opp=PlayerState(has_declared_skill=False),
            ),
            State(
                me=PlayerState(hands=2, quick_level=2),
                opp=PlayerState(has_declared_skill=False),
            ),
            # Skipped player holding Time (the skip+time interrupt).
            State(
                me=PlayerState(has_declared_skill=True, skip_phases=1,
                               time_active=True, guard_active=True,
                               stock=frozenset({FLASH})),
                opp=PlayerState(has_declared_skill=True),
                me_guard_extra_used_this_phase=True,
            ),
            # Skipped player without Time (normal pass).
            State(
                me=PlayerState(has_declared_skill=True, skip_phases=2),
                opp=PlayerState(has_declared_skill=True, time_active=True),
            ),
            # Guard already fired this phase; previous skill stockable.
            State(
                me=PlayerState(has_declared_skill=True,
                               stock=frozenset({F, L})),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=GUARD,
                me_guard_extra_used_this_phase=True,
            ),
            # Anti-counter skills in stock with choice bookkeeping.
            State(
                me=PlayerState(has_declared_skill=True,
                               stock=frozenset({F, L, CEMENT}),
                               choice_used_this_phase=frozenset({F})),
                opp=PlayerState(has_declared_skill=True, lock_pending=True),
                previous_skill=F,
            ),
        ]
        for state in specials:
            assert_state_matches(
                self, FULL_CONFIG, FULL_ALPHABET_MASK, state, events
            )

    def test_stock_size_cap_playouts(self):
        config = RulesConfig(False, False, stock_alphabet=A5, max_stock_size=1)
        run_playouts(self, config, A5_MASK, seed=4, games=25, max_stock=1)

    def test_endgame_heavy_playouts(self):
        # Start from (1,1) endgame roots to densely cover endgame branches.
        from complete_solver.endgame_abstraction import h11_root

        rng = random.Random(3)
        events: set = set()
        for _ in range(40):
            state = h11_root()
            for _ in range(30):
                assert_state_matches(
                    self, FULL_CONFIG, FULL_ALPHABET_MASK, state, events
                )
                tp_actions = legal_tp_actions(state, FULL_CONFIG)
                ntp_actions = legal_ntp_actions(state, FULL_CONFIG)
                skills_only = [
                    a for a in tp_actions if not isinstance(a.skill, int)
                ]
                pool = (
                    skills_only
                    if skills_only and rng.random() < 0.7
                    else tp_actions
                )
                tp_action = rng.choice(pool)
                ntp_action = rng.choice(ntp_actions)
                result = transition(state, tp_action, ntp_action, FULL_CONFIG)
                if result.next_state is None:
                    break
                state = result.next_state


if __name__ == "__main__":
    unittest.main()

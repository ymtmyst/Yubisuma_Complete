"""Gate tests for the CHOICE post-reaction (second-mover) fix.

CHOICE (チョイス) is a single declaration (thumb only); the opponent's
reaction is revealed, and only THEN does the mover pick which stocked skill
fires. The engine used to pre-commit the fired skill at declaration time
(one TPAction per stocked skill). The fix collapses same-thumb CHOICE rows
to their per-column (per-reaction) max before every matrix-game solve (see
``complete_solver/choice_collapse.py``).

These tests are the correctness anchor for the fix:
  1. Directional deltas: {feint,flash} and {feint,guard} (a punish-on-counter
     skill paired with a plain skill) must value ABOVE the pre-fix
     (uncollapsed) matrix; single-stock and {feint,lock} (two anti-counter,
     punish-only-on-counter skills) must be UNCHANGED (their raw matrices
     never disagree with their collapse, or the equilibrium never visits the
     column where they would).
  2. Hardcode sanity (user-specified): with {feint, guard} stocked, the
     post-reaction pick fires FEINT under COUNTER and GUARD otherwise.
  3. General dominance: the collapsed (fixed) value is >= holding just ONE of
     the two stocked skills (a fully pre-committed strategy is a special case
     the post-reaction player can always replicate).
  4. Reference (collapsed) == packed VI (collapsed) parity on a small universe
     where CHOICE with 2 choosable stocked skills actually occurs (the
     existing regression suite's universes are deliberately stockless, so
     they cannot exercise this).
"""

from __future__ import annotations

import unittest

import numpy as np

from complete_solver.actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from complete_solver.choice_collapse import collapse_choice_actions
from complete_solver.constants import COUNTER, FEINT, FLASH, GUARD, LOCK, NONE
from complete_solver.finite_horizon import FiniteHorizonSolver, material_leaf_evaluator
from complete_solver.matrix_game import solve_zero_sum_matrix
from complete_solver.packed_engine import unpack_state
from complete_solver.packed_vi import solve_universe
from complete_solver.small_matrix import solve_small_zero_sum
from complete_solver.state import PlayerState, State
from complete_solver.transition import transition

CONFIG = RulesConfig(False, False)
GAMMA = 0.999


def _make_state(stock: frozenset) -> State:
    return State(
        me=PlayerState(hands=2, stock=stock, has_declared_skill=True),
        opp=PlayerState(hands=2, has_declared_skill=True),
        previous_skill=None,
    )


def _old_value(state: State, depth: int) -> float:
    """Pre-fix value: NO collapse, one fully pre-committed row per stocked
    skill (exactly what the engine did before this fix)."""
    tp_actions = legal_tp_actions(state, CONFIG)
    ntp_actions = legal_ntp_actions(state, CONFIG)
    matrix = np.zeros((len(tp_actions), len(ntp_actions)))
    for r, tp in enumerate(tp_actions):
        for c, ntp in enumerate(ntp_actions):
            result = transition(state, tp, ntp, CONFIG)
            if result.terminal_reward is not None:
                matrix[r, c] = result.terminal_reward
                continue
            if depth <= 1:
                v = material_leaf_evaluator(result.next_state)
            else:
                v = _old_value(result.next_state, depth - 1)
            payoff = GAMMA * v
            if not result.same_turn_player:
                payoff = -payoff
            matrix[r, c] = payoff
    return float(solve_zero_sum_matrix(matrix).value)


class ChoiceDeltaDirectionTests(unittest.TestCase):
    """Gate item 1 + item 5 (fix value >= old value, always)."""

    def test_two_choosable_stocks_value_at_least_old(self):
        for stock in (
            frozenset({FEINT, FLASH}),
            frozenset({FEINT, GUARD}),
            frozenset({FEINT, LOCK}),
        ):
            state = _make_state(stock)
            solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
            new_v = solver.value(state, 2)
            old_v = _old_value(state, 2)
            self.assertGreaterEqual(
                new_v, old_v - 1e-9, msg=f"stock={sorted(stock)}: {new_v} < {old_v}"
            )

    def test_feint_flash_and_feint_guard_strictly_improve_at_depth2(self):
        for stock in (frozenset({FEINT, FLASH}), frozenset({FEINT, GUARD})):
            state = _make_state(stock)
            solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
            new_v = solver.value(state, 2)
            old_v = _old_value(state, 2)
            self.assertGreater(
                new_v, old_v + 1e-4,
                msg=f"stock={sorted(stock)}: expected a real improvement, "
                    f"got old={old_v} new={new_v}",
            )

    def test_single_stock_and_double_anti_counter_are_unchanged(self):
        """Single choosable stock (<=1 CHOICE row per thumb) and {feint,lock}
        (both anti-counter — inert except under COUNTER, where the
        opponent's equilibrium avoids countering) must have delta == 0."""
        for stock in (frozenset({FEINT}), frozenset({FEINT, LOCK})):
            state = _make_state(stock)
            solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
            new_v = solver.value(state, 2)
            old_v = _old_value(state, 2)
            self.assertAlmostEqual(
                new_v, old_v, delta=1e-9,
                msg=f"stock={sorted(stock)} expected Delta=0, got {new_v - old_v}",
            )


class HardcodeSanityTests(unittest.TestCase):
    """Gate item 2 (user-specified): {feint, guard} fires FEINT under
    COUNTER and GUARD otherwise, via the post-reaction resolver."""

    def test_feint_guard_post_reaction_pick(self):
        state = _make_state(frozenset({FEINT, GUARD}))
        solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
        solver.value(state, 2)  # warm the recursive _value memo
        ntp_actions = legal_ntp_actions(state, CONFIG)
        counter_actions = [a for a in ntp_actions if a.reaction == COUNTER]
        none_actions = [a for a in ntp_actions if a.reaction == NONE]
        self.assertTrue(counter_actions and none_actions)
        for ntp in counter_actions:
            picked = solver.resolve_choice(state, thumb=ntp.thumb, ntp_action=ntp, depth=2)
            self.assertEqual(picked.choice, FEINT, msg=f"ntp={ntp}")
        for ntp in none_actions:
            picked = solver.resolve_choice(state, thumb=ntp.thumb, ntp_action=ntp, depth=2)
            self.assertEqual(picked.choice, GUARD, msg=f"ntp={ntp}")


class ChoiceDominanceTests(unittest.TestCase):
    """Gate item 3: the collapsed value dominates every fully pre-committed
    single-skill strategy (a max always dominates any one of its rows)."""

    def test_collapsed_value_dominates_each_single_skill_commitment(self):
        stock = frozenset({FEINT, GUARD})
        state = _make_state(stock)
        solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
        fixed_value = solver.value(state, 2)
        for single in (frozenset({FEINT}), frozenset({GUARD})):
            single_state = _make_state(single)
            single_solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
            single_value = single_solver.value(single_state, 2)
            # Holding only one of the two skills is a legal (weaker) subset
            # of options relative to holding both with the corrected CHOICE;
            # the two-skill value must be >= either one-skill value.
            self.assertGreaterEqual(fixed_value, single_value - 1e-6)


class NoOpOnLowStockTests(unittest.TestCase):
    """Gate item 4: 0/1 choosable stock states are UNAFFECTED byte-for-byte
    (the collapse is a mathematical no-op — group size < 2)."""

    def test_no_stock_collapse_is_identity(self):
        state = _make_state(frozenset())
        tp_actions = legal_tp_actions(state, CONFIG)
        matrix = np.zeros((len(tp_actions), 1))
        collapsed_actions, collapsed_matrix, groups = collapse_choice_actions(
            tp_actions, matrix
        )
        self.assertEqual(collapsed_actions, tp_actions)
        self.assertEqual(groups, {})

    def test_single_choosable_stock_collapse_is_identity(self):
        state = _make_state(frozenset({FEINT}))
        tp_actions = legal_tp_actions(state, CONFIG)
        matrix = np.arange(len(tp_actions)).reshape(-1, 1).astype(float)
        collapsed_actions, collapsed_matrix, groups = collapse_choice_actions(
            tp_actions, matrix
        )
        self.assertEqual(collapsed_actions, tp_actions)
        self.assertEqual(groups, {})
        self.assertTrue(np.array_equal(collapsed_matrix, matrix))


class ReferencePackedParityWithActiveChoiceTests(unittest.TestCase):
    """Gate item: reference == packed parity, specifically on states where
    CHOICE has 2 choosable stocked skills (the existing regression suite's
    universes are stockless and never exercise this). Slow (~70s): builds and
    solves a real (1,1)-hands, 2-skill-alphabet closed universe."""

    @classmethod
    def setUpClass(cls):
        root = State(
            me=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
            opp=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
        )
        cls.db, cls.info = solve_universe(
            root, alphabet=frozenset({FEINT, GUARD}), max_stock_size=2,
            gamma=GAMMA, max_states=2_000_000, verbose=False,
        )
        cls.config = RulesConfig(
            False, False, stock_alphabet=frozenset({FEINT, GUARD}), max_stock_size=2
        )

    def test_universe_converged(self):
        self.assertLess(self.info["max_delta"], 1e-9)

    def test_choice_active_states_exist(self):
        n = len(self.db)
        count = 0
        for pos in range(0, n, 37):  # sparse scan, keep it fast
            state = unpack_state(int(self.db.keys0[pos]), int(self.db.keys1[pos]))
            if len(state.me.stock - state.me.choice_used_this_phase) >= 2:
                count += 1
        self.assertGreater(count, 0)

    def test_bellman_residual_via_reference_engine_on_choice_active_states(self):
        rng = np.random.default_rng(0)
        n = len(self.db)
        # Restrict the sample to states with 2 choosable stocked skills (the
        # case the fix actually changes) via a coarse pre-filter, then sample.
        candidates = []
        for pos in range(0, n, 11):
            state = unpack_state(int(self.db.keys0[pos]), int(self.db.keys1[pos]))
            if len(state.me.stock - state.me.choice_used_this_phase) >= 2:
                candidates.append(pos)
                if len(candidates) >= 400:
                    break
        self.assertGreater(len(candidates), 0)
        sample = rng.choice(candidates, size=min(40, len(candidates)), replace=False)

        for pos in sample:
            state = unpack_state(int(self.db.keys0[pos]), int(self.db.keys1[pos]))
            tp_actions = legal_tp_actions(state, self.config)
            ntp_actions = legal_ntp_actions(state, self.config)
            raw = np.empty((len(tp_actions), len(ntp_actions)))
            for r, tp in enumerate(tp_actions):
                for c, ntp in enumerate(ntp_actions):
                    result = transition(state, tp, ntp, self.config)
                    if result.terminal_reward is not None:
                        raw[r, c] = result.terminal_reward
                    else:
                        child_value = self.db.get(result.next_state)
                        self.assertIsNotNone(
                            child_value, f"closure leak: {result.next_state}"
                        )
                        sign = 1.0 if result.same_turn_player else -1.0
                        raw[r, c] = sign * GAMMA * child_value
            _, matrix, _ = collapse_choice_actions(tp_actions, raw)
            expected, _, _ = solve_small_zero_sum(matrix)
            actual = self.db.get(state)
            self.assertAlmostEqual(
                expected, actual, delta=1e-6, msg=f"state={state}"
            )


if __name__ == "__main__":
    unittest.main()

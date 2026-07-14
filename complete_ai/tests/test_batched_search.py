"""BatchedSearcher must equal the reference depth-2 solver; pruning rules."""

from __future__ import annotations

import unittest

import numpy as np

from complete_ai.batched_search import BatchedSearcher
from complete_ai.packed_eval import material_leaf_bits
from complete_ai.tests.test_features_and_targets import random_states
from complete_solver.actions import RulesConfig
from complete_solver.constants import CEMENT, FEINT, FLASH, GUARD, LOCK
from complete_solver.fast_solver import FastHorizonSolver
from complete_solver.finite_horizon import (
    FiniteHorizonSolver,
    material_leaf_evaluator,
)
from complete_solver.packed_engine import (
    SKILL_ID,
    legal_ntp_codes,
    legal_tp_codes,
    pack_state,
    step,
)
from complete_solver.small_matrix import solve_small_zero_sum
from complete_solver.state import PlayerState, State

CONFIG = RulesConfig(False, False)
GAMMA = 0.999
STOCK_CODES = set(range(64 + 9 * 4, 64 + 9 * 4 + 4))


class MaterialLeafSearcher(BatchedSearcher):
    """Searcher whose 'net' is the material leaf — enables exact comparison
    against FastHorizonSolver without any neural nondeterminism."""

    def __init__(self, prune_stock: bool = False):
        super().__init__(model=None, device="cpu", gamma=GAMMA,
                         prune_stock=prune_stock)

    def _net_values(self, keys0, keys1):
        out = np.empty(len(keys0), dtype=np.float32)
        for i in range(len(keys0)):
            out[i] = material_leaf_bits(np.int64(keys0[i]), np.int64(keys1[i]))
        return out


class TestBatchedSearchMatchesReference(unittest.TestCase):
    def test_depth2_values_match_fast_solver(self):
        searcher = MaterialLeafSearcher(prune_stock=False)
        reference = FastHorizonSolver(
            CONFIG, gamma=GAMMA, leaf_evaluator=material_leaf_evaluator
        )
        for state in random_states(seed=11, count=15):
            lane0, lane1 = pack_state(state)
            value, tp_codes, ntp_codes, tp_policy, ntp_policy = searcher.solve(
                lane0, lane1
            )
            self.assertAlmostEqual(
                value, reference.value(state, 2), delta=1e-6,
                msg=f"state={state}",
            )
            self.assertAlmostEqual(sum(tp_policy), 1.0, delta=1e-6)
            self.assertAlmostEqual(sum(ntp_policy), 1.0, delta=1e-6)


class TestDepth3Value(unittest.TestCase):
    def test_depth3_values_match_fast_solver(self):
        searcher = MaterialLeafSearcher(prune_stock=False)
        reference = FastHorizonSolver(
            CONFIG, gamma=GAMMA, leaf_evaluator=material_leaf_evaluator
        )
        for state in random_states(seed=13, count=6):
            lane0, lane1 = pack_state(state)
            self.assertAlmostEqual(
                searcher.value_depth3(lane0, lane1),
                reference.value(state, 3),
                delta=1e-6,
                msg=f"state={state}",
            )


class TestDepth4Value(unittest.TestCase):
    def test_depth4_extends_depth3_composition(self):
        """value_depth4(s) must equal a one-ply LP over the root whose child
        values are value_depth3 — i.e. expand_depth4 is a faithful one-level
        extension of expand_depth3. Holds regardless of the extra-turn
        transposition bug below (both sides inherit it identically)."""
        searcher = MaterialLeafSearcher(prune_stock=True)
        full = np.int64(255)
        no_cap = np.int64(99)
        tp_buf = np.zeros(96, dtype=np.int64)
        ntp_buf = np.zeros(16, dtype=np.int64)
        for state in random_states(seed=17, count=20):
            lane0, lane1 = pack_state(state)
            n_tp = legal_tp_codes(np.int64(lane0), np.int64(lane1),
                                  full, no_cap, tp_buf)
            n_ntp = legal_ntp_codes(np.int64(lane0), np.int64(lane1), ntp_buf)
            matrix = np.empty((n_tp, n_ntp))
            for a in range(n_tp):
                for b in range(n_ntp):
                    c0, c1, status, reward = step(
                        np.int64(lane0), np.int64(lane1),
                        tp_buf[a], ntp_buf[b], full,
                    )
                    if status == 2:
                        matrix[a, b] = reward
                    else:
                        sign = 1.0 if status == 1 else -1.0
                        matrix[a, b] = (
                            sign * GAMMA * searcher.value_depth3(int(c0), int(c1))
                        )
            composed, _, _ = solve_small_zero_sum(matrix)
            self.assertAlmostEqual(
                searcher.value_depth4(lane0, lane1), composed, delta=1e-6,
                msg=f"state={state}",
            )

class TestDegenerateMatrixRepair(unittest.TestCase):
    """Regression for the 2026-07-14 fix. backup_children/root_value_only
    previously discarded _matrix_value's success flag, so when the fast packed
    simplex could not certify a degenerate subgame it silently used the
    (maximin+minimax)/2 PLACEHOLDER as the value — a ~0.04 error that surfaced
    on skip / extra-turn TARGET states (found via depth-4 parity). The fix
    re-solves those rare nodes exactly with solve_small_zero_sum. This state's
    depth-3 tree contains exactly such a 39x9 degenerate matrix."""

    def test_extra_turn_state_matches_exact_solver(self):
        searcher = MaterialLeafSearcher(prune_stock=False)
        exact = FiniteHorizonSolver(
            CONFIG, gamma=GAMMA, leaf_evaluator=material_leaf_evaluator
        )
        state = State(
            me=PlayerState(hands=2, used_ultimate=True, has_declared_skill=True),
            opp=PlayerState(hands=2, has_declared_skill=True),
            previous_skill="ブースト", me_extra_turns=2,
        )
        lane0, lane1 = pack_state(state)
        self.assertAlmostEqual(
            searcher.value_depth3(lane0, lane1), exact.value(state, 3),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            searcher.value_depth4(lane0, lane1), exact.value(state, 4),
            delta=1e-6,
        )


class TestBatchedParity(unittest.TestCase):
    """Batched/threaded helpers must equal the per-state path exactly."""

    def test_solve_batch_matches_per_state(self):
        searcher = MaterialLeafSearcher(prune_stock=True)
        states = random_states(seed=21, count=40)
        packed = [pack_state(s) for s in states]
        k0 = np.array([p[0] for p in packed], dtype=np.int64)
        k1 = np.array([p[1] for p in packed], dtype=np.int64)
        batched = searcher.solve_batch(k0, k1)
        for (lane0, lane1), (bv, btp, bntp, btpp, bntpp) in zip(packed, batched):
            rv, rtp, rntp, rtpp, rntpp = searcher.solve(lane0, lane1)
            self.assertAlmostEqual(bv, rv, delta=1e-9)
            self.assertTrue(np.array_equal(btp, rtp))
            self.assertTrue(np.array_equal(bntp, rntp))
            self.assertTrue(np.allclose(btpp, rtpp, atol=1e-9))
            self.assertTrue(np.allclose(bntpp, rntpp, atol=1e-9))

    def test_value_depth3_batch_matches_per_state(self):
        searcher = MaterialLeafSearcher(prune_stock=True)
        states = random_states(seed=23, count=24)
        packed = [pack_state(s) for s in states]
        k0 = np.array([p[0] for p in packed], dtype=np.int64)
        k1 = np.array([p[1] for p in packed], dtype=np.int64)
        batched = searcher.value_depth3_batch(k0, k1, leaf_budget=5000)
        for (lane0, lane1), bv in zip(packed, batched):
            self.assertAlmostEqual(
                bv, searcher.value_depth3(lane0, lane1), delta=1e-9
            )

    def test_material_depth3_values_matches_naive(self):
        # The threaded deduped anchor path (used by dataset.py) must equal the
        # naive full-width depth-3 material recursion it replaced.
        from complete_ai.packed_eval import (
            depth3_values, material_depth3_values,
        )
        states = random_states(seed=27, count=30)
        packed = [pack_state(s) for s in states]
        k0 = np.ascontiguousarray(np.array([p[0] for p in packed], dtype=np.int64))
        k1 = np.ascontiguousarray(np.array([p[1] for p in packed], dtype=np.int64))
        naive = depth3_values(k0, k1, GAMMA)
        fast = material_depth3_values(k0, k1, GAMMA, n_threads=4)
        self.assertTrue(np.allclose(naive, fast, atol=1e-5))


class TestStockPruning(unittest.TestCase):
    def _tp_codes(self, stock: frozenset, prune: bool = True):
        state = State(
            me=PlayerState(hands=2, stock=stock, has_declared_skill=True),
            opp=PlayerState(hands=2, has_declared_skill=True),
            previous_skill=GUARD,  # stockable previous → STOCK normally legal
        )
        searcher = MaterialLeafSearcher(prune_stock=prune)
        lane0, lane1 = pack_state(state)
        _, tp_codes, _, _, _ = searcher.solve(lane0, lane1)
        return set(int(c) for c in tp_codes)

    def test_two_normal_stocks_block_third(self):
        codes = self._tp_codes(frozenset({FLASH, CEMENT}))
        self.assertFalse(codes & STOCK_CODES)

    def test_anti_counter_pair_allows_third(self):
        codes = self._tp_codes(frozenset({FEINT, LOCK}))
        self.assertTrue(codes & STOCK_CODES)

    def test_three_stocks_block_fourth(self):
        codes = self._tp_codes(frozenset({FEINT, LOCK, FLASH}))
        self.assertFalse(codes & STOCK_CODES)

    def test_one_stock_allows_second(self):
        codes = self._tp_codes(frozenset({FLASH}))
        self.assertTrue(codes & STOCK_CODES)

    def test_prune_off_keeps_stock(self):
        codes = self._tp_codes(frozenset({FLASH, CEMENT}), prune=False)
        self.assertTrue(codes & STOCK_CODES)


if __name__ == "__main__":
    unittest.main()

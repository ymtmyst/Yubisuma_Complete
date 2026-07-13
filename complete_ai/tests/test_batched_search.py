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
from complete_solver.finite_horizon import material_leaf_evaluator
from complete_solver.packed_engine import SKILL_ID, pack_state
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

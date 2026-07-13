"""complete_ai: feature semantics + leaf/target equivalence with references."""

from __future__ import annotations

import random
import unittest

import numpy as np

from complete_ai.features import FEATURE_SIZE, N_PLAYER, features_from_state
from complete_ai.packed_eval import depth2_values, material_leaf_bits
from complete_solver.actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from complete_solver.constants import FEINT, FLASH, GUARD
from complete_solver.fast_solver import FastHorizonSolver
from complete_solver.finite_horizon import material_leaf_evaluator
from complete_solver.packed_engine import pack_state
from complete_solver.state import PlayerState, State, initial_state
from complete_solver.transition import transition

CONFIG = RulesConfig(False, False)
GAMMA = 0.999


def random_states(seed: int, count: int) -> list[State]:
    rng = random.Random(seed)
    states = [initial_state()]
    while len(states) < count:
        state = initial_state()
        for _ in range(rng.randrange(1, 40)):
            tp_actions = legal_tp_actions(state, CONFIG)
            skills = [a for a in tp_actions if not isinstance(a.skill, int)]
            pool = skills if skills and rng.random() < 0.7 else tp_actions
            result = transition(
                state, rng.choice(pool),
                rng.choice(legal_ntp_actions(state, CONFIG)), CONFIG,
            )
            if result.next_state is None:
                break
            state = result.next_state
            if rng.random() < 0.4 and len(states) < count:
                states.append(state)
    return states


class TestFeatures(unittest.TestCase):
    def test_shape_and_range(self):
        for state in random_states(seed=1, count=50):
            feats = features_from_state(state)
            self.assertEqual(feats.shape, (FEATURE_SIZE,))
            self.assertGreaterEqual(feats.min(), 0.0)
            self.assertLessEqual(feats.max(), 1.0)

    def test_semantic_spot_checks(self):
        state = State(
            me=PlayerState(
                hands=1, cement=1, guard_active=True,
                stock=frozenset({FEINT, FLASH}), used_ultimate=True,
                has_declared_skill=True,
            ),
            opp=PlayerState(hands=2, quick_level=2, has_declared_skill=True),
            previous_skill=GUARD,
            me_extra_turns=2,
        )
        feats = features_from_state(state)
        self.assertAlmostEqual(feats[0], 0.5)      # me hands 1/2
        self.assertAlmostEqual(feats[1], 0.5)      # me cement 1/2
        self.assertAlmostEqual(feats[2], 1.0)      # me guard
        self.assertAlmostEqual(feats[8], 1.0)      # me ultimate used
        self.assertAlmostEqual(feats[20], 0.5)     # |stock|=2 → 2/4
        self.assertAlmostEqual(feats[21], 1.0)     # feint stocked → anti-counter
        self.assertAlmostEqual(feats[22], 1.0)     # flash stocked → normal
        self.assertAlmostEqual(feats[N_PLAYER + 0], 1.0)   # opp hands 2/2
        self.assertAlmostEqual(feats[N_PLAYER + 4], 1.0)   # opp quick 2/2
        self.assertAlmostEqual(feats[2 * N_PLAYER + 21], 2 / 7)  # extra turns
        # previous_skill one-hot: exactly one bit set in the prev block.
        prev_block = feats[2 * N_PLAYER: 2 * N_PLAYER + 21]
        self.assertEqual(int(prev_block.sum()), 1)


class TestPackedEvalMatchesReference(unittest.TestCase):
    def test_material_leaf_differential(self):
        for state in random_states(seed=2, count=120):
            lane0, lane1 = pack_state(state)
            fast = material_leaf_bits(np.int64(lane0), np.int64(lane1))
            self.assertAlmostEqual(
                fast, material_leaf_evaluator(state), delta=1e-12,
                msg=f"state={state}",
            )

    def test_wide_action_state_does_not_overflow(self):
        # Full stock + 3 thumbs yields ~78 TP actions — above the old 64-row
        # buffers that caused a native heap-corruption crash (exit 116).
        from complete_solver.constants import CEMENT, CHARGE, LOCK, QUICK, SKIP

        full = frozenset({FEINT, FLASH, GUARD, LOCK, CEMENT, CHARGE, QUICK, SKIP})
        state = State(
            me=PlayerState(hands=2, stock=full, has_declared_skill=True),
            opp=PlayerState(hands=2, has_declared_skill=True),
            previous_skill=GUARD,
        )
        n_actions = len(legal_tp_actions(state, CONFIG))
        self.assertGreater(n_actions, 64)
        self.assertLessEqual(n_actions, 96)
        lane0, lane1 = pack_state(state)
        keys0 = np.array([lane0], dtype=np.int64)
        keys1 = np.array([lane1], dtype=np.int64)
        value = float(depth2_values(keys0, keys1, GAMMA)[0])
        solver = FastHorizonSolver(
            CONFIG, gamma=GAMMA, leaf_evaluator=material_leaf_evaluator
        )
        self.assertAlmostEqual(value, solver.value(state, 2), delta=1e-5)

    def test_depth2_targets_match_fast_solver(self):
        states = random_states(seed=3, count=12)
        keys0 = np.empty(len(states), dtype=np.int64)
        keys1 = np.empty(len(states), dtype=np.int64)
        for i, state in enumerate(states):
            keys0[i], keys1[i] = pack_state(state)
        batch = depth2_values(keys0, keys1, GAMMA)
        solver = FastHorizonSolver(
            CONFIG, gamma=GAMMA, leaf_evaluator=material_leaf_evaluator
        )
        for i, state in enumerate(states):
            self.assertAlmostEqual(
                float(batch[i]), solver.value(state, 2), delta=1e-5,
                msg=f"state={state}",
            )


if __name__ == "__main__":
    unittest.main()

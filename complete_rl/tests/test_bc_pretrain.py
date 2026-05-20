"""Tests for behavioral cloning warm-start from the Complete solver."""

from __future__ import annotations

import unittest

import numpy as np

from complete_rl.bc_pretrain import generate_bc_dataset, bc_pretrain
from complete_rl.env import build_canonical_tp_actions
from complete_rl.maskable_ppo import build_model, has_maskable_ppo_dependencies, make_env
from complete_rl.obs import OBS_SIZE
from complete_solver import RulesConfig


class TestGenerateBCDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Small state cap so tests run quickly.
        self.dataset = generate_bc_dataset(max_states=30, vi_epsilon=1e-2)

    def test_dataset_is_non_empty(self) -> None:
        self.assertGreater(len(self.dataset), 0)

    def test_obs_shape(self) -> None:
        for obs, _ in self.dataset:
            self.assertEqual(obs.shape, (OBS_SIZE,))
            self.assertEqual(obs.dtype, np.float32)

    def test_probs_shape_matches_canonical(self) -> None:
        config = RulesConfig()
        n_actions = len(build_canonical_tp_actions(config))
        for _, probs in self.dataset:
            self.assertEqual(probs.shape, (n_actions,))
            self.assertEqual(probs.dtype, np.float32)

    def test_probs_sum_to_one(self) -> None:
        for _, probs in self.dataset:
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)

    def test_probs_non_negative(self) -> None:
        for _, probs in self.dataset:
            self.assertTrue((probs >= 0).all())

    def test_mirror_on_config(self) -> None:
        ds = generate_bc_dataset(
            config=RulesConfig(enable_mirror=True), max_states=20, vi_epsilon=1e-2
        )
        self.assertGreater(len(ds), 0)
        config = RulesConfig(enable_mirror=True)
        n_actions = len(build_canonical_tp_actions(config))
        for _, probs in ds:
            self.assertEqual(probs.shape, (n_actions,))
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)


@unittest.skipUnless(has_maskable_ppo_dependencies(), "MaskablePPO deps not installed")
class TestBCPretrain(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = generate_bc_dataset(max_states=20, vi_epsilon=1e-2)
        env = make_env(seed=0, max_steps=20)
        self.model = build_model(env, seed=0, n_steps=8, batch_size=4, n_epochs=1, verbose=0)

    def test_returns_epoch_losses(self) -> None:
        losses = bc_pretrain(
            self.model, self.dataset, n_epochs=2, learning_rate=1e-3, batch_size=8, seed=0
        )
        self.assertEqual(len(losses), 2)

    def test_losses_are_positive_and_finite(self) -> None:
        losses = bc_pretrain(
            self.model, self.dataset, n_epochs=2, learning_rate=1e-3, batch_size=8, seed=0
        )
        for loss in losses:
            self.assertGreater(loss, 0.0)
            self.assertTrue(np.isfinite(loss))

    def test_empty_dataset_returns_empty_losses(self) -> None:
        losses = bc_pretrain(self.model, [], n_epochs=3)
        self.assertEqual(losses, [])

    def test_model_still_predicts_legal_action_after_bc(self) -> None:
        bc_pretrain(self.model, self.dataset, n_epochs=1, batch_size=8, seed=0)
        env = make_env(seed=42, max_steps=20)
        obs, _ = env.reset(seed=42)
        mask = env.action_masks()
        action, _ = self.model.predict(obs, action_masks=mask)
        self.assertTrue(mask[int(action)])


if __name__ == "__main__":
    unittest.main()

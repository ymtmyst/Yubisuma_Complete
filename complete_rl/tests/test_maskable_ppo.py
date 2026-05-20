"""Smoke tests for the MaskablePPO training baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complete_rl.maskable_ppo import (
    ALL_CONFIGS,
    TRAINING_PRESETS,
    build_model,
    evaluate_saved_model,
    evaluate_saved_model_directory,
    has_maskable_ppo_dependencies,
    make_env,
    train_maskable_ppo,
    train_maskable_ppo_all_configs,
    train_maskable_ppo_multi_seed,
)


@unittest.skipUnless(has_maskable_ppo_dependencies(), "MaskablePPO deps are not installed")
class MaskablePPOBaselineTests(unittest.TestCase):
    def test_untrained_model_predicts_legal_masked_action(self) -> None:
        env = make_env(seed=123, max_steps=20)
        model = build_model(
            env,
            seed=123,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )
        obs, _ = env.reset(seed=123)
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask)
        self.assertTrue(mask[int(action)])

    def test_tiny_training_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo(
                total_timesteps=8,
                output_dir=Path(tmp),
                seed=321,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
            )
            self.assertEqual(result.evaluation.episodes, 1)
            self.assertTrue(result.artifacts.model_path)
            self.assertTrue(result.artifacts.metrics_path)
            self.assertTrue(result.artifacts.model_path.exists())
            self.assertTrue(result.artifacts.metrics_path.exists())

    def test_smoke_preset_has_tiny_training_shape(self) -> None:
        preset = TRAINING_PRESETS["smoke"]
        self.assertEqual(preset.total_timesteps, 8)
        self.assertEqual(preset.eval_episodes, 1)

    def test_multi_seed_training_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo_multi_seed(
                seeds=(11, 12),
                output_dir=Path(tmp),
                total_timesteps=8,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
            )
            self.assertEqual(len(result.runs), 2)
            self.assertTrue(result.summary_path)
            self.assertTrue(result.metrics_path)
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.metrics_path.exists())
            for run in result.runs:
                self.assertTrue(run.model_path)
                self.assertTrue(run.metrics_path)
                self.assertTrue(run.model_path.exists())
                self.assertTrue(run.metrics_path.exists())

            reused = train_maskable_ppo_multi_seed(
                seeds=(11, 12),
                output_dir=Path(tmp),
                total_timesteps=8,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
            )
            self.assertTrue(all(run.reused_existing for run in reused.runs))

            different_policy = train_maskable_ppo_multi_seed(
                seeds=(11,),
                output_dir=Path(tmp),
                total_timesteps=8,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                ntp_policy="none",
                verbose=0,
            )
            self.assertFalse(different_policy.runs[0].reused_existing)

            different_reward = train_maskable_ppo_multi_seed(
                seeds=(11,),
                output_dir=Path(tmp),
                total_timesteps=8,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                reward_mode="material",
                verbose=0,
            )
            self.assertFalse(different_reward.runs[0].reused_existing)

    def test_all_configs_training_writes_combined_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo_all_configs(
                seeds=(7,),
                output_dir=Path(tmp),
                total_timesteps=8,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
            )
            self.assertEqual(len(result.configs), len(ALL_CONFIGS))
            self.assertTrue(result.summary_path)
            self.assertTrue(result.metrics_path)
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.metrics_path.exists())
            for config_run in result.configs:
                self.assertEqual(len(config_run.result.runs), 1)
                self.assertTrue(config_run.result.runs[0].model_path)
                self.assertTrue(config_run.result.runs[0].model_path.exists())

    def test_saved_model_can_be_evaluated_with_named_ntp_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo(
                total_timesteps=8,
                output_dir=Path(tmp),
                seed=99,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
            )
            evaluation = evaluate_saved_model(
                result.artifacts.model_path,
                episodes=1,
                seed=1234,
                max_steps=20,
                ntp_policy="none",
            )
            self.assertEqual(evaluation.episodes, 1)

    def test_saved_model_directory_evaluation_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_maskable_ppo_multi_seed(
                seeds=(21,),
                output_dir=root,
                total_timesteps=8,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
            )
            result = evaluate_saved_model_directory(
                root,
                output_dir=root / "eval",
                ntp_policies=("random", "none"),
                episodes=1,
                max_steps=20,
            )
            self.assertEqual(len(result.rows), 2)
            self.assertTrue(result.summary_path)
            self.assertTrue(result.metrics_path)
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.metrics_path.exists())


if __name__ == "__main__":
    unittest.main()

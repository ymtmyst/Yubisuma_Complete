"""Smoke tests for the MaskablePPO training baseline."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from complete_rl.maskable_ppo import (
    ALL_CONFIGS,
    TRAINING_PRESETS,
    build_model,
    evaluate_model,
    evaluate_saved_model,
    evaluate_saved_model_directory,
    finetune_saved_model,
    has_maskable_ppo_dependencies,
    make_env,
    train_maskable_ppo,
    train_maskable_ppo_all_configs,
    train_maskable_ppo_multi_seed,
)
from complete_rl.env import build_canonical_tp_actions
from complete_solver import RulesConfig
from complete_solver.constants import COPY, FEINT


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
            payload = json.loads(result.artifacts.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("hyperparameters", payload)
            self.assertEqual(payload["hyperparameters"]["n_steps"], 8)
            self.assertEqual(payload["hyperparameters"]["batch_size"], 4)
            self.assertEqual(payload["hyperparameters"]["ent_coef"], 0.0)
            self.assertEqual(payload["hyperparameters"]["max_steps"], 20)

    def test_evaluation_scores_terminal_reward_from_start_side(self) -> None:
        actions = build_canonical_tp_actions(RulesConfig())
        feint_idx = next(i for i, action in enumerate(actions) if action.skill == FEINT)
        copy_idx = next(i for i, action in enumerate(actions) if action.skill == COPY)

        class FeintThenCopyModel:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, _obs, **_kwargs):
                action = feint_idx if self.calls == 0 else copy_idx
                self.calls += 1
                return action, None

        result = evaluate_model(
            FeintThenCopyModel(),
            episodes=1,
            seed=9123,
            max_steps=20,
            deterministic=True,
            ntp_policy="none_lowest",
        )

        self.assertEqual(result.wins, 0)
        self.assertEqual(result.losses, 1)
        self.assertEqual(result.average_reward, -1.0)

    def test_bc_checkpoint_can_be_saved_before_ppo_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo(
                total_timesteps=8,
                output_dir=Path(tmp),
                seed=4321,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
                use_bc_pretrain=True,
                bc_depth=1,
                bc_max_states=1,
                bc_epochs=1,
                save_bc_checkpoint=True,
            )

            self.assertTrue(result.artifacts.bc_checkpoint_path)
            self.assertTrue(result.artifacts.bc_checkpoint_path.exists())
            self.assertEqual(
                result.artifacts.bc_checkpoint_path.name,
                "maskable_ppo_after_bc.zip",
            )
            payload = json.loads(result.artifacts.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["bc_checkpoint_path"],
                str(result.artifacts.bc_checkpoint_path),
            )
            self.assertTrue(payload["hyperparameters"]["save_bc_checkpoint"])

    def test_finetune_artifacts_keep_loaded_model_hyperparameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = train_maskable_ppo(
                total_timesteps=8,
                output_dir=Path(tmp) / "source",
                seed=654,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                ent_coef=0.02,
                eval_episodes=1,
                verbose=0,
            )
            result = finetune_saved_model(
                source.artifacts.model_path,
                output_dir=Path(tmp) / "finetuned",
                seed=655,
                total_timesteps=8,
                max_steps=20,
                eval_episodes=1,
                verbose=0,
            )

            payload = json.loads(result.artifacts.metrics_path.read_text(encoding="utf-8"))
            hyperparameters = payload["hyperparameters"]
            self.assertEqual(hyperparameters["n_steps"], 8)
            self.assertEqual(hyperparameters["batch_size"], 4)
            self.assertEqual(hyperparameters["n_epochs"], 1)
            self.assertEqual(hyperparameters["ent_coef"], 0.02)
            self.assertEqual(
                hyperparameters["source_model_path"],
                str(source.artifacts.model_path),
            )

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

    def test_curriculum_warmup_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo(
                total_timesteps=16,
                output_dir=Path(tmp),
                seed=77,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                verbose=0,
                curriculum_warmup_steps=8,
                curriculum_warmup_policy="none",
            )
            self.assertEqual(result.evaluation.episodes, 1)
            self.assertTrue(result.artifacts.model_path.exists())

    def test_episode_mixed_ntp_training_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train_maskable_ppo(
                total_timesteps=16,
                output_dir=Path(tmp),
                seed=55,
                max_steps=20,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                eval_episodes=1,
                ntp_policy="episode_mixed_basic",
                verbose=0,
            )
            self.assertEqual(result.evaluation.episodes, 1)
            self.assertTrue(result.artifacts.model_path.exists())

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

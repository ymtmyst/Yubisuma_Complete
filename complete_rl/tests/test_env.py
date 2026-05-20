"""Tests for the Complete Gymnasium environment."""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from complete_solver import RulesConfig
from complete_rl import CompleteEnv, MIXED_NTP_POLICIES, OBS_SIZE, REWARD_MODES
from complete_rl.env import (
    NAMED_NTP_POLICIES,
    block_first_ntp_policy,
    build_action_mask,
    build_canonical_tp_actions,
    counter_first_ntp_policy,
    none_ntp_policy,
    resolve_canonical_action,
)
from complete_solver.constants import BLOCK, COUNTER, NONE, PASS
from complete_solver.state import PlayerState, State

_ALL_CONFIGS = (
    RulesConfig(enable_mirror=False, enable_reversi=False),
    RulesConfig(enable_mirror=True,  enable_reversi=False),
    RulesConfig(enable_mirror=False, enable_reversi=True),
    RulesConfig(enable_mirror=True,  enable_reversi=True),
)


class CanonicalActionListTests(unittest.TestCase):
    def test_canonical_list_grows_with_mirror(self) -> None:
        off = build_canonical_tp_actions(RulesConfig(enable_mirror=False))
        on  = build_canonical_tp_actions(RulesConfig(enable_mirror=True))
        self.assertGreater(len(on), len(off))

    def test_canonical_list_grows_with_reversi(self) -> None:
        off = build_canonical_tp_actions(RulesConfig(enable_reversi=False))
        on  = build_canonical_tp_actions(RulesConfig(enable_reversi=True))
        self.assertGreater(len(on), len(off))

    def test_canonical_list_contains_pass(self) -> None:
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                actions = build_canonical_tp_actions(config)
                self.assertTrue(any(a.skill == PASS for a in actions))

    def test_canonical_list_is_unique(self) -> None:
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                actions = build_canonical_tp_actions(config)
                # ALL entries have all_order=() so tuple set-comparison works
                self.assertEqual(len(actions), len(set(actions)))

    def test_all_action_placeholder_filled_on_resolve(self) -> None:
        """resolve_canonical_action fills all_order for ALL from current stock."""
        from complete_solver.constants import ALL, FLASH, CHARGE
        state = State(me=PlayerState(stock=frozenset({FLASH, CHARGE})))
        canonical = next(
            a for a in build_canonical_tp_actions(RulesConfig())
            if a.skill == ALL
        )
        resolved = resolve_canonical_action(canonical, state)
        self.assertEqual(resolved.all_order, (CHARGE, FLASH))  # sorted


class ObservationTests(unittest.TestCase):
    def test_obs_shape_and_dtype(self) -> None:
        env = CompleteEnv()
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (OBS_SIZE,))
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_in_unit_range(self) -> None:
        env = CompleteEnv()
        for _ in range(20):
            obs, _ = env.reset()
            self.assertGreaterEqual(float(obs.min()), 0.0)
            self.assertLessEqual(float(obs.max()), 1.0)

    def test_obs_is_consistent_for_same_state(self) -> None:
        from complete_rl.obs import encode_state
        state = State()
        obs1 = encode_state(state)
        obs2 = encode_state(state)
        np.testing.assert_array_equal(obs1, obs2)

    def test_obs_changes_after_step(self) -> None:
        env = CompleteEnv()
        obs0, _ = env.reset()
        mask = env.action_masks()
        action = int(np.where(mask)[0][0])
        obs1, _, terminated, _, _ = env.step(action)
        if not terminated:
            self.assertFalse(np.array_equal(obs0, obs1))


class ActionMaskTests(unittest.TestCase):
    def test_mask_shape_matches_action_space(self) -> None:
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                env = CompleteEnv(config=config)
                env.reset()
                mask = env.action_masks()
                self.assertEqual(mask.shape, (env.action_space.n,))

    def test_mask_dtype_is_bool(self) -> None:
        env = CompleteEnv()
        env.reset()
        self.assertEqual(env.action_masks().dtype, bool)

    def test_mask_has_at_least_one_legal_action(self) -> None:
        env = CompleteEnv()
        env.reset()
        self.assertGreater(env.action_masks().sum(), 0)

    def test_pass_mask_only_when_skip_active(self) -> None:
        config = RulesConfig()
        canonical = build_canonical_tp_actions(config)
        pass_idx = next(i for i, a in enumerate(canonical) if a.skill == PASS)

        # Normal initial state: PASS should be illegal
        normal_state = State()
        normal_mask = build_action_mask(canonical, normal_state, config)
        self.assertFalse(normal_mask[pass_idx])

        # Skip active: PASS should be the only legal action
        skip_state = State(me=PlayerState(skip_phases=1))
        skip_mask = build_action_mask(canonical, skip_state, config)
        self.assertTrue(skip_mask[pass_idx])
        self.assertEqual(skip_mask.sum(), 1)


class EpisodeTests(unittest.TestCase):
    def _run_random_episode(self, env: CompleteEnv, max_steps: int = 500) -> dict:
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while steps < max_steps:
            mask = env.action_masks()
            idx = int(np.random.choice(np.where(mask)[0]))
            obs, reward, terminated, truncated, info = env.step(idx)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        return {"steps": steps, "reward": total_reward, "terminated": terminated}

    def test_episode_terminates_with_binary_reward(self) -> None:
        env = CompleteEnv()
        for _ in range(10):
            result = self._run_random_episode(env)
            if result["terminated"]:
                self.assertIn(result["reward"], {1.0, -1.0})

    def test_all_configs_complete_episodes(self) -> None:
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                env = CompleteEnv(config=config)
                result = self._run_random_episode(env)
                # Should terminate (win/loss) within max_steps
                self.assertGreater(result["steps"], 0)

    def test_obs_dtype_throughout_episode(self) -> None:
        env = CompleteEnv()
        obs, _ = env.reset()
        self.assertEqual(obs.dtype, np.float32)
        mask = env.action_masks()
        idx = int(np.where(mask)[0][0])
        obs, _, _, _, _ = env.step(idx)
        self.assertEqual(obs.dtype, np.float32)


class SpaceCompatibilityTests(unittest.TestCase):
    def test_obs_contained_in_observation_space(self) -> None:
        env = CompleteEnv()
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))

    def test_legal_actions_contained_in_action_space(self) -> None:
        env = CompleteEnv()
        env.reset()
        mask = env.action_masks()
        for idx in np.where(mask)[0]:
            self.assertIn(int(idx), env.action_space)

    def test_repr_includes_config(self) -> None:
        env = CompleteEnv(config=RulesConfig(enable_mirror=True))
        self.assertIn("mirror=ON", repr(env))
        self.assertIn("reward_mode=terminal", repr(env))

    def test_four_configs_have_different_n_actions(self) -> None:
        n_actions = [CompleteEnv(config=c).n_actions for c in _ALL_CONFIGS]
        # At minimum the two extremes should differ
        self.assertLess(min(n_actions), max(n_actions))

    def test_reset_seed_makes_step_deterministic(self) -> None:
        env = CompleteEnv()
        obs0, _ = env.reset(seed=42)
        action = int(np.where(env.action_masks())[0][0])
        obs1, reward1, terminated1, truncated1, info1 = env.step(action)

        obs0_repeat, _ = env.reset(seed=42)
        obs2, reward2, terminated2, truncated2, info2 = env.step(action)

        np.testing.assert_array_equal(obs0, obs0_repeat)
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(reward1, reward2)
        self.assertEqual(terminated1, terminated2)
        self.assertEqual(truncated1, truncated2)
        self.assertEqual(info1["events"], info2["events"])

    def test_gymnasium_check_env_smoke(self) -> None:
        check_env(CompleteEnv(), skip_render_check=True)

    def test_material_reward_mode_is_accepted(self) -> None:
        env = CompleteEnv(reward_mode="material")
        obs, _ = env.reset(seed=3)
        action = int(np.where(env.action_masks())[0][0])
        _, reward, _, _, info = env.step(action)
        self.assertIn("material", REWARD_MODES)
        self.assertIsInstance(reward, float)
        self.assertEqual(info["reward_mode"], "material")

    def test_invalid_reward_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompleteEnv(reward_mode="not_a_mode")


class NamedNTPPolicyTests(unittest.TestCase):
    def test_named_policy_strings_are_accepted(self) -> None:
        for name in ("random", *MIXED_NTP_POLICIES, *sorted(NAMED_NTP_POLICIES)):
            with self.subTest(name=name):
                env = CompleteEnv(opponent_policy=name)
                obs, _ = env.reset(seed=1)
                self.assertEqual(obs.shape, (OBS_SIZE,))

    def test_mixed_policy_is_seed_deterministic(self) -> None:
        env = CompleteEnv(opponent_policy="mixed_basic")
        obs0, _ = env.reset(seed=42)
        action = int(np.where(env.action_masks())[0][0])
        obs1, reward1, terminated1, truncated1, info1 = env.step(action)

        obs0_repeat, _ = env.reset(seed=42)
        obs2, reward2, terminated2, truncated2, info2 = env.step(action)

        np.testing.assert_array_equal(obs0, obs0_repeat)
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(reward1, reward2)
        self.assertEqual(terminated1, terminated2)
        self.assertEqual(truncated1, truncated2)
        self.assertEqual(info1["events"], info2["events"])

    def test_invalid_named_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompleteEnv(opponent_policy="not_a_policy")

    def test_basic_named_policy_preferences(self) -> None:
        state = State()
        config = RulesConfig()
        self.assertEqual(none_ntp_policy(state, config).reaction, NONE)
        self.assertEqual(counter_first_ntp_policy(state, config).reaction, COUNTER)
        self.assertEqual(block_first_ntp_policy(state, config).reaction, BLOCK)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Complete Gymnasium environment."""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from complete_solver import RulesConfig
from complete_rl import CompleteEnv, MIXED_NTP_POLICIES, OBS_SIZE, REWARD_MODES, SEPARATED_NTP_POLICIES
from complete_rl.env import (
    NAMED_NTP_POLICIES,
    block_first_ntp_policy,
    build_action_mask,
    build_canonical_tp_actions,
    counter_first_ntp_policy,
    none_ntp_policy,
    resolve_canonical_action,
    separated_ntp_policy,
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

    def test_pass_is_never_legal(self) -> None:
        # True skip (2026-07-13): skipped phases are consumed inside the
        # turn switch, so PASS is not a real action and mover states with
        # skip_phases > 0 do not occur in reachable play.
        config = RulesConfig()
        canonical = build_canonical_tp_actions(config)
        pass_idx = next(
            (i for i, a in enumerate(canonical) if a.skill == PASS), None
        )

        normal_state = State()
        normal_mask = build_action_mask(canonical, normal_state, config)
        if pass_idx is not None:
            self.assertFalse(normal_mask[pass_idx])

        skip_state = State(me=PlayerState(skip_phases=1))
        skip_mask = build_action_mask(canonical, skip_state, config)
        if pass_idx is not None:
            self.assertFalse(skip_mask[pass_idx])
        self.assertGreater(skip_mask.sum(), 0)


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

    def test_step_info_contains_resolved_actions(self) -> None:
        env = CompleteEnv(opponent_policy="none_lowest")
        env.reset(seed=7)
        action = int(np.where(env.action_masks())[0][0])
        _, _, _, _, info = env.step(action)
        self.assertEqual(info["ntp_reaction"], NONE)
        self.assertEqual(info["ntp_thumb"], 0)
        self.assertIn("tp_action", info)

    def test_separated_ntp_policy_reaction_and_thumb(self) -> None:
        state = State()
        config = RulesConfig()
        none_lowest = separated_ntp_policy(state, config, 0.0, "lowest")
        counter_lowest = separated_ntp_policy(state, config, 1.0, "lowest")
        self.assertEqual((none_lowest.reaction, none_lowest.thumb), (NONE, 0))
        self.assertEqual((counter_lowest.reaction, counter_lowest.thumb), (COUNTER, 0))

    def test_separated_ntp_policies_are_accepted(self) -> None:
        for name in SEPARATED_NTP_POLICIES:
            with self.subTest(name=name):
                env = CompleteEnv(opponent_policy=name)
                obs, _ = env.reset(seed=5)
                self.assertEqual(obs.shape, (OBS_SIZE,))

    def test_separated_uniform_policy_is_seed_deterministic(self) -> None:
        env = CompleteEnv(opponent_policy="counter50_uniform")
        obs0, _ = env.reset(seed=123)
        action = int(np.where(env.action_masks())[0][0])
        obs1, r1, t1, tr1, _ = env.step(action)

        obs0b, _ = env.reset(seed=123)
        obs2, r2, t2, tr2, _ = env.step(action)

        np.testing.assert_array_equal(obs0, obs0b)
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(r1, r2)
        self.assertEqual(t1, t2)
        self.assertEqual(tr1, tr2)

    def test_episode_mixed_policies_are_accepted(self) -> None:
        for name in (
            "episode_mixed_basic",
            "episode_separated_basic",
            "episode_weighted_none_counter",
        ):
            with self.subTest(name=name):
                env = CompleteEnv(opponent_policy=name)
                obs, _ = env.reset(seed=5)
                self.assertEqual(obs.shape, (OBS_SIZE,))

    def test_episode_mixed_is_seed_deterministic(self) -> None:
        env = CompleteEnv(opponent_policy="episode_mixed_basic")
        obs0, _ = env.reset(seed=99)
        action = int(np.where(env.action_masks())[0][0])
        obs1, r1, t1, tr1, _ = env.step(action)

        obs0b, _ = env.reset(seed=99)
        obs2, r2, t2, tr2, _ = env.step(action)

        np.testing.assert_array_equal(obs0, obs0b)
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(r1, r2)
        self.assertEqual(t1, t2)
        self.assertEqual(tr1, tr2)

    def test_episode_mixed_changes_ntp_per_episode(self) -> None:
        env = CompleteEnv(opponent_policy="episode_mixed_basic", max_steps=500)
        seen_ids: set[int] = set()
        for seed in range(20):
            env.reset(seed=seed * 1000)
            seen_ids.add(id(env._ntp_policy))
        self.assertGreater(len(seen_ids), 1, "NTP policy fn should vary across episodes")

    def test_episode_separated_policy_uses_seeded_separated_policies(self) -> None:
        env = CompleteEnv(opponent_policy="episode_separated_basic")
        seen_separated: set[str] = set()
        for seed in range(50):
            env.reset(seed=seed)
            if env._separated_ntp_policy_name is not None:
                seen_separated.add(env._separated_ntp_policy_name)
                self.assertEqual(env._ntp_policy, env._seeded_separated_ntp_policy)

        self.assertEqual(
            seen_separated,
            {"none_uniform", "counter50_uniform", "counter_uniform"},
        )

    def test_nash_optimal_policy_returns_legal_action(self) -> None:
        from complete_rl.nash_ntp import compute_nash_ntp_strategies
        from complete_rl.env import _NASH_NTP_STRATEGY_CACHE
        config = RulesConfig()
        _NASH_NTP_STRATEGY_CACHE[config] = compute_nash_ntp_strategies(
            config, max_states=30, vi_epsilon=1e-2
        )
        env = CompleteEnv(opponent_policy="nash_optimal")
        obs, _ = env.reset(seed=7)
        mask = env.action_masks()
        legal = int(np.where(mask)[0][0])
        obs2, reward, terminated, truncated, _ = env.step(legal)
        self.assertEqual(obs2.shape, (OBS_SIZE,))

    def test_nash_optimal_policy_is_seed_deterministic(self) -> None:
        from complete_rl.nash_ntp import compute_nash_ntp_strategies
        from complete_rl.env import _NASH_NTP_STRATEGY_CACHE
        config = RulesConfig()
        _NASH_NTP_STRATEGY_CACHE[config] = compute_nash_ntp_strategies(
            config, max_states=30, vi_epsilon=1e-2
        )
        env = CompleteEnv(opponent_policy="nash_optimal")
        obs0, _ = env.reset(seed=42)
        action = int(np.where(env.action_masks())[0][0])
        obs1, r1, t1, tr1, _ = env.step(action)

        obs0b, _ = env.reset(seed=42)
        obs2, r2, t2, tr2, _ = env.step(action)

        np.testing.assert_array_equal(obs0, obs0b)
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(r1, r2)
        self.assertEqual(t1, t2)
        self.assertEqual(tr1, tr2)


if __name__ == "__main__":
    unittest.main()

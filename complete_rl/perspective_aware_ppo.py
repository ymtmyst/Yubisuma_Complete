"""Perspective-aware MaskablePPO for self-play with turn-switching.

In zero-sum self-play, when the turn switches between players, the next state's
value function is from the OPPONENT's perspective. The standard PPO bootstrap
formula incorrectly treats it as if from the same player's perspective:

    standard:  A_t = R_t + γ · V(s_{t+1}) - V(s_t)
    correct:   A_t = R_t + γ · sign · V(s_{t+1}) - V(s_t)
               where sign = -1 if turn switched between t and t+1, else +1

The same sign correction propagates through the GAE-λ recursion:

    A_t = δ_t + γ · λ · sign · A_{t+1}

This module provides ``PerspectiveMaskableRolloutBuffer`` and
``PerspectiveMaskablePPO`` as drop-in replacements for the standard
``MaskableRolloutBuffer`` / ``MaskablePPO`` from sb3_contrib.

The env (``complete_rl.env.CompleteEnv``) must emit ``same_turn_player`` in
its info dict; ``False`` means the turn switched on this transition.

Usage
-----
>>> from complete_rl.perspective_aware_ppo import PerspectiveMaskablePPO
>>> model = PerspectiveMaskablePPO("MlpPolicy", env, ...)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class PerspectiveMaskableRolloutBuffer(MaskableRolloutBuffer):
    """MaskableRolloutBuffer with turn-switch tracking for GAE sign correction.

    Adds a ``turn_switched`` array of shape (buffer_size, n_envs) and overrides
    ``compute_returns_and_advantage`` to apply the sign flip described in the
    module docstring.
    """

    turn_switched: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.turn_switched = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        *args: Any,
        turn_switched: np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Add transition with optional turn-switch flag.

        ``turn_switched`` must be a bool/float array of shape (n_envs,);
        ``True``/``1`` means the action at this step caused a perspective change
        (env returned ``same_turn_player=False``).
        """
        if turn_switched is not None:
            arr = np.asarray(turn_switched).reshape((self.n_envs,)).astype(np.float32)
            self.turn_switched[self.pos] = arr
        super().add(*args, action_masks=action_masks, **kwargs)

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """Compute returns and advantages with perspective-aware sign correction.

        The recurrence is:
            sign_t  = -1 if turn_switched[t] else +1
            δ_t     = R_t + γ * sign_t * V(s_{t+1}) * non_terminal - V(s_t)
            A_t     = δ_t + γ * λ * sign_t * non_terminal * A_{t+1}
        """
        last_values_np = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            # Convert {0.0, 1.0} → {+1.0, -1.0}
            sign = 1.0 - 2.0 * self.turn_switched[step]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal * sign
                - self.values[step]
            )
            last_gae_lam = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * sign
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


def _extract_turn_switched(infos: list[dict], n_envs: int) -> np.ndarray:
    """Extract ``same_turn_player`` flags from env infos and convert to switch flags.

    Returns an array of shape (n_envs,) where 1.0 = turn switched, 0.0 = same TP.
    For terminal transitions (game ended), info may not contain ``same_turn_player``;
    we default to 0 (no switch) — this is safe because terminal transitions zero
    out the bootstrap via ``next_non_terminal``.
    """
    flags = np.zeros((n_envs,), dtype=np.float32)
    for idx in range(min(len(infos), n_envs)):
        info = infos[idx] if infos[idx] is not None else {}
        same_tp = info.get("same_turn_player", True)
        if not same_tp:
            flags[idx] = 1.0
    return flags


class PerspectiveMaskablePPO(MaskablePPO):
    """MaskablePPO that uses ``PerspectiveMaskableRolloutBuffer`` and forwards
    ``same_turn_player`` from env infos to the buffer.

    Drop-in replacement: same constructor signature as ``MaskablePPO``.
    """

    def __init__(
        self,
        *args: Any,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        **kwargs: Any,
    ) -> None:
        if rollout_buffer_class is None:
            rollout_buffer_class = PerspectiveMaskableRolloutBuffer
        super().__init__(
            *args,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskableRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Collect rollouts and pass per-step turn-switch flags to the buffer.

        Identical to ``MaskablePPO.collect_rollouts`` except for extracting
        ``same_turn_player`` from infos and passing ``turn_switched=...`` to
        ``rollout_buffer.add``.
        """
        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. "
                "Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=action_masks
                )

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            turn_switched = _extract_turn_switched(infos, env.num_envs)

            # Use the buffer's perspective-aware add when available.
            if isinstance(rollout_buffer, PerspectiveMaskableRolloutBuffer):
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    turn_switched=turn_switched,
                    action_masks=action_masks,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    action_masks=action_masks,
                )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)  # type: ignore[arg-type]
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

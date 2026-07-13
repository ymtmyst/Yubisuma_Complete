"""Best-response attack environment (N6, gate ④ 非搾取性の実測).

The ATTACKER (a PPO learner) controls one full seat — declarations AND
reactions — against the frozen search agent. Every ply contains exactly one
attacker decision: its own TP declaration when it holds the turn, or its NTP
reaction when the frozen agent declares. The frozen agent samples from its
root LP mixtures (the deployed policy, stochastic by design).

Action space: Discrete(204) = packed TP codes 0..191 ∪ (192 + ntp codes
0..11), masked by context. Observation: the 103 value-net features (mover
perspective) + one "attacker is mover" flag.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from complete_solver.packed_engine import (
    legal_ntp_codes,
    legal_tp_codes,
    pack_state,
    step,
)
from complete_solver.state import initial_state

from .batched_search import BatchedSearcher, _FULL_MASK, _NO_CAP
from .features import FEATURE_SIZE, features_from_lanes

N_TP_CODES = 192
N_NTP_CODES = 12
N_ACTIONS = N_TP_CODES + N_NTP_CODES
OBS_SIZE = FEATURE_SIZE + 1


class BRAttackEnv(gym.Env):
    """One-seat attacker vs the frozen search agent."""

    metadata = {"render_modes": []}

    def __init__(self, searcher: BatchedSearcher, max_plies: int = 120,
                 seed: int = 0):
        super().__init__()
        self.searcher = searcher
        self.max_plies = max_plies
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self._tp_buf = np.zeros(96, dtype=np.int64)
        self._ntp_buf = np.zeros(16, dtype=np.int64)
        self._episode = 0
        self._init = pack_state(initial_state())

    # ── helpers ───────────────────────────────────────────────────────────

    def _frozen_pick(self, codes, policy) -> int:
        p = np.clip(np.asarray(policy, dtype=np.float64), 0.0, None)
        total = p.sum()
        if total <= 0:
            return int(codes[self.rng.integers(0, len(codes))])
        return int(codes[self.rng.choice(len(p), p=p / total)])

    def _obs(self) -> np.ndarray:
        keys0 = np.array([self._lane0], dtype=np.int64)
        keys1 = np.array([self._lane1], dtype=np.int64)
        feats = features_from_lanes(keys0, keys1)[0]
        return np.concatenate(
            [feats, np.array([1.0 if self._attacker_is_mover else 0.0],
                             dtype=np.float32)]
        )

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self._attacker_is_mover:
            n = legal_tp_codes(np.int64(self._lane0), np.int64(self._lane1),
                               _FULL_MASK, _NO_CAP, self._tp_buf)
            for i in range(n):
                mask[int(self._tp_buf[i])] = True
        else:
            n = legal_ntp_codes(np.int64(self._lane0), np.int64(self._lane1),
                                self._ntp_buf)
            for i in range(n):
                mask[N_TP_CODES + int(self._ntp_buf[i])] = True
        return mask

    # ── gym API ───────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._lane0, self._lane1 = (np.int64(self._init[0]),
                                    np.int64(self._init[1]))
        # Alternate which seat the attacker takes; seat 0 moves first.
        self._attacker_seat = self._episode % 2
        self._episode += 1
        self._mover = 0
        self._attacker_is_mover = (self._mover == self._attacker_seat)
        self._plies = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if self._attacker_is_mover:
            tp_code = action  # masked to legal TP codes
            ntp_code = self._frozen_ntp()
        else:
            tp_code = self._frozen_tp()
            ntp_code = action - N_TP_CODES

        child0, child1, status, reward = step(
            self._lane0, self._lane1,
            np.int64(tp_code), np.int64(ntp_code), _FULL_MASK,
        )
        self._plies += 1

        if int(status) == 2:
            winner = self._mover if reward > 0 else 1 - self._mover
            attacker_reward = 1.0 if winner == self._attacker_seat else -1.0
            return self._obs(), attacker_reward, True, False, {}

        if int(status) == 0:
            self._mover = 1 - self._mover
        self._lane0, self._lane1 = child0, child1
        self._attacker_is_mover = (self._mover == self._attacker_seat)
        truncated = self._plies >= self.max_plies
        return self._obs(), 0.0, False, truncated, {}

    def _frozen_tp(self) -> int:
        _, tp_codes, _, tp_policy, _ = self.searcher.solve(
            int(self._lane0), int(self._lane1)
        )
        return self._frozen_pick(tp_codes, tp_policy)

    def _frozen_ntp(self) -> int:
        _, _, ntp_codes, _, ntp_policy = self.searcher.solve(
            int(self._lane0), int(self._lane1)
        )
        return self._frozen_pick(ntp_codes, ntp_policy)

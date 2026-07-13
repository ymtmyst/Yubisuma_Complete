"""Play agents for evaluation and interactive play (N5).

``SearchAgent`` wraps the depth-2 net-leaf search: it MIXES according to the
root LP equilibrium (the core anti-exploitability property of this project).
``ScriptedAgent`` implements clean fixed baselines — both seats fully
specified, unlike the legacy RL-env "named NTP policies" where the model
under test also played the opponent's declarations.
"""

from __future__ import annotations

import numpy as np

from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes

from .batched_search import BatchedSearcher, _FULL_MASK, _NO_CAP


class SearchAgent:
    """LP-mixture play from the batched net-leaf search."""

    def __init__(self, searcher: BatchedSearcher, rng: np.random.Generator,
                 epsilon: float = 0.0, deterministic: bool = False):
        self.searcher = searcher
        self.rng = rng
        self.epsilon = epsilon
        self.deterministic = deterministic

    def _pick(self, codes, policy) -> int:
        if self.deterministic:
            return int(codes[int(np.argmax(policy))])
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(codes[self.rng.integers(0, len(codes))])
        p = np.clip(np.asarray(policy, dtype=np.float64), 0.0, None)
        total = p.sum()
        if total <= 0:
            return int(codes[self.rng.integers(0, len(codes))])
        return int(codes[self.rng.choice(len(p), p=p / total)])

    def tp_action(self, lane0: int, lane1: int) -> int:
        _, tp_codes, _, tp_policy, _ = self.searcher.solve(lane0, lane1)
        return self._pick(tp_codes, tp_policy)

    def ntp_action(self, lane0: int, lane1: int) -> int:
        _, _, ntp_codes, _, ntp_policy = self.searcher.solve(lane0, lane1)
        return self._pick(ntp_codes, ntp_policy)


class ScriptedAgent:
    """Fixed baseline: uniform-random TP declarations, styled NTP reactions.

    reaction_style:
      "random"  — uniform over legal reactions,
      "none"    — never reacts (reaction なし, random thumb),
      "counter" — counters whenever legal,
      "block"   — blocks if available, else counters, else none.
    """

    def __init__(self, reaction_style: str, rng: np.random.Generator):
        assert reaction_style in ("random", "none", "counter", "block")
        self.style = reaction_style
        self.rng = rng
        self._tp_buf = np.zeros(96, dtype=np.int64)
        self._ntp_buf = np.zeros(16, dtype=np.int64)

    def tp_action(self, lane0: int, lane1: int) -> int:
        n = legal_tp_codes(np.int64(lane0), np.int64(lane1), _FULL_MASK,
                           _NO_CAP, self._tp_buf)
        return int(self._tp_buf[self.rng.integers(0, n)])

    def ntp_action(self, lane0: int, lane1: int) -> int:
        n = legal_ntp_codes(np.int64(lane0), np.int64(lane1), self._ntp_buf)
        codes = [int(c) for c in self._ntp_buf[:n]]
        if self.style == "random":
            return codes[self.rng.integers(0, len(codes))]
        by_reaction: dict[int, list[int]] = {}
        for code in codes:
            by_reaction.setdefault(code // 4, []).append(code)
        if self.style == "none":
            pool = by_reaction[0]
        elif self.style == "counter":
            pool = by_reaction.get(1) or by_reaction[0]
        else:  # block
            pool = by_reaction.get(2) or by_reaction.get(1) or by_reaction[0]
        return pool[self.rng.integers(0, len(pool))]

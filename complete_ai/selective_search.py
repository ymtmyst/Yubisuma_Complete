"""N7-C: selective / best-first deepening (smarter, not deeper).

A uniform depth-d search expands EVERY action to depth d. But in a mixed
equilibrium almost all actions carry ~0 probability, and deepening them barely
moves the root value. Selective deepening instead:

  1. solves the node's matrix with shallow (net) children to get an equilibrium
     mixed strategy;
  2. deepens ONLY the cells in the strategy's support (row support × column
     support) and re-solves.

So the search budget concentrates on the lines actually played, reaching a
larger *effective* depth on critical lines for a fraction of a uniform search's
node count. See ``N7_STRATEGY_AND_DESIGN.md`` Part 4.

This is a correctness-first pure-Python prototype (mirrors how N7-A began):
``tau < 0`` deepens all actions and therefore reproduces the uniform depth-d
value exactly (the reference test); ``tau >= 0`` keeps only actions with
equilibrium probability ``> tau``. A compiled version is future work.
"""

from __future__ import annotations

import numpy as np

from complete_solver.packed_engine import legal_ntp_codes, step
from complete_solver.small_matrix import solve_small_zero_sum

from .batched_search import _FULL_MASK, _legal_tp_pruned
from .graph_teacher import net_values


class SelectiveSearcher:
    """Support-restricted iterative-deepening value search with net leaves."""

    def __init__(self, model, device: str, gamma: float = 0.999,
                 prune: bool = False, depth: int = 4, tau: float = 0.05):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.prune = prune
        self.depth = depth      # acting depth used by solve()
        self.tau = tau          # acting support threshold used by solve()
        self._tp_buf = np.zeros(96, dtype=np.int64)
        self._ntp_buf = np.zeros(16, dtype=np.int64)
        self._solve_cache: dict[tuple[int, int], tuple] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        # deep_cells: matrix cells that were recursively deepened (the search
        # cost); nodes: internal (depth>0) node expansions.
        self.stats = {"deep_cells": 0, "nodes": 0, "solves": 0}

    def _net_one(self, lane0: int, lane1: int) -> float:
        key = (lane0, lane1)
        cached = self._net_memo.get(key)
        if cached is None:
            cached = float(net_values(
                self.model, self.device,
                np.array([lane0], np.int64), np.array([lane1], np.int64))[0])
            self._net_memo[key] = cached
        return cached

    def _legal(self, lane0: int, lane1: int):
        n_tp = _legal_tp_pruned(np.int64(lane0), np.int64(lane1),
                                self.prune, self._tp_buf)
        n_ntp = legal_ntp_codes(np.int64(lane0), np.int64(lane1), self._ntp_buf)
        return (self._tp_buf[:n_tp].copy(), self._ntp_buf[:n_ntp].copy())

    def _children(self, lane0: int, lane1: int, tp_codes, ntp_codes):
        """Build the (n_tp × n_ntp) transition layout for one node.

        Returns arrays over cells: child0/child1 (packed, or -1 if terminal),
        sign (±1 for non-terminal, terminal reward stored in ``const``), and a
        list of unique non-terminal child (key -> local index) for batched net.
        """
        n_tp, n_ntp = len(tp_codes), len(ntp_codes)
        child0 = np.full(n_tp * n_ntp, -1, dtype=np.int64)
        child1 = np.full(n_tp * n_ntp, -1, dtype=np.int64)
        const = np.zeros(n_tp * n_ntp, dtype=np.float64)  # sign, or terminal payoff
        is_term = np.zeros(n_tp * n_ntp, dtype=bool)
        for a in range(n_tp):
            for b in range(n_ntp):
                c0, c1, status, reward = step(
                    np.int64(lane0), np.int64(lane1),
                    np.int64(tp_codes[a]), np.int64(ntp_codes[b]), _FULL_MASK,
                )
                pos = a * n_ntp + b
                if status == 2:
                    is_term[pos] = True
                    const[pos] = reward
                else:
                    child0[pos] = c0
                    child1[pos] = c1
                    const[pos] = 1.0 if status == 1 else -1.0
        return child0, child1, const, is_term

    def _shallow_matrix(self, n_tp, n_ntp, child0, child1, const, is_term):
        """Matrix with net-valued (depth-1) children (transposition-memoized)."""
        cellvals = np.empty(n_tp * n_ntp)
        for pos in range(n_tp * n_ntp):
            if is_term[pos]:
                cellvals[pos] = const[pos]
            else:
                cellvals[pos] = const[pos] * self.gamma * self._net_one(
                    int(child0[pos]), int(child1[pos]))
        return cellvals.reshape(n_tp, n_ntp)

    def value(self, lane0: int, lane1: int, depth: int, tau: float = 0.02) -> float:
        """Selective value of a state to ``depth`` plies (net leaves at depth 0).

        ``tau``: keep actions with equilibrium probability > tau in the support.
        ``tau < 0`` deepens every action ⇒ exact uniform depth-``depth`` value.

        Transposition-memoized within the call (dedup does not change the value,
        only the cost — same invariant the compiled ``expand_depth3`` relies on).
        """
        self._net_memo: dict[tuple[int, int], float] = {}
        self._val_memo: dict[tuple[int, int, int], float] = {}
        return self._value(int(lane0), int(lane1), depth, tau)

    def solve(self, lane0: int, lane1: int):
        """Acting interface compatible with ``BatchedSearcher.solve``.

        Returns ``(value, tp_codes, ntp_codes, tp_policy, ntp_policy)`` using
        selective deepening to ``self.depth`` (support threshold ``self.tau``).
        The mixed strategy comes from the DEEPENED root matrix, so acting reads
        the critical lines deeper than the depth-2 agent — more accurate play at
        a fraction of a uniform deep search's cost. Cached per state (the model
        is fixed during a match/game), matching ``play_match``'s own caching."""
        key = (int(lane0), int(lane1))
        cached = self._solve_cache.get(key)
        if cached is not None:
            return cached
        self._net_memo = {}
        self._val_memo = {}
        tp_codes, ntp_codes = self._legal(key[0], key[1])
        n_tp, n_ntp = len(tp_codes), len(ntp_codes)
        child0, child1, const, is_term = self._children(
            key[0], key[1], tp_codes, ntp_codes)
        shallow = self._shallow_matrix(n_tp, n_ntp, child0, child1, const, is_term)
        v_sh, tp_pol, ntp_pol = solve_small_zero_sum(shallow)
        if self.depth <= 1:
            res = (float(v_sh), tp_codes, ntp_codes, tp_pol, ntp_pol)
            self._solve_cache[key] = res
            return res
        if self.tau < 0:
            sup_tp = range(n_tp)
            sup_ntp = range(n_ntp)
        else:
            sup_tp = [a for a in range(n_tp) if tp_pol[a] > self.tau]
            sup_ntp = [b for b in range(n_ntp) if ntp_pol[b] > self.tau]
        deep = shallow.copy()
        for a in sup_tp:
            for b in sup_ntp:
                pos = a * n_ntp + b
                if is_term[pos]:
                    continue
                deep[a, b] = const[pos] * self.gamma * self._value(
                    int(child0[pos]), int(child1[pos]), self.depth - 1, self.tau)
        v, tp_pol2, ntp_pol2 = solve_small_zero_sum(deep)
        res = (float(v), tp_codes, ntp_codes, tp_pol2, ntp_pol2)
        self._solve_cache[key] = res
        return res

    def _value(self, lane0: int, lane1: int, depth: int, tau: float) -> float:
        if depth <= 0:
            return self._net_one(lane0, lane1)
        memo_key = (lane0, lane1, depth)
        cached = self._val_memo.get(memo_key)
        if cached is not None:
            return cached

        self.stats["nodes"] += 1
        tp_codes, ntp_codes = self._legal(lane0, lane1)
        n_tp, n_ntp = len(tp_codes), len(ntp_codes)
        child0, child1, const, is_term = self._children(
            lane0, lane1, tp_codes, ntp_codes)

        shallow = self._shallow_matrix(n_tp, n_ntp, child0, child1, const, is_term)
        v_sh, tp_pol, ntp_pol = solve_small_zero_sum(shallow)
        self.stats["solves"] += 1
        if depth == 1:
            self._val_memo[memo_key] = float(v_sh)
            return float(v_sh)

        # Support (tau<0 ⇒ deepen all, giving the uniform depth-d value).
        if tau < 0:
            sup_tp = range(n_tp)
            sup_ntp = range(n_ntp)
        else:
            sup_tp = [a for a in range(n_tp) if tp_pol[a] > tau]
            sup_ntp = [b for b in range(n_ntp) if ntp_pol[b] > tau]

        deep = shallow.copy()
        for a in sup_tp:
            for b in sup_ntp:
                pos = a * n_ntp + b
                if is_term[pos]:
                    continue
                self.stats["deep_cells"] += 1
                deep[a, b] = const[pos] * self.gamma * self._value(
                    int(child0[pos]), int(child1[pos]), depth - 1, tau)
        v, _, _ = solve_small_zero_sum(deep)
        self.stats["solves"] += 1
        self._val_memo[memo_key] = float(v)
        return float(v)

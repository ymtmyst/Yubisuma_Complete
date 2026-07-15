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

from complete_solver.choice_collapse import (
    collapse_choice_matrix_packed,
    is_choice_meta_code,
)
from complete_solver.packed_engine import legal_ntp_codes, step
from complete_solver.small_matrix import solve_small_zero_sum

from .batched_search import _FULL_MASK, _legal_tp_pruned
from .graph_teacher import net_values


def _collapse_solve(tp_codes, matrix, collapse=True):
    """Collapse same-thumb CHOICE rows (post-reaction skill pick; see
    choice_collapse.py) and solve the reduced LP. Returns (value,
    collapsed_codes, row_policy, col_policy, groups).

    ``collapse=False`` reproduces the pre-fix (pre-committed CHOICE) behavior
    — solve the raw matrix with no collapse — used ONLY by the CHOICE-fix
    impact measurement harness to get an apples-to-apples old-vs-new value
    through the identical depth-d engine. Production always uses the default
    (``collapse=True``); the searcher's ``collapse_choice`` attribute is True
    unless a measurement explicitly flips it."""
    if not collapse:
        value, row_policy, col_policy = solve_small_zero_sum(matrix)
        return value, np.asarray(tp_codes), row_policy, col_policy, {}
    collapsed_codes, collapsed_matrix, groups = collapse_choice_matrix_packed(
        tp_codes, matrix
    )
    value, row_policy, col_policy = solve_small_zero_sum(collapsed_matrix)
    return value, collapsed_codes, row_policy, col_policy, groups


def _expand_support(tp_codes, collapsed_codes, row_policy, groups, tau):
    """Map the collapsed-row support (probability > tau) back to RAW row
    indices into ``tp_codes`` — the deepening loop indexes cells by raw
    (uncollapsed) row position. A collapsed CHOICE row's whole group is kept
    in support (any member might become the post-reaction argmax once
    deepened)."""
    code_to_idx = {int(c): i for i, c in enumerate(tp_codes)}
    support: list[int] = []
    for k, code in enumerate(collapsed_codes):
        if row_policy[k] <= tau:
            continue
        code = int(code)
        members = groups.get(code)
        if members is None:
            support.append(code_to_idx[code])
        else:
            support.extend(code_to_idx[int(c)] for c in members)
    return support


class SelectiveSearcher:
    """Support-restricted iterative-deepening value search with net leaves."""

    def __init__(self, model, device: str, gamma: float = 0.999,
                 prune: bool = False, depth: int = 4, tau: float = 0.05,
                 collapse_choice: bool = True):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.prune = prune
        self.depth = depth      # acting depth used by solve()
        self.tau = tau          # acting support threshold used by solve()
        # CHOICE post-reaction collapse (the rule fix). Default True = fixed
        # engine. Set False ONLY for the fix-impact measurement harness to
        # reproduce the pre-fix (pre-committed CHOICE) values on the same
        # states/engine. See _collapse_solve.
        self.collapse_choice = collapse_choice
        self._tp_buf = np.zeros(96, dtype=np.int64)
        self._ntp_buf = np.zeros(16, dtype=np.int64)
        self._solve_cache: dict[tuple[int, int], tuple] = {}
        # Per-state CHOICE post-reaction resolution data from solve() (the
        # UNCOLLAPSED final root matrix + tp_codes + collapse groups), keyed
        # like _solve_cache — see resolve_tp_code / choice_collapse.py.
        self._choice_resolve_cache: dict[tuple[int, int], tuple] = {}
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
        v_sh, tp_codes_c, tp_pol, ntp_pol, groups = _collapse_solve(tp_codes, shallow, self.collapse_choice)
        if self.depth <= 1:
            res = (float(v_sh), tp_codes_c, ntp_codes, tp_pol, ntp_pol)
            self._solve_cache[key] = res
            self._choice_resolve_cache[key] = (tp_codes, shallow, groups)
            return res
        if self.tau < 0:
            sup_tp = range(n_tp)
            sup_ntp = range(n_ntp)
        else:
            sup_tp = _expand_support(tp_codes, tp_codes_c, tp_pol, groups, self.tau)
            sup_ntp = [b for b in range(n_ntp) if ntp_pol[b] > self.tau]
        deep = shallow.copy()
        for a in sup_tp:
            for b in sup_ntp:
                pos = a * n_ntp + b
                if is_term[pos]:
                    continue
                deep[a, b] = const[pos] * self.gamma * self._value(
                    int(child0[pos]), int(child1[pos]), self.depth - 1, self.tau)
        v, tp_codes_c2, tp_pol2, ntp_pol2, groups2 = _collapse_solve(tp_codes, deep, self.collapse_choice)
        res = (float(v), tp_codes_c2, ntp_codes, tp_pol2, ntp_pol2)
        self._solve_cache[key] = res
        self._choice_resolve_cache[key] = (tp_codes, deep, groups2)
        return res

    def resolve_tp_code(self, lane0: int, lane1: int, tp_code: int,
                        ntp_code: int) -> int:
        """Post-reaction resolution (mirrors ``BatchedSearcher.resolve_tp_code``):
        turn a sampled TP code from ``solve()`` into a concrete code ``step``
        can execute, picking the stocked skill that maximizes the payoff
        against the REALIZED opponent reaction. No-op for non-CHOICE-meta
        codes. Must be called with the SAME (lane0, lane1) as a preceding
        ``solve()`` call (results are cached, so this also works after
        several other states were solved in between)."""
        if not is_choice_meta_code(tp_code):
            return tp_code
        key = (int(lane0), int(lane1))
        cached = self._choice_resolve_cache.get(key)
        if cached is None:
            self.solve(lane0, lane1)
            cached = self._choice_resolve_cache[key]
        tp_codes, matrix, groups = cached
        candidates = groups[int(tp_code)]
        if len(candidates) == 1:
            return int(candidates[0])
        code_to_row = {int(c): i for i, c in enumerate(tp_codes)}
        ntp_codes = self._solve_cache[key][2]
        col = int(np.nonzero(np.asarray(ntp_codes) == ntp_code)[0][0])
        best_code, best_value = candidates[0], -float("inf")
        for code in candidates:
            value = float(matrix[code_to_row[int(code)], col])
            if value > best_value:
                best_value = value
                best_code = code
        return int(best_code)

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
        v_sh, tp_codes_c, tp_pol, ntp_pol, groups = _collapse_solve(tp_codes, shallow, self.collapse_choice)
        self.stats["solves"] += 1
        if depth == 1:
            self._val_memo[memo_key] = float(v_sh)
            return float(v_sh)

        # Support (tau<0 ⇒ deepen all, giving the uniform depth-d value).
        if tau < 0:
            sup_tp = range(n_tp)
            sup_ntp = range(n_ntp)
        else:
            sup_tp = _expand_support(tp_codes, tp_codes_c, tp_pol, groups, tau)
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
        v, _, _, _, _ = _collapse_solve(tp_codes, deep, self.collapse_choice)
        self.stats["solves"] += 1
        self._val_memo[memo_key] = float(v)
        return float(v)

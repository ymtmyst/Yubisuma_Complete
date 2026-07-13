"""Batched depth-2 LP-backup search with neural leaves (N4 core).

The compiled engine expands a root's depth-2 tree with transposition dedup;
all unique leaves are evaluated by the value network in ONE batch; values are
backed up through LP solves (children: value only, root: value + mixed
strategies). The root value doubles as the training target for fitted
Nash-VI, so acting and target generation share the same computation.

Self-play stock pruning (user domain knowledge, 2026-07-12): STOCK is pruned
when already holding 2 skills — except when those two are exactly the
anti-counter pair (feint+lock), where a third stock is essential. A fourth
stock is never considered. The pruning shapes the AGENT's own policy space;
opponents/verifiers remain free to use unpruned actions.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit
from numba.typed import Dict as NumbaDict
from numba.core import types

from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes, step
from complete_solver.packed_vi import _matrix_value
from complete_solver.small_matrix import solve_small_zero_sum

from .features import features_from_lanes

_KEY_TYPE = types.UniTuple(types.int64, 2)
_FULL_MASK = np.int64(255)
_NO_CAP = np.int64(99)
_STOCK_CODE_BASE = 64 + 9 * 4  # STOCK action codes: base..base+3
_ANTI_PAIR = np.int64((1 << 6) | (1 << 7))  # feint+lock stock mask

MAX_ACTIONS = 96
MAX_CHILDREN = 1200
MAX_CELLS = 2_000_000
MAX_LEAVES = 400_000
# Depth-3 expansion (training targets): level-2 nodes and their cells.
MAX_L2 = 60_000
MAX_L2_CELLS = 8_000_000


@njit(cache=True, nogil=True, inline="always")
def _prune_stock(lane0, code, prune):
    """True → drop this TP action code under self-play stock pruning."""
    if not prune:
        return False
    if code < _STOCK_CODE_BASE or code > _STOCK_CODE_BASE + 3:
        return False
    stock = lane0 >> 18 & 255
    held = 0
    for i in range(8):
        held += stock >> i & 1
    if held < 2:
        return False
    if held == 2 and stock == _ANTI_PAIR:
        return False  # feint+lock pair: the third stock is essential
    return True


@njit(cache=True, nogil=True)
def _legal_tp_pruned(lane0, lane1, prune, buf):
    n = legal_tp_codes(lane0, lane1, _FULL_MASK, _NO_CAP, buf)
    if not prune:
        return n
    kept = 0
    for i in range(n):
        if not _prune_stock(lane0, buf[i], prune):
            buf[kept] = buf[i]
            kept += 1
    return kept


@njit(cache=True, nogil=True)
def expand_depth2(root0, root1, prune,
                  root_tp, root_ntp,
                  root_const, root_child,
                  child_keys0, child_keys1,
                  child_tp_n, child_ntp_n, child_offset,
                  cell_const, cell_leaf,
                  leaf_keys0, leaf_keys1):
    """Expand a depth-2 tree with dedup. Returns packed counts, or (-1,...)
    on buffer overflow (caller should fall back to a smaller depth).

    Cell encoding (root_const/root_child and cell_const/cell_leaf):
      child/leaf index  -1 → the const array holds sign*terminal_reward…
      actually: *_child/*_leaf = -1 → *_const holds the terminal payoff;
      otherwise *_const holds the SIGN (±1) and *_child/*_leaf the index.
    """
    n_tp = _legal_tp_pruned(root0, root1, prune, root_tp)
    n_ntp = legal_ntp_codes(root0, root1, root_ntp)

    child_index = NumbaDict.empty(_KEY_TYPE, types.int64)
    n_children = 0
    for a in range(n_tp):
        for b in range(n_ntp):
            c0, c1, status, reward = step(
                root0, root1, root_tp[a], root_ntp[b], _FULL_MASK
            )
            pos = a * n_ntp + b
            if status == 2:
                root_const[pos] = reward
                root_child[pos] = -1
            else:
                key = (c0, c1)
                if key in child_index:
                    idx = child_index[key]
                else:
                    if n_children >= MAX_CHILDREN:
                        return -1, 0, 0, 0
                    idx = n_children
                    child_index[key] = idx
                    child_keys0[idx] = c0
                    child_keys1[idx] = c1
                    n_children += 1
                root_const[pos] = 1.0 if status == 1 else -1.0
                root_child[pos] = idx

    leaf_index = NumbaDict.empty(_KEY_TYPE, types.int64)
    n_leaves = 0
    cell_pos = 0
    tp_buf = np.zeros(MAX_ACTIONS, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    for i in range(n_children):
        lane0 = child_keys0[i]
        lane1 = child_keys1[i]
        c_tp = _legal_tp_pruned(lane0, lane1, prune, tp_buf)
        c_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
        child_tp_n[i] = c_tp
        child_ntp_n[i] = c_ntp
        child_offset[i] = cell_pos
        if cell_pos + c_tp * c_ntp > MAX_CELLS:
            return -1, 0, 0, 0
        for a in range(c_tp):
            for b in range(c_ntp):
                g0, g1, status, reward = step(
                    lane0, lane1, tp_buf[a], ntp_buf[b], _FULL_MASK
                )
                if status == 2:
                    cell_const[cell_pos] = reward
                    cell_leaf[cell_pos] = -1
                else:
                    key = (g0, g1)
                    if key in leaf_index:
                        idx = leaf_index[key]
                    else:
                        if n_leaves >= MAX_LEAVES:
                            return -1, 0, 0, 0
                        idx = n_leaves
                        leaf_index[key] = idx
                        leaf_keys0[idx] = g0
                        leaf_keys1[idx] = g1
                        n_leaves += 1
                    cell_const[cell_pos] = 1.0 if status == 1 else -1.0
                    cell_leaf[cell_pos] = idx
                cell_pos += 1
    return n_tp, n_ntp, n_children, n_leaves


@njit(cache=True, nogil=True)
def expand_depth3(root0, root1, prune,
                  root_tp, root_ntp, root_const, root_child,
                  l1_keys0, l1_keys1, l1_tp_n, l1_ntp_n, l1_offset,
                  l1_const, l1_idx,
                  l2_keys0, l2_keys1, l2_tp_n, l2_ntp_n, l2_offset,
                  l2_const, l2_idx,
                  leaf_keys0, leaf_keys1):
    """Three-level expansion with per-level transposition dedup.

    Cell encoding as in expand_depth2: idx -1 → const holds the terminal
    payoff; otherwise const holds the sign (±1) and idx the child index in
    the next level (L1 cells point into L2, L2 cells point into leaves).
    Returns (n_tp, n_ntp, nL1, nL2, n_leaves) or (-1, 0, 0, 0, 0) on
    buffer overflow (caller falls back to depth 2).
    """
    n_tp = _legal_tp_pruned(root0, root1, prune, root_tp)
    n_ntp = legal_ntp_codes(root0, root1, root_ntp)

    l1_index = NumbaDict.empty(_KEY_TYPE, types.int64)
    n_l1 = 0
    for a in range(n_tp):
        for b in range(n_ntp):
            c0, c1, status, reward = step(
                root0, root1, root_tp[a], root_ntp[b], _FULL_MASK
            )
            pos = a * n_ntp + b
            if status == 2:
                root_const[pos] = reward
                root_child[pos] = -1
            else:
                key = (c0, c1)
                if key in l1_index:
                    idx = l1_index[key]
                else:
                    if n_l1 >= MAX_CHILDREN:
                        return -1, 0, 0, 0, 0
                    idx = n_l1
                    l1_index[key] = idx
                    l1_keys0[idx] = c0
                    l1_keys1[idx] = c1
                    n_l1 += 1
                root_const[pos] = 1.0 if status == 1 else -1.0
                root_child[pos] = idx

    l2_index = NumbaDict.empty(_KEY_TYPE, types.int64)
    n_l2 = 0
    cell_pos = 0
    tp_buf = np.zeros(MAX_ACTIONS, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    for i in range(n_l1):
        lane0 = l1_keys0[i]
        lane1 = l1_keys1[i]
        c_tp = _legal_tp_pruned(lane0, lane1, prune, tp_buf)
        c_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
        l1_tp_n[i] = c_tp
        l1_ntp_n[i] = c_ntp
        l1_offset[i] = cell_pos
        if cell_pos + c_tp * c_ntp > MAX_CELLS:
            return -1, 0, 0, 0, 0
        for a in range(c_tp):
            for b in range(c_ntp):
                g0, g1, status, reward = step(
                    lane0, lane1, tp_buf[a], ntp_buf[b], _FULL_MASK
                )
                if status == 2:
                    l1_const[cell_pos] = reward
                    l1_idx[cell_pos] = -1
                else:
                    key = (g0, g1)
                    if key in l2_index:
                        idx = l2_index[key]
                    else:
                        if n_l2 >= MAX_L2:
                            return -1, 0, 0, 0, 0
                        idx = n_l2
                        l2_index[key] = idx
                        l2_keys0[idx] = g0
                        l2_keys1[idx] = g1
                        n_l2 += 1
                    l1_const[cell_pos] = 1.0 if status == 1 else -1.0
                    l1_idx[cell_pos] = idx
                cell_pos += 1

    leaf_index = NumbaDict.empty(_KEY_TYPE, types.int64)
    n_leaves = 0
    cell_pos = 0
    for i in range(n_l2):
        lane0 = l2_keys0[i]
        lane1 = l2_keys1[i]
        c_tp = _legal_tp_pruned(lane0, lane1, prune, tp_buf)
        c_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
        l2_tp_n[i] = c_tp
        l2_ntp_n[i] = c_ntp
        l2_offset[i] = cell_pos
        if cell_pos + c_tp * c_ntp > MAX_L2_CELLS:
            return -1, 0, 0, 0, 0
        for a in range(c_tp):
            for b in range(c_ntp):
                g0, g1, status, reward = step(
                    lane0, lane1, tp_buf[a], ntp_buf[b], _FULL_MASK
                )
                if status == 2:
                    l2_const[cell_pos] = reward
                    l2_idx[cell_pos] = -1
                else:
                    key = (g0, g1)
                    if key in leaf_index:
                        idx = leaf_index[key]
                    else:
                        if n_leaves >= MAX_LEAVES:
                            return -1, 0, 0, 0, 0
                        idx = n_leaves
                        leaf_index[key] = idx
                        leaf_keys0[idx] = g0
                        leaf_keys1[idx] = g1
                        n_leaves += 1
                    l2_const[cell_pos] = 1.0 if status == 1 else -1.0
                    l2_idx[cell_pos] = idx
                cell_pos += 1
    return n_tp, n_ntp, n_l1, n_l2, n_leaves


@njit(cache=True, nogil=True)
def backup_children(n_children, child_tp_n, child_ntp_n, child_offset,
                    cell_const, cell_leaf, leaf_values, gamma,
                    matrix, tableau, basis, out_values):
    for i in range(n_children):
        n_tp = child_tp_n[i]
        n_ntp = child_ntp_n[i]
        pos = child_offset[i]
        for a in range(n_tp):
            for b in range(n_ntp):
                leaf = cell_leaf[pos]
                if leaf < 0:
                    matrix[a, b] = cell_const[pos]
                else:
                    matrix[a, b] = cell_const[pos] * gamma * leaf_values[leaf]
                pos += 1
        value, _ = _matrix_value(matrix, n_tp, n_ntp, tableau, basis)
        out_values[i] = value


@njit(cache=True, nogil=True)
def root_value_only(n_tp, n_ntp, root_const, root_child, child_values, gamma,
                    matrix, tableau, basis):
    """Root game VALUE only (no policies) — used by value_depth3, which needs
    just the target. Kept nogil so the whole depth-3 backup is GIL-free and
    the threaded target path scales past the per-state-forward contention."""
    for a in range(n_tp):
        for b in range(n_ntp):
            pos = a * n_ntp + b
            child = root_child[pos]
            if child < 0:
                matrix[a, b] = root_const[pos]
            else:
                matrix[a, b] = root_const[pos] * gamma * child_values[child]
    value, _ = _matrix_value(matrix, n_tp, n_ntp, tableau, basis)
    return value


class BatchedSearcher:
    """Depth-2 LP search with neural leaves for one process/GPU."""

    def __init__(self, model, device: str, gamma: float = 0.999,
                 prune_stock: bool = True):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.prune_stock = prune_stock
        # Reusable expansion buffers.
        self._root_tp = np.zeros(MAX_ACTIONS, dtype=np.int64)
        self._root_ntp = np.zeros(16, dtype=np.int64)
        self._root_const = np.zeros(MAX_ACTIONS * 16, dtype=np.float64)
        self._root_child = np.zeros(MAX_ACTIONS * 16, dtype=np.int64)
        self._child_keys0 = np.zeros(MAX_CHILDREN, dtype=np.int64)
        self._child_keys1 = np.zeros(MAX_CHILDREN, dtype=np.int64)
        self._child_tp_n = np.zeros(MAX_CHILDREN, dtype=np.int16)
        self._child_ntp_n = np.zeros(MAX_CHILDREN, dtype=np.int16)
        self._child_offset = np.zeros(MAX_CHILDREN, dtype=np.int64)
        self._cell_const = np.zeros(MAX_CELLS, dtype=np.float64)
        self._cell_leaf = np.zeros(MAX_CELLS, dtype=np.int64)
        self._leaf_keys0 = np.zeros(MAX_LEAVES, dtype=np.int64)
        self._leaf_keys1 = np.zeros(MAX_LEAVES, dtype=np.int64)
        self._matrix = np.zeros((MAX_ACTIONS, 16), dtype=np.float64)
        self._tableau = np.zeros((97, 120), dtype=np.float64)
        self._basis = np.zeros(96, dtype=np.int64)

    def _net_values(self, keys0: np.ndarray, keys1: np.ndarray) -> np.ndarray:
        feats = features_from_lanes(keys0, keys1)
        with torch.no_grad():
            out = []
            for i in range(0, len(feats), 131072):
                chunk = torch.from_numpy(feats[i:i + 131072]).to(self.device)
                out.append(self.model(chunk).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def solve(self, lane0: int, lane1: int):
        """Depth-2 net-leaf solve. Returns (value, tp_codes, ntp_codes,
        tp_policy, ntp_policy)."""
        n_tp, n_ntp, n_children, n_leaves = expand_depth2(
            np.int64(lane0), np.int64(lane1), self.prune_stock,
            self._root_tp, self._root_ntp,
            self._root_const, self._root_child,
            self._child_keys0, self._child_keys1,
            self._child_tp_n, self._child_ntp_n, self._child_offset,
            self._cell_const, self._cell_leaf,
            self._leaf_keys0, self._leaf_keys1,
        )
        if n_tp < 0:
            raise RuntimeError("expansion buffers exceeded")

        if n_leaves > 0:
            leaf_values = self._net_values(
                self._leaf_keys0[:n_leaves].copy(),
                self._leaf_keys1[:n_leaves].copy(),
            ).astype(np.float64)
        else:
            leaf_values = np.zeros(0, dtype=np.float64)

        child_values = np.zeros(max(n_children, 1), dtype=np.float64)
        if n_children > 0:
            backup_children(
                n_children, self._child_tp_n, self._child_ntp_n,
                self._child_offset, self._cell_const, self._cell_leaf,
                leaf_values, self.gamma,
                self._matrix, self._tableau, self._basis, child_values,
            )

        root_matrix = np.empty((n_tp, n_ntp))
        for a in range(n_tp):
            for b in range(n_ntp):
                pos = a * n_ntp + b
                child = self._root_child[pos]
                if child < 0:
                    root_matrix[a, b] = self._root_const[pos]
                else:
                    root_matrix[a, b] = (
                        self._root_const[pos] * self.gamma * child_values[child]
                    )
        value, tp_policy, ntp_policy = solve_small_zero_sum(root_matrix)
        return (
            float(value),
            self._root_tp[:n_tp].copy(),
            self._root_ntp[:n_ntp].copy(),
            tp_policy,
            ntp_policy,
        )

    def _expand_depth2_snapshot(self, lane0: int, lane1: int):
        """Expand one depth-2 tree and copy out the slices needed to back it
        up after a deferred, batched net forward. Returns (snapshot,
        leaf_keys0, leaf_keys1) or (None, None, None) on buffer overflow."""
        n_tp, n_ntp, n_children, n_leaves = expand_depth2(
            np.int64(lane0), np.int64(lane1), self.prune_stock,
            self._root_tp, self._root_ntp,
            self._root_const, self._root_child,
            self._child_keys0, self._child_keys1,
            self._child_tp_n, self._child_ntp_n, self._child_offset,
            self._cell_const, self._cell_leaf,
            self._leaf_keys0, self._leaf_keys1,
        )
        if n_tp < 0:
            return None, None, None
        if n_children > 0:
            cells = int(self._child_offset[n_children - 1]) + \
                int(self._child_tp_n[n_children - 1]) * \
                int(self._child_ntp_n[n_children - 1])
        else:
            cells = 0
        rc = n_tp * n_ntp
        snap = (
            n_tp, n_ntp, n_children, n_leaves,
            self._root_tp[:n_tp].copy(), self._root_ntp[:n_ntp].copy(),
            self._root_const[:rc].copy(), self._root_child[:rc].copy(),
            self._child_tp_n[:n_children].copy(),
            self._child_ntp_n[:n_children].copy(),
            self._child_offset[:n_children].copy(),
            self._cell_const[:cells].copy(), self._cell_leaf[:cells].copy(),
        )
        return (
            snap,
            self._leaf_keys0[:n_leaves].copy(),
            self._leaf_keys1[:n_leaves].copy(),
        )

    def _backup_depth2(self, snap, leaf_values: np.ndarray):
        """Back up a depth-2 snapshot. Returns (value, tp_codes, ntp_codes,
        tp_policy, ntp_policy) — identical to solve()."""
        (n_tp, n_ntp, n_children, n_leaves,
         root_tp, root_ntp, root_const, root_child,
         child_tp_n, child_ntp_n, child_offset,
         cell_const, cell_leaf) = snap
        child_values = np.zeros(max(n_children, 1), dtype=np.float64)
        if n_children > 0:
            backup_children(
                n_children, child_tp_n, child_ntp_n, child_offset,
                cell_const, cell_leaf, leaf_values, self.gamma,
                self._matrix, self._tableau, self._basis, child_values,
            )
        root_matrix = np.empty((n_tp, n_ntp))
        for a in range(n_tp):
            for b in range(n_ntp):
                pos = a * n_ntp + b
                child = root_child[pos]
                if child < 0:
                    root_matrix[a, b] = root_const[pos]
                else:
                    root_matrix[a, b] = root_const[pos] * self.gamma * child_values[child]
        value, tp_policy, ntp_policy = solve_small_zero_sum(root_matrix)
        return float(value), root_tp, root_ntp, tp_policy, ntp_policy

    def solve_batch(self, keys0, keys1, leaf_budget: int = 120_000):
        """Depth-2 net-leaf solve for many states in one shot — identical
        per-state results to solve(), but each chunk's leaves are pooled into
        ONE network forward (the net is ~half of solve()'s cost and per-state
        forwards are overhead-bound). Returns a list of solve()-style tuples
        (value, tp_codes, ntp_codes, tp_policy, ntp_policy), one per input
        state, in order. Caller should deduplicate states first when games
        share positions."""
        keys0 = np.asarray(keys0, dtype=np.int64)
        keys1 = np.asarray(keys1, dtype=np.int64)
        n = len(keys0)
        results = [None] * n

        pending = []          # (state_index, snapshot)
        seg0, seg1 = [], []
        acc = [0]

        def flush():
            if not pending:
                return
            if seg0:
                cat0 = np.concatenate(seg0)
                cat1 = np.concatenate(seg1)
            else:
                cat0 = np.zeros(0, dtype=np.int64)
                cat1 = np.zeros(0, dtype=np.int64)
            if len(cat0) > 0:
                stacked = np.stack([cat0, cat1], axis=1)
                uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
                inv = np.asarray(inv).reshape(-1)
                gu_vals = self._net_values(
                    np.ascontiguousarray(uniq[:, 0]),
                    np.ascontiguousarray(uniq[:, 1]),
                ).astype(np.float64)
            else:
                inv = np.zeros(0, dtype=np.int64)
                gu_vals = np.zeros(0, dtype=np.float64)
            pos = 0
            for j, snap in pending:
                nl = snap[3]
                if nl > 0:
                    leaf_values = gu_vals[inv[pos:pos + nl]]
                    pos += nl
                else:
                    leaf_values = np.zeros(0, dtype=np.float64)
                results[j] = self._backup_depth2(snap, leaf_values)
            pending.clear()
            seg0.clear()
            seg1.clear()
            acc[0] = 0

        for j in range(n):
            snap, lk0, lk1 = self._expand_depth2_snapshot(
                int(keys0[j]), int(keys1[j])
            )
            if snap is None:
                results[j] = self.solve(int(keys0[j]), int(keys1[j]))
                continue
            pending.append((j, snap))
            seg0.append(lk0)
            seg1.append(lk1)
            acc[0] += snap[3]
            if acc[0] >= leaf_budget:
                flush()
        flush()
        return results

    def value_depth3(self, lane0: int, lane1: int) -> float:
        """Depth-3 net-leaf VALUE (training target; acting stays depth-2).

        Rationale (designer diagnosis 2026-07-13): with depth-2 targets only
        the endgame payoff of slow skills (cement/stock/lock) falls inside
        the credit-assignment horizon — e.g. endgame cement was learnt while
        opening cement (the stronger use) was not. Deeper targets extend the
        horizon each generation. Falls back to depth 2 on buffer overflow.
        """
        self._ensure_depth3_buffers()

        n_tp, n_ntp, n_l1, n_l2, n_leaves = expand_depth3(
            np.int64(lane0), np.int64(lane1), self.prune_stock,
            self._root_tp, self._root_ntp, self._root_const, self._root_child,
            self._child_keys0, self._child_keys1,
            self._child_tp_n, self._child_ntp_n, self._child_offset,
            self._l1_const, self._l1_idx,
            self._l2_keys0, self._l2_keys1,
            self._l2_tp_n, self._l2_ntp_n, self._l2_offset,
            self._l2_const, self._l2_idx,
            self._leaf_keys0, self._leaf_keys1,
        )
        if n_tp < 0:
            value, _, _, _, _ = self.solve(lane0, lane1)
            return value

        if n_leaves > 0:
            leaf_values = self._net_values(
                self._leaf_keys0[:n_leaves].copy(),
                self._leaf_keys1[:n_leaves].copy(),
            ).astype(np.float64)
        else:
            leaf_values = np.zeros(0, dtype=np.float64)

        l2_values = np.zeros(max(n_l2, 1), dtype=np.float64)
        if n_l2 > 0:
            backup_children(
                n_l2, self._l2_tp_n, self._l2_ntp_n, self._l2_offset,
                self._l2_const, self._l2_idx, leaf_values, self.gamma,
                self._matrix, self._tableau, self._basis, l2_values,
            )
        l1_values = np.zeros(max(n_l1, 1), dtype=np.float64)
        if n_l1 > 0:
            backup_children(
                n_l1, self._child_tp_n, self._child_ntp_n, self._child_offset,
                self._l1_const, self._l1_idx, l2_values, self.gamma,
                self._matrix, self._tableau, self._basis, l1_values,
            )

        value = root_value_only(
            n_tp, n_ntp, self._root_const, self._root_child, l1_values,
            self.gamma, self._matrix, self._tableau, self._basis,
        )
        return float(value)

    def _ensure_depth3_buffers(self) -> None:
        if not hasattr(self, "_l2_keys0"):
            self._l1_const = np.zeros(MAX_CELLS, dtype=np.float64)
            self._l1_idx = np.zeros(MAX_CELLS, dtype=np.int64)
            self._l2_keys0 = np.zeros(MAX_L2, dtype=np.int64)
            self._l2_keys1 = np.zeros(MAX_L2, dtype=np.int64)
            self._l2_tp_n = np.zeros(MAX_L2, dtype=np.int16)
            self._l2_ntp_n = np.zeros(MAX_L2, dtype=np.int16)
            self._l2_offset = np.zeros(MAX_L2, dtype=np.int64)
            self._l2_const = np.zeros(MAX_L2_CELLS, dtype=np.float64)
            self._l2_idx = np.zeros(MAX_L2_CELLS, dtype=np.int64)

    def _expand_depth3_snapshot(self, lane0: int, lane1: int):
        """Expand one depth-3 tree and copy out the slices needed to back it
        up later (so the net forward can be deferred and batched across many
        states). Returns (snapshot, leaf_keys0, leaf_keys1), or (None, None,
        None) on buffer overflow."""
        n_tp, n_ntp, n_l1, n_l2, n_leaves = expand_depth3(
            np.int64(lane0), np.int64(lane1), self.prune_stock,
            self._root_tp, self._root_ntp, self._root_const, self._root_child,
            self._child_keys0, self._child_keys1,
            self._child_tp_n, self._child_ntp_n, self._child_offset,
            self._l1_const, self._l1_idx,
            self._l2_keys0, self._l2_keys1,
            self._l2_tp_n, self._l2_ntp_n, self._l2_offset,
            self._l2_const, self._l2_idx,
            self._leaf_keys0, self._leaf_keys1,
        )
        if n_tp < 0:
            return None, None, None
        # Total cells actually written at each level (offset of last node +
        # its own tp*ntp block); expand_depth3 fills them contiguously.
        if n_l1 > 0:
            l1_cells = int(self._child_offset[n_l1 - 1]) + \
                int(self._child_tp_n[n_l1 - 1]) * int(self._child_ntp_n[n_l1 - 1])
        else:
            l1_cells = 0
        if n_l2 > 0:
            l2_cells = int(self._l2_offset[n_l2 - 1]) + \
                int(self._l2_tp_n[n_l2 - 1]) * int(self._l2_ntp_n[n_l2 - 1])
        else:
            l2_cells = 0
        rc = n_tp * n_ntp
        snap = (
            n_tp, n_ntp, n_l1, n_l2, n_leaves,
            self._root_const[:rc].copy(), self._root_child[:rc].copy(),
            self._child_tp_n[:n_l1].copy(), self._child_ntp_n[:n_l1].copy(),
            self._child_offset[:n_l1].copy(),
            self._l1_const[:l1_cells].copy(), self._l1_idx[:l1_cells].copy(),
            self._l2_tp_n[:n_l2].copy(), self._l2_ntp_n[:n_l2].copy(),
            self._l2_offset[:n_l2].copy(),
            self._l2_const[:l2_cells].copy(), self._l2_idx[:l2_cells].copy(),
        )
        return (
            snap,
            self._leaf_keys0[:n_leaves].copy(),
            self._leaf_keys1[:n_leaves].copy(),
        )

    def _backup_snapshot(self, snap, leaf_values: np.ndarray) -> float:
        """Back up a snapshot from _expand_depth3_snapshot given its leaf
        values. Mirrors value_depth3's backup exactly."""
        (n_tp, n_ntp, n_l1, n_l2, n_leaves,
         root_const, root_child, child_tp_n, child_ntp_n, child_offset,
         l1_const, l1_idx, l2_tp_n, l2_ntp_n, l2_offset,
         l2_const, l2_idx) = snap

        l2_values = np.zeros(max(n_l2, 1), dtype=np.float64)
        if n_l2 > 0:
            backup_children(
                n_l2, l2_tp_n, l2_ntp_n, l2_offset,
                l2_const, l2_idx, leaf_values, self.gamma,
                self._matrix, self._tableau, self._basis, l2_values,
            )
        l1_values = np.zeros(max(n_l1, 1), dtype=np.float64)
        if n_l1 > 0:
            backup_children(
                n_l1, child_tp_n, child_ntp_n, child_offset,
                l1_const, l1_idx, l2_values, self.gamma,
                self._matrix, self._tableau, self._basis, l1_values,
            )
        value = root_value_only(
            n_tp, n_ntp, root_const, root_child, l1_values,
            self.gamma, self._matrix, self._tableau, self._basis,
        )
        return float(value)

    def value_depth3_batch(self, keys0, keys1, leaf_budget: int = 80_000):
        """Depth-3 net-leaf values for many states — same result as calling
        value_depth3 per state, but pools each chunk's leaves into ONE network
        forward (per-state forwards are dominated by launch/transfer overhead:
        measured ~28x on the net portion, RTX 4070 Ti). States accumulate into
        a chunk until their combined leaf count reaches leaf_budget, then the
        chunk's leaves are globally deduplicated, evaluated in one forward, and
        scattered back for per-state LP backup. Memory stays bounded to roughly
        one leaf_budget's worth of expansion cells."""
        self._ensure_depth3_buffers()
        keys0 = np.asarray(keys0, dtype=np.int64)
        keys1 = np.asarray(keys1, dtype=np.int64)
        n = len(keys0)
        out = np.empty(n, dtype=np.float64)

        pending = []          # (state_index, snapshot)
        seg0, seg1 = [], []   # per-state leaf key segments (concat order)
        acc = [0]             # accumulated leaf count (list for closure)

        def flush():
            if not pending:
                return
            if seg0:
                cat0 = np.concatenate(seg0)
                cat1 = np.concatenate(seg1)
            else:
                cat0 = np.zeros(0, dtype=np.int64)
                cat1 = np.zeros(0, dtype=np.int64)
            if len(cat0) > 0:
                stacked = np.stack([cat0, cat1], axis=1)
                uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
                inv = np.asarray(inv).reshape(-1)
                gu_vals = self._net_values(
                    np.ascontiguousarray(uniq[:, 0]),
                    np.ascontiguousarray(uniq[:, 1]),
                ).astype(np.float64)
            else:
                inv = np.zeros(0, dtype=np.int64)
                gu_vals = np.zeros(0, dtype=np.float64)
            pos = 0
            for j, snap in pending:
                nl = snap[4]
                if nl > 0:
                    leaf_values = gu_vals[inv[pos:pos + nl]]
                    pos += nl
                else:
                    leaf_values = np.zeros(0, dtype=np.float64)
                out[j] = self._backup_snapshot(snap, leaf_values)
            pending.clear()
            seg0.clear()
            seg1.clear()
            acc[0] = 0

        for j in range(n):
            snap, lk0, lk1 = self._expand_depth3_snapshot(
                int(keys0[j]), int(keys1[j])
            )
            if snap is None:
                # Rare buffer overflow: fall back to the single-state path.
                out[j] = self.value_depth3(int(keys0[j]), int(keys1[j]))
                continue
            pending.append((j, snap))
            seg0.append(lk0)
            seg1.append(lk1)
            acc[0] += snap[4]
            if acc[0] >= leaf_budget:
                flush()
        flush()
        return out


def parallel_depth3_values(model, device, keys0, keys1, *,
                           prune_stock: bool = True, gamma: float = 0.999,
                           n_threads: int = 6):
    """Depth-3 net-leaf targets for many states, computed across threads.

    The heavy work per state is compiled (expand_depth3 + LP backups, ~70%)
    and the net forward releases the GIL during CUDA, so the nogil njit path
    parallelises cleanly. Each thread owns its own BatchedSearcher (private
    expansion buffers) but shares the read-only model. Results are identical
    to a serial value_depth3 loop (verified bit-exact). Measured ~3.7x at 6
    threads on an RTX 4070 Ti; throughput saturates around 6.
    """
    from concurrent.futures import ThreadPoolExecutor

    keys0 = np.asarray(keys0, dtype=np.int64)
    keys1 = np.asarray(keys1, dtype=np.int64)
    n = len(keys0)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    n_threads = max(1, min(n_threads, n))
    searchers = [
        BatchedSearcher(model, device, gamma=gamma, prune_stock=prune_stock)
        for _ in range(n_threads)
    ]
    chunks = np.array_split(np.arange(n), n_threads)

    def work(t: int) -> None:
        s = searchers[t]
        for i in chunks[t]:
            out[i] = s.value_depth3(int(keys0[i]), int(keys1[i]))

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(work, range(n_threads)))
    return out

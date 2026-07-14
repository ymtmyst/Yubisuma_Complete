"""Nash value iteration over an OPEN, sampled sub-graph of reachable states.

This is the N7-A engine (see ``N7_STRATEGY_AND_DESIGN.md``). Where
``packed_vi.solve_universe`` solves a *closed* universe (every child is inside
the set), this module solves an arbitrary *sampled* set of interior states and
closes the graph with fixed **frontier boundary values** — the value net V(s)
in production, or exact table values in tests.

Mechanism (maximal reuse of the proven ``packed_vi`` machinery):

- interior states  = the sampled seed set, indices ``0 .. n_seed-1``.
- frontier states  = children of interior states that are NOT seeds; they get
                     indices ``n_seed .. n_total-1`` and NEVER get updated —
                     their value is a fixed boundary condition.
- One Jacobi sweep (``packed_vi._jacobi_sweep``) applied with a state count of
  ``n_seed`` updates only interior states; child lookups into the value buffer
  reach frontier slots, which hold the boundary values persistently (they are
  written into both double-buffers up front and never overwritten).

For a closed seed set (no frontier) this reduces EXACTLY to
``solve_universe`` — that is the reference test in ``tests/test_subgraph_vi.py``.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict as NumbaDict

from .packed_engine import legal_ntp_codes, legal_tp_codes, step
from .packed_vi import (
    _KEY_TYPE,
    _count_cells,
    _jacobi_sweep,
    _repair_state_value,
)


@njit(cache=True)
def _fill_open(seed0, seed1, n_seed, alphabet_mask, max_stock,
               offsets, child_idx, cell_val, front0, front1):
    """Fill transition tables for the interior seeds, discovering frontier.

    Interior seeds occupy global indices ``0 .. n_seed-1``. Any non-terminal
    child that is not itself a seed is appended to the frontier
    (``front0/front1``) and assigned index ``n_seed + k``. Terminal children
    are stored as ``child_idx = -1`` with the terminal reward in ``cell_val``
    (identical convention to ``packed_vi._fill_tables``). Returns the frontier
    count, or -1 if ``front0`` overflowed."""
    index = NumbaDict.empty(_KEY_TYPE, types.int64)
    for i in range(n_seed):
        index[(seed0[i], seed1[i])] = i
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    cap_front = front0.shape[0]
    n_front = 0
    for i in range(n_seed):
        lane0 = seed0[i]
        lane1 = seed1[i]
        n_tp = legal_tp_codes(lane0, lane1, alphabet_mask, max_stock, tp_buf)
        n_ntp = legal_ntp_codes(lane0, lane1, ntp_buf)
        pos = offsets[i]
        for a in range(n_tp):
            for b in range(n_ntp):
                child0, child1, status, reward = step(
                    lane0, lane1, tp_buf[a], ntp_buf[b], alphabet_mask
                )
                if status == 2:
                    child_idx[pos] = -1
                    cell_val[pos] = reward
                else:
                    key = (child0, child1)
                    if key in index:
                        child_idx[pos] = index[key]
                    else:
                        if n_front >= cap_front:
                            return -1
                        gi = n_seed + n_front
                        index[key] = gi
                        front0[n_front] = child0
                        front1[n_front] = child1
                        n_front += 1
                        child_idx[pos] = gi
                    cell_val[pos] = 1 if status == 1 else -1
                pos += 1
    return n_front


class SubgraphTables:
    """Flattened transition tables for a sampled sub-graph.

    ``keys0/keys1`` are ordered ``[interior seeds ..., frontier ...]`` so the
    global index of a state equals its position here.
    """

    def __init__(self, seed0, seed1, front0, front1,
                 tp_counts, ntp_counts, offsets, child_idx, cell_val,
                 alphabet_mask, max_stock):
        self.n_seed = int(seed0.shape[0])
        self.n_front = int(front0.shape[0])
        self.n_total = self.n_seed + self.n_front
        self.keys0 = np.concatenate([seed0, front0])
        self.keys1 = np.concatenate([seed1, front1])
        self.tp_counts = tp_counts
        self.ntp_counts = ntp_counts
        self.offsets = offsets
        self.child_idx = child_idx
        self.cell_val = cell_val
        self.alphabet_mask = int(alphabet_mask)
        self.max_stock = int(max_stock)

    @property
    def frontier_fraction(self) -> float:
        return self.n_front / self.n_total if self.n_total else 0.0

    def frontier_keys(self):
        """(front0, front1) — the states whose boundary value must be supplied."""
        return self.keys0[self.n_seed:], self.keys1[self.n_seed:]


def build_subgraph(seed0, seed1, alphabet_mask, max_stock) -> SubgraphTables:
    """Build the open sub-graph induced by the seed set.

    ``seed0/seed1`` are packed lanes of the sampled interior states (must be
    non-terminal and distinct). ``alphabet_mask``/``max_stock`` MUST match how
    the children are meant to be generated (same convention as
    ``solve_universe``)."""
    seed0 = np.ascontiguousarray(seed0, dtype=np.int64)
    seed1 = np.ascontiguousarray(seed1, dtype=np.int64)
    n_seed = seed0.shape[0]
    mask = np.int64(alphabet_mask)
    ms = np.int64(max_stock)

    tp_counts, ntp_counts, total_cells = _count_cells(seed0, seed1, n_seed, mask, ms)
    total_cells = int(total_cells)
    offsets = np.zeros(n_seed, dtype=np.int64)
    if n_seed > 1:
        np.cumsum(
            (tp_counts.astype(np.int64) * ntp_counts.astype(np.int64))[:-1],
            out=offsets[1:],
        )
    child_idx = np.empty(total_cells, dtype=np.int32)
    cell_val = np.empty(total_cells, dtype=np.int8)
    # Upper bound on frontier size: at most one new state per cell.
    front0 = np.empty(total_cells, dtype=np.int64)
    front1 = np.empty(total_cells, dtype=np.int64)

    n_front = _fill_open(
        seed0, seed1, n_seed, mask, ms,
        offsets, child_idx, cell_val, front0, front1,
    )
    if n_front < 0:  # pragma: no cover — bounded by total_cells, unreachable
        raise RuntimeError("frontier overflow (should be bounded by cell count)")
    front0 = front0[:n_front].copy()
    front1 = front1[:n_front].copy()
    return SubgraphTables(
        seed0, seed1, front0, front1,
        tp_counts, ntp_counts, offsets, child_idx, cell_val, mask, ms,
    )


def run_subgraph_vi(
    tab: SubgraphTables,
    boundary_values: np.ndarray,
    gamma: float = 0.999,
    epsilon: float = 1e-6,
    max_iterations: int = 3000,
    interior_init: np.ndarray | None = None,
    omega: float = 1.0,
    stall_window: int = 100,
    stall_rtol: float = 1e-3,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Jacobi Nash-VI over the interior with fixed frontier boundary values.

    ``boundary_values`` has length ``tab.n_front`` and gives V(s) for each
    frontier state (net or exact). ``interior_init`` (length ``tab.n_seed``)
    optionally warm-starts the interior (e.g. from the net) — it changes the
    convergence speed only, not the unique fixed point.

    Unlike the closed ``solve_universe`` (which reaches 1e-9), an OPEN sub-graph
    with an arbitrary-real net boundary can develop a slow cyclic component
    and/or a tiny limit cycle around states whose matrix cannot be certified
    (repaired every sweep). Since these values feed a regression teacher — where
    a ~1e-4 residual is utterly negligible against the net's own MSE — we stop
    early when the residual plateaus (no relative improvement of ``max_delta``
    over ``stall_window`` sweeps), and report ``stalled`` in the info.

    ``omega`` < 1 applies successive under-relaxation
    ``v ← (1-ω)·v_old + ω·Bellman(v_old)`` on the interior. This preserves the
    fixed point exactly (a fixed point of Bellman is a fixed point of the damped
    map) but damps the limit cycle around uncertified/repaired states that
    otherwise pins ``max_delta`` at a plateau (observed 1.3e-1 at 40k seeds with
    a raw-real net boundary). ``omega=1.0`` (default) reproduces the closed
    ``solve_universe`` dynamics so the reference tests are unaffected. Returns
    ``(interior_values, info)``."""
    n_seed = tab.n_seed
    n_total = tab.n_total
    if boundary_values.shape[0] != tab.n_front:
        raise ValueError(
            f"boundary_values length {boundary_values.shape[0]} != "
            f"frontier count {tab.n_front}"
        )

    values = np.zeros(n_total, dtype=np.float64)
    v_next = np.zeros(n_total, dtype=np.float64)
    if interior_init is not None:
        values[:n_seed] = interior_init
        v_next[:n_seed] = interior_init
    # Frontier boundary lives in BOTH buffers and is never written by the
    # sweep (which only writes indices 0..n_seed-1), so it persists across the
    # buffer swap below.
    values[n_seed:] = boundary_values
    v_next[n_seed:] = boundary_values

    fail_idx = np.zeros(4096, dtype=np.int64)
    max_delta = np.inf
    total_repairs = 0
    iterations = 0
    best_delta = np.inf
    stall_count = 0
    stalled = False
    t0 = time.perf_counter()
    for iterations in range(1, max_iterations + 1):
        max_delta, n_failed = _jacobi_sweep(
            tab.tp_counts, tab.ntp_counts, tab.offsets,
            tab.child_idx, tab.cell_val, values, v_next, gamma, fail_idx,
        )
        for k in range(n_failed):
            i = int(fail_idx[k])
            repaired = _repair_state_value(
                i, tab.tp_counts, tab.ntp_counts, tab.offsets,
                tab.child_idx, tab.cell_val, values, gamma,
            )
            v_next[i] = repaired
        if omega < 1.0:
            # Under-relax the whole interior (repairs included), then take the
            # damped step's residual as the convergence metric. Frontier slots
            # (>= n_seed) are boundary and left untouched.
            v_next[:n_seed] = (
                (1.0 - omega) * values[:n_seed] + omega * v_next[:n_seed]
            )
            max_delta = float(np.abs(v_next[:n_seed] - values[:n_seed]).max())
        else:
            # Fold repair residuals into max_delta (matches solve_universe).
            for k in range(n_failed):
                i = int(fail_idx[k])
                delta = abs(v_next[i] - values[i])
                if delta > max_delta:
                    max_delta = delta
        total_repairs += n_failed
        values, v_next = v_next, values
        if verbose and (iterations % 50 == 0 or max_delta < epsilon):
            print(f"sweep {iterations} max_delta {max_delta:.3e} "
                  f"repairs {n_failed}", flush=True)
        if max_delta < epsilon:
            break
        # Plateau detection: stop when max_delta stops improving materially.
        if max_delta < best_delta * (1.0 - stall_rtol):
            best_delta = max_delta
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= stall_window:
                stalled = True
                break
    vi_seconds = time.perf_counter() - t0
    converged = max_delta < epsilon

    info = {
        "n_seed": n_seed,
        "n_frontier": tab.n_front,
        "frontier_fraction": tab.frontier_fraction,
        "iterations": int(iterations),
        "max_delta": float(max_delta),
        "vi_seconds": vi_seconds,
        "repairs": int(total_repairs),
        "converged": bool(converged),
        "stalled": bool(stalled),
        "gamma": gamma,
    }
    return values[:n_seed].copy(), info

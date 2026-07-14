"""N7-E: njit-compiled coupled VI for graph-BR (fast line).

This is the COMPILED backend for the graph-BR value iteration. It does NOT
replace :mod:`complete_ai.graph_br` — the pure-Python ``solve_br`` stays as the
reference/fallback.

There are TWO distinct fast paths here; do not conflate them:

* ``solve_br_njit`` / ``solve_flat`` — the njit VI. Consumes the SAME graph as
  the Python ``solve_br`` and is proven IDENTICAL to it on that graph
  (``tests/test_graph_br_fast.py``). This is the drop-in VI used by
  ``n7_graph_br --engine njit`` (which still enumerates with ``enumerate_br``).
* ``enumerate_solve_fast`` — a SEPARATE, level-synchronous enumerator that caps
  depth by a state's shortest BFS distance (each state expanded once, at its
  shortest depth, and always evaluated at its best-response value). This is
  DELIBERATELY NOT identical to ``enumerate_br``, which caps by edge path-length
  and truncates deep edges to already-known states as frozen leaves.

  Consequence (measured, models/value_gvi_latest.pt): the two agree exactly at
  D=2, then ``enumerate_solve_fast`` returns a strictly HIGHER exploitability as
  D grows (it lets a best-responding attacker keep its full value at revisited
  states, so it is the TIGHTER / more-correct lower bound). Its state count ``n``
  also includes the depth-D leaf layer, so fast(D)['n'] ≈ enumerate_br(D+1)['n'].
  This mismatch is BY DESIGN — do NOT "fix" ``enumerate_solve_fast`` to match
  ``enumerate_br``; a naive enum-vs-enum parity check WILL report a difference
  and that is expected, not a bug.

Two speedups over the Python VI (which ran ~55 min at 250k states):

1. **Leaf-constant folding.** A cell whose child is terminal / endgame-exact /
   depth-capped / over-cap contributes a value that does NOT depend on the
   iterate (Vm/Vr). We fold those into a per-row constant at flatten time, so
   the sweep only iterates INTERNAL edges (child index ≥ 0) — a huge cut when
   leaves dominate (e.g. 19M leaf cells vs a few M internal at depth 6).
2. **njit sweep.** The coupled Jacobi update over flat CSR arrays compiles to
   native code (nogil), turning the per-sweep cost from minutes to ms.

Sign convention is copied verbatim from ``graph_br.solve_br`` (value TO THE
ATTACKER; mover/reactor roles; status 1 keeps the mover, 0 flips it). A parity
test (``tests/test_graph_br_fast.py``) asserts identical results to the Python
VI on a small graph.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from complete_solver.packed_engine import (
    legal_ntp_codes,
    legal_tp_codes,
    pack_state,
    step,
)
from complete_solver.state import initial_state

_FULL = np.int64(255)
_NOCAP = np.int64(99)


# ── leaf constants (child value independent of the iterate) ──────────────────

def _leaf_const_mover(ci: int, st: int, rew: float, gamma: float) -> float:
    """Attacker is mover here; contribution of a NON-internal child."""
    if st == 2:
        return rew                                  # terminal → reward to mover
    if ci == -3:
        return gamma * (rew if st == 1 else -rew)   # endgame / depth-cap leaf
    return 0.0                                        # over-cap (-2): optimistic 0


def _leaf_const_reactor(ci: int, st: int, rew: float, gamma: float) -> float:
    """Attacker is reactor here (frozen is mover)."""
    if st == 2:
        return -rew
    if ci == -3:
        return gamma * (-rew if st == 1 else rew)
    return 0.0


def _flatten_side(side_rows, n, gamma, leaf_const):
    """Flatten one role's ragged (state → rows → cells) structure into CSR:
    per-state row span, per-row (folded constant + internal-cell span), and flat
    internal-cell arrays (child index, status, weight)."""
    srow_off = np.zeros(n, dtype=np.int64)
    srow_n = np.zeros(n, dtype=np.int64)
    row_const: list[float] = []
    row_ioff: list[int] = []
    row_ilen: list[int] = []
    ci_list: list[int] = []
    st_list: list[int] = []
    w_list: list[float] = []
    r = 0
    for i in range(n):
        rows = side_rows[i]
        srow_off[i] = r
        if not rows:
            srow_n[i] = 0
            continue
        srow_n[i] = len(rows)
        for cells in rows:
            const = 0.0
            ioff = len(ci_list)
            ilen = 0
            for (ci, st, rew, w) in cells:
                if ci >= 0:
                    ci_list.append(ci)
                    st_list.append(st)
                    w_list.append(w)
                    ilen += 1
                else:
                    const += w * leaf_const(ci, st, rew, gamma)
            row_const.append(const)
            row_ioff.append(ioff)
            row_ilen.append(ilen)
            r += 1
    return (
        srow_off, srow_n,
        np.asarray(row_const, dtype=np.float64),
        np.asarray(row_ioff, dtype=np.int64),
        np.asarray(row_ilen, dtype=np.int64),
        np.asarray(ci_list, dtype=np.int64),
        np.asarray(st_list, dtype=np.int8),
        np.asarray(w_list, dtype=np.float64),
    )


@njit(cache=True, nogil=True)
def _coupled_sweep(n,
                   m_soff, m_sn, m_rconst, m_rioff, m_rilen, m_ci, m_st, m_w,
                   r_soff, r_sn, r_rconst, r_rioff, r_rilen, r_ci, r_st, r_w,
                   Vm, Vr, Vm_new, Vr_new, gamma, omega):
    """One damped coupled Jacobi sweep. Reads Vm/Vr (old), writes Vm_new/Vr_new.
    Returns max |Δ|."""
    max_delta = 0.0
    for i in range(n):
        # V_mover: attacker maximises over its TP rows.
        nr = m_sn[i]
        if nr == 0:
            vm = Vm[i]
        else:
            best = -1e18
            base = m_soff[i]
            for rr in range(base, base + nr):
                acc = m_rconst[rr]
                io = m_rioff[rr]
                for c in range(io, io + m_rilen[rr]):
                    ci = m_ci[c]
                    if m_st[c] == 1:
                        acc += m_w[c] * gamma * Vm[ci]
                    else:
                        acc += m_w[c] * gamma * Vr[ci]
                if acc > best:
                    best = acc
            vm = best
        # V_reactor: attacker maximises over its NTP rows.
        nr2 = r_sn[i]
        if nr2 == 0:
            vr = Vr[i]
        else:
            best = -1e18
            base = r_soff[i]
            for rr in range(base, base + nr2):
                acc = r_rconst[rr]
                io = r_rioff[rr]
                for c in range(io, io + r_rilen[rr]):
                    ci = r_ci[c]
                    if r_st[c] == 1:
                        acc += r_w[c] * gamma * Vr[ci]
                    else:
                        acc += r_w[c] * gamma * Vm[ci]
                if acc > best:
                    best = acc
            vr = best
        vm = (1.0 - omega) * Vm[i] + omega * vm
        vr = (1.0 - omega) * Vr[i] + omega * vr
        Vm_new[i] = vm
        Vr_new[i] = vr
        dm = vm - Vm[i]
        if dm < 0.0:
            dm = -dm
        dr = vr - Vr[i]
        if dr < 0.0:
            dr = -dr
        if dm > max_delta:
            max_delta = dm
        if dr > max_delta:
            max_delta = dr
    return max_delta


def solve_br_njit(data, gamma: float = 0.999, max_iters: int = 4000,
                  eps: float = 1e-7, omega: float = 0.6, verbose: bool = True):
    """njit coupled VI — same result shape as :func:`graph_br.solve_br`."""
    n = data["n"]
    t0 = time.perf_counter()
    mv = _flatten_side(data["mover_rows"], n, gamma, _leaf_const_mover)
    rv = _flatten_side(data["reactor_cols"], n, gamma, _leaf_const_reactor)
    if verbose:
        n_int = mv[5].shape[0] + rv[5].shape[0]
        print(f"  flattened: {n_int:,} internal edges "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    Vm = np.zeros(n); Vr = np.zeros(n)
    Vm2 = np.empty(n); Vr2 = np.empty(n)
    d = np.inf
    it = 0
    for it in range(1, max_iters + 1):
        d = _coupled_sweep(n, *mv, *rv, Vm, Vr, Vm2, Vr2, gamma, omega)
        Vm, Vm2 = Vm2, Vm
        Vr, Vr2 = Vr2, Vr
        if verbose and (it % 100 == 0 or d < eps):
            print(f"  BR-VI(njit) sweep {it}: max_delta {d:.2e}", flush=True)
        if d < eps:
            break
    root_seat0 = float(Vm[0])
    root_seat1 = float(Vr[0])
    attacker_value = 0.5 * (root_seat0 + root_seat1)
    return {
        "attacker_value_seat0": root_seat0,
        "attacker_value_seat1": root_seat1,
        "attacker_value": attacker_value,
        "attacker_winrate": 0.5 * (attacker_value + 1.0),
        "sweeps": it, "converged": bool(d < eps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fast enumeration line: level-synchronous BFS + threaded solve + njit child
# generation + folded flat storage. Produces the SAME flat structure the njit VI
# consumes (no Python-tuple graph → memory-lean enough for D=7-8). Selectable
# alongside the pure-Python enumerate_br; results match it (parity test).
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True, nogil=True)
def _grid_children(s0, s1, tp_codes, n_tp, ntp_codes, n_ntp, mask,
                   out_c0, out_c1, out_st, out_rew):
    """All (tp × ntp) children of one state in a compiled loop (replaces the
    per-cell Python ``step`` calls). Fills row-major out_[k=a*n_ntp+b]."""
    for a in range(n_tp):
        for b in range(n_ntp):
            g0, g1, stt, rew = step(s0, s1, tp_codes[a], ntp_codes[b], mask)
            k = a * n_ntp + b
            out_c0[k] = g0
            out_c1[k] = g1
            out_st[k] = stt
            out_rew[k] = rew


class _Grow:
    """Amortized-O(1) growable 1-D numpy buffer (avoids GB-scale Python lists
    for the cell arrays at D=7-8)."""

    def __init__(self, dtype, cap=1 << 16):
        self.buf = np.empty(cap, dtype=dtype)
        self.n = 0

    def append(self, v):
        if self.n == self.buf.shape[0]:
            self.buf = np.concatenate((self.buf, np.empty_like(self.buf)))
        self.buf[self.n] = v
        self.n += 1

    def arr(self):
        return self.buf[:self.n]


def _parallel_solve(searchers, keys0, keys1, n_threads):
    """Policies for many states across threads (per-thread searcher owns its
    buffers; solve_batch pools each chunk's net forward). Returns a list of
    solve()-style tuples aligned to input order."""
    n = len(keys0)
    results = [None] * n
    chunks = np.array_split(np.arange(n), n_threads)

    def work(t):
        s = searchers[t]
        idx = chunks[t]
        if len(idx) == 0:
            return
        sub = s.solve_batch(keys0[idx], keys1[idx])
        for local, gi in enumerate(idx):
            results[gi] = sub[local]

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(work, range(n_threads)))
    return results


def enumerate_solve_fast(searcher_factory, endgame, max_depth,
                         cap=4_000_000, support_eps=1e-3, gamma=0.999,
                         n_threads=8, verbose=True):
    """Level-synchronous BFS that solves each level's frozen policies in
    parallel and folds the BR graph straight into the flat VI format.

    Depth-``max_depth`` states are kept as FIXED-VALUE leaves (0 rows, initial
    V = frozen value / −value), which the njit sweep never updates — identical
    semantics to graph_br's depth cap. Endgame states are folded as exact
    constants (A0). Returns a dict consumable by :func:`solve_flat`."""
    searchers = [searcher_factory() for _ in range(n_threads)]
    init0, init1 = pack_state(initial_state())
    index = {(int(init0), int(init1)): 0}
    order0 = [int(init0)]; order1 = [int(init1)]
    depth = [0]

    # Per-state folded structure (both roles).
    m_soff = _Grow(np.int64); m_sn = _Grow(np.int32)
    r_soff = _Grow(np.int64); r_sn = _Grow(np.int32)
    Vm0 = _Grow(np.float64); Vr0 = _Grow(np.float64)
    # Per-row.
    m_rconst = _Grow(np.float64); m_rioff = _Grow(np.int64); m_rilen = _Grow(np.int32)
    r_rconst = _Grow(np.float64); r_rioff = _Grow(np.int64); r_rilen = _Grow(np.int32)
    # Per internal cell.
    m_ci = _Grow(np.int64); m_st = _Grow(np.int8); m_w = _Grow(np.float64)
    r_ci = _Grow(np.int64); r_st = _Grow(np.int8); r_w = _Grow(np.float64)

    g_c0 = np.empty(96 * 16, np.int64); g_c1 = np.empty(96 * 16, np.int64)
    g_st = np.empty(96 * 16, np.int8); g_rew = np.empty(96 * 16, np.float64)
    tp_buf = np.empty(96, np.int64); ntp_buf = np.empty(16, np.int64)

    def child_index(c0, c1, d):
        key = (c0, c1)
        ci = index.get(key)
        if ci is None:
            if len(order0) >= cap:
                return -2
            ci = len(order0)
            index[key] = ci
            order0.append(c0); order1.append(c1)
            depth.append(d)
        return ci

    head = 0
    n_endgame = 0
    t0 = time.perf_counter()
    while head < len(order0):
        stop = len(order0)
        k0 = np.asarray(order0[head:stop], np.int64)
        k1 = np.asarray(order1[head:stop], np.int64)
        policies = _parallel_solve(searchers, k0, k1,
                                   min(n_threads, max(1, len(k0))))
        for li, i in enumerate(range(head, stop)):
            val, tpc, ntpc, tpp, ntpp = policies[li]
            di = depth[i]
            m_soff.append(m_rconst.n); r_soff.append(r_rconst.n)
            if di >= max_depth:
                m_sn.append(0); r_sn.append(0)
                Vm0.append(float(val)); Vr0.append(-float(val))
                continue
            Vm0.append(0.0); Vr0.append(0.0)
            n_tp = len(tpc); n_ntp = len(ntpc)
            tpc = np.asarray(tpc, np.int64); ntpc = np.asarray(ntpc, np.int64)
            tpp = np.asarray(tpp, np.float64); ntpp = np.asarray(ntpp, np.float64)
            _grid_children(np.int64(order0[i]), np.int64(order1[i]),
                           tpc, n_tp, ntpc, n_ntp, _FULL, g_c0, g_c1, g_st, g_rew)
            supp_tp = np.where(tpp > support_eps)[0]
            supp_ntp = np.where(ntpp > support_eps)[0]

            # mover rows: all tp × support ntp, weight = σ_ntp.
            for a in range(n_tp):
                const = 0.0; ioff = m_ci.n; ilen = 0
                for b in supp_ntp:
                    k = a * n_ntp + b
                    st = int(g_st[k]); w = float(ntpp[b])
                    if st == 2:
                        const += w * float(g_rew[k])
                    else:
                        c0 = int(g_c0[k]); c1 = int(g_c1[k])
                        a0 = endgame.value(c0, c1) if endgame is not None else None
                        if a0 is not None:
                            const += w * gamma * (a0 if st == 1 else -a0)
                            n_endgame += 1
                        else:
                            ci = child_index(c0, c1, di + 1)
                            if ci < 0:
                                pass  # over-cap → optimistic 0
                            else:
                                m_ci.append(ci); m_st.append(st); m_w.append(w); ilen += 1
                m_rconst.append(const); m_rioff.append(ioff); m_rilen.append(ilen)
            m_sn.append(n_tp)

            # reactor rows: support tp × all ntp, weight = σ_tp.
            for b in range(n_ntp):
                const = 0.0; ioff = r_ci.n; ilen = 0
                for a in supp_tp:
                    k = a * n_ntp + b
                    st = int(g_st[k]); w = float(tpp[a])
                    if st == 2:
                        const += w * (-float(g_rew[k]))
                    else:
                        c0 = int(g_c0[k]); c1 = int(g_c1[k])
                        a0 = endgame.value(c0, c1) if endgame is not None else None
                        if a0 is not None:
                            const += w * gamma * (-a0 if st == 1 else a0)
                        else:
                            ci = child_index(c0, c1, di + 1)
                            if ci >= 0:
                                r_ci.append(ci); r_st.append(st); r_w.append(w); ilen += 1
                r_rconst.append(const); r_rioff.append(ioff); r_rilen.append(ilen)
            r_sn.append(n_ntp)

        if verbose:
            print(f"  fast-enum depth≤{depth[stop-1]} states {stop} "
                  f"(frontier {len(order0)-stop}, {time.perf_counter()-t0:.0f}s)",
                  flush=True)
        head = stop

    n = len(order0)
    if verbose:
        print(f"fast-enum {n} states, {m_ci.n + r_ci.n} internal edges, "
              f"endgame folds {n_endgame} ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    return {
        "n": n, "hit_cap": n >= cap,
        "Vm0": Vm0.arr(), "Vr0": Vr0.arr(),
        "m": (m_soff.arr(), m_sn.arr(), m_rconst.arr(), m_rioff.arr(),
              m_rilen.arr(), m_ci.arr(), m_st.arr(), m_w.arr()),
        "r": (r_soff.arr(), r_sn.arr(), r_rconst.arr(), r_rioff.arr(),
              r_rilen.arr(), r_ci.arr(), r_st.arr(), r_w.arr()),
    }


def solve_flat(flat, gamma=0.999, max_iters=6000, eps=1e-7, omega=0.6,
               verbose=True):
    """njit VI over the fast-enum flat structure (fixed-value leaves via V0)."""
    n = flat["n"]
    Vm = flat["Vm0"].copy(); Vr = flat["Vr0"].copy()
    Vm2 = Vm.copy(); Vr2 = Vr.copy()
    mv = flat["m"]; rv = flat["r"]
    d = np.inf; it = 0
    for it in range(1, max_iters + 1):
        d = _coupled_sweep(n, *mv, *rv, Vm, Vr, Vm2, Vr2, gamma, omega)
        Vm, Vm2 = Vm2, Vm
        Vr, Vr2 = Vr2, Vr
        if verbose and (it % 100 == 0 or d < eps):
            print(f"  BR-VI(fast) sweep {it}: max_delta {d:.2e}", flush=True)
        if d < eps:
            break
    root_seat0 = float(Vm[0]); root_seat1 = float(Vr[0])
    attacker_value = 0.5 * (root_seat0 + root_seat1)
    return {
        "attacker_value_seat0": root_seat0,
        "attacker_value_seat1": root_seat1,
        "attacker_value": attacker_value,
        "attacker_winrate": 0.5 * (attacker_value + 1.0),
        "sweeps": it, "converged": bool(d < eps),
    }

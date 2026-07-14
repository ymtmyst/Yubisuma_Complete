"""N7: graph-BR — EXACT best-response exploitability of a frozen agent.

The PPO attacker (``br_attack``) is a weak, noisy LOWER bound on exploitability
and — per the regime-blindness finding (N7 Part 6-3c) — can miss タイム-regime
exploits entirely. graph-BR computes the attacker's best-response value by
dynamic programming over the reachable game graph, giving the EXACT exploit
rate (no learning, no regime blind spots).

Formulation (mirrors ``br_env``: attacker owns one seat, frozen owns the other,
each ply the mover declares TP and the reactor reacts NTP; status 0 flips the
mover, status 1 keeps it, status 2 is terminal with reward to the mover). Two
value functions, both = value TO THE ATTACKER:

  V_mover(s)   = max_tp   Σ_{ntp∈supp} σ_ntp(s)·g(tp,ntp)     (attacker is mover)
  V_reactor(s) = max_ntp  Σ_{tp∈supp}  σ_tp(s)·g'(tp,ntp)     (attacker is reactor)

where g uses the child value TO THE ATTACKER: terminal → ±reward (sign by whose
turn it was); non-terminal → γ·V_mover(child) if the mover is kept (status 1),
else γ·V_reactor(child) (status 0 flips who moves). The frozen agent plays only
its SUPPORT (σ>eps), so the reachable set (attacker-any × frozen-support) is
bounded — unlike the full any×any game graph.

Attacker win-rate ≈ (V+1)/2, averaged over the attacker taking seat 0 (root
value = V_mover(root)) and seat 1 (root value = V_reactor(root)).

This is a correctness-first NumPy version; enumeration calls the frozen
``searcher.solve`` per state (the cost driver). Bounded by ``cap``.
"""

from __future__ import annotations

import time

import numpy as np

from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes, pack_state, step
from complete_solver.state import initial_state

_FULL = np.int64(255)
_NOCAP = np.int64(99)


def _solve_cached(searcher, cache, key):
    c = cache.get(key)
    if c is None:
        c = searcher.solve(key[0], key[1])
        cache[key] = c
    return c


def enumerate_br(searcher, cap: int = 200_000, support_eps: float = 1e-3,
                 endgame=None, max_depth=None, verbose: bool = True):
    """BFS the reachable graph under attacker-any × frozen-support.

    If ``endgame`` (an :class:`~complete_ai.endgame_table.EndgameTablebase`) is
    given, any child state it contains is treated as an EXACT leaf and NOT
    expanded (pincer / N7-F): the frozen agent plays the certified-optimal
    endgame there, so a best-responding attacker gets exactly the game value —
    ``A0[child]`` to the mover — with no room to exploit. This caps the back of
    the graph at the endgame boundary, replacing the optimistic over-cap leaf
    (value 0) with the true value and dramatically shrinking the reachable set.
    Endgame leaves are recorded with child index ``-3`` and the A0 value carried
    in the reward slot; :func:`solve_br` decodes them by role/status.

    Returns a dict of parallel arrays/lists keyed by state index.
    """
    init0, init1 = pack_state(initial_state())
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)
    solve_cache: dict[tuple[int, int], tuple] = {}
    n_endgame_leaves = 0
    n_depth_leaves = 0

    index: dict[tuple[int, int], int] = {(int(init0), int(init1)): 0}
    order = [(int(init0), int(init1))]
    depth = [0]                       # BFS depth per state (parallel to order)
    # per-state cell records for the two roles:
    #   mover_cells[i]   = list of (child_index_or_-1, status, reward, weight)
    #     grouped by tp-row → we store as (n_tp, list per row of (idx,status,rew,w))
    # To keep it simple we store flat cell lists plus row/col structure.
    mover_rows: list = [None]        # per state: list over tp of list of (idx,status,rew,sigma_ntp)
    reactor_cols: list = [None]      # per state: list over ntp of list of (idx,status,rew,sigma_tp)

    head = 0
    t0 = time.perf_counter()
    while head < len(order):
        s0, s1 = order[head]
        i = head
        head += 1
        _, tp_codes, ntp_codes, tp_pol, ntp_pol = _solve_cached(
            searcher, solve_cache, (s0, s1))
        tp_codes = np.asarray(tp_codes); ntp_codes = np.asarray(ntp_codes)
        tp_pol = np.asarray(tp_pol, float); ntp_pol = np.asarray(ntp_pol, float)
        supp_tp = np.where(tp_pol > support_eps)[0]
        supp_ntp = np.where(ntp_pol > support_eps)[0]

        def child_of(tc, nc):
            nonlocal n_endgame_leaves, n_depth_leaves
            c0, c1, status, reward = step(np.int64(s0), np.int64(s1),
                                          np.int64(tc), np.int64(nc), _FULL)
            if int(status) == 2:
                return -1, 2, float(reward)
            key = (int(c0), int(c1))
            if endgame is not None:
                a0 = endgame.value(key[0], key[1])
                if a0 is not None:
                    # Exact endgame leaf: carry A0 value (mover's view) in the
                    # reward slot; -3 marks it for solve_br's role decode.
                    n_endgame_leaves += 1
                    return -3, int(status), float(a0)
            if max_depth is not None and depth[i] + 1 >= max_depth:
                # Depth-limited leaf: beyond the horizon, cap with the FROZEN
                # agent's own value estimate V_net(child) (mover's view) — i.e.
                # assume play is game-fair from here. Decoded identically to an
                # endgame leaf (-3). This UNDER-counts the attacker (it stops
                # best-responding past depth D), so the estimate rises toward the
                # true exploitability as max_depth grows.
                n_depth_leaves += 1
                v = _solve_cached(searcher, solve_cache, key)[0]
                return -3, int(status), float(v)
            ci = index.get(key)
            if ci is None:
                if len(order) >= cap:
                    return -2, int(status), 0.0  # over cap → treat as leaf later
                ci = len(order)
                index[key] = ci
                order.append(key)
                depth.append(depth[i] + 1)
                mover_rows.append(None)
                reactor_cols.append(None)
            return ci, int(status), 0.0

        # V_mover: attacker chooses tp (all rows), frozen plays supp_ntp.
        rows = []
        for a in range(len(tp_codes)):
            cells = []
            for b in supp_ntp:
                ci, st, rew = child_of(int(tp_codes[a]), int(ntp_codes[b]))
                cells.append((ci, st, rew, float(ntp_pol[b])))
            rows.append(cells)
        mover_rows[i] = rows

        # V_reactor: frozen plays supp_tp, attacker chooses ntp (all cols).
        cols = []
        for b in range(len(ntp_codes)):
            cells = []
            for a in supp_tp:
                ci, st, rew = child_of(int(tp_codes[a]), int(ntp_codes[b]))
                cells.append((ci, st, rew, float(tp_pol[a])))
            cols.append(cells)
        reactor_cols[i] = cols

        if verbose and head % 2000 == 0:
            print(f"  enum {head} states ({time.perf_counter()-t0:.0f}s, "
                  f"frontier {len(order)-head})", flush=True)

    if verbose:
        print(f"enumerated {len(order)} states in "
              f"{time.perf_counter()-t0:.0f}s (cap {cap}, "
              f"endgame leaves {n_endgame_leaves}, depth leaves {n_depth_leaves})",
              flush=True)
    return {"order": order, "mover_rows": mover_rows,
            "reactor_cols": reactor_cols, "n": len(order),
            "endgame_leaves": n_endgame_leaves,
            "depth_leaves": n_depth_leaves}


def solve_br(data, gamma: float = 0.999, max_iters: int = 4000,
             eps: float = 1e-7, omega: float = 0.6, verbose: bool = True):
    """Coupled Jacobi VI for (V_mover, V_reactor) = value to the attacker."""
    n = data["n"]
    Vm = np.zeros(n); Vr = np.zeros(n)
    mover_rows = data["mover_rows"]; reactor_cols = data["reactor_cols"]

    def child_val(ci, st, rew, Vm, Vr):
        if st == 2:
            return rew            # terminal reward to the mover (=attacker here)
        if ci == -3:
            # Exact endgame leaf: rew = A0 value (mover's view). Attacker was
            # mover here; if the turn is kept (st 1) the attacker is still mover
            # at the child (V_mover = A0), else it flips to reactor (V = -A0).
            return gamma * (rew if st == 1 else -rew)
        if ci < 0:
            # over-cap leaf: value unknown. Returning 0 (neutral) is OPTIMISTIC
            # for a maximizing attacker (0 > many real losses) ⇒ hitting the cap
            # OVER-estimates the attacker. Only trust results with hit_cap=False.
            return 0.0
        # status 1 keeps mover, status 0 flips it.
        return gamma * (Vm[ci] if st == 1 else Vr[ci])

    def child_val_r(ci, st, rew, Vm, Vr):
        # reactor role: current mover is FROZEN, terminal reward is to frozen →
        # value to attacker is -rew; status 1 keeps frozen-mover (child reactor),
        # status 0 flips to attacker-mover.
        if st == 2:
            return -rew
        if ci == -3:
            # Endgame leaf, attacker was reactor: frozen (mover) secures A0, so
            # attacker gets -A0 if frozen keeps the move (st 1, attacker stays
            # reactor → V_reactor=-A0); if it flips (st 0) attacker becomes mover
            # → V_mover = A0.
            return gamma * (-rew if st == 1 else rew)
        if ci < 0:
            return 0.0
        return gamma * (Vr[ci] if st == 1 else Vm[ci])

    for it in range(1, max_iters + 1):
        Vm_new = np.empty(n); Vr_new = np.empty(n)
        for i in range(n):
            # V_mover: max over tp rows of Σ_ntp σ · childval
            best = -1e18
            for cells in mover_rows[i]:
                acc = 0.0
                for (ci, st, rew, w) in cells:
                    acc += w * child_val(ci, st, rew, Vm, Vr)
                if acc > best:
                    best = acc
            Vm_new[i] = best if best > -1e17 else Vm[i]
            # V_reactor: max over ntp cols of Σ_tp σ · childval_r
            best = -1e18
            for cells in reactor_cols[i]:
                acc = 0.0
                for (ci, st, rew, w) in cells:
                    acc += w * child_val_r(ci, st, rew, Vm, Vr)
                if acc > best:
                    best = acc
            Vr_new[i] = best if best > -1e17 else Vr[i]
        if omega < 1.0:
            Vm_new = (1 - omega) * Vm + omega * Vm_new
            Vr_new = (1 - omega) * Vr + omega * Vr_new
        d = max(np.abs(Vm_new - Vm).max(), np.abs(Vr_new - Vr).max())
        Vm, Vr = Vm_new, Vr_new
        if verbose and (it % 50 == 0 or d < eps):
            print(f"  BR-VI sweep {it}: max_delta {d:.2e}", flush=True)
        if d < eps:
            break
    root_seat0 = float(Vm[0])   # attacker is seat 0 = mover at root
    root_seat1 = float(Vr[0])   # attacker is seat 1 = reactor at root
    attacker_value = 0.5 * (root_seat0 + root_seat1)
    return {
        "attacker_value_seat0": root_seat0,
        "attacker_value_seat1": root_seat1,
        "attacker_value": attacker_value,
        "attacker_winrate": 0.5 * (attacker_value + 1.0),
        "sweeps": it, "converged": bool(d < eps),
    }

"""N7-A: graph value-iteration teacher (net-wired sub-graph Nash-VI).

Wires the value net into ``complete_solver.subgraph_vi`` as the frontier
boundary condition (and the interior warm-start), turning a sampled set of
interior states into **long-horizon teacher values**: values that have had the
game's structure propagated across the whole sampled sub-graph, not just a
fixed 2–4 ply look-ahead. See ``N7_STRATEGY_AND_DESIGN.md`` Part 2.

Production use (later): replace the depth-3/4 backup teacher in the generation
loop with these values. This module is the bridge; ``subgraph_vi`` is the
proven, net-free engine underneath.
"""

from __future__ import annotations

import numpy as np
import torch

from complete_solver.endgame_abstraction import h11_root
from complete_solver.packed_engine import (
    FULL_ALPHABET_MASK,
    legal_ntp_codes,
    legal_tp_codes,
    pack_state,
    step,
)
from complete_solver.state import initial_state
from complete_solver.subgraph_vi import build_subgraph, run_subgraph_vi

from .features import features_from_lanes

_NET_CHUNK = 131_072
_FULL = np.int64(255)
_NOCAP = np.int64(99)


def random_walk_seeds(n_states: int, seed: int = 0, max_steps: int = 40,
                      endgame_frac: float = 0.3):
    """Collect diverse reachable states by uniform-random legal play.

    N7-A coverage (neutral means only — NO skill forcing, per the domain
    constraint): walks use the FULL alphabet with NO stock pruning and pick
    each action uniformly, so lines involving long-horizon skills (stock /
    cement / lock) are visited at their natural random frequency — far more
    than on-policy self-play, which never funnels there (the chicken-and-egg
    coverage gap). These states seed the sub-graph's interior so graph-VI can
    propagate their eventual payoff inward. Returns unique (keys0, keys1)."""
    rng = np.random.default_rng(seed)
    init0, init1 = pack_state(initial_state())
    end0, end1 = pack_state(h11_root())
    tp = np.zeros(96, dtype=np.int64)
    ntp = np.zeros(16, dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    while len(seen) < n_states:
        if rng.random() < endgame_frac:
            l0, l1 = np.int64(end0), np.int64(end1)
        else:
            l0, l1 = np.int64(init0), np.int64(init1)
        for _ in range(max_steps):
            seen.add((int(l0), int(l1)))
            if len(seen) >= n_states:
                break
            n_tp = legal_tp_codes(l0, l1, _FULL, _NOCAP, tp)
            n_ntp = legal_ntp_codes(l0, l1, ntp)
            a = tp[int(rng.integers(n_tp))]
            b = ntp[int(rng.integers(n_ntp))]
            c0, c1, status, _ = step(l0, l1, np.int64(a), np.int64(b), _FULL)
            if status == 2:
                break
            l0, l1 = c0, c1
    arr = np.array(sorted(seen), dtype=np.int64)
    return np.ascontiguousarray(arr[:, 0]), np.ascontiguousarray(arr[:, 1])


def net_values(model, device: str, keys0: np.ndarray, keys1: np.ndarray) -> np.ndarray:
    """V(s) from the value net for packed states, as a float64 1-D array.

    Mirrors ``BatchedSearcher._net_values`` (same chunking) but standalone so
    the teacher does not need a full searcher instance."""
    keys0 = np.ascontiguousarray(keys0, dtype=np.int64)
    keys1 = np.ascontiguousarray(keys1, dtype=np.int64)
    feats = features_from_lanes(keys0, keys1)
    if len(feats) == 0:
        return np.zeros(0, dtype=np.float64)
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), _NET_CHUNK):
            chunk = torch.from_numpy(feats[i:i + _NET_CHUNK]).to(device)
            out.append(model(chunk).float().cpu().numpy().ravel())
    return np.concatenate(out).astype(np.float64)


def graph_vi_teacher(
    model,
    device: str,
    seed0: np.ndarray,
    seed1: np.ndarray,
    *,
    alphabet_mask: int = FULL_ALPHABET_MASK,
    max_stock: int = 3,
    gamma: float = 0.999,
    epsilon: float = 1e-6,
    max_iterations: int = 3000,
    omega: float = 0.5,
    warm_start: bool = True,
    verbose: bool = False,
):
    """Teacher values for the ``seed`` states via net-boundary sub-graph Nash-VI.

    - frontier (unexpanded children) get their value from ``model``;
    - the interior is warm-started from ``model`` (speed only — the fixed point
      is unique regardless of the start);
    - Jacobi Nash-VI runs to ``epsilon``.

    ``alphabet_mask``/``max_stock`` control which children exist and MUST match
    the intended game variant (default: full alphabet, stock ≤ 3 per the domain
    knowledge that 4+ held skills are never worth considering).

    Returns ``(interior_values, tables, info)``.
    """
    tab = build_subgraph(seed0, seed1, alphabet_mask, max_stock)

    if tab.n_front:
        f0, f1 = tab.frontier_keys()
        boundary = net_values(model, device, f0, f1)
    else:
        boundary = np.empty(0, dtype=np.float64)

    interior_init = None
    if warm_start:
        interior_init = net_values(
            model, device, tab.keys0[:tab.n_seed], tab.keys1[:tab.n_seed]
        )

    values, info = run_subgraph_vi(
        tab, boundary, gamma=gamma, epsilon=epsilon,
        max_iterations=max_iterations, interior_init=interior_init,
        omega=omega, verbose=verbose,
    )
    return values, tab, info

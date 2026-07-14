"""N7-F(a): exact endgame tablebase wired into play.

We hold the exact stockless (1,1)-hands endgame solution (``endgame_h11_A0``)
and, whenever the LIVE state is in that table, play the CERTIFIED-optimal LP
mixture instead of the value net's depth-2 approximation. This is an endgame
tablebase (chess-style), not skill forcing: the state key already encodes every
relevant context (guard / time / used-ultimate / previous-skill — verified
present in A0), so the lookup returns the true optimum for that exact context.

Scope (where the table applies): both players at 1 hand AND no stock held.
Anything else (2+ hands, or any stocked skill) is not in A0 and falls through to
the net search. In A0 the legal alphabet has no stock/copy actions, so the
mixture is computed with the table's own ``alphabet_mask`` (0).

The optimum at a state is a ONE-ply zero-sum LP whose children are evaluated by
their EXACT table value — reproducing the stored value to ~1e-9 (asserted in
tests), which certifies the mixture is the true equilibrium.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes, step
from complete_solver.small_matrix import solve_small_zero_sum

_DEFAULT_DB = Path("data/endgame_h11_A0.npz")


class EndgameTablebase:
    """Exact optimal policy lookup over a solved endgame universe.

    ``solve(lane0, lane1)`` mirrors ``BatchedSearcher.solve``'s return shape
    ``(value, tp_codes, ntp_codes, tp_policy, ntp_policy)`` so it is a drop-in
    inside :class:`SearchAgent`. Returns exact values; raises ``KeyError`` only
    if asked to solve a state that is not ``contains``-ed (caller must guard).
    """

    def __init__(self, path: str | Path = _DEFAULT_DB):
        data = np.load(path)
        self.values: dict[tuple[int, int], float] = {
            (int(a), int(b)): float(v)
            for a, b, v in zip(data["keys0"], data["keys1"], data["values"])
        }
        self.gamma = float(data["gamma"][0])
        self.mask = np.int64(int(data["alphabet_mask"][0]))
        self._nocap = np.int64(0)
        self._tp = np.zeros(96, dtype=np.int64)
        self._ntp = np.zeros(16, dtype=np.int64)
        self._cache: dict[tuple[int, int], tuple] = {}

    def __len__(self) -> int:
        return len(self.values)

    def contains(self, lane0: int, lane1: int) -> bool:
        return (int(lane0), int(lane1)) in self.values

    def value(self, lane0: int, lane1: int) -> float | None:
        """Exact game value (mover's view) if known, else None. Usable as an
        exact search leaf for the pincer variant (b)."""
        return self.values.get((int(lane0), int(lane1)))

    def solve(self, lane0: int, lane1: int) -> tuple:
        key = (int(lane0), int(lane1))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        l0, l1 = np.int64(lane0), np.int64(lane1)
        n_tp = legal_tp_codes(l0, l1, self.mask, self._nocap, self._tp)
        n_ntp = legal_ntp_codes(l0, l1, self._ntp)
        matrix = np.zeros((n_tp, n_ntp), dtype=np.float64)
        for a in range(n_tp):
            for b in range(n_ntp):
                c0, c1, status, reward = step(
                    l0, l1, np.int64(self._tp[a]), np.int64(self._ntp[b]),
                    self.mask,
                )
                if status == 2:
                    matrix[a, b] = reward
                else:
                    sign = 1.0 if status == 1 else -1.0
                    matrix[a, b] = sign * self.gamma * self.values[(int(c0), int(c1))]

        value, row, col = solve_small_zero_sum(matrix)
        result = (
            float(value),
            self._tp[:n_tp].copy(),
            self._ntp[:n_ntp].copy(),
            np.ascontiguousarray(row, dtype=np.float64),
            np.ascontiguousarray(col, dtype=np.float64),
        )
        self._cache[key] = result
        return result


class PincerSearcher:
    """N7-F(b) pincer: a search whose leaves that fall inside the endgame table
    are evaluated by their EXACT value instead of the net estimate.

    Wraps a :class:`~complete_ai.batched_search.BatchedSearcher` and overrides
    only ``_net_values`` (the leaf evaluator): for each leaf key present in the
    tablebase we substitute the certified value, so the depth-2 backup for
    states *leading into* the endgame no longer inherits the net's endgame
    error. Non-endgame leaves keep the net value. Used for PLAY only — the
    training/teacher path (nogil-threaded) is deliberately left untouched.

    The subclass is built lazily so importing this module does not require
    torch. Instances behave like a BatchedSearcher (same ``solve`` signature).
    """

    def __new__(cls, model, device, endgame: EndgameTablebase,
                gamma: float = 0.999, prune_stock: bool = True):
        from .batched_search import BatchedSearcher

        table = endgame.values

        class _Impl(BatchedSearcher):
            def _net_values(self, keys0, keys1):
                vals = super()._net_values(keys0, keys1)
                flat = np.reshape(vals, -1)
                k0 = np.ascontiguousarray(keys0)
                k1 = np.ascontiguousarray(keys1)
                for i in range(len(k0)):
                    exact = table.get((int(k0[i]), int(k1[i])))
                    if exact is not None:
                        flat[i] = exact
                return vals

        return _Impl(model=model, device=device, gamma=gamma,
                     prune_stock=prune_stock)


_SHARED: EndgameTablebase | None = None


def load_endgame_tablebase(path: str | Path = _DEFAULT_DB) -> EndgameTablebase | None:
    """Process-wide shared tablebase (loaded once). Returns None if the DB file
    is absent, so callers can degrade gracefully to pure net search."""
    global _SHARED
    if _SHARED is None:
        p = Path(path)
        if not p.exists():
            return None
        _SHARED = EndgameTablebase(p)
    return _SHARED

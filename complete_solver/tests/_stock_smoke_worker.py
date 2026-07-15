"""Subprocess smoke worker: a run_selfplay/play_match pass must COMPLETE
without error under a composed set of "broken stock" toggles (see
_stock_worker.py's docstring for why this must run in a fresh subprocess
with the numba cache cleared).

This does not assert any particular game-theoretic outcome — it only proves
the full compiled pipeline (packed_engine.step/legal_tp_codes with the new
160..191 targeted-STOCK code range, choice_collapse's CHOICE-vs-STOCK-target
bounded grouping, batched_search's njit expand/backup, small_matrix's LP
solves) runs end to end without crashing, hanging, or raising, under a
realistic composition of toggles.

Usage: ``python -m complete_solver.tests._stock_smoke_worker``
(env vars for the toggles under test must already be set on the subprocess
before it starts). Exits 0 and prints "OK" on success.
"""

from __future__ import annotations

import numpy as np

from complete_ai.arena import play_match
from complete_ai.batched_search import BatchedSearcher
from complete_ai.packed_eval import material_leaf_bits
from complete_ai.selfplay import run_selfplay


class MaterialLeafSearcher(BatchedSearcher):
    """Searcher whose 'net' is the cheap material-count leaf — no trained
    model needed, keeps this smoke test fast and deterministic."""

    def __init__(self, prune_stock: bool = True):
        super().__init__(model=None, device="cpu", gamma=0.999,
                         prune_stock=prune_stock)

    def _net_values(self, keys0, keys1):
        out = np.empty(len(keys0), dtype=np.float32)
        for i in range(len(keys0)):
            out[i] = material_leaf_bits(np.int64(keys0[i]), np.int64(keys1[i]))
        return out


searcher_a = MaterialLeafSearcher()
searcher_b = MaterialLeafSearcher()

selfplay_result = run_selfplay(searcher_a, n_games=3, max_plies=25, seed=1)
assert selfplay_result["outcomes"]["terminal"] + selfplay_result["outcomes"]["truncated"] == 3, (
    f"run_selfplay did not account for all games: {selfplay_result}"
)

match_result = play_match(searcher_a, searcher_b, n_games=3, max_plies=25, seed=2)
assert match_result["games"] == 3, f"play_match did not complete all games: {match_result}"
assert match_result["wins_a"] + match_result["wins_b"] + match_result["truncations"] == 3

print("OK")

"""N7-C selective search: tau<0 reproduces uniform depth-d exactly.

The selective searcher with ``tau < 0`` deepens every action, so its value
must equal the trusted uniform depth-d values from ``BatchedSearcher``
(depth-2 = ``solve``; depth-3 = ``value_depth3``, itself reference-checked
against ``FiniteHorizonSolver``). Both run with stock pruning OFF so the action
models match. A positive ``tau`` must then cost strictly fewer deepened cells.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes, step
from complete_solver.state import initial_state
from complete_solver.packed_engine import pack_state

from complete_ai.batched_search import BatchedSearcher
from complete_ai.generation_loop import load_model
from complete_ai.selective_search import SelectiveSearcher

MODEL = Path("models/value_v0.pt")


def sample_states(n: int = 6):
    """Initial state plus a few reachable descendants (varied nodes)."""
    lane0, lane1 = pack_state(initial_state())
    states = [(int(lane0), int(lane1))]
    tp = np.zeros(96, dtype=np.int64)
    ntp = np.zeros(16, dtype=np.int64)
    cur0, cur1 = np.int64(lane0), np.int64(lane1)
    rng = np.random.default_rng(0)
    while len(states) < n:
        nt = legal_tp_codes(cur0, cur1, np.int64(255), np.int64(99), tp)
        nn = legal_ntp_codes(cur0, cur1, ntp)
        a = int(rng.integers(0, nt))
        b = int(rng.integers(0, nn))
        c0, c1, status, _ = step(cur0, cur1, tp[a], ntp[b], np.int64(255))
        if status == 2:
            cur0, cur1 = np.int64(lane0), np.int64(lane1)
            continue
        states.append((int(c0), int(c1)))
        cur0, cur1 = c0, c1
    return states


@unittest.skipUnless(MODEL.exists(), "value_v0.pt required")
class TestSelectiveSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.model = load_model(MODEL, cls.device)
        cls.searcher = BatchedSearcher(cls.model, cls.device, prune_stock=False)
        cls.sel = SelectiveSearcher(cls.model, cls.device, prune=False)
        cls.states = sample_states()

    def test_tau_negative_matches_uniform_depth2(self):
        for s0, s1 in self.states:
            uniform = self.searcher.solve(s0, s1)[0]
            selective = self.sel.value(s0, s1, depth=2, tau=-1.0)
            self.assertAlmostEqual(uniform, selective, delta=1e-6,
                                   msg=f"state=({s0},{s1})")

    def test_tau_negative_matches_uniform_depth3(self):
        # Depth-3 uniform is heavy in pure Python; use the later (smaller-
        # branching) sampled states. Memoization keeps each tractable.
        for s0, s1 in self.states[3:]:
            uniform = self.searcher.value_depth3(s0, s1)
            selective = self.sel.value(s0, s1, depth=3, tau=-1.0)
            self.assertAlmostEqual(uniform, selective, delta=1e-6,
                                   msg=f"state=({s0},{s1})")

    def test_support_pruning_reduces_deepened_cells(self):
        s0, s1 = self.states[0]
        self.sel.reset_stats()
        self.sel.value(s0, s1, depth=3, tau=-1.0)
        uniform_cells = self.sel.stats["deep_cells"]
        self.sel.reset_stats()
        self.sel.value(s0, s1, depth=3, tau=0.02)
        pruned_cells = self.sel.stats["deep_cells"]
        self.assertLess(pruned_cells, uniform_cells)


if __name__ == "__main__":
    unittest.main()

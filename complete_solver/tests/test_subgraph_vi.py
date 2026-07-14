"""N7-A sub-graph Nash-VI: reference-consistency against the closed solver.

Two reference tests, both using the exact stockless (1,1) universe from
``solve_universe`` (= the A0 endgame DB) as ground truth:

1. CLOSED: seed the sub-graph with the ENTIRE universe. No frontier can exist,
   and the interior values must reproduce the exact universe values.
2. OPEN: seed with a SUBSET; inject the exact universe values as the frontier
   boundary. The interior values must still recover the exact universe values —
   this validates the open-boundary mechanism with no value net involved.
"""

from __future__ import annotations

import unittest

import numpy as np

from complete_solver.packed_vi import alphabet_to_mask, solve_universe
from complete_solver.state import PlayerState, State
from complete_solver.subgraph_vi import build_subgraph, run_subgraph_vi

GAMMA = 0.999


def no_stock_root() -> State:
    return State(
        me=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
        opp=PlayerState(hands=1, used_ultimate=True, has_declared_skill=True),
    )


class TestSubgraphVI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exact ground truth: the closed stockless (1,1) universe.
        cls.mask = alphabet_to_mask(frozenset())
        cls.max_stock = 99  # solve_universe default (mask=0 forbids stocking)
        cls.db, cls.info = solve_universe(
            no_stock_root(), alphabet=frozenset(), gamma=GAMMA,
            max_states=500_000, verbose=False,
        )
        # Values keyed by (lane0, lane1) for lookups regardless of ordering.
        cls.truth = {
            (int(k0), int(k1)): float(v)
            for k0, k1, v in zip(cls.db.keys0, cls.db.keys1, cls.db.values)
        }
        cls.all0 = cls.db.keys0.astype(np.int64)
        cls.all1 = cls.db.keys1.astype(np.int64)

    def test_closed_seed_set_matches_solve_universe(self):
        """Whole universe as seeds ⇒ zero frontier ⇒ exact reproduction."""
        tab = build_subgraph(self.all0, self.all1, self.mask, self.max_stock)
        self.assertEqual(tab.n_front, 0, "closed set must have no frontier")
        values, info = run_subgraph_vi(
            tab, np.empty(0, dtype=np.float64), gamma=GAMMA,
        )
        self.assertTrue(info["converged"])
        truth = np.array([self.truth[(int(a), int(b))]
                          for a, b in zip(tab.keys0[:tab.n_seed],
                                          tab.keys1[:tab.n_seed])])
        self.assertLess(np.abs(values - truth).max(), 1e-6)

    def test_open_subset_with_exact_boundary_recovers_truth(self):
        """A subset with the exact universe values as boundary must recover
        the exact interior values (Bellman fixed point with true frontier)."""
        rng = np.random.default_rng(0)
        n = self.all0.shape[0]
        sel = rng.choice(n, size=min(4000, n // 2), replace=False)
        seed0 = self.all0[sel]
        seed1 = self.all1[sel]

        tab = build_subgraph(seed0, seed1, self.mask, self.max_stock)
        self.assertGreater(tab.n_front, 0, "a strict subset must have frontier")

        f0, f1 = tab.frontier_keys()
        boundary = np.array([self.truth[(int(a), int(b))]
                             for a, b in zip(f0, f1)], dtype=np.float64)

        values, info = run_subgraph_vi(tab, boundary, gamma=GAMMA)
        self.assertTrue(info["converged"])
        truth = np.array([self.truth[(int(a), int(b))]
                          for a, b in zip(tab.keys0[:tab.n_seed],
                                          tab.keys1[:tab.n_seed])])
        self.assertLess(np.abs(values - truth).max(), 1e-6)

    def test_warm_start_does_not_change_fixed_point(self):
        """Interior warm-start changes speed, not the converged values."""
        rng = np.random.default_rng(1)
        n = self.all0.shape[0]
        sel = rng.choice(n, size=min(2000, n // 2), replace=False)
        tab = build_subgraph(self.all0[sel], self.all1[sel],
                             self.mask, self.max_stock)
        f0, f1 = tab.frontier_keys()
        boundary = np.array([self.truth[(int(a), int(b))]
                             for a, b in zip(f0, f1)], dtype=np.float64)
        cold, _ = run_subgraph_vi(tab, boundary, gamma=GAMMA)
        warm_init = rng.uniform(-1.0, 1.0, size=tab.n_seed)
        warm, _ = run_subgraph_vi(tab, boundary, gamma=GAMMA,
                                  interior_init=warm_init)
        self.assertLess(np.abs(cold - warm).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()

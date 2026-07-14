"""N7-F(a): exact endgame tablebase — correctness + agent wiring."""

import unittest
from pathlib import Path

import numpy as np

from complete_solver.endgame_abstraction import h11_root
from complete_solver.state import initial_state
from complete_solver.packed_engine import pack_state

DB = Path("data/endgame_h11_A0.npz")


@unittest.skipUnless(DB.exists(), "A0 endgame DB not built")
class TestEndgameTablebase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from complete_ai.endgame_table import EndgameTablebase
        cls.tb = EndgameTablebase(DB)
        cls.rng = np.random.default_rng(0)

    def _sample_keys(self, n):
        keys = list(self.tb.values.keys())
        idx = self.rng.choice(len(keys), size=min(n, len(keys)), replace=False)
        return [keys[i] for i in idx]

    def test_value_matches_db_exactly(self):
        """The one-ply LP over exact child values must REPRODUCE the stored
        endgame value — this certifies the returned mixture is the true
        equilibrium (not an approximation)."""
        max_err = 0.0
        for k0, k1 in self._sample_keys(3000):
            value, *_ = self.tb.solve(k0, k1)
            max_err = max(max_err, abs(value - self.tb.values[(k0, k1)]))
        self.assertLess(max_err, 1e-6, f"LP value diverges from DB: {max_err:.2e}")

    def test_policies_are_valid_distributions(self):
        for k0, k1 in self._sample_keys(2000):
            _, tp_codes, ntp_codes, tp_pol, ntp_pol = self.tb.solve(k0, k1)
            self.assertEqual(len(tp_codes), len(tp_pol))
            self.assertEqual(len(ntp_codes), len(ntp_pol))
            for pol in (tp_pol, ntp_pol):
                self.assertTrue((pol >= -1e-9).all())
                self.assertAlmostEqual(float(pol.sum()), 1.0, places=6)

    def test_contains_boundary(self):
        """h11_root (1,1 stockless) is in the table; the opening (more hands)
        is not."""
        e0, e1 = pack_state(h11_root())
        self.assertTrue(self.tb.contains(e0, e1))
        i0, i1 = pack_state(initial_state())
        self.assertFalse(self.tb.contains(i0, i1))

    def test_agent_uses_table_in_endgame(self):
        """A SearchAgent with the tablebase must route in-table states through
        the exact optimum (its move must be a legal TP of that mixture)."""
        from complete_ai.agents import SearchAgent

        class _StubSearcher:
            def solve(self, l0, l1):
                raise AssertionError("net search must NOT be called in-table")

        agent = SearchAgent(_StubSearcher(), np.random.default_rng(1),
                            endgame=self.tb, deterministic=True)
        e0, e1 = pack_state(h11_root())
        _, tp_codes, _, _, _ = self.tb.solve(e0, e1)
        move = agent.tp_action(e0, e1)
        self.assertIn(move, [int(c) for c in tp_codes])


_MODEL = Path("models/value_gvi_latest.pt")


@unittest.skipUnless(DB.exists() and _MODEL.exists(), "A0 DB or model missing")
class TestPincerSearcher(unittest.TestCase):
    """N7-F(b): exact endgame leaves inside the depth-2 search must drive the
    backed-up value of A0 states to their EXACT value (net-only cannot)."""

    def test_pincer_value_matches_exact_on_a0(self):
        import torch
        from complete_ai.generation_loop import load_model
        from complete_ai.endgame_table import EndgameTablebase, PincerSearcher

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(_MODEL, device)
        tb = EndgameTablebase(DB)
        pin = PincerSearcher(model, device, tb, prune_stock=True)

        data = np.load(DB)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data["keys0"]), size=200, replace=False)
        max_err = 0.0
        for i in idx:
            k0, k1 = int(data["keys0"][i]), int(data["keys1"][i])
            v = pin.solve(k0, k1)[0]
            max_err = max(max_err, abs(v - float(data["values"][i])))
        # All depth-2 leaves of an A0 state are themselves in A0 → exact leaves
        # → the backup reproduces the exact value (small tol for LP/degenerate).
        self.assertLess(max_err, 0.02, f"pincer value not exact: {max_err:.4f}")


if __name__ == "__main__":
    unittest.main()

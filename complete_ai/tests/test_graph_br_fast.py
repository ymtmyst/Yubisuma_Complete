"""N7-E: the njit graph-BR VI must match the pure-Python reference exactly."""

import unittest

import numpy as np


def _toy_graph():
    """A hand-built tiny BR graph exercising every cell kind: internal edges
    (both status), terminal, endgame-exact (-3) with both statuses, over-cap
    (-2). Two states, so the coupled VI has something to iterate."""
    # state 0 mover rows (attacker as mover): two tp-rows.
    mover_rows = [
        [
            # row A: internal→state1 (st1), endgame leaf (st0, A0=0.3)
            [(1, 1, 0.0, 0.6), (-3, 0, 0.3, 0.4)],
            # row B: terminal win (st2 rew +1), over-cap (-2)
            [(-1, 2, 1.0, 0.5), (-2, 0, 0.0, 0.5)],
        ],
        # state 1 mover rows
        [
            [(-3, 1, -0.2, 1.0)],                      # endgame leaf, st1
        ],
    ]
    reactor_cols = [
        [
            [(1, 0, 0.0, 0.7), (-1, 2, -1.0, 0.3)],    # internal st0 + terminal
        ],
        [
            [(-3, 0, 0.5, 0.5), (0, 1, 0.0, 0.5)],     # endgame + internal→state0
        ],
    ]
    return {"n": 2, "mover_rows": mover_rows, "reactor_cols": reactor_cols,
            "endgame_leaves": 0, "depth_leaves": 0}


class TestGraphBrFastParity(unittest.TestCase):
    def test_njit_matches_python_vi(self):
        from complete_ai.graph_br import solve_br
        from complete_ai.graph_br_fast import solve_br_njit

        data = _toy_graph()
        ref = solve_br(dict(data), gamma=0.9, omega=0.6, verbose=False)
        fast = solve_br_njit(dict(data), gamma=0.9, omega=0.6, verbose=False)
        for key in ("attacker_value_seat0", "attacker_value_seat1",
                    "attacker_value", "attacker_winrate"):
            self.assertAlmostEqual(ref[key], fast[key], places=6,
                                   msg=f"{key}: py={ref[key]} njit={fast[key]}")

    def test_random_graphs_match(self):
        """Fuzz several random small graphs — njit and python VI must agree."""
        from complete_ai.graph_br import solve_br
        from complete_ai.graph_br_fast import solve_br_njit

        rng = np.random.default_rng(0)

        def _norm_row(cells):
            """Weights are the frozen policy's probabilities within a row — they
            sum to 1. Normalising keeps the VI a contraction (gamma<1) so values
            stay bounded and the parity check is meaningful."""
            tot = sum(c[3] for c in cells) or 1.0
            return [(c[0], c[1], c[2], c[3] / tot) for c in cells]

        for trial in range(8):
            n = int(rng.integers(2, 6))
            mover_rows = []
            reactor_cols = []
            for _ in range(n):
                mrows = []
                for _ in range(int(rng.integers(1, 4))):
                    cells = []
                    for _ in range(int(rng.integers(1, 4))):
                        kind = rng.integers(0, 4)
                        w = float(rng.random()) + 0.1
                        if kind == 0:
                            cells.append((int(rng.integers(0, n)),
                                          int(rng.integers(0, 2)), 0.0, w))
                        elif kind == 1:
                            cells.append((-1, 2, float(rng.uniform(-1, 1)), w))
                        elif kind == 2:
                            cells.append((-3, int(rng.integers(0, 2)),
                                          float(rng.uniform(-1, 1)), w))
                        else:
                            cells.append((-2, int(rng.integers(0, 2)), 0.0, w))
                    mrows.append(_norm_row(cells))
                mover_rows.append(mrows)
                rcols = []
                for _ in range(int(rng.integers(1, 4))):
                    cells = []
                    for _ in range(int(rng.integers(1, 4))):
                        cells.append((int(rng.integers(0, n)),
                                      int(rng.integers(0, 2)), 0.0,
                                      float(rng.random()) + 0.1))
                    rcols.append(_norm_row(cells))
                reactor_cols.append(rcols)
            data = {"n": n, "mover_rows": mover_rows,
                    "reactor_cols": reactor_cols}
            ref = solve_br(dict(data), gamma=0.95, omega=0.5, verbose=False)
            fast = solve_br_njit(dict(data), gamma=0.95, omega=0.5, verbose=False)
            self.assertAlmostEqual(ref["attacker_winrate"],
                                   fast["attacker_winrate"], places=5,
                                   msg=f"trial {trial}")


from pathlib import Path

_DB = Path("data/endgame_h11_A0.npz")
_MODEL = Path("models/value_gvi_latest.pt")


@unittest.skipUnless(_DB.exists() and _MODEL.exists(), "A0 DB or model missing")
class TestFastEnumEndToEnd(unittest.TestCase):
    """The fast level-synchronous enum + njit VI must produce a converged, sane
    exploitability and use the exact endgame cap. (Exact parity to the Python
    ``enumerate_br`` is NOT expected: the fast engine caps depth by BFS-distance
    — each state expanded once at its shortest depth — whereas the Python
    reference caps by path length, so the two treat transpositions differently.
    The fast/BFS-distance semantics is the intended, more-correct one.)"""

    def test_fast_enum_converges_and_is_monotone(self):
        import torch
        from complete_ai.generation_loop import load_model
        from complete_ai.endgame_table import load_endgame_tablebase, PincerSearcher
        from complete_ai.graph_br_fast import enumerate_solve_fast, solve_flat

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(_MODEL, dev)
        eg = load_endgame_tablebase()
        def factory():
            return PincerSearcher(model, dev, eg, prune_stock=True)

        prev = None
        for D in (2, 3, 4):
            flat = enumerate_solve_fast(factory, eg, max_depth=D, cap=200000,
                                        n_threads=4, verbose=False)
            res = solve_flat(flat, gamma=0.999, verbose=False)
            self.assertTrue(res["converged"])
            self.assertGreater(res["attacker_winrate"], 0.45)
            self.assertLess(res["attacker_winrate"], 0.65)
            # deeper best-response can only find MORE exploit (monotone up).
            if prev is not None:
                self.assertGreaterEqual(res["attacker_winrate"] + 1e-6, prev)
            prev = res["attacker_winrate"]

    def test_fast_matches_reference_at_base_depth(self):
        """The BFS-distance fast enum and the path-length reference enum agree
        EXACTLY at D=2 (the shallowest cap, where no revisited interior state is
        reached deep enough for the two depth-cap philosophies to diverge). This
        pins the intended relationship: fast == ref at D=2, then fast >= ref as D
        grows. A regression here means the two engines have genuinely drifted
        apart at the base case — a real bug, unlike the by-design deeper gap."""
        import torch
        from complete_ai.generation_loop import load_model
        from complete_ai.endgame_table import load_endgame_tablebase, PincerSearcher
        from complete_ai.graph_br import enumerate_br, solve_br
        from complete_ai.graph_br_fast import enumerate_solve_fast, solve_flat

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(_MODEL, dev)
        eg = load_endgame_tablebase()
        def factory():
            return PincerSearcher(model, dev, eg, prune_stock=True)

        ref = solve_br(enumerate_br(factory(), cap=200000, endgame=eg,
                                    max_depth=2, verbose=False),
                       gamma=0.999, verbose=False)
        fast = solve_flat(enumerate_solve_fast(factory, eg, max_depth=2,
                                               cap=200000, n_threads=4,
                                               verbose=False),
                          gamma=0.999, verbose=False)
        self.assertAlmostEqual(ref["attacker_value"], fast["attacker_value"],
                               places=4,
                               msg=f"D=2 base case drifted: ref={ref['attacker_value']} "
                                   f"fast={fast['attacker_value']}")


if __name__ == "__main__":
    unittest.main()

"""Fast depth-limited LP-backup search for Complete rules.

This module is the N1 deliverable of AI_MASTER_PLAN_V2: a drop-in,
value-identical replacement for :class:`FiniteHorizonSolver` that is fast
enough to drive self-play data generation and real-time play.

Speed techniques (all exact — the computed game value matches the reference
solver up to LP tolerance):

1. Persistent transposition table keyed by (state, depth). Results survive
   across searches, so iterative deepening and self-play reuse earlier work.
2. Lazy, cached node expansion: `transition()` for a (state, tp, ntp) cell is
   computed at most once per process, and only when the search actually needs
   that cell.
3. Double-oracle node solving: instead of filling the full payoff matrix
   (~45×9 at the opening), solve a small restricted matrix game and grow it
   with best responses until neither player can deviate profitably. At
   convergence the restricted equilibrium is an exact equilibrium of the full
   matrix, but untouched cells (and their whole subtrees) are never computed.
4. LP avoidance for easy nodes: pure saddle-point detection, 1×N / N×1
   maximin, and a closed-form 2×2 mixed solution. The scipy LP only runs for
   genuinely mixed nodes of size ≥ 2×3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .actions import (
    NTPAction,
    RulesConfig,
    TPAction,
    legal_ntp_actions,
    legal_tp_actions,
)
from .choice_collapse import choice_row_groups, resolve_choice_action
from .finite_horizon import material_leaf_evaluator
from .small_matrix import solve_small_zero_sum
from .state import State
from .transition import transition

LeafEvaluator = Callable[[State], float]

_EPS = 1e-9


@dataclass(frozen=True)
class FastStatePolicy:
    """Root solve result. Policies are full-length over the action tuples."""

    value: float
    tp_actions: tuple[TPAction, ...]
    ntp_actions: tuple[NTPAction, ...]
    tp_policy: tuple[float, ...]
    ntp_policy: tuple[float, ...]


@dataclass
class _Expansion:
    """Lazily filled per-state child table shared across depths/searches."""

    tp_actions: tuple[TPAction, ...]
    ntp_actions: tuple[NTPAction, ...]
    # (row, col) -> ("T", reward) | ("S", sign, child_state)
    cells: dict[tuple[int, int], tuple] = field(default_factory=dict)
    # CHOICE post-reaction collapse (see choice_collapse.py): collapsed_tp_actions
    # is tp_actions with same-thumb CHOICE groups (>=2 stocked skills) replaced
    # by one pseudo action; row_sources[k] gives the RAW tp_actions indices the
    # collapsed row k aggregates (max). Computed once per expansion.
    collapsed_tp_actions: tuple[TPAction, ...] = ()
    row_sources: tuple[tuple[int, ...], ...] = ()


@dataclass
class SolverStats:
    expansions: int = 0
    cell_evals: int = 0
    tt_hits: int = 0
    matrix_solves: int = 0
    do_iterations: int = 0
    exact_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "expansions": self.expansions,
            "cell_evals": self.cell_evals,
            "tt_hits": self.tt_hits,
            "matrix_solves": self.matrix_solves,
            "do_iterations": self.do_iterations,
            "exact_hits": self.exact_hits,
        }


class FastHorizonSolver:
    """Depth-limited exact subgame solver with persistent caches.

    Values match :class:`FiniteHorizonSolver` semantics exactly:
    ``value(s, 0) = leaf(s)`` and one-step payoffs are
    ``terminal_reward`` or ``sign * gamma * value(child, depth-1)``.
    """

    def __init__(
        self,
        config: RulesConfig = RulesConfig(),
        gamma: float = 0.999,
        leaf_evaluator: LeafEvaluator | None = None,
        use_double_oracle: bool = True,
        full_matrix_threshold: int = 16,
        exact_values: dict[State, float] | None = None,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma <= 1")
        self.config = config
        self.gamma = gamma
        self.leaf_evaluator = leaf_evaluator or material_leaf_evaluator
        self.use_double_oracle = use_double_oracle
        self.full_matrix_threshold = full_matrix_threshold
        # Exact game values (endgame DB). Must be computed with the same gamma;
        # a hit short-circuits search at any depth because the value is exact.
        self.exact_values = exact_values if exact_values is not None else {}

        self._expansions: dict[State, _Expansion] = {}
        self._tt: dict[tuple[State, int], float] = {}
        self._leaf_cache: dict[State, float] = {}
        # Warm-start supports per state: (row indices, col indices)
        self._support: dict[State, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self.stats = SolverStats()

    # ── public API ────────────────────────────────────────────────────────

    def value(self, state: State, depth: int) -> float:
        exact = self.exact_values.get(state)
        if exact is not None:
            self.stats.exact_hits += 1
            return exact
        if depth <= 0:
            return self._leaf(state)
        key = (state, depth)
        cached = self._tt.get(key)
        if cached is not None:
            self.stats.tt_hits += 1
            return cached
        value, _, _, _ = self._solve_node(state, depth)
        self._tt[key] = value
        return value

    def solve_state(self, state: State, depth: int) -> FastStatePolicy:
        """Solve the root and return value plus full-length mixed policies.

        ``tp_actions``/``tp_policy`` are over the COLLAPSED action space (see
        choice_collapse.py): a CHOICE thumb backed by 2+ choosable stock is
        ONE pseudo action (``choice=None``) whose probability is the mover's
        probability of declaring that thumb — which concrete skill fires is a
        post-reaction pick (``resolve_choice``), not part of this policy.
        """
        if depth < 1:
            return FastStatePolicy(self._leaf(state), (), (), (), ())
        value, tp_actions, row_policy, col_policy = self._solve_node(state, depth)
        self._tt[(state, depth)] = value
        exp = self._expand(state)
        return FastStatePolicy(
            value=value,
            tp_actions=tp_actions,
            ntp_actions=exp.ntp_actions,
            tp_policy=tuple(row_policy),
            ntp_policy=tuple(col_policy),
        )

    def resolve_choice(
        self, state: State, thumb: int, ntp_action: NTPAction, depth: int
    ) -> TPAction:
        """Post-reaction skill pick for a CHOICE(thumb) declaration, now that
        the opponent's realized reaction ``ntp_action`` is known."""
        exp = self._expand(state)
        col = exp.ntp_actions.index(ntp_action)

        def payoff_fn(candidate: TPAction, _ntp: NTPAction) -> float:
            row = exp.tp_actions.index(candidate)
            return self._payoff(state, exp, row, col, depth)

        return resolve_choice_action(state, thumb, ntp_action, payoff_fn)

    def clear_caches(self) -> None:
        self._expansions.clear()
        self._tt.clear()
        self._leaf_cache.clear()
        self._support.clear()
        self.stats = SolverStats()

    def cache_sizes(self) -> dict:
        return {
            "expansions": len(self._expansions),
            "transposition": len(self._tt),
            "leaves": len(self._leaf_cache),
        }

    # ── internals ─────────────────────────────────────────────────────────

    def _leaf(self, state: State) -> float:
        cached = self._leaf_cache.get(state)
        if cached is None:
            cached = float(self.leaf_evaluator(state))
            self._leaf_cache[state] = cached
        return cached

    def _expand(self, state: State) -> _Expansion:
        exp = self._expansions.get(state)
        if exp is None:
            tp_actions = legal_tp_actions(state, self.config)
            collapsed_tp_actions, row_sources = choice_row_groups(tp_actions)
            exp = _Expansion(
                tp_actions=tp_actions,
                ntp_actions=legal_ntp_actions(state, self.config),
                collapsed_tp_actions=collapsed_tp_actions,
                row_sources=row_sources,
            )
            self._expansions[state] = exp
            self.stats.expansions += 1
        return exp

    def _cell(self, state: State, exp: _Expansion, row: int, col: int) -> tuple:
        cell = exp.cells.get((row, col))
        if cell is None:
            result = transition(
                state, exp.tp_actions[row], exp.ntp_actions[col], self.config
            )
            if result.terminal_reward is not None:
                cell = ("T", float(result.terminal_reward))
            else:
                sign = 1.0 if result.same_turn_player else -1.0
                cell = ("S", sign, result.next_state)
            exp.cells[(row, col)] = cell
        return cell

    def _payoff(self, state: State, exp: _Expansion, row: int, col: int, depth: int) -> float:
        cell = self._cell(state, exp, row, col)
        self.stats.cell_evals += 1
        if cell[0] == "T":
            return cell[1]
        _, sign, child = cell
        return sign * self.gamma * self.value(child, depth - 1)

    def _solve_node(
        self, state: State, depth: int
    ) -> tuple[float, tuple[TPAction, ...], np.ndarray, np.ndarray]:
        """Solve one node's (collapsed) matrix game.

        Returns ``(value, collapsed_tp_actions, row_policy, col_policy)``.
        CHOICE rows sharing a thumb (2+ choosable stock) are collapsed to one
        row = the per-column max BEFORE the LP/double-oracle solve — the
        post-reaction skill pick (see choice_collapse.py). ``row_policy`` is
        over ``collapsed_tp_actions``, not the raw ``exp.tp_actions``.
        """
        exp = self._expand(state)
        n_rows = len(exp.collapsed_tp_actions)
        n_cols = len(exp.ntp_actions)
        row_sources = exp.row_sources

        if not self.use_double_oracle or n_rows * n_cols <= self.full_matrix_threshold:
            matrix = np.empty((n_rows, n_cols), dtype=float)
            for i in range(n_rows):
                sources = row_sources[i]
                if len(sources) == 1:
                    for j in range(n_cols):
                        matrix[i, j] = self._payoff(state, exp, sources[0], j, depth)
                else:
                    for j in range(n_cols):
                        matrix[i, j] = max(
                            self._payoff(state, exp, raw_i, j, depth)
                            for raw_i in sources
                        )
            value, row_policy, col_policy = self._solve_matrix(matrix)
            return value, exp.collapsed_tp_actions, row_policy, col_policy

        return self._solve_double_oracle(state, exp, depth)

    def _solve_double_oracle(
        self, state: State, exp: _Expansion, depth: int
    ) -> tuple[float, tuple[TPAction, ...], np.ndarray, np.ndarray]:
        row_sources = exp.row_sources
        n_rows = len(exp.collapsed_tp_actions)
        n_cols = len(exp.ntp_actions)

        warm = self._support.get(state)
        if warm is not None:
            rows = [i for i in warm[0] if i < n_rows] or [0]
            cols = [j for j in warm[1] if j < n_cols] or [0]
        else:
            rows, cols = [0], [0]

        # Column generation: computed payoffs are cached as numpy vectors so
        # repeated best-response scans across DO iterations cost numpy dots,
        # not fresh child evaluations.
        cells = exp.cells
        tp_actions = exp.tp_actions
        ntp_actions = exp.ntp_actions
        gamma = self.gamma
        config = self.config
        value_fn = self.value
        child_depth = depth - 1
        cell_evals = 0

        def raw_payoff(i: int, j: int) -> float:
            """Payoff of a RAW (uncollapsed) tp_actions row i vs column j."""
            nonlocal cell_evals
            cell_evals += 1
            cell = cells.get((i, j))
            if cell is None:
                result = transition(state, tp_actions[i], ntp_actions[j], config)
                if result.terminal_reward is not None:
                    cell = ("T", float(result.terminal_reward))
                else:
                    sign = 1.0 if result.same_turn_player else -1.0
                    cell = ("S", sign, result.next_state)
                cells[(i, j)] = cell
            if cell[0] == "T":
                return cell[1]
            return cell[1] * gamma * value_fn(cell[2], child_depth)

        def payoff(i: int, j: int) -> float:
            """Payoff of a COLLAPSED row i (index into exp.collapsed_tp_actions)
            vs column j — the per-column max over its raw source rows (a
            single-element max for pass-through rows)."""
            sources = row_sources[i]
            if len(sources) == 1:
                return raw_payoff(sources[0], j)
            return max(raw_payoff(raw_i, j) for raw_i in sources)

        # col_vectors[j] = payoffs of every row against column j (length n_rows)
        # row_vectors[i] = payoffs of row i against every column (length n_cols)
        col_vectors: dict[int, np.ndarray] = {}
        row_vectors: dict[int, np.ndarray] = {}

        def col_vector(j: int) -> np.ndarray:
            vec = col_vectors.get(j)
            if vec is None:
                vec = np.fromiter(
                    (payoff(i, j) for i in range(n_rows)), dtype=float, count=n_rows
                )
                col_vectors[j] = vec
            return vec

        def row_vector(i: int) -> np.ndarray:
            vec = row_vectors.get(i)
            if vec is None:
                vec = np.fromiter(
                    (payoff(i, j) for j in range(n_cols)), dtype=float, count=n_cols
                )
                row_vectors[i] = vec
            return vec

        value = 0.0
        sub_x = np.array([1.0])
        sub_y = np.array([1.0])

        # Each iteration adds at least one row or column, so this terminates.
        for _ in range(n_rows + n_cols + 1):
            self.stats.do_iterations += 1
            sub = np.empty((len(rows), len(cols)), dtype=float)
            for b, j in enumerate(cols):
                sub[:, b] = col_vector(j)[rows]
            self.stats.matrix_solves += 1
            value, sub_x, sub_y = solve_small_zero_sum(sub)

            # Row player's best response against the current column mixture.
            row_payoffs = np.zeros(n_rows)
            for b, j in enumerate(cols):
                weight = sub_y[b]
                if weight > _EPS:
                    row_payoffs += weight * col_vector(j)
            best_row = int(np.argmax(row_payoffs))
            best_row_value = float(row_payoffs[best_row])

            # Column player's best response against the current row mixture.
            col_payoffs = np.zeros(n_cols)
            for a, i in enumerate(rows):
                weight = sub_x[a]
                if weight > _EPS:
                    col_payoffs += weight * row_vector(i)
            best_col = int(np.argmin(col_payoffs))
            best_col_value = float(col_payoffs[best_col])

            grew = False
            if best_row_value > value + 1e-7 and best_row not in rows:
                rows.append(best_row)
                grew = True
            if best_col_value < value - 1e-7 and best_col not in cols:
                cols.append(best_col)
                grew = True
            if not grew:
                break

        self.stats.cell_evals += cell_evals
        self._support[state] = (tuple(rows), tuple(cols))

        row_policy = np.zeros(n_rows)
        col_policy = np.zeros(n_cols)
        for a, i in enumerate(rows):
            row_policy[i] = sub_x[a]
        for b, j in enumerate(cols):
            col_policy[j] = sub_y[b]
        return value, exp.collapsed_tp_actions, row_policy, col_policy

    def _solve_matrix(self, matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        self.stats.matrix_solves += 1
        return solve_small_zero_sum(matrix)

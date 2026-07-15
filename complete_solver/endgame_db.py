"""Reachability-driven exact endgame database (N2 of AI_MASTER_PLAN_V2).

Hands never increase in this game (nothing restores a lowered hand with
mirror/reversi off), so the set of states reachable from any given state is
"closed": play can never leave it. If that closure is small enough to
enumerate, discounted Shapley value iteration over it yields values that are
EXACTLY the full-game values of every state inside — the rest of the game is
unreachable from there and cannot matter.

A comprehensive chess-style tablebase is combinatorially impossible here
(stock subsets alone give ~2^8 per player), so instead the database grows
on demand: feed it endgame states actually encountered in play or training;
each state whose closure fits a state budget gets its whole closure solved
and merged into a persistent value store. Probing is an O(1) dict lookup.
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from .choice_collapse import choice_row_groups, collapse_rows_by_sources
from .small_matrix import solve_small_zero_sum
from .state import State
from .transition import transition


@dataclass(frozen=True)
class ClosureResult:
    states: frozenset[State] | None   # None → budget exceeded
    visited: int


def compute_closure(
    root: State,
    config: RulesConfig = RulesConfig(),
    max_states: int = 100_000,
) -> ClosureResult:
    """BFS every non-terminal state reachable from *root* (root included)."""
    visited: set[State] = {root}
    queue: deque[State] = deque([root])
    while queue:
        state = queue.popleft()
        for tp_action in legal_tp_actions(state, config):
            for ntp_action in legal_ntp_actions(state, config):
                result = transition(state, tp_action, ntp_action, config)
                child = result.next_state
                if child is not None and child not in visited:
                    if len(visited) >= max_states:
                        return ClosureResult(None, len(visited))
                    visited.add(child)
                    queue.append(child)
    return ClosureResult(frozenset(visited), len(visited))


@dataclass(frozen=True)
class ClosureSolution:
    values: dict[State, float]
    iterations: int
    max_delta: float
    converged: bool
    max_bellman_residual: float


def solve_closure(
    states: frozenset[State],
    config: RulesConfig = RulesConfig(),
    gamma: float = 0.999,
    epsilon: float = 1e-9,
    max_iterations: int = 20_000,
    known_values: dict[State, float] | None = None,
) -> ClosureSolution:
    """Gauss-Seidel Shapley value iteration over a closed state set.

    *known_values* (e.g. an existing database) seeds the sweep and pins
    already-solved states so they are not recomputed.
    """
    known = known_values or {}
    unsolved = [s for s in states if s not in known]
    index = {s: i for i, s in enumerate(unsolved)}
    n = len(unsolved)

    # Precompute per-state cell structure. Cells are either a constant payoff
    # (terminal, or a state pinned by *known*) or (sign, child index).
    const_m: list[np.ndarray] = []
    idx_m: list[np.ndarray] = []
    sign_m: list[np.ndarray] = []
    # CHOICE fix: the row grouping (which raw rows collapse to a per-column
    # max — see choice_collapse.py) depends only on the action LIST, which is
    # fixed across VI iterations, so it is precomputed once per state here.
    row_sources_m: list[list[tuple[int, ...]]] = []
    for state in unsolved:
        tp_actions = legal_tp_actions(state, config)
        ntp_actions = legal_ntp_actions(state, config)
        _, row_sources = choice_row_groups(tp_actions)
        row_sources_m.append(row_sources)
        rows, cols = len(tp_actions), len(ntp_actions)
        const = np.zeros((rows, cols))
        idx = np.full((rows, cols), -1, dtype=np.int64)
        sign = np.zeros((rows, cols))
        for r, tp_action in enumerate(tp_actions):
            for c, ntp_action in enumerate(ntp_actions):
                result = transition(state, tp_action, ntp_action, config)
                if result.terminal_reward is not None:
                    const[r, c] = result.terminal_reward
                    continue
                child = result.next_state
                cell_sign = 1.0 if result.same_turn_player else -1.0
                child_index = index.get(child)
                if child_index is None:
                    known_value = known.get(child)
                    if known_value is None:
                        raise ValueError(
                            "state set is not closed: unknown successor "
                            f"{child} from {state}"
                        )
                    const[r, c] = cell_sign * gamma * known_value
                else:
                    idx[r, c] = child_index
                    sign[r, c] = cell_sign
        const_m.append(const)
        idx_m.append(idx)
        sign_m.append(sign)

    values = np.zeros(n)
    max_delta = 0.0
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        max_delta = 0.0
        for i in range(n):
            idx = idx_m[i]
            matrix = const_m[i] + np.where(
                idx >= 0, sign_m[i] * gamma * values[np.maximum(idx, 0)], 0.0
            )
            matrix = collapse_rows_by_sources(matrix, row_sources_m[i])
            new_value, _, _ = solve_small_zero_sum(matrix)
            delta = abs(new_value - values[i])
            if delta > max_delta:
                max_delta = delta
            values[i] = new_value  # Gauss-Seidel: reuse immediately
        if max_delta < epsilon:
            break

    # Bellman residual check: converged values must reproduce themselves.
    max_residual = 0.0
    for i in range(n):
        idx = idx_m[i]
        matrix = const_m[i] + np.where(
            idx >= 0, sign_m[i] * gamma * values[np.maximum(idx, 0)], 0.0
        )
        matrix = collapse_rows_by_sources(matrix, row_sources_m[i])
        value, _, _ = solve_small_zero_sum(matrix)
        max_residual = max(max_residual, abs(value - values[i]))

    solved = {state: float(values[index[state]]) for state in unsolved}
    return ClosureSolution(
        values=solved,
        iterations=iteration,
        max_delta=max_delta,
        converged=max_delta < epsilon,
        max_bellman_residual=max_residual,
    )


class EndgameDB:
    """Persistent exact-value store, grown closure by closure."""

    def __init__(
        self,
        config: RulesConfig = RulesConfig(),
        gamma: float = 0.999,
        max_closure_states: int = 100_000,
    ):
        self.config = config
        self.gamma = gamma
        self.max_closure_states = max_closure_states
        self.values: dict[State, float] = {}
        self.rejected_roots = 0
        self.solved_closures = 0

    # ── growth ────────────────────────────────────────────────────────────

    def add_state(self, root: State, epsilon: float = 1e-9) -> bool:
        """Solve *root*'s closure exactly and merge it. False if over budget."""
        if root in self.values:
            return True
        closure = compute_closure(root, self.config, self.max_closure_states)
        if closure.states is None:
            self.rejected_roots += 1
            return False
        solution = solve_closure(
            closure.states,
            self.config,
            gamma=self.gamma,
            epsilon=epsilon,
            known_values=self.values,
        )
        if not solution.converged or solution.max_bellman_residual > 1e-6:
            raise RuntimeError(
                f"closure VI did not converge cleanly: {solution.iterations} its, "
                f"delta={solution.max_delta:.2e}, "
                f"residual={solution.max_bellman_residual:.2e}"
            )
        self.values.update(solution.values)
        self.solved_closures += 1
        return True

    # ── probing ───────────────────────────────────────────────────────────

    def probe(self, state: State) -> float | None:
        return self.values.get(state)

    def __len__(self) -> int:
        return len(self.values)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        payload = {
            "gamma": self.gamma,
            "enable_mirror": self.config.enable_mirror,
            "enable_reversi": self.config.enable_reversi,
            "values": self.values,
            "saved_at": time.time(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: RulesConfig = RulesConfig(),
        max_closure_states: int = 100_000,
    ) -> "EndgameDB":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload["enable_mirror"] != config.enable_mirror or (
            payload["enable_reversi"] != config.enable_reversi
        ):
            raise ValueError("endgame DB was built for a different rules config")
        db = cls(config, gamma=payload["gamma"], max_closure_states=max_closure_states)
        db.values = payload["values"]
        return db

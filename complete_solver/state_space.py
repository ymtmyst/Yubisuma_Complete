"""Reachable-state enumeration and value iteration for the Complete solver."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from .choice_collapse import choice_row_groups, collapse_rows_by_sources
from .matrix_game import solve_zero_sum_matrix
from .state import State
from .transition import transition


@dataclass(frozen=True)
class StateSpaceStats:
    total_states: int
    terminal_states: int
    max_me_hands_seen: int
    max_opp_hands_seen: int

    def __str__(self) -> str:
        return (
            f"states={self.total_states} "
            f"terminals={self.terminal_states} "
            f"max_hands=({self.max_me_hands_seen}, {self.max_opp_hands_seen})"
        )


def enumerate_reachable_states(
    initial: State | None = None,
    config: RulesConfig = RulesConfig(),
    max_states: int | None = None,
) -> frozenset[State]:
    """BFS from *initial* to collect all reachable non-terminal states.

    Terminal states (where transition returns terminal_reward != None) are not
    included in the returned set because they have no further successors.

    *max_states* is a safety cap; enumeration stops once that many states are
    visited (useful for large configs during exploratory runs).
    """
    if initial is None:
        initial = State()

    visited: set[State] = {initial}
    queue: deque[State] = deque([initial])

    while queue:
        if max_states is not None and len(visited) >= max_states:
            break
        state = queue.popleft()
        for tp_action in legal_tp_actions(state, config):
            for ntp_action in legal_ntp_actions(state, config):
                result = transition(state, tp_action, ntp_action, config)
                if result.next_state is not None and result.next_state not in visited:
                    visited.add(result.next_state)
                    queue.append(result.next_state)

    return frozenset(visited)


def state_space_stats(
    states: frozenset[State],
    config: RulesConfig = RulesConfig(),
) -> StateSpaceStats:
    """Compute summary statistics over a set of enumerated states."""
    terminal_count = 0
    max_me = 0
    max_opp = 0

    for state in states:
        max_me = max(max_me, state.me.hands)
        max_opp = max(max_opp, state.opp.hands)
        for tp_action in legal_tp_actions(state, config):
            for ntp_action in legal_ntp_actions(state, config):
                result = transition(state, tp_action, ntp_action, config)
                if result.terminal_reward is not None:
                    terminal_count += 1
                    break
            else:
                continue
            break

    return StateSpaceStats(
        total_states=len(states),
        terminal_states=terminal_count,
        max_me_hands_seen=max_me,
        max_opp_hands_seen=max_opp,
    )


@dataclass(frozen=True)
class ValueIterationResult:
    values: dict[State, float]
    iterations: int
    max_delta: float
    converged: bool

    def __str__(self) -> str:
        status = "converged" if self.converged else "stopped (max_iter)"
        return (
            f"ValueIteration({status}, "
            f"iterations={self.iterations}, "
            f"max_delta={self.max_delta:.2e}, "
            f"states={len(self.values)})"
        )


def value_iteration(
    states: frozenset[State],
    config: RulesConfig = RulesConfig(),
    gamma: float = 0.999,
    epsilon: float = 1e-6,
    max_iterations: int = 1000,
    initial_values: dict[State, float] | None = None,
    leaf_evaluator=None,
) -> ValueIterationResult:
    """Discounted value iteration over a pre-enumerated set of states.

    Values are from the current turn player's perspective. States outside
    *states* are evaluated once with *leaf_evaluator* and treated as fixed
    constants (they are not updated during iteration). Terminal transitions use
    their reward directly.

    Transitions are pre-computed once to avoid redundant work across iterations.
    Returns a :class:`ValueIterationResult` with the converged value table.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 < gamma <= 1")

    leaf_fn = leaf_evaluator if leaf_evaluator is not None else lambda s: 0.0
    state_list = list(states)
    n = len(state_list)
    state_index: dict[State, int] = {s: i for i, s in enumerate(state_list)}

    # Pre-compute transition table.
    # Each entry is one of:
    #   (float, None)          → constant payoff (terminal or out-of-set leaf)
    #   (float, int)           → sign_factor * gamma * values[next_idx]
    #                            sign_factor stored, next_idx in state_list
    shapes: list[tuple[int, int]] = []
    tables: list[list[list[tuple]]] = []
    # CHOICE fix: row grouping depends only on the (fixed) action list, so it
    # is precomputed once per state (see choice_collapse.py).
    row_sources_list: list[list[tuple[int, ...]]] = []

    for state in state_list:
        tp_actions = legal_tp_actions(state, config)
        ntp_actions = legal_ntp_actions(state, config)
        _, row_sources = choice_row_groups(tp_actions)
        row_sources_list.append(row_sources)
        shapes.append((len(tp_actions), len(ntp_actions)))
        rows: list[list[tuple]] = []
        for tp_action in tp_actions:
            row: list[tuple] = []
            for ntp_action in ntp_actions:
                result = transition(state, tp_action, ntp_action, config)
                if result.terminal_reward is not None:
                    row.append((float(result.terminal_reward), None))
                else:
                    assert result.next_state is not None
                    nxt = result.next_state
                    sign = 1.0 if result.same_turn_player else -1.0
                    if nxt in state_index:
                        row.append((sign, state_index[nxt]))
                    else:
                        # Out-of-set: freeze at leaf value now (won't be updated)
                        leaf_payoff = gamma * sign * leaf_fn(nxt)
                        row.append((leaf_payoff, None))
            rows.append(row)
        tables.append(rows)

    # Initialize values.
    values = np.array(
        [initial_values[s] for s in state_list] if initial_values is not None
        else [leaf_fn(s) for s in state_list],
        dtype=float,
    )

    max_delta = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        new_values = np.empty(n, dtype=float)

        for i, ((n_tp, n_ntp), rows) in enumerate(zip(shapes, tables)):
            matrix = np.empty((n_tp, n_ntp), dtype=float)
            for j, row in enumerate(rows):
                for k, entry in enumerate(row):
                    sign_or_const, next_idx = entry
                    if next_idx is None:
                        matrix[j, k] = sign_or_const
                    else:
                        matrix[j, k] = gamma * sign_or_const * values[next_idx]
            matrix = collapse_rows_by_sources(matrix, row_sources_list[i])
            new_values[i] = solve_zero_sum_matrix(matrix).value

        max_delta = float(np.max(np.abs(new_values - values)))
        values = new_values
        if max_delta < epsilon:
            break

    return ValueIterationResult(
        values={state_list[i]: float(values[i]) for i in range(n)},
        iterations=iteration,
        max_delta=max_delta,
        converged=max_delta < epsilon,
    )

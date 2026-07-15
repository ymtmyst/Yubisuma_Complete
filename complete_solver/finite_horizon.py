"""Depth-limited exact subgame solving for Complete rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from .actions import NTPAction, RulesConfig, TPAction, legal_ntp_actions, legal_tp_actions
from .choice_collapse import collapse_choice_actions, resolve_choice_action
from .matrix_game import solve_zero_sum_matrix
from .state import State
from .transition import transition

LeafEvaluator = Callable[[State], float]


@dataclass(frozen=True)
class StatePolicy:
    value: float
    tp_actions: tuple[TPAction, ...]
    ntp_actions: tuple[NTPAction, ...]
    tp_policy: tuple[float, ...]
    ntp_policy: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]
    # thumb -> original (pre-collapse) TPActions a collapsed CHOICE row
    # replaced. Empty for states with 0/1 choosable stock (no collapse).
    choice_groups: dict[int, tuple[TPAction, ...]] = field(default_factory=dict)


class FiniteHorizonSolver:
    """Solve a depth-limited zero-sum subgame from any public state.

    Values are from the current turn player's perspective. On normal turn
    switch the next state's value is negated, while extra turns keep the sign.
    """

    def __init__(
        self,
        config: RulesConfig = RulesConfig(),
        gamma: float = 1.0,
        leaf_evaluator: LeafEvaluator | None = None,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma <= 1")
        self.config = config
        self.gamma = gamma
        self.leaf_evaluator = leaf_evaluator or material_leaf_evaluator

    def solve_state(self, state: State, depth: int) -> StatePolicy:
        if depth < 1:
            value = float(self.leaf_evaluator(state))
            return StatePolicy(value, (), (), (), (), (), {})

        matrix, tp_actions, ntp_actions, choice_groups = self.build_payoff_matrix(
            state, depth
        )
        solution = solve_zero_sum_matrix(matrix)
        return StatePolicy(
            value=solution.value,
            tp_actions=tp_actions,
            ntp_actions=ntp_actions,
            tp_policy=tuple(float(x) for x in solution.row_policy),
            ntp_policy=tuple(float(x) for x in solution.col_policy),
            matrix=tuple(tuple(float(x) for x in row) for row in matrix),
            choice_groups=choice_groups,
        )

    def value(self, state: State, depth: int) -> float:
        return self._value(state, depth)

    def _cell_payoff(
        self, state: State, tp_action: TPAction, ntp_action: NTPAction, depth: int
    ) -> float:
        result = transition(state, tp_action, ntp_action, self.config)
        if result.terminal_reward is not None:
            return result.terminal_reward
        assert result.next_state is not None
        if depth <= 1:
            payoff = self.gamma * self.leaf_evaluator(result.next_state)
        else:
            payoff = self.gamma * self._value(result.next_state, depth - 1)
        if not result.same_turn_player:
            payoff = -payoff
        return payoff

    def build_payoff_matrix(
        self,
        state: State,
        depth: int,
    ) -> tuple[
        np.ndarray,
        tuple[TPAction, ...],
        tuple[NTPAction, ...],
        dict[int, tuple[TPAction, ...]],
    ]:
        """Build the (collapsed) payoff matrix.

        CHOICE rows are pre-committed per legal_tp_actions (one row per
        stocked skill); this collapses same-thumb groups to their per-column
        max BEFORE the LP solve, implementing the post-reaction (second-
        mover) skill pick. See choice_collapse.py for the rule rationale.
        """
        tp_actions = legal_tp_actions(state, self.config)
        ntp_actions = legal_ntp_actions(state, self.config)
        raw = np.zeros((len(tp_actions), len(ntp_actions)), dtype=float)

        for row, tp_action in enumerate(tp_actions):
            for col, ntp_action in enumerate(ntp_actions):
                raw[row, col] = self._cell_payoff(state, tp_action, ntp_action, depth)

        collapsed_actions, matrix, choice_groups = collapse_choice_actions(
            tp_actions, raw
        )
        return matrix, collapsed_actions, ntp_actions, choice_groups

    def resolve_choice(
        self, state: State, thumb: int, ntp_action: NTPAction, depth: int
    ) -> TPAction:
        """Post-reaction skill pick: given the realized opponent reaction,
        return the concrete TPAction(CHOICE, thumb, choice=<best skill>) —
        the argmax the column-max collapse used for the matrix value."""
        return resolve_choice_action(
            state, thumb, ntp_action,
            lambda candidate, ntp: self._cell_payoff(state, candidate, ntp, depth),
        )

    @lru_cache(maxsize=None)
    def _value(self, state: State, depth: int) -> float:
        if depth <= 0:
            return float(self.leaf_evaluator(state))
        return self.solve_state(state, depth).value


def solve_state(
    state: State,
    depth: int,
    config: RulesConfig = RulesConfig(),
    gamma: float = 1.0,
    leaf_evaluator: LeafEvaluator | None = None,
) -> StatePolicy:
    return FiniteHorizonSolver(config, gamma, leaf_evaluator).solve_state(state, depth)


def material_leaf_evaluator(state: State) -> float:
    """Small bounded heuristic used only at a depth-limited frontier."""

    if state.me.hands <= 0 and state.opp.hands <= 0:
        return 0.0
    if state.me.hands <= 0:
        return 1.0 if state.me.has_declared_skill and state.opp.has_declared_skill else 0.0
    if state.opp.hands <= 0:
        return -1.0 if state.me.has_declared_skill and state.opp.has_declared_skill else 0.0

    hand_value = (state.opp.hands - state.me.hands) / 2.0
    buff_value = 0.0
    if state.me.guard_active:
        buff_value += 0.03
    if state.opp.guard_active:
        buff_value -= 0.03
    if state.me.charge_active:
        buff_value += 0.02
    if state.opp.charge_active:
        buff_value -= 0.02
    buff_value += 0.015 * (state.me.quick_level - state.opp.quick_level)
    buff_value += 0.01 * (len(state.me.stock) - len(state.opp.stock))
    if state.me.used_ultimate:
        buff_value -= 0.04
    if state.opp.used_ultimate:
        buff_value += 0.04
    return max(-1.0, min(1.0, hand_value + buff_value))

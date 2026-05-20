"""Depth-limited exact subgame solving for Complete rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from .actions import NTPAction, RulesConfig, TPAction, legal_ntp_actions, legal_tp_actions
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
            return StatePolicy(value, (), (), (), (), ())

        matrix, tp_actions, ntp_actions = self.build_payoff_matrix(state, depth)
        solution = solve_zero_sum_matrix(matrix)
        return StatePolicy(
            value=solution.value,
            tp_actions=tp_actions,
            ntp_actions=ntp_actions,
            tp_policy=tuple(float(x) for x in solution.row_policy),
            ntp_policy=tuple(float(x) for x in solution.col_policy),
            matrix=tuple(tuple(float(x) for x in row) for row in matrix),
        )

    def value(self, state: State, depth: int) -> float:
        return self._value(state, depth)

    def build_payoff_matrix(
        self,
        state: State,
        depth: int,
    ) -> tuple[np.ndarray, tuple[TPAction, ...], tuple[NTPAction, ...]]:
        tp_actions = legal_tp_actions(state, self.config)
        ntp_actions = legal_ntp_actions(state, self.config)
        matrix = np.zeros((len(tp_actions), len(ntp_actions)), dtype=float)

        for row, tp_action in enumerate(tp_actions):
            for col, ntp_action in enumerate(ntp_actions):
                result = transition(state, tp_action, ntp_action, self.config)
                if result.terminal_reward is not None:
                    payoff = result.terminal_reward
                elif depth <= 1:
                    assert result.next_state is not None
                    leaf_value = self.leaf_evaluator(result.next_state)
                    payoff = self.gamma * leaf_value
                    if not result.same_turn_player:
                        payoff = -payoff
                else:
                    assert result.next_state is not None
                    next_value = self._value(result.next_state, depth - 1)
                    payoff = self.gamma * next_value
                    if not result.same_turn_player:
                        payoff = -payoff
                matrix[row, col] = payoff

        return matrix, tp_actions, ntp_actions

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

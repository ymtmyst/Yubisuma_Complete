"""Nash-optimal NTP policy derived from the exact value iteration solver.

Computes Nash mixed strategies for NTP at every reachable state by:
1. Running value iteration to get V*(s) for all reachable states.
2. For each state, building the payoff matrix payoff[tp_i, ntp_j].
3. Solving the zero-sum LP to extract the NTP column player's mixed strategy.

The resulting policy is the best possible opponent for TP: it plays the exact
Nash NTP strategy, forcing TP to learn the Nash TP strategy when trained
against it.
"""

from __future__ import annotations

import numpy as np

from complete_solver import RulesConfig, State
from complete_solver.actions import NTPAction, legal_ntp_actions, legal_tp_actions
from complete_solver.finite_horizon import material_leaf_evaluator
from complete_solver.matrix_game import solve_zero_sum_matrix
from complete_solver.state_space import enumerate_reachable_states, value_iteration
from complete_solver.transition import transition


def compute_nash_ntp_strategies(
    config: RulesConfig = RulesConfig(),
    *,
    max_states: int = 500,
    gamma: float = 0.999,
    vi_epsilon: float = 1e-4,
    vi_max_iter: int = 500,
) -> dict[State, tuple[tuple[NTPAction, ...], tuple[float, ...]]]:
    """Compute Nash NTP mixed strategies for all reachable states.

    Returns a dict mapping each reachable state to a pair
    ``(ntp_actions, ntp_probs)`` where ``ntp_probs`` sums to 1.
    States where no NTP action is legal are omitted.
    """
    states = enumerate_reachable_states(config=config, max_states=max_states)
    vi = value_iteration(
        states,
        config=config,
        gamma=gamma,
        epsilon=vi_epsilon,
        max_iterations=vi_max_iter,
        leaf_evaluator=material_leaf_evaluator,
    )
    V = vi.values

    result: dict[State, tuple[tuple[NTPAction, ...], tuple[float, ...]]] = {}

    for state in states:
        tp_acts = legal_tp_actions(state, config)
        ntp_acts = legal_ntp_actions(state, config)

        if not tp_acts or not ntp_acts:
            continue

        n_tp = len(tp_acts)
        n_ntp = len(ntp_acts)
        matrix = np.zeros((n_tp, n_ntp), dtype=float)

        for i, tp_a in enumerate(tp_acts):
            for j, ntp_a in enumerate(ntp_acts):
                res = transition(state, tp_a, ntp_a, config)
                if res.terminal_reward is not None:
                    payoff = float(res.terminal_reward)
                else:
                    nxt = res.next_state
                    assert nxt is not None
                    nxt_v = V.get(nxt, float(material_leaf_evaluator(nxt)))
                    sign = 1.0 if res.same_turn_player else -1.0
                    payoff = gamma * sign * nxt_v
                matrix[i, j] = payoff

        try:
            sol = solve_zero_sum_matrix(matrix)
            probs = sol.col_policy
        except RuntimeError:
            probs = np.full(n_ntp, 1.0 / n_ntp)

        result[state] = (tuple(ntp_acts), tuple(float(p) for p in probs))

    return result

"""Pure solver-facing rules for Complete Yubisuma."""

from .actions import NTPAction, RulesConfig, TPAction, legal_ntp_actions, legal_tp_actions
from .finite_horizon import FiniteHorizonSolver, StatePolicy, solve_state
from .state import PlayerState, State, initial_state
from .state_space import (
    StateSpaceStats,
    ValueIterationResult,
    enumerate_reachable_states,
    state_space_stats,
    value_iteration,
)
from .transition import Transition, transition

__all__ = [
    "FiniteHorizonSolver",
    "NTPAction",
    "PlayerState",
    "RulesConfig",
    "State",
    "StatePolicy",
    "StateSpaceStats",
    "TPAction",
    "Transition",
    "ValueIterationResult",
    "enumerate_reachable_states",
    "initial_state",
    "legal_ntp_actions",
    "legal_tp_actions",
    "solve_state",
    "state_space_stats",
    "transition",
    "value_iteration",
]

"""Stock-alphabet abstraction for the endgame database (N2b).

The endgame state space is dominated by stock combinatorics, but most stocked
skills barely matter at (1,1) hands. We therefore solve the endgame universe
of a RESTRICTED game where only an "alphabet" of high-impact skills may be
stocked (declaring skills is never restricted), and then MEASURE how much
value anyone could gain by deviating into an excluded stock — instead of
assuming the alphabet is right.

Pipeline:
1. ``build_h11_db``    — enumerate + exactly solve the restricted (1,1)
                         universe (closure BFS + Shapley VI from endgame_db).
2. ``project_state``   — map a full-game state onto the abstraction (drop
                         excluded skills from stocks) for probing.
3. ``deviation_scan``  — for DB states where the full game would allow
                         stocking an excluded skill, compute the gain of that
                         deviation against the abstract equilibrium reaction
                         mixture, valuing the off-abstraction child with a
                         depth-limited full-game search whose leaves probe the
                         projected DB. Reports per-skill gain statistics.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from .actions import RulesConfig, TPAction, legal_ntp_actions, legal_tp_actions
from .constants import (
    FEINT,
    FLASH,
    GUARD,
    LOCK,
    MIRROR_PREP,
    PASS,
    REFERENCEABLE_SKILLS,
    SKIP,
    STOCK,
)
from .choice_collapse import collapse_choice_actions
from .endgame_db import EndgameDB, compute_closure, solve_closure
from .fast_solver import FastHorizonSolver
from .finite_horizon import material_leaf_evaluator
from .small_matrix import solve_small_zero_sum
from .state import PlayerState, State
from .transition import transition

DEFAULT_STOCK_ALPHABET: frozenset[str] = frozenset({FEINT, LOCK, FLASH, GUARD, SKIP})


def h11_root() -> State:
    """Canonical generator of the (1,1) endgame universe."""
    return State(
        me=PlayerState(hands=1, has_declared_skill=True),
        opp=PlayerState(hands=1, has_declared_skill=True),
    )


def abstract_config(
    alphabet: frozenset[str] = DEFAULT_STOCK_ALPHABET,
) -> RulesConfig:
    return RulesConfig(
        enable_mirror=False, enable_reversi=False, stock_alphabet=alphabet
    )


# ── projection ─────────────────────────────────────────────────────────────


def project_player(player: PlayerState, alphabet: frozenset[str]) -> PlayerState | None:
    """Project one player's state onto the abstraction, or None if unsafe.

    Stocks and choice bookkeeping are intersected with the alphabet.
    ``drop_blocked_skills`` outside the alphabet cannot be represented (the
    abstract game never blocks them), so such transient states are not probed.
    """
    if not player.drop_blocked_skills <= alphabet:
        return None
    if player.stock <= alphabet and player.choice_used_this_phase <= alphabet:
        return player
    return player._replace(
        stock=player.stock & alphabet,
        choice_used_this_phase=player.choice_used_this_phase & alphabet,
    )


def project_state(state: State, alphabet: frozenset[str]) -> State | None:
    me = project_player(state.me, alphabet)
    opp = project_player(state.opp, alphabet)
    if me is None or opp is None:
        return None
    if me is state.me and opp is state.opp:
        return state
    return state._replace(me=me, opp=opp)


# ── database construction ──────────────────────────────────────────────────


def build_h11_db(
    alphabet: frozenset[str] = DEFAULT_STOCK_ALPHABET,
    gamma: float = 0.999,
    max_states: int = 800_000,
    epsilon: float = 1e-9,
    verbose: bool = True,
) -> tuple[EndgameDB, dict]:
    """Solve the restricted (1,1) universe exactly. Returns (db, build info)."""
    config = abstract_config(alphabet)
    t0 = time.perf_counter()
    closure = compute_closure(h11_root(), config, max_states=max_states)
    enum_seconds = time.perf_counter() - t0
    if closure.states is None:
        raise RuntimeError(
            f"restricted (1,1) universe exceeds {max_states} states — "
            "shrink the alphabet or raise the budget"
        )

    t0 = time.perf_counter()
    solution = solve_closure(closure.states, config, gamma=gamma, epsilon=epsilon)
    solve_seconds = time.perf_counter() - t0
    if not solution.converged or solution.max_bellman_residual > 1e-6:
        raise RuntimeError(
            f"closure VI did not converge cleanly: {solution.iterations} its, "
            f"delta={solution.max_delta:.2e}, "
            f"residual={solution.max_bellman_residual:.2e}"
        )

    db = EndgameDB(config, gamma=gamma, max_closure_states=max_states)
    db.values = solution.values
    db.solved_closures = 1
    info = {
        "alphabet": sorted(alphabet),
        "states": len(closure.states),
        "enum_seconds": enum_seconds,
        "solve_seconds": solve_seconds,
        "iterations": solution.iterations,
        "max_delta": solution.max_delta,
        "max_bellman_residual": solution.max_bellman_residual,
    }
    if verbose:
        print(f"h11 universe: {info}", flush=True)
    return db, info


# ── deviation scan ─────────────────────────────────────────────────────────


@dataclass
class DeviationStats:
    skill: str
    states_scanned: int = 0
    max_gain: float = -np.inf
    mean_gain: float = 0.0
    positive_gains: int = 0          # gain > 1e-6
    significant_gains: int = 0       # gain > 0.01
    worst_state: State | None = None

    def as_dict(self) -> dict:
        return {
            "skill": self.skill,
            "states_scanned": self.states_scanned,
            "max_gain": None if self.states_scanned == 0 else float(self.max_gain),
            "mean_gain": float(self.mean_gain),
            "positive_gains": self.positive_gains,
            "significant_gains": self.significant_gains,
        }


def deviation_scan(
    db: EndgameDB,
    alphabet: frozenset[str] = DEFAULT_STOCK_ALPHABET,
    search_depth: int = 3,
    max_states_per_skill: int = 400,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, DeviationStats]:
    """Measure the value of stocking each EXCLUDED skill inside the endgame.

    For every sampled DB state whose ``previous_skill`` is an excluded (but in
    the full game stockable) skill, the deviator plays STOCK against the
    abstract equilibrium reaction mixture. The resulting off-abstraction child
    is valued by a depth-limited FULL-game search whose leaves probe the
    projected database. ``gain = deviation payoff − abstract state value``.
    A positive gain means the abstraction loses real value at that state.
    """
    full_config = RulesConfig(enable_mirror=False, enable_reversi=False)
    excluded = (REFERENCEABLE_SKILLS - {MIRROR_PREP}) - alphabet
    gamma = db.gamma
    rng = random.Random(seed)

    def leaf(state: State) -> float:
        projected = project_state(state, alphabet)
        if projected is not None:
            value = db.values.get(projected)
            if value is not None:
                return value
        return material_leaf_evaluator(state)

    searcher = FastHorizonSolver(full_config, gamma=gamma, leaf_evaluator=leaf)

    # Group candidate states by the excluded skill they could stock.
    candidates: dict[str, list[State]] = {skill: [] for skill in excluded}
    for state in db.values:
        previous = state.previous_skill
        if (
            isinstance(previous, str)
            and previous in excluded
            and previous not in state.me.stock
            and state.me.skip_phases == 0
        ):
            candidates[previous].append(state)

    results: dict[str, DeviationStats] = {}
    for skill in sorted(excluded):
        stats = DeviationStats(skill=skill)
        pool = candidates[skill]
        if len(pool) > max_states_per_skill:
            pool = rng.sample(pool, max_states_per_skill)
        gain_sum = 0.0
        for state in pool:
            abstract_value = db.values[state]

            # Abstract equilibrium reaction mixture at this state.
            abstract_cfg = db.config
            tp_actions = legal_tp_actions(state, abstract_cfg)
            ntp_actions = legal_ntp_actions(state, abstract_cfg)
            matrix = np.empty((len(tp_actions), len(ntp_actions)))
            for r, tp_action in enumerate(tp_actions):
                for c, ntp_action in enumerate(ntp_actions):
                    result = transition(state, tp_action, ntp_action, abstract_cfg)
                    if result.terminal_reward is not None:
                        matrix[r, c] = result.terminal_reward
                    else:
                        sign = 1.0 if result.same_turn_player else -1.0
                        matrix[r, c] = sign * gamma * db.values[result.next_state]
            # CHOICE fix: collapse before solving (only the column/NTP
            # mixture is used below, so the row collapse is transparent to
            # the caller — see choice_collapse.py).
            _, collapsed_matrix, _ = collapse_choice_actions(tp_actions, matrix)
            _, _, ntp_mixture = solve_small_zero_sum(collapsed_matrix)

            # Deviation: STOCK the excluded skill (thumb choice: best gain).
            thumbs = sorted({action.thumb for action in tp_actions})
            best_payoff = -np.inf
            for thumb in thumbs:
                deviation = TPAction(STOCK, thumb)
                payoff = 0.0
                for c, ntp_action in enumerate(ntp_actions):
                    weight = ntp_mixture[c]
                    if weight <= 1e-12:
                        continue
                    result = transition(state, deviation, ntp_action, full_config)
                    if result.terminal_reward is not None:
                        child_value = result.terminal_reward
                    else:
                        sign = 1.0 if result.same_turn_player else -1.0
                        child_value = sign * gamma * searcher.value(
                            result.next_state, search_depth
                        )
                    payoff += weight * child_value
                best_payoff = max(best_payoff, payoff)

            gain = best_payoff - abstract_value
            stats.states_scanned += 1
            gain_sum += gain
            if gain > stats.max_gain:
                stats.max_gain = gain
                stats.worst_state = state
            if gain > 1e-6:
                stats.positive_gains += 1
            if gain > 0.01:
                stats.significant_gains += 1
        if stats.states_scanned:
            stats.mean_gain = gain_sum / stats.states_scanned
        results[skill] = stats
        if verbose:
            print(f"deviation[{skill}]: {stats.as_dict()}", flush=True)
    return results

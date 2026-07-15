"""Shared CHOICE-row collapse for the post-reaction (second-mover) fix.

RULE FIX (2026-07-15): CHOICE (チョイス) is a single declaration — thumb only.
The opponent's reaction is revealed, and only THEN does the mover pick WHICH
stocked skill fires. The engine previously pre-committed the fired skill at
declaration time (one TPAction per stocked skill, ``choice=<skill>``), which
lets the reactor punish a skill it hasn't seen yet. That is wrong.

Fix (column-max collapse, verified game-theoretically correct): in every
per-state zero-sum matrix (mover TP rows x opponent NTP cols), group the
CHOICE rows (multiple stocked skills, same thumb) and replace each group with
ONE row equal to the per-column (per-reaction) MAXIMUM of the group. Solving
the LP on the reduced matrix makes CHOICE's value = "mover picks the best
stocked skill after seeing the reaction." Thumb stays committed (rows are
grouped BY thumb); only the fired skill's identity becomes post-reaction.

States with 0 or 1 choosable stock have at most one CHOICE row per thumb, so
the collapse is a mathematical no-op for them (a group of size 1 is returned
unchanged) — this is what keeps existing single/no-stock tests byte-identical.

This module provides the grouping/collapse logic in the two representations
used across the codebase:
  * TPAction-object based (the reference solvers: finite_horizon.py,
    fast_solver.py).
  * packed int-code based (the compiled engines: packed_engine/packed_vi/
    subgraph_vi/batched_search/selective_search). Packed TP codes encode
    CHOICE as ``128 + choice_id*4 + thumb`` (see packed_engine.py), so
    ``code & 3`` is always the thumb and ``code >= 128`` always means CHOICE.

``CHOICE_META_BASE`` (packed side) marks a *collapsed* pseudo-row: it stands
for "declare CHOICE with this thumb, skill TBD until the reaction is seen."
It is never a legal input to ``packed_engine.step`` — callers must resolve it
to a concrete code (see ``resolve_choice_code`` and the reference solvers'
``resolve_choice``) once the opponent's realized NTP action is known, before
executing the actual game step.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .actions import TPAction
from .constants import CHOICE

# Packed TP codes: 0..63 numbers, 64..127 skills, 128..159 CHOICE (8 choice
# ids x 4 thumbs), 160..191 targeted-STOCK (YS_STOCK_FREECHOICE only; see
# packed_engine.STOCK_TARGET_BASE), 255 PASS. 192..195 is unused by the real
# engine and reserved here for "collapsed CHOICE, thumb=code-192" pseudo-rows.
#
# IMPORTANT: the CHOICE-code range check below is deliberately bounded
# (CHOICE_CODE_BASE <= code < CHOICE_CODE_END), NOT a loose "code >= 128"
# — codes 160..191 are a real, unrelated action (targeted STOCK) and must
# never be swept into a CHOICE collapse group.
CHOICE_CODE_BASE = 128
CHOICE_CODE_END = 160
CHOICE_META_BASE = 192


def is_choice_meta_code(code: int) -> bool:
    return CHOICE_META_BASE <= code < CHOICE_META_BASE + 4


def choice_meta_code(thumb: int) -> int:
    return CHOICE_META_BASE + thumb


def choice_meta_thumb(code: int) -> int:
    return code - CHOICE_META_BASE


# ── TPAction-object representation (reference solvers) ─────────────────────


def choice_row_groups(
    tp_actions: tuple[TPAction, ...],
) -> tuple[tuple[TPAction, ...], list[tuple[int, ...]]]:
    """Group CHOICE rows by thumb; collapse groups of size >= 2.

    Returns ``(collapsed_actions, row_sources)``: ``row_sources[k]`` is the
    tuple of indices into the ORIGINAL ``tp_actions`` that collapsed row ``k``
    aggregates (max over them). Pass-through rows (including CHOICE groups of
    size < 2, i.e. 0 or 1 choosable stock) have a 1-tuple source — a
    mathematical no-op, matching pre-fix behaviour exactly. A collapsed row's
    action is a pseudo ``TPAction(CHOICE, thumb, choice=None)`` — never a real
    action; it must be resolved (see ``resolve_choice_action``) before it can
    be executed via ``transition``.
    """
    by_thumb: dict[int, list[int]] = {}
    for i, action in enumerate(tp_actions):
        if action.skill == CHOICE:
            by_thumb.setdefault(action.thumb, []).append(i)
    groups = {thumb: idxs for thumb, idxs in by_thumb.items() if len(idxs) >= 2}
    if not groups:
        return tuple(tp_actions), [(i,) for i in range(len(tp_actions))]

    grouped_idx = {i for idxs in groups.values() for i in idxs}
    collapsed: list[TPAction] = []
    sources: list[tuple[int, ...]] = []
    for thumb in sorted(groups):
        collapsed.append(TPAction(CHOICE, thumb, choice=None))
        sources.append(tuple(groups[thumb]))
    for i, action in enumerate(tp_actions):
        if i in grouped_idx:
            continue
        collapsed.append(action)
        sources.append((i,))
    return tuple(collapsed), sources


def collapse_rows_by_sources(
    matrix: np.ndarray, row_sources: list[tuple[int, ...]]
) -> np.ndarray:
    """Apply a precomputed ``row_sources`` grouping (from ``choice_row_groups``,
    which only depends on the action LIST, not payoffs) to a fresh payoff
    matrix — the per-iteration collapse step for VI loops that rebuild the
    matrix every sweep but whose action set (and hence grouping) is fixed."""
    if all(len(sources) == 1 for sources in row_sources):
        return matrix
    return np.array(
        [
            matrix[sources[0], :] if len(sources) == 1
            else matrix[list(sources), :].max(axis=0)
            for sources in row_sources
        ]
    )


def collapse_choice_actions(
    tp_actions: tuple[TPAction, ...], matrix: np.ndarray
) -> tuple[tuple[TPAction, ...], np.ndarray, dict[int, tuple[TPAction, ...]]]:
    """Collapse an already-built (rows=tp_actions) matrix.

    Returns ``(collapsed_actions, collapsed_matrix, groups)`` where ``groups``
    maps a collapsed row's thumb -> the original TPActions it replaced (for
    post-reaction resolution via ``resolve_choice_action``).
    """
    collapsed_actions, row_sources = choice_row_groups(tp_actions)
    n_cols = matrix.shape[1]
    out = np.empty((len(collapsed_actions), n_cols), dtype=matrix.dtype)
    groups: dict[int, tuple[TPAction, ...]] = {}
    for k, sources in enumerate(row_sources):
        if len(sources) == 1:
            out[k, :] = matrix[sources[0], :]
        else:
            out[k, :] = matrix[list(sources), :].max(axis=0)
            groups[collapsed_actions[k].thumb] = tuple(tp_actions[i] for i in sources)
    return collapsed_actions, out, groups


def resolve_choice_action(
    state,
    thumb: int,
    ntp_action,
    payoff_fn,
) -> TPAction:
    """Pick the concrete stocked skill to fire for a CHOICE(thumb) declaration
    now that the opponent's reaction ``ntp_action`` is known (the post-reaction
    / second-mover choice this whole module exists to implement).

    ``payoff_fn(candidate_tp_action, ntp_action) -> float`` must be the SAME
    per-cell payoff the caller's matrix build uses (so the pick is exactly the
    argmax the column-max collapse used for the value).
    """
    available = sorted(state.me.stock - state.me.choice_used_this_phase)
    if not available:
        raise ValueError("resolve_choice_action: no choosable stock")
    best_action = None
    best_value = -float("inf")
    for skill in available:
        candidate = TPAction(CHOICE, thumb, choice=skill)
        value = payoff_fn(candidate, ntp_action)
        if value > best_value:
            best_value = value
            best_action = candidate
    assert best_action is not None
    return best_action


# ── packed int-code representation (compiled engines) ──────────────────────


def collapse_choice_matrix_packed(
    tp_codes: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, ...]]]:
    """Packed-code twin of ``collapse_choice_actions``.

    ``tp_codes``: 1-D int array of packed TP codes (see packed_engine.py),
    rows of ``matrix``. Groups codes in ``[CHOICE_CODE_BASE, CHOICE_CODE_END)``
    (CHOICE) by ``code & 3`` (thumb); groups of size >= 2 collapse to one row =
    the per-column max, represented by the pseudo code
    ``CHOICE_META_BASE + thumb``. Groups of size 0 or 1 (single/no choosable
    stock) pass through unchanged — a mathematical no-op. Codes outside this
    range (e.g. targeted-STOCK codes under YS_STOCK_FREECHOICE) always pass
    through untouched, never grouped with CHOICE.

    Returns ``(new_codes, new_matrix, groups)`` where ``groups`` maps the meta
    code -> the tuple of original concrete choice codes it replaced.
    """
    n_rows, n_cols = matrix.shape
    by_thumb: dict[int, list[int]] = {}
    for i in range(n_rows):
        code = int(tp_codes[i])
        if CHOICE_CODE_BASE <= code < CHOICE_CODE_END:
            by_thumb.setdefault(code & 3, []).append(i)
    collapse_rows = {t: idxs for t, idxs in by_thumb.items() if len(idxs) >= 2}
    if not collapse_rows:
        return np.asarray(tp_codes), matrix, {}

    collapsed_idx = {i for idxs in collapse_rows.values() for i in idxs}
    new_codes: list[int] = []
    new_rows: list[np.ndarray] = []
    groups: dict[int, tuple[int, ...]] = {}
    for thumb in sorted(collapse_rows):
        idxs = collapse_rows[thumb]
        new_codes.append(choice_meta_code(thumb))
        new_rows.append(matrix[idxs, :].max(axis=0))
        groups[choice_meta_code(thumb)] = tuple(int(tp_codes[i]) for i in idxs)
    for i in range(n_rows):
        if i in collapsed_idx:
            continue
        new_codes.append(int(tp_codes[i]))
        new_rows.append(matrix[i, :])

    return (
        np.array(new_codes, dtype=np.int64),
        np.array(new_rows, dtype=matrix.dtype),
        groups,
    )


def resolve_choice_code(
    lane0: int,
    meta_code: int,
    groups: dict[int, tuple[int, ...]],
    column_payoffs,
) -> int:
    """Pick the concrete choice code for a collapsed meta row, given the
    per-candidate payoff callable ``column_payoffs(code) -> float`` evaluated
    at the realized opponent column. ``groups`` must be the mapping returned
    by ``collapse_choice_matrix_packed`` for this exact state/thumb."""
    candidates = groups[meta_code]
    best_code, best_value = candidates[0], -float("inf")
    for code in candidates:
        value = column_payoffs(code)
        if value > best_value:
            best_value = value
            best_code = code
    return best_code


@njit(cache=True, inline="always")
def collapse_choice_rows_njit(codes, matrix, n_rows, n_cols, out_codes, out_matrix):
    """Njit twin of ``collapse_choice_matrix_packed`` for hot-loop matrix
    builds (packed_vi's Jacobi sweep, batched_search's backup chain).

    ``codes``/``matrix`` are read-only over ``[:n_rows]``/``[:n_rows,:n_cols]``.
    ``out_codes``/``out_matrix`` are caller-owned scratch buffers (reused
    across calls) sized for at least ``n_rows`` rows. Returns the new row
    count. Value-equivalent to the Python version; groups of size < 2 (0 or 1
    choosable stock) pass through with their ORIGINAL code untouched — a
    mathematical no-op, matching every other call site exactly. Only codes in
    ``[128, 160)`` (real CHOICE codes) are eligible for grouping — codes
    ``[160, 192)`` (targeted-STOCK, YS_STOCK_FREECHOICE) always pass through
    untouched, never grouped with CHOICE."""
    counts = np.zeros(4, dtype=np.int64)
    for i in range(n_rows):
        code = codes[i]
        if code >= CHOICE_CODE_BASE and code < CHOICE_CODE_END:
            counts[code & 3] += 1

    slot = np.full(4, -1, dtype=np.int64)
    n_out = 0
    for i in range(n_rows):
        code = codes[i]
        if code >= CHOICE_CODE_BASE and code < CHOICE_CODE_END and counts[code & 3] >= 2:
            thumb = code & 3
            s = slot[thumb]
            if s == -1:
                s = n_out
                slot[thumb] = s
                out_codes[s] = 192 + thumb
                for j in range(n_cols):
                    out_matrix[s, j] = matrix[i, j]
                n_out += 1
            else:
                for j in range(n_cols):
                    v = matrix[i, j]
                    if v > out_matrix[s, j]:
                        out_matrix[s, j] = v
        else:
            out_codes[n_out] = code
            for j in range(n_cols):
                out_matrix[n_out, j] = matrix[i, j]
            n_out += 1
    return n_out

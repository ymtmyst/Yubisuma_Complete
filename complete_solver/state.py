"""Immutable state representation for solver-facing Complete rules.

``PlayerState`` and ``State`` are ``NamedTuple``s: construction, hashing and
equality run at C speed, which matters because search creates hundreds of
thousands of states per second. Field order is part of the public contract —
never reorder, only append.
"""

from __future__ import annotations

from typing import NamedTuple, Union

from .constants import MIRROR_PREP

SkillRef = Union[int, str]

_EMPTY: frozenset[str] = frozenset()


class PlayerState(NamedTuple):
    """State for one player from the canonical current-player perspective."""

    hands: int = 2
    cement: int = 0

    guard_active: bool = False
    charge_active: bool = False
    quick_level: int = 0
    mirror_ready: bool = False

    lock_pending: bool = False
    lock_active: bool = False
    skip_phases: int = 0
    drop_blocked_skills: frozenset[str] = _EMPTY

    used_ultimate: bool = False
    stock: frozenset[str] = _EMPTY
    stock_alpha_used_this_phase: bool = False
    choice_used_this_phase: frozenset[str] = _EMPTY
    time_active: bool = False

    has_declared_skill: bool = False

    def stock_without_disabled_rules(self, enable_mirror: bool) -> frozenset[str]:
        if enable_mirror:
            return self.stock
        return frozenset(skill for skill in self.stock if skill != MIRROR_PREP)


_DEFAULT_PLAYER = PlayerState()


class State(NamedTuple):
    """Canonical public state.

    Values and transitions are from the current turn player's perspective:
    `me` is the turn player and `opp` is the non-turn player.
    """

    me: PlayerState = _DEFAULT_PLAYER
    opp: PlayerState = _DEFAULT_PLAYER
    previous_skill: SkillRef | None = None
    me_extra_turns: int = 0
    opp_extra_turns: int = 0
    me_guard_extra_used_this_phase: bool = False
    opp_guard_extra_used_this_phase: bool = False


def initial_state() -> State:
    return State()

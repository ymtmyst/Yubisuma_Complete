"""Legal complete-action generation for Complete rules."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    ALL,
    ANTI_COUNTER_SKILLS,
    BLOCK,
    BOOST,
    CHOICE,
    COPY,
    COUNTER,
    DROP,
    MIRROR_MAIN,
    MIRROR_PREP,
    NONE,
    NORMAL_SKILLS,
    PASS,
    REFERENCE_SKILLS,
    REFERENCEABLE_SKILLS,
    REVERSI,
    STOCK,
    STOCK_ALPHA_SKILLS,
    ULTIMATE_TP_SKILLS,
)
from .state import SkillRef, State


@dataclass(frozen=True)
class RulesConfig:
    enable_mirror: bool = False
    enable_reversi: bool = False


@dataclass(frozen=True, order=True)
class TPAction:
    skill: SkillRef
    thumb: int = 0
    choice: str | None = None
    all_order: tuple[str, ...] = ()

    def key(self) -> str:
        if isinstance(self.skill, int):
            return f"NUM(total={self.skill},thumb={self.thumb})"
        if self.skill == CHOICE:
            return f"CHOICE({self.choice},thumb={self.thumb})"
        if self.skill == ALL:
            return f"ALL(order={self.all_order},thumb={self.thumb})"
        return f"{self.skill}(thumb={self.thumb})"


@dataclass(frozen=True, order=True)
class NTPAction:
    reaction: str = NONE
    thumb: int = 0

    def key(self) -> str:
        return f"NTP({self.reaction},thumb={self.thumb})"


def legal_thumb_values(hands: int, cement: int) -> range:
    lower = min(max(cement, 0), max(hands, 0))
    return range(lower, max(hands, 0) + 1)


def legal_tp_actions(state: State, config: RulesConfig = RulesConfig()) -> tuple[TPAction, ...]:
    if state.me.skip_phases > 0:
        return (TPAction(PASS, 0),)

    thumbs = tuple(legal_thumb_values(state.me.hands, state.me.cement))
    total_max = state.me.hands + state.opp.hands
    actions: list[TPAction] = []

    for thumb in thumbs:
        for total in range(total_max + 1):
            actions.append(TPAction(total, thumb))

    skills = set(NORMAL_SKILLS | ANTI_COUNTER_SKILLS | REFERENCE_SKILLS | ULTIMATE_TP_SKILLS)
    if not config.enable_mirror:
        skills.discard(MIRROR_PREP)
    if not config.enable_reversi:
        skills.discard(REVERSI)
    if state.me.used_ultimate:
        skills.difference_update(ULTIMATE_TP_SKILLS)

    skills.difference_update(state.me.drop_blocked_skills)

    previous = state.previous_skill
    for skill in sorted(skills):
        if skill == COPY:
            if previous is None:
                continue
            if isinstance(previous, str) and previous not in REFERENCEABLE_SKILLS:
                continue
        elif skill == STOCK:
            if not isinstance(previous, str) or previous not in _stockable_skills(config):
                continue
            if previous in state.me.stock:
                continue
        elif skill in STOCK_ALPHA_SKILLS and state.me.stock_alpha_used_this_phase:
            continue
        elif skill == CHOICE:
            available = sorted(state.me.stock - state.me.choice_used_this_phase)
            for choice in available:
                for thumb in thumbs:
                    actions.append(TPAction(skill, thumb, choice=choice))
            continue
        elif skill == ALL:
            if not state.me.stock:
                continue
            for thumb in thumbs:
                actions.append(TPAction(skill, thumb, all_order=tuple(sorted(state.me.stock))))
            continue
        elif skill == DROP and not state.me.stock:
            continue

        for thumb in thumbs:
            actions.append(TPAction(skill, thumb))

    return tuple(sorted(set(actions), key=lambda action: action.key()))


def legal_ntp_actions(state: State, config: RulesConfig = RulesConfig()) -> tuple[NTPAction, ...]:
    thumbs = tuple(legal_thumb_values(state.opp.hands, state.opp.cement))
    lock_blocks_reactions = state.opp.lock_active or state.opp.lock_pending
    reactions = [NONE]

    if not lock_blocks_reactions:
        reactions.append(COUNTER)
        if not state.opp.used_ultimate:
            reactions.append(BLOCK)
        if config.enable_mirror and state.opp.mirror_ready:
            reactions.append(MIRROR_MAIN)

    return tuple(sorted(NTPAction(reaction, thumb) for reaction in reactions for thumb in thumbs))


def _stockable_skills(config: RulesConfig) -> frozenset[str]:
    if config.enable_mirror:
        return REFERENCEABLE_SKILLS
    return REFERENCEABLE_SKILLS - {MIRROR_PREP}

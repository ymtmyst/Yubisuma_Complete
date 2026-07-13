"""Pure one-turn transition for solver-facing Complete rules."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import NTPAction, RulesConfig, TPAction
from .constants import (
    ALL,
    ANTI_COUNTER_SKILLS,
    BLOCK,
    BOOST,
    CEMENT,
    CHARGE,
    CHOICE,
    COPY,
    COUNTER,
    DROP,
    FEINT,
    FLASH,
    GUARD,
    LOCK,
    MIRROR_MAIN,
    MIRROR_PREP,
    NONE,
    PASS,
    QUICK,
    REFERENCEABLE_SKILLS,
    REVERSI,
    SKIP,
    STOCK,
    STOCK_ALPHA_SKILLS,
    TIME,
    ULTIMATE_TP_SKILLS,
)
from .state import PlayerState, SkillRef, State


@dataclass(frozen=True)
class Transition:
    next_state: State | None
    terminal_reward: float | None
    same_turn_player: bool
    events: tuple[str, ...] = ()


@dataclass(slots=True)
class _Work:
    me: PlayerState
    opp: PlayerState
    previous_skill: SkillRef | None
    me_extra_turns: int
    opp_extra_turns: int
    me_guard_extra_used: bool
    opp_guard_extra_used: bool
    me_blocked: bool = False
    opp_blocked: bool = False
    added_extra_turns: int = 0
    events: list[str] | None = None

    def event(self, value: str) -> None:
        assert self.events is not None
        self.events.append(value)


def transition(
    state: State,
    tp_action: TPAction,
    ntp_action: NTPAction,
    config: RulesConfig = RulesConfig(),
) -> Transition:
    """Resolve one simultaneous turn from the current player's perspective."""

    work = _Work(
        me=state.me,
        opp=state.opp,
        previous_skill=state.previous_skill,
        me_extra_turns=state.me_extra_turns,
        opp_extra_turns=state.opp_extra_turns,
        me_guard_extra_used=state.me_guard_extra_used_this_phase,
        opp_guard_extra_used=state.opp_guard_extra_used_this_phase,
        events=[],
    )

    # Lock becomes active for the non-turn player at the start of this turn.
    if work.opp.lock_pending:
        work.opp = work.opp._replace(lock_pending=False, lock_active=True)

    skill = tp_action.skill
    previous_skill = state.previous_skill
    quick_before = work.me.quick_level

    work.me = work.me._replace(has_declared_skill=True)
    if isinstance(skill, str) and skill in ULTIMATE_TP_SKILLS:
        work.me = work.me._replace(used_ultimate=True)
    if isinstance(skill, str) and skill in STOCK_ALPHA_SKILLS:
        work.me = work.me._replace(stock_alpha_used_this_phase=True)

    charge_was_active = False
    if isinstance(skill, int) and work.me.charge_active:
        charge_was_active = True
        work.me = work.me._replace(charge_active=False)
        work.event("charge_consumed")

    if ntp_action.reaction == MIRROR_MAIN:
        work.opp = work.opp._replace(mirror_ready=False)
        _resolve_mirror(work, tp_action, ntp_action, charge_was_active)
    elif ntp_action.reaction == BLOCK:
        work.opp = work.opp._replace(used_ultimate=True)
        if _is_skip_effect(skill, previous_skill):
            work.event("block_failed_against_skip")
            _resolve_skill_effect(work, tp_action, ntp_action, charge_was_active)
        else:
            work.event("blocked")
    elif ntp_action.reaction == COUNTER:
        if isinstance(skill, str) and skill in ANTI_COUNTER_SKILLS:
            _resolve_anti_counter(work, skill)
        elif skill == COPY:
            _resolve_copy_countered(work, tp_action, ntp_action)
        elif skill in (CHOICE, ALL):
            _resolve_stock_alpha_countered(work, tp_action, ntp_action)
        else:
            _resolve_counter(work, tp_action, ntp_action, charge_was_active)
    else:
        _resolve_skill_effect(work, tp_action, ntp_action, charge_was_active)

    # Record history after resolving references against the previous turn.
    work.previous_skill = skill

    _cleanup_end_of_turn(work, quick_before, skill)
    return _finish_turn(work, skill)


def _resolve_counter(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
    charge_was_active: bool,
) -> None:
    skill = tp_action.skill
    total = tp_action.thumb + ntp_action.thumb

    if isinstance(skill, int):
        fire_count = 2 if charge_was_active else 1
        for _ in range(fire_count):
            if total == skill:
                work.opp = _lower_one(work.opp, "opp", work)
                work.event("counter_number_hit_opp_scores")
            else:
                work.me = _lower_one(work.me, "me", work)
                work.event("counter_number_miss_me_scores")
        return

    if skill == FLASH:
        if tp_action.thumb == ntp_action.thumb:
            _attempt_two_hand_drop(work, "opp", "counter_flash")
        else:
            work.event("counter_flash_miss")
        return

    work.event("counter_no_effect")


def _resolve_anti_counter(work: _Work, skill: str) -> None:
    if skill == FEINT:
        work.me = _lower_one(work.me, "me", work)
        _add_extra_turns(work, 1)
        work.event("feint_success")
    elif skill == LOCK:
        work.opp = work.opp._replace(lock_pending=True)
        work.event("lock_success")


def _resolve_skill_effect(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
    charge_was_active: bool,
) -> None:
    skill = tp_action.skill
    total = tp_action.thumb + ntp_action.thumb

    if isinstance(skill, int):
        fire_count = 2 if charge_was_active else 1
        for _ in range(fire_count):
            if total == skill:
                work.me = _lower_one(work.me, "me", work)
                work.event("number_hit")
            else:
                work.event("number_miss")
                break
        return

    if skill == FLASH:
        if tp_action.thumb == ntp_action.thumb:
            _attempt_two_hand_drop(work, "me", "flash")
        else:
            work.event("flash_miss")
        return

    if skill == CEMENT:
        work.me = _apply_cement(work.me, tp_action.thumb)
        work.opp = _apply_cement(work.opp, ntp_action.thumb)
        work.event("cement_applied")
        return

    if skill == GUARD:
        # New rules 2026-07-13: the WHOLE guard effect (shield + extra turn)
        # fires only once per phase. A second guard in the same phase —
        # declared directly or via Copy/Choice/All — is fully inert.
        if not work.me_guard_extra_used:
            work.me = work.me._replace(guard_active=True)
            _add_extra_turns(work, 1)
            work.me_guard_extra_used = True
            work.event("guard_extra_turn")
        else:
            work.event("guard_inert_this_phase")
        return

    if skill == CHARGE:
        work.me = work.me._replace(charge_active=True)
        work.event("charge_set")
        return

    if skill == QUICK:
        _resolve_quick_for(work, "me", "quick")
        return

    if skill == SKIP:
        work.opp = work.opp._replace(skip_phases=work.opp.skip_phases + 1)
        work.event("skip_applied")
        return

    if skill == MIRROR_PREP:
        work.me = work.me._replace(mirror_ready=True)
        work.event("mirror_ready")
        return

    if skill == FEINT:
        work.event("feint_no_counter")
        return

    if skill == LOCK:
        work.event("lock_no_counter")
        return

    if skill == COPY:
        _resolve_copy(work, tp_action, ntp_action)
        return

    if skill == STOCK:
        previous = work.previous_skill
        if isinstance(previous, str) and previous in REFERENCEABLE_SKILLS:
            work.me = work.me._replace(stock=frozenset(set(work.me.stock) | {previous}))
            work.event("stock_added")
        else:
            work.event("stock_failed")
        return

    if skill == CHOICE:
        if tp_action.choice and tp_action.choice in work.me.stock:
            work.me = work.me._replace(choice_used_this_phase=frozenset(
                    set(work.me.choice_used_this_phase) | {tp_action.choice}
                ),
            )
            _execute_referenced_skill(work, tp_action.choice, tp_action, ntp_action)
            work.event("choice_used")
        else:
            work.event("choice_failed")
        return

    if skill == ALL:
        for stock_skill in _all_order(work.me.stock, tp_action.all_order):
            _execute_referenced_skill(work, stock_skill, tp_action, ntp_action)
        work.me = work.me._replace(stock=frozenset())
        work.event("all_used")
        return

    if skill == DROP:
        work.opp = work.opp._replace(drop_blocked_skills=work.me.stock)
        _add_extra_turns(work, 1)
        work.event("drop_applied")
        return

    if skill == BOOST:
        _add_extra_turns(work, 3)
        work.event("boost")
        return

    if skill == REVERSI:
        work.me, work.opp = _swap_reversi_state(work.me, work.opp)
        work.event("reversi")
        return

    if skill == TIME:
        work.me = work.me._replace(time_active=True)
        _add_extra_turns(work, 1)
        work.event("time_set")


def _resolve_copy(work: _Work, tp_action: TPAction, ntp_action: NTPAction) -> None:
    previous = work.previous_skill
    if previous is None:
        work.event("copy_failed")
        return
    if isinstance(previous, str) and previous not in REFERENCEABLE_SKILLS:
        work.event("copy_failed")
        return
    for _ in range(2):
        _execute_referenced_skill(work, previous, tp_action, ntp_action)
    work.event("copy_used")


def _resolve_copy_countered(work: _Work, tp_action: TPAction, ntp_action: NTPAction) -> None:
    previous = work.previous_skill
    if previous is None:
        work.event("copy_counter_failed")
        return
    if isinstance(previous, str) and previous not in REFERENCEABLE_SKILLS:
        work.event("copy_counter_failed")
        return

    total = tp_action.thumb + ntp_action.thumb
    if isinstance(previous, int):
        for _ in range(2):
            if total == previous:
                work.opp = _lower_one(work.opp, "opp", work)
            else:
                work.me = _lower_one(work.me, "me", work)
        work.event("copy_counter_number")
        return

    for _ in range(2):
        _apply_counter_to_skill(work, previous, tp_action, ntp_action)
    work.event("copy_counter_used")


def _resolve_stock_alpha_countered(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    if tp_action.skill == CHOICE:
        if tp_action.choice:
            _apply_counter_to_skill(work, tp_action.choice, tp_action, ntp_action)
            work.event("choice_countered")
        return

    if tp_action.skill == ALL:
        for stock_skill in _all_order(work.me.stock, tp_action.all_order):
            _apply_counter_to_skill(work, stock_skill, tp_action, ntp_action)
        work.me = work.me._replace(stock=frozenset())
        work.event("all_countered")


def _execute_referenced_skill(
    work: _Work,
    skill: SkillRef,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    total = tp_action.thumb + ntp_action.thumb

    if isinstance(skill, int):
        if total == skill:
            work.me = _lower_one(work.me, "me", work)
        return

    if skill == FLASH:
        if tp_action.thumb == ntp_action.thumb:
            _attempt_two_hand_drop(work, "me", "referenced_flash")
    elif skill == CEMENT:
        work.me = _apply_cement(work.me, tp_action.thumb)
        work.opp = _apply_cement(work.opp, ntp_action.thumb)
    elif skill == GUARD:
        # Whole effect once per phase (see the direct-declaration branch).
        if not work.me_guard_extra_used:
            work.me = work.me._replace(guard_active=True)
            _add_extra_turns(work, 1)
            work.me_guard_extra_used = True
    elif skill == CHARGE:
        work.me = work.me._replace(charge_active=True)
    elif skill == QUICK:
        work.me = work.me._replace(quick_level=2)
    elif skill == SKIP:
        work.opp = work.opp._replace(skip_phases=work.opp.skip_phases + 1)
    elif skill == MIRROR_PREP:
        work.me = work.me._replace(mirror_ready=True)
    elif skill in (FEINT, LOCK):
        # Anti-counter skills only activate when the opponent declares
        # Counter. Referenced without a counter (Copy/Choice/All resolved
        # under no reaction), they do NOTHING — no hand drop, no extra turn,
        # no lock. (Rules doc: 「相手がカウンターを宣言しなければ何も起こらない」;
        # designer ruling 2026-07-13. The countered paths live in
        # _apply_counter_to_skill.)
        work.event("referenced_anti_counter_inert")


def _apply_counter_to_skill(
    work: _Work,
    skill: str,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    if skill == FLASH:
        if tp_action.thumb == ntp_action.thumb:
            _attempt_two_hand_drop(work, "opp", "counter_referenced_flash")
    elif skill == FEINT:
        work.me = _lower_one(work.me, "me", work)
        _add_extra_turns(work, 1)
    elif skill == LOCK:
        work.opp = work.opp._replace(lock_pending=True)


def _resolve_mirror(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
    charge_was_active: bool,
) -> None:
    skill = tp_action.skill

    if skill == COPY:
        _resolve_copy_under_mirror(work, tp_action, ntp_action)
    elif skill == STOCK:
        _resolve_stock_under_mirror(work, tp_action, ntp_action)
    elif skill == CHOICE:
        _resolve_choice_under_mirror(work, tp_action, ntp_action)
    elif skill == ALL:
        _resolve_all_under_mirror(work, tp_action, ntp_action)
    elif not _try_reflect_skill(work, skill, tp_action, ntp_action, charge_was_active):
        _resolve_skill_effect(work, tp_action, ntp_action, charge_was_active)
        work.event("mirror_not_reflectable")

    work.event("mirror_used")


def _try_reflect_skill(
    work: _Work,
    skill: SkillRef,
    tp_action: TPAction,
    ntp_action: NTPAction,
    charge_was_active: bool = False,
) -> bool:
    total = tp_action.thumb + ntp_action.thumb

    if isinstance(skill, int):
        fire_count = 2 if charge_was_active else 1
        for _ in range(fire_count):
            if total == skill:
                work.opp = _lower_one(work.opp, "opp", work)
        work.event("mirror_number")
        return True

    if skill == FLASH:
        if tp_action.thumb == ntp_action.thumb:
            _attempt_two_hand_drop(work, "opp", "mirror_flash")
        return True

    if skill == CEMENT:
        work.me = _apply_cement(work.me, tp_action.thumb)
        work.opp = _apply_cement(work.opp, ntp_action.thumb)
        work.event("mirror_cement")
        return True

    if skill == DROP:
        work.me = work.me._replace(drop_blocked_skills=work.opp.stock)
        work.opp_extra_turns += 1
        work.event("mirror_drop")
        return True

    return False


def _resolve_copy_under_mirror(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    previous = work.previous_skill
    if previous is None:
        work.event("copy_failed")
        return
    if isinstance(previous, str) and previous not in REFERENCEABLE_SKILLS:
        work.event("copy_failed")
        return

    for _ in range(2):
        if not _try_reflect_skill(work, previous, tp_action, ntp_action):
            _execute_referenced_skill(work, previous, tp_action, ntp_action)
            work.event("mirror_reference_not_reflectable")
    work.event("copy_used")


def _resolve_stock_under_mirror(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    previous = work.previous_skill
    if not isinstance(previous, str) or previous not in REFERENCEABLE_SKILLS:
        work.event("stock_failed")
        return

    if not _try_reflect_skill(work, previous, tp_action, ntp_action):
        work.me = work.me._replace(stock=frozenset(set(work.me.stock) | {previous}))
        work.event("stock_added")
        work.event("mirror_reference_not_reflectable")


def _resolve_choice_under_mirror(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    if not tp_action.choice or tp_action.choice not in work.me.stock:
        work.event("choice_failed")
        return

    work.me = work.me._replace(choice_used_this_phase=frozenset(
            set(work.me.choice_used_this_phase) | {tp_action.choice}
        ),
    )
    if not _try_reflect_skill(work, tp_action.choice, tp_action, ntp_action):
        _execute_referenced_skill(work, tp_action.choice, tp_action, ntp_action)
        work.event("mirror_reference_not_reflectable")
    work.event("choice_used")


def _resolve_all_under_mirror(
    work: _Work,
    tp_action: TPAction,
    ntp_action: NTPAction,
) -> None:
    for stock_skill in _all_order(work.me.stock, tp_action.all_order):
        if not _try_reflect_skill(work, stock_skill, tp_action, ntp_action):
            _execute_referenced_skill(work, stock_skill, tp_action, ntp_action)
            work.event("mirror_reference_not_reflectable")
    work.me = work.me._replace(stock=frozenset())
    work.event("all_used")


def _resolve_quick_for(
    work: _Work,
    target: str,
    event_name: str,
    source_player: str | None = None,
) -> None:
    state_attr = source_player or target
    player = work.me if state_attr == "me" else work.opp

    if player.quick_level == 2:
        _attempt_two_hand_drop(work, target, event_name)
        if state_attr == "me":
            work.me = work.me._replace(quick_level=0)
        else:
            work.opp = work.opp._replace(quick_level=0)
    elif player.quick_level == 1:
        if target == "me":
            work.me = _lower_one(work.me, "me", work)
        else:
            work.opp = _lower_one(work.opp, "opp", work)
        if state_attr == "me":
            work.me = work.me._replace(quick_level=0)
        else:
            work.opp = work.opp._replace(quick_level=0)
    else:
        if target == "me":
            work.me = work.me._replace(quick_level=2)
        else:
            work.opp = work.opp._replace(quick_level=2)
    work.event(event_name)


def _attempt_two_hand_drop(work: _Work, target: str, event_name: str) -> None:
    dropper = work.me if target == "me" else work.opp
    opponent = work.opp if target == "me" else work.me
    blocked = work.me_blocked if target == "me" else work.opp_blocked

    if blocked:
        work.event(f"{event_name}_blocked_this_turn")
        return

    if dropper.hands < 2:
        lowered = _lower_one(dropper, target, work)
        if target == "me":
            work.me = lowered
        else:
            work.opp = lowered
        work.event(f"{event_name}_one_hand")
        return

    if opponent.guard_active:
        opponent = opponent._replace(guard_active=False)
        if target == "me":
            work.opp = opponent
            work.me_blocked = True
        else:
            work.me = opponent
            work.opp_blocked = True
        work.event(f"{event_name}_guarded")
        return

    dropper = dropper._replace(hands=0, cement=0)
    if target == "me":
        work.me = dropper
    else:
        work.opp = dropper
    work.event(f"{event_name}_two_hands")


def _lower_one(player: PlayerState, owner: str, work: _Work) -> PlayerState:
    if owner == "me" and work.me_blocked:
        return player
    if owner == "opp" and work.opp_blocked:
        return player
    hands = max(0, player.hands - 1)
    return player._replace(hands=hands, cement=min(player.cement, hands))


def _apply_cement(player: PlayerState, thumb: int) -> PlayerState:
    if thumb <= 0:
        return player
    return player._replace(cement=max(player.cement, min(thumb, player.hands)))


def _add_extra_turns(work: _Work, count: int) -> None:
    work.added_extra_turns += count


def _cleanup_end_of_turn(work: _Work, quick_before: int, skill: SkillRef) -> None:
    if quick_before > 0 and skill != QUICK:
        work.me = work.me._replace(quick_level=max(0, quick_before - 1))
    if work.opp.lock_active:
        work.opp = work.opp._replace(lock_active=False)


def _finish_turn(work: _Work, skill: SkillRef | None) -> Transition:
    both_declared = work.me.has_declared_skill and work.opp.has_declared_skill
    if both_declared:
        if work.me.hands <= 0:
            return Transition(None, 1.0, False, tuple(work.events or ()))
        if work.opp.hands <= 0:
            return Transition(None, -1.0, False, tuple(work.events or ()))

    pending = work.me_extra_turns + work.added_extra_turns
    if work.opp.time_active and pending > 0:
        work.opp = work.opp._replace(time_active=False)
        pending = 0
        work.event("time_interrupted_extra_turn")

    if pending > 0:
        next_state = State(
            me=work.me,
            opp=work.opp,
            previous_skill=work.previous_skill,
            me_extra_turns=pending - 1,
            opp_extra_turns=work.opp_extra_turns,
            me_guard_extra_used_this_phase=work.me_guard_extra_used,
            opp_guard_extra_used_this_phase=work.opp_guard_extra_used,
        )
        return Transition(next_state, None, True, tuple(work.events or ()))

    # True skip: when the turn would pass to a player whose next phase is
    # skipped, that phase is consumed INSTANTLY — no pseudo-turn, no extra
    # discount ply (designer confirmation 2026-07-13). Time still passes for
    # the skipped player (quick decays; shield/drop-block/phase flags expire
    # at their phase boundary). If the skipped player holds Time, they take
    # the turn instead (fresh phase); otherwise the mover continues with a
    # fresh phase of their own.
    incoming = work.opp
    if incoming.skip_phases > 0:
        skipped = incoming._replace(
            skip_phases=incoming.skip_phases - 1,
            quick_level=max(0, incoming.quick_level - 1),
            guard_active=False,
            stock_alpha_used_this_phase=False,
            choice_used_this_phase=frozenset(),
            drop_blocked_skills=frozenset(),
        )
        work.event("phase_skipped")
        if skipped.time_active:
            skipped = skipped._replace(time_active=False)
            work.event("time_skip_interrupt")
            next_state = State(
                me=skipped,
                opp=work.me,
                previous_skill=work.previous_skill,
                me_extra_turns=work.opp_extra_turns,
                opp_extra_turns=0,
                me_guard_extra_used_this_phase=False,
                opp_guard_extra_used_this_phase=work.me_guard_extra_used,
            )
            return Transition(next_state, None, False, tuple(work.events or ()))
        fresh_me = work.me._replace(
            guard_active=False,
            stock_alpha_used_this_phase=False,
            choice_used_this_phase=frozenset(),
            drop_blocked_skills=frozenset(),
        )
        next_state = State(
            me=fresh_me,
            opp=skipped,
            previous_skill=work.previous_skill,
            me_extra_turns=0,
            opp_extra_turns=work.opp_extra_turns,
            me_guard_extra_used_this_phase=False,
            opp_guard_extra_used_this_phase=False,
        )
        return Transition(next_state, None, True, tuple(work.events or ()))

    next_state = _switch_perspective(work)
    return Transition(next_state, None, False, tuple(work.events or ()))


def _switch_perspective(work: _Work) -> State:
    ended_me = work.me
    next_me = work.opp

    next_me = next_me._replace(guard_active=False,
        stock_alpha_used_this_phase=False,
        choice_used_this_phase=frozenset(),
        drop_blocked_skills=frozenset(),
    )

    return State(
        me=next_me,
        opp=ended_me,
        previous_skill=work.previous_skill,
        me_extra_turns=work.opp_extra_turns,
        opp_extra_turns=0,
        me_guard_extra_used_this_phase=False,
        opp_guard_extra_used_this_phase=work.me_guard_extra_used,
    )


def _swap_reversi_state(a: PlayerState, b: PlayerState) -> tuple[PlayerState, PlayerState]:
    new_a = a._replace(hands=b.hands,
        cement=b.cement,
        guard_active=b.guard_active,
        charge_active=b.charge_active,
        quick_level=b.quick_level,
        mirror_ready=b.mirror_ready,
        lock_pending=b.lock_pending,
        lock_active=b.lock_active,
        drop_blocked_skills=b.drop_blocked_skills,
    )
    new_b = b._replace(hands=a.hands,
        cement=a.cement,
        guard_active=a.guard_active,
        charge_active=a.charge_active,
        quick_level=a.quick_level,
        mirror_ready=a.mirror_ready,
        lock_pending=a.lock_pending,
        lock_active=a.lock_active,
        drop_blocked_skills=a.drop_blocked_skills,
    )
    return new_a, new_b


def _all_order(stock: frozenset[str], requested: tuple[str, ...]) -> tuple[str, ...]:
    order: list[str] = []
    remaining = set(stock)
    for skill in requested:
        if skill in remaining:
            order.append(skill)
            remaining.remove(skill)
    order.extend(sorted(remaining))
    return tuple(order)


def _is_skip_effect(skill: SkillRef, previous: SkillRef | None) -> bool:
    return skill == SKIP or (skill == COPY and previous == SKIP)

"""Numba-compiled rules engine over bit-packed states (mirror/reversi OFF).

This is a performance port of ``transition.py`` + ``actions.py`` for the
mirror-OFF / reversi-OFF configuration only. States are packed into two
int64 lanes; actions into small int codes. Semantics are verified against
the reference implementation by an exhaustive differential test
(``tests/test_packed_engine.py``) — any behavioural difference is a bug in
THIS file, never a rules decision.

Bit layout, per player (42 bits):
  hands:0-1  cement:2-3  guard:4  charge:5  quick:6-7  lock_pending:8
  lock_active:9  skip:10-13  ult:14  time:15  declared:16  alpha:17
  stock:18-25  choice_used:26-33  drop_blocked:34-41
Lane 0: me bits | previous_skill<<42 (5b) | me_extra<<47 (4b)
        | me_guard_extra<<51 | opp_guard_extra<<52
Lane 1: opp bits | opp_extra<<42 (4b)

Skill ids (stockable ids are 0..7 — stock masks index by these):
  0 FLASH  1 CEMENT  2 GUARD  3 CHARGE  4 QUICK  5 SKIP  6 FEINT  7 LOCK
  8 COPY   9 STOCK  10 CHOICE 11 ALL    12 DROP  13 BOOST 14 TIME

previous_skill code: 0 = None, 1..5 = declared totals 0..4, 6..20 = skill id+6.

TP action codes: number  = total*4 + thumb              (0..63)
                 skill   = 64 + skill_id*4 + thumb      (64..127)
                 choice  = 128 + choice_id*4 + thumb    (128..159)
                 stock_target (YS_STOCK_FREECHOICE only)
                         = 160 + skill_id*4 + thumb      (160..191, skill_id 0..7)
                 [reserved: 192..195 = choice_collapse.CHOICE_META_BASE,
                  collapsed-CHOICE pseudo-rows, never a real input to step()]
                 pass    = 255
NTP action codes: reaction*4 + thumb, reaction 0=none 1=counter 2=block.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit

from .actions import NTPAction, TPAction
from .constants import (
    ALL,
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
    NONE,
    PASS,
    QUICK,
    SKIP,
    STOCK,
    TIME,
)
from .state import PlayerState, State
from .toggles import STOCK_FREECHOICE, STOCK_FREETEMPO, STOCK_UNLIMITED_ALPHA

# ── skill tables ───────────────────────────────────────────────────────────

SKILL_NAMES: tuple[str, ...] = (
    FLASH, CEMENT, GUARD, CHARGE, QUICK, SKIP, FEINT, LOCK,
    COPY, STOCK, CHOICE, ALL, DROP, BOOST, TIME,
)
SKILL_ID: dict[str, int] = {name: i for i, name in enumerate(SKILL_NAMES)}
N_STOCKABLE = 8  # ids 0..7 are exactly the mirror-off stockable skills

ID_FLASH, ID_CEMENT, ID_GUARD, ID_CHARGE, ID_QUICK, ID_SKIP = 0, 1, 2, 3, 4, 5
ID_FEINT, ID_LOCK, ID_COPY, ID_STOCK, ID_CHOICE, ID_ALL = 6, 7, 8, 9, 10, 11
ID_DROP, ID_BOOST, ID_TIME = 12, 13, 14

PREV_NONE = 0  # previous codes: 1..5 totals, 6..20 skill id + 6

REACTION_NONE, REACTION_COUNTER, REACTION_BLOCK = 0, 1, 2
PASS_CODE = 255

FULL_ALPHABET_MASK = (1 << N_STOCKABLE) - 1

# ── counter-piercing toggle (experimental, defaults OFF) ───────────────────
# Bitmask over skill ids (see SKILL_ID above): when bit `skill_id` is set,
# that skill's effect fires under reaction COUNTER exactly as it would under
# reaction NONE (BLOCK is unaffected). Default 0 = base game, byte-identical.
#
# This is read ONCE at module import into a Python global, which `step`
# below captures as a compile-time constant when numba JIT-compiles it (this
# is how numba treats module globals referenced from @njit code). Changing
# the env var therefore has NO EFFECT on an already-compiled/cached `step` —
# the on-disk numba cache (`complete_solver/__pycache__/*.nb*`,
# `complete_ai/__pycache__/*.nb*`) must be deleted so `step` and its callers
# recompile and pick up the new mask. See module docstring / README note.
_CP_MASK = int(os.environ.get("YS_COUNTER_PIERCE", "0"))

# ── "broken stock" toggles (experimental, default OFF) ─────────────────────
# See complete_solver/toggles.py for the full description of each flag. Same
# numba-global-capture caveat as _CP_MASK above applies to all three: changing
# the env var requires deleting the on-disk numba cache before the next run.
_STOCK_FREECHOICE = STOCK_FREECHOICE
_STOCK_UNLIMITED_ALPHA = STOCK_UNLIMITED_ALPHA
_STOCK_FREETEMPO = STOCK_FREETEMPO

# Targeted-STOCK TP code range (YS_STOCK_FREECHOICE): 160 + skill_id*4 +
# thumb, skill_id 0..7. Codes 128..159 are CHOICE (8 choice_id x 4 thumb) and
# 192..195 are choice_collapse.CHOICE_META_BASE (collapsed CHOICE pseudo-rows)
# — 160..191 is exactly the unused gap between them.
STOCK_TARGET_BASE = 160

# ── packing (python side) ──────────────────────────────────────────────────

_P_HANDS, _P_CEMENT, _P_GUARD, _P_CHARGE = 0, 2, 4, 5
_P_QUICK, _P_LOCKP, _P_LOCKA, _P_SKIP = 6, 8, 9, 10
_P_ULT, _P_TIME, _P_DECLARED, _P_ALPHA = 14, 15, 16, 17
_P_STOCK, _P_CHOICE_USED, _P_DROP_BLOCKED = 18, 26, 34
_PLAYER_BITS = 42
_L0_PREV, _L0_EXTRA, _L0_MEG, _L0_OPPG = 42, 47, 51, 52
_L1_EXTRA = 42


def _mask_of(skills: frozenset[str]) -> int:
    mask = 0
    for name in skills:
        mask |= 1 << SKILL_ID[name]
    assert mask < (1 << N_STOCKABLE), f"non-stockable skill in mask: {skills}"
    return mask


def _skills_of(mask: int) -> frozenset[str]:
    return frozenset(
        SKILL_NAMES[i] for i in range(N_STOCKABLE) if mask & (1 << i)
    )


def pack_player(player: PlayerState) -> int:
    assert not player.mirror_ready, "packed engine is mirror-OFF only"
    assert player.skip_phases < 16 and player.hands <= 3 and player.cement <= 3
    bits = (
        player.hands << _P_HANDS
        | player.cement << _P_CEMENT
        | int(player.guard_active) << _P_GUARD
        | int(player.charge_active) << _P_CHARGE
        | player.quick_level << _P_QUICK
        | int(player.lock_pending) << _P_LOCKP
        | int(player.lock_active) << _P_LOCKA
        | player.skip_phases << _P_SKIP
        | int(player.used_ultimate) << _P_ULT
        | int(player.time_active) << _P_TIME
        | int(player.has_declared_skill) << _P_DECLARED
        | int(player.stock_alpha_used_this_phase) << _P_ALPHA
        | _mask_of(player.stock) << _P_STOCK
        | _mask_of(player.choice_used_this_phase) << _P_CHOICE_USED
        | _mask_of(player.drop_blocked_skills) << _P_DROP_BLOCKED
    )
    return bits


def unpack_player(bits: int) -> PlayerState:
    return PlayerState(
        hands=bits >> _P_HANDS & 3,
        cement=bits >> _P_CEMENT & 3,
        guard_active=bool(bits >> _P_GUARD & 1),
        charge_active=bool(bits >> _P_CHARGE & 1),
        quick_level=bits >> _P_QUICK & 3,
        mirror_ready=False,
        lock_pending=bool(bits >> _P_LOCKP & 1),
        lock_active=bool(bits >> _P_LOCKA & 1),
        skip_phases=bits >> _P_SKIP & 15,
        drop_blocked_skills=_skills_of(bits >> _P_DROP_BLOCKED & 255),
        used_ultimate=bool(bits >> _P_ULT & 1),
        stock=_skills_of(bits >> _P_STOCK & 255),
        stock_alpha_used_this_phase=bool(bits >> _P_ALPHA & 1),
        choice_used_this_phase=_skills_of(bits >> _P_CHOICE_USED & 255),
        time_active=bool(bits >> _P_TIME & 1),
        has_declared_skill=bool(bits >> _P_DECLARED & 1),
    )


def prev_to_code(previous) -> int:
    if previous is None:
        return PREV_NONE
    if isinstance(previous, int):
        assert 0 <= previous <= 4
        return 1 + previous
    return 6 + SKILL_ID[previous]


def code_to_prev(code: int):
    if code == PREV_NONE:
        return None
    if code <= 5:
        return code - 1
    return SKILL_NAMES[code - 6]


def pack_state(state: State) -> tuple[int, int]:
    assert state.me_extra_turns < 16 and state.opp_extra_turns < 16
    lane0 = (
        pack_player(state.me)
        | prev_to_code(state.previous_skill) << _L0_PREV
        | state.me_extra_turns << _L0_EXTRA
        | int(state.me_guard_extra_used_this_phase) << _L0_MEG
        | int(state.opp_guard_extra_used_this_phase) << _L0_OPPG
    )
    lane1 = pack_player(state.opp) | state.opp_extra_turns << _L1_EXTRA
    return lane0, lane1


def unpack_state(lane0: int, lane1: int) -> State:
    player_mask = (1 << _PLAYER_BITS) - 1
    return State(
        me=unpack_player(lane0 & player_mask),
        opp=unpack_player(lane1 & player_mask),
        previous_skill=code_to_prev(lane0 >> _L0_PREV & 31),
        me_extra_turns=lane0 >> _L0_EXTRA & 15,
        opp_extra_turns=lane1 >> _L1_EXTRA & 15,
        me_guard_extra_used_this_phase=bool(lane0 >> _L0_MEG & 1),
        opp_guard_extra_used_this_phase=bool(lane0 >> _L0_OPPG & 1),
    )


def tp_action_to_code(action: TPAction) -> int:
    if action.skill == PASS:
        return PASS_CODE
    if isinstance(action.skill, int):
        return action.skill * 4 + action.thumb
    if action.skill == CHOICE:
        return 128 + SKILL_ID[action.choice] * 4 + action.thumb
    if action.skill == STOCK and action.stock_target is not None:
        return STOCK_TARGET_BASE + SKILL_ID[action.stock_target] * 4 + action.thumb
    return 64 + SKILL_ID[action.skill] * 4 + action.thumb


def code_to_tp_action(code: int, stock_mask: int = 0) -> TPAction:
    if code == PASS_CODE:
        return TPAction(PASS, 0)
    if code < 64:
        return TPAction(code // 4, code % 4)
    if code < 128:
        return TPAction(SKILL_NAMES[(code - 64) // 4], code % 4)
    if code < STOCK_TARGET_BASE:
        choice_id = (code - 128) // 4
        return TPAction(CHOICE, code % 4, choice=SKILL_NAMES[choice_id])
    if code < STOCK_TARGET_BASE + 32:
        skill_id = (code - STOCK_TARGET_BASE) // 4
        return TPAction(STOCK, code % 4, stock_target=SKILL_NAMES[skill_id])
    raise ValueError(f"code_to_tp_action: reserved/unused code {code}")


def ntp_action_to_code(action: NTPAction) -> int:
    reaction = {NONE: 0, COUNTER: 1, BLOCK: 2}[action.reaction]
    return reaction * 4 + action.thumb


def code_to_ntp_action(code: int) -> NTPAction:
    return NTPAction((NONE, COUNTER, BLOCK)[code // 4], code % 4)


# ── compiled legal-action generation ───────────────────────────────────────


@njit(cache=True)
def legal_tp_codes(lane0, lane1, alphabet_mask, max_stock, out):
    """Fill *out* (int64[96]) with legal TP codes; returns the count.

    *max_stock* caps how many skills may be held in stock (99 = unlimited).
    """
    me = lane0 & ((1 << 42) - 1)
    opp = lane1 & ((1 << 42) - 1)
    me_hands = me & 3
    me_cement = me >> 2 & 3
    # Skipped movers do not occur: skip is consumed inside the turn switch
    # (designer confirmation 2026-07-13: PASS is not a real action).

    n = 0
    thumb_lo = me_cement if me_cement < me_hands else me_hands
    total_max = me_hands + (opp & 3)
    prev = lane0 >> 42 & 31
    stock = me >> 18 & 255
    choice_used = me >> 26 & 255
    drop_blocked = me >> 34 & 255
    ult_used = me >> 14 & 1
    alpha_used = me >> 17 & 1

    for thumb in range(thumb_lo, me_hands + 1):
        # numbers
        for total in range(total_max + 1):
            out[n] = total * 4 + thumb
            n += 1
        # plain skills 0..7 (stockable base skills), gated by drop block
        for skill_id in range(8):
            if drop_blocked & (1 << skill_id):
                continue
            out[n] = 64 + skill_id * 4 + thumb
            n += 1
        # COPY: previous must be a number or a referenceable skill (id 0..7).
        # (drop_blocked ⊆ stockable ids 0..7, so COPY itself is never blocked)
        if prev != 0 and (prev <= 5 or prev - 6 < 8):
            out[n] = 64 + 8 * 4 + thumb
            n += 1
        # STOCK: base behaviour = previous is a stockable (alphabet) skill
        # not already stocked. Targeted behaviour (YS_STOCK_FREECHOICE) =
        # any alphabet skill not already stocked, regardless of previous.
        # Either way the holder must be below the stock-size cap.
        if _STOCK_FREECHOICE:
            held = 0
            for bit in range(8):
                held += stock >> bit & 1
            if held < max_stock:
                for skill_id2 in range(8):
                    if (alphabet_mask >> skill_id2 & 1) and not (stock >> skill_id2 & 1):
                        out[n] = STOCK_TARGET_BASE + skill_id2 * 4 + thumb
                        n += 1
        elif prev >= 6 and prev - 6 < 8:
            prev_id = prev - 6
            if (alphabet_mask >> prev_id & 1) and not (stock >> prev_id & 1):
                held = 0
                for bit in range(8):
                    held += stock >> bit & 1
                if held < max_stock:
                    out[n] = 64 + 9 * 4 + thumb
                    n += 1
        # CHOICE / ALL / DROP share the one-per-phase limit (removed by
        # YS_STOCK_UNLIMITED_ALPHA).
        if alpha_used == 0 or _STOCK_UNLIMITED_ALPHA:
            for choice_id in range(8):
                if (stock >> choice_id & 1) and not (choice_used >> choice_id & 1):
                    out[n] = 128 + choice_id * 4 + thumb
                    n += 1
            if stock != 0:
                out[n] = 64 + 11 * 4 + thumb  # ALL
                n += 1
                out[n] = 64 + 12 * 4 + thumb  # DROP
                n += 1
        # ultimates
        if ult_used == 0:
            out[n] = 64 + 13 * 4 + thumb  # BOOST
            n += 1
            out[n] = 64 + 14 * 4 + thumb  # TIME
            n += 1
    return n


@njit(cache=True)
def legal_ntp_codes(lane0, lane1, out):
    """Fill *out* (int64[16]) with legal NTP codes; returns the count."""
    opp = lane1 & ((1 << 42) - 1)
    hands = opp & 3
    cement = opp >> 2 & 3
    thumb_lo = cement if cement < hands else hands
    lock_blocked = (opp >> 8 & 1) or (opp >> 9 & 1)
    ult_used = opp >> 14 & 1

    n = 0
    for thumb in range(thumb_lo, hands + 1):
        out[n] = 0 * 4 + thumb  # none
        n += 1
        if not lock_blocked:
            out[n] = 1 * 4 + thumb  # counter
            n += 1
            if ult_used == 0:
                out[n] = 2 * 4 + thumb  # block
                n += 1
    return n


# ── compiled one-turn transition ───────────────────────────────────────────
# Helper conventions: players are int64 bitfields, mutated functionally.


@njit(cache=True, inline="always")
def _lower_one_bits(player, blocked):
    if blocked:
        return player
    hands = player & 3
    if hands > 0:
        hands -= 1
    cement = player >> 2 & 3
    if cement > hands:
        cement = hands
    return (player & ~np.int64(15)) | hands | (cement << 2)


@njit(cache=True, inline="always")
def _apply_cement_bits(player, thumb):
    if thumb <= 0:
        return player
    hands = player & 3
    cap = thumb if thumb < hands else hands
    cement = player >> 2 & 3
    if cap > cement:
        cement = cap
    return (player & ~np.int64(12)) | (cement << 2)


@njit(cache=True)
def step(lane0, lane1, tp_code, ntp_code, alphabet_mask):
    """One simultaneous turn. Returns (child0, child1, status, reward).

    status: 0 = turn switch, 1 = same turn player, 2 = terminal.
    reward is from the current turn player's perspective (only status 2).
    """
    player_mask = (np.int64(1) << 42) - 1
    me = lane0 & player_mask
    opp = lane1 & player_mask
    prev = lane0 >> 42 & 31
    me_extra = lane0 >> 47 & 15
    me_guard_extra = lane0 >> 51 & 1
    opp_guard_extra = lane0 >> 52 & 1
    opp_extra = lane1 >> 42 & 15

    me_blocked = False
    opp_blocked = False
    added_extra = 0

    # Lock pending → active for the non-turn player.
    if opp >> 8 & 1:
        opp = (opp & ~np.int64(1 << 8)) | (np.int64(1) << 9)

    tp_thumb = tp_code & 3
    ntp_thumb = ntp_code & 3
    reaction = ntp_code >> 2
    total = tp_thumb + ntp_thumb

    new_prev = prev

    if True:
        quick_before = me >> 6 & 3
        me |= np.int64(1) << 16  # has_declared

        is_number = tp_code < 64
        skill_id = -1
        choice_id = -1
        stock_target_id = -1
        declared_total = -1
        if is_number:
            declared_total = tp_code // 4
        elif tp_code < 128:
            skill_id = (tp_code - 64) // 4
        elif tp_code < STOCK_TARGET_BASE:
            skill_id = ID_CHOICE
            choice_id = (tp_code - 128) // 4
        else:
            # Targeted STOCK (YS_STOCK_FREECHOICE only; legal_tp_codes never
            # emits this range otherwise). code < 192 always here — 192+ is
            # choice_collapse.CHOICE_META_BASE and is never a real input.
            skill_id = ID_STOCK
            stock_target_id = (tp_code - STOCK_TARGET_BASE) // 4

        if skill_id == ID_BOOST or skill_id == ID_TIME:
            me |= np.int64(1) << 14  # used_ultimate
        if skill_id == ID_CHOICE or skill_id == ID_ALL or skill_id == ID_DROP:
            me |= np.int64(1) << 17  # stock_alpha_used

        charge_was_active = False
        if is_number and (me >> 5 & 1):
            charge_was_active = True
            me &= ~np.int64(1 << 5)

        # ---- resolution ------------------------------------------------
        if reaction == REACTION_BLOCK:
            opp |= np.int64(1) << 14  # block consumes the ultimate
            skip_effect = skill_id == ID_SKIP or (
                skill_id == ID_COPY and prev == 6 + ID_SKIP
            )
            if skip_effect:
                # resolve normally (skip cannot be blocked)
                me, opp, me_blocked, opp_blocked, added_extra, me_guard_extra = (
                    _resolve_no_reaction(
                        me, opp, me_blocked, opp_blocked, added_extra,
                        me_guard_extra, prev, is_number, declared_total,
                        skill_id, choice_id, stock_target_id, charge_was_active,
                        tp_thumb, ntp_thumb, total,
                    )
                )
            # else: fully blocked, no effect
        elif reaction == REACTION_COUNTER and skill_id >= 0 and (
            (_CP_MASK >> skill_id) & 1
        ) == 1:
            # Counter-piercing (experimental, toggle via YS_COUNTER_PIERCE):
            # this skill fires exactly as under reaction NONE; COUNTER does
            # not negate it. BLOCK (handled above) is unaffected.
            me, opp, me_blocked, opp_blocked, added_extra, me_guard_extra = (
                _resolve_no_reaction(
                    me, opp, me_blocked, opp_blocked, added_extra,
                    me_guard_extra, prev, is_number, declared_total,
                    skill_id, choice_id, stock_target_id, charge_was_active,
                    tp_thumb, ntp_thumb, total,
                )
            )
        elif reaction == REACTION_COUNTER:
            if skill_id == ID_FEINT:
                me = _lower_one_bits(me, me_blocked)
                added_extra += 1
            elif skill_id == ID_LOCK:
                opp |= np.int64(1) << 8  # lock_pending
            elif skill_id == ID_COPY:
                if prev != 0 and (prev <= 5 or prev - 6 < 8):
                    if prev <= 5:
                        prev_total = prev - 1
                        for _ in range(2):
                            if total == prev_total:
                                opp = _lower_one_bits(opp, opp_blocked)
                            else:
                                me = _lower_one_bits(me, me_blocked)
                    else:
                        for _ in range(2):
                            me, opp, me_blocked, opp_blocked, added_extra = (
                                _counter_referenced(
                                    me, opp, me_blocked, opp_blocked,
                                    added_extra, prev - 6, tp_thumb, ntp_thumb,
                                )
                            )
            elif skill_id == ID_CHOICE:
                # NOTE: the reference does NOT mark choice_used under counter.
                if choice_id >= 0 and (me >> 18 >> choice_id & 1):
                    me, opp, me_blocked, opp_blocked, added_extra = (
                        _counter_referenced(
                            me, opp, me_blocked, opp_blocked, added_extra,
                            choice_id, tp_thumb, ntp_thumb,
                        )
                    )
            elif skill_id == ID_ALL:
                stock = me >> 18 & 255
                for ref_id in range(8):
                    if stock >> ref_id & 1:
                        me, opp, me_blocked, opp_blocked, added_extra = (
                            _counter_referenced(
                                me, opp, me_blocked, opp_blocked, added_extra,
                                ref_id, tp_thumb, ntp_thumb,
                            )
                        )
                me &= ~np.int64(255 << 18)
            elif is_number:
                fire = 2 if charge_was_active else 1
                for _ in range(fire):
                    if total == declared_total:
                        opp = _lower_one_bits(opp, opp_blocked)
                    else:
                        me = _lower_one_bits(me, me_blocked)
            elif skill_id == ID_FLASH:
                # Countered flash reflects onto the opponent (target = opp).
                if tp_thumb == ntp_thumb:
                    me, opp, me_blocked, opp_blocked = _two_hand_drop(
                        me, opp, me_blocked, opp_blocked, True
                    )
            # other skills: counter has no effect
        else:  # no reaction
            me, opp, me_blocked, opp_blocked, added_extra, me_guard_extra = (
                _resolve_no_reaction(
                    me, opp, me_blocked, opp_blocked, added_extra,
                    me_guard_extra, prev, is_number, declared_total,
                    skill_id, choice_id, stock_target_id, charge_was_active,
                    tp_thumb, ntp_thumb, total,
                )
            )

        # record previous skill
        if is_number:
            new_prev = 1 + declared_total
        else:
            new_prev = 6 + skill_id

        # end-of-turn cleanup
        if quick_before > 0 and skill_id != ID_QUICK:
            me = (me & ~np.int64(3 << 6)) | ((quick_before - 1) << 6)
        if opp >> 9 & 1:
            opp &= ~np.int64(1 << 9)  # lock_active expires

    # ---- finish turn ----------------------------------------------------
    both_declared = (me >> 16 & 1) and (opp >> 16 & 1)
    # Both players must have declared before either can reach zero hands.
    # Restore the final hand when an opening combo would otherwise create a
    # zero-hand, non-terminal state with no valid continuation.
    if not both_declared:
        if (me & 3) <= 0:
            me = (me & ~np.int64(15)) | np.int64(1)
        if (opp & 3) <= 0:
            opp = (opp & ~np.int64(15)) | np.int64(1)
    if both_declared:
        if (me & 3) <= 0:
            return np.int64(0), np.int64(0), np.int64(2), np.int64(1)
        if (opp & 3) <= 0:
            return np.int64(0), np.int64(0), np.int64(2), np.int64(-1)

    pending = me_extra + added_extra
    if (opp >> 15 & 1) and pending > 0:
        opp &= ~np.int64(1 << 15)  # time interrupts the extra turns
        pending = 0

    if pending > 0:
        if pending > 15:
            pending = 15
        child0 = (
            me
            | np.int64(new_prev) << 42
            | np.int64(pending - 1) << 47
            | np.int64(me_guard_extra) << 51
            | np.int64(opp_guard_extra) << 52
        )
        child1 = opp | np.int64(opp_extra) << 42
        return child0, child1, np.int64(1), np.int64(0)

    # Turn switch with TRUE skip consumption (designer confirmation
    # 2026-07-13): a skipped incoming player's phase is consumed instantly —
    # quick decays, shield/drop-block/phase flags expire — with no pseudo
    # turn. If they hold Time they take the turn instead (fresh phase);
    # otherwise the mover continues with a fresh phase of their own.
    ended_me = me
    incoming = opp
    if (incoming >> 10 & 15) > 0:
        skips = incoming >> 10 & 15
        incoming = (incoming & ~np.int64(15 << 10)) | ((skips - 1) << 10)
        quick = incoming >> 6 & 3
        if quick > 0:
            incoming = (incoming & ~np.int64(3 << 6)) | ((quick - 1) << 6)
        incoming &= ~np.int64(1 << 4)       # shield expires
        incoming &= ~np.int64(1 << 17)      # stock_alpha resets
        incoming &= ~np.int64(255 << 26)    # choice_used clears
        incoming &= ~np.int64(255 << 34)    # drop_blocked clears
        if incoming >> 15 & 1:              # Time interrupt
            incoming &= ~np.int64(1 << 15)
            child0 = (
                incoming
                | np.int64(new_prev) << 42
                | np.int64(opp_extra) << 47
                | np.int64(0) << 51
                | np.int64(me_guard_extra) << 52
            )
            child1 = ended_me  # extras 0
            return child0, child1, np.int64(0), np.int64(0)
        fresh_me = ended_me
        fresh_me &= ~np.int64(1 << 4)
        fresh_me &= ~np.int64(1 << 17)
        fresh_me &= ~np.int64(255 << 26)
        fresh_me &= ~np.int64(255 << 34)
        child0 = (
            fresh_me
            | np.int64(new_prev) << 42
            | np.int64(0) << 47
            | np.int64(0) << 51
            | np.int64(0) << 52
        )
        child1 = incoming | np.int64(opp_extra) << 42
        return child0, child1, np.int64(1), np.int64(0)

    next_me = incoming
    next_me &= ~np.int64(1 << 4)          # guard_active off
    next_me &= ~np.int64(1 << 17)         # stock_alpha_used off
    next_me &= ~np.int64(255 << 26)       # choice_used cleared
    next_me &= ~np.int64(255 << 34)       # drop_blocked cleared

    child0 = (
        next_me
        | np.int64(new_prev) << 42
        | np.int64(opp_extra) << 47
        | np.int64(0) << 51                # me_guard_extra resets
        | np.int64(me_guard_extra) << 52   # becomes opp side flag
    )
    child1 = ended_me  # opp_extra becomes 0
    return child0, child1, np.int64(0), np.int64(0)


@njit(cache=True)
def _two_hand_drop(me, opp, me_blocked, opp_blocked, target_is_opp):
    """Attempt a two-hand drop for the target player."""
    if target_is_opp:
        dropper, opponent, blocked = opp, me, opp_blocked
    else:
        dropper, opponent, blocked = me, opp, me_blocked

    if blocked:
        return me, opp, me_blocked, opp_blocked

    hands = dropper & 3
    both_declared = (me >> 16 & 1) and (opp >> 16 & 1)
    if hands <= 2 and not both_declared:
        return me, opp, me_blocked, opp_blocked
    if hands < 2:
        dropper = _lower_one_bits(dropper, blocked)
    elif opponent >> 4 & 1:  # opponent's guard cancels and blocks this side
        opponent &= ~np.int64(1 << 4)
        if target_is_opp:
            opp_blocked = True
        else:
            me_blocked = True
        if target_is_opp:
            return opponent, dropper, me_blocked, opp_blocked
        return dropper, opponent, me_blocked, opp_blocked
    else:
        dropper &= ~np.int64(15)  # hands=0, cement=0

    if target_is_opp:
        return opponent, dropper, me_blocked, opp_blocked
    return dropper, opponent, me_blocked, opp_blocked


@njit(cache=True)
def _execute_referenced(me, opp, me_blocked, opp_blocked, added_extra,
                        me_guard_extra, ref_id, tp_thumb, ntp_thumb):
    """_execute_referenced_skill: referenced by COPY / CHOICE / ALL."""
    if ref_id == ID_FLASH:
        if tp_thumb == ntp_thumb:
            me, opp, me_blocked, opp_blocked = _two_hand_drop(
                me, opp, me_blocked, opp_blocked, False
            )
    elif ref_id == ID_CEMENT:
        me = _apply_cement_bits(me, tp_thumb)
        opp = _apply_cement_bits(opp, ntp_thumb)
    elif ref_id == ID_GUARD:
        # Whole guard effect fires once per phase (rules 2026-07-13).
        if me_guard_extra == 0:
            me |= np.int64(1) << 4
            added_extra += 1
            me_guard_extra = 1
    elif ref_id == ID_CHARGE:
        me |= np.int64(1) << 5
    elif ref_id == ID_QUICK:
        me = (me & ~np.int64(3 << 6)) | (np.int64(2) << 6)
    elif ref_id == ID_SKIP:
        skips = opp >> 10 & 15
        if skips < 15:
            opp = (opp & ~np.int64(15 << 10)) | ((skips + 1) << 10)
    # ID_FEINT / ID_LOCK: anti-counter skills are INERT when referenced
    # without a counter (designer ruling 2026-07-13); the countered paths
    # live in _counter_referenced.
    return me, opp, me_blocked, opp_blocked, added_extra, me_guard_extra


@njit(cache=True)
def _counter_referenced(me, opp, me_blocked, opp_blocked, added_extra,
                        ref_id, tp_thumb, ntp_thumb):
    """_apply_counter_to_skill: referenced skill resolved under counter."""
    if ref_id == ID_FLASH:
        if tp_thumb == ntp_thumb:
            me, opp, me_blocked, opp_blocked = _two_hand_drop(
                me, opp, me_blocked, opp_blocked, True
            )
    elif ref_id == ID_FEINT:
        me = _lower_one_bits(me, me_blocked)
        added_extra += 1
    elif ref_id == ID_LOCK:
        opp |= np.int64(1) << 8
    return me, opp, me_blocked, opp_blocked, added_extra


@njit(cache=True)
def _resolve_no_reaction(me, opp, me_blocked, opp_blocked, added_extra,
                         me_guard_extra, prev, is_number, declared_total,
                         skill_id, choice_id, stock_target_id, charge_was_active,
                         tp_thumb, ntp_thumb, total):
    """_resolve_skill_effect: TP's declaration with no (effective) reaction."""
    if is_number:
        fire = 2 if charge_was_active else 1
        for _ in range(fire):
            if total == declared_total:
                me = _lower_one_bits(me, me_blocked)
            else:
                break
        return me, opp, me_blocked, opp_blocked, added_extra, me_guard_extra

    if skill_id == ID_FLASH:
        if tp_thumb == ntp_thumb:
            me, opp, me_blocked, opp_blocked = _two_hand_drop(
                me, opp, me_blocked, opp_blocked, False
            )
    elif skill_id == ID_CEMENT:
        me = _apply_cement_bits(me, tp_thumb)
        opp = _apply_cement_bits(opp, ntp_thumb)
    elif skill_id == ID_GUARD:
        # Whole guard effect fires once per phase (rules 2026-07-13).
        if me_guard_extra == 0:
            me |= np.int64(1) << 4
            added_extra += 1
            me_guard_extra = 1
    elif skill_id == ID_CHARGE:
        me |= np.int64(1) << 5
    elif skill_id == ID_QUICK:
        quick = me >> 6 & 3
        if quick == 2:
            me = (me & ~np.int64(3 << 6))
            me, opp, me_blocked, opp_blocked = _two_hand_drop(
                me, opp, me_blocked, opp_blocked, False
            )
        elif quick == 1:
            me = _lower_one_bits(me, me_blocked)
            me = (me & ~np.int64(3 << 6))
        else:
            me = (me & ~np.int64(3 << 6)) | (np.int64(2) << 6)
    elif skill_id == ID_SKIP:
        skips = opp >> 10 & 15
        if skips < 15:
            opp = (opp & ~np.int64(15 << 10)) | ((skips + 1) << 10)
    elif skill_id == ID_COPY:
        if prev != 0 and (prev <= 5 or prev - 6 < 8):
            if prev <= 5:
                prev_total = prev - 1
                for _ in range(2):
                    if total == prev_total:
                        me = _lower_one_bits(me, me_blocked)
            else:
                for _ in range(2):
                    (me, opp, me_blocked, opp_blocked, added_extra,
                     me_guard_extra) = _execute_referenced(
                        me, opp, me_blocked, opp_blocked, added_extra,
                        me_guard_extra, prev - 6, tp_thumb, ntp_thumb,
                    )
    elif skill_id == ID_STOCK:
        if _STOCK_FREECHOICE and stock_target_id >= 0:
            # Targeted STOCK (YS_STOCK_FREECHOICE): legal_tp_codes only ever
            # emits a target not already held, so this always succeeds.
            if not (me >> 18 >> stock_target_id & 1):
                me |= np.int64(1) << (18 + stock_target_id)
        elif prev >= 6 and prev - 6 < 8:
            me |= np.int64(1) << (18 + (prev - 6))
        if _STOCK_FREETEMPO:
            # YS_STOCK_FREETEMPO: STOCK does not pass initiative (base and
            # targeted alike).
            added_extra += 1
    elif skill_id == ID_CHOICE:
        if choice_id >= 0 and (me >> 18 >> choice_id & 1):
            me |= np.int64(1) << (26 + choice_id)
            (me, opp, me_blocked, opp_blocked, added_extra,
             me_guard_extra) = _execute_referenced(
                me, opp, me_blocked, opp_blocked, added_extra,
                me_guard_extra, choice_id, tp_thumb, ntp_thumb,
            )
    elif skill_id == ID_ALL:
        stock = me >> 18 & 255
        for ref_id in range(8):
            if stock >> ref_id & 1:
                (me, opp, me_blocked, opp_blocked, added_extra,
                 me_guard_extra) = _execute_referenced(
                    me, opp, me_blocked, opp_blocked, added_extra,
                    me_guard_extra, ref_id, tp_thumb, ntp_thumb,
                )
        me &= ~np.int64(255 << 18)
    elif skill_id == ID_DROP:
        stock = me >> 18 & 255
        opp = (opp & ~np.int64(255 << 34)) | (stock << 34)
        added_extra += 1
    elif skill_id == ID_BOOST:
        added_extra += 3
    elif skill_id == ID_TIME:
        me |= np.int64(1) << 15
        added_extra += 1
    return me, opp, me_blocked, opp_blocked, added_extra, me_guard_extra

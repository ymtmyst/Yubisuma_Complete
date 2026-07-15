"""Subprocess worker for counter-piercing (YS_COUNTER_PIERCE) golden checks.

Why a subprocess: numba's on-disk cache (``cache=True``) is keyed on the
compiled function's bytecode, NOT on the runtime value of the module-level
globals it closes over (see packed_engine._CP_MASK docstring). A process
that has already imported/compiled ``packed_engine.step`` with one mask
value will keep using that compiled specialization even if the Python
global is changed afterwards, and a fresh process can silently load a STALE
on-disk cache compiled under a different mask. The only reliable way to
prove "mask X actually took effect" is: delete the on-disk numba cache,
then start a FRESH interpreter with YS_COUNTER_PIERCE=X already set in the
environment before anything imports packed_engine.

Usage: ``python -m complete_solver.tests._cp_worker <expected_mask>``
Exits 0 and prints "OK" on success; raises (non-zero exit, traceback on
stderr) on any failure, including "the mask I read doesn't match what the
caller expected" (proves the cache-clear + reimport actually worked).
"""

from __future__ import annotations

import sys
import unittest

expected_mask = int(sys.argv[1])

from complete_solver.actions import NTPAction, RulesConfig, TPAction  # noqa: E402
from complete_solver.constants import (  # noqa: E402
    ALL,
    BLOCK,
    CEMENT,
    CHOICE,
    COUNTER,
    DROP,
    FLASH,
    STOCK,
)
from complete_solver.packed_engine import (  # noqa: E402
    FULL_ALPHABET_MASK,
    _CP_MASK as p_mask,
    ntp_action_to_code,
    pack_state,
    step,
    tp_action_to_code,
    unpack_state,
)
from complete_solver.state import PlayerState, State  # noqa: E402
from complete_solver.transition import _CP_MASK as t_mask  # noqa: E402
from complete_solver.transition import transition  # noqa: E402

case = unittest.TestCase()
case.maxDiff = None

# 0. Prove the mask actually took effect in THIS process (both engines).
case.assertEqual(
    t_mask, expected_mask,
    f"transition._CP_MASK={t_mask} != expected {expected_mask} "
    "(env var / reimport did not take effect)",
)
case.assertEqual(
    p_mask, expected_mask,
    f"packed_engine._CP_MASK={p_mask} != expected {expected_mask} "
    "(env var / reimport did not take effect)",
)

FULL_CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)
STOCK_BIT = 1 << 9
CEMENT_BIT = 1 << 1


def run_both(state: State, tp_action: TPAction, ntp_action: NTPAction):
    """Run reference transition() and packed step(); assert they agree."""
    ref = transition(state, tp_action, ntp_action, FULL_CONFIG)
    lane0, lane1 = pack_state(state)
    tp_code = tp_action_to_code(tp_action)
    ntp_code = ntp_action_to_code(ntp_action)
    child0, child1, status, reward = step(
        lane0, lane1, tp_code, ntp_code, FULL_ALPHABET_MASK
    )
    if ref.terminal_reward is not None:
        case.assertEqual(int(status), 2, "packed engine missed terminal")
        case.assertEqual(float(reward), ref.terminal_reward, "reward mismatch")
    else:
        case.assertNotEqual(int(status), 2, "packed engine false terminal")
        packed_next = unpack_state(int(child0), int(child1))
        case.assertEqual(
            packed_next, ref.next_state,
            f"packed/reference mismatch: packed={packed_next} ref={ref.next_state}",
        )
    return ref


# NOTE ON PERSPECTIVE SWITCHING: transition() resolves one turn from the
# DECLARER's ("me") viewpoint. When the turn ends with no pending extra
# turns and no skip (true for every scenario below), _switch_perspective
# swaps sides: the declarer's PlayerState (with this turn's effects
# applied) becomes `next_state.opp`, and the reactor's PlayerState becomes
# `next_state.me`. So "did the effect land on the declarer" must be
# checked via `next_state.opp`, and "did it land on the reactor" via
# `next_state.me` — the OPPOSITE of the pre-turn me/opp naming. Getting
# this backwards is the single easiest way to write a golden test that
# looks right and is wrong; check both sides where cheap and prefer the
# switch-invariant `events` list as the primary signal.


def check_stock_under_counter() -> None:
    """STOCK declared (by the declarer/"me"), previous=FLASH (stockable),
    opp reacts COUNTER. After the turn-end perspective switch, the
    declarer's (possibly updated) stock lands in next_state.opp.stock."""
    state = State(
        me=PlayerState(hands=2, cement=0),
        opp=PlayerState(hands=2, cement=0),
        previous_skill=FLASH,
    )
    tp = TPAction(STOCK, thumb=0)
    ntp = NTPAction(COUNTER, thumb=0)
    ref = run_both(state, tp, ntp)
    stocked = bool(
        ref.next_state and FLASH in ref.next_state.opp.stock
    )
    not_stocked_other_side = bool(
        ref.next_state and FLASH not in ref.next_state.me.stock
    )
    case.assertTrue(not_stocked_other_side, "FLASH must never land in the reactor's stock")
    if expected_mask & STOCK_BIT:
        case.assertTrue(stocked, "STOCK bit set: expected counter-pierce to stock FLASH")
        case.assertIn("stock_added", ref.events)
    else:
        case.assertFalse(stocked, "STOCK bit unset: counter must NOT stock (baseline)")
        case.assertIn("counter_no_effect", ref.events)


def check_stock_under_block_still_blocked() -> None:
    """BLOCK must still prevent stocking, regardless of the pierce mask."""
    state = State(
        me=PlayerState(hands=2, cement=0),
        opp=PlayerState(hands=2, cement=0),
        previous_skill=FLASH,
    )
    tp = TPAction(STOCK, thumb=0)
    ntp = NTPAction(BLOCK, thumb=0)
    ref = run_both(state, tp, ntp)
    stocked_either_side = bool(
        ref.next_state
        and (FLASH in ref.next_state.me.stock or FLASH in ref.next_state.opp.stock)
    )
    case.assertFalse(stocked_either_side, "BLOCK must still prevent stocking even with STOCK bit set")
    case.assertIn("blocked", ref.events)


def check_cement_under_counter() -> None:
    """CEMENT declared under COUNTER: the declarer raises their OWN cement
    by their own thumb, the reactor raises their OWN cement by their own
    thumb. After the perspective switch, the declarer's result lands in
    next_state.opp and the reactor's in next_state.me."""
    state = State(
        me=PlayerState(hands=2, cement=0),
        opp=PlayerState(hands=2, cement=0),
    )
    tp = TPAction(CEMENT, thumb=2)   # declarer's own thumb
    ntp = NTPAction(COUNTER, thumb=1)  # reactor's own thumb
    ref = run_both(state, tp, ntp)
    declarer_cement = ref.next_state.opp.cement if ref.next_state else -1
    reactor_cement = ref.next_state.me.cement if ref.next_state else -1
    if expected_mask & CEMENT_BIT:
        case.assertEqual(declarer_cement, 2, "CEMENT bit set: expected cement to apply to declarer")
        case.assertEqual(reactor_cement, 1, "CEMENT bit set: expected cement to apply to reactor")
        case.assertIn("cement_applied", ref.events)
    else:
        case.assertEqual(declarer_cement, 0, "CEMENT bit unset: counter must NOT apply cement (baseline)")
        case.assertEqual(reactor_cement, 0, "CEMENT bit unset: counter must NOT apply cement (baseline)")
        case.assertIn("counter_no_effect", ref.events)


def check_cement_under_block_still_blocked() -> None:
    state = State(
        me=PlayerState(hands=2, cement=0),
        opp=PlayerState(hands=2, cement=0),
    )
    tp = TPAction(CEMENT, thumb=2)
    ntp = NTPAction(BLOCK, thumb=1)
    ref = run_both(state, tp, ntp)
    case.assertEqual(ref.next_state.me.cement, 0, "BLOCK must still block cement")
    case.assertEqual(ref.next_state.opp.cement, 0, "BLOCK must still block cement")
    case.assertIn("blocked", ref.events)


def check_choice_all_drop_unaffected_under_counter() -> None:
    """CHOICE(10)/ALL(11)/DROP(12) behavior under COUNTER must be identical
    to the baseline no-pierce behavior in every run, since only bit 9
    (STOCK) or bit 1 (CEMENT) is ever set in this experiment — never bits
    10/11/12. This is a fixed-outcome assertion (not conditioned on
    expected_mask): if it passes identically whether run under mask 0, 2,
    or 512, that IS the proof of "unaffected".

    A countered FLASH reference reflects onto the REACTOR (see
    packed_engine._counter_referenced / transition._apply_counter_to_skill,
    target="opp" = the reactor from the declarer's viewpoint). After the
    end-of-turn perspective switch the reactor becomes next_state.me, so
    the dropped hands show up on next_state.me, not next_state.opp.
    """
    # CHOICE referencing FLASH, matching thumbs -> counter-flash two-hand
    # drop targeting the reactor (2 hands, no guard -> both hands drop).
    state = State(
        me=PlayerState(hands=2, cement=0, stock=frozenset({FLASH})),
        opp=PlayerState(hands=2, cement=0),
    )
    tp = TPAction(CHOICE, thumb=1, choice=FLASH)
    ntp = NTPAction(COUNTER, thumb=1)
    ref = run_both(state, tp, ntp)
    case.assertEqual(ref.next_state.me.hands, 0, "choice-under-counter flash should two-hand-drop the reactor")
    case.assertIn("choice_countered", ref.events)

    # ALL with stock={FLASH}, matching thumbs -> same two-hand drop, and
    # stock is cleared regardless of counter (declarer's stock -> opp side
    # after the switch).
    state = State(
        me=PlayerState(hands=2, cement=0, stock=frozenset({FLASH})),
        opp=PlayerState(hands=2, cement=0),
    )
    tp = TPAction(ALL, thumb=1, all_order=(FLASH,))
    ntp = NTPAction(COUNTER, thumb=1)
    ref = run_both(state, tp, ntp)
    case.assertEqual(ref.next_state.me.hands, 0, "all-under-counter flash should two-hand-drop the reactor")
    case.assertEqual(ref.next_state.opp.stock, frozenset(), "all clears the declarer's stock regardless of counter")
    case.assertIn("all_countered", ref.events)

    # DROP under counter: baseline says "counter has no effect" -> nobody's
    # drop_blocked_skills changes (checked on both sides).
    state = State(
        me=PlayerState(hands=2, cement=0, stock=frozenset({FLASH})),
        opp=PlayerState(hands=2, cement=0),
    )
    tp = TPAction(DROP, thumb=0)
    ntp = NTPAction(COUNTER, thumb=0)
    ref = run_both(state, tp, ntp)
    case.assertEqual(
        ref.next_state.me.drop_blocked_skills, frozenset(),
        "drop-under-counter must have no effect regardless of the STOCK/CEMENT pierce mask",
    )
    case.assertEqual(
        ref.next_state.opp.drop_blocked_skills, frozenset(),
        "drop-under-counter must have no effect regardless of the STOCK/CEMENT pierce mask",
    )
    case.assertIn("counter_no_effect", ref.events)


def run_parity_fuzz() -> None:
    """Reuse the project's reference-vs-packed differential harness (the
    same one that guards the base game) under the CURRENT mask, so a fuzz
    of random playouts through the pierced skill(s) is also cross-checked
    engine-vs-engine, not just the hand-picked scenarios above."""
    sys.path.insert(0, "")
    from complete_solver.tests.test_packed_engine import (  # noqa: E402
        FULL_ALPHABET_MASK as _FAM,
        run_playouts,
    )

    run_playouts(case, FULL_CONFIG, _FAM, seed=12345 + expected_mask, games=20, max_steps=40)


check_stock_under_counter()
check_stock_under_block_still_blocked()
check_cement_under_counter()
check_cement_under_block_still_blocked()
check_choice_all_drop_unaffected_under_counter()
run_parity_fuzz()

print("OK")

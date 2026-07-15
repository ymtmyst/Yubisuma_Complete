"""Subprocess worker for the "broken stock" toggle golden checks
(YS_STOCK_FREECHOICE / YS_STOCK_UNLIMITED_ALPHA / YS_STOCK_FREETEMPO).

Same rationale as ``_cp_worker.py`` (see its docstring): numba's on-disk
``cache=True`` is keyed on bytecode, not on the runtime value of the module
globals a compiled function closes over. The only reliable way to prove
"these flags actually took effect" is: delete the on-disk numba cache, then
start a FRESH interpreter with the env vars already set before anything
imports ``complete_solver.packed_engine``.

Usage: ``python -m complete_solver.tests._stock_worker <fc> <ua> <ft>``
where each of ``fc``/``ua``/``ft`` is "1" or "0" for YS_STOCK_FREECHOICE /
YS_STOCK_UNLIMITED_ALPHA / YS_STOCK_FREETEMPO respectively (must match the
env vars the caller set for this subprocess). Exits 0 and prints "OK" on
success.
"""

from __future__ import annotations

import sys
import unittest

expected_fc = sys.argv[1] not in ("0", "")
expected_ua = sys.argv[2] not in ("0", "")
expected_ft = sys.argv[3] not in ("0", "")

import numpy as np  # noqa: E402

from complete_solver.actions import (  # noqa: E402
    NTPAction, RulesConfig, TPAction, legal_tp_actions,
)
from complete_solver.constants import (  # noqa: E402
    ALL, CHOICE, DROP, FEINT, GUARD, LOCK, NONE, STOCK,
)
from complete_solver.packed_engine import (  # noqa: E402
    FULL_ALPHABET_MASK,
    STOCK_TARGET_BASE,
    _STOCK_FREECHOICE as p_fc,
    _STOCK_FREETEMPO as p_ft,
    _STOCK_UNLIMITED_ALPHA as p_ua,
    legal_tp_codes,
    ntp_action_to_code,
    pack_state,
    step,
    tp_action_to_code,
    unpack_state,
)
from complete_solver.state import PlayerState, State  # noqa: E402
from complete_solver.toggles import (  # noqa: E402
    STOCK_FREECHOICE as t_fc,
    STOCK_FREETEMPO as t_ft,
    STOCK_UNLIMITED_ALPHA as t_ua,
)
from complete_solver.transition import transition  # noqa: E402

case = unittest.TestCase()
case.maxDiff = None

# 0. Prove the flags actually took effect in THIS process (both engines read
# from the same complete_solver.toggles module, but assert both import paths
# independently in case a future refactor breaks the shared source).
for name, got, want in (
    ("packed_engine._STOCK_FREECHOICE", p_fc, expected_fc),
    ("packed_engine._STOCK_UNLIMITED_ALPHA", p_ua, expected_ua),
    ("packed_engine._STOCK_FREETEMPO", p_ft, expected_ft),
    ("toggles.STOCK_FREECHOICE (imported by transition/actions)", t_fc, expected_fc),
    ("toggles.STOCK_UNLIMITED_ALPHA", t_ua, expected_ua),
    ("toggles.STOCK_FREETEMPO", t_ft, expected_ft),
):
    case.assertEqual(got, want, f"{name}={got} != expected {want} (env/reimport did not take effect)")

FULL_CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)


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
        case.assertEqual(
            bool(status == 1), ref.same_turn_player,
            f"same-turn flag mismatch: packed_status={int(status)} ref={ref.same_turn_player}",
        )
        packed_next = unpack_state(int(child0), int(child1))
        case.assertEqual(
            packed_next, ref.next_state,
            f"packed/reference mismatch: packed={packed_next} ref={ref.next_state}",
        )
    return ref


def assert_legal_action_sets_match(state: State) -> tuple:
    """Reference legal_tp_actions and packed legal_tp_codes must agree
    exactly (as codes). Returns (ref_actions, ref_codes)."""
    ref_actions = legal_tp_actions(state, FULL_CONFIG)
    ref_codes = sorted(tp_action_to_code(a) for a in ref_actions)
    lane0, lane1 = pack_state(state)
    tp_buf = np.zeros(96, dtype=np.int64)
    n_tp = legal_tp_codes(
        np.int64(lane0), np.int64(lane1), FULL_ALPHABET_MASK, np.int64(99), tp_buf
    )
    case.assertLessEqual(n_tp, 96, "legal_tp_codes overflowed its 96-slot buffer")
    packed_codes = sorted(int(c) for c in tp_buf[:n_tp])
    case.assertEqual(packed_codes, ref_codes, f"TP action set mismatch at {state}")
    return ref_actions, ref_codes


# ── Toggle 1: YS_STOCK_FREECHOICE ───────────────────────────────────────────


def check_freechoice_from_fresh_state() -> None:
    """From a truly fresh initial state (previous_skill=None), STOCK-target
    is legal under FREECHOICE (illegal under baseline, since base STOCK
    requires previous_skill to already be a stockable skill)."""
    state = State()  # default players, previous_skill=None
    ref_actions, _ = assert_legal_action_sets_match(state)

    targeted = [
        a for a in ref_actions
        if a.skill == STOCK and a.stock_target is not None
    ]
    if expected_fc:
        case.assertEqual(
            len(targeted), 8 * 3,  # 8 targets x 3 legal thumbs (hands=2,cement=0)
            f"expected one STOCK-target action per stockable skill per thumb, got {targeted}",
        )
        feint_action = next(
            a for a in targeted if a.stock_target == FEINT and a.thumb == 0
        )
        ref = run_both(state, feint_action, NTPAction(NONE, thumb=0))
        # Perspective-switch caveat: FREETEMPO (if also on) keeps the move
        # with the mover (same_turn_player True, no switch => stays .me).
        landed = ref.next_state.me.stock if ref.same_turn_player else ref.next_state.opp.stock
        case.assertIn(FEINT, landed, "STOCK-target(FEINT) must stock FEINT")
        case.assertIn("stock_added", ref.events)
    else:
        case.assertEqual(targeted, [], "baseline: no targeted STOCK actions should exist at all")
        plain_stock = [a for a in ref_actions if a.skill == STOCK]
        case.assertEqual(
            plain_stock, [],
            "baseline: STOCK must be illegal from a fresh state (previous_skill=None)",
        )


def check_freechoice_cannot_target_already_held() -> None:
    """A skill already in stock must never appear as a STOCK-target,
    regardless of previous_skill."""
    state = State(
        me=PlayerState(hands=2, cement=0, stock=frozenset({FEINT})),
        opp=PlayerState(hands=2, cement=0),
        previous_skill=FEINT,
    )
    ref_actions, _ = assert_legal_action_sets_match(state)
    targeted = [
        a for a in ref_actions
        if a.skill == STOCK and a.stock_target is not None
    ]
    if expected_fc:
        held_targets = [a for a in targeted if a.stock_target == FEINT]
        case.assertEqual(held_targets, [], "must never target an already-held skill")
        case.assertTrue(
            any(a.stock_target == LOCK for a in targeted),
            "an unheld stockable skill (LOCK) must still be targetable",
        )
        # Actually declare it and check the packed encoding round-trips
        # through the dedicated 160..191 code range.
        lock_action = next(a for a in targeted if a.stock_target == LOCK and a.thumb == 0)
        code = tp_action_to_code(lock_action)
        case.assertGreaterEqual(code, STOCK_TARGET_BASE)
        case.assertLess(code, STOCK_TARGET_BASE + 32)
        ref = run_both(state, lock_action, NTPAction(NONE, thumb=0))
        landed = ref.next_state.me.stock if ref.same_turn_player else ref.next_state.opp.stock
        case.assertIn(LOCK, landed)
        case.assertIn(FEINT, landed, "existing stock must be preserved")
    else:
        plain_stock = [a for a in ref_actions if a.skill == STOCK]
        case.assertEqual(
            plain_stock, [],
            "baseline: STOCK must be illegal when previous_skill is already held",
        )


# ── Toggle 2: YS_STOCK_UNLIMITED_ALPHA ──────────────────────────────────────


def check_unlimited_alpha_allows_repeat() -> None:
    """After CHOICE has already fired once this phase (stock_alpha_used_this_
    phase=True, one skill already choice_used), a second CHOICE (of the OTHER
    stocked skill) / ALL / DROP is still legal only when the toggle is on."""
    state = State(
        me=PlayerState(
            hands=2, cement=0, stock=frozenset({FEINT, LOCK}),
            choice_used_this_phase=frozenset({FEINT}),
            stock_alpha_used_this_phase=True,
        ),
        opp=PlayerState(hands=2, cement=0),
    )
    ref_actions, _ = assert_legal_action_sets_match(state)
    choice_actions = [a for a in ref_actions if a.skill == CHOICE]
    all_actions = [a for a in ref_actions if a.skill == ALL]
    drop_actions = [a for a in ref_actions if a.skill == DROP]

    if expected_ua:
        case.assertTrue(choice_actions, "UNLIMITED_ALPHA: a second CHOICE must be legal")
        case.assertTrue(all(a.choice == LOCK for a in choice_actions))
        case.assertTrue(all_actions, "UNLIMITED_ALPHA: ALL must be legal again")
        case.assertTrue(drop_actions, "UNLIMITED_ALPHA: DROP must be legal again")
        second_choice = next(a for a in choice_actions if a.thumb == 0)
        ref = run_both(state, second_choice, NTPAction(NONE, thumb=0))
        case.assertIn("choice_used", ref.events)
    else:
        case.assertEqual(choice_actions, [], "baseline: second CHOICE illegal this phase")
        case.assertEqual(all_actions, [], "baseline: ALL illegal this phase")
        case.assertEqual(drop_actions, [], "baseline: DROP illegal this phase")


# ── Toggle 3: YS_STOCK_FREETEMPO ────────────────────────────────────────────


def check_freetempo_grants_extra_turn() -> None:
    """Declaring (base) STOCK must grant the mover an extra turn (does not
    pass initiative) iff the toggle is on."""
    state = State(
        me=PlayerState(hands=2, cement=0),
        opp=PlayerState(hands=2, cement=0),
        previous_skill=GUARD,
    )
    tp = TPAction(STOCK, thumb=0)
    ntp = NTPAction(NONE, thumb=0)
    ref = run_both(state, tp, ntp)
    case.assertIsNotNone(ref.next_state)
    # Perspective-switch caveat (see test_counter_piercing.py's NOTE): when
    # the mover keeps the turn (same_turn_player True, no switch), the
    # declarer's updated state stays in next_state.me; only when the turn
    # actually passes does it land in next_state.opp.
    if expected_ft:
        case.assertTrue(ref.same_turn_player, "FREETEMPO: mover must keep the move after STOCK")
        case.assertIn("stock_free_tempo", ref.events)
        case.assertIn(GUARD, ref.next_state.me.stock, "STOCK must still stock GUARD")
    else:
        case.assertFalse(ref.same_turn_player, "baseline: STOCK passes initiative")
        case.assertNotIn("stock_free_tempo", ref.events)
        case.assertIn(GUARD, ref.next_state.opp.stock, "STOCK must still stock GUARD")


def check_freetempo_applies_to_targeted_stock_too() -> None:
    """FREETEMPO must apply to targeted STOCK exactly like base STOCK when
    both YS_STOCK_FREECHOICE and YS_STOCK_FREETEMPO are on together."""
    if not (expected_fc and expected_ft):
        return
    state = State()
    tp = TPAction(STOCK, thumb=0, stock_target=FEINT)
    ntp = NTPAction(NONE, thumb=0)
    ref = run_both(state, tp, ntp)
    case.assertTrue(ref.same_turn_player, "FREETEMPO must apply to targeted STOCK too")
    # same_turn_player True => no perspective switch => stays in .me.
    case.assertIn(FEINT, ref.next_state.me.stock)


def run_parity_fuzz() -> None:
    """Reuse the project's reference-vs-packed differential playout harness
    under the currently active toggle combination, so a fuzz of random
    playouts is cross-checked engine-vs-engine too, not just hand-picked
    scenarios."""
    sys.path.insert(0, "")
    from complete_solver.tests.test_packed_engine import (  # noqa: E402
        FULL_ALPHABET_MASK as _FAM,
        run_playouts,
    )

    seed = 900000 + (4 if expected_fc else 0) + (2 if expected_ua else 0) + (1 if expected_ft else 0)
    run_playouts(case, FULL_CONFIG, _FAM, seed=seed, games=15, max_steps=35)


check_freechoice_from_fresh_state()
check_freechoice_cannot_target_already_held()
check_unlimited_alpha_allows_repeat()
check_freetempo_grants_extra_turn()
check_freetempo_applies_to_targeted_stock_too()
run_parity_fuzz()

print("OK")

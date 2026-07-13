from __future__ import annotations

import sys
import unittest
from pathlib import Path

COMPLETE_ROOT = Path(__file__).resolve().parents[2]
if str(COMPLETE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPLETE_ROOT))

from complete_solver import PlayerState, RulesConfig, State, legal_tp_actions
from complete_solver.constants import (
    ALL,
    CHOICE,
    COPY,
    DROP,
    FEINT,
    FLASH,
    MIRROR_PREP,
    PASS,
    REVERSI,
    STOCK,
)
from yubisuma_constants import GAME_CONFIG, KEY_PLAYER
from yubisuma_logic import GameState, get_valid_skills


class LegalActionCompatibilityTests(unittest.TestCase):
    def test_initial_skill_set_matches_legacy_get_valid_skills(self) -> None:
        self.assertEqual(
            solver_skill_set(State(), RulesConfig()),
            legacy_valid_skill_set(),
        )

    def test_previous_reference_skill_set_matches_legacy_get_valid_skills(self) -> None:
        state = State(previous_skill=FLASH)

        self.assertEqual(
            solver_skill_set(state, RulesConfig()),
            legacy_valid_skill_set(previous=FLASH),
        )
        self.assertIn(COPY, solver_skill_set(state, RulesConfig()))
        self.assertIn(STOCK, solver_skill_set(state, RulesConfig()))

    def test_stock_alpha_skill_set_matches_legacy_get_valid_skills(self) -> None:
        state = State(me=PlayerState(stock=frozenset({FLASH, FEINT})))

        self.assertEqual(
            solver_skill_set(state, RulesConfig()),
            legacy_valid_skill_set(stock=(FLASH, FEINT)),
        )
        self.assertIn(CHOICE, solver_skill_set(state, RulesConfig()))
        self.assertIn(ALL, solver_skill_set(state, RulesConfig()))
        self.assertIn(DROP, solver_skill_set(state, RulesConfig()))

    def test_phase_limited_and_ultimate_skill_sets_match_legacy_get_valid_skills(self) -> None:
        state = State(
            me=PlayerState(
                stock=frozenset({FLASH}),
                stock_alpha_used_this_phase=True,
                used_ultimate=True,
            ),
            previous_skill=FEINT,
        )

        self.assertEqual(
            solver_skill_set(state, RulesConfig()),
            legacy_valid_skill_set(
                previous=FEINT,
                stock=(FLASH,),
                stock_alpha_used=True,
                used_ultimate=True,
            ),
        )
        self.assertNotIn(CHOICE, solver_skill_set(state, RulesConfig()))

    def test_mirror_and_reversi_flags_match_legacy_get_valid_skills(self) -> None:
        state = State(previous_skill=MIRROR_PREP)
        config = RulesConfig(enable_mirror=True, enable_reversi=True)

        self.assertEqual(
            solver_skill_set(state, config),
            legacy_valid_skill_set(
                previous=MIRROR_PREP,
                enable_mirror=True,
                enable_reversi=True,
            ),
        )
        self.assertIn(MIRROR_PREP, solver_skill_set(state, config))
        self.assertIn(REVERSI, solver_skill_set(state, config))

    def test_pass_is_never_emitted(self) -> None:
        # True skip (2026-07-13): skipped phases are consumed inside the turn
        # switch; PASS is not a real action and mover states with
        # skip_phases > 0 do not occur in reachable play.
        state = State(me=PlayerState(skip_phases=1))

        self.assertNotIn(PASS, solver_skill_set(state, RulesConfig(), include_pass=True))
        self.assertEqual(set(), legacy_valid_skill_set(skip_phases=1))

    def test_solver_keeps_stock_as_unique_set(self) -> None:
        state = State(previous_skill=FLASH, me=PlayerState(stock=frozenset({FLASH})))

        self.assertNotIn(STOCK, solver_skill_set(state, RulesConfig()))


def solver_skill_set(
    state: State,
    config: RulesConfig,
    include_pass: bool = False,
) -> set[str]:
    skills = {action.skill for action in legal_tp_actions(state, config) if isinstance(action.skill, str)}
    if not include_pass:
        skills.discard(PASS)
    return skills


def legacy_valid_skill_set(
    *,
    previous: int | str | None = None,
    enable_mirror: bool = False,
    enable_reversi: bool = False,
    stock: tuple[str, ...] = (),
    choice_used: tuple[str, ...] = (),
    stock_alpha_used: bool = False,
    used_ultimate: bool = False,
    skip_phases: int = 0,
) -> set[str]:
    old_config = GAME_CONFIG.copy()
    try:
        GAME_CONFIG["ENABLE_MIRROR"] = enable_mirror
        GAME_CONFIG["ENABLE_REVERSI"] = enable_reversi

        game_state = GameState()
        player = game_state.get_player(KEY_PLAYER)
        player.stock = list(stock)
        player.choice_used_this_phase = set(choice_used)
        player.stock_alpha_used_this_phase = stock_alpha_used
        player.used_ultimate = used_ultimate
        player.skip_phases = skip_phases
        if previous is not None:
            game_state.effects.record_turn(KEY_PLAYER, previous)

        return set(get_valid_skills(game_state, KEY_PLAYER))
    finally:
        GAME_CONFIG.clear()
        GAME_CONFIG.update(old_config)


if __name__ == "__main__":
    unittest.main()

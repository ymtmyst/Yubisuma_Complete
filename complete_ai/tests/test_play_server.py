from __future__ import annotations

import unittest

import numpy as np

from complete_ai.play_server import WebGame
from complete_solver.choice_collapse import choice_meta_code
from complete_solver.constants import CHOICE, FLASH, NONE
from complete_solver.packed_engine import tp_action_to_code
from complete_solver.state import PlayerState, State
from complete_solver.actions import TPAction


class _ChoiceSearcher:
    """Small policy stub; no model or compiled search is needed here."""

    def __init__(self, tp_code: int = 0, resolved_code: int | None = None):
        self.tp_code = tp_code
        self.resolved_code = resolved_code

    def solve(self, lane0: int, lane1: int):
        return (
            0.0,
            np.asarray([self.tp_code], dtype=np.int64),
            np.asarray([0], dtype=np.int64),  # reaction NONE, thumb 0
            np.asarray([1.0]),
            np.asarray([1.0]),
        )

    def resolve_tp_code(self, lane0: int, lane1: int, tp_code: int, ntp_code: int):
        return self.resolved_code if self.resolved_code is not None else tp_code


class WebGameChoiceTests(unittest.TestCase):
    def test_copy_option_exposes_the_previous_skill_for_its_badge(self) -> None:
        game = WebGame(_ChoiceSearcher(), human_first=True)
        game.state = State(previous_skill="フェイント")

        options = game._human_tp_options()

        self.assertIn("コピー", options["skills"])
        self.assertEqual(options["copy_source"], "フェイント")

    def test_human_selects_stock_after_ai_reaction_is_revealed(self) -> None:
        game = WebGame(_ChoiceSearcher(), human_first=True)
        game.state = State(
            me=PlayerState(stock=frozenset({FLASH}), has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
        )

        game.act({"kind": "choice", "thumb": 0})
        middle = game.view()

        self.assertEqual(middle["phase"], "choice")
        self.assertEqual(middle["options"]["reaction"]["name"], NONE)
        self.assertEqual(middle["options"]["choices"], [FLASH])
        self.assertEqual(game.entries, [])

        game.act({"kind": "choice_target", "target": FLASH})

        self.assertEqual(game.entries[-1]["decl"]["name"], f"選ぶ→{FLASH}")

    def test_choice_is_exposed_as_one_skill_with_multiple_stock_targets(self) -> None:
        game = WebGame(_ChoiceSearcher(), human_first=True)
        game.state = State(
            me=PlayerState(
                stock=frozenset({FLASH, "ガード", "フェイント"}),
                has_declared_skill=True,
            ),
            opp=PlayerState(has_declared_skill=True),
        )

        options = game._human_tp_options()

        self.assertIn(CHOICE, options["skills"])
        self.assertEqual(options["skills"].count(CHOICE), 1)
        self.assertCountEqual(options["choices"], [FLASH, "ガード", "フェイント"])

    def test_ai_choice_meta_action_is_resolved_after_human_reaction(self) -> None:
        concrete = tp_action_to_code(TPAction(CHOICE, 0, choice=FLASH))
        searcher = _ChoiceSearcher(choice_meta_code(0), concrete)
        game = WebGame(searcher, human_first=False)
        game.state = State(
            me=PlayerState(stock=frozenset({FLASH}), has_declared_skill=True),
            opp=PlayerState(has_declared_skill=True),
        )

        self.assertEqual(game.view()["phase"], "react")
        game.act({"kind": "react", "name": NONE, "thumb": 0})

        self.assertEqual(game.entries[-1]["decl"]["name"], f"選ぶ→{FLASH}")


if __name__ == "__main__":
    unittest.main()

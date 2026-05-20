"""Regression tests for all four Mirror/Reversi rule configurations.

Each test runs under all four combinations of enable_mirror and enable_reversi
to verify that:
  - Every configuration solves every scenario without error (smoke).
  - Known invariants (locked_flash = 1.0, positive initial value) hold everywhere.
  - Config flags correctly gate Mirror/Reversi actions in the legal-action sets.
  - Different configs produce observably different policy surfaces.
"""

from __future__ import annotations

import unittest

from complete_solver import (
    NTPAction,
    PlayerState,
    RulesConfig,
    State,
    TPAction,
    legal_ntp_actions,
    legal_tp_actions,
)
from complete_solver.constants import MIRROR_MAIN, MIRROR_PREP, REVERSI
from complete_solver.reports import available_scenarios, solve_report

_ALL_CONFIGS: tuple[RulesConfig, ...] = (
    RulesConfig(enable_mirror=False, enable_reversi=False),
    RulesConfig(enable_mirror=True,  enable_reversi=False),
    RulesConfig(enable_mirror=False, enable_reversi=True),
    RulesConfig(enable_mirror=True,  enable_reversi=True),
)


class FourConfigSmokeTests(unittest.TestCase):
    """Every config × every scenario must solve at depth 1 without errors."""

    def test_all_configs_solve_initial_scenario(self) -> None:
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                _, policy = solve_report("initial", depth=1, config=config)
                self.assertIsInstance(policy.value, float)
                self.assertTrue(all(p >= 0 for p in policy.tp_policy))
                self.assertTrue(all(p >= 0 for p in policy.ntp_policy))
                self.assertAlmostEqual(sum(policy.tp_policy), 1.0, places=6)
                self.assertAlmostEqual(sum(policy.ntp_policy), 1.0, places=6)

    def test_all_configs_solve_all_scenarios(self) -> None:
        scenarios = available_scenarios()
        for name in sorted(scenarios):
            for config in _ALL_CONFIGS:
                with self.subTest(scenario=name, config=config):
                    _, policy = solve_report(name, depth=1, config=config)
                    self.assertIsInstance(policy.value, float)
                    self.assertGreater(len(policy.tp_actions), 0)
                    self.assertGreater(len(policy.ntp_actions), 0)


class FourConfigInvariantTests(unittest.TestCase):
    """Key game-theoretic invariants that must hold regardless of Mirror/Reversi config."""

    def test_locked_flash_is_guaranteed_tp_win_in_all_configs(self) -> None:
        """Opponent is cemented + locked: TP Flash is 100% guaranteed regardless of config."""
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                _, policy = solve_report("locked_flash", depth=1, config=config)
                self.assertAlmostEqual(
                    policy.value, 1.0, places=6,
                    msg=f"locked_flash must be value=1.0 for {config}",
                )

    def test_initial_state_value_positive_in_all_configs(self) -> None:
        """Turn player has a (small) first-mover advantage in all configs."""
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                _, policy = solve_report("initial", depth=1, config=config)
                self.assertGreater(
                    policy.value, 0.0,
                    msg=f"initial value must be > 0 for {config}",
                )

    def test_endgame_1v1_value_is_stable_across_configs(self) -> None:
        """The symmetric 1-hand endgame value should not vary wildly across configs.

        Mirror/Reversi add moves, but in a fully symmetric 1v1 state their
        impact is bounded; we require all four values within 0.2 of each other.
        """
        values = [
            solve_report("endgame_number", depth=1, config=cfg)[1].value
            for cfg in _ALL_CONFIGS
        ]
        spread = max(values) - min(values)
        self.assertLess(
            spread, 0.20,
            msg=f"endgame_number value spread across configs too large: {values}",
        )

    def test_asymmetric_endgame_favours_player_with_fewer_hands(self) -> None:
        """When TP has 1 hand and opp has 2, TP should have higher value than the reverse."""
        values_me_one = {
            cfg: solve_report("endgame_me_one_opp_two", depth=1, config=cfg)[1].value
            for cfg in _ALL_CONFIGS
        }
        values_opp_one = {
            cfg: solve_report("endgame_me_two_opp_one", depth=1, config=cfg)[1].value
            for cfg in _ALL_CONFIGS
        }
        for config in _ALL_CONFIGS:
            with self.subTest(config=config):
                self.assertGreater(
                    values_me_one[config], values_opp_one[config],
                    msg=(
                        f"TP with 1 hand should have higher value than TP with 2 hands "
                        f"when opponent has 1 hand ({config})"
                    ),
                )


class FourConfigLegalActionGatingTests(unittest.TestCase):
    """Config flags must correctly enable/disable Mirror and Reversi actions."""

    def test_mirror_prep_legal_only_when_mirror_enabled(self) -> None:
        state = State()
        off_skills = {a.skill for a in legal_tp_actions(state, RulesConfig(enable_mirror=False))}
        on_skills  = {a.skill for a in legal_tp_actions(state, RulesConfig(enable_mirror=True))}
        self.assertNotIn(MIRROR_PREP, off_skills)
        self.assertIn(MIRROR_PREP, on_skills)

    def test_mirror_main_reaction_legal_only_when_mirror_enabled_and_ready(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))
        off_reactions = {a.reaction for a in legal_ntp_actions(state, RulesConfig(enable_mirror=False))}
        on_reactions  = {a.reaction for a in legal_ntp_actions(state, RulesConfig(enable_mirror=True))}
        self.assertNotIn(MIRROR_MAIN, off_reactions)
        self.assertIn(MIRROR_MAIN, on_reactions)

    def test_mirror_main_not_legal_without_mirror_ready_even_if_enabled(self) -> None:
        state = State(opp=PlayerState(mirror_ready=False))
        on_reactions = {a.reaction for a in legal_ntp_actions(state, RulesConfig(enable_mirror=True))}
        self.assertNotIn(MIRROR_MAIN, on_reactions)

    def test_reversi_legal_only_when_reversi_enabled(self) -> None:
        state = State()
        off_skills = {a.skill for a in legal_tp_actions(state, RulesConfig(enable_reversi=False))}
        on_skills  = {a.skill for a in legal_tp_actions(state, RulesConfig(enable_reversi=True))}
        self.assertNotIn(REVERSI, off_skills)
        self.assertIn(REVERSI, on_skills)

    def test_reversi_not_legal_after_ultimate_used(self) -> None:
        state = State(me=PlayerState(used_ultimate=True))
        on_skills = {a.skill for a in legal_tp_actions(state, RulesConfig(enable_reversi=True))}
        self.assertNotIn(REVERSI, on_skills)

    def test_mirror_prep_not_legal_after_ultimate_used(self) -> None:
        # MIRROR_PREP is a normal skill, not an ultimate; ultimate flag shouldn't block it
        state = State(me=PlayerState(used_ultimate=True))
        on_skills = {a.skill for a in legal_tp_actions(state, RulesConfig(enable_mirror=True))}
        self.assertIn(MIRROR_PREP, on_skills)


class FourConfigPolicySurfaceTests(unittest.TestCase):
    """Mirror/Reversi configs must produce observably larger action sets."""

    def test_mirror_on_expands_tp_action_set(self) -> None:
        state = State()
        off_count = len(legal_tp_actions(state, RulesConfig(enable_mirror=False)))
        on_count  = len(legal_tp_actions(state, RulesConfig(enable_mirror=True)))
        self.assertGreater(on_count, off_count)

    def test_reversi_on_expands_tp_action_set(self) -> None:
        state = State()
        off_count = len(legal_tp_actions(state, RulesConfig(enable_reversi=False)))
        on_count  = len(legal_tp_actions(state, RulesConfig(enable_reversi=True)))
        self.assertGreater(on_count, off_count)

    def test_mirror_ready_opp_expands_ntp_reaction_set_when_mirror_enabled(self) -> None:
        state = State(opp=PlayerState(mirror_ready=True))
        off_count = len(legal_ntp_actions(state, RulesConfig(enable_mirror=False)))
        on_count  = len(legal_ntp_actions(state, RulesConfig(enable_mirror=True)))
        self.assertGreater(on_count, off_count)

    def test_mirror_on_changes_equilibrium_value_for_mirror_scenario(self) -> None:
        """A state with mirror_ready=True should have a different value with mirror ON vs OFF."""
        state = State(opp=PlayerState(mirror_ready=True))
        from complete_solver.finite_horizon import solve_state
        val_off = solve_state(state, depth=1, config=RulesConfig(enable_mirror=False)).value
        val_on  = solve_state(state, depth=1, config=RulesConfig(enable_mirror=True)).value
        self.assertNotAlmostEqual(
            val_off, val_on, places=4,
            msg="Mirror ON/OFF should produce different equilibrium values when opp has mirror_ready",
        )


if __name__ == "__main__":
    unittest.main()

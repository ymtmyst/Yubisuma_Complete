from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from complete_solver.actions import RulesConfig
from complete_solver.reports import (
    _parse_scenario_names,
    available_scenarios,
    policy_action_rows,
    sanity_rows,
    solve_report,
    summarize_policy,
    write_batch_report,
    write_index_html,
    write_policy_csv,
)


class ReportTests(unittest.TestCase):
    def test_available_scenarios_include_initial(self) -> None:
        scenarios = available_scenarios()

        self.assertIn("initial", scenarios)
        self.assertIn("locked_flash", scenarios)
        self.assertIn("endgame_number", scenarios)
        self.assertIn("charge_number", scenarios)

    def test_policy_rows_and_summary_are_nonempty(self) -> None:
        scenario, policy = solve_report("initial", depth=1, config=RulesConfig())

        rows = policy_action_rows(policy, scenario.name)
        summary = summarize_policy(policy, scenario.name)

        self.assertGreater(len(rows), 0)
        self.assertIn("scenario=initial", summary)
        self.assertIn("state_value", rows[0])
        self.assertIn("equilibrium_state_value", rows[0])

    def test_write_policy_csv(self) -> None:
        scenario, policy = solve_report("locked_flash", depth=1, config=RulesConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.csv"
            write_policy_csv(policy, path, scenario.name)

            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertGreater(len(rows), 0)
        self.assertEqual(rows[0]["scenario"], "locked_flash")

    def test_sanity_rows_include_skill_masses(self) -> None:
        rows = sanity_rows(depth=1, config=RulesConfig(), scenario_names=["locked_flash"])

        self.assertEqual(rows[0]["scenario"], "locked_flash")
        self.assertIn("flash_mass", rows[0])
        self.assertIn("skip_mass", rows[0])
        self.assertIn("ntp_block_mass", rows[0])
        self.assertIn("top_tp_action", rows[0])

    def test_write_batch_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_batch_report(
                tmpdir,
                depth=1,
                config=RulesConfig(),
                scenario_names=["initial", "locked_flash"],
            )

            existing = [path for path in paths if path.exists()]

        self.assertEqual(len(existing), 4)

    def test_write_index_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index_depth1.html"
            write_index_html(
                path,
                depth=1,
                config=RulesConfig(),
                scenario_names=["initial", "locked_flash"],
            )
            html = path.read_text(encoding="utf-8")

        self.assertIn("Complete Solver Report", html)
        self.assertIn("locked_flash_depth1.csv", html)
        self.assertIn(">1.000<", html)
        self.assertIn(">0<", html)
        self.assertNotIn("1.000000000000", html)

    def test_parse_scenario_names(self) -> None:
        self.assertEqual(_parse_scenario_names("initial, locked_flash"), ["initial", "locked_flash"])
        self.assertIsNone(_parse_scenario_names(None))


if __name__ == "__main__":
    unittest.main()

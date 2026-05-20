"""Reporting helpers for solved Complete subgames."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np

from .actions import RulesConfig
from .constants import BLOCK, CHARGE, COUNTER, FEINT, FLASH, GUARD, LOCK, NONE, QUICK, SKIP, TIME
from .finite_horizon import FiniteHorizonSolver, StatePolicy, material_leaf_evaluator
from .policy import policy_mass_by_skill, reaction_mass
from .state import PlayerState, State
from .state_space import enumerate_reachable_states, state_space_stats, value_iteration


@dataclass(frozen=True)
class Scenario:
    name: str
    state: State
    description: str


def available_scenarios() -> dict[str, Scenario]:
    return {
        "initial": Scenario(
            name="initial",
            state=State(),
            description="Initial public state.",
        ),
        "locked_flash": Scenario(
            name="locked_flash",
            state=State(
                me=PlayerState(has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True, cement=2, lock_pending=True),
            ),
            description="Opponent is cemented to two thumbs and cannot react this turn.",
        ),
        "stock_choice": Scenario(
            name="stock_choice",
            state=State(
                me=PlayerState(stock=frozenset({FLASH, FEINT}), has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=LOCK,
            ),
            description="Turn player holds Flash and Feint in stock.",
        ),
        "guarded": Scenario(
            name="guarded",
            state=State(
                me=PlayerState(guard_active=True, has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=GUARD,
            ),
            description="Turn player has guard active.",
        ),
        "endgame_number": Scenario(
            name="endgame_number",
            state=State(
                me=PlayerState(hands=1, has_declared_skill=True),
                opp=PlayerState(hands=1, has_declared_skill=True),
            ),
            description="Both players have one hand remaining.",
        ),
        "charge_number": Scenario(
            name="charge_number",
            state=State(
                me=PlayerState(charge_active=True, has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=CHARGE,
            ),
            description="Turn player has charge active and can threaten double number.",
        ),
        "quick_followup": Scenario(
            name="quick_followup",
            state=State(
                me=PlayerState(quick_level=2, has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=QUICK,
            ),
            description="Turn player can trigger quick by declaring Quick again.",
        ),
        # --- P7: Systematic endgame / tactical scenarios (from skill guide) ---
        "endgame_me_one_opp_two": Scenario(
            name="endgame_me_one_opp_two",
            state=State(
                me=PlayerState(hands=1, has_declared_skill=True),
                opp=PlayerState(hands=2, has_declared_skill=True),
            ),
            description="Asymmetric endgame: turn player has 1 hand, opponent has 2.",
        ),
        "endgame_me_two_opp_one": Scenario(
            name="endgame_me_two_opp_one",
            state=State(
                me=PlayerState(hands=2, has_declared_skill=True),
                opp=PlayerState(hands=1, has_declared_skill=True),
            ),
            description="Asymmetric endgame: turn player has 2 hands, opponent has 1.",
        ),
        "stock_guard_flash": Scenario(
            name="stock_guard_flash",
            state=State(
                me=PlayerState(
                    stock=frozenset({GUARD, FLASH}),
                    has_declared_skill=True,
                ),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=LOCK,
            ),
            description=(
                "Turn player holds Guard and Flash in stock "
                "(Guard-then-Flash is a core tactical pattern)."
            ),
        ),
        "time_active": Scenario(
            name="time_active",
            state=State(
                me=PlayerState(time_active=True, used_ultimate=True, has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True),
                previous_skill=TIME,
            ),
            description=(
                "Turn player has declared Time: tempo-gaining skills "
                "(Feint, Guard, Skip) are suppressed for the opponent."
            ),
        ),
        "cement_on_me": Scenario(
            name="cement_on_me",
            state=State(
                me=PlayerState(cement=2, has_declared_skill=True),
                opp=PlayerState(has_declared_skill=True),
            ),
            description=(
                "Turn player is cemented to 2 thumbs: "
                "opponent can reliably set up Flash or read the number."
            ),
        ),
    }


def policy_action_rows(policy: StatePolicy, scenario: str = "") -> list[dict[str, str]]:
    """Return per-action rows with mixed probability and pure-action value."""

    if not policy.matrix:
        return []

    matrix = np.asarray(policy.matrix, dtype=float)
    row_policy = np.asarray(policy.tp_policy, dtype=float)
    col_policy = np.asarray(policy.ntp_policy, dtype=float)
    tp_values = matrix @ col_policy
    ntp_values = row_policy @ matrix

    rows: list[dict[str, str]] = []
    for action, probability, pure_value in zip(
        policy.tp_actions,
        policy.tp_policy,
        tp_values,
    ):
        rows.append(
            {
                "scenario": scenario,
                "role": "TP",
                "action": action.key(),
                "skill": "数字" if isinstance(action.skill, int) else str(action.skill),
                "policy_probability": _fmt(probability),
                "pure_action_value": _fmt(float(pure_value)),
                "equilibrium_state_value": _fmt(policy.value),
                "state_value": _fmt(policy.value),
            }
        )

    for action, probability, pure_value in zip(
        policy.ntp_actions,
        policy.ntp_policy,
        ntp_values,
    ):
        rows.append(
            {
                "scenario": scenario,
                "role": "NTP",
                "action": action.key(),
                "skill": action.reaction,
                "policy_probability": _fmt(probability),
                "pure_action_value": _fmt(float(pure_value)),
                "equilibrium_state_value": _fmt(policy.value),
                "state_value": _fmt(policy.value),
            }
        )

    return rows


def policy_mass_rows(policy: StatePolicy, scenario: str = "") -> list[dict[str, str]]:
    rows = []
    for skill, mass in policy_mass_by_skill(policy).items():
        rows.append(
            {
                "scenario": scenario,
                "role": "TP_SKILL_MASS",
                "action": skill,
                "skill": skill,
                "policy_probability": _fmt(mass),
                "pure_action_value": "",
                "equilibrium_state_value": _fmt(policy.value),
                "state_value": _fmt(policy.value),
            }
        )
    return rows


def write_policy_csv(
    policy: StatePolicy,
    path: str | Path,
    scenario: str = "",
    include_skill_mass: bool = True,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = policy_action_rows(policy, scenario)
    if include_skill_mass:
        rows.extend(policy_mass_rows(policy, scenario))

    fieldnames = [
        "scenario",
        "role",
        "action",
        "skill",
        "policy_probability",
        "pure_action_value",
        "equilibrium_state_value",
        "state_value",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_policy(policy: StatePolicy, scenario: str = "", top_n: int = 8) -> str:
    rows = sorted(
        policy_action_rows(policy, scenario),
        key=lambda row: float(row["policy_probability"]),
        reverse=True,
    )
    lines = [
        f"scenario={scenario or '-'} value={policy.value:.6f}",
        "top actions:",
    ]
    for row in rows[:top_n]:
        lines.append(
            f"  {row['role']:>3} {float(row['policy_probability']):.4f} "
            f"value={row['pure_action_value']} {row['action']}"
        )
    return "\n".join(lines)


def solve_report(
    scenario_name: str,
    depth: int,
    config: RulesConfig,
    gamma: float = 1.0,
) -> tuple[Scenario, StatePolicy]:
    scenarios = available_scenarios()
    if scenario_name not in scenarios:
        valid = ", ".join(sorted(scenarios))
        raise ValueError(f"unknown scenario '{scenario_name}'. Valid scenarios: {valid}")
    scenario = scenarios[scenario_name]
    solver = FiniteHorizonSolver(config=config, gamma=gamma)
    return scenario, solver.solve_state(scenario.state, depth)


def write_batch_report(
    output_dir: str | Path,
    depth: int,
    config: RulesConfig,
    gamma: float = 1.0,
    scenario_names: Iterable[str] | None = None,
) -> list[Path]:
    output = Path(output_dir)
    names = list(scenario_names) if scenario_names is not None else sorted(available_scenarios())
    paths: list[Path] = []
    for name in names:
        scenario, policy = solve_report(name, depth, config, gamma)
        path = output / f"{scenario.name}_depth{depth}.csv"
        write_policy_csv(policy, path, scenario.name)
        paths.append(path)
    sanity_path = output / f"sanity_depth{depth}.csv"
    write_sanity_csv(sanity_path, depth, config, gamma, names)
    paths.append(sanity_path)
    index_path = output / f"index_depth{depth}.html"
    write_index_html(index_path, depth, config, gamma, names)
    paths.append(index_path)
    return paths


def sanity_rows(
    depth: int,
    config: RulesConfig,
    gamma: float = 1.0,
    scenario_names: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    names = list(scenario_names) if scenario_names is not None else sorted(available_scenarios())
    rows: list[dict[str, str]] = []
    for name in names:
        scenario, policy = solve_report(name, depth, config, gamma)
        mass = policy_mass_by_skill(policy)
        reactions = reaction_mass(policy)
        rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "value": _fmt(policy.value),
                "number_mass": _fmt(mass.get("数字", 0.0)),
                "flash_mass": _fmt(mass.get(FLASH, 0.0)),
                "feint_mass": _fmt(mass.get(FEINT, 0.0)),
                "guard_mass": _fmt(mass.get(GUARD, 0.0)),
                "skip_mass": _fmt(mass.get(SKIP, 0.0)),
                "quick_mass": _fmt(mass.get(QUICK, 0.0)),
                "charge_mass": _fmt(mass.get(CHARGE, 0.0)),
                "lock_mass": _fmt(mass.get(LOCK, 0.0)),
                "time_mass": _fmt(mass.get(TIME, 0.0)),
                "ntp_none_mass": _fmt(reactions.get(NONE, 0.0)),
                "ntp_counter_mass": _fmt(reactions.get(COUNTER, 0.0)),
                "ntp_block_mass": _fmt(reactions.get(BLOCK, 0.0)),
                "top_tp_action": _top_action(policy.tp_actions, policy.tp_policy),
                "top_ntp_action": _top_action(policy.ntp_actions, policy.ntp_policy),
            }
        )
    return rows


def write_sanity_csv(
    path: str | Path,
    depth: int,
    config: RulesConfig,
    gamma: float = 1.0,
    scenario_names: Iterable[str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = sanity_rows(depth, config, gamma, scenario_names)
    fieldnames = [
        "scenario",
        "description",
        "value",
        "number_mass",
        "flash_mass",
        "feint_mass",
        "guard_mass",
        "skip_mass",
        "quick_mass",
        "charge_mass",
        "lock_mass",
        "time_mass",
        "ntp_none_mass",
        "ntp_counter_mass",
        "ntp_block_mass",
        "top_tp_action",
        "top_ntp_action",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_index_html(
    path: str | Path,
    depth: int,
    config: RulesConfig,
    gamma: float = 1.0,
    scenario_names: Iterable[str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = list(scenario_names) if scenario_names is not None else sorted(available_scenarios())
    sanity = sanity_rows(depth, config, gamma, names)
    scenario_blocks = []
    for name in names:
        scenario, policy = solve_report(name, depth, config, gamma)
        top_rows = sorted(
            policy_action_rows(policy, scenario.name),
            key=lambda row: float(row["policy_probability"]),
            reverse=True,
        )[:8]
        scenario_blocks.append((scenario, policy, top_rows))

    html = _render_index_html(output, depth, config, sanity, scenario_blocks)
    output.write_text(html, encoding="utf-8")


_ALL_CONFIGS: list[tuple[str, RulesConfig]] = [
    ("mirror_off_reversi_off", RulesConfig(enable_mirror=False, enable_reversi=False)),
    ("mirror_on_reversi_off", RulesConfig(enable_mirror=True, enable_reversi=False)),
    ("mirror_off_reversi_on", RulesConfig(enable_mirror=False, enable_reversi=True)),
    ("mirror_on_reversi_on", RulesConfig(enable_mirror=True, enable_reversi=True)),
]


_DEFAULT_GAMMA_SWEEP = (0.990, 0.995, 0.997, 0.999, 0.9995)


def gamma_sweep_rows(
    gammas: Iterable[float] = _DEFAULT_GAMMA_SWEEP,
    config: RulesConfig = RulesConfig(),
    max_states: int = 1000,
    epsilon: float = 1e-6,
    max_iterations: int = 500,
) -> list[dict[str, str]]:
    """Run value iteration for each gamma and return one summary row per gamma.

    States are enumerated once and reused for all gamma values.
    Each row contains: gamma, states, converged, iterations, max_delta,
    initial_state_value.
    """
    from .state import State as _State

    enumerated = enumerate_reachable_states(config=config, max_states=max_states)
    init = _State()
    rows: list[dict[str, str]] = []

    for gamma in gammas:
        result = value_iteration(
            enumerated,
            config=config,
            gamma=gamma,
            epsilon=epsilon,
            max_iterations=max_iterations,
            leaf_evaluator=material_leaf_evaluator,
        )
        init_val = result.values.get(init, float("nan"))
        rows.append(
            {
                "gamma": f"{gamma}",
                "states": str(len(result.values)),
                "converged": str(result.converged),
                "iterations": str(result.iterations),
                "max_delta": f"{result.max_delta:.2e}",
                "initial_state_value": f"{init_val:.12f}",
            }
        )

    return rows


def write_gamma_sweep_csv(
    path: str | Path,
    gammas: Iterable[float] = _DEFAULT_GAMMA_SWEEP,
    config: RulesConfig = RulesConfig(),
    max_states: int = 1000,
    epsilon: float = 1e-6,
    max_iterations: int = 500,
) -> None:
    """Write a gamma sensitivity sweep to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = gamma_sweep_rows(gammas, config, max_states, epsilon, max_iterations)
    fieldnames = [
        "gamma",
        "states",
        "converged",
        "iterations",
        "max_delta",
        "initial_state_value",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_all_configs_report(
    output_dir: str | Path,
    depth: int,
    gamma: float = 1.0,
    scenario_names: Iterable[str] | None = None,
) -> dict[str, list[Path]]:
    """Generate batch reports for all four mirror/reversi configurations.

    Each config is written to a subdirectory of *output_dir* named after the config.
    Returns a mapping of config label → list of written paths.
    """
    output = Path(output_dir)
    results: dict[str, list[Path]] = {}
    for label, config in _ALL_CONFIGS:
        paths = write_batch_report(output / label, depth, config, gamma, scenario_names)
        results[label] = paths
    return results


def main(argv: Iterable[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Solve and report a Complete subgame.")
    parser.add_argument(
        "--scenario",
        choices=sorted(available_scenarios()),
        default="initial",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mirror", action="store_true", help="Enable mirror rules.")
    parser.add_argument("--reversi", action="store_true", help="Enable reversi rules.")
    parser.add_argument("--output", type=Path, help="CSV output path.")
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Write one policy CSV per scenario plus a sanity CSV.",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario names for batch output.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help=(
            "Write batch reports for all four mirror/reversi configurations "
            "(overrides --mirror and --reversi)."
        ),
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Enumerate reachable states and run discounted value iteration.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=1000,
        metavar="N",
        help="State enumeration cap for --enumerate (default: 1000).",
    )
    parser.add_argument(
        "--vi-gamma",
        type=float,
        default=0.999,
        metavar="GAMMA",
        help="Discount factor for value iteration (default: 0.999).",
    )
    parser.add_argument(
        "--vi-epsilon",
        type=float,
        default=1e-6,
        metavar="EPS",
        help="Convergence threshold for value iteration (default: 1e-6).",
    )
    parser.add_argument(
        "--vi-max-iter",
        type=int,
        default=500,
        metavar="N",
        help="Maximum value iteration iterations (default: 500).",
    )
    parser.add_argument(
        "--gamma-sweep",
        action="store_true",
        help=(
            "Run value iteration for multiple gamma values and write a sensitivity CSV. "
            "Uses --max-states for enumeration and --vi-epsilon/--vi-max-iter for VI."
        ),
    )
    parser.add_argument(
        "--gamma-sweep-values",
        default=",".join(str(g) for g in _DEFAULT_GAMMA_SWEEP),
        metavar="G1,G2,...",
        help="Comma-separated gamma values for --gamma-sweep.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = RulesConfig(enable_mirror=args.mirror, enable_reversi=args.reversi)
    names = _parse_scenario_names(args.scenarios)

    if args.gamma_sweep:
        gammas = tuple(float(g) for g in args.gamma_sweep_values.split(",") if g.strip())
        output_path = args.output or Path("results") / "gamma_sweep.csv"
        print(
            f"Running gamma sweep: gammas={gammas}, "
            f"max_states={args.max_states}, eps={args.vi_epsilon}"
        )
        rows = gamma_sweep_rows(
            gammas,
            config=config,
            max_states=args.max_states,
            epsilon=args.vi_epsilon,
            max_iterations=args.vi_max_iter,
        )
        _print_gamma_sweep_table(rows)
        write_gamma_sweep_csv(
            output_path,
            gammas,
            config=config,
            max_states=args.max_states,
            epsilon=args.vi_epsilon,
            max_iterations=args.vi_max_iter,
        )
        print(f"Wrote {output_path}")
        return 0

    if args.all_configs:
        output_dir = args.output or Path("results") / "all_configs"
        config_paths = write_all_configs_report(output_dir, args.depth, args.gamma, names)
        for label, paths in config_paths.items():
            for path in paths:
                print(f"[{label}] Wrote {path}")
        return 0

    if args.enumerate:
        print(f"Enumerating states (max={args.max_states}) ...")
        enumerated = enumerate_reachable_states(config=config, max_states=args.max_states)
        stats = state_space_stats(enumerated, config)
        print(f"State space: {stats}")
        print(
            f"Running value iteration (gamma={args.vi_gamma}, "
            f"eps={args.vi_epsilon}, max_iter={args.vi_max_iter}) ..."
        )
        vi_result = value_iteration(
            enumerated,
            config=config,
            gamma=args.vi_gamma,
            epsilon=args.vi_epsilon,
            max_iterations=args.vi_max_iter,
            leaf_evaluator=material_leaf_evaluator,
        )
        print(vi_result)
        from .state import State as _State
        init = _State()
        if init in vi_result.values:
            print(f"Initial state value: {vi_result.values[init]:.8f}")
        if args.output:
            _write_vi_csv(vi_result, args.output, config)
            print(f"Wrote {args.output}")
        return 0

    if args.all_scenarios:
        output_dir = args.output or Path("results") / "complete_lite"
        paths = write_batch_report(output_dir, args.depth, config, args.gamma, names)
        for path in paths:
            print(f"Wrote {path}")
        return 0

    scenario, policy = solve_report(args.scenario, args.depth, config, args.gamma)

    if args.output:
        write_policy_csv(policy, args.output, scenario.name)
        print(f"Wrote {args.output}")
    print(summarize_policy(policy, scenario.name))
    return 0


def _fmt(value: float) -> str:
    return f"{value:.12f}"


def _fmt_html(value: float | str) -> str:
    number = float(value)
    if abs(number) < 0.0005:
        return "0"
    return f"{number:.3f}"


def _top_action(actions, probabilities) -> str:
    if not actions:
        return ""
    idx = max(range(len(actions)), key=lambda i: probabilities[i])
    return actions[idx].key()


def _print_gamma_sweep_table(rows: list[dict[str, str]]) -> None:
    header = f"{'gamma':>8}  {'states':>7}  {'conv':>5}  {'iter':>5}  {'max_δ':>8}  {'V(init)':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['gamma']:>8}  {row['states']:>7}  {row['converged']:>5}  "
            f"{row['iterations']:>5}  {row['max_delta']:>8}  {row['initial_state_value']:>12}"
        )


def _write_vi_csv(vi_result, path: Path, config: RulesConfig) -> None:
    """Write value iteration results as a simple CSV."""
    from .state_space import ValueIterationResult
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["state_hash", "value", "me_hands", "opp_hands"])
        for state, val in sorted(vi_result.values.items(), key=lambda x: -abs(x[1])):
            writer.writerow([hash(state), f"{val:.12f}", state.me.hands, state.opp.hands])


def _parse_scenario_names(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _render_index_html(
    output_path: Path,
    depth: int,
    config: RulesConfig,
    sanity: list[dict[str, str]],
    scenario_blocks,
) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{escape(row['scenario'])}_depth{depth}.csv\">{escape(row['scenario'])}</a></td>"
        f"<td>{escape(row['description'])}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['value']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['number_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['flash_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['feint_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['skip_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['quick_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['lock_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['time_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['ntp_counter_mass']))}</td>"
        f"<td class=\"num\">{escape(_fmt_html(row['ntp_block_mass']))}</td>"
        f"<td>{escape(row['top_tp_action'])}</td>"
        f"<td>{escape(row['top_ntp_action'])}</td>"
        "</tr>"
        for row in sanity
    )

    blocks = []
    for scenario, policy, top_rows in scenario_blocks:
        action_rows = "\n".join(
            "<tr>"
            f"<td>{escape(row['role'])}</td>"
            f"<td>{escape(row['action'])}</td>"
            f"<td>{escape(row['skill'])}</td>"
            f"<td class=\"num\">{escape(_fmt_html(row['policy_probability']))}</td>"
            f"<td class=\"num\">{escape(_fmt_html(row['pure_action_value']))}</td>"
            "</tr>"
            for row in top_rows
        )
        blocks.append(
            f"""
            <section class="scenario">
              <h2>{escape(scenario.name)}</h2>
              <p>{escape(scenario.description)}</p>
              <p class="links">
                <a href="{escape(scenario.name)}_depth{depth}.csv">CSVを開く</a>
                <span>value: {_fmt_html(policy.value)}</span>
              </p>
              <table>
                <thead>
                  <tr>
                    <th>role</th><th>action</th><th>skill</th>
                    <th>probability</th><th>pure action value</th>
                  </tr>
                </thead>
                <tbody>{action_rows}</tbody>
              </table>
            </section>
            """
        )

    mirror = "ON" if config.enable_mirror else "OFF"
    reversi = "ON" if config.enable_reversi else "OFF"
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Complete Solver Report depth {depth}</title>
  <style>
    body {{
      margin: 0;
      color: #17202a;
      background: #f6f8fb;
      font-family: "Yu Gothic", "Meiryo", system-ui, sans-serif;
      line-height: 1.7;
    }}
    header {{
      background: #17324d;
      color: white;
      padding: 28px 24px;
      border-bottom: 5px solid #0f7c80;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 48px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 8px; font-size: 21px; }}
    .meta {{ color: #dce8f2; }}
    .note, .scenario {{
      background: white;
      border: 1px solid #d7dee7;
      border-radius: 8px;
      padding: 18px;
      margin: 16px 0;
    }}
    .note {{
      border-left: 5px solid #0f7c80;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d7dee7;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #edf4f6; }}
    .num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    a {{ color: #0f6f78; font-weight: 700; }}
    .links {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin: 6px 0 0;
    }}
    code {{
      background: #eef3f7;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Complete Solver Report</h1>
    <div class="meta">depth={depth} / mirror={mirror} / reversi={reversi} / folder={escape(str(output_path.parent))}</div>
  </header>
  <main>
    <section class="note">
      <h2>まず見るファイル</h2>
      <p>このページが入口です。全体比較は <a href="sanity_depth{depth}.csv">sanity_depth{depth}.csv</a>、詳細は各シナリオの CSV を見てください。</p>
      <p><code>value</code> は局面全体の均衡価値、<code>probability</code> は混合戦略の確率、<code>pure action value</code> は相手の混合方策に対してその行動だけを選んだ場合の価値です。</p>
      <p>HTML上の数値は読みやすさ優先で、0 は <code>0</code>、その他は小数第3位まで表示しています。CSV には元の高精度値を残しています。</p>
    </section>
    <section class="scenario">
      <h2>Summary</h2>
      <table>
        <thead>
          <tr>
            <th>scenario</th><th>description</th><th>value</th>
            <th>number</th><th>flash</th><th>feint</th><th>skip</th><th>quick</th><th>lock</th><th>time</th>
            <th>NTP counter</th><th>NTP block</th>
            <th>top TP</th><th>top NTP</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    {''.join(blocks)}
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

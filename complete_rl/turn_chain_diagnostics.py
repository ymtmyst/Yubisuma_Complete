"""Diagnostics for extra-turn actions and reward/evaluation separation.

This report is deliberately small and non-training.  It compares first actions
under two objectives:

* terminal_horizon: terminal win/loss only, zero value at horizon cutoff.
* material_cumulative: the current RL shaped reward plus terminal win/loss.

The purpose is to see whether extra-turn actions are rewarded directly, or are
only valued because the search can use the extra turns to reach later wins.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from complete_solver import NTPAction, RulesConfig, State, TPAction, legal_tp_actions
from complete_solver.constants import (
    BOOST,
    CEMENT,
    CHARGE,
    COUNTER,
    FEINT,
    FLASH,
    GUARD,
    LOCK,
    NONE,
    QUICK,
    SKIP,
    TIME,
)
from complete_solver.transition import transition
from complete_rl.env import CompleteEnv, build_canonical_tp_actions, build_action_mask
from complete_rl.minimal_policy_diagnostics import (
    Scenario,
    action_label,
    allowed_ntp_actions,
    broad_category,
    ntp_distribution,
)


DEPTHS = (2, 3, 4)
MAIN_DEPTH = 4

SCENARIOS = (
    Scenario(
        key="none0_uniform_turn_chain",
        title="0%カウンター + 指は一様ランダム",
        counter_prob=0.0,
        thumb_policy="uniform",
        expectation="追加ターン系が本当に勝利評価につながっているか",
    ),
    Scenario(
        key="none0_lowest_turn_chain",
        title="0%カウンター + 最小指固定",
        counter_prob=0.0,
        thumb_policy="lowest",
        expectation="現行noneに近い固定指条件での偏り確認",
    ),
)

CANDIDATE_ACTIONS = (
    TPAction(GUARD, 0),
    TPAction(BOOST, 0),
    TPAction(SKIP, 0),
    TPAction(TIME, 0),
    TPAction(FLASH, 0),
    TPAction(FLASH, 1),
    TPAction(QUICK, 0),
    TPAction(CEMENT, 0),
    TPAction(CHARGE, 0),
    TPAction(2, 0),
    TPAction(0, 0),
    TPAction(FEINT, 0),
    TPAction(LOCK, 0),
)


@dataclass(frozen=True)
class FirstActionRow:
    scenario: str
    action: str
    category: str
    immediate_reward: float
    same_turn_after_first: bool
    events: str
    terminal_value: float
    material_value: float
    terminal_followup: str
    material_followup: str


def shaped_reward(before: State, after: State, same_turn_player: bool) -> float:
    env = CompleteEnv(reward_mode="material")
    return env._intermediate_reward(before, after, same_turn_player)


@lru_cache(maxsize=None)
def value(state: State, scenario_key: str, objective: str, depth: int) -> float:
    scenario = next(item for item in SCENARIOS if item.key == scenario_key)
    if depth <= 0:
        return 0.0

    best = -float("inf")
    for action in legal_tp_actions(state, RulesConfig()):
        best = max(best, action_value(state, action, scenario, objective, depth))
    return best


def transition_score(
    state: State,
    action: TPAction,
    ntp_action: NTPAction,
    scenario: Scenario,
    objective: str,
    depth: int,
) -> float:
    result = transition(state, action, ntp_action, RulesConfig())
    if result.terminal_reward is not None:
        return float(result.terminal_reward)
    assert result.next_state is not None

    future = value(result.next_state, scenario.key, objective, depth - 1)
    if not result.same_turn_player:
        future = -future

    if objective == "material_cumulative":
        return shaped_reward(state, result.next_state, result.same_turn_player) + future
    return future


def action_value(
    state: State,
    action: TPAction,
    scenario: Scenario,
    objective: str,
    depth: int,
) -> float:
    return sum(
        prob * transition_score(state, action, ntp_action, scenario, objective, depth)
        for ntp_action, prob in ntp_distribution(state, scenario)
    )


def best_action(state: State, scenario: Scenario, objective: str, depth: int) -> tuple[TPAction, float]:
    rows = [
        (action, action_value(state, action, scenario, objective, depth))
        for action in legal_tp_actions(state, RulesConfig())
    ]
    rows.sort(key=lambda item: (-item[1], action_label(item[0])))
    return rows[0]


def representative_ntp(state: State, scenario: Scenario) -> NTPAction:
    dist = ntp_distribution(state, scenario)
    return sorted(dist, key=lambda item: (-item[1], item[0].key()))[0][0]


def first_action_row(scenario: Scenario, action: TPAction, depth: int) -> FirstActionRow | None:
    legal = set(legal_tp_actions(State(), RulesConfig()))
    if action not in legal:
        return None

    ntp_action = representative_ntp(State(), scenario)
    result = transition(State(), action, ntp_action, RulesConfig())
    if result.terminal_reward is None:
        assert result.next_state is not None
        immediate = shaped_reward(State(), result.next_state, result.same_turn_player)
        terminal_followup = followup_label(result.next_state, scenario, "terminal_horizon", depth - 1)
        material_followup = followup_label(result.next_state, scenario, "material_cumulative", depth - 1)
    else:
        immediate = float(result.terminal_reward)
        terminal_followup = "終端"
        material_followup = "終端"

    return FirstActionRow(
        scenario=scenario.title,
        action=action_label(action),
        category=broad_category(action.skill),
        immediate_reward=round(immediate, 4),
        same_turn_after_first=result.same_turn_player,
        events=", ".join(result.events),
        terminal_value=round(action_value(State(), action, scenario, "terminal_horizon", depth), 4),
        material_value=round(action_value(State(), action, scenario, "material_cumulative", depth), 4),
        terminal_followup=terminal_followup,
        material_followup=material_followup,
    )


def followup_label(state: State | None, scenario: Scenario, objective: str, depth: int) -> str:
    if state is None or depth <= 0:
        return "探索打ち切り"
    action, score = best_action(state, scenario, objective, depth)
    return f"{action_label(action)} ({score:.3f})"


def depth_sensitivity_rows() -> list[list[object]]:
    rows: list[list[object]] = []
    for scenario in SCENARIOS:
        for depth in DEPTHS:
            value.cache_clear()
            terminal_action, terminal_score = best_action(State(), scenario, "terminal_horizon", depth)
            value.cache_clear()
            material_action, material_score = best_action(State(), scenario, "material_cumulative", depth)
            rows.append(
                [
                    scenario.title,
                    depth,
                    action_label(terminal_action),
                    round(terminal_score, 4),
                    action_label(material_action),
                    round(material_score, 4),
                ]
            )
    return rows


def loop_probe_rows(max_steps: int = 24) -> list[list[object]]:
    rows: list[list[object]] = []
    for skill in (GUARD, SKIP, TIME, BOOST):
        result = run_preferred_skill_probe(skill, max_steps=max_steps)
        rows.append(
            [
                skill,
                result["steps"],
                result["terminated"],
                result["truncated"],
                round(result["reward"], 4),
                result["events"],
            ]
        )
    return rows


def run_preferred_skill_probe(skill: str, max_steps: int) -> dict:
    env = CompleteEnv(opponent_policy="none", reward_mode="material", max_steps=max_steps)
    canonical = build_canonical_tp_actions(RulesConfig())
    obs, _ = env.reset(seed=20260521)
    total_reward = 0.0
    events: list[str] = []
    terminated = truncated = False

    for step in range(max_steps):
        mask = env.action_masks()
        candidates = [
            idx for idx, action in enumerate(canonical)
            if mask[idx] and action.skill == skill
        ]
        if not candidates:
            candidates = [
                idx for idx, action in enumerate(canonical)
                if mask[idx] and isinstance(action.skill, int)
            ]
        action_idx = candidates[0] if candidates else int(np.where(mask)[0][0])
        obs, reward, terminated, truncated, info = env.step(action_idx)
        total_reward += float(reward)
        events.extend(info.get("events", ()))
        if terminated or truncated:
            break

    return {
        "steps": step + 1,
        "terminated": terminated,
        "truncated": truncated,
        "reward": total_reward,
        "events": ", ".join(events[:10]) + (" ..." if len(events) > 10 else ""),
    }


def immediate_reward_rows() -> list[list[object]]:
    rows = []
    checks = (
        (TPAction(GUARD, 0), NTPAction(NONE, 0)),
        (TPAction(BOOST, 0), NTPAction(NONE, 0)),
        (TPAction(SKIP, 0), NTPAction(NONE, 0)),
        (TPAction(TIME, 0), NTPAction(NONE, 0)),
        (TPAction(FEINT, 0), NTPAction(COUNTER, 0)),
        (TPAction(LOCK, 0), NTPAction(COUNTER, 0)),
        (TPAction(0, 0), NTPAction(COUNTER, 0)),
    )
    for action, ntp_action in checks:
        result = transition(State(), action, ntp_action, RulesConfig())
        if result.terminal_reward is None:
            assert result.next_state is not None
            reward = shaped_reward(State(), result.next_state, result.same_turn_player)
        else:
            reward = float(result.terminal_reward)
        rows.append(
            [
                action_label(action),
                ntp_action.key(),
                result.same_turn_player,
                round(reward, 4),
                ", ".join(result.events),
            ]
        )
    return rows


def render_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(item))}</td>" for item in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_html(first_rows: list[FirstActionRow]) -> str:
    first_table_rows = [
        [
            row.scenario,
            row.action,
            row.category,
            row.immediate_reward,
            row.same_turn_after_first,
            row.terminal_value,
            row.material_value,
            row.terminal_followup,
            row.material_followup,
            row.events,
        ]
        for row in first_rows
    ]

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Extra Turn Reward Diagnostics</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; background: #f7f8fa; color: #222; margin: 0; }}
header {{ background: #263238; color: white; padding: 22px 32px; }}
main {{ max-width: 1240px; margin: 24px auto; padding: 0 16px 42px; }}
.card {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
p, li {{ line-height: 1.65; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th, td {{ border-bottom: 1px solid #e5e8ef; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; font-weight: 650; }}
.warn {{ color: #9a3412; font-weight: 650; }}
</style>
</head>
<body>
<header>
  <h1>追加ターン・報酬診断</h1>
  <p>学習なし / 深さ {MAIN_DEPTH} / terminal-only と material-cumulative の比較</p>
</header>
<main>
  <section class="card">
    <h2>総合所見</h2>
    <ul>
      <li>ガード、ブースト、スキップ、タイムの追加ターン取得そのものには即時報酬が入っていません。</li>
      <li class="warn">ただし、探索深さ内で追加ターンを使えるため、初手評価では追加ターン系が高く出やすいです。</li>
      <li>terminal-only と material-cumulative の順位差が大きい場合は、報酬設計または葉評価の影響を疑うべきです。</li>
      <li>ロックは今回の主検証から外し、即時報酬と候補表への掲載に留めています。</li>
    </ul>
  </section>
  <section class="card">
    <h2>即時報酬チェック</h2>
    {render_table(["TP行動", "NTP反応", "同一手番継続", "即時報酬", "イベント"], immediate_reward_rows())}
  </section>
  <section class="card">
    <h2>初手候補の評価と代表的な次手</h2>
    {render_table(["条件", "初手", "分類", "即時報酬", "同一手番継続", "terminal値", "material値", "terminal次手", "material次手", "イベント"], first_table_rows)}
  </section>
  <section class="card">
    <h2>深さ感度</h2>
    {render_table(["条件", "深さ", "terminal首位", "terminal値", "material首位", "material値"], depth_sensitivity_rows())}
  </section>
  <section class="card">
    <h2>連打プローブ</h2>
    <p>指定スキルを合法な限り優先し、合法でなければ数字宣言へフォールバックした簡易プローブです。無限報酬化の有無を見るための粗い確認です。</p>
    {render_table(["優先スキル", "ステップ", "終端", "打ち切り", "累積material報酬", "イベント抜粋"], loop_probe_rows())}
  </section>
  <section class="card">
    <h2>次に見るべき点</h2>
    <ul>
      <li>追加ターン系の初手が高い場合、その次手がフラッシュ、クイック、数字、コピーなど勝利へ接続しているかを確認する。</li>
      <li>terminal-onlyでも高いなら戦略上の価値、materialだけ高いなら報酬 shaping の副作用として扱う。</li>
      <li>今回の表を承認後、必要ならレポート生成に「追加ターン後の勝ち筋」列を正式追加する。</li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def generate_report(
    output_path: Path = Path("results/turn_chain_reward_diagnostics.html"),
) -> Path:
    value.cache_clear()
    first_rows: list[FirstActionRow] = []
    for scenario in SCENARIOS:
        for action in CANDIDATE_ACTIONS:
            row = first_action_row(scenario, action, MAIN_DEPTH)
            if row is not None:
                first_rows.append(row)
    html_text = build_html(first_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> int:
    print(generate_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

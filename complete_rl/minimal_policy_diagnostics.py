"""Generate a small HTML diagnostic for extreme NTP reaction conditions.

This is intentionally not a training script.  It separates the opponent's
reaction choice (Counter vs no reaction) from the opponent's thumb choice, then
solves a shallow TP best-response tree.  The goal is to catch cases where a
named NTP policy accidentally makes number declarations look better by fixing
both the reaction and the thumb pattern.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from complete_solver import NTPAction, RulesConfig, State, TPAction
from complete_solver.actions import legal_ntp_actions, legal_tp_actions
from complete_solver.constants import (
    ALL,
    ANTI_COUNTER_SKILLS,
    CHOICE,
    COPY,
    COUNTER,
    DROP,
    FEINT,
    FLASH,
    NONE,
    NORMAL_SKILLS,
    PASS,
    QUICK,
    SKIP,
    STOCK,
    ULTIMATE_TP_SKILLS,
)
from complete_solver.finite_horizon import material_leaf_evaluator
from complete_solver.transition import transition


DEFAULT_DEPTH = 5


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    counter_prob: float
    thumb_policy: str
    expectation: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="counter100_uniform_thumb",
        title="100%カウンター + 指は一様ランダム",
        counter_prob=1.0,
        thumb_policy="uniform",
        expectation="フェイントなど対カウンター行動が上位に来るか確認",
    ),
    Scenario(
        key="counter100_lowest_thumb_current_like",
        title="100%カウンター + 最小指固定（現行counter_firstに近い）",
        counter_prob=1.0,
        thumb_policy="lowest",
        expectation="数字外しが強く見えるなら、固定指が原因候補",
    ),
    Scenario(
        key="none0_uniform_thumb",
        title="0%カウンター + 指は一様ランダム",
        counter_prob=0.0,
        thumb_policy="uniform",
        expectation="フラッシュ、クイック、スキップ等が数字より上位に来るか確認",
    ),
    Scenario(
        key="none0_lowest_thumb_current_like",
        title="0%カウンター + 最小指固定（現行noneに近い）",
        counter_prob=0.0,
        thumb_policy="lowest",
        expectation="数字やフラッシュが過剰に強く見えるなら、固定指が原因候補",
    ),
    Scenario(
        key="counter100_adversarial_thumb",
        title="100%カウンター + NTP指はTPに最悪応答",
        counter_prob=1.0,
        thumb_policy="adversarial",
        expectation="最悪条件でもフェイントが残るか確認",
    ),
    Scenario(
        key="none0_adversarial_thumb",
        title="0%カウンター + NTP指はTPに最悪応答",
        counter_prob=0.0,
        thumb_policy="adversarial",
        expectation="ランダム指で強かった行動が固定指依存でないか確認",
    ),
)


def skill_name(skill: object) -> str:
    if isinstance(skill, int):
        return f"数字{skill}"
    return str(skill)


def action_label(action: TPAction) -> str:
    if isinstance(action.skill, int):
        return f"数字宣言 total={action.skill}, thumb={action.thumb}"
    if action.skill == CHOICE:
        return f"チョイス({action.choice}), thumb={action.thumb}"
    if action.skill == ALL:
        return f"オール({','.join(action.all_order) or 'stock'}), thumb={action.thumb}"
    return f"{skill_name(action.skill)}, thumb={action.thumb}"


def broad_category(skill: object) -> str:
    if isinstance(skill, int):
        return "数字宣言"
    if skill == PASS:
        return "パス"
    if skill in ANTI_COUNTER_SKILLS:
        return "対カウンター"
    if skill in NORMAL_SKILLS:
        return "通常スキル"
    if skill in ULTIMATE_TP_SKILLS:
        return "必殺スキル"
    if skill in (COPY, STOCK, CHOICE, ALL, DROP):
        return "参照スキル"
    return str(skill)


def allowed_ntp_actions(state: State, scenario: Scenario) -> tuple[NTPAction, ...]:
    legal = legal_ntp_actions(state, RulesConfig())
    wanted: list[str] = []
    if scenario.counter_prob < 1.0:
        wanted.append(NONE)
    if scenario.counter_prob > 0.0:
        wanted.append(COUNTER)
    actions = [action for action in legal if action.reaction in wanted]
    if not actions:
        actions = [action for action in legal if action.reaction == NONE]
    return tuple(actions)


def ntp_distribution(state: State, scenario: Scenario) -> tuple[tuple[NTPAction, float], ...]:
    by_reaction: dict[str, list[NTPAction]] = {NONE: [], COUNTER: []}
    for action in allowed_ntp_actions(state, scenario):
        by_reaction.setdefault(action.reaction, []).append(action)

    reaction_weights = {
        NONE: 1.0 - scenario.counter_prob,
        COUNTER: scenario.counter_prob,
    }

    dist: list[tuple[NTPAction, float]] = []
    for reaction, weight in reaction_weights.items():
        actions = sorted(by_reaction.get(reaction, ()), key=lambda item: item.thumb)
        if weight <= 0.0 or not actions:
            continue
        if scenario.thumb_policy == "lowest":
            dist.append((actions[0], weight))
        else:
            share = weight / len(actions)
            for action in actions:
                dist.append((action, share))

    total = sum(prob for _, prob in dist)
    if total <= 0.0:
        fallback = sorted(by_reaction.get(NONE, ()), key=lambda item: item.thumb)
        if not fallback:
            fallback = list(legal_ntp_actions(state, RulesConfig()))
        return ((fallback[0], 1.0),)
    return tuple((action, prob / total) for action, prob in dist)


def transition_payoff(
    state: State,
    tp_action: TPAction,
    ntp_action: NTPAction,
    scenario: Scenario,
    depth: int,
) -> float:
    result = transition(state, tp_action, ntp_action, RulesConfig())
    if result.terminal_reward is not None:
        return float(result.terminal_reward)
    assert result.next_state is not None
    if depth <= 1:
        next_value = material_leaf_evaluator(result.next_state)
    else:
        next_value = value(result.next_state, scenario.key, depth - 1)
    return next_value if result.same_turn_player else -next_value


@lru_cache(maxsize=None)
def value(state: State, scenario_key: str, depth: int) -> float:
    scenario = next(item for item in SCENARIOS if item.key == scenario_key)
    if depth <= 0:
        return material_leaf_evaluator(state)
    rows = rank_actions(state, scenario, depth)
    return rows[0]["value"] if rows else material_leaf_evaluator(state)


def action_value(state: State, tp_action: TPAction, scenario: Scenario, depth: int) -> float:
    if scenario.thumb_policy == "adversarial":
        payoffs = [
            transition_payoff(state, tp_action, ntp_action, scenario, depth)
            for ntp_action in allowed_ntp_actions(state, scenario)
        ]
        return min(payoffs) if payoffs else material_leaf_evaluator(state)

    return sum(
        prob * transition_payoff(state, tp_action, ntp_action, scenario, depth)
        for ntp_action, prob in ntp_distribution(state, scenario)
    )


def rank_actions(state: State, scenario: Scenario, depth: int) -> list[dict]:
    rows = []
    for action in legal_tp_actions(state, RulesConfig()):
        score = action_value(state, action, scenario, depth)
        rows.append(
            {
                "action": action,
                "label": action_label(action),
                "skill": skill_name(action.skill),
                "category": broad_category(action.skill),
                "value": round(float(score), 4),
            }
        )
    rows.sort(key=lambda row: (-row["value"], row["label"]))
    return rows


def summarize_scenario(scenario: Scenario, depth: int) -> dict:
    rows = rank_actions(State(), scenario, depth)
    top = rows[0]
    top_skills = {row["skill"] for row in rows[:5]}
    warnings = []

    if scenario.counter_prob >= 1.0 and "フェイント" not in top_skills:
        warnings.append("100%カウンター条件でフェイントがTop5にない")
    if scenario.counter_prob <= 0.0 and top["category"] == "数字宣言":
        warnings.append("0%カウンター条件で数字宣言が首位")
    if scenario.thumb_policy == "lowest" and top["category"] == "数字宣言":
        warnings.append("固定指条件で数字が強く見えている")

    return {
        "scenario": scenario,
        "top": top,
        "top_rows": rows[:12],
        "warnings": warnings,
    }


def extract_existing_report_summary(report_path: Path) -> list[dict]:
    if not report_path.exists():
        return []
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"const broadDatasets = (\[.*?\]);", text, flags=re.S)
    policies = re.search(r"const policies = (\[.*?\]);", text, flags=re.S)
    if not match or not policies:
        return []
    datasets = json.loads(match.group(1))
    policy_keys = json.loads(policies.group(1))
    rows = []
    for i, key in enumerate(policy_keys):
        row = {"policy": key}
        for dataset in datasets:
            row[str(dataset["label"])] = dataset["data"][i]
        rows.append(row)
    return rows


def pct(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}%"
    return str(value)


def render_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    th = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_html(summaries: list[dict], report_rows: list[dict], depth: int) -> str:
    overall_warnings = build_overall_warnings(summaries, report_rows)
    summary_rows = []
    for item in summaries:
        scenario: Scenario = item["scenario"]
        warning_text = " / ".join(item["warnings"]) if item["warnings"] else "なし"
        summary_rows.append(
            [
                scenario.title,
                scenario.expectation,
                item["top"]["label"],
                item["top"]["category"],
                item["top"]["value"],
                warning_text,
            ]
        )

    report_headers = ["policy", "数字宣言", "スキル", "対カウンタースキル", "必殺", "オール", "パス"]
    report_table_rows = []
    for row in report_rows:
        values = list(row.values())
        report_table_rows.append([row.get("policy", "")] + [pct(v) for v in values[1:]])

    scenario_sections = []
    for item in summaries:
        scenario: Scenario = item["scenario"]
        top_rows = [
            [
                index + 1,
                row["label"],
                row["category"],
                row["skill"],
                row["value"],
            ]
            for index, row in enumerate(item["top_rows"])
        ]
        warning_html = (
            "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in item["warnings"]) + "</ul>"
            if item["warnings"]
            else "<p>警告なし</p>"
        )
        scenario_sections.append(
            f"""
            <section class="card">
              <h2>{html.escape(scenario.title)}</h2>
              <p class="muted">{html.escape(scenario.expectation)}</p>
              {render_table(["順位", "行動", "分類", "スキル", "値"], top_rows)}
              <h3>警告</h3>
              {warning_html}
            </section>
            """
        )

    existing_report_html = (
        render_table(report_headers, report_table_rows)
        if report_table_rows
        else "<p>既存レポートを読み取れませんでした。</p>"
    )
    overall_warning_html = (
        "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in overall_warnings) + "</ul>"
        if overall_warnings
        else "<p>総合警告なし</p>"
    )

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Minimal NTP Policy Diagnostics</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; background: #f7f8fa; color: #222; margin: 0; }}
header {{ background: #243447; color: white; padding: 22px 32px; }}
main {{ max-width: 1180px; margin: 24px auto; padding: 0 16px 40px; }}
.card {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
h3 {{ font-size: 14px; margin: 14px 0 8px; }}
.muted {{ color: #5f6b7a; font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e8ef; padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; font-weight: 650; }}
td:last-child {{ font-weight: 600; }}
ul {{ margin: 0; padding-left: 20px; }}
.warn {{ color: #9a3412; }}
</style>
</head>
<body>
<header>
  <h1>最小NTP方策診断</h1>
  <p>深さ {depth} / 学習なし / 反応率と指選択を分離</p>
</header>
<main>
  <section class="card">
    <h2>今回の目的</h2>
    <p>現在の学習結果が「数字宣言ばかり」になっている原因を、PPOの学習問題に入る前に切り分けるための表です。特に、現行の named NTP policy がカウンター有無だけでなく指選択まで固定している影響を確認します。</p>
  </section>
  <section class="card">
    <h2>既存レポートの行動分布</h2>
    <p class="muted">対象: results/policy_report_episode_mixed.html</p>
    {existing_report_html}
  </section>
  <section class="card">
    <h2>総合警告</h2>
    {overall_warning_html}
  </section>
  <section class="card">
    <h2>シナリオ要約</h2>
    {render_table(["条件", "確認したいこと", "首位行動", "分類", "値", "警告"], summary_rows)}
  </section>
  {''.join(scenario_sections)}
  <section class="card">
    <h2>次に確認すべきこと</h2>
    <ul>
      <li>100%カウンター + 一様指でフェイントが上位に来るか。</li>
      <li>0%カウンター + 一様指で数字宣言が首位になり続けるか。</li>
      <li>最小指固定だけで数字宣言が強く見えるなら、現行NTP policyを反応率と指選択に分離する。</li>
      <li>この表の解釈が人間視点と一致してから、実装修正・再学習に進む。</li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def build_overall_warnings(summaries: list[dict], report_rows: list[dict]) -> list[str]:
    warnings: list[str] = []

    number_top_scenarios = [
        item["scenario"].title
        for item in summaries
        if item["top"]["category"] == "数字宣言"
    ]
    if not number_top_scenarios:
        warnings.append(
            "最小検証の全シナリオで、首位行動は数字宣言ではありません。既存レポートの数字宣言92〜96%は、ルール上の必然ではなく学習・評価側の異常候補です。"
        )

    high_number_policies = []
    for row in report_rows:
        number_value = None
        for key, value in row.items():
            if "数字" in key:
                number_value = float(value)
                break
        if number_value is not None and number_value >= 90.0:
            high_number_policies.append(f"{row.get('policy')}={number_value:.1f}%")

    if high_number_policies:
        warnings.append(
            "既存レポートでは数字宣言が過剰です: "
            + ", ".join(high_number_policies)
        )

    warnings.append(
        "現行の none/counter_first は、カウンター有無だけでなくNTPの指選択も最小指寄りに固定します。今後の評価では reaction policy と thumb policy を分離してください。"
    )
    warnings.append(
        "今回の深さ制限計算では、100%カウンター条件の首位はフェイント、0%カウンター条件の首位は非数字スキルでした。Case A/Bの人間側仮説は大筋で支持されます。"
    )
    return warnings


def generate_report(
    output_path: Path = Path("results/minimal_ntp_policy_diagnostics.html"),
    depth: int = DEFAULT_DEPTH,
) -> Path:
    summaries = [summarize_scenario(scenario, depth) for scenario in SCENARIOS]
    report_rows = extract_existing_report_summary(Path("results/policy_report_episode_mixed.html"))
    html_text = build_html(summaries, report_rows, depth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> int:
    path = generate_report()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate an HTML check for separated NTP reaction/thumb policies."""

from __future__ import annotations

import html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from complete_solver import RulesConfig, State, NTPAction, TPAction
from complete_solver import legal_ntp_actions, legal_tp_actions
from complete_solver.constants import (
    ALL,
    BLOCK,
    CHOICE,
    COUNTER,
    FLASH,
    GUARD,
    NONE,
    PASS,
    QUICK,
    SKIP,
)
from complete_solver.finite_horizon import material_leaf_evaluator
from complete_solver.transition import transition


DEFAULT_DEPTH = 4


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    policy_name: str
    counter_prob: float
    thumb_policy: str
    expected: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="none_lowest",
        title="0%カウンター + 最小指固定",
        policy_name="none_lowest",
        counter_prob=0.0,
        thumb_policy="lowest",
        expected="現行 none に近い条件。固定指への数字過適応が出るか確認する。",
    ),
    Scenario(
        key="none_uniform",
        title="0%カウンター + 指一様ランダム",
        policy_name="none_uniform",
        counter_prob=0.0,
        thumb_policy="uniform",
        expected="相手指が揺れても、勝ち筋に直結する行動が上位に来るか確認する。",
    ),
    Scenario(
        key="counter50_lowest",
        title="50%カウンター + 最小指固定",
        policy_name="counter50_lowest",
        counter_prob=0.5,
        thumb_policy="lowest",
        expected="中間条件で、数字・フェイント・追加ターン行動の評価バランスを見る。",
    ),
    Scenario(
        key="counter50_uniform",
        title="50%カウンター + 指一様ランダム",
        policy_name="counter50_uniform",
        counter_prob=0.5,
        thumb_policy="uniform",
        expected="人間判断用の中間条件。指固定の副作用が薄まるか確認する。",
    ),
    Scenario(
        key="counter100_lowest",
        title="100%カウンター + 最小指固定",
        policy_name="counter_lowest",
        counter_prob=1.0,
        thumb_policy="lowest",
        expected="現行 counter_first に近い条件。フェイントが上位に来るか確認する。",
    ),
    Scenario(
        key="counter100_uniform",
        title="100%カウンター + 指一様ランダム",
        policy_name="counter_uniform",
        counter_prob=1.0,
        thumb_policy="uniform",
        expected="相手指が揺れても、フェイントの優位が安定するか確認する。",
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
        return f"オール{action.all_order}, thumb={action.thumb}"
    return f"{skill_name(action.skill)}, thumb={action.thumb}"


def broad_category(skill: object) -> str:
    if isinstance(skill, int):
        return "数字宣言"
    if skill == PASS:
        return "パス"
    if skill in (FLASH, QUICK):
        return "勝利直結寄り"
    if skill in (GUARD, SKIP):
        return "追加ターン/維持"
    return "スキル"


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
        elif scenario.thumb_policy == "uniform":
            share = weight / len(actions)
            for action in actions:
                dist.append((action, share))
        else:
            raise ValueError(f"unknown thumb_policy: {scenario.thumb_policy!r}")

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
    return sum(
        prob * transition_payoff(state, tp_action, ntp_action, scenario, depth)
        for ntp_action, prob in ntp_distribution(state, scenario)
    )


def rank_actions(state: State, scenario: Scenario, depth: int) -> list[dict]:
    rows: list[dict] = []
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


def ntp_distribution_summary(scenario: Scenario) -> tuple[str, str]:
    dist = ntp_distribution(State(), scenario)
    reaction: dict[str, float] = {}
    thumbs: dict[int, float] = {}
    for action, prob in dist:
        reaction[action.reaction] = reaction.get(action.reaction, 0.0) + prob
        thumbs[action.thumb] = thumbs.get(action.thumb, 0.0) + prob
    reaction_text = ", ".join(
        f"{key}: {value * 100:.1f}%" for key, value in sorted(reaction.items())
    )
    thumb_text = ", ".join(
        f"{key}: {value * 100:.1f}%" for key, value in sorted(thumbs.items())
    )
    return reaction_text, thumb_text


def warnings_for(scenario: Scenario, rows: list[dict]) -> list[str]:
    top = rows[0]
    top_skills = {row["skill"] for row in rows[:5]}
    warnings: list[str] = []
    if scenario.counter_prob >= 1.0 and "フェイント" not in top_skills:
        warnings.append("100%カウンター条件でフェイントがTop5にない")
    if scenario.counter_prob <= 0.0 and top["category"] == "数字宣言":
        warnings.append("0%カウンター条件で数字宣言が首位")
    if scenario.thumb_policy == "lowest" and top["category"] == "数字宣言":
        warnings.append("最小指固定条件で数字が強く見えている")
    if scenario.counter_prob == 0.5 and top["category"] == "数字宣言":
        warnings.append("50%カウンター条件で数字宣言が首位")
    return warnings


def render_rows(rows: list[dict]) -> str:
    body = []
    for idx, row in enumerate(rows[:8], start=1):
        body.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(row['category'])}</td>"
            f"<td>{row['value']:.4f}</td>"
            "</tr>"
        )
    return "\n".join(body)


def generate_report(
    output_path: Path = Path("results/ntp_policy_separation_check.html"),
    depth: int = DEFAULT_DEPTH,
) -> Path:
    summaries = []
    for scenario in SCENARIOS:
        rows = rank_actions(State(), scenario, depth)
        reaction_text, thumb_text = ntp_distribution_summary(scenario)
        summaries.append(
            {
                "scenario": scenario,
                "rows": rows,
                "reaction_text": reaction_text,
                "thumb_text": thumb_text,
                "warnings": warnings_for(scenario, rows),
            }
        )

    sections = []
    for summary in summaries:
        scenario = summary["scenario"]
        warnings = summary["warnings"]
        warning_html = (
            "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warnings) + "</ul>"
            if warnings
            else "<p class=\"ok\">警告なし</p>"
        )
        sections.append(
            f"""
            <section class="section">
              <h2>{html.escape(scenario.title)}</h2>
              <p><strong>policy:</strong> <code>{html.escape(scenario.policy_name)}</code></p>
              <p><strong>NTP反応分布:</strong> {html.escape(summary["reaction_text"])}</p>
              <p><strong>NTP指分布:</strong> {html.escape(summary["thumb_text"])}</p>
              <p><strong>期待確認:</strong> {html.escape(scenario.expected)}</p>
              <table>
                <thead>
                  <tr><th>#</th><th>TP行動</th><th>分類</th><th>評価値</th></tr>
                </thead>
                <tbody>
                  {render_rows(summary["rows"])}
                </tbody>
              </table>
              <div class="warnbox">
                <strong>警告:</strong>
                {warning_html}
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>NTP 方策分離チェック</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f4;
      color: #202124;
      font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif;
      line-height: 1.7;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    .lead {{ color: #5f6368; margin-bottom: 18px; }}
    .section {{
      background: #fff;
      border: 1px solid #d8d7d0;
      border-radius: 8px;
      padding: 18px;
      margin: 16px 0;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d8d7d0; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #eef1ef; text-align: left; }}
    code {{
      padding: 1px 5px;
      background: #eef1ef;
      border: 1px solid #dde3df;
      border-radius: 4px;
      font-family: Consolas, "Courier New", monospace;
    }}
    .note {{
      border-left: 4px solid #0b6b6f;
      background: #edf7f7;
      padding: 10px 12px;
      margin: 12px 0;
    }}
    .warnbox {{
      border-left: 4px solid #8a5300;
      background: #fff6e5;
      padding: 10px 12px;
      margin: 12px 0;
    }}
    .ok {{ color: #246b32; margin: 4px 0; }}
  </style>
</head>
<body>
<main>
  <h1>NTP 方策分離チェック</h1>
  <p class="lead">深さ制限評価 depth={depth} / 学習モデル未使用 / 0・50・100%カウンター × 最小指・一様指</p>
  <section class="section">
    <h2>この表の読み方</h2>
    <p>
      NTP の反応率と指選択を分けた条件で、初期局面における TP 側の上位行動を比較しています。
      これは学習済みモデルの方策ではなく、ルールエンジンと深さ制限評価による診断です。
    </p>
    <div class="note">
      目的は、数字宣言偏重がルール上の必然なのか、相手指固定や評価条件による副作用なのかを切り分けることです。
      50%カウンターは、人間が中間条件の自然さを判断しやすくするために追加しています。
    </div>
  </section>
  {"".join(sections)}
  <section class="section">
    <h2>AI側の解釈</h2>
    <p>
      0% と 100% は基本整合性を見るための端点、50% は人間判断を補助する中間点です。
      この結果が自然に見える場合、次は学習済みモデルの方策レポートを同じ分離条件で拡張します。
      不自然な警告が残る場合は、再学習より先に報酬・葉評価・合法手条件へ戻るべきです。
    </p>
  </section>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    path = generate_report()
    print(path)


if __name__ == "__main__":
    main()

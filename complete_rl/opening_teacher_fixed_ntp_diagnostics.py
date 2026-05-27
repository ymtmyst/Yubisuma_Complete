"""Compare opening BC teachers with fixed-NTP qualitative opening checks."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from complete_solver import State, TPAction
from complete_solver.constants import (
    ANTI_COUNTER_SKILLS,
    BOOST,
    COPY,
    FEINT,
    FLASH,
    GUARD,
    LOCK,
    QUICK,
    SKIP,
    TIME,
)
from complete_solver.finite_horizon import FiniteHorizonSolver, material_leaf_evaluator
from complete_solver.transition import transition
from complete_rl.bc_objective_diagnostics import miss_number_policy_note
from complete_rl.ntp_policy_separation_check import (
    SCENARIOS as FIXED_NTP_SCENARIOS,
    ntp_distribution,
    rank_actions,
)
from complete_rl.separated_policy_report import action_label, policy_label, pct


DEFAULT_DEPTH = 4
FIXED_POLICY_NAMES = ("none_uniform", "counter_uniform")
TURN_SETUP_SKILLS = {GUARD, SKIP, BOOST, TIME}
DIRECT_SKILLS = {FLASH, QUICK}


def fixed_scenarios() -> tuple:
    return tuple(
        scenario
        for scenario in FIXED_NTP_SCENARIOS
        if scenario.policy_name in FIXED_POLICY_NAMES
    )


def skill_group(skill: object) -> str:
    if isinstance(skill, int):
        return "数字宣言"
    if skill in ANTI_COUNTER_SKILLS:
        return "フェイント/ロック"
    if skill in TURN_SETUP_SKILLS:
        return "ターン/防御/準備"
    if skill in DIRECT_SKILLS:
        return "勝利直結寄り"
    if skill == COPY:
        return "参照"
    return "その他スキル"


def teacher_stats(depth: int = DEFAULT_DEPTH) -> dict:
    solver = FiniteHorizonSolver(leaf_evaluator=material_leaf_evaluator)
    policy = solver.solve_state(State(), depth)
    masses: Counter[str] = Counter()
    for action, prob in zip(policy.tp_actions, policy.tp_policy):
        masses[skill_group(action.skill)] += float(prob)
    rows = []
    for action, prob in zip(policy.tp_actions, policy.tp_policy):
        if float(prob) <= 1e-6:
            continue
        rows.append(
            {
                "label": action_label(action),
                "category": skill_group(action.skill),
                "prob": round(float(prob) * 100.0, 1),
                "note": miss_number_policy_note(action),
            }
        )
    rows.sort(key=lambda row: (-row["prob"], row["label"]))
    return {
        "depth": depth,
        "value": float(policy.value),
        "rows": rows[:12],
        "masses": masses,
    }


def fixed_rows(scenario, depth: int) -> list[dict]:
    rows = rank_actions(State(), scenario, depth)
    selected = list(enumerate(rows[:10], start=1))
    representative_feint = next(
        (
            (index, row)
            for index, row in enumerate(rows, start=1)
            if row["action"].skill == FEINT and row["action"].thumb == 1
        ),
        None,
    )
    selected_ranks = {index for index, _ in selected}
    if representative_feint is not None and representative_feint[0] not in selected_ranks:
        selected.append(representative_feint)
    rendered = []
    for rank, row in selected:
        action = row["action"]
        rendered.append(
            {
                "rank": rank,
                "action": action,
                "label": action_label(action),
                "skill": row["skill"],
                "category": skill_group(action.skill),
                "value": row["value"],
                "shape": one_step_shape(action, scenario),
            }
        )
    return rendered


def transition_outcome_label(result) -> str:
    if result.terminal_reward is not None:
        return f"終局報酬 {float(result.terminal_reward):+.1f}"
    if result.same_turn_player:
        return "開始側が追加ターン"
    return "最初の手番交代"


def weighted_summary(counter: Counter[str], weights: dict[str, float]) -> str:
    if not counter:
        return "-"
    return " / ".join(
        f"{label} {weights[label] * 100.0:.1f}%"
        for label, _ in counter.most_common()
    )


def one_step_shape(action: TPAction, scenario) -> dict[str, str]:
    outcome_order: Counter[str] = Counter()
    event_order: Counter[str] = Counter()
    outcome_weights: dict[str, float] = Counter()
    event_weights: dict[str, float] = Counter()
    for ntp_action, prob in ntp_distribution(State(), scenario):
        result = transition(State(), action, ntp_action)
        outcome = transition_outcome_label(result)
        outcome_order[outcome] += 1
        outcome_weights[outcome] += float(prob)
        if result.events:
            events = ", ".join(result.events)
        else:
            events = "-"
        event_order[events] += 1
        event_weights[events] += float(prob)
    return {
        "outcome": weighted_summary(outcome_order, outcome_weights),
        "events": weighted_summary(event_order, event_weights),
    }


def rows_html(rows: list[dict]) -> str:
    if not rows:
        return '<tr><td colspan="6">なし</td></tr>'
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['rank']}</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(row['category'])}</td>"
            f"<td>{row['value']:+.4f}</td>"
            f"<td>{html.escape(row['shape']['events'])}</td>"
            f"<td>{html.escape(row['shape']['outcome'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def teacher_rows_html(rows: list[dict]) -> str:
    if not rows:
        return '<tr><td colspan="4">なし</td></tr>'
    return "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['label']))}</td>"
        f"<td>{html.escape(str(row['category']))}</td>"
        f"<td>{row['prob']:.1f}%</td>"
        f"<td>{html.escape(str(row['note']))}</td>"
        "</tr>"
        for row in rows
    )


def mass_rows_html(masses: Counter[str]) -> str:
    if not masses:
        return '<tr><td colspan="2">なし</td></tr>'
    return "\n".join(
        "<tr>"
        f"<td>{html.escape(group)}</td>"
        f"<td>{mass * 100.0:.1f}%</td>"
        "</tr>"
        for group, mass in masses.most_common()
    )


def find_skill_rank(rows: list[dict], skills: set[str]) -> str:
    hits = [
        str(row["rank"])
        for row in rows
        if row["skill"] in skills
    ]
    return ", ".join(hits[:5]) if hits else "Top10外"


def qualitative_findings(teacher: dict, fixed: dict[str, list[dict]]) -> list[str]:
    findings: list[str] = []
    anti_mass = teacher["masses"].get("フェイント/ロック", 0.0) * 100.0
    setup_mass = teacher["masses"].get("ターン/防御/準備", 0.0) * 100.0
    findings.append(
        "BC depth="
        f"{teacher['depth']} 教師は初期局面でターン/防御/準備 {setup_mass:.1f}%、"
        f"フェイント/ロック {anti_mass:.1f}% を置く。固定NTP専用の最適応答ではなく、"
        "相手反応も手を選ぶゼロ和教師である。"
    )

    none_rows = fixed["none_uniform"]
    counter_rows = fixed["counter_uniform"]
    none_top = none_rows[0]
    counter_feint_ranks = find_skill_rank(counter_rows, {FEINT, LOCK})
    none_feint_ranks = find_skill_rank(none_rows, {FEINT, LOCK})
    findings.append(
        "カウンター0%・指一様の固定評価では先頭が "
        f"{none_top['label']}。フェイント/ロック順位は {none_feint_ranks} で、"
        "この条件では空振りイベントと手番交代を直接確認する。"
    )
    findings.append(
        "カウンター100%・指一様の固定評価ではフェイント/ロック順位が "
        f"{counter_feint_ranks}。対カウンター開幕が質的に正しいかは、"
        "`feint_success` と追加ターンが並ぶかで読む。"
    )
    return findings


def render_fixed_section(scenario, rows: list[dict], depth: int) -> str:
    expected = (
        "0%カウンターでは、フェイントが何も進めず手番を渡す形を警戒する。"
        if scenario.policy_name.startswith("none_")
        else "100%カウンターでは、対カウンター行動が追加ターンへ変わる形を確認する。"
    )
    return f"""
    <section class="section">
      <h2>{html.escape(policy_label(scenario.policy_name))}</h2>
      <p class="meta">内部表記: <code>{html.escape(scenario.policy_name)}</code> / fixed-NTP depth={depth}</p>
      <p>{html.escape(expected)}</p>
      <table>
        <thead>
          <tr><th>#</th><th>TP初手</th><th>分類</th><th>固定NTP評価</th><th>一次イベント</th><th>一手後の形</th></tr>
        </thead>
        <tbody>{rows_html(rows)}</tbody>
      </table>
    </section>
    """


def render_report(depth: int = DEFAULT_DEPTH) -> str:
    teacher = teacher_stats(depth)
    fixed = {
        scenario.policy_name: fixed_rows(scenario, depth)
        for scenario in fixed_scenarios()
    }
    finding_html = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in qualitative_findings(teacher, fixed)
    )
    fixed_sections = "".join(
        render_fixed_section(scenario, fixed[scenario.policy_name], depth)
        for scenario in fixed_scenarios()
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>開幕教師 vs 固定NTP 診断</title>
  <style>
    body {{ margin: 0; background: #f6f6f2; color: #202124; font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif; line-height: 1.68; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ font-size: 28px; letter-spacing: 0; margin: 0 0 6px; }}
    h2 {{ font-size: 20px; letter-spacing: 0; margin: 0 0 10px; }}
    h3 {{ font-size: 15px; letter-spacing: 0; margin: 12px 0 7px; }}
    .lead, .meta {{ color: #5f6368; }}
    .section {{ background: #fff; border: 1px solid #d7d6cf; border-radius: 8px; margin: 15px 0; padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .note {{ background: #edf6f5; border-left: 4px solid #0a6b6f; margin: 10px 0; padding: 10px 12px; }}
    .read {{ background: #fff5e2; border-left: 4px solid #8a5300; margin: 10px 0; padding: 10px 12px; }}
    table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    th, td {{ border: 1px solid #d7d6cf; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1ef; }}
    code {{ background: #eef1ef; border: 1px solid #dce3dd; border-radius: 4px; font-family: Consolas, "Courier New", monospace; padding: 1px 5px; }}
  </style>
</head>
<body>
<main>
  <h1>開幕教師 vs 固定NTP 診断</h1>
  <p class="lead">初期局面のみ / depth={depth} / 勝率ではなく、教師分布と固定相手に対する初手の意味を比較</p>
  <section class="section">
    <h2>確認意図</h2>
    <p>
      現在の BC depth={depth} 教師は、NTPも最悪応答を選ぶゼロ和の有限地平線教師です。
      一方、固定NTP評価は「相手がカウンターしない」「相手が必ずカウンターする」を分けた
      exploit確認です。この2つを混同せず、どの段階で開幕方策が固定されるかを切る前準備にします。
    </p>
    <div class="note">
      固定NTP表の「一次イベント」と「一手後の形」は初手だけを見ます。
      評価値は参考で、定性分析では `feint_no_counter`、`feint_success`、
      追加ターン、最初の手番交代を優先して読みます。
    </div>
  </section>
  <section class="section">
    <h2>BC finite-horizon 教師</h2>
    <p class="meta">初期局面 / value={teacher["value"]:+.4f} / depth={teacher["depth"]}</p>
    <div class="grid">
      <section>
        <h3>教師の行動分布</h3>
        <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{teacher_rows_html(teacher["rows"])}</tbody></table>
      </section>
      <section>
        <h3>教師のスキル群</h3>
        <table><thead><tr><th>群</th><th>確率</th></tr></thead><tbody>{mass_rows_html(teacher["masses"])}</tbody></table>
      </section>
    </div>
  </section>
  {fixed_sections}
  <section class="section">
    <h2>定性読み</h2>
    <div class="read"><ul>{finding_html}</ul></div>
    <p>
      次の確認では BC直後モデルと PPO後モデルの開始側手順を同じ診断で比較し、
      教師分布の混合読みがそのままモデル開幕に残るのか、PPOでさらに固定されるのかを切る。
    </p>
  </section>
</main>
</body>
</html>
"""


def generate_report(
    output_path: str | Path = "results/opening_teacher_fixed_ntp_diagnostics.html",
    depth: int = DEFAULT_DEPTH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(depth), encoding="utf-8")
    return output_path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/opening_teacher_fixed_ntp_diagnostics.html")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(generate_report(args.output, args.depth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

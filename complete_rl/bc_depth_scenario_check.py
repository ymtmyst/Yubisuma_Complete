"""Targeted depth comparison for Guard-alternative scenarios.

Compares depth=3/4/5 for:
  - initial: baseline
  - time_active: me has used Time (extra turn). Guard still somewhat valid.
  - opp_one_hand: opponent has 1 hand left. Guard utility low.

Zero leaf VI is excluded (論外). Material leaf VI shown as reference only.
Focus: Guard rate, Feint/Skip rate, other skill distribution.
"""

from __future__ import annotations

import html
from pathlib import Path

from complete_solver.constants import BOOST, FEINT, FLASH, GUARD, LOCK, QUICK, SKIP, TIME
from complete_solver.state import PlayerState, State
from complete_rl.bc_objective_diagnostics import (
    finite_horizon_rows,
    vi_teacher_rows,
)


TARGETED_SCENARIOS: list[dict] = [
    {
        "key": "initial",
        "title": "初期局面（ベースライン）",
        "state": State(),
        "note": "ガード72%/フェイント22%が depth=4 の基準。フラッシュ0%、数字0%。",
    },
    {
        "key": "time_active",
        "title": "タイム使用中（me.time_active=True）",
        "state": State(me=PlayerState(time_active=True)),
        "note": "タイムで追加ターンを得ている局面。相手はターン終了後に反撃可能なのでガードに一定の価値はある。",
    },
    {
        "key": "opp_one_hand",
        "title": "相手の手が残り1本",
        "state": State(opp=PlayerState(hands=1)),
        "note": "相手が1本しかないため宣言できる数字が限られ、フラッシュも脅威が減る。ガードの選択価値が低いはず。",
    },
]

DEPTHS = (3, 4, 5)

SKILL_GROUPS = {
    "guard": {GUARD},
    "feint_lock": {FEINT, LOCK},
    "skip_boost_time": {SKIP, BOOST, TIME},
    "quick_flash": {QUICK, FLASH},
}


def skill_masses(actions, probs) -> dict[str, float]:
    masses: dict[str, float] = {k: 0.0 for k in SKILL_GROUPS}
    masses["number"] = 0.0
    for action, prob in zip(actions, probs):
        p = float(prob)
        if isinstance(action.skill, int):
            masses["number"] += p
        else:
            for group_key, skills in SKILL_GROUPS.items():
                if action.skill in skills:
                    masses[group_key] += p
    return masses


def pct(v: float) -> str:
    if v < 0.005:
        return "—"
    return f"{v * 100:.1f}%"


def render_top_actions(rows: list[dict], limit: int = 5) -> str:
    return "<br>".join(
        f"{html.escape(r['action'])} ({r['prob']:.1f}%)" for r in rows[:limit]
    )


def render_mass_row(masses: dict[str, float]) -> str:
    return (
        f"<td class=\"guard\">{pct(masses['guard'])}</td>"
        f"<td>{pct(masses['feint_lock'])}</td>"
        f"<td>{pct(masses['skip_boost_time'])}</td>"
        f"<td>{pct(masses['quick_flash'])}</td>"
        f"<td>{pct(masses['number'])}</td>"
    )


def generate_report(
    output_path: Path = Path("results/bc_depth_scenario_check.html"),
) -> Path:
    sections = []

    for scenario in TARGETED_SCENARIOS:
        state: State = scenario["state"]
        title: str = scenario["title"]
        note: str = scenario["note"]

        # Collect depth=3/4/5 results
        depth_results = []
        for depth in DEPTHS:
            value, rows = finite_horizon_rows(state, depth)
            # Re-solve to get raw probs for mass calculation
            from complete_solver.finite_horizon import FiniteHorizonSolver, material_leaf_evaluator
            policy = FiniteHorizonSolver(leaf_evaluator=material_leaf_evaluator).solve_state(state, depth)
            masses = skill_masses(policy.tp_actions, policy.tp_policy)
            depth_results.append({"depth": depth, "value": value, "rows": rows, "masses": masses})

        # Material leaf VI for reference
        mat_vi = vi_teacher_rows(state, 400, use_material_leaf=True)
        from complete_solver.finite_horizon import FiniteHorizonSolver, material_leaf_evaluator
        from complete_solver.actions import legal_tp_actions
        import numpy as np
        # Recompute masses from VI
        vi_policy_rows = mat_vi[4]
        # Approximate masses from top_rows (only for reference)
        vi_guard_mass = sum(r["prob"] for r in vi_policy_rows if "ガード" in r["action"]) / 100.0
        vi_feint_mass = sum(r["prob"] for r in vi_policy_rows if "フェイント" in r["action"] or "ロック" in r["action"]) / 100.0
        vi_skip_mass = sum(r["prob"] for r in vi_policy_rows if "スキップ" in r["action"] or "ブースト" in r["action"] or "タイム" in r["action"]) / 100.0
        vi_win_mass = sum(r["prob"] for r in vi_policy_rows if "フラッシュ" in r["action"] or "クイック" in r["action"]) / 100.0
        vi_num_mass = sum(r["prob"] for r in vi_policy_rows if "数字" in r["action"]) / 100.0
        vi_masses = {
            "guard": vi_guard_mass,
            "feint_lock": vi_feint_mass,
            "skip_boost_time": vi_skip_mass,
            "quick_flash": vi_win_mass,
            "number": vi_num_mass,
        }

        # Depth comparison table
        depth_rows_html = ""
        for dr in depth_results:
            depth = dr["depth"]
            masses = dr["masses"]
            guard_class = ""
            if scenario["key"] == "opp_one_hand" and depth >= 4 and masses["guard"] > 0.5:
                guard_class = " warn-cell"
            depth_rows_html += (
                f"<tr>"
                f"<td><strong>depth={depth}</strong></td>"
                f"<td>{dr['value']:+.4f}</td>"
                f"<td class=\"guard{guard_class}\">{pct(masses['guard'])}</td>"
                f"<td>{pct(masses['feint_lock'])}</td>"
                f"<td>{pct(masses['skip_boost_time'])}</td>"
                f"<td>{pct(masses['quick_flash'])}</td>"
                f"<td>{pct(masses['number'])}</td>"
                f"<td>{render_top_actions(dr['rows'])}</td>"
                f"</tr>"
            )

        # VI material reference row
        vi_guard_flag = ""
        if scenario["key"] == "opp_one_hand" and vi_masses["guard"] > 0.5:
            vi_guard_flag = " warn-cell"
        depth_rows_html += (
            f"<tr class=\"vi-row\">"
            f"<td>VI material leaf（参考）</td>"
            f"<td>{mat_vi[0]:+.4f}</td>"
            f"<td class=\"guard{vi_guard_flag}\">{pct(vi_masses['guard'])}</td>"
            f"<td>{pct(vi_masses['feint_lock'])}</td>"
            f"<td>{pct(vi_masses['skip_boost_time'])}</td>"
            f"<td>{pct(vi_masses['quick_flash'])}</td>"
            f"<td>{pct(vi_masses['number'])}</td>"
            f"<td>{render_top_actions(mat_vi[4])}</td>"
            f"</tr>"
        )

        sections.append(f"""
  <section class="section">
    <h2>{html.escape(title)}</h2>
    <p class="lead">{html.escape(note)}</p>
    <table>
      <thead>
        <tr>
          <th>教師</th><th>価値</th>
          <th class="guard">ガード率</th>
          <th>フェイント/ロック率</th>
          <th>スキップ/ブースト/タイム率</th>
          <th>勝利直結率</th>
          <th>数字率</th>
          <th>上位行動（上位5）</th>
        </tr>
      </thead>
      <tbody>{depth_rows_html}</tbody>
    </table>
  </section>
""")

    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>深さ別シナリオ教師チェック（ガード代替分析）</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #202124; font-family: "Segoe UI","Yu Gothic","Meiryo",sans-serif; line-height: 1.7; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .lead {{ color: #5f6368; margin-bottom: 14px; }}
    .section {{ background: #fff; border: 1px solid #d8d7d0; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .note {{ border-left: 4px solid #0b6b6f; background: #edf7f7; padding: 10px 12px; margin: 12px 0; }}
    .warn {{ border-left: 4px solid #9b1c1c; background: #fff0f0; padding: 10px 12px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d8d7d0; padding: 7px 9px; vertical-align: top; }}
    th {{ background: #eef1ef; text-align: left; }}
    th.guard, td.guard {{ background: #e8f4ff; font-weight: bold; }}
    td.warn-cell {{ background: #ffe0e0; font-weight: bold; color: #9b1c1c; }}
    tr.vi-row td {{ color: #5f6368; background: #f9f9f6; font-style: italic; }}
    code {{ padding: 1px 5px; background: #eef1ef; border: 1px solid #dde3df; border-radius: 4px; font-family: Consolas,"Courier New",monospace; }}
  </style>
</head>
<body>
<main>
  <h1>深さ別シナリオ教師チェック（ガード代替分析）</h1>
  <p class="lead">depth=3/4/5 × 3シナリオ。ガード率が状況に応じて変化するかを確認。zero leaf VI は論外のため除外。</p>

  <section class="section">
    <h2>確認意図</h2>
    <div class="note">
      <ul>
        <li><strong>ガード率の妥当性</strong>: 相手の手が残り1本の局面では、ガードの選択価値は低い（攻撃パターンが限られるため）。depth=4/5 でガード率が下がるかを確認。</li>
        <li><strong>タイム使用中</strong>: 追加ターン中はガードにも一定の価値あり。ガード率が高くてもおかしくない。</li>
        <li><strong>depth=5 の必要性</strong>: depth=4 と depth=5 で結果が大きく変わらなければ、BC 教師は depth=4 で十分。</li>
        <li><strong>zero leaf VI</strong>: 数字100%のため除外。material leaf VI は参考行として掲載。</li>
      </ul>
    </div>
  </section>

  {"".join(sections)}

  <section class="section">
    <h2>AI側の解釈欄（確認後に記入）</h2>
    <p>各シナリオについて、depth=4/5 のガード率変化と上位行動の妥当性を確認し、ここに記録する。</p>
  </section>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    print("depth=3/4/5 を計算中（depth=5 は少し時間がかかります）...")
    path = generate_report()
    print(path)


if __name__ == "__main__":
    main()

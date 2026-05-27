"""Generate a focused diagnostic for opening Flash evaluation.

This report treats opening Flash as legal but strategically bad, then checks
whether short-horizon teachers are overvaluing it.
"""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

import numpy as np

from complete_solver import NTPAction, State, TPAction
from complete_solver.actions import legal_ntp_actions, legal_tp_actions
from complete_solver.constants import (
    BOOST,
    CHOICE,
    COPY,
    COUNTER,
    FEINT,
    FLASH,
    GUARD,
    LOCK,
    NONE,
    QUICK,
    SKIP,
    STOCK,
    TIME,
)
from complete_solver.finite_horizon import FiniteHorizonSolver, material_leaf_evaluator
from complete_solver.matrix_game import solve_zero_sum_matrix
from complete_solver.state_space import enumerate_reachable_states, value_iteration
from complete_solver.transition import transition


SKILL_LABELS = {
    FLASH: "フラッシュ",
    FEINT: "フェイント",
    LOCK: "ロック",
    SKIP: "スキップ",
    GUARD: "ガード",
    QUICK: "クイック",
    BOOST: "ブースト",
    TIME: "タイム",
    COPY: "コピー",
    STOCK: "ストック",
    CHOICE: "チョイス",
}


def action_label(action: TPAction) -> str:
    if isinstance(action.skill, int):
        return f"数字 total={action.skill}, thumb={action.thumb}"
    base = SKILL_LABELS.get(action.skill, str(action.skill))
    if action.skill == CHOICE and action.choice is not None:
        choice = SKILL_LABELS.get(action.choice, str(action.choice))
        return f"チョイス({choice}), thumb={action.thumb}"
    return f"{base}, thumb={action.thumb}"


def action_category(action: TPAction) -> str:
    if isinstance(action.skill, int):
        return "数字"
    if action.skill in {FEINT, LOCK}:
        return "対カウンター"
    if action.skill in {SKIP, GUARD, BOOST, TIME}:
        return "ターン/防御/準備"
    if action.skill in {FLASH, QUICK}:
        return "勝利直結"
    if action.skill in {COPY, STOCK, CHOICE}:
        return "参照"
    return "その他スキル"


def policy_stats(depth: int) -> dict:
    policy = FiniteHorizonSolver(leaf_evaluator=material_leaf_evaluator).solve_state(State(), depth)
    rows = []
    masses = {
        "flash": 0.0,
        "number": 0.0,
        "anti_counter": 0.0,
        "turn_setup": 0.0,
        "direct_win": 0.0,
    }
    for action, prob in zip(policy.tp_actions, policy.tp_policy):
        p = float(prob)
        if isinstance(action.skill, int):
            masses["number"] += p
        if action.skill == FLASH:
            masses["flash"] += p
        if action.skill in {FEINT, LOCK}:
            masses["anti_counter"] += p
        if action.skill in {SKIP, GUARD, BOOST, TIME}:
            masses["turn_setup"] += p
        if action.skill in {FLASH, QUICK}:
            masses["direct_win"] += p
        if p > 1e-6:
            rows.append(
                {
                    "action": action_label(action),
                    "category": action_category(action),
                    "prob": p * 100.0,
                }
            )
    rows.sort(key=lambda row: (-row["prob"], row["action"]))
    return {"depth": depth, "value": float(policy.value), "masses": masses, "rows": rows[:10]}


@lru_cache(maxsize=None)
def cached_vi(use_material_leaf: bool):
    states = enumerate_reachable_states(max_states=400)
    vi = value_iteration(
        states,
        gamma=0.999,
        epsilon=1e-4,
        max_iterations=500,
        leaf_evaluator=material_leaf_evaluator if use_material_leaf else None,
    )
    return states, vi


def vi_policy_stats(use_material_leaf: bool) -> dict:
    states, vi = cached_vi(use_material_leaf)
    state = State()
    tp_actions = legal_tp_actions(state)
    ntp_actions = legal_ntp_actions(state)
    matrix = np.zeros((len(tp_actions), len(ntp_actions)), dtype=float)
    for i, tp_action in enumerate(tp_actions):
        for j, ntp_action in enumerate(ntp_actions):
            result = transition(state, tp_action, ntp_action)
            if result.terminal_reward is not None:
                matrix[i, j] = float(result.terminal_reward)
                continue
            assert result.next_state is not None
            fallback = material_leaf_evaluator(result.next_state) if use_material_leaf else 0.0
            next_value = vi.values.get(result.next_state, fallback)
            matrix[i, j] = 0.999 * (next_value if result.same_turn_player else -next_value)

    solution = solve_zero_sum_matrix(matrix)
    rows = []
    masses = {"flash": 0.0, "number": 0.0}
    for action, prob in zip(tp_actions, solution.row_policy):
        p = float(prob)
        if isinstance(action.skill, int):
            masses["number"] += p
        if action.skill == FLASH:
            masses["flash"] += p
        if p > 1e-6:
            rows.append(
                {
                    "action": action_label(action),
                    "category": action_category(action),
                    "prob": p * 100.0,
                }
            )
    rows.sort(key=lambda row: (-row["prob"], row["action"]))
    return {
        "value": float(solution.value),
        "states": len(states),
        "converged": vi.converged,
        "delta": float(vi.max_delta),
        "masses": masses,
        "rows": rows[:10],
    }


def teacher_rows() -> list[dict]:
    rows = []
    for name, use_material in (("zero leaf BC教師", False), ("material leaf BC教師", True)):
        stats = vi_policy_stats(use_material)
        rows.append(
            {
                "name": name,
                "value": stats["value"],
                "states": stats["states"],
                "converged": stats["converged"],
                "delta": stats["delta"],
                "flash_mass": stats["masses"]["flash"] * 100.0,
                "number_mass": stats["masses"]["number"] * 100.0,
                "top": stats["rows"][:8],
            }
        )
    return rows


def flash_followup_rows() -> list[dict]:
    cases = [
        (
            "初手フラッシュが外れた後",
            TPAction(FLASH, 1),
            NTPAction(NONE, 0),
            TPAction(COPY, 1),
            NTPAction(NONE, 1),
            "相手がコピーでフラッシュを参照すると、即勝ち筋になり得る。",
        ),
        (
            "初手フラッシュが外れた後",
            TPAction(FLASH, 1),
            NTPAction(NONE, 0),
            TPAction(STOCK, 1),
            NTPAction(NONE, 1),
            "相手にフラッシュをストックされ、チョイス等の長期的な圧を与える。",
        ),
        (
            "初手フラッシュが当たった後",
            TPAction(FLASH, 1),
            NTPAction(NONE, 1),
            TPAction(COPY, 1),
            NTPAction(NONE, 1),
            "初手の勝利判定が保留されるため、相手がコピーで逆に即勝ちできる形がある。",
        ),
    ]
    rows = []
    for title, first_tp, first_ntp, reply_tp, reply_ntp, note in cases:
        first = transition(State(), first_tp, first_ntp)
        if first.next_state is None:
            reply_desc = "初手で終局"
            events = ", ".join(first.events)
            original_tp_result = "未確認"
        else:
            reply = transition(first.next_state, reply_tp, reply_ntp)
            events = ", ".join(first.events + reply.events)
            if reply.terminal_reward is None:
                original_tp_result = "相手に参照資産を渡す"
            elif reply.terminal_reward > 0:
                original_tp_result = "元TP視点では敗北"
            elif reply.terminal_reward < 0:
                original_tp_result = "元TP視点では勝利"
            else:
                original_tp_result = "引き分け相当"
            reply_desc = action_label(reply_tp)
        rows.append(
            {
                "title": title,
                "first": f"{action_label(first_tp)} / NTP {first_ntp.reaction}, thumb={first_ntp.thumb}",
                "reply": reply_desc,
                "events": events,
                "result": original_tp_result,
                "note": note,
            }
        )
    return rows


def pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def render_depth_table(stats: list[dict]) -> str:
    body = []
    for item in stats:
        masses = item["masses"]
        top = "<br>".join(
            f"{html.escape(row['action'])} ({row['prob']:.1f}%)" for row in item["rows"][:5]
        )
        warning = ""
        if item["depth"] == 3 and masses["flash"] > 0.05:
            warning = "警告: depth=3だけ初手フラッシュが高い。"
        if item["depth"] == 4 and masses["flash"] <= 1e-6:
            warning = "depth=4では消えるため、短期探索の境界アーティファクト疑い。"
        body.append(
            "<tr>"
            f"<td>depth={item['depth']}</td>"
            f"<td>{item['value']:+.4f}</td>"
            f"<td>{pct(masses['flash'])}</td>"
            f"<td>{pct(masses['number'])}</td>"
            f"<td>{pct(masses['anti_counter'])}</td>"
            f"<td>{pct(masses['turn_setup'])}</td>"
            f"<td>{top}</td>"
            f"<td>{html.escape(warning)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def render_followup_table(rows: list[dict]) -> str:
    return "\n".join(
        "<tr>"
        f"<td>{html.escape(row['title'])}</td>"
        f"<td>{html.escape(row['first'])}</td>"
        f"<td>{html.escape(row['reply'])}</td>"
        f"<td>{html.escape(row['events'])}</td>"
        f"<td>{html.escape(row['result'])}</td>"
        f"<td>{html.escape(row['note'])}</td>"
        "</tr>"
        for row in rows
    )


def render_teacher_table(rows: list[dict]) -> str:
    body = []
    for item in rows:
        top = "<br>".join(
            f"{html.escape(row['action'])} ({row['prob']:.1f}%)" for row in item["top"]
        )
        body.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{item['value']:+.4f}</td>"
            f"<td>{item['states']}</td>"
            f"<td>{item['converged']}</td>"
            f"<td>{item['delta']:.2e}</td>"
            f"<td>{item['number_mass']:.1f}%</td>"
            f"<td>{item['flash_mass']:.1f}%</td>"
            f"<td>{top}</td>"
            "</tr>"
        )
    return "\n".join(body)


def generate_report(output_path: Path = Path("results/opening_flash_trap_diagnostics.html")) -> Path:
    depth_stats = [policy_stats(depth) for depth in (1, 2, 3, 4)]
    followups = flash_followup_rows()
    teachers = teacher_rows()
    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>初手フラッシュ過大評価診断</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #202124; font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif; line-height: 1.7; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    .lead {{ color: #5f6368; margin-bottom: 18px; }}
    .section {{ background: #fff; border: 1px solid #d8d7d0; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .note {{ border-left: 4px solid #0b6b6f; background: #edf7f7; padding: 10px 12px; margin: 12px 0; }}
    .warn {{ border-left: 4px solid #9b1c1c; background: #fff0f0; padding: 10px 12px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d8d7d0; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #eef1ef; text-align: left; }}
    code {{ padding: 1px 5px; background: #eef1ef; border: 1px solid #dde3df; border-radius: 4px; font-family: Consolas, "Courier New", monospace; }}
  </style>
</head>
<body>
<main>
  <h1>初手フラッシュ過大評価診断</h1>
  <p class="lead">初手フラッシュは合法だが、相手にコピー/ストックを許すため人間視点では通常採用しない、という前提で確認。</p>

  <section class="section">
    <h2>結論</h2>
    <div class="warn">
      finite horizon depth=3 が初手フラッシュを約33%混ぜるのは妥当ではありません。
      depth=4 ではフラッシュが0%に落ちるため、少なくとも depth=3 の教師分布をそのまま採用するのは危険です。
      material leaf BC教師は zero leaf より改善していますが、人間らしい初期方策としてはまだ不十分です。
    </div>
    <p>
      原因候補は「合法手生成」ではなく、短い探索深さと葉評価が、初手フラッシュ後に相手へ参照スキル資産を渡すリスクを十分に評価できていないことです。
      次の工程では、学習ではなく教師生成・評価器側を先に修正対象にします。
    </p>
  </section>

  <section class="section">
    <h2>深さ別の初期方策</h2>
    <table>
      <thead>
        <tr><th>評価</th><th>価値</th><th>フラッシュ率</th><th>数字率</th><th>対カウンター率</th><th>準備/ターン率</th><th>上位行動</th><th>判定</th></tr>
      </thead>
      <tbody>{render_depth_table(depth_stats)}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>初手フラッシュ後の危険</h2>
    <table>
      <thead>
        <tr><th>状況</th><th>初手</th><th>相手の次手例</th><th>イベント</th><th>元TP視点</th><th>解釈</th></tr>
      </thead>
      <tbody>{render_followup_table(followups)}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>BC教師の現状</h2>
    <table>
      <thead>
        <tr><th>教師</th><th>価値</th><th>状態数</th><th>収束</th><th>delta</th><th>数字率</th><th>フラッシュ率</th><th>上位行動</th></tr>
      </thead>
      <tbody>{render_teacher_table(teachers)}</tbody>
    </table>
    <div class="note">
      zero leaf は数字100%に寄るため不採用候補です。
      material leaf は数字100%を崩しますが、数字やスキップへの寄りが残り、人間の初期方策とはまだズレています。
    </div>
  </section>

  <section class="section">
    <h2>次にやるべきこと</h2>
    <ol>
      <li>depth=3 の教師を採用対象から外し、少なくとも depth=4 相当または参照リスクを含む葉評価で比較する。</li>
      <li>初手フラッシュ、初手数字連打、追加ターン維持を警告する教師分布チェックを gate として追加する。</li>
      <li>material leaf を最終案にせず、コピー/ストックされる危険、相手に強い参照資産を渡す危険を評価器に入れる案を作る。</li>
      <li>修正後の教師分布HTMLを確認してから、BC smoke 学習に進む。</li>
    </ol>
  </section>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    print(generate_report())


if __name__ == "__main__":
    main()

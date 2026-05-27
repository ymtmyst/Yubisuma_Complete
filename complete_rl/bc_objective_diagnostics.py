"""Diagnose BC teacher distributions and swapped-perspective PPO objectives."""

from __future__ import annotations

import html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from complete_solver import RulesConfig, State, NTPAction, TPAction
from complete_solver.actions import legal_ntp_actions, legal_tp_actions
from complete_solver.constants import (
    ANTI_COUNTER_SKILLS,
    COUNTER,
    FEINT,
    FLASH,
    LOCK,
    NONE,
    QUICK,
    SKIP,
)
from complete_solver.finite_horizon import FiniteHorizonSolver, material_leaf_evaluator
from complete_solver.matrix_game import solve_zero_sum_matrix
from complete_solver.state import PlayerState
from complete_solver.state_space import enumerate_reachable_states, value_iteration
from complete_solver.transition import transition


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    state: State
    note: str


def _after(tp_action: TPAction, ntp_action: NTPAction) -> State:
    result = transition(State(), tp_action, ntp_action)
    assert result.next_state is not None
    return result.next_state


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="initial",
        title="初期局面",
        state=State(),
        note="フェイント等が使用可能な通常の初期局面。",
    ),
    Scenario(
        key="after_miss_number_vs_counter",
        title="外し数字がカウンターされた直後",
        state=_after(TPAction(4, 1), NTPAction(COUNTER, 0)),
        note="数字4 thumb=1 は初期局面のカウンター指0/1/2に必ず外れる。",
    ),
    Scenario(
        key="feint_drop_blocked",
        title="フェイントがドロップ封印された局面",
        state=State(me=PlayerState(drop_blocked_skills=frozenset({FEINT}))),
        note="外し数字をフェイント代替として許容し得る代表例。",
    ),
    Scenario(
        key="opp_has_stock_flash",
        title="相手がフラッシュをストックしている局面",
        state=State(opp=PlayerState(stock=frozenset({FLASH}))),
        note="参照スキルへのメタとして数字が理論上価値を持ち得る代表例。",
    ),
)


def action_label(action: TPAction) -> str:
    if isinstance(action.skill, int):
        return f"数字宣言 total={action.skill}, thumb={action.thumb}"
    return f"{action.skill}, thumb={action.thumb}"


def category(action: TPAction) -> str:
    if isinstance(action.skill, int):
        return "数字宣言"
    if action.skill in ANTI_COUNTER_SKILLS:
        return "対カウンター"
    if action.skill in {FLASH, QUICK}:
        return "勝利直結寄り"
    if action.skill == SKIP:
        return "追加ターン/妨害"
    return "スキル"


def miss_number_policy_note(action: TPAction) -> str:
    if not isinstance(action.skill, int):
        return ""
    if action.skill == 4:
        return "初期2本同士ではカウンターに外れやすい数字。常用は警告対象。"
    if action.skill == 0:
        return "初期2本同士では通常宣言として当たりにくい数字。BC教師で100%なら警告対象。"
    return "数字宣言。状況依存で許容されるが、フェイントを圧倒する常用は要注意。"


def top_rows_from_policy(actions: tuple[TPAction, ...], probs: tuple[float, ...] | np.ndarray, limit: int = 8) -> list[dict]:
    rows = []
    for action, prob in zip(actions, probs):
        if float(prob) <= 1e-6:
            continue
        rows.append(
            {
                "action": action_label(action),
                "category": category(action),
                "prob": round(float(prob) * 100.0, 1),
                "note": miss_number_policy_note(action),
            }
        )
    rows.sort(key=lambda row: (-row["prob"], row["action"]))
    return rows[:limit]


def finite_horizon_rows(state: State, depth: int) -> tuple[float, list[dict]]:
    policy = FiniteHorizonSolver().solve_state(state, depth=depth)
    return policy.value, top_rows_from_policy(policy.tp_actions, policy.tp_policy)


@lru_cache(maxsize=None)
def cached_vi(max_states: int, use_material_leaf: bool):
    config = RulesConfig()
    states = enumerate_reachable_states(config=config, max_states=max_states)
    vi = value_iteration(
        states,
        config=config,
        gamma=0.999,
        epsilon=1e-4,
        max_iterations=500,
        leaf_evaluator=material_leaf_evaluator if use_material_leaf else None,
    )
    return states, vi


def vi_teacher_rows(
    state: State,
    max_states: int,
    *,
    gamma: float = 0.999,
    vi_epsilon: float = 1e-4,
    vi_max_iter: int = 500,
    use_material_leaf: bool = False,
) -> tuple[float, int, bool, float, list[dict]]:
    config = RulesConfig()
    if gamma != 0.999 or vi_epsilon != 1e-4 or vi_max_iter != 500:
        states = enumerate_reachable_states(config=config, max_states=max_states)
        vi = value_iteration(
            states,
            config=config,
            gamma=gamma,
            epsilon=vi_epsilon,
            max_iterations=vi_max_iter,
            leaf_evaluator=material_leaf_evaluator if use_material_leaf else None,
        )
    else:
        states, vi = cached_vi(max_states, use_material_leaf)
    V = vi.values
    tp_actions = legal_tp_actions(state, config)
    ntp_actions = legal_ntp_actions(state, config)
    matrix = np.zeros((len(tp_actions), len(ntp_actions)), dtype=float)
    for i, tp_action in enumerate(tp_actions):
        for j, ntp_action in enumerate(ntp_actions):
            result = transition(state, tp_action, ntp_action, config)
            if result.terminal_reward is not None:
                matrix[i, j] = float(result.terminal_reward)
            else:
                assert result.next_state is not None
                next_value = V.get(
                    result.next_state,
                    float(material_leaf_evaluator(result.next_state)) if use_material_leaf else 0.0,
                )
                matrix[i, j] = gamma * (next_value if result.same_turn_player else -next_value)
    solution = solve_zero_sum_matrix(matrix)
    return (
        solution.value,
        len(states),
        vi.converged,
        vi.max_delta,
        top_rows_from_policy(tp_actions, solution.row_policy),
    )


def render_rows(rows: list[dict]) -> str:
    if not rows:
        return "<tr><td colspan=\"4\">方策なし</td></tr>"
    return "\n".join(
        "<tr>"
        f"<td>{html.escape(row['action'])}</td>"
        f"<td>{html.escape(row['category'])}</td>"
        f"<td>{row['prob']:.1f}%</td>"
        f"<td>{html.escape(row['note'])}</td>"
        "</tr>"
        for row in rows
    )


def render_policy_section() -> str:
    parts = []
    for scenario in SCENARIOS:
        finite1_value, finite1_rows = finite_horizon_rows(scenario.state, 1)
        finite3_value, finite3_rows = finite_horizon_rows(scenario.state, 3)
        vi_default = vi_teacher_rows(scenario.state, 400, use_material_leaf=False)
        vi_material = vi_teacher_rows(scenario.state, 400, use_material_leaf=True)
        parts.append(
            f"""
            <section class="section">
              <h2>{html.escape(scenario.title)}</h2>
              <p>{html.escape(scenario.note)}</p>
              <div class="grid">
                <div>
                  <h3>有限深さ depth=1 / value={finite1_value:+.4f}</h3>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(finite1_rows)}</tbody></table>
                </div>
                <div>
                  <h3>有限深さ depth=3 / value={finite3_value:+.4f}</h3>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(finite3_rows)}</tbody></table>
                </div>
                <div>
                  <h3>VI教師 default leaf / value={vi_default[0]:+.4f}</h3>
                  <p class="meta">states={vi_default[1]}, converged={vi_default[2]}, delta={vi_default[3]:.2e}</p>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(vi_default[4])}</tbody></table>
                </div>
                <div>
                  <h3>VI教師 material leaf / value={vi_material[0]:+.4f}</h3>
                  <p class="meta">states={vi_material[1]}, converged={vi_material[2]}, delta={vi_material[3]:.2e}</p>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(vi_material[4])}</tbody></table>
                </div>
              </div>
            </section>
            """
        )
    return "".join(parts)


def perspective_rows() -> str:
    rows = []
    checks = [
        ("外し数字 vs カウンター", TPAction(4, 1), NTPAction(COUNTER, 0)),
        ("フェイント vs カウンター", TPAction(FEINT, 0), NTPAction(COUNTER, 0)),
        ("フラッシュ vs 反応なし", TPAction(FLASH, 0), NTPAction(NONE, 0)),
    ]
    for label, tp_action, ntp_action in checks:
        result = transition(State(), tp_action, ntp_action)
        if result.next_state is None:
            next_desc = "終端"
        else:
            next_desc = (
                f"次手番me.hands={result.next_state.me.hands}, "
                f"opp.hands={result.next_state.opp.hands}, "
                f"same_turn={result.same_turn_player}"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(action_label(tp_action))}</td>"
            f"<td>{html.escape(ntp_action.key())}</td>"
            f"<td>{html.escape(', '.join(result.events))}</td>"
            f"<td>{html.escape(next_desc)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def generate_report(output_path: Path = Path("results/bc_objective_diagnostics.html")) -> Path:
    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>BC教師分布・PPO目的診断</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #202124; font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif; line-height: 1.7; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 12px 0 8px; font-size: 15px; letter-spacing: 0; }}
    .lead {{ color: #5f6368; margin-bottom: 18px; }}
    .section {{ background: #fff; border: 1px solid #d8d7d0; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 14px; }}
    .note {{ border-left: 4px solid #0b6b6f; background: #edf7f7; padding: 10px 12px; margin: 12px 0; }}
    .warn {{ border-left: 4px solid #9b1c1c; background: #fff0f0; padding: 10px 12px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d8d7d0; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #eef1ef; text-align: left; }}
    .meta {{ color: #5f6368; font-size: 12px; margin: 0 0 6px; }}
    code {{ padding: 1px 5px; background: #eef1ef; border: 1px solid #dde3df; border-radius: 4px; font-family: Consolas, "Courier New", monospace; }}
  </style>
</head>
<body>
<main>
  <h1>BC教師分布・PPO目的診断</h1>
  <p class="lead">学習前診断 / 外し数字の許容方針を反映 / 再学習前の確認用</p>

  <section class="section">
    <h2>外し数字の扱い</h2>
    <div class="note">
      方針: 特定局面では許容するが、基本的にはあまり許容しない。
      フェイントがドロップで封印されている、参照スキルへのメタになる、カウンター有無が半々で当たりも期待する、といった状況では許容する。
      ただし、フェイントを遥かに上回る形で意図的な外し数字を常用する方策は警告対象とする。
    </div>
  </section>

  <section class="section">
    <h2>視点切替/PPO目的の確認</h2>
    <p>
      CompleteEnv は手番交代時に状態の視点を入れ替える。
      exact solver は手番交代時に価値の符号を反転するが、通常の単一エージェントPPOは次状態価値を同じ目的の継続として扱う。
      このため、交代後の次プレイヤーに有利な価値が、直前行動の価値として混ざる可能性がある。
    </p>
    <table>
      <thead><tr><th>確認</th><th>TP行動</th><th>NTP行動</th><th>イベント</th><th>次状態</th></tr></thead>
      <tbody>{perspective_rows()}</tbody>
    </table>
    <div class="warn">
      要確認: 現在の環境は「共有方策が両者を操作する」形としては動くが、
      直前の手番プレイヤーのゼロサム目的をPPOのリターンに正しく反映しているかは未確定。
    </div>
  </section>

  {render_policy_section()}

  <section class="section">
    <h2>AI側の解釈</h2>
    <p>
      有限深さ評価とVI教師で初期局面の方策が大きく異なる場合、数字偏重は学習済みモデルだけの問題ではなく、
      BC教師生成または価値反復の境界条件から入っている可能性が高い。
      また、視点切替とPPO目的の不一致がある場合、再学習しても同じ局所解に戻る危険がある。
    </p>
    <p>
      次に進む前に、BC教師の作り方を修正すべきか、RL環境を「現在手番プレイヤーの報酬」ではなく
      固定プレイヤー視点または符号反転済みの自己対戦形式に変えるべきかを判断する。
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

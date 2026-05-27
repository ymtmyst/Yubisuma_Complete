"""Generate a focused policy report for separated NTP conditions."""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path

from complete_solver import RulesConfig
from complete_solver.constants import (
    ALL,
    ANTI_COUNTER_SKILLS,
    BOOST,
    CHOICE,
    COPY,
    DROP,
    FLASH,
    GUARD,
    NORMAL_SKILLS,
    PASS,
    QUICK,
    SKIP,
    STOCK,
    TIME,
    ULTIMATE_TP_SKILLS,
)
from complete_rl.env import CompleteEnv, build_canonical_tp_actions, resolve_canonical_action


POLICIES = [
    "none_lowest",
    "none_uniform",
    "counter50_lowest",
    "counter50_uniform",
    "counter_lowest",
    "counter_uniform",
]
MODES = [("deterministic", True), ("stochastic", False)]
IMPORTANT_SKILLS = [FLASH, QUICK, GUARD, SKIP, BOOST, TIME, COPY, STOCK]
MODE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "deterministic": (
        "決定論評価",
        "方策が最も高い確率を付けた合法手を毎手選ぶ。固定開幕や局所解の確認向け。",
    ),
    "stochastic": (
        "確率サンプル評価",
        "方策分布から合法手をサンプルする。学習済み分布に残った選択幅の確認向け。",
    ),
}
POLICY_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "none_lowest": (
        "カウンター0%・指最小",
        "相手NTP反応は常に「なし」。合法な指のうち最小の指を選ぶ。",
    ),
    "none_uniform": (
        "カウンター0%・指一様",
        "相手NTP反応は常に「なし」。合法な指を一様ランダムに選ぶ。",
    ),
    "counter50_lowest": (
        "カウンター50%・指最小",
        "相手NTP反応は「なし」と「カウンター」を50%ずつ選び、指は最小にする。",
    ),
    "counter50_uniform": (
        "カウンター50%・指一様",
        "相手NTP反応は「なし」と「カウンター」を50%ずつ選び、指は一様にする。",
    ),
    "counter_lowest": (
        "カウンター100%・指最小",
        "相手NTP反応は常に「カウンター」。合法な指のうち最小の指を選ぶ。",
    ),
    "counter_uniform": (
        "カウンター100%・指一様",
        "相手NTP反応は常に「カウンター」。合法な指を一様ランダムに選ぶ。",
    ),
}


def skill_label(skill: object) -> str:
    if isinstance(skill, int):
        return f"数字{skill}"
    return str(skill)


def action_label(action) -> str:
    if isinstance(action.skill, int):
        return f"数字宣言 合計={action.skill}, 指={action.thumb}"
    if action.skill == CHOICE:
        return f"チョイス({action.choice}), 指={action.thumb}"
    if action.skill == ALL:
        return f"オール{action.all_order}, 指={action.thumb}"
    return f"{skill_label(action.skill)}, 指={action.thumb}"


def mode_label(mode_name: str) -> str:
    return MODE_DESCRIPTIONS.get(mode_name, (mode_name, ""))[0]


def policy_label(policy_name: str) -> str:
    return POLICY_DESCRIPTIONS.get(policy_name, (policy_name, ""))[0]


def broad_category(skill: object) -> str:
    if isinstance(skill, int):
        return "数字宣言"
    if skill == PASS:
        return "パス"
    if skill in ANTI_COUNTER_SKILLS:
        return "対カウンター"
    if skill in ULTIMATE_TP_SKILLS:
        return "アルティメット"
    if skill in (COPY, DROP, STOCK):
        return "参照スキル"
    if skill in NORMAL_SKILLS:
        return "通常スキル"
    return "その他"


def pct(count: int, total: int) -> float:
    return round(100.0 * count / max(total, 1), 1)


def analyze(
    model_path: str | Path,
    episodes: int = 80,
    max_steps: int = 300,
    seed_base: int = 9000000,
) -> dict:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("sb3-contrib required") from exc

    model = MaskablePPO.load(str(model_path))
    canonical = build_canonical_tp_actions(RulesConfig())
    results: dict = {}

    for mode_name, deterministic in MODES:
        for policy in POLICIES:
            env = CompleteEnv(opponent_policy=policy, max_steps=max_steps)
            wins = losses = truncations = total_steps = 0
            start_wins = start_losses = start_truncations = 0
            actions: Counter[str] = Counter()
            first_actions: Counter[str] = Counter()
            broad: Counter[str] = Counter()
            skills: Counter[str] = Counter()
            start_side_skills: Counter[str] = Counter()
            switched_side_skills: Counter[str] = Counter()
            important: Counter[str] = Counter()
            ntp_reactions: Counter[str] = Counter()
            ntp_thumbs: Counter[int] = Counter()
            after_extra: Counter[str] = Counter()
            trace: list[dict[str, str]] = []
            pending_extra_next = False

            for ep in range(episodes):
                obs, _ = env.reset(seed=seed_base + ep * 7919)
                done = False
                step = 0
                final_reward = 0.0
                actor_side = "開始側"
                pending_extra_next = False
                while not done:
                    state_before = env._state
                    mask = env.action_masks()
                    action_idx, _ = model.predict(
                        obs,
                        action_masks=mask,
                        deterministic=deterministic,
                    )
                    tp_action = resolve_canonical_action(canonical[int(action_idx)], state_before)
                    label = action_label(tp_action)
                    skill = skill_label(tp_action.skill)
                    actions[label] += 1
                    broad[broad_category(tp_action.skill)] += 1
                    skills[skill] += 1
                    if actor_side == "開始側":
                        start_side_skills[skill] += 1
                    else:
                        switched_side_skills[skill] += 1
                    if step == 0:
                        first_actions[label] += 1
                    if skill in IMPORTANT_SKILLS:
                        important[skill] += 1
                    if pending_extra_next:
                        after_extra[label] += 1
                        pending_extra_next = False

                    obs, reward, terminated, truncated, info = env.step(int(action_idx))
                    ntp_reactions[str(info.get("ntp_reaction", ""))] += 1
                    ntp_thumbs[int(info.get("ntp_thumb", 0))] += 1
                    pending_extra_next = bool(info.get("same_turn_player", False))
                    if ep == 0:
                        previous = (
                            skill_label(state_before.previous_skill)
                            if state_before.previous_skill is not None
                            else "-"
                        )
                        copy_target = previous if tp_action.skill == COPY else "-"
                        if terminated:
                            result_label = f"終端報酬={float(reward):+.1f}"
                        elif truncated:
                            result_label = "打切"
                        elif info.get("same_turn_player", False):
                            result_label = "追加ターン"
                        else:
                            result_label = "手番交代"
                        trace.append(
                            {
                                "step": str(step + 1),
                                "actor": actor_side,
                                "before": (
                                    f"TP手={state_before.me.hands}, "
                                    f"NTP手={state_before.opp.hands}, 直前={previous}"
                                ),
                                "action": label,
                                "ntp": (
                                    f"{info.get('ntp_reaction', '')}, "
                                    f"指={info.get('ntp_thumb', '')}"
                                ),
                                "copy_target": copy_target,
                                "events": ", ".join(info.get("events", ())) or "-",
                                "result": result_label,
                            }
                        )
                    final_reward = reward
                    step += 1
                    done = terminated or truncated
                    if not done and not info.get("same_turn_player", False):
                        actor_side = (
                            "手番交代後側" if actor_side == "開始側" else "開始側"
                        )

                total_steps += step
                if final_reward > 0:
                    wins += 1
                elif final_reward < 0:
                    losses += 1
                else:
                    truncations += 1
                if final_reward == 0:
                    start_truncations += 1
                elif (
                    actor_side == "開始側" and final_reward > 0
                ) or (
                    actor_side == "手番交代後側" and final_reward < 0
                ):
                    start_wins += 1
                else:
                    start_losses += 1

            total_actions = sum(actions.values())
            results[(mode_name, policy)] = {
                "mode": mode_name,
                "policy": policy,
                "episodes": episodes,
                "wins": wins,
                "losses": losses,
                "truncations": truncations,
                "start_wins": start_wins,
                "start_losses": start_losses,
                "start_truncations": start_truncations,
                "avg_steps": round(total_steps / episodes, 1),
                "total_actions": total_actions,
                "actions": actions,
                "first_actions": first_actions,
                "broad": broad,
                "skills": skills,
                "start_side_skills": start_side_skills,
                "switched_side_skills": switched_side_skills,
                "important": important,
                "ntp_reactions": ntp_reactions,
                "ntp_thumbs": ntp_thumbs,
                "after_extra": after_extra,
                "trace": trace,
            }
    return results


def table_rows(counter: Counter, total: int, limit: int | None = 8) -> str:
    if not counter:
        return "<tr><td colspan=\"3\">なし</td></tr>"
    rows = []
    entries = counter.most_common(limit) if limit is not None else counter.most_common()
    for key, count in entries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(key))}</td>"
            f"<td>{count}</td>"
            f"<td>{pct(count, total):.1f}%</td>"
            "</tr>"
        )
    return "\n".join(rows)


def trace_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<tr><td colspan=\"8\">なし</td></tr>"
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(row['step'])}</td>"
            f"<td>{html.escape(row['actor'])}</td>"
            f"<td>{html.escape(row['before'])}</td>"
            f"<td>{html.escape(row['action'])}</td>"
            f"<td>{html.escape(row['ntp'])}</td>"
            f"<td>{html.escape(row['copy_target'])}</td>"
            f"<td>{html.escape(row['events'])}</td>"
            f"<td>{html.escape(row['result'])}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def warning_list(data: dict) -> list[str]:
    total = data["total_actions"]
    broad = data["broad"]
    warnings: list[str] = []
    episodes = max(int(data.get("episodes", 0)), 1)
    wins = int(data.get("start_wins", data.get("wins", 0)))
    truncations = int(data.get("start_truncations", data.get("truncations", 0)))
    win_pct = 100.0 * wins / episodes
    trunc_pct = 100.0 * truncations / episodes
    number_pct = pct(broad.get("数字宣言", 0), total)
    anti_pct = pct(broad.get("対カウンター", 0), total)
    extra_pct = pct(data["important"].get(GUARD, 0) + data["important"].get(BOOST, 0) + data["important"].get(SKIP, 0) + data["important"].get(TIME, 0), total)
    first_actions = data.get("first_actions", Counter())
    if wins == 0:
        warnings.append("勝利なし: 0.0%")
    elif win_pct < 50.0:
        warnings.append(f"勝率が低い: {win_pct:.1f}%")
    if trunc_pct >= 20.0:
        warnings.append(f"打ち切りが多い: {trunc_pct:.1f}%")
    if first_actions:
        first_action, first_count = first_actions.most_common(1)[0]
        first_pct = 100.0 * first_count / episodes
        if first_pct >= 90.0:
            warnings.append(
                f"初手が単一行動に集中: {html.escape(str(first_action))} {first_pct:.1f}%"
            )
    if number_pct >= 80.0:
        warnings.append(f"数字宣言が過多: {number_pct:.1f}%")
    if extra_pct >= 50.0:
        warnings.append(f"追加ターン/維持系が多い: {extra_pct:.1f}%")
    if anti_pct < 5.0 and data["ntp_reactions"].get("カウンター", 0) > 0:
        warnings.append(f"カウンター条件に対して対カウンター使用率が低い: {anti_pct:.1f}%")
    return warnings


def render_report(results: dict, model_path: Path, episodes: int) -> str:
    sections = []
    for mode_name, _ in MODES:
        for policy in POLICIES:
            data = results[(mode_name, policy)]
            total = data["total_actions"]
            broad_total = sum(data["broad"].values())
            ntp_total = sum(data["ntp_reactions"].values())
            warn = warning_list(data)
            warn_html = (
                "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warn) + "</ul>"
                if warn else "<p class=\"ok\">警告なし</p>"
            )
            sections.append(
                f"""
                <details class="section condition">
                  <summary>
                    <strong>{html.escape(mode_label(mode_name))} / {html.escape(policy_label(policy))}</strong>
                    <span>開始側 {data["start_wins"]}勝 {data["start_losses"]}敗 {data["start_truncations"]}打切 / 平均{data["avg_steps"]}手</span>
                  </summary>
                  <p class="internal-name">内部表記: {html.escape(mode_name)} / {html.escape(policy)}</p>
                  <div class="summary">
                    <span>開始側成績: {data["start_wins"]}勝 {data["start_losses"]}敗 {data["start_truncations"]}打切</span>
                    <span>終端TP報酬: + {data["wins"]} / - {data["losses"]} / 0 {data["truncations"]}</span>
                    <span>平均手数: {data["avg_steps"]}</span>
                    <span>総TP行動: {total}</span>
                  </div>
                  <div class="grid">
                    <div>
                      <h3>スキル別使用率</h3>
                      <table><thead><tr><th>スキル</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["skills"], total, None)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>開始側スキル別使用率</h3>
                      <table><thead><tr><th>スキル</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["start_side_skills"], max(sum(data["start_side_skills"].values()), 1), None)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>手番交代後側スキル別使用率</h3>
                      <table><thead><tr><th>スキル</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["switched_side_skills"], max(sum(data["switched_side_skills"].values()), 1), None)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>行動明細</h3>
                      <table><thead><tr><th>行動</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["actions"], total, None)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>カテゴリ集計（参考）</h3>
                      <table><thead><tr><th>分類</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["broad"], broad_total, 10)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>初手分布</h3>
                      <table><thead><tr><th>行動</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["first_actions"], episodes, None)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>追加ターン後の次手</h3>
                      <table><thead><tr><th>行動</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["after_extra"], max(sum(data["after_extra"].values()), 1), None)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>NTP反応分布</h3>
                      <table><thead><tr><th>反応</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["ntp_reactions"], ntp_total, 8)}
                      </tbody></table>
                    </div>
                    <div>
                      <h3>NTP指分布</h3>
                      <table><thead><tr><th>指</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {table_rows(data["ntp_thumbs"], ntp_total, 8)}
                      </tbody></table>
                    </div>
                  </div>
                  <h3>代表トレース（最初の1エピソード）</h3>
                  <table class="trace"><thead><tr><th>手</th><th>TP側</th><th>直前状態</th><th>TP行動</th><th>NTP反応</th><th>コピー対象</th><th>イベント</th><th>結果</th></tr></thead><tbody>
                    {trace_rows(data["trace"])}
                  </tbody></table>
                  <div class="warn"><strong>警告:</strong>{warn_html}</div>
                </details>
                """
            )
    mode_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{html.escape(description)}</td>"
        "</tr>"
        for name, (label, description) in MODE_DESCRIPTIONS.items()
    )
    policy_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{html.escape(description)}</td>"
        "</tr>"
        for name, (label, description) in POLICY_DESCRIPTIONS.items()
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>分離NTP 方策レポート</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #202124; font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif; line-height: 1.7; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 12px 0 8px; font-size: 15px; letter-spacing: 0; }}
    .lead {{ color: #5f6368; margin-bottom: 18px; }}
    .section {{ background: #fff; border: 1px solid #d8d7d0; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    details.section > summary {{ align-items: baseline; cursor: pointer; display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between; list-style-position: outside; }}
    details.section > summary strong {{ font-size: 20px; }}
    details.section > summary span {{ color: #5f6368; font-size: 14px; }}
    details.section[open] > summary {{ border-bottom: 1px solid #d8d7d0; margin-bottom: 12px; padding-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 12px; }}
    .summary span {{ background: #eef1ef; border: 1px solid #dde3df; border-radius: 999px; padding: 3px 10px; font-size: 13px; }}
    .internal-name {{ color: #5f6368; font-size: 13px; margin: 0 0 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d8d7d0; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #eef1ef; text-align: left; }}
    .trace {{ display: block; overflow-x: auto; white-space: nowrap; }}
    .trace td {{ white-space: normal; min-width: 84px; }}
    .trace td:nth-child(3), .trace td:nth-child(7) {{ min-width: 180px; }}
    .note {{ border-left: 4px solid #0b6b6f; background: #edf7f7; padding: 10px 12px; margin: 12px 0; }}
    .warn {{ border-left: 4px solid #8a5300; background: #fff6e5; padding: 10px 12px; margin: 12px 0 0; }}
    .ok {{ color: #246b32; margin: 4px 0; }}
  </style>
</head>
<body>
<main>
  <h1>分離NTP 方策レポート</h1>
  <p class="lead">モデル: {html.escape(str(model_path))} / 各条件 {episodes} episodes / 決定論評価と確率サンプル評価を比較</p>
  <section class="section">
    <h2>確認意図</h2>
    <p>
      0%/50%/100%カウンターと、最小指/一様指を分離した条件で、学習済みモデルの実際の行動分布を確認します。
      勝率だけでなく、初手、スキル別使用率、行動明細、追加ターン後の次手、NTP反応・指分布、警告を並べています。
    </p>
    <div class="note">
      この環境では手番が交代すると同じ方策が次のTP側も操作します。したがって全体のスキル使用率は開始側と手番交代後側の合算です。
      成績は「開始側成績」を主表示し、旧来の符号確認用に終端TP報酬も併記します。代表トレースでコピー対象と手番側を確認してください。
      警告は採用判断の補助です。
    </div>
  </section>
  <section class="section">
    <h2>内部表記の対応表</h2>
    <p>NTP は相手側の反応です。ここでは「反応なし」と「カウンター」を分離し、指選択も別条件として確認しています。</p>
    <h3>評価モード</h3>
    <table><thead><tr><th>内部表記</th><th>日本語表示</th><th>意味</th></tr></thead><tbody>{mode_rows}</tbody></table>
    <h3>相手NTP条件</h3>
    <table><thead><tr><th>内部表記</th><th>日本語表示</th><th>意味</th></tr></thead><tbody>{policy_rows}</tbody></table>
  </section>
  {"".join(sections)}
</main>
</body>
</html>
"""


def generate_report(
    model_path: str | Path,
    output_path: str | Path = "results/separated_policy_report_episode_mixed.html",
    episodes: int = 80,
    max_steps: int = 300,
) -> Path:
    model_path = Path(model_path)
    output_path = Path(output_path)
    results = analyze(model_path, episodes=episodes, max_steps=max_steps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(results, model_path, episodes), encoding="utf-8")
    return output_path


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--output", default="results/separated_policy_report_episode_mixed.html")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = generate_report(
        args.model_path,
        output_path=args.output,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

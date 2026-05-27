"""Generate opening-chain diagnostics from the starting TP side."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from complete_solver import RulesConfig
from complete_solver.constants import ANTI_COUNTER_SKILLS, COPY
from complete_rl.env import CompleteEnv, build_canonical_tp_actions, resolve_canonical_action
from complete_rl.separated_policy_report import (
    MODE_DESCRIPTIONS,
    POLICY_DESCRIPTIONS,
    action_label,
    mode_label,
    pct,
    policy_label,
    skill_label,
    trace_rows,
)


MODES = (("deterministic", True), ("stochastic", False))
POLICIES = ("none_lowest", "none_uniform", "counter_lowest", "counter_uniform")
EXPECTED_READS = {
    "none_lowest": (
        "0%カウンターへの開始側手順",
        "相手はカウンターしない。開始側初手がフェイント/ロックに寄るなら、"
        "固定相手を exploit する前に対カウンター開幕を空振りしている。",
    ),
    "none_uniform": (
        "0%カウンターへの開始側手順",
        "相手はカウンターせず、指だけ揺れる。指固定の癖を除いても"
        "対カウンター開幕が残るかを見る。",
    ),
    "counter_lowest": (
        "100%カウンターへの開始側手順",
        "相手は必ずカウンターする。開始側がフェイント/ロックから"
        "追加ターンや参照手順へ進めるかを見る。",
    ),
    "counter_uniform": (
        "100%カウンターへの開始側手順",
        "相手は必ずカウンターし、指だけ揺れる。対カウンター手順が"
        "指条件に依存せず残るかを見る。",
    ),
}


def copy_target_label(state_before, skill: object) -> str:
    if skill != COPY:
        return "-"
    if state_before.previous_skill is None:
        return "-"
    return skill_label(state_before.previous_skill)


def opening_action_label(action, state_before) -> str:
    label = action_label(action)
    target = copy_target_label(state_before, action.skill)
    if action.skill == COPY:
        return f"{label} / 参照={target}"
    return label


def side_result(actor_side: str, reward: float) -> str:
    if reward == 0:
        return "打切"
    start_won = (actor_side == "開始側" and reward > 0) or (
        actor_side != "開始側" and reward < 0
    )
    return "開始側勝利" if start_won else "開始側敗北"


def outcome_summary(counter: Counter[str]) -> str:
    return " / ".join(f"{label} {count}" for label, count in counter.most_common())


def opening_handoff_label(env: CompleteEnv) -> str:
    state_after = env._state
    previous = (
        skill_label(state_after.previous_skill)
        if state_after.previous_skill is not None
        else "-"
    )
    return (
        "手番交代: "
        f"開始側手={state_after.opp.hands}, 相手手={state_after.me.hands}, "
        f"直前={previous}"
    )


def analyze(
    model_path: str | Path,
    episodes: int = 80,
    max_steps: int = 300,
    seed_base: int = 9_500_000,
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
            first_actions: Counter[str] = Counter()
            first_skills: Counter[str] = Counter()
            opening_chains: Counter[tuple[str, str]] = Counter()
            opening_results: Counter[str] = Counter()
            opening_copy_targets: Counter[str] = Counter()
            start_copy_targets: Counter[str] = Counter()
            start_outcomes: Counter[str] = Counter()
            traces: list[dict[str, str]] = []
            total_steps = 0

            for episode in range(episodes):
                obs, _ = env.reset(seed=seed_base + episode * 7919)
                actor_side = "開始側"
                final_reward = 0.0
                final_truncated = False
                opening_open = True
                opening_actions: list[str] = []
                step = 0

                while True:
                    state_before = env._state
                    mask = env.action_masks()
                    action_idx, _ = model.predict(
                        obs,
                        action_masks=mask,
                        deterministic=deterministic,
                    )
                    tp_action = resolve_canonical_action(
                        canonical[int(action_idx)],
                        state_before,
                    )
                    label = action_label(tp_action)
                    copy_target = copy_target_label(state_before, tp_action.skill)

                    if actor_side == "開始側":
                        if tp_action.skill == COPY:
                            start_copy_targets[copy_target] += 1
                        if opening_open:
                            opening_label = opening_action_label(tp_action, state_before)
                            opening_actions.append(opening_label)
                            if step == 0:
                                first_actions[label] += 1
                                first_skills[skill_label(tp_action.skill)] += 1
                            if tp_action.skill == COPY:
                                opening_copy_targets[copy_target] += 1

                    obs, reward, terminated, truncated, info = env.step(int(action_idx))
                    final_reward = float(reward)
                    final_truncated = bool(truncated and not terminated)
                    step += 1

                    if episode == 0:
                        previous = (
                            skill_label(state_before.previous_skill)
                            if state_before.previous_skill is not None
                            else "-"
                        )
                        if terminated:
                            transition_result = f"終端報酬={final_reward:+.1f}"
                        elif truncated:
                            transition_result = "打切"
                        elif info.get("same_turn_player", False):
                            transition_result = "追加ターン"
                        else:
                            transition_result = "手番交代"
                        traces.append(
                            {
                                "step": str(step),
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
                                "result": transition_result,
                            }
                        )

                    if actor_side == "開始側" and opening_open:
                        if terminated:
                            opening_result = side_result(actor_side, final_reward)
                            opening_chains[(" -> ".join(opening_actions), opening_result)] += 1
                            opening_results[opening_result] += 1
                            opening_open = False
                        elif truncated:
                            opening_result = "開始側開幕中に打切"
                            opening_chains[(" -> ".join(opening_actions), opening_result)] += 1
                            opening_results[opening_result] += 1
                            opening_open = False
                        elif not info.get("same_turn_player", False):
                            opening_result = opening_handoff_label(env)
                            opening_chains[(" -> ".join(opening_actions), opening_result)] += 1
                            opening_results[opening_result] += 1
                            opening_open = False

                    if terminated or truncated:
                        break
                    if not info.get("same_turn_player", False):
                        actor_side = (
                            "手番交代後側" if actor_side == "開始側" else "開始側"
                        )

                total_steps += step
                start_outcome = "打切" if final_truncated else side_result(actor_side, final_reward)
                start_outcomes[start_outcome] += 1

            results[(mode_name, policy)] = {
                "mode": mode_name,
                "policy": policy,
                "episodes": episodes,
                "avg_steps": round(total_steps / max(episodes, 1), 1),
                "first_actions": first_actions,
                "first_skills": first_skills,
                "opening_chains": opening_chains,
                "opening_results": opening_results,
                "opening_copy_targets": opening_copy_targets,
                "start_copy_targets": start_copy_targets,
                "start_outcomes": start_outcomes,
                "trace": traces,
            }
    return results


def counter_rows(counter: Counter, total: int, limit: int | None = None) -> str:
    if not counter:
        return '<tr><td colspan="3">なし</td></tr>'
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


def opening_chain_rows(counter: Counter[tuple[str, str]], episodes: int) -> str:
    if not counter:
        return '<tr><td colspan="4">なし</td></tr>'
    rows = []
    for (chain, result), count in counter.most_common(10):
        rows.append(
            "<tr>"
            f"<td>{html.escape(chain or '-')}</td>"
            f"<td>{html.escape(result)}</td>"
            f"<td>{count}</td>"
            f"<td>{pct(count, episodes):.1f}%</td>"
            "</tr>"
        )
    return "\n".join(rows)


def expectation_warnings(data: dict) -> list[str]:
    first_total = max(sum(data["first_skills"].values()), 1)
    anti_openings = sum(
        count
        for skill, count in data["first_skills"].items()
        if skill in ANTI_COUNTER_SKILLS
    )
    anti_pct = pct(anti_openings, first_total)
    warnings: list[str] = []
    policy = data["policy"]

    if policy.startswith("none_") and anti_openings > 0:
        warnings.append(
            f"0%カウンターで初手フェイント/ロックが残る: {anti_pct:.1f}%"
        )
    if policy.startswith("counter_") and anti_pct < 80.0:
        warnings.append(
            f"100%カウンターで初手フェイント/ロックが不足: {anti_pct:.1f}%"
        )
    return warnings


def render_report(results: dict, model_path: Path, episodes: int) -> str:
    sections = []
    for mode_name, _ in MODES:
        for policy in POLICIES:
            data = results[(mode_name, policy)]
            warning_html = expectation_warnings(data)
            warn = (
                "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warning_html) + "</ul>"
                if warning_html
                else '<p class="ok">期待確認上の警告なし</p>'
            )
            expected_title, expected_text = EXPECTED_READS[policy]
            sections.append(
                f"""
                <details class="condition">
                  <summary>
                    <strong>{html.escape(mode_label(mode_name))} / {html.escape(policy_label(policy))}</strong>
                    <span>{html.escape(outcome_summary(data["start_outcomes"]))}</span>
                  </summary>
                  <p class="internal-name">内部表記: {html.escape(mode_name)} / {html.escape(policy)}</p>
                  <div class="expect">
                    <strong>{html.escape(expected_title)}</strong>
                    <p>{html.escape(expected_text)}</p>
                  </div>
                  <div class="chips">
                    <span>平均総手数 {data["avg_steps"]}</span>
                    <span>開始側コピー参照 {sum(data["start_copy_targets"].values())}回</span>
                    <span>開幕コピー参照 {sum(data["opening_copy_targets"].values())}回</span>
                  </div>
                  <div class="grid">
                    <section>
                      <h3>開始側最初の連続手番</h3>
                      <table><thead><tr><th>手順</th><th>区切り</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {opening_chain_rows(data["opening_chains"], episodes)}
                      </tbody></table>
                    </section>
                    <section>
                      <h3>開始側初手</h3>
                      <table><thead><tr><th>行動</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {counter_rows(data["first_actions"], episodes)}
                      </tbody></table>
                    </section>
                    <section>
                      <h3>開幕の区切り</h3>
                      <table><thead><tr><th>区切り</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {counter_rows(data["opening_results"], episodes)}
                      </tbody></table>
                    </section>
                    <section>
                      <h3>開始側の勝敗</h3>
                      <table><thead><tr><th>結果</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {counter_rows(data["start_outcomes"], episodes)}
                      </tbody></table>
                    </section>
                    <section>
                      <h3>開幕コピー対象</h3>
                      <table><thead><tr><th>参照先</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {counter_rows(data["opening_copy_targets"], max(sum(data["opening_copy_targets"].values()), 1))}
                      </tbody></table>
                    </section>
                    <section>
                      <h3>開始側コピー対象 全体</h3>
                      <table><thead><tr><th>参照先</th><th>回数</th><th>割合</th></tr></thead><tbody>
                        {counter_rows(data["start_copy_targets"], max(sum(data["start_copy_targets"].values()), 1))}
                      </tbody></table>
                    </section>
                  </div>
                  <h3>代表トレース</h3>
                  <table class="trace"><thead><tr><th>手</th><th>TP側</th><th>直前状態</th><th>TP行動</th><th>NTP反応</th><th>コピー対象</th><th>イベント</th><th>結果</th></tr></thead><tbody>
                    {trace_rows(data["trace"])}
                  </tbody></table>
                  <div class="warn"><strong>期待確認:</strong>{warn}</div>
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
        if name in {mode_name for mode_name, _ in MODES}
    )
    policy_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{html.escape(description)}</td>"
        "</tr>"
        for name, (label, description) in POLICY_DESCRIPTIONS.items()
        if name in set(POLICIES)
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>開始側手順診断</title>
  <style>
    body {{ margin: 0; background: #f6f6f2; color: #202124; font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif; line-height: 1.65; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ font-size: 28px; letter-spacing: 0; margin: 0 0 6px; }}
    h2 {{ font-size: 19px; letter-spacing: 0; margin: 0 0 10px; }}
    h3 {{ font-size: 15px; letter-spacing: 0; margin: 12px 0 7px; }}
    .lead, .internal-name {{ color: #5f6368; }}
    .panel, details.condition {{ background: #fff; border: 1px solid #d7d6cf; border-radius: 8px; margin: 15px 0; padding: 18px; }}
    details.condition > summary {{ align-items: baseline; cursor: pointer; display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between; }}
    details.condition > summary strong {{ font-size: 19px; }}
    details.condition > summary span {{ color: #5f6368; font-size: 13px; }}
    details.condition[open] > summary {{ border-bottom: 1px solid #d7d6cf; margin-bottom: 12px; padding-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .grid section {{ min-width: 0; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 12px; }}
    .chips span {{ background: #eef1ef; border: 1px solid #dce3dd; border-radius: 999px; font-size: 13px; padding: 3px 10px; }}
    .note, .expect {{ background: #edf6f5; border-left: 4px solid #0a6b6f; margin: 10px 0; padding: 10px 12px; }}
    .expect p {{ margin: 4px 0 0; }}
    .warn {{ background: #fff5e2; border-left: 4px solid #8a5300; margin-top: 12px; padding: 10px 12px; }}
    .ok {{ color: #246b32; margin: 4px 0; }}
    table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    th, td {{ border: 1px solid #d7d6cf; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1ef; }}
    .trace {{ display: block; overflow-x: auto; white-space: nowrap; }}
    .trace td {{ min-width: 82px; white-space: normal; }}
    .trace td:nth-child(3), .trace td:nth-child(7) {{ min-width: 180px; }}
  </style>
</head>
<body>
<main>
  <h1>開始側手順診断</h1>
  <p class="lead">モデル: {html.escape(str(model_path))} / 各条件 {episodes} episodes</p>
  <section class="panel">
    <h2>見るもの</h2>
    <p>
      開始側の最初の連続手番を切り出し、初手、追加ターン中の追撃、Copy の参照先、
      最初に手番を渡す時点の手数を並べます。勝率だけでなく、固定相手に対する
      開幕 exploit がどの方向へ進んでいるかを読むための診断です。
    </p>
    <div class="note">
      「開始側最初の連続手番」は初期局面から最初の手番交代または終局までです。
      0%カウンターではフェイント空振りを、100%カウンターでは対カウンター開幕と
      その後の参照手順を重点的に見ます。
    </div>
    <h3>評価モード</h3>
    <table><thead><tr><th>内部表記</th><th>日本語</th><th>意味</th></tr></thead><tbody>{mode_rows}</tbody></table>
    <h3>NTP条件</h3>
    <table><thead><tr><th>内部表記</th><th>日本語</th><th>意味</th></tr></thead><tbody>{policy_rows}</tbody></table>
  </section>
  {"".join(sections)}
</main>
</body>
</html>
"""


def generate_report(
    model_path: str | Path,
    output_path: str | Path = "results/start_side_policy_diagnostics.html",
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
    parser.add_argument("--output", default="results/start_side_policy_diagnostics.html")
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

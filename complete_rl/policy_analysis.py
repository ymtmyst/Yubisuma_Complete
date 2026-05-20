"""Policy analysis: visualise the action distribution of a trained TP model.

Simulates game episodes against various NTP policies and records every TP
action taken.  Produces a self-contained HTML report with Chart.js charts.

Usage
-----
>>> from complete_rl.policy_analysis import generate_policy_report
>>> generate_policy_report("results/my_model/maskable_ppo_complete.zip",
...                        output_path="results/policy_report.html")
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from complete_solver import RulesConfig
from complete_solver.constants import (
    ALL, ANTI_COUNTER_SKILLS, CHOICE, COPY, DROP, NORMAL_SKILLS,
    PASS, STOCK, ULTIMATE_TP_SKILLS,
)
from complete_rl.env import CompleteEnv, build_canonical_tp_actions, resolve_canonical_action


_NTP_POLICIES = ["none", "counter_first", "block_first", "random", "episode_mixed_basic"]
_NTP_LABELS = {
    "none": "none（反応なし）",
    "counter_first": "counter（カウンター優先）",
    "block_first": "block（ブロック優先）",
    "random": "random（ランダム）",
    "episode_mixed_basic": "episode_mixed（混合）",
}
_NTP_COLORS = {
    "none": "#4CAF50",
    "counter_first": "#F44336",
    "block_first": "#2196F3",
    "random": "#FF9800",
    "episode_mixed_basic": "#9C27B0",
}


def _categorize_skill(skill) -> str:
    if isinstance(skill, int) or (isinstance(skill, str) and skill.isdigit()):
        return "数字宣言"
    if skill == PASS:
        return "パス"
    if skill == ALL:
        return "オール"
    if skill == CHOICE:
        return "チョイス"
    if skill in NORMAL_SKILLS:
        return f"通常スキル/{skill}"
    if skill in ANTI_COUNTER_SKILLS:
        return f"対カウンター/{skill}"
    if skill in ULTIMATE_TP_SKILLS:
        return f"アルティメット/{skill}"
    if skill in (COPY, DROP, STOCK):
        return f"参照スキル/{skill}"
    return str(skill)


def _broad_category(skill) -> str:
    if isinstance(skill, int) or (isinstance(skill, str) and skill.isdigit()):
        return "数字宣言"
    if skill == PASS:
        return "パス"
    if skill == ALL:
        return "オール"
    if skill in ANTI_COUNTER_SKILLS:
        return "対カウタースキル"
    if skill in ULTIMATE_TP_SKILLS:
        return "アルティメット"
    return "スキル"


def analyze_policy(
    model_path: str | Path,
    config: RulesConfig = RulesConfig(),
    n_episodes: int = 200,
    max_steps: int = 300,
    seed_base: int = 0,
    ntp_policies: list[str] | None = None,
) -> dict:
    """Simulate episodes and collect action statistics."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("sb3-contrib required") from exc

    model = MaskablePPO.load(str(model_path))
    canonical = build_canonical_tp_actions(config)
    policies = ntp_policies or _NTP_POLICIES

    data: dict[str, dict] = {}
    for ntp in policies:
        env = CompleteEnv(
            config=config,
            opponent_policy=ntp,
            max_steps=max_steps,
        )
        wins = losses = truncations = 0
        total_steps = 0
        # Recorded per action
        all_skills: list[str] = []
        broad_cats: list[str] = []
        number_values: list[int] = []  # when skill is a number
        step_broad: list[tuple[int, str]] = []   # (step, broad_cat)

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_base + ep * 7919)
            done = False
            step = 0
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                can = canonical[int(action)]
                tp_act = resolve_canonical_action(can, env._state)
                skill = tp_act.skill

                all_skills.append(_categorize_skill(skill))
                bc = _broad_category(skill)
                broad_cats.append(bc)
                step_broad.append((step, bc))
                if isinstance(skill, int):
                    number_values.append(skill)

                obs, reward, terminated, truncated, _ = env.step(int(action))
                step += 1
                done = terminated or truncated

            total_steps += step
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                truncations += 1

        # Aggregate broad categories
        from collections import Counter
        broad_count = Counter(broad_cats)
        skill_count = Counter(all_skills)
        num_count = Counter(number_values)

        # Step-wise broad (first 3 steps vs rest)
        early = Counter(bc for s, bc in step_broad if s < 3)
        late = Counter(bc for s, bc in step_broad if s >= 3)

        data[ntp] = {
            "wins": wins,
            "losses": losses,
            "truncations": truncations,
            "win_rate": wins / n_episodes,
            "avg_steps": total_steps / n_episodes,
            "broad": dict(broad_count),
            "skills": dict(skill_count),
            "numbers": dict(num_count),
            "early_broad": dict(early),
            "late_broad": dict(late),
            "total_actions": len(all_skills),
        }

    return data


# ── HTML generation ──────────────────────────────────────────────────────────

_BROAD_COLORS = {
    "数字宣言": "#4CAF50",
    "スキル": "#2196F3",
    "対カウタースキル": "#FF9800",
    "アルティメット": "#9C27B0",
    "パス": "#9E9E9E",
    "オール": "#00BCD4",
}

_ALL_BROAD = ["数字宣言", "スキル", "対カウタースキル", "アルティメット", "オール", "パス"]


def generate_policy_report(
    model_path: str | Path,
    output_path: str | Path | None = None,
    config: RulesConfig = RulesConfig(),
    n_episodes: int = 200,
    max_steps: int = 300,
    ntp_policies: list[str] | None = None,
) -> Path:
    """Run analysis and write a self-contained HTML report."""
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.parent / "policy_report.html"
    output_path = Path(output_path)

    print(f"Analysing {n_episodes} episodes per NTP policy …")
    data = analyze_policy(
        model_path, config=config, n_episodes=n_episodes,
        max_steps=max_steps, ntp_policies=ntp_policies,
    )

    html = _build_html(data, model_path, n_episodes)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written → {output_path}")
    return output_path


def _pct(count: dict, key: str, total: int) -> float:
    return round(100.0 * count.get(key, 0) / max(total, 1), 1)


def _build_html(data: dict, model_path: Path, n_episodes: int) -> str:
    policies = list(data.keys())

    # ── Dataset: broad category stacked bar per NTP ───────────────────────
    broad_datasets = []
    for cat in _ALL_BROAD:
        values = []
        for ntp in policies:
            d = data[ntp]
            values.append(_pct(d["broad"], cat, d["total_actions"]))
        broad_datasets.append({
            "label": cat,
            "data": values,
            "backgroundColor": _BROAD_COLORS.get(cat, "#607D8B"),
        })

    # ── Dataset: number declarations histogram per NTP ────────────────────
    num_labels = [str(i) for i in range(5)]
    num_datasets = []
    for ntp in policies:
        d = data[ntp]
        total = d["total_actions"]
        vals = [_pct(d["numbers"], i, total) for i in range(5)]
        num_datasets.append({
            "label": _NTP_LABELS.get(ntp, ntp),
            "data": vals,
            "backgroundColor": _NTP_COLORS.get(ntp, "#607D8B") + "CC",
            "borderColor": _NTP_COLORS.get(ntp, "#607D8B"),
            "borderWidth": 1,
        })

    # ── Dataset: early (steps 0-2) vs late (steps 3+) broad per NTP ──────
    early_late_data = {}
    for ntp in policies:
        d = data[ntp]
        early_t = sum(d["early_broad"].values()) or 1
        late_t = sum(d["late_broad"].values()) or 1
        early_late_data[ntp] = {
            "early": {c: _pct(d["early_broad"], c, early_t) for c in _ALL_BROAD},
            "late": {c: _pct(d["late_broad"], c, late_t) for c in _ALL_BROAD},
        }

    # ── Dataset: top skills per NTP ───────────────────────────────────────
    skill_tables = {}
    for ntp in policies:
        d = data[ntp]
        total = d["total_actions"]
        sorted_skills = sorted(d["skills"].items(), key=lambda x: -x[1])
        skill_tables[ntp] = [
            {"skill": k, "count": v, "pct": _pct(d["skills"], k, total)}
            for k, v in sorted_skills[:15]
        ]

    # Serialise for JS
    js_broad = json.dumps(broad_datasets, ensure_ascii=False)
    js_num = json.dumps(num_datasets, ensure_ascii=False)
    js_el = json.dumps(early_late_data, ensure_ascii=False)
    js_skill = json.dumps(skill_tables, ensure_ascii=False)
    js_ntp_labels = json.dumps([_NTP_LABELS.get(p, p) for p in policies], ensure_ascii=False)
    js_policies = json.dumps(policies, ensure_ascii=False)
    js_summary = json.dumps(
        {
            ntp: {
                "wins": d["wins"],
                "losses": d["losses"],
                "truncations": d["truncations"],
                "win_rate": round(d["win_rate"] * 100, 1),
                "avg_steps": round(d["avg_steps"], 1),
            }
            for ntp, d in data.items()
        },
        ensure_ascii=False,
    )
    js_broad_cats = json.dumps(_ALL_BROAD, ensure_ascii=False)
    js_broad_colors = json.dumps([_BROAD_COLORS.get(c, "#607D8B") for c in _ALL_BROAD], ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>方策分析レポート — {model_path.parent.name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #333; }}
  header {{ background: #1976D2; color: #fff; padding: 20px 32px; }}
  header h1 {{ font-size: 1.4em; }}
  header p {{ font-size: 0.85em; opacity: 0.85; margin-top: 4px; }}
  .container {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
  .card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.12);
            padding: 20px; margin-bottom: 20px; }}
  h2 {{ font-size: 1.1em; color: #1976D2; margin-bottom: 14px; border-bottom: 2px solid #E3F2FD;
        padding-bottom: 6px; }}
  h3 {{ font-size: 0.95em; color: #555; margin: 14px 0 6px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr));
                   gap: 12px; }}
  .summary-cell {{ border-radius: 6px; padding: 14px; text-align: center; }}
  .summary-cell .label {{ font-size: 0.75em; color: #777; margin-bottom: 4px; }}
  .summary-cell .big {{ font-size: 1.6em; font-weight: 700; }}
  .summary-cell .sub {{ font-size: 0.75em; color: #555; margin-top: 2px; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media(max-width:700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  canvas {{ max-height: 340px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
  .tab {{ padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 0.82em;
           background: #E3F2FD; color: #1976D2; border: none; }}
  .tab.active {{ background: #1976D2; color: #fff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #F5F5F5; font-weight: 600; color: #555; }}
  .bar-cell {{ position: relative; }}
  .bar-inner {{ height: 14px; border-radius: 3px; background: #4CAF50; display: inline-block; min-width: 2px; }}
  .note {{ font-size: 0.78em; color: #888; margin-top: 8px; }}
</style>
</head>
<body>
<header>
  <h1>方策分析レポート</h1>
  <p>モデル: {model_path.parent.name} &nbsp;|&nbsp; エピソード数: {n_episodes} / NTP方策 &nbsp;|&nbsp;
     最大ステップ: 300</p>
</header>
<div class="container">

<!-- 1. Summary -->
<div class="card" id="summary-section">
  <h2>1. 勝敗サマリー</h2>
  <div id="summary-grid" class="summary-grid"></div>
</div>

<!-- 2. Broad action distribution -->
<div class="card">
  <h2>2. 行動カテゴリ分布（NTP方策別）</h2>
  <p class="note" style="margin-bottom:10px">各バーは全ステップ中の割合（%）。数字宣言 vs スキル の対比に注目。</p>
  <canvas id="broadChart"></canvas>
</div>

<!-- 3. Early vs Late -->
<div class="card">
  <h2>3. ゲーム序盤（0-2ステップ）vs 後半（3+ステップ）の行動比較</h2>
  <p class="note" style="margin-bottom:10px">OBS=123 の反応履歴が蓄積した後、行動パターンがどう変わるかを確認。</p>
  <div class="tabs" id="el-tabs"></div>
  <div class="chart-row">
    <div><h3>序盤（0-2ステップ）</h3><canvas id="earlyChart"></canvas></div>
    <div><h3>後半（3+ステップ）</h3><canvas id="lateChart"></canvas></div>
  </div>
</div>

<!-- 4. Number values -->
<div class="card">
  <h2>4. 宣言した数字の分布（0〜4）</h2>
  <p class="note" style="margin-bottom:10px">数字宣言がある場合、どの数字（0〜4）を選ぶかの割合（全行動に対する%）。</p>
  <canvas id="numChart"></canvas>
</div>

<!-- 5. Skill detail table -->
<div class="card">
  <h2>5. 使用スキル詳細（上位15件）</h2>
  <div class="tabs" id="skill-tabs"></div>
  <div id="skill-table-container"></div>
</div>

</div><!-- /container -->
<script>
const policies = {js_policies};
const ntpLabels = {js_ntp_labels};
const broadDatasets = {js_broad};
const numDatasets = {js_num};
const elData = {js_el};
const skillTables = {js_skill};
const summary = {js_summary};
const broadCats = {js_broad_cats};
const broadColors = {js_broad_colors};

// ── 1. Summary cards ───────────────────────────────────────────────────────
const grid = document.getElementById('summary-grid');
const statusColors = ['#4CAF50','#F44336','#FF9800','#2196F3','#9C27B0'];
policies.forEach((p, i) => {{
  const s = summary[p];
  const bg = statusColors[i % statusColors.length] + '18';
  const fg = statusColors[i % statusColors.length];
  const cell = document.createElement('div');
  cell.className = 'summary-cell';
  cell.style.background = bg;
  cell.innerHTML = `
    <div class="label">${{ntpLabels[i]}}</div>
    <div class="big" style="color:${{fg}}">${{s.win_rate}}%</div>
    <div class="sub">勝率 | ${{s.wins}}勝 ${{s.losses}}敗 ${{s.truncations}}打切</div>
    <div class="sub">平均 ${{s.avg_steps}} ステップ</div>`;
  grid.appendChild(cell);
}});

// ── 2. Broad stacked bar ───────────────────────────────────────────────────
new Chart(document.getElementById('broadChart'), {{
  type: 'bar',
  data: {{ labels: ntpLabels, datasets: broadDatasets }},
  options: {{
    plugins: {{ legend: {{ position: 'right' }} }},
    responsive: true,
    scales: {{
      x: {{ stacked: true }},
      y: {{ stacked: true, max: 100, ticks: {{ callback: v => v+'%' }} }}
    }}
  }}
}});

// ── 3. Early vs Late pie charts ────────────────────────────────────────────
let activeElPolicy = policies[0];
const elTabs = document.getElementById('el-tabs');
let earlyChart, lateChart;

function makePie(canvasId, vals) {{
  return new Chart(document.getElementById(canvasId), {{
    type: 'doughnut',
    data: {{
      labels: broadCats,
      datasets: [{{ data: vals, backgroundColor: broadColors, borderWidth: 1 }}]
    }},
    options: {{
      plugins: {{
        legend: {{ position: 'right', labels: {{ font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}}: ${{ctx.raw}}%` }} }}
      }},
      responsive: true,
    }}
  }});
}}

function updateElCharts(policy) {{
  const d = elData[policy];
  const earlyVals = broadCats.map(c => d.early[c] || 0);
  const lateVals  = broadCats.map(c => d.late[c]  || 0);
  if (earlyChart) earlyChart.destroy();
  if (lateChart)  lateChart.destroy();
  earlyChart = makePie('earlyChart', earlyVals);
  lateChart  = makePie('lateChart',  lateVals);
}}

policies.forEach((p, i) => {{
  const btn = document.createElement('button');
  btn.className = 'tab' + (i === 0 ? ' active' : '');
  btn.textContent = ntpLabels[i];
  btn.onclick = () => {{
    document.querySelectorAll('#el-tabs .tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    updateElCharts(p);
  }};
  elTabs.appendChild(btn);
}});
updateElCharts(policies[0]);

// ── 4. Number values grouped bar ──────────────────────────────────────────
new Chart(document.getElementById('numChart'), {{
  type: 'bar',
  data: {{ labels: ['0','1','2','3','4'], datasets: numDatasets }},
  options: {{
    plugins: {{ legend: {{ position: 'right' }} }},
    responsive: true,
    scales: {{
      y: {{ ticks: {{ callback: v => v+'%' }}, title: {{ display: true, text: '全行動中の割合 (%)' }} }}
    }}
  }}
}});

// ── 5. Skill detail tables ─────────────────────────────────────────────────
const skillTabsEl = document.getElementById('skill-tabs');
const skillContainer = document.getElementById('skill-table-container');

function makeSkillTable(policy) {{
  const rows = skillTables[policy];
  const maxPct = rows.length ? rows[0].pct : 1;
  let html = '<table><thead><tr><th>スキル</th><th>回数</th><th colspan="2">割合</th></tr></thead><tbody>';
  rows.forEach(r => {{
    const w = Math.round(r.pct / Math.max(maxPct, 0.1) * 180);
    html += `<tr>
      <td>${{r.skill}}</td>
      <td>${{r.count}}</td>
      <td>${{r.pct}}%</td>
      <td class="bar-cell"><span class="bar-inner" style="width:${{w}}px"></span></td>
    </tr>`;
  }});
  html += '</tbody></table>';
  return html;
}}

policies.forEach((p, i) => {{
  const btn = document.createElement('button');
  btn.className = 'tab' + (i === 0 ? ' active' : '');
  btn.textContent = ntpLabels[i];
  btn.onclick = () => {{
    document.querySelectorAll('#skill-tabs .tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    skillContainer.innerHTML = makeSkillTable(p);
  }};
  skillTabsEl.appendChild(btn);
}});
skillContainer.innerHTML = makeSkillTable(policies[0]);
</script>
</body>
</html>"""
    return html


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate policy analysis HTML report for a trained MaskablePPO model."
    )
    parser.add_argument("model_path")
    parser.add_argument("--output", help="Output HTML path (default: model_dir/policy_report.html)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--reversi", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = RulesConfig(enable_mirror=args.mirror, enable_reversi=args.reversi)
    generate_policy_report(
        args.model_path,
        output_path=args.output,
        config=config,
        n_episodes=args.episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

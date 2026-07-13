"""Generate results/index.html — a single clean hub linking every report.

Open results/index.html in a browser and everything worth viewing is one
click away. Re-run after producing new reports to refresh the hub:

    python -m complete_ai.build_report_hub
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "index.html"

# Curated primary reports (title, filename-relative-to-results, one-liner).
CURRENT = [
    ("現行AIの方策分布(最新)", "policy_report_value_latest_rules_v2.html",
     "使用スキル分布・全スキル統計(タイム等の合法vs選択)・決着分析。まずここ。"),
    ("N1 高速探索基盤ベンチマーク", "n1_fast_solver_benchmark.html",
     "深さ制限LP探索の高速化(値の一致検証と速度比)。"),
    ("N2 終盤厳密データベース", "n2_endgame_db_report.html",
     "終盤の厳密解テーブルと状態空間の規模。"),
    ("N3 価値ネット v0 精度検証", "n3_value_net_v0_report.html",
     "手作り評価との精度対決(厳密値・深い探索との相関)。"),
    ("N4 自己対戦世代ループ", "n4_generation_loop_report.html",
     "fitted Nash-VI の世代推移と診断。"),
    ("N5 実戦エージェント評価", "n5_baseline_eval_report.html",
     "固定相手への勝率・旧PPO対戦・対戦CLIの説明。"),
    ("ルール監査・裁定シート", "rules_audit_2026-07-13.html",
     "ルール実装の訂正記録(フェイント参照/ガード/タイム/スキップ)。"),
]

# Planning / logs (markdown & plan — open as text/HTML in the browser).
DOCS = [
    ("計画書 AI_MASTER_PLAN_V2", "../AI_MASTER_PLAN_V2.md",
     "現行アーキテクチャの全体計画とチェックリスト。"),
    ("作業ログ WORK_LOG", "../WORK_LOG.md",
     "N0〜N6 の全作業記録(時系列)。"),
    ("ルール文書 完全ルール(新)", "../完全ルール（新）",
     "ゲームの正式ルール。"),
]

# Endgame scenario solver outputs (Complete rules, not the Basic project).
ENDGAME = [
    ("終盤シナリオ depth1", "_legacy/endgame_scenarios/complete_lite/index_depth1.html"),
    ("終盤シナリオ depth2", "_legacy/endgame_scenarios/complete_lite_depth2/index_depth2.html"),
    ("終盤シナリオ v2", "_legacy/endgame_scenarios/complete_lite_v2/index_depth1.html"),
]

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN",
  "Yu Gothic", Meiryo, sans-serif; line-height: 1.6; color: #1a1a1a;
  background: #f6f7f9; }
.wrap { max-width: 880px; margin: 0 auto; padding: 48px 24px 80px; }
header h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px; letter-spacing: .01em; }
header p { margin: 0; color: #6b7280; font-size: 14px; }
.hero { display: block; margin: 28px 0 40px; padding: 22px 24px; border-radius: 14px;
  background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff;
  text-decoration: none; box-shadow: 0 8px 24px rgba(79,70,229,.28); transition: transform .12s; }
.hero:hover { transform: translateY(-2px); }
.hero .k { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; opacity: .85; }
.hero .t { font-size: 21px; font-weight: 700; margin: 4px 0 4px; }
.hero .d { font-size: 14px; opacity: .92; }
h2 { font-size: 15px; font-weight: 700; color: #374151; margin: 36px 0 14px;
  padding-bottom: 8px; border-bottom: 2px solid #e5e7eb; letter-spacing: .02em; }
.grid { display: grid; gap: 12px; }
a.card { display: block; padding: 15px 18px; border-radius: 11px; background: #fff;
  border: 1px solid #e5e7eb; text-decoration: none; color: inherit; transition: all .12s; }
a.card:hover { border-color: #a5b4fc; box-shadow: 0 4px 14px rgba(0,0,0,.06); transform: translateY(-1px); }
a.card .t { font-weight: 600; font-size: 15px; color: #111827; }
a.card .d { font-size: 13px; color: #6b7280; margin-top: 3px; }
.mini { display: flex; flex-wrap: wrap; gap: 8px; }
.mini a { font-size: 13px; padding: 7px 13px; border-radius: 8px; background: #fff;
  border: 1px solid #e5e7eb; text-decoration: none; color: #374151; }
.mini a:hover { border-color: #a5b4fc; color: #4f46e5; }
details { margin-top: 8px; }
summary { cursor: pointer; font-size: 13px; color: #6b7280; padding: 6px 0; }
footer { margin-top: 48px; font-size: 12px; color: #9ca3af; text-align: center; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1115; color: #e5e7eb; }
  header p { color: #9ca3af; }
  h2 { color: #cbd5e1; border-color: #1f2530; }
  a.card { background: #171a21; border-color: #232936; }
  a.card .t { color: #f3f4f6; }
  a.card:hover { border-color: #6366f1; }
  .mini a { background: #171a21; border-color: #232936; color: #cbd5e1; }
}
"""


def card(title: str, href: str, desc: str = "") -> str:
    d = f'<div class="d">{html.escape(desc)}</div>' if desc else ""
    return (f'<a class="card" href="{html.escape(href)}">'
            f'<div class="t">{html.escape(title)}</div>{d}</a>')


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    hero_title, hero_href, hero_desc = CURRENT[0]
    hero = (f'<a class="hero" href="{html.escape(hero_href)}">'
            f'<div class="k">最新・まずここ</div>'
            f'<div class="t">{html.escape(hero_title)}</div>'
            f'<div class="d">{html.escape(hero_desc)}</div></a>')

    reports = '<div class="grid">' + "".join(
        card(t, h, d) for t, h, d in CURRENT[1:]) + "</div>"

    docs = '<div class="grid">' + "".join(
        card(t, h, d) for t, h, d in DOCS) + "</div>"

    endgame = '<div class="mini">' + "".join(
        f'<a href="{html.escape(h)}">{html.escape(t)}</a>' for t, h in ENDGAME
    ) + "</div>"

    # Auto-discover the archived May diagnostics so nothing is silently lost.
    legacy_dir = RESULTS / "_legacy" / "diagnostics_may"
    legacy_files = sorted(legacy_dir.glob("*.html")) if legacy_dir.exists() else []
    legacy_links = "".join(
        f'<a href="_legacy/diagnostics_may/{html.escape(f.name)}">'
        f'{html.escape(f.stem)}</a>' for f in legacy_files
    )
    legacy = (
        f'<details><summary>過去の診断レポート（5月の旧BC+PPO路線・{len(legacy_files)}件）'
        f'</summary><div class="mini" style="margin-top:10px">{legacy_links}</div></details>'
    )

    doc = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yubisuma Complete AI — レポート集約</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>Yubisuma Complete AI レポート</h1>
  <p>ミラー・リバーシ OFF / 探索中心アーキテクチャ。このページ1枚から全レポートへ。</p>
</header>
{hero}
<h2>フェーズ別レポート・ルール監査</h2>
{reports}
<h2>計画・記録・ルール</h2>
{docs}
<h2>終盤シナリオ解析（Complete ソルバー）</h2>
{endgame}
<h2>アーカイブ</h2>
{legacy}
<footer>results/index.html — <code>python -m complete_ai.build_report_hub</code> で再生成</footer>
</div></body></html>"""

    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} (current reports: {len(CURRENT)}, "
          f"legacy diagnostics: {len(legacy_files)})", flush=True)


if __name__ == "__main__":
    main()

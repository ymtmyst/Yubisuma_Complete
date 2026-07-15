"""Generate results/index.html — a single clean hub linking every report.

Open results/index.html in a browser and everything worth viewing is one
click away. Re-run after producing new reports to refresh the hub:

    python -m complete_ai.build_report_hub

The hub has three ergonomic features, all managed from this file:

1. Collapsible sections (``details.sec``) so long lists fold away.
2. Readable docs: Markdown (.md) and the custom-notation 完全ルール are
   pre-rendered to styled HTML under ``results/docs/`` — clicking a card
   opens a clean page instead of raw text (raw .md is unreadable in a
   browser, and file:// cannot fetch to render client-side).
3. Status badges (🔥 進行中 / 🧱 基盤 / 📚 参考) + auto update dates so you
   can tell at a glance what is live vs. stable vs. archival.
"""

from __future__ import annotations

import html
import re
import sys
import time
from pathlib import Path

import markdown

RESULTS = Path("results")
DOCS_OUT = RESULTS / "docs"
OUT = RESULTS / "index.html"

# Status badges. Edit an entry's badge key below to re-classify it.
#   hot  = いま動かしている最新情報
#   base = 土台・安定（随時参照）
#   ref  = 参考・アーカイブ寄り
BADGES = {
    "hot":  ("hot",  "🔥 進行中"),
    "base": ("base", "🧱 基盤"),
    "ref":  ("ref",  "📚 参考"),
}

# Curated primary reports: (title, filename-relative-to-results, one-liner, badge).
CURRENT = [
    ("N7-A graph-vi モデルの方策分布", "policy_report_gvi.html",
     "サブグラフNash-VI教師で学習したモデルの方策分布(深さ教師版と比較用)。", "hot"),
    ("N7-A 深さ教師ベースラインの方策分布", "policy_report_depth_baseline.html",
     "並行完走した深さ教師12世代モデルの方策分布(graph-vi との対照)。", "base"),
    ("現行AIの方策分布(最新)", "policy_report_value_latest_rules_v2.html",
     "使用スキル分布・全スキル統計(タイム等の合法vs選択)・決着分析。まずここ。", "hot"),
    ("N7-G2 スキル別 setup-cost/価値プローブ", "skill_value_probe_2026-07-14.html",
     "未使用スキルの『発動済み価値 vs 宣言コスト vs AI使用率』一覧。ガードが最も break-even 近く・ロックは発動時強い・ストック消化はフェイント偏重。", "hot"),
    ("N7-G3 現行モデルのストック込み方策分布", "stock_policy_distribution_2026-07-15.html",
     "実ゲーム自己対戦の行動分布。ストック宣言率0.03%＝ゲームレベルでは事実上未使用。ガード/コピー/フラッシュは実戦で使用(9〜15%)。", "hot"),
    ("壊れストック因果実験（行動分布）", "broken_stock_experiment_2026-07-15.html",
     "ストックを段階的に壊れ性能にして学習→AIが使うか。基本0%→壊すと保持78〜87%で即採用＝0%は真に妥当。CHOICE後出しバグ修正込み。P1/P2/P3の詳細分布。", "hot"),
    ("N1 高速探索基盤ベンチマーク", "n1_fast_solver_benchmark.html",
     "深さ制限LP探索の高速化(値の一致検証と速度比)。", "base"),
    ("N2 終盤厳密データベース", "n2_endgame_db_report.html",
     "終盤の厳密解テーブルと状態空間の規模。", "ref"),
    ("N3 価値ネット v0 精度検証", "n3_value_net_v0_report.html",
     "手作り評価との精度対決(厳密値・深い探索との相関)。", "base"),
    ("N4 自己対戦世代ループ", "n4_generation_loop_report.html",
     "fitted Nash-VI の世代推移と診断。", "hot"),
    ("N5 実戦エージェント評価", "n5_baseline_eval_report.html",
     "固定相手への勝率・旧PPO対戦・対戦CLIの説明。", "hot"),
    ("ルール監査・裁定シート", "rules_audit_2026-07-13.html",
     "ルール実装の訂正記録(フェイント参照/ガード/タイム/スキップ)。", "base"),
]

# Planning / logs. Source docs are pre-rendered to styled HTML under docs/.
#   (title, source-relative-to-Complete, kind, out-name-in-docs, one-liner, badge)
#   kind: "md" = Markdown, "rule" = 完全ルール custom notation.
DOCS = [
    ("計画書 AI_MASTER_PLAN_V2", "AI_MASTER_PLAN_V2.md", "md",
     "AI_MASTER_PLAN_V2.html",
     "現行アーキテクチャの全体計画とチェックリスト。", "hot"),
    ("作業ログ WORK_LOG", "WORK_LOG.md", "md",
     "WORK_LOG.html",
     "N0〜N6 の全作業記録(時系列)。", "hot"),
    ("ルール文書 完全ルール(新)", "完全ルール（新）", "rule",
     "完全ルール.html",
     "ゲームの正式ルール(全スキルの厳密な効果)。", "base"),
    ("今後のアイデア FUTURE_IDEAS", "FUTURE_IDEAS.md", "md",
     "FUTURE_IDEAS.html",
     "未実装の拡張候補(ヒント/詰めパズル/感想戦/人対人/N7強化 ほか)。", "ref"),
    ("全体像 PROJECT_SUMMARY", "PROJECT_SUMMARY.md", "md",
     "PROJECT_SUMMARY.html",
     "プロジェクト全体のわかりやすい要約。", "base"),
    ("N7 戦略考察と設計案", "N7_STRATEGY_AND_DESIGN.md", "md",
     "N7_STRATEGY_AND_DESIGN.html",
     "深さを掘る以外で強くする路線の考察と設計案 A(サブグラフNash-VI)/B/C/D。", "hot"),
]

# Rulebook (child-friendly, served from the game's webplay folder).
RULEBOOK = ("📖 かんたんルールブック", "../complete_ai/webplay/rulebook.html",
            "わざの効果をやさしく説明（ビジュアル版）。", "base")

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
.hero { display: block; margin: 22px 0 10px; padding: 22px 24px; border-radius: 14px;
  background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff;
  text-decoration: none; box-shadow: 0 8px 24px rgba(79,70,229,.28); transition: transform .12s; }
.hero:hover { transform: translateY(-2px); }
.hero .k { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; opacity: .85; }
.hero .t { font-size: 21px; font-weight: 700; margin: 4px 0 4px; }
.hero .d { font-size: 14px; opacity: .92; }
.hero.play { background: linear-gradient(135deg, #16a34a, #059669);
  box-shadow: 0 8px 24px rgba(5,150,105,.32); margin: 28px 0 10px; }
.hero.play .t { font-size: 24px; }
.note { font-size: 13px; color: #6b7280; background: #fff; border: 1px solid #e5e7eb;
  border-left: 4px solid #16a34a; border-radius: 10px; padding: 12px 14px; margin: 8px 0 0; line-height: 1.6; }
.note code { background: #f3f4f6; padding: 1px 6px; border-radius: 5px; font-size: 12px; }
/* 開閉トグルのセクション */
details.sec { margin: 30px 0 0; }
details.sec > summary { list-style: none; cursor: pointer; font-size: 15px; font-weight: 700;
  color: #374151; padding: 8px 0; border-bottom: 2px solid #e5e7eb; letter-spacing: .02em;
  display: flex; align-items: center; gap: 8px; user-select: none; }
details.sec > summary::-webkit-details-marker { display: none; }
details.sec > summary .caret { transition: transform .15s; color: #9ca3af; font-size: 12px; }
details.sec[open] > summary .caret { transform: rotate(90deg); }
details.sec > summary .cnt { margin-left: auto; font-size: 12px; font-weight: 400; color: #9ca3af; }
details.sec > .grid, details.sec > .mini { margin-top: 14px; }
/* 状態バッジ */
.badge { display: inline-block; font-size: 11px; font-weight: 700; line-height: 1;
  padding: 4px 8px; border-radius: 999px; white-space: nowrap; }
.badge.hot  { background: #fee2e2; color: #b91c1c; }
.badge.base { background: #dbeafe; color: #1d4ed8; }
.badge.ref  { background: #eef0f3; color: #6b7280; }
.ch { display: flex; align-items: center; gap: 8px; }
.ch .t { flex: 1; }
.date { font-size: 11px; color: #9ca3af; margin-top: 6px; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 26px 0 0;
  font-size: 12px; color: #6b7280; }
.grid { display: grid; gap: 12px; }
@media (min-width: 620px) { .grid.two { grid-template-columns: 1fr 1fr; } }
a.card { display: block; padding: 15px 18px; border-radius: 11px; background: #fff;
  border: 1px solid #e5e7eb; text-decoration: none; color: inherit; transition: all .12s; }
a.card:hover { border-color: #a5b4fc; box-shadow: 0 4px 14px rgba(0,0,0,.06); transform: translateY(-1px); }
a.card .t { font-weight: 600; font-size: 15px; color: #111827; }
a.card .d { font-size: 13px; color: #6b7280; margin-top: 3px; }
.mini { display: flex; flex-wrap: wrap; gap: 8px; }
.mini a { font-size: 13px; padding: 7px 13px; border-radius: 8px; background: #fff;
  border: 1px solid #e5e7eb; text-decoration: none; color: #374151; }
.mini a:hover { border-color: #a5b4fc; color: #4f46e5; }
footer { margin-top: 48px; font-size: 12px; color: #9ca3af; text-align: center; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1115; color: #e5e7eb; }
  header p { color: #9ca3af; }
  .note { background: #171a21; border-color: #232936; color: #9ca3af; }
  .note code { background: #232936; }
  details.sec > summary { color: #cbd5e1; border-color: #1f2530; }
  .badge.hot  { background: #3b1618; color: #fca5a5; }
  .badge.base { background: #152238; color: #93c5fd; }
  .badge.ref  { background: #1b2028; color: #9ca3af; }
  a.card { background: #171a21; border-color: #232936; }
  a.card .t { color: #f3f4f6; }
  a.card .d { color: #9ca3af; }
  a.card:hover { border-color: #6366f1; }
  .mini a { background: #171a21; border-color: #232936; color: #cbd5e1; }
}
"""

# ── Rendered-document (docs/*.html) stylesheet ────────────────────────────
# Clean, readable typography for the Markdown / rule pages so opening a doc
# no longer dumps raw text at the reader.
DOC_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN",
  "Yu Gothic", Meiryo, sans-serif; line-height: 1.75; color: #1f2328;
  background: #f6f7f9; }
.wrap { max-width: 820px; margin: 0 auto; padding: 32px 24px 96px; }
.top { display: flex; align-items: center; gap: 12px; margin-bottom: 22px;
  padding-bottom: 16px; border-bottom: 1px solid #e5e7eb; }
.top a.back { font-size: 13px; color: #4f46e5; text-decoration: none; font-weight: 600;
  padding: 6px 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }
.top a.back:hover { border-color: #a5b4fc; }
.top .meta { font-size: 12px; color: #9ca3af; margin-left: auto; }
.doc { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px;
  padding: 36px 40px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.doc h1 { font-size: 26px; font-weight: 700; margin: 0 0 18px; padding-bottom: 12px;
  border-bottom: 2px solid #e5e7eb; letter-spacing: .01em; }
.doc h2 { font-size: 20px; font-weight: 700; margin: 34px 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid #eceef1; }
.doc h3 { font-size: 16px; font-weight: 700; margin: 26px 0 10px; color: #374151; }
.doc h4 { font-size: 14px; font-weight: 700; margin: 20px 0 8px; color: #4b5563; }
.doc p { margin: 12px 0; }
.doc ul, .doc ol { margin: 12px 0; padding-left: 26px; }
.doc li { margin: 5px 0; }
.doc a { color: #4f46e5; text-decoration: none; }
.doc a:hover { text-decoration: underline; }
.doc code { background: #f3f4f6; padding: 2px 6px; border-radius: 5px; font-size: 90%;
  font-family: "SFMono-Regular", "Consolas", "Menlo", monospace; }
.doc pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 10px;
  padding: 14px 16px; overflow-x: auto; line-height: 1.55; }
.doc pre code { background: none; padding: 0; font-size: 13px; }
.doc blockquote { margin: 14px 0; padding: 4px 16px; border-left: 4px solid #a5b4fc;
  color: #6b7280; background: #f9fafb; border-radius: 0 8px 8px 0; }
.doc table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px;
  display: block; overflow-x: auto; }
.doc th, .doc td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left;
  vertical-align: top; }
.doc th { background: #f3f4f6; font-weight: 700; }
.doc tr:nth-child(even) td { background: #fafbfc; }
.doc hr { border: none; border-top: 1px solid #e5e7eb; margin: 28px 0; }
.doc img { max-width: 100%; }
/* 完全ルール 専用 */
.doc details.rb { border: 1px solid #e5e7eb; border-radius: 10px; margin: 10px 0;
  background: #fbfbfc; overflow: hidden; }
.doc details.rb > summary { cursor: pointer; list-style: none; padding: 12px 16px;
  font-weight: 700; font-size: 15px; user-select: none; display: flex; align-items: center; gap: 8px; }
.doc details.rb > summary::-webkit-details-marker { display: none; }
.doc details.rb > summary::before { content: "▶"; color: #9ca3af; font-size: 11px;
  transition: transform .15s; }
.doc details.rb[open] > summary::before { transform: rotate(90deg); }
.doc details.rb > summary:hover { background: #f3f4f6; }
.doc .rbbody { padding: 4px 18px 14px; border-top: 1px solid #eceef1; }
.doc .rbbody > *:first-child { margin-top: 12px; }
.doc .skill { font-weight: 700; background: #eef2ff; color: #4338ca;
  padding: 1px 7px; border-radius: 6px; font-size: 92%; }
.doc .rnote { font-size: 13px; color: #6b7280; margin: 8px 0; }
.doc .rh3 { font-size: 15px; }
footer { margin-top: 40px; font-size: 12px; color: #9ca3af; text-align: center; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1115; color: #d1d5db; }
  .top { border-color: #232936; }
  .top a.back { background: #171a21; border-color: #232936; color: #a5b4fc; }
  .doc { background: #171a21; border-color: #232936; box-shadow: none; }
  .doc h1 { border-color: #232936; }
  .doc h2 { border-color: #1f2530; }
  .doc h3 { color: #cbd5e1; }
  .doc h4 { color: #b6bdc8; }
  .doc code { background: #232936; }
  .doc pre { background: #12151b; border-color: #232936; }
  .doc blockquote { background: #12151b; color: #9ca3af; border-color: #4338ca; }
  .doc th, .doc td { border-color: #232936; }
  .doc th { background: #1b2028; }
  .doc tr:nth-child(even) td { background: #12151b; }
  .doc hr { border-color: #232936; }
  .doc details.rb { background: #12151b; border-color: #232936; }
  .doc details.rb > summary:hover { background: #1b2028; }
  .doc .rbbody { border-color: #232936; }
  .doc .skill { background: #1e1b3a; color: #c4b5fd; }
}
"""


def fmt_date(path: Path) -> str:
    """'MM-DD' from a file's mtime, or '' when the file is missing."""
    if not path.exists():
        return ""
    return time.strftime("%m-%d", time.localtime(path.stat().st_mtime))


def badge_span(key: str) -> str:
    cls, label = BADGES[key]
    return f'<span class="badge {cls}">{label}</span>'


def card(title: str, href: str, desc: str = "", badge: str = "",
         date: str = "") -> str:
    """A report card with an optional status badge and update date."""
    b = badge_span(badge) if badge else ""
    head = (f'<div class="ch"><span class="t">{html.escape(title)}</span>{b}</div>'
            if badge else f'<div class="t">{html.escape(title)}</div>')
    d = f'<div class="d">{html.escape(desc)}</div>' if desc else ""
    dt = f'<div class="date">更新 {date}</div>' if date else ""
    return (f'<a class="card" href="{html.escape(href)}">{head}{d}{dt}</a>')


def section(title: str, count: int, body: str, is_open: bool = True) -> str:
    op = " open" if is_open else ""
    return (f'<details class="sec"{op}><summary><span class="caret">▶</span>'
            f'{html.escape(title)}<span class="cnt">{count}件</span></summary>'
            f'{body}</details>')


# ── Document rendering (make .md / 完全ルール readable) ────────────────────

def doc_page(title: str, source_label: str, date: str, body_html: str) -> str:
    meta = f'更新 {date} ・ {html.escape(source_label)}' if date else html.escape(source_label)
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{DOC_CSS}</style></head>
<body><div class="wrap">
<div class="top"><a class="back" href="../index.html">← ハブに戻る</a>
<span class="meta">{meta}</span></div>
<article class="doc"><h1>{html.escape(title)}</h1>
{body_html}
</article>
<footer>results/docs/ — <code>python -m complete_ai.build_report_hub</code> で再生成</footer>
</div></body></html>"""


def render_markdown(src: Path) -> str:
    md = markdown.Markdown(extensions=["extra", "sane_lists", "toc", "admonition"])
    return md.convert(src.read_text(encoding="utf-8"))


# --- 完全ルール custom-notation renderer -----------------------------------
# The rules doc uses a PukiWiki-ish notation:
#   *見出し / **小見出し   [+]title..[END] / [-]title..[END] (開閉ブロック)
#   ''太字''  %%%スキル%%%  &color(#hex){…}  &align(pos){…}  |セル|セル|  -箇条書き

def _balanced(s: str, start: int) -> tuple[str, int]:
    """Return (text inside braces, index past the matching '}')."""
    depth, i = 1, start
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], len(s)


def _inline(s: str) -> str:
    """Convert inline tokens to HTML, escaping all literal text."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s.startswith("''", i):
            j = s.find("''", i + 2)
            if j != -1:
                out.append("<strong>" + _inline(s[i + 2:j]) + "</strong>")
                i = j + 2
                continue
        if s.startswith("%%%", i):
            j = s.find("%%%", i + 3)
            if j != -1:
                out.append('<span class="skill">' + _inline(s[i + 3:j]) + "</span>")
                i = j + 3
                continue
        m = re.match(r"&color\(([^)]*)\)\{", s[i:])
        if m:
            inner, end = _balanced(s, i + m.end())
            color = html.escape(m.group(1).strip())
            out.append(f'<span style="color:{color}">' + _inline(inner) + "</span>")
            i = end
            continue
        m = re.match(r"&align\(([^)]*)\)\{", s[i:])
        if m:  # alignment: keep the text, drop the directive
            inner, end = _balanced(s, i + m.end())
            out.append(_inline(inner))
            i = end
            continue
        out.append(html.escape(s[i]))
        i += 1
    return "".join(out)


def render_rulebook(src: Path) -> str:
    lines = src.read_text(encoding="utf-8").splitlines()
    parts: list[str] = []
    li: list[str] = []
    tbl: list[str] = []

    def flush_list() -> None:
        if li:
            parts.append("<ul>" + "".join(f"<li>{x}</li>" for x in li) + "</ul>")
            li.clear()

    def flush_table() -> None:
        if tbl:
            rows = []
            for row in tbl:
                cells = [c for c in row.split("|")[1:-1]]
                rows.append("<tr>" + "".join(
                    f"<td>{_inline(c.strip())}</td>" for c in cells) + "</tr>")
            parts.append('<table>' + "".join(rows) + "</table>")
            tbl.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_list()
            flush_table()
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_list()
            tbl.append(stripped)
            continue
        flush_table()
        if line.startswith("[+]") or line.startswith("[-]"):
            flush_list()
            op = " open" if line.startswith("[-]") else ""
            title = _inline(line[3:].strip())
            parts.append(f'<details class="rb"{op}><summary>{title}</summary>'
                         f'<div class="rbbody">')
        elif stripped == "[END]":
            flush_list()
            parts.append("</div></details>")
        elif line.startswith("**"):
            flush_list()
            parts.append(f'<h3 class="rh3">{_inline(line[2:].strip())}</h3>')
        elif line.startswith("*"):
            flush_list()
            parts.append(f"<h2>{_inline(line[1:].strip())}</h2>")
        elif line.startswith("-"):
            li.append(_inline(line[1:].strip()))
        elif stripped.startswith("※"):
            flush_list()
            parts.append(f'<p class="rnote">{_inline(stripped)}</p>')
        else:
            flush_list()
            parts.append(f"<p>{_inline(stripped)}</p>")

    flush_list()
    flush_table()
    return "\n".join(parts)


def build_docs() -> dict[str, str]:
    """Render each planning/log/rule doc to docs/*.html. Returns title->href."""
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    hrefs: dict[str, str] = {}
    for title, src_rel, kind, out_name, _desc, _badge in DOCS:
        src = Path(src_rel)  # relative to Complete/ (cwd)
        if not src.exists():
            print(f"  ! skip {title}: source missing ({src})", flush=True)
            continue
        body = render_rulebook(src) if kind == "rule" else render_markdown(src)
        page = doc_page(title, src.name, fmt_date(src), body)
        (DOCS_OUT / out_name).write_text(page, encoding="utf-8")
        hrefs[title] = f"docs/{out_name}"
    return hrefs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    doc_hrefs = build_docs()

    hero_title, hero_href, hero_desc, _ = CURRENT[0]
    hero = (f'<a class="hero" href="{html.escape(hero_href)}">'
            f'<div class="k">最新・まずここ</div>'
            f'<div class="t">{html.escape(hero_title)}</div>'
            f'<div class="d">{html.escape(hero_desc)}</div></a>')

    # Launch button — uses the yubisuma:// custom protocol (register once via
    # AI対戦リンクを有効化.reg). Browsers cannot run a .bat from a file://
    # link directly, so this protocol handler is the standard workaround.
    play = (
        '<a class="hero play" href="yubisuma://play">'
        '<div class="k">クリックで起動</div>'
        '<div class="t">▶ AI と対戦する</div>'
        '<div class="d">初回のみ「AI対戦リンクを有効化.reg」を実行してください（下の注記）。'
        '起動には十数秒かかり、ブラウザで対戦画面が開きます。</div></a>')

    legend = ('<p class="legend">'
              f'{badge_span("hot")} いま動かしている最新情報　'
              f'{badge_span("base")} 土台・安定（随時参照）　'
              f'{badge_span("ref")} 参考・アーカイブ寄り</p>')

    # フェーズ別レポート・ルール監査
    report_cards = "".join(
        card(t, h, d, badge=b, date=fmt_date(RESULTS / h))
        for t, h, d, b in CURRENT[1:])
    reports = section("フェーズ別レポート・ルール監査", len(CURRENT) - 1,
                      f'<div class="grid two">{report_cards}</div>', is_open=True)

    # 計画・記録・ルール（ルールブック + 整形済みドキュメント）
    rb_t, rb_h, rb_d, rb_b = RULEBOOK
    doc_cards = [card(rb_t, rb_h, rb_d, badge=rb_b,
                      date=fmt_date(RESULTS / ".." / "complete_ai" / "webplay" / "rulebook.html"))]
    for t, src_rel, _kind, _out, d, b in DOCS:
        href = doc_hrefs.get(t)
        if href is None:
            continue
        doc_cards.append(card(t, href, d, badge=b, date=fmt_date(Path(src_rel))))
    docs_html = section("計画・記録・ルール", len(doc_cards),
                        f'<div class="grid two">{"".join(doc_cards)}</div>', is_open=True)

    # 終盤シナリオ解析
    endgame_links = "".join(
        f'<a href="{html.escape(h)}">{html.escape(t)}</a>' for t, h in ENDGAME)
    endgame = section("終盤シナリオ解析（Complete ソルバー）", len(ENDGAME),
                      f'<div class="mini">{endgame_links}</div>', is_open=False)

    # アーカイブ（5月の旧診断を自動収集）
    legacy_dir = RESULTS / "_legacy" / "diagnostics_may"
    legacy_files = sorted(legacy_dir.glob("*.html")) if legacy_dir.exists() else []
    legacy_links = "".join(
        f'<a href="_legacy/diagnostics_may/{html.escape(f.name)}">'
        f'{html.escape(f.stem)}</a>' for f in legacy_files)
    legacy = section("アーカイブ（5月の旧BC+PPO路線）", len(legacy_files),
                     f'<div class="mini">{legacy_links}</div>', is_open=False)

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
{play}
<p class="note">💡 <b>「▶ AI と対戦する」を使うための初回セットアップ</b>：
Complete フォルダの <code>AI対戦リンクを有効化.reg</code> をダブルクリックし「はい」を押してください（1回だけ）。
以降このボタンでゲームが起動します。うまくいかない時は <code>AIと対戦.bat</code> を直接ダブルクリックでもOK。</p>
{hero}
{legend}
{reports}
{docs_html}
{endgame}
{legacy}
<footer>results/index.html — <code>python -m complete_ai.build_report_hub</code> で再生成。バッジ🔥/🧱/📚はこのファイルで管理、更新日は各ファイルの更新時刻。</footer>
</div></body></html>"""

    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} (reports: {len(CURRENT)}, docs: {len(doc_hrefs)}, "
          f"legacy: {len(legacy_files)})", flush=True)


if __name__ == "__main__":
    main()

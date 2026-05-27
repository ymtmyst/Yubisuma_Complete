"""Generate an HTML comparison of zero/material BC teacher distributions."""

from __future__ import annotations

import html
from pathlib import Path

from complete_rl.bc_objective_diagnostics import (
    SCENARIOS,
    finite_horizon_rows,
    render_rows,
    vi_teacher_rows,
)


def warning_text(zero_rows: list[dict], material_rows: list[dict]) -> str:
    warnings: list[str] = []
    if zero_rows:
        top = zero_rows[0]
        if top["category"] == "数字宣言" and top["prob"] >= 80.0:
            warnings.append(f"zero leaf の教師が数字宣言に過集中: {top['prob']:.1f}%")
    if material_rows:
        top = material_rows[0]
        if top["category"] == "数字宣言" and top["prob"] >= 80.0:
            warnings.append(f"material leaf でも数字宣言に過集中: {top['prob']:.1f}%")
    if not warnings:
        return "<p class=\"ok\">強い数字過集中はなし</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warnings) + "</ul>"


def generate_report(
    output_path: Path = Path("results/bc_teacher_distribution_check.html"),
) -> Path:
    sections = []
    for scenario in SCENARIOS:
        depth3_value, depth3_rows = finite_horizon_rows(scenario.state, 3)
        depth4_value, depth4_rows = finite_horizon_rows(scenario.state, 4)
        zero = vi_teacher_rows(scenario.state, 400, use_material_leaf=False)
        material = vi_teacher_rows(scenario.state, 400, use_material_leaf=True)
        sections.append(
            f"""
            <section class="section">
              <h2>{html.escape(scenario.title)}</h2>
              <p>{html.escape(scenario.note)}</p>
              <div class="grid">
                <div>
                  <h3>参考: finite horizon depth=3 / value={depth3_value:+.4f}</h3>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(depth3_rows)}</tbody></table>
                </div>
                <div>
                  <h3>★採用候補: finite horizon depth=4 / value={depth4_value:+.4f}</h3>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(depth4_rows)}</tbody></table>
                </div>
                <div>
                  <h3>zero leaf BC教師 / value={zero[0]:+.4f}</h3>
                  <p class="meta">states={zero[1]}, converged={zero[2]}, delta={zero[3]:.2e}</p>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(zero[4])}</tbody></table>
                </div>
                <div>
                  <h3>material leaf BC教師 / value={material[0]:+.4f}</h3>
                  <p class="meta">states={material[1]}, converged={material[2]}, delta={material[3]:.2e}</p>
                  <table><thead><tr><th>行動</th><th>分類</th><th>確率</th><th>メモ</th></tr></thead><tbody>{render_rows(material[4])}</tbody></table>
                </div>
              </div>
              <div class="warn"><strong>警告:</strong>{warning_text(zero[4], material[4])}</div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>BC教師分布チェック</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #202124; font-family: "Segoe UI", "Yu Gothic", "Meiryo", sans-serif; line-height: 1.7; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 12px 0 8px; font-size: 15px; letter-spacing: 0; }}
    .lead {{ color: #5f6368; margin-bottom: 18px; }}
    .section {{ background: #fff; border: 1px solid #d8d7d0; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .note {{ border-left: 4px solid #0b6b6f; background: #edf7f7; padding: 10px 12px; margin: 12px 0; }}
    .warn {{ border-left: 4px solid #8a5300; background: #fff6e5; padding: 10px 12px; margin: 12px 0; }}
    .ok {{ color: #246b32; margin: 4px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d8d7d0; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #eef1ef; text-align: left; }}
    .meta {{ color: #5f6368; font-size: 12px; margin: 0 0 6px; }}
    code {{ padding: 1px 5px; background: #eef1ef; border: 1px solid #dde3df; border-radius: 4px; font-family: Consolas, "Courier New", monospace; }}
  </style>
</head>
<body>
<main>
  <h1>BC教師分布チェック</h1>
  <p class="lead">zero leaf と material leaf のBC教師を代表局面で比較 / 再学習前の確認用</p>
  <section class="section">
    <h2>確認意図</h2>
    <p>
      現行の <code>zero leaf</code> 教師が数字宣言へ過集中していたため、
      <code>material leaf</code> に切り替えることで教師分布が人間視点に近づくか確認します。
      外し数字は完全禁止ではありませんが、通常局面で80%以上に集中する場合は警告します。
    </p>
  </section>
  {"".join(sections)}
  <section class="section">
    <h2>AI側の解釈</h2>
    <p>
      material leaf で数字100%が崩れるなら、次は短時間のBC smoke学習でモデル方策に反映されるかを確認します。
      material leaf でも数字過集中が残る場合は、finite horizon教師またはシナリオ教師とのハイブリッドを検討します。
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

"""Exact stock-alphabet importance analysis over solved (1,1) universes.

Every A5-reachable state is also A8-reachable (restricting the stock alphabet
only removes legal actions), so for shared states the difference
``V_restricted − V_full`` is the EXACT value cost of forbidding the excluded
stocks for both players. Comparing the alphabet ladder A0 ⊂ A4 ⊂ A5 ⊂ A6 ⊂ A8
attributes that cost skill by skill — this replaces intuition ("stocked cement
is worthless in the endgame") with exact numbers.

Run after the batch job:  python -m complete_solver.abstraction_error_report
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

from .packed_vi import PackedEndgameDB

DATA_DIR = Path("data")
OUT_PATH = Path("results/n2b_endgame_table_report.html")

LADDER = [
    ("A0", "ストック全面禁止"),
    ("s1_A4", "保持1個まで/フェイント・ロック・フラッシュ・ガード"),
    ("s1_A5", "保持1個まで/A4+スキップ"),
    ("s1_A6", "保持1個まで/A5+クイック"),
    ("s1_A8", "保持1個まで/全8種"),
    ("s2_A4", "保持2個まで/A4"),
]
BASELINE_ORDER = ["s2_A4", "s1_A8", "s1_A6", "s1_A5", "s1_A4", "A0"]


def join_values(small: PackedEndgameDB, big: PackedEndgameDB):
    """Return (v_small, v_big) aligned on the states shared by both DBs."""
    # Merge on (keys0, keys1), both sorted lexicographically.
    small_view = np.stack([small.keys0, small.keys1], axis=1)
    big_view = np.stack([big.keys0, big.keys1], axis=1)

    # Positions of small's keys inside big via two-level searchsorted.
    pos = np.searchsorted(big.keys0, small.keys0, side="left")
    matched_small = []
    matched_big = []
    # Vectorised refinement: for equal keys0 runs, binary search keys1.
    # (loop in numpy-chunks; small enough at ≤ tens of millions)
    hi_all = np.searchsorted(big.keys0, small.keys0, side="right")
    for i in range(len(small.keys0)):
        lo = pos[i]
        hi = hi_all[i]
        if lo == hi:
            continue
        j = lo + np.searchsorted(big.keys1[lo:hi], small.keys1[i], side="left")
        if j < hi and big.keys1[j] == small.keys1[i]:
            matched_small.append(i)
            matched_big.append(j)
    small_idx = np.array(matched_small, dtype=np.int64)
    big_idx = np.array(matched_big, dtype=np.int64)
    return small.values[small_idx], big.values[big_idx], len(small_view), len(big_view)


def describe(diff: np.ndarray) -> dict:
    abs_diff = np.abs(diff)
    return {
        "shared_states": int(diff.size),
        "max_abs": float(abs_diff.max()) if diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if diff.size else 0.0,
        "p999": float(np.quantile(abs_diff, 0.999)) if diff.size else 0.0,
        "over_0_01": int((abs_diff > 0.01).sum()),
        "over_0_05": int((abs_diff > 0.05).sum()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    dbs: dict[str, PackedEndgameDB] = {}
    infos: dict[str, dict] = {}
    summary_path = DATA_DIR / "h11_batch_summary.json"
    batch_info = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    for name, _ in LADDER:
        path = DATA_DIR / f"endgame_h11_{name}.npz"
        if path.exists():
            dbs[name] = PackedEndgameDB.load(path)
            infos[name] = batch_info.get(name, {})
            print(f"loaded {name}: {len(dbs[name])} states")

    baseline_name = next((n for n in BASELINE_ORDER if n in dbs), None)
    if baseline_name is None:
        raise SystemExit("no solved DB found — run the batch job first")
    full = dbs[baseline_name]
    print(f"baseline (richest solved universe): {baseline_name}")

    ladder_rows = []
    for name, label in LADDER:
        if name not in dbs:
            continue
        info = infos.get(name, {})
        ladder_rows.append(
            f"<tr><td>{name}</td><td>{label}</td><td>{len(dbs[name]):,}</td>"
            f"<td>{info.get('iterations', '?')}</td>"
            f"<td>{info.get('vi_seconds', 0):.0f}s</td></tr>"
        )

    # Error of each restricted DB vs the full solution.
    error_rows = []
    step_rows = []
    for name, label in LADDER:
        if name not in dbs or name == baseline_name:
            continue
        v_small, v_big, n_small, _ = join_values(dbs[name], full)
        stats = describe(v_small - v_big)
        error_rows.append(
            f"<tr><td>{name}({label})</td><td>{stats['shared_states']:,}"
            f" / {n_small:,}</td><td>{stats['max_abs']:.4f}</td>"
            f"<td>{stats['mean_abs']:.5f}</td><td>{stats['p999']:.4f}</td>"
            f"<td>{stats['over_0_01']:,}</td><td>{stats['over_0_05']:,}</td></tr>"
        )
        print(f"{name} vs {baseline_name}: {stats}")

    # Axis-by-axis attribution along the ladder (exact on shared states).
    attribution = [
        ("A0→s1_A4", "重要4種を1個持てる価値", "A0", "s1_A4"),
        ("s1_A4→s1_A5", "スキップのストック価値", "s1_A4", "s1_A5"),
        ("s1_A5→s1_A6", "クイックのストック価値", "s1_A5", "s1_A6"),
        ("s1_A6→s1_A8", "セメント/チャージのストック価値", "s1_A6", "s1_A8"),
        ("s1_A4→s2_A4", "重要4種の2個目保持の価値", "s1_A4", "s2_A4"),
    ]
    for key, label, small_name, big_name in attribution:
        if small_name not in dbs or big_name not in dbs:
            continue
        v_small, v_big, _, _ = join_values(dbs[small_name], dbs[big_name])
        stats = describe(v_small - v_big)
        step_rows.append(
            f"<tr><td>{key}</td><td>{label}</td><td>{stats['max_abs']:.4f}</td>"
            f"<td>{stats['mean_abs']:.5f}</td><td>{stats['over_0_01']:,}</td></tr>"
        )
        print(f"{key}: {stats}")

    content = f"""<meta charset="utf-8">
<title>N2b (1,1)終盤テーブルとストック重要度の厳密分析</title>
<h1>N2b 終盤テーブル構築とストック重要度の厳密分析</h1>
<p>対象: ミラー・リバーシOFF / (1,1)終盤宇宙 / gamma=0.999。
値はすべて<b>厳密</b>(閉集合上のShapley価値反復、Bellman残差検証済み)。</p>

<h2>1. 解いた宇宙(アルファベット段階)</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>名前</th><th>ストック可能スキル</th><th>状態数</th><th>収束スイープ数</th><th>VI時間</th></tr>
{''.join(ladder_rows)}
</table>

<h2>2. 制限版の誤差(A8_full = 真の値 と比較)</h2>
<p>「両者がそのストックを禁じられた場合に失われる/得られる価値」の分布。
max_abs が小さいほど、そのアルファベットで十分ということ。</p>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>アルファベット</th><th>共有状態数/全状態</th><th>最大|誤差|</th><th>平均|誤差|</th><th>99.9%点</th><th>|誤差|&gt;0.01の状態数</th><th>&gt;0.05</th></tr>
{''.join(error_rows)}
</table>

<h2>3. スキル別ストック重要度(はしご差分)</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>段階</th><th>意味</th><th>最大|差|</th><th>平均|差|</th><th>|差|&gt;0.01の状態数</th></tr>
{''.join(step_rows)}
</table>

<h2>4. AIの解釈</h2>
<p>(数値確認後に追記)</p>
"""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

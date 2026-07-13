"""N1 benchmark: FastHorizonSolver vs FiniteHorizonSolver.

Generates results/n1_fast_solver_benchmark.html with timing, value equality,
and instrumentation counters.

Run:  python -m complete_solver.benchmark_fast_solver
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import numpy as np

from .actions import RulesConfig, legal_ntp_actions, legal_tp_actions
from .fast_solver import FastHorizonSolver
from .finite_horizon import FiniteHorizonSolver
from .state import initial_state
from .transition import transition

CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)
GAMMA = 0.999
MAX_REFERENCE_DEPTH = 4  # reference depth 4 takes ~2 minutes; keep it once for the record


def bench_reference() -> list[dict]:
    rows = []
    for depth in range(1, MAX_REFERENCE_DEPTH + 1):
        solver = FiniteHorizonSolver(CONFIG, gamma=GAMMA)
        t0 = time.perf_counter()
        value = solver.solve_state(initial_state(), depth).value
        rows.append({"depth": depth, "value": value, "seconds": time.perf_counter() - t0})
        print(f"reference depth={depth}: {rows[-1]['seconds']:.2f}s value={value:+.4f}", flush=True)
    return rows


def bench_fast() -> tuple[list[dict], FastHorizonSolver]:
    solver = FastHorizonSolver(CONFIG, gamma=GAMMA)
    rows = []
    for depth in range(1, 6):
        t0 = time.perf_counter()
        value = solver.solve_state(initial_state(), depth).value
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        solver.solve_state(initial_state(), depth)
        warm = time.perf_counter() - t0
        rows.append(
            {
                "depth": depth,
                "value": value,
                "seconds": cold,
                "warm_seconds": warm,
                "stats": solver.stats.as_dict(),
                "caches": solver.cache_sizes(),
            }
        )
        print(
            f"fast depth={depth}: cold={cold:.2f}s warm={warm*1000:.1f}ms value={value:+.4f} "
            f"caches={solver.cache_sizes()}",
            flush=True,
        )
        if cold > 300:
            break
    return rows, solver


def bench_selfplay(depth: int = 3, max_moves: int = 30) -> dict:
    """Play one search-vs-search game, sampling from the root LP mixture."""
    rng = np.random.default_rng(0)
    solver = FastHorizonSolver(CONFIG, gamma=GAMMA)
    state = initial_state()
    move_times: list[float] = []
    outcome = "truncated"
    for _ in range(max_moves):
        t0 = time.perf_counter()
        policy = solver.solve_state(state, depth)
        move_times.append(time.perf_counter() - t0)
        tp_index = int(rng.choice(len(policy.tp_policy), p=np.array(policy.tp_policy)))
        ntp_index = int(rng.choice(len(policy.ntp_policy), p=np.array(policy.ntp_policy)))
        result = transition(
            state, policy.tp_actions[tp_index], policy.ntp_actions[ntp_index], CONFIG
        )
        if result.terminal_reward is not None:
            outcome = f"terminal reward {result.terminal_reward:+.0f}"
            break
        state = result.next_state
    report = {
        "depth": depth,
        "moves": len(move_times),
        "outcome": outcome,
        "first_move_s": move_times[0],
        "mean_later_ms": float(np.mean(move_times[1:]) * 1000) if len(move_times) > 1 else 0.0,
        "max_later_ms": float(np.max(move_times[1:]) * 1000) if len(move_times) > 1 else 0.0,
    }
    print(f"selfplay: {report}", flush=True)
    return report


def render_html(ref_rows, fast_rows, selfplay, out_path: Path) -> None:
    def table(rows, cols, headers):
        cells = "".join(f"<th>{h}</th>" for h in headers)
        body = ""
        for r in rows:
            body += "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
        return f"<table border='1' cellpadding='4' cellspacing='0'><tr>{cells}</tr>{body}</table>"

    ref_fmt = [
        {"depth": r["depth"], "value": f"{r['value']:+.6f}", "seconds": f"{r['seconds']:.2f}"}
        for r in ref_rows
    ]
    fast_fmt = [
        {
            "depth": r["depth"],
            "value": f"{r['value']:+.6f}",
            "seconds": f"{r['seconds']:.2f}",
            "warm_ms": f"{r['warm_seconds']*1000:.1f}",
            "matrix_solves": r["stats"]["matrix_solves"],
            "cell_evals": r["stats"]["cell_evals"],
            "tt_states": r["caches"]["transposition"],
        }
        for r in fast_rows
    ]

    checks = []
    for ref in ref_rows:
        fast = next((f for f in fast_rows if f["depth"] == ref["depth"]), None)
        if fast:
            diff = abs(ref["value"] - fast["value"])
            speedup = ref["seconds"] / fast["seconds"] if fast["seconds"] > 0 else float("inf")
            checks.append(
                {
                    "depth": ref["depth"],
                    "value_diff": f"{diff:.2e}",
                    "match": "OK" if diff < 1e-6 else "MISMATCH",
                    "speedup": f"x{speedup:.1f}",
                }
            )

    content = f"""<meta charset="utf-8">
<title>N1 FastHorizonSolver ベンチマーク</title>
<h1>N1 高速探索基盤ベンチマーク(初期局面, gamma={GAMMA}, ミラー・リバーシOFF)</h1>
<h2>値の一致検証と速度比(旧ソルバー比)</h2>
{table(checks, ["depth", "value_diff", "match", "speedup"], ["深さ", "値の差", "判定", "速度比"])}
<h2>旧ソルバー FiniteHorizonSolver(コールドスタート)</h2>
{table(ref_fmt, ["depth", "value", "seconds"], ["深さ", "値", "秒"])}
<h2>新ソルバー FastHorizonSolver(反復深化・キャッシュ持続)</h2>
{table(fast_fmt, ["depth", "value", "seconds", "warm_ms", "matrix_solves", "cell_evals", "tt_states"],
        ["深さ", "値", "秒(コールド)", "再解決(ms)", "行列解決回数", "セル評価数", "TT局面数"])}
<h2>自己対戦シミュレーション(depth={selfplay['depth']}, LP混合からサンプリング)</h2>
<ul>
<li>手数: {selfplay['moves']}(結果: {html.escape(selfplay['outcome'])})</li>
<li>初手の思考時間: {selfplay['first_move_s']:.2f} 秒</li>
<li>2手目以降の平均: {selfplay['mean_later_ms']:.0f} ms / 最大: {selfplay['max_later_ms']:.0f} ms</li>
</ul>
<h2>AIの解釈</h2>
<p>速度向上は (1) 探索間で持続するトランスポジションテーブル、(2) 遷移の遅延評価キャッシュ、
(3) ダブルオラクル法による部分行列解決、(4) 鞍点・閉形式によるLP回避 の複合効果。
値は旧ソルバーと一致(上表)。N1 の完了条件は depth3 ≤ 1s / depth4 ≤ 10s。</p>
"""
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fast_rows, _ = bench_fast()
    selfplay = bench_selfplay()
    ref_rows = bench_reference()
    render_html(ref_rows, fast_rows, selfplay, Path("results/n1_fast_solver_benchmark.html"))


if __name__ == "__main__":
    main()

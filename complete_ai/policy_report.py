"""Skill-usage / policy distribution report for a value-net agent (standing
user requirement: every new model gets one of these, in HTML).

Two views:
1. DIRECT policy — the exact root LP mixture (probabilities, not samples) at
   representative states (opening, (1,1) endgame root).
2. PLAY statistics — self-play games sampled from the LP mixture: TP action
   categories and NTP reactions, broken down by game phase (total hands:
   4 = 序盤, 3 = 中盤, ≤2 = 終盤).

Run:  python -m complete_ai.policy_report --model models/value_latest.pt \
          --out results/policy_report_value_latest.html
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from complete_solver.endgame_abstraction import h11_root
from complete_solver.packed_engine import (
    PASS_CODE,
    SKILL_NAMES,
    code_to_ntp_action,
    pack_state,
    step,
)
from complete_solver.state import initial_state

from .agents import SearchAgent
from .batched_search import BatchedSearcher, _FULL_MASK, _NO_CAP
from .generation_loop import load_model

PHASES = ("序盤(手4)", "中盤(手3)", "終盤(手2以下)")
REACTIONS = ("なし", "カウンター", "ブロック")


def tp_label(code: int) -> str:
    if code == PASS_CODE:
        return "パス"
    if code < 64:
        return f"数字{code // 4}"
    if code < 128:
        return SKILL_NAMES[(code - 64) // 4]
    return f"チョイス→{SKILL_NAMES[(code - 128) // 4]}"


def tp_category(code: int) -> str:
    if code == PASS_CODE:
        return "パス"
    if code < 64:
        return "数字宣言"
    if code < 128:
        return SKILL_NAMES[(code - 64) // 4]
    return f"チョイス→{SKILL_NAMES[(code - 128) // 4]}"


def phase_of(lane0: int, lane1: int) -> int:
    total = (lane0 & 3) + (lane1 & 3)
    if total >= 4:
        return 0
    if total == 3:
        return 1
    return 2


def direct_mixture_table(searcher: BatchedSearcher, lane0: int, lane1: int,
                         min_prob: float = 0.005) -> str:
    value, tp_codes, ntp_codes, tp_policy, ntp_policy = searcher.solve(
        int(lane0), int(lane1)
    )
    rows = sorted(
        ((float(p), tp_label(int(c))) for c, p in zip(tp_codes, tp_policy)),
        reverse=True,
    )
    body = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{p * 100:.1f}%</td></tr>"
        for p, label in rows if p >= min_prob
    )
    ntp_rows = sorted(
        ((float(p), f"{code_to_ntp_action(int(c)).reaction}"
                    f"(親指{int(c) % 4})")
         for c, p in zip(ntp_codes, ntp_policy)),
        reverse=True,
    )
    ntp_body = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{p * 100:.1f}%</td></tr>"
        for p, label in ntp_rows if p >= min_prob
    )
    return (
        f"<p>局面評価値(手番側視点): {value:+.4f}</p>"
        "<table border='1' cellpadding='4' cellspacing='0'><tr>"
        "<th>宣言(TP)</th><th>確率</th></tr>" + body + "</table>"
        "<p>反応側(NTP)の混合:</p>"
        "<table border='1' cellpadding='4' cellspacing='0'><tr>"
        "<th>反応</th><th>確率</th></tr>" + ntp_body + "</table>"
    )


_BOOST_CODES = set(range(64 + 13 * 4, 64 + 13 * 4 + 4))

# Canonical TP category order — every skill is always listed (even at 0%) so
# a skill the AI never picks (e.g. タイム) is visibly 0.0%, not silently
# absent. "数字宣言" groups all number totals; チョイス→X are added lazily.
_ALL_TP_CATEGORIES = (
    "数字宣言", "フラッシュ", "セメント", "ガード", "チャージ", "クイック",
    "スキップ", "フェイント", "ロック", "コピー", "ストック", "オール",
    "ドロップ", "ブースト", "タイム",
)

# Long-horizon / conditional skills whose value only appears deep in the tree
# (designer diagnosis 2026-07-13). We track how often each was LEGAL vs
# actually SELECTED, so "0% because bad" is distinguishable from "0% because
# the shallow search can't see its payoff".
_WATCH_SKILLS = ("セメント", "ロック", "ストック", "チャージ", "タイム")
_SKILL_CODE_BASE = 64  # skill TP codes: 64 + skill_id*4 + thumb


def _watch_legal_selected(lane0, lane1, tp_buf):
    """Return (set of watch skills that are LEGAL here)."""
    from complete_solver.packed_engine import legal_tp_codes
    n = legal_tp_codes(np.int64(lane0), np.int64(lane1), _FULL_MASK,
                       _NO_CAP, tp_buf)
    legal = set()
    for i in range(n):
        legal.add(tp_category(int(tp_buf[i])))
    return {s for s in _WATCH_SKILLS if s in legal}


def collect_play_stats(agent: SearchAgent, n_games: int, max_plies: int = 120,
                       seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    init0, init1 = pack_state(initial_state())
    tp_counts = [Counter() for _ in PHASES]
    ntp_counts = [Counter() for _ in PHASES]
    win_moves = Counter()        # the declaration that won (winner was mover)
    loss_moves = Counter()       # 宣言→反応 that lost (winner was reactor)
    first_mover_wins = 0
    decided = 0
    plies_list = []
    truncations = 0
    # Boost-countered tracking: seat 0/1, mover alternates via status.
    boost_countered_games = 0
    boost_countered_wins = 0
    boost_countered_win_moves = Counter()
    boost_ok_games = 0
    boost_ok_wins = 0
    # Watch-skill legal-vs-selected: how often each long-horizon/conditional
    # skill was available but the AI chose something else.
    watch_legal = Counter()
    watch_selected = Counter()
    ultimate_choice = Counter()   # among plies where an ultimate was legal
    tp_buf = np.zeros(96, dtype=np.int64)

    for game in range(n_games):
        lane0, lane1 = np.int64(init0), np.int64(init1)
        plies = 0
        mover = 0  # seat index; seat 0 always moves first (self-play, same agent)
        boost_countered_seat = -1
        boost_ok_seat = -1
        for _ in range(max_plies):
            tp_code = agent.tp_action(int(lane0), int(lane1))
            ntp_code = agent.ntp_action(int(lane0), int(lane1))
            phase = phase_of(int(lane0), int(lane1))
            category = tp_category(tp_code)
            tp_counts[phase][category] += 1
            # Watch-skill availability vs selection.
            legal_watch = _watch_legal_selected(lane0, lane1, tp_buf)
            for s in legal_watch:
                watch_legal[s] += 1
            if category in _WATCH_SKILLS:
                watch_selected[category] += 1
            # Ultimate choice: タイム being legal implies the ultimate slot is
            # unused (so ブースト is legal too). Record what got chosen.
            if "タイム" in legal_watch:
                ultimate_choice["_available"] += 1
                if category == "ブースト":
                    ultimate_choice["ブースト"] += 1
                elif category == "タイム":
                    ultimate_choice["タイム"] += 1
                else:
                    ultimate_choice["温存/他"] += 1
            reaction = code_to_ntp_action(int(ntp_code)).reaction
            ntp_counts[phase][reaction] += 1
            if int(tp_code) in _BOOST_CODES:
                if reaction == "カウンター" and boost_countered_seat < 0:
                    boost_countered_seat = mover
                elif reaction != "カウンター" and boost_ok_seat < 0:
                    boost_ok_seat = mover
            plies += 1
            child0, child1, status, reward = step(
                lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
            )
            if status == 2:
                winner = mover if reward > 0 else 1 - mover
                decided += 1
                if winner == 0:
                    first_mover_wins += 1
                label = tp_label(int(tp_code))
                if reward > 0:
                    win_moves[f"{label}(反応:{reaction})"] += 1
                else:
                    loss_moves[f"{label} → {reaction}に討たれ"] += 1
                if boost_countered_seat >= 0:
                    boost_countered_games += 1
                    if winner == boost_countered_seat:
                        boost_countered_wins += 1
                        winning_label = (f"{label}(反応:{reaction})"
                                         if winner == mover else f"相手の{label}を{reaction}")
                        boost_countered_win_moves[winning_label] += 1
                if boost_ok_seat >= 0:
                    boost_ok_games += 1
                    if winner == boost_ok_seat:
                        boost_ok_wins += 1
                break
            if status == 0:
                mover = 1 - mover
            lane0, lane1 = child0, child1
        else:
            truncations += 1
        plies_list.append(plies)

    return {
        "tp_counts": tp_counts,
        "ntp_counts": ntp_counts,
        "win_moves": win_moves,
        "loss_moves": loss_moves,
        "first_mover_winrate": first_mover_wins / max(decided, 1),
        "boost_countered": {
            "games": boost_countered_games,
            "wins": boost_countered_wins,
            "win_moves": boost_countered_win_moves,
        },
        "boost_ok": {"games": boost_ok_games, "wins": boost_ok_wins},
        "watch_legal": watch_legal,
        "watch_selected": watch_selected,
        "ultimate_choice": ultimate_choice,
        "mean_plies": float(np.mean(plies_list)),
        "truncations": truncations,
        "games": n_games,
    }


def simple_counter_table(counter: Counter, total: int | None = None) -> str:
    total = total or sum(counter.values()) or 1
    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td><td>{v / total * 100:.1f}%</td></tr>"
        for k, v in counter.most_common(20)
    )
    return ("<table border='1' cellpadding='4' cellspacing='0'>"
            "<tr><th>決着手</th><th>回数</th><th>割合</th></tr>" + body + "</table>")


def counts_table(counters: list[Counter], all_keys: tuple[str, ...] | None = None) -> str:
    seen = set().union(*[set(c) for c in counters])
    if all_keys is not None:
        # Always list the canonical categories (0% included), then any extra
        # keys actually seen (e.g. チョイス→X), sorted by frequency.
        extra = sorted(seen - set(all_keys),
                       key=lambda k: -sum(c[k] for c in counters))
        keys = list(all_keys) + extra
    else:
        keys = sorted(seen, key=lambda k: -sum(c[k] for c in counters))
    header = "<tr><th>行動</th>" + "".join(
        f"<th>{p}</th>" for p in PHASES) + "<th>全体</th></tr>"
    totals = [sum(c.values()) or 1 for c in counters]
    grand = sum(sum(c.values()) for c in counters) or 1
    body = ""
    for key in keys:
        cells = "".join(
            f"<td>{counters[i][key] / totals[i] * 100:.1f}%</td>"
            for i in range(len(PHASES))
        )
        overall = sum(c[key] for c in counters) / grand * 100
        emphasis = ' style="color:#999"' if overall == 0 else ""
        body += (f"<tr{emphasis}><td>{html.escape(str(key))}</td>{cells}"
                 f"<td><b>{overall:.1f}%</b></td></tr>")
    return f"<table border='1' cellpadding='4' cellspacing='0'>{header}{body}</table>"


def watch_skill_table(stats: dict) -> str:
    """Legal-vs-selected for long-horizon skills: distinguishes 'never chosen
    because weak' from 'never chosen because the shallow search is blind to
    its payoff' (the skill was legal but passed over)."""
    legal = stats["watch_legal"]
    selected = stats["watch_selected"]
    rows = ""
    for s in _WATCH_SKILLS:
        n_legal = legal.get(s, 0)
        n_sel = selected.get(s, 0)
        rate = (n_sel / n_legal * 100) if n_legal else 0.0
        rows += (f"<tr><td>{s}</td><td>{n_legal:,}</td><td>{n_sel:,}</td>"
                 f"<td>{rate:.2f}%</td></tr>")
    uc = stats["ultimate_choice"]
    avail = uc.get("_available", 0) or 1
    ult = (
        "<p>必殺技が使える手番での選択(タイムが合法 = 必殺枠が未使用の手番):</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>選択</th><th>回数</th><th>割合</th></tr>"
        f"<tr><td>ブースト</td><td>{uc.get('ブースト',0):,}</td>"
        f"<td>{uc.get('ブースト',0)/avail*100:.1f}%</td></tr>"
        f"<tr><td>タイム</td><td>{uc.get('タイム',0):,}</td>"
        f"<td>{uc.get('タイム',0)/avail*100:.1f}%</td></tr>"
        f"<tr><td>温存(他の手)</td><td>{uc.get('温存/他',0):,}</td>"
        f"<td>{uc.get('温存/他',0)/avail*100:.1f}%</td></tr>"
        "</table>"
    )
    return (
        "<p>長期・条件付きスキルが「合法だったのに選ばれなかった」率。"
        "選択率がほぼ0%なら、弱いのか探索の地平線が届いていないのかの切り分け材料。</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>スキル</th><th>合法だった手番</th><th>選択された回数</th>"
        "<th>選択率</th></tr>" + rows + "</table>" + ult
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/value_latest.pt")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)
    searcher = BatchedSearcher(model, device, prune_stock=True)
    agent = SearchAgent(searcher, np.random.default_rng(3))

    model_name = Path(args.model).stem
    out_path = Path(args.out or f"results/policy_report_{model_name}.html")

    init_lanes = pack_state(initial_state())
    endgame_lanes = pack_state(h11_root())
    initial_table = direct_mixture_table(searcher, *init_lanes)
    endgame_table = direct_mixture_table(searcher, *endgame_lanes)

    stats = collect_play_stats(agent, n_games=args.games)
    print(f"games={stats['games']} mean_plies={stats['mean_plies']:.1f} "
          f"truncations={stats['truncations']}", flush=True)

    content = f"""<meta charset="utf-8">
<title>方策分布レポート {html.escape(model_name)}</title>
<h1>方策分布レポート — {html.escape(model_name)}</h1>
<p>深さ2ネット葉探索+LP混合(ストック枝刈りON)。
自己対戦 {stats['games']} 局(平均 {stats['mean_plies']:.1f} 手、
打ち切り {stats['truncations']})。</p>

<h2>1. 初期局面の混合戦略(確率の直接参照、サンプリングではない)</h2>
{initial_table}

<h2>2. (1,1)終盤基準局面の混合戦略(直接参照)</h2>
{endgame_table}

<h2>3. 自己対戦でのTP宣言分布(フェーズ別・全スキル表示)</h2>
{counts_table(stats['tp_counts'], all_keys=_ALL_TP_CATEGORIES)}

<h2>3b. 長期・条件付きスキルの合法 vs 選択(タイム/セメント/ロック/ストック/チャージ)</h2>
{watch_skill_table(stats)}

<h2>4. 自己対戦でのNTP反応分布(フェーズ別)</h2>
{counts_table(stats['ntp_counts'])}

<h2>5. 決着の統計(何が勝負を決めているか)</h2>
<p>先手勝率: <b>{stats['first_mover_winrate'] * 100:.1f}%</b></p>
<h3>5a. 勝った側の決着宣言(手番側が勝った場合)</h3>
{simple_counter_table(stats['win_moves'])}
<h3>5b. 負けた側の最後の宣言(反応側に討たれた場合)</h3>
{simple_counter_table(stats['loss_moves'])}

<h2>6. ブーストがカウンターされた後(必殺技の空振り)</h2>
<p>ブーストがカウンターされたゲーム: {stats['boost_countered']['games']} 局、
ブースト側勝率: <b>{(stats['boost_countered']['wins'] / max(stats['boost_countered']['games'], 1)) * 100:.1f}%</b>
(対照: ブーストが通ったゲーム {stats['boost_ok']['games']} 局、
ブースト側勝率 {(stats['boost_ok']['wins'] / max(stats['boost_ok']['games'], 1)) * 100:.1f}%)</p>
<h3>6a. 空振り後にブースト側が勝った時の決着手</h3>
{simple_counter_table(stats['boost_countered']['win_moves'])}

<h2>7. AIの解釈</h2>
<p>(生成時に観察を追記する欄)</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

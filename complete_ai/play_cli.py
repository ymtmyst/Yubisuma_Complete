"""Human vs AI interactive play (N5).

Simultaneity is preserved: when you are the turn player the AI commits to its
reaction mixture without seeing your choice, and when the AI declares you are
asked for your reaction BEFORE its declaration is revealed.

Run:  python -m complete_ai.play_cli [--model models/value_latest.pt] [--ai-first]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from complete_solver.packed_engine import (
    PASS_CODE,
    SKILL_NAMES,
    code_to_ntp_action,
    pack_state,
    step,
    unpack_state,
)
from complete_solver.state import initial_state

from .agents import SearchAgent
from .batched_search import BatchedSearcher, _FULL_MASK
from .generation_loop import load_model


def describe_tp(code: int) -> str:
    if code == PASS_CODE:
        return "パス(スキップ中)"
    thumb = code % 4
    if code < 64:
        return f"数字 {code // 4}(親指{thumb})"
    if code < 128:
        return f"{SKILL_NAMES[(code - 64) // 4]}(親指{thumb})"
    return f"チョイス→{SKILL_NAMES[(code - 128) // 4]}(親指{thumb})"


def describe_ntp(code: int) -> str:
    action = code_to_ntp_action(code)
    return f"{action.reaction}(親指{action.thumb})"


def show_state(lane0: int, lane1: int, human_is_me: bool) -> None:
    state = unpack_state(lane0, lane1)
    me, opp = state.me, state.opp
    you, ai = (me, opp) if human_is_me else (opp, me)

    def line(label, p):
        buffs = []
        if p.guard_active:
            buffs.append("ガード")
        if p.charge_active:
            buffs.append("チャージ")
        if p.quick_level:
            buffs.append(f"クイック{p.quick_level}")
        if p.lock_pending or p.lock_active:
            buffs.append("被ロック")
        if p.time_active:
            buffs.append("タイム")
        if p.used_ultimate:
            buffs.append("必殺技済")
        stock = "・".join(sorted(p.stock)) or "なし"
        return (f"  {label}: 手{p.hands} セメント{p.cement} "
                f"ストック[{stock}] {' '.join(buffs)}")

    print(line("あなた", you))
    print(line("  AI  ", ai))
    if state.previous_skill is not None:
        print(f"  直前の宣言: {state.previous_skill}")
    extra = state.me_extra_turns
    if extra:
        print(f"  手番側の残り追加ターン: {extra}")


def choose(prompt: str, options: list[str]) -> int:
    for i, option in enumerate(options):
        print(f"    [{i}] {option}")
    while True:
        try:
            raw = input(f"{prompt} > ").strip()
        except EOFError:
            print("\n入力が終了したため対局を中断します")
            sys.exit(0)
        try:
            index = int(raw)
            if 0 <= index < len(options):
                return index
        except ValueError:
            pass
        print("  番号を入力してください")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/value_latest.pt")
    parser.add_argument("--ai-first", action="store_true")
    parser.add_argument("--selective-depth", type=int, default=0,
                        help="0 = depth-2 search (default). >2 = N7-C selective "
                             "deepening: read the equilibrium-support lines to "
                             "this depth (stronger, ~0.3s/move at depth 5).")
    parser.add_argument("--selective-tau", type=float, default=0.05)
    parser.add_argument("--no-endgame-table", action="store_true",
                        help="disable the exact endgame tablebase (N7-F): by "
                             "default the AI plays certified-optimal moves in "
                             "the (1,1)-hands stockless endgame.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    endgame = None
    if not args.no_endgame_table:
        from .endgame_table import load_endgame_tablebase
        endgame = load_endgame_tablebase()

    if args.selective_depth > 2:
        from .selective_search import SelectiveSearcher
        searcher = SelectiveSearcher(model, device, prune=True,
                                     depth=args.selective_depth,
                                     tau=args.selective_tau)
        print(f"selective search: depth {args.selective_depth}, "
              f"tau {args.selective_tau}")
    elif endgame is not None:
        # Pincer (N7-F(b)): exact endgame values as search leaves, so decisions
        # LEADING INTO the endgame use the true value, not the net's estimate.
        from .endgame_table import PincerSearcher
        searcher = PincerSearcher(model, device, endgame, prune_stock=True)
    else:
        searcher = BatchedSearcher(model, device, prune_stock=True)
    if endgame is not None:
        print(f"endgame tablebase: exact play + exact search leaves in "
              f"{len(endgame):,} solved (1,1) states")
    agent = SearchAgent(searcher, np.random.default_rng(), endgame=endgame)

    init0, init1 = pack_state(initial_state())
    lane0, lane1 = np.int64(init0), np.int64(init1)
    human_mover = not args.ai_first

    print("=== ユビスマ Complete(ミラー・リバーシOFF)人間 vs AI ===")
    print("手を0にした側の勝ちです。\n")

    from complete_solver.packed_engine import legal_ntp_codes, legal_tp_codes
    from .batched_search import _NO_CAP
    tp_buf = np.zeros(96, dtype=np.int64)
    ntp_buf = np.zeros(16, dtype=np.int64)

    for ply in range(300):
        print(f"\n―― 第{ply + 1}手 ――")
        show_state(int(lane0), int(lane1), human_is_me=human_mover)

        if human_mover:
            n = legal_tp_codes(lane0, lane1, _FULL_MASK, _NO_CAP, tp_buf)
            codes = [int(c) for c in tp_buf[:n]]
            # AI commits to its reaction before seeing your declaration.
            ai_ntp = agent.ntp_action(int(lane0), int(lane1))
            index = choose("あなたの宣言", [describe_tp(c) for c in codes])
            tp_code, ntp_code = codes[index], ai_ntp
            print(f"  AIの反応: {describe_ntp(ntp_code)}")
        else:
            ai_tp = agent.tp_action(int(lane0), int(lane1))
            n = legal_ntp_codes(lane0, lane1, ntp_buf)
            codes = [int(c) for c in ntp_buf[:n]]
            print("  AIが宣言を選びました(あなたの反応を先に決めてください)")
            index = choose("あなたの反応", [describe_ntp(c) for c in codes])
            tp_code, ntp_code = ai_tp, codes[index]
            print(f"  AIの宣言: {describe_tp(tp_code)}")

        child0, child1, status, reward = step(
            lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
        )
        if status == 2:
            mover_won = reward > 0
            human_won = mover_won == human_mover
            print("\n=== " + ("あなたの勝ち!" if human_won else "AIの勝ち") + " ===")
            return
        if status == 0:
            human_mover = not human_mover
        lane0, lane1 = child0, child1
    print("長引いたため打ち切りです")


if __name__ == "__main__":
    main()

"""Counter-piercing action-distribution measurement (Phase 2).

Loads a value-net model and plays self-play games with the pure LP-mixture
policy (SearchAgent, epsilon=0), then reports the action distribution UNDER
WHATEVER RULE MASK THE PROCESS IS RUNNING (YS_COUNTER_PIERCE). This is the
tool that answers "did buffing skill X change how often the trained model
uses it".

CRITICAL: this asserts `_CP_MASK` (in BOTH engines) equals the mask passed
on the command line, so a stale numba cache or a forgotten env var can never
silently make the measurement run under the wrong rules. Clear the numba
cache and set YS_COUNTER_PIERCE before launching.

Metrics (all over self-play mover-plies):
  - TP category distribution overall + by phase (STOCK-declaration rate,
    CEMENT rate, FLASH rate, every skill, number-declaration);
  - thumb distribution overall and for FLASH / number declarations
    (the cement "don't-raise-fingers / flash-zero" meta signal);
  - stock-hold rate: fraction of plies where the mover holds >=1 stocked
    skill, and fraction of games where any stock is ever held;
  - NTP reaction distribution;
  - direct buff-firing evidence: how often a STOCK declaration (resp. a
    CEMENT declaration) actually met a COUNTER reaction this game.

Usage:
  python -m scratchpad.cp_policy_dist --model models/value_cp_stock_latest.pt \
      --expect-mask 512 --games 4000 --out data/cp_stock_dist.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from complete_solver.packed_engine import (
    _CP_MASK as PACKED_MASK,
    pack_state,
    step,
    code_to_ntp_action,
)
from complete_solver.state import initial_state
from complete_solver.transition import _CP_MASK as REF_MASK

from complete_ai.agents import SearchAgent
from complete_ai.batched_search import BatchedSearcher, _FULL_MASK
from complete_ai.generation_loop import load_model
from complete_ai.policy_report import phase_of, tp_category, PHASES

_STOCK_ID, _CEMENT_ID = 9, 1


def measure(model, device, n_games: int, seed: int) -> dict:
    searcher = BatchedSearcher(model, device, prune_stock=True)
    agent = SearchAgent(searcher, np.random.default_rng(seed), epsilon=0.0)
    init0, init1 = pack_state(initial_state())

    tp_cat = [Counter() for _ in PHASES]           # by phase
    tp_cat_all = Counter()                         # overall
    ntp_reactions = Counter()
    thumb_all = Counter()
    thumb_flash = Counter()
    thumb_number = Counter()
    plies = 0
    plies_with_stock = 0
    games_with_stock = 0
    games_stock_declared_under_counter = 0
    games_cement_declared_under_counter = 0
    stock_decl_under_counter_plies = 0
    cement_decl_under_counter_plies = 0
    truncations = 0

    for game in range(n_games):
        lane0, lane1 = np.int64(init0), np.int64(init1)
        held_this_game = False
        stock_uc_this_game = False
        cement_uc_this_game = False
        for _ in range(120):
            tp_code = agent.tp_action(int(lane0), int(lane1))
            ntp_code = agent.ntp_action(int(lane0), int(lane1))
            phase = phase_of(int(lane0), int(lane1))
            cat = tp_category(int(tp_code))
            thumb = int(tp_code) % 4
            reaction = code_to_ntp_action(int(ntp_code)).reaction

            tp_cat[phase][cat] += 1
            tp_cat_all[cat] += 1
            ntp_reactions[reaction] += 1
            thumb_all[thumb] += 1
            if cat == "数字宣言":
                thumb_number[thumb] += 1
            if cat == "フラッシュ":
                thumb_flash[thumb] += 1

            # stock-hold: mover is lane0's "me"; stock bits are 18..25.
            if (int(lane0) >> 18) & 0xFF:
                plies_with_stock += 1
                held_this_game = True

            # direct buff-firing evidence (declaration + opponent COUNTER).
            is_stock_decl = 64 + _STOCK_ID * 4 <= int(tp_code) < 64 + _STOCK_ID * 4 + 4
            is_cement_decl = 64 + _CEMENT_ID * 4 <= int(tp_code) < 64 + _CEMENT_ID * 4 + 4
            if reaction == "カウンター":
                if is_stock_decl:
                    stock_decl_under_counter_plies += 1
                    stock_uc_this_game = True
                if is_cement_decl:
                    cement_decl_under_counter_plies += 1
                    cement_uc_this_game = True

            plies += 1
            child0, child1, status, _ = step(
                lane0, lane1, np.int64(tp_code), np.int64(ntp_code), _FULL_MASK
            )
            if status == 2:
                break
            lane0, lane1 = child0, child1
        else:
            truncations += 1
        if held_this_game:
            games_with_stock += 1
        if stock_uc_this_game:
            games_stock_declared_under_counter += 1
        if cement_uc_this_game:
            games_cement_declared_under_counter += 1

    def pct(counter, total):
        return {k: 100.0 * v / max(total, 1) for k, v in counter.items()}

    total_tp = sum(tp_cat_all.values())
    return {
        "n_games": n_games,
        "plies": plies,
        "truncations": truncations,
        "tp_category_overall_pct": pct(tp_cat_all, total_tp),
        "tp_category_counts": dict(tp_cat_all),
        "stock_decl_pct": 100.0 * tp_cat_all.get("ストック", 0) / max(total_tp, 1),
        "cement_decl_pct": 100.0 * tp_cat_all.get("セメント", 0) / max(total_tp, 1),
        "flash_decl_pct": 100.0 * tp_cat_all.get("フラッシュ", 0) / max(total_tp, 1),
        "number_decl_pct": 100.0 * tp_cat_all.get("数字宣言", 0) / max(total_tp, 1),
        "stock_hold_ply_pct": 100.0 * plies_with_stock / max(plies, 1),
        "stock_hold_game_pct": 100.0 * games_with_stock / max(n_games, 1),
        "thumb_overall_pct": pct(thumb_all, sum(thumb_all.values())),
        "thumb_number_pct": pct(thumb_number, sum(thumb_number.values())),
        "thumb_flash_pct": pct(thumb_flash, sum(thumb_flash.values())),
        "ntp_reaction_pct": pct(ntp_reactions, sum(ntp_reactions.values())),
        "stock_decl_under_counter_plies": stock_decl_under_counter_plies,
        "cement_decl_under_counter_plies": cement_decl_under_counter_plies,
        "games_stock_declared_under_counter": games_stock_declared_under_counter,
        "games_cement_declared_under_counter": games_cement_declared_under_counter,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expect-mask", type=int, required=True,
                        help="assert both engines' _CP_MASK equal this")
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    assert PACKED_MASK == args.expect_mask, (
        f"packed_engine._CP_MASK={PACKED_MASK} != expected {args.expect_mask} "
        "(clear numba cache and set YS_COUNTER_PIERCE before running)"
    )
    assert REF_MASK == args.expect_mask, (
        f"transition._CP_MASK={REF_MASK} != expected {args.expect_mask}"
    )
    print(f"mask OK: both engines _CP_MASK={args.expect_mask}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.model), device)
    result = measure(model, device, args.games, args.seed)
    result["model"] = args.model
    result["mask"] = args.expect_mask

    print(json.dumps({
        "model": args.model, "mask": args.expect_mask,
        "plies": result["plies"], "truncations": result["truncations"],
        "stock_decl_pct": round(result["stock_decl_pct"], 4),
        "stock_hold_ply_pct": round(result["stock_hold_ply_pct"], 4),
        "stock_hold_game_pct": round(result["stock_hold_game_pct"], 4),
        "cement_decl_pct": round(result["cement_decl_pct"], 4),
        "flash_decl_pct": round(result["flash_decl_pct"], 4),
        "number_decl_pct": round(result["number_decl_pct"], 4),
        "thumb_overall_pct": {k: round(v, 2) for k, v in sorted(result["thumb_overall_pct"].items())},
        "thumb_flash_pct": {k: round(v, 2) for k, v in sorted(result["thumb_flash_pct"].items())},
        "ntp_reaction_pct": {k: round(v, 2) for k, v in result["ntp_reaction_pct"].items()},
        "stock_decl_under_counter_plies": result["stock_decl_under_counter_plies"],
        "cement_decl_under_counter_plies": result["cement_decl_under_counter_plies"],
    }, ensure_ascii=False, indent=2), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

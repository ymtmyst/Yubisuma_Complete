"""Compare two generation runs (e.g. depth teacher vs N7-A graph-vi teacher).

Reads two n4_*generations.jsonl logs and prints per-generation a0-spearman and
arena win-rate side by side, plus final deltas. This is the quantitative half
of the graph-vi vs depth comparison; the qualitative litmus (early-cement >
late-cement) comes from the policy_report on each run's final model.

Run:  python -m complete_ai.n7a_compare --a data/n4_generations.jsonl --b data/n4_gvi_generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="data/n4_generations.jsonl",
                        help="baseline log (depth teacher)")
    parser.add_argument("--b", default="data/n4_gvi_generations.jsonl",
                        help="experiment log (graph-vi teacher)")
    parser.add_argument("--a-name", default="depth")
    parser.add_argument("--b-name", default="graph-vi")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    a = load(Path(args.a))
    b = load(Path(args.b))
    print(f"{args.a_name}: {len(a)} gens   {args.b_name}: {len(b)} gens\n")

    def a0(r):
        return r.get("a0_spearman", float("nan"))

    def arena(r):
        return r.get("arena_new_winrate", float("nan"))

    n = max(len(a), len(b))
    print(f"{'gen':>4} | {args.a_name:>18} | {args.b_name:>18}")
    print(f"{'':>4} | {'a0':>8}{'arena':>10} | {'a0':>8}{'arena':>10}")
    for i in range(n):
        ra = a[i] if i < len(a) else None
        rb = b[i] if i < len(b) else None
        la = f"{a0(ra):>8.4f}{arena(ra):>10.3f}" if ra else f"{'-':>8}{'-':>10}"
        lb = f"{a0(rb):>8.4f}{arena(rb):>10.3f}" if rb else f"{'-':>8}{'-':>10}"
        print(f"{i+1:>4} | {la} | {lb}")

    if a and b:
        print("\n── final ──")
        print(f"a0-spearman   {args.a_name} {a0(a[-1]):.4f}   "
              f"{args.b_name} {a0(b[-1]):.4f}   Δ {a0(b[-1]) - a0(a[-1]):+.4f}")
        # a0 best across gens (accuracy on the exact stockless slice)
        print(f"a0 best       {args.a_name} {max(a0(r) for r in a):.4f}   "
              f"{args.b_name} {max(a0(r) for r in b):.4f}")
        print("\nNote: a0 measures accuracy on the EXACT stockless slice; the "
              "graph-vi thesis is about LONG-horizon (stock/cement) value, which "
              "a0 only partially reflects. The decisive test is the policy "
              "litmus (early-cement > late-cement) from policy_report.")


if __name__ == "__main__":
    main()

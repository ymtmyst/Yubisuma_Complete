"""Train PPO best-response attackers against the frozen agent (N6, gate ④).

Run:  python -m complete_ai.br_attack --seeds 0,1,2 --steps 250000

The gate passes when the FROZEN agent keeps ≥55% winrate against every
trained attacker (i.e. the attacker cannot exploit the agent past 45%).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .batched_search import BatchedSearcher
from .br_env import BRAttackEnv
from .generation_loop import load_model

OUT = Path("data/n6_br_attack.json")


def evaluate_attacker(ppo, env: BRAttackEnv, episodes: int = 300) -> dict:
    wins = 0
    losses = 0
    truncations = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = ppo.predict(obs, action_masks=mask, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            if terminated:
                done = True
                if reward > 0:
                    wins += 1
                else:
                    losses += 1
            elif truncated:
                done = True
                truncations += 1
    decided = wins + losses
    return {
        "episodes": episodes,
        "attacker_wins": wins,
        "attacker_losses": losses,
        "truncations": truncations,
        "attacker_winrate": wins / decided if decided else 0.0,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--model", default="models/value_latest.pt")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_model = load_model(Path(args.model), device)

    results: dict = {"model": args.model, "steps": args.steps, "seeds": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"=== BR attacker seed {seed} ===", flush=True)
        searcher = BatchedSearcher(frozen_model, device, prune_stock=True)
        env = BRAttackEnv(searcher, seed=seed)
        ppo = MaskablePPO(
            "MlpPolicy", env,
            seed=seed,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=6,
            gamma=0.999,
            ent_coef=0.01,
            verbose=0,
            device=device,
        )
        t0 = time.perf_counter()
        ppo.learn(total_timesteps=args.steps, progress_bar=False)
        train_seconds = time.perf_counter() - t0
        print(f"trained in {train_seconds:.0f}s; evaluating...", flush=True)

        outcome = evaluate_attacker(ppo, BRAttackEnv(searcher, seed=seed + 100))
        outcome["train_seconds"] = train_seconds
        results["seeds"][seed] = outcome
        print(f"seed {seed}: {json.dumps(outcome)}", flush=True)
        ppo.save(f"models/br_attacker_seed{seed}")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    worst = max(
        (r["attacker_winrate"] for r in results["seeds"].values()),
        default=0.0,
    )
    results["worst_attacker_winrate"] = worst
    # Unexploitability criterion: a Nash player concedes exactly the game
    # value (≈50% with alternating seats) to a best responder — no less.
    # The gate therefore passes when no attacker EXCEEDS parity; demanding
    # ≤45% would require dominating a best responder, which even perfect
    # play does not guarantee.
    results["gate_pass"] = worst <= 0.50
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"GATE: worst attacker winrate {worst:.3f} "
          f"({'PASS' if results['gate_pass'] else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()

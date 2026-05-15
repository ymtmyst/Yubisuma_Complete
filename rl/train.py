# rl/train.py - メイン訓練スクリプト
"""
指スマ完全ルール版 PPO Self-Play 訓練。

使い方:
  python -m rl.train                         # 新規訓練開始
  python -m rl.train --resume PATH           # 前回の続きから再開
  python -m rl.train --steps 20000000        # ステップ数指定
  python -m rl.train --wandb                 # W&B ログ有効化
  python -m rl.train --wandb --run-name exp1 # 実験名指定
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from rl.config import (
    PPO_CONFIG, NETWORK_CONFIG, TOTAL_TIMESTEPS,
    MODEL_DIR, LOG_DIR, ANALYSIS_DIR, LEAGUE_CONFIG,
    CHECKPOINT_FREQ, OPPONENT_UPDATE_FREQ, EVAL_FREQ, EVAL_EPISODES,
    SNAPSHOT_PATH,
)
from rl.env import YubisumaEnv
from rl.network import YubisumaFeaturesExtractor
from rl.model_utils import import_maskable_ppo, load_maskable_ppo
from rl.opponents import LeagueManager
from rl.analysis import AnalysisDB
from rl.callbacks import (
    SelfPlayCallback, AnalysisCallback, CheckpointCallback, AuxLossCallback,
    EntCoefScheduleCallback, RandomEvalCallback, SkillSnapshotCallback,
    DiversityLossCallback,
)


def make_env(rank=0):
    def _init():
        env = YubisumaEnv(opponent_policy=None)
        return env
    return _init


def _make_run_id(args):
    raw = args.run_id or args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    return safe.strip("_") or datetime.now().strftime("%Y%m%d_%H%M%S")



def train(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    run_id = _make_run_id(args)
    analysis_db_path = args.db_path or os.path.join(ANALYSIS_DIR, f"{run_id}.db")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # === W&B 初期化 ===
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="yubisuma-ppo",
                name=args.run_name,
                config={
                    **PPO_CONFIG,
                    "network": NETWORK_CONFIG,
                    "league": LEAGUE_CONFIG,
                    "n_envs": args.n_envs,
                    "total_steps": args.steps,
                    "resume": args.resume,
                    "run_id": run_id,
                    "analysis_db": analysis_db_path,
                },
                sync_tensorboard=True,  # SB3 の TensorBoard ログを自動同期
                save_code=False,
                resume="allow" if args.resume else None,
            )
            print(f"[W&B] Run: {wandb_run.url}")
        except ImportError:
            print("[W&B] wandb がインストールされていません。pip install wandb")
            wandb_run = None

    # === 分析DB ===
    analysis_db = AnalysisDB(db_path=analysis_db_path, run_id=run_id)
    analysis_cb = AnalysisCallback(analysis_db, verbose=1)

    # === 環境作成 (SubprocVecEnv: CPUコアを並列活用) ===
    from stable_baselines3.common.vec_env import SubprocVecEnv

    n_envs = args.n_envs
    print(f"[Train] Envs: {n_envs}")
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)],
                        start_method="spawn")

    # === MaskablePPO ===
    MaskablePPO = import_maskable_ppo()

    policy_kwargs = {
        "features_extractor_class": YubisumaFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": NETWORK_CONFIG["feature_dim"],
        },
        "net_arch": {
            "pi": NETWORK_CONFIG["policy_layers"],
            "vf": NETWORK_CONFIG["value_layers"],
        },
        "activation_fn": torch.nn.ReLU,
    }

    if args.resume:
        print(f"[Train] Resume from: {args.resume}")
        model = load_maskable_ppo(args.resume, env=env, device=device)
        # --steps が指定されていない場合は残りステップを継続
        # (model.num_timesteps が保存済みステップ数を保持している)
        remaining = args.steps - model.num_timesteps
        if remaining <= 0:
            print(f"[Train] 既に {model.num_timesteps:,} ステップ完了。"
                  f"追加する場合は --steps を大きくしてください。")
            env.close()
            return
        print(f"[Train] 再開: {model.num_timesteps:,} 済 → "
              f"あと {remaining:,} ステップ")
    else:
        print("[Train] New model")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            **PPO_CONFIG,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device=device,
        )

    # === リーグマネージャー ===
    league = LeagueManager(model_class=MaskablePPO)

    # === コールバック ===
    selfplay_cb = SelfPlayCallback(
        env, league_manager=league,
        update_freq=OPPONENT_UPDATE_FREQ, verbose=1
    )
    checkpoint_cb = CheckpointCallback(save_freq=CHECKPOINT_FREQ, verbose=1)
    aux_loss_cb = AuxLossCallback(aux_lr=3e-4, verbose=1)
    # ent_coef線形減衰: SB3はent_coefにcallableを受け付けないためコールバックで実装
    ent_coef_cb = EntCoefScheduleCallback(
        initial=PPO_CONFIG["ent_coef"],
        final=0.005,
        total_steps=args.steps,
    )
    eval_cb = RandomEvalCallback(
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False,
        verbose=1,
    )
    snapshot_cb = SkillSnapshotCallback(
        analysis_db, SNAPSHOT_PATH,
        snapshot_freq=CHECKPOINT_FREQ, last_n=100, verbose=1,
    )
    # DIAYN (Eysenbach et al. ICLR 2019): persona z 別に区別可能な行動分布へ自然分化
    # 学習開始時 discriminator accuracy はランダム (1/7≈0.143) から始まり、
    # 分化が成功していれば徐々に上昇する。wandb で diversity/disc_tp_acc を監視。
    diversity_cb = DiversityLossCallback(verbose=1) if args.diversity else None

    callbacks = [selfplay_cb, analysis_cb, checkpoint_cb, aux_loss_cb, ent_coef_cb,
                 eval_cb, snapshot_cb]
    if diversity_cb is not None:
        callbacks.append(diversity_cb)

    # W&B コールバック追加
    if wandb_run is not None:
        try:
            from wandb.integration.sb3 import WandbCallback
            wandb_cb = WandbCallback(
                gradient_save_freq=0,   # 勾配ログは無効（重くなるため）
                verbose=0,
            )
            callbacks.append(wandb_cb)
        except ImportError:
            pass

    # === 訓練情報表示 ===
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[Train] Parameters: {total_params:,}")
    print(f"[Train] Run ID: {run_id}")
    print(f"[Train] Target steps: {args.steps:,}")
    print(f"[Train] Analysis DB: {analysis_db.db_path}")
    print()

    # === 訓練 ===
    try:
        # resume時: SB3は reset_num_timesteps=False のとき
        #   total_timesteps += self.num_timesteps を内部で行うため
        #   args.steps をそのまま渡すと目標が (args.steps + num_timesteps) になりオーバーシュートする。
        #   remaining を渡すことで SB3 内部で remaining + num_timesteps = args.steps が成立し正確になる。
        learn_steps = remaining if args.resume else args.steps
        model.learn(
            total_timesteps=learn_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        print("\n[Train] Interrupted. Saving model...")
    finally:
        # 最終モデル保存（中断時も保存）
        final_path = os.path.join(MODEL_DIR, "yubisuma_ppo_final")
        model.save(final_path)
        print(f"[Train] Model saved: {final_path}")

        summary = analysis_db.get_summary()
        print(f"\n=== Result ===")
        print(f"Episodes : {summary['total_episodes']}")
        print(f"Win rate : {summary['overall_win_rate']:.1%}")
        print(f"Recent100: {summary['recent_100_win_rate']:.1%}")

        if wandb_run is not None:
            import wandb
            wandb.finish()

        env.close()


def main():
    parser = argparse.ArgumentParser(description="Yubisuma PPO Training")
    parser.add_argument(
        "--steps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total timesteps (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (e.g. rl_models/yubisuma_ppo_500000)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=16,
        help="Number of parallel environments (default: 16)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="W&B run name (optional)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Stable run identifier for analysis DB naming/filtering",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Analysis DB path (default: rl_analysis/<run-id>.db)",
    )
    parser.add_argument(
        "--diversity", action="store_true",
        help="Enable DIAYN diversity loss for persona分化 (Eysenbach et al. ICLR 2019)",
    )
    parser.add_argument(
        "--no-diversity", dest="diversity", action="store_false",
        help="Disable DIAYN diversity loss (baseline run)",
    )
    parser.set_defaults(diversity=True)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

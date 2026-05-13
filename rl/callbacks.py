# rl/callbacks.py - 訓練コールバック
"""
SB3用カスタムコールバック:
- SelfPlayCallback: 定期的に対戦相手を更新 (SubprocVecEnv/DummyVecEnv両対応)
- AuxLossCallback: 補助タスク損失の計算・適用
- AnalysisCallback: 分析DBへの記録
- CheckpointCallback: チェックポイント保存
"""

import os
import numpy as np
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from rl.config import (
    CHECKPOINT_FREQ, OPPONENT_UPDATE_FREQ,
    MODEL_DIR,
)


class SelfPlayCallback(BaseCallback):
    """
    Self-Play用コールバック。
    定期的にモデルのコピーを対戦相手として設定する。

    SubprocVecEnv/DummyVecEnv両対応:
    - モデルをディスクに保存し、パス文字列をenv_method経由で各環境に配布
    - 各環境が自前でモデルをロード (大きなオブジェクトのIPC送信を回避)
    """

    def __init__(self, env, league_manager=None, update_freq=OPPONENT_UPDATE_FREQ,
                 verbose=1):
        super().__init__(verbose)
        self.env = env
        self.league_manager = league_manager
        self.update_freq = update_freq
        self.generation = 0
        self._last_update = 0
        # リーグなし時の一時保存先
        self._selfplay_path = os.path.join(MODEL_DIR, "_selfplay_current")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # ロールアウト境界でのみ更新 → GAEの計算中に環境が変わることを防止
        if self.num_timesteps - self._last_update >= self.update_freq:
            self._update_opponent()
            self._last_update = self.num_timesteps

    def _update_opponent(self):
        """対戦相手を更新"""
        self.generation += 1

        if self.league_manager is not None:
            # リーグにチェックポイントを保存 (step番号で命名)
            self.league_manager.save_checkpoint(self.model, self.num_timesteps)
            # リーグからパスを選択してサブプロセスに配布
            path, step = self.league_manager.select_opponent()
            if path is not None:
                self._set_opponent_by_path(path)
                if self.verbose > 0:
                    print(f"[SelfPlay] 対戦相手をstep {step:,}に更新 "
                          f"(リーグサイズ: {self.league_manager.num_opponents})")
            else:
                self._set_opponent_from_self()
        else:
            self._set_opponent_from_self()

    def _set_opponent_from_self(self):
        """現在のモデルを一時保存してパス経由で各環境に設定"""
        self.model.save(self._selfplay_path)
        self._set_opponent_by_path(self._selfplay_path)
        if self.verbose > 0:
            print(f"[SelfPlay] 対戦相手を現在のモデルに更新 (世代{self.generation})")

    def _set_opponent_by_path(self, path):
        """SubprocVecEnv/DummyVecEnv両対応: env_method経由でパスを全環境に配布。
        各環境の set_opponent_path() がサブプロセス内でモデルをロードする。"""
        self.env.env_method('set_opponent_path', path)


class AnalysisCallback(BaseCallback):
    """
    分析DBへの記録コールバック。

    SubprocVecEnv対応: env側でinfo['episode_summary']に詰めたデータを
    主プロセスのlocals['infos']から収集する。
    """

    def __init__(self, analysis_db, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.analysis_db = analysis_db
        self.log_freq = log_freq
        self._episode_buffer = deque(maxlen=200)
        self._wins = 0
        self._total = 0
        self._last_log_step = 0    # しきい値方式: n_envs=16でも正確に発火
        self._log_count = 0        # コンソール出力は10回に1回

    def _on_step(self) -> bool:
        # SubprocVecEnv/DummyVecEnvどちらでも、エピソード終了時に
        # info['episode_summary']が主プロセスのinfosに届く
        for info in self.locals.get('infos', []):
            ep_data = info.get('episode_summary')
            if ep_data is not None:
                self._episode_buffer.append(ep_data)
                if ep_data.get('agent_won'):
                    self._wins += 1
                self._total += 1
                self.analysis_db.record_episode(
                    ep_data, training_step=self.num_timesteps,
                )

        # 定期的に訓練統計を記録
        # % 演算はn_envs倍数でないlog_freqで発火しないため、しきい値方式を使う
        if self.num_timesteps - self._last_log_step >= self.log_freq and self._total > 0:
            self._last_log_step = self.num_timesteps
            self._log_count += 1
            recent = list(self._episode_buffer)[-100:]
            recent_wins = sum(1 for e in recent if e.get('agent_won'))
            recent_lengths = [e.get('total_turns', 0) for e in recent]

            win_rate = recent_wins / len(recent) if recent else 0.0
            avg_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0

            stats = {
                'win_rate': win_rate,
                'avg_episode_length': avg_length,
            }

            if self.logger is not None:
                try:
                    self.logger.record("game/win_rate_100", win_rate)
                    self.logger.record("game/avg_episode_length", avg_length)
                    self.logger.record("game/total_episodes", self._total)
                    stats['policy_loss'] = self.logger.name_to_value.get(
                        'train/policy_gradient_loss', None
                    )
                    stats['value_loss'] = self.logger.name_to_value.get(
                        'train/value_loss', None
                    )
                    stats['entropy'] = self.logger.name_to_value.get(
                        'train/entropy_loss', None
                    )
                except Exception:
                    pass

            self.analysis_db.record_training_stats(self.num_timesteps, stats)

            if self.verbose > 0 and self._log_count % 10 == 0:
                print(f"[Analysis] Step {self.num_timesteps}: "
                      f"勝率={win_rate:.1%}, "
                      f"平均ターン={avg_length:.1f}")

        return True


class AuxLossCallback(BaseCallback):
    """
    補助タスク損失コールバック。

    各ロールアウト収集中に (観測, 相手行動) ペアを蓄積し、
    ロールアウト終了後に相手行動予測ヘッドの損失を計算・適用する。
    補助損失はPPOの更新とは独立した別オプティマイザで行う。

    オプティマイザは2つの学習率グループを持つ:
    - 補助ヘッド: aux_lr (フル速度)
    - shared_net: aux_lr * 0.1 (低速) → 相手モデリング向け特徴を徐々に改善しつつ
                                          PPOの10エポック更新への干渉を最小化
    """

    def __init__(self, aux_lr: float = 3e-4, verbose: int = 0):
        super().__init__(verbose)
        self.aux_lr = aux_lr
        self._aux_computer = None
        self._aux_optimizer = None

    def _on_training_start(self) -> None:
        import torch
        from rl.network import AuxiliaryLossComputer

        fe = self.model.policy.features_extractor
        device = next(fe.parameters()).device

        self._aux_computer = AuxiliaryLossComputer(
            feature_extractor=fe,
            device=str(device),
        )

        # 補助ヘッドはfull lr、shared_netは低lr(1/10)で同時更新
        # → aux損失が特徴表現を相手モデリング向けに改善しつつ、PPOへの干渉を最小化
        self._aux_optimizer = torch.optim.Adam([
            {'params': (list(fe.aux_reaction_head.parameters()) +
                        list(fe.aux_thumbs_head.parameters()) +
                        list(fe.aux_skill_head.parameters())),
             'lr': self.aux_lr},
            {'params': list(fe.shared_net.parameters()),
             'lr': self.aux_lr * 0.1},
        ])

    def _on_step(self) -> bool:
        """ロールアウト中: 相手行動ラベルと観測を蓄積"""
        if self._aux_computer is None:
            return True

        infos = self.locals.get('infos', [])
        if not infos:
            return True

        obs_tensor = self.locals.get('obs_tensor', None)
        if obs_tensor is not None and hasattr(obs_tensor, 'detach'):
            obs_array = obs_tensor.detach().cpu().numpy()
        else:
            obs_array = self.locals.get('new_obs', None)

        if obs_array is None:
            return True

        for i, info in enumerate(infos):
            if 'opponent_action' not in info or i >= len(obs_array):
                continue

            opp_action = info['opponent_action']
            obs_i = np.asarray(obs_array[i], dtype=np.float32)
            is_agent_tp = (opp_action.get('role') == 'ntp')
            self._aux_computer.record_opponent_action(obs_i, opp_action, is_agent_tp)

        return True

    def _on_rollout_end(self) -> None:
        """ロールアウト終了後: 補助損失を計算・適用"""
        import torch

        if self._aux_computer is None:
            return

        loss = self._aux_computer.compute_loss()

        if loss.requires_grad and loss.item() > 1e-8:
            self._aux_optimizer.zero_grad()
            loss.backward()

            all_params: list = []
            for pg in self._aux_optimizer.param_groups:
                all_params.extend(pg['params'])
            torch.nn.utils.clip_grad_norm_(all_params, 0.5)

            self._aux_optimizer.step()

            if self.logger is not None:
                self.logger.record("train/aux_loss", loss.item())
            if self.verbose > 0:
                print(f"[AuxLoss] Step {self.num_timesteps}: "
                      f"aux_loss={loss.item():.4f}")

        self._aux_computer.clear_buffer()


class CheckpointCallback(BaseCallback):
    """チェックポイント保存コールバック"""

    def __init__(self, save_freq=CHECKPOINT_FREQ, save_path=None, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path or MODEL_DIR
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"yubisuma_ppo_{self.num_timesteps}"
            )
            self.model.save(path)
            if self.verbose > 0:
                print(f"[Checkpoint] モデル保存: {path}")
        return True


class EntCoefScheduleCallback(BaseCallback):
    """
    ent_coefを学習進捗に応じて線形減衰させるコールバック。

    SB3のPPOはent_coefにcallableを渡せない(learning_rate/clip_rangeのみ対応)ため、
    コールバック経由でmodel.ent_coefを直接書き換えることでスケジュールを実現する。
    """

    def __init__(self, initial: float, final: float, total_steps: int):
        super().__init__()
        self.initial = initial
        self.final = final
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.total_steps))
        # progress: 0.0(開始) → 1.0(終了) で initial → final に線形減衰
        self.model.ent_coef = self.final + (1.0 - progress) * (self.initial - self.final)
        return True

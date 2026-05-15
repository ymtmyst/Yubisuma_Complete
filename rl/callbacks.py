# rl/callbacks.py - 訓練コールバック
"""
SB3用カスタムコールバック:
- SelfPlayCallback: 定期的に対戦相手を更新 (SubprocVecEnv/DummyVecEnv両対応)
- AuxLossCallback: 補助タスク損失の計算・適用
- AnalysisCallback: 分析DBへの記録
- CheckpointCallback: チェックポイント保存
- SkillSnapshotCallback: スキル分布をJSONに上書き出力（ゲームロジック確認用）
"""

import json
import os
import numpy as np
from collections import deque
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback

from rl.config import (
    CHECKPOINT_FREQ, OPPONENT_UPDATE_FREQ,
    MODEL_DIR, LOOKAHEAD_N,
    OBS_PERSONA, NUM_PERSONA_TP, NUM_PERSONA_NTP,
    LAMBDA_DIVERSITY_TP, LAMBDA_DIVERSITY_NTP,
    DIAYN_SHARED_LR_RATIO,
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
        # 終盤勝敗予測用: env_idx -> 直近LOOKAHEAD_N観測の deque
        self._recent_obs_by_env: dict = {}

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
                        list(fe.aux_skill_head.parameters()) +
                        list(fe.aux_lookahead_head.parameters())),
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
            if i >= len(obs_array):
                continue
            obs_i = np.asarray(obs_array[i], dtype=np.float32)

            # 相手行動の記録 (既存)
            if 'opponent_action' in info:
                opp_action = info['opponent_action']
                is_agent_tp = (opp_action.get('role') == 'ntp')
                self._aux_computer.record_opponent_action(obs_i, opp_action, is_agent_tp)

            # 終盤勝敗予測: per-env deque にこのステップの観測を蓄積
            if i not in self._recent_obs_by_env:
                self._recent_obs_by_env[i] = deque(maxlen=LOOKAHEAD_N)
            self._recent_obs_by_env[i].append(obs_i)

            # エピソード終了時: deque 内の全観測（末尾LOOKAHEAD_N手）をラベル付け
            ep_data = info.get('episode_summary')
            if ep_data is not None and i in self._recent_obs_by_env:
                outcome = 1.0 if ep_data.get('agent_won') else 0.0
                for past_obs in self._recent_obs_by_env[i]:
                    self._aux_computer.add_lookahead(past_obs, outcome)
                self._recent_obs_by_env[i].clear()

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


class RandomEvalCallback(BaseCallback):
    """
    固定ランダム相手に対する定期評価。

    self-play中の勝率は相手リーグの強さや分布に依存するため、
    学習進捗の外部指標としてランダム相手への勝率・平均ターン数を記録する。
    混合戦略を評価するため、デフォルトでは確率的に行動をサンプルする。
    """

    def __init__(self, eval_freq: int, n_eval_episodes: int,
                 deterministic: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True

        self._last_eval = self.num_timesteps
        win_rate, avg_turns = self._evaluate_random_opponent()

        if self.logger is not None:
            self.logger.record("eval/random_win_rate", win_rate)
            self.logger.record("eval/random_avg_turns", avg_turns)

        if self.verbose > 0:
            print(f"[Eval] Step {self.num_timesteps}: "
                  f"random_win_rate={win_rate:.1%}, avg_turns={avg_turns:.1f}")

        return True

    def _evaluate_random_opponent(self):
        import numpy as np
        from rl.env import YubisumaEnv

        env = YubisumaEnv(opponent_policy=None)
        wins = 0
        total_turns = []

        try:
            for _ in range(self.n_eval_episodes):
                obs, _ = env.reset()
                done = False
                reward = 0.0

                while not done:
                    mask = env.action_masks()
                    action, _ = self.model.predict(
                        obs,
                        action_masks=mask,
                        deterministic=self.deterministic,
                    )
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated

                if reward > 0:
                    wins += 1
                total_turns.append(env.turn_count)
        finally:
            env.close()

        win_rate = wins / self.n_eval_episodes if self.n_eval_episodes > 0 else 0.0
        avg_turns = float(np.mean(total_turns)) if total_turns else 0.0
        return win_rate, avg_turns


class SkillSnapshotCallback(BaseCallback):
    """
    直近 last_n エピソードのスキル使用分布を JSON に上書き出力。
    ゲームロジックが正しく機能しているかの確認用。
    snapshot_freq ステップごとに更新し、訓練終了時にも最終出力する。
    """

    def __init__(self, analysis_db, output_path, snapshot_freq=50_000,
                 last_n=100, verbose=0):
        super().__init__(verbose)
        self.analysis_db = analysis_db
        self.output_path = output_path
        self.snapshot_freq = snapshot_freq
        self.last_n = last_n
        self._last_snapshot = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot >= self.snapshot_freq:
            self._write_snapshot()
            self._last_snapshot = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._write_snapshot()

    def _write_snapshot(self):
        try:
            skills    = self.analysis_db.get_skill_opportunity_stats(last_n_episodes=self.last_n)
            reactions = self.analysis_db.get_reaction_stats(last_n_episodes=self.last_n)
            summary   = self.analysis_db.get_summary()
            ep_stats  = self.analysis_db.get_episode_length_stats(last_n_episodes=self.last_n)

            # スキル使用分布（TP行動）
            total_skill = sum(s['usage_count'] for s in skills)
            skill_dist = {}
            for s in skills:
                pct = s['usage_count'] / total_skill * 100 if total_skill else 0
                skill_dist[s['skill']] = {
                    "count": s['usage_count'],
                    "turn_pct": round(pct, 1),
                    "legal_count": s.get('opportunity_count', 0),
                    "legal_use%": round(s.get('opportunity_usage_rate', 0.0) * 100, 1),
                    "win%":  round(s['win_rate'] * 100, 1),
                }

            # リアクション分布（NTP行動）
            total_react = sum(r['count'] for r in reactions)
            react_dist = {}
            for r in reactions:
                pct = r['count'] / total_react * 100 if total_react else 0
                react_dist[r['reaction']] = {
                    "count": r['count'],
                    "pct":   round(pct, 1),
                }

            snapshot = {
                "step":             self.num_timesteps,
                "updated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "episodes_sampled": self.last_n,
                "win_rate_%":       round(summary['recent_100_win_rate'] * 100, 1),
                "avg_turns":        round(ep_stats.get('mean', 0), 1),
                "skills_as_tp":     skill_dist,
                "reactions_as_ntp": react_dist,
            }

            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

            if self.verbose > 0:
                print(f"[Snapshot] {self.output_path} 更新 (step={self.num_timesteps:,})")
        except Exception as e:
            if self.verbose > 0:
                print(f"[Snapshot] 書き込みエラー: {e}")


class DiversityLossCallback(BaseCallback):
    """
    DIAYN-style diversity loss (Eysenbach et al. "Diversity is All You Need", ICLR 2019).

    Persona-conditioned policy が persona z 毎に区別可能な行動を取るよう、
    discriminator q(z|s) の対数尤度を補助損失として最大化する。

    実装上の注意:
    - 観測末尾 OBS_PERSONA 次元 (persona one-hot) を zero-mask して shared_net に通す
      → discriminator が観測の persona one-hot を trivial に passthrough できないようにする
    - shared_net への back-prop は DIAYN_SHARED_LR_RATIO (1/10) 倍速
      → PPO の policy/value 更新への干渉を最小化
    - PPO の reward には介入しない (純勝敗報酬を保持)
    """

    def __init__(self, lambda_tp: float = LAMBDA_DIVERSITY_TP,
                 lambda_ntp: float = LAMBDA_DIVERSITY_NTP,
                 lr: float = 3e-4,
                 shared_lr_ratio: float = DIAYN_SHARED_LR_RATIO,
                 buffer_max: int = 8192,
                 verbose: int = 0):
        super().__init__(verbose)
        self.lambda_tp = lambda_tp
        self.lambda_ntp = lambda_ntp
        self.lr = lr
        self.shared_lr_ratio = shared_lr_ratio
        self.buffer_max = buffer_max
        self._optimizer = None
        self._device = "cpu"
        self._obs_buf: list = []
        self._tp_labels: list = []
        self._ntp_labels: list = []

    def _on_training_start(self) -> None:
        import torch
        fe = self.model.policy.features_extractor
        device = next(fe.parameters()).device
        self._device = str(device)

        # 2 lr group: discriminator heads (full lr), shared_net (低速)
        self._optimizer = torch.optim.Adam([
            {'params': (list(fe.persona_disc_shared.parameters()) +
                        list(fe.persona_tp_head.parameters()) +
                        list(fe.persona_ntp_head.parameters())),
             'lr': self.lr},
            {'params': list(fe.shared_net.parameters()),
             'lr': self.lr * self.shared_lr_ratio},
        ])

    def _on_step(self) -> bool:
        if self._optimizer is None:
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
            if i >= len(obs_array):
                continue
            if 'agent_persona_tp' not in info:
                continue
            obs_i = np.asarray(obs_array[i], dtype=np.float32).copy()
            # persona one-hot を zero-mask (trivial 解防止)
            obs_i[-OBS_PERSONA:] = 0.0
            self._obs_buf.append(obs_i)
            self._tp_labels.append(int(info['agent_persona_tp']))
            self._ntp_labels.append(int(info['agent_persona_ntp']))

        # バッファ上限
        if len(self._obs_buf) > self.buffer_max:
            drop = len(self._obs_buf) - self.buffer_max
            self._obs_buf = self._obs_buf[drop:]
            self._tp_labels = self._tp_labels[drop:]
            self._ntp_labels = self._ntp_labels[drop:]

        return True

    def _on_rollout_end(self) -> None:
        if self._optimizer is None or len(self._obs_buf) < 64:
            return

        import torch
        import torch.nn.functional as F

        fe = self.model.policy.features_extractor
        obs = torch.FloatTensor(np.array(self._obs_buf)).to(self._device)
        tp_lbl = torch.LongTensor(self._tp_labels).to(self._device)
        ntp_lbl = torch.LongTensor(self._ntp_labels).to(self._device)

        features = fe.shared_net(obs)
        tp_logits, ntp_logits = fe.persona_predictions(features)

        loss_tp = F.cross_entropy(tp_logits, tp_lbl)
        loss_ntp = F.cross_entropy(ntp_logits, ntp_lbl)
        loss = self.lambda_tp * loss_tp + self.lambda_ntp * loss_ntp

        self._optimizer.zero_grad()
        loss.backward()
        all_params: list = []
        for pg in self._optimizer.param_groups:
            all_params.extend(pg['params'])
        torch.nn.utils.clip_grad_norm_(all_params, 0.5)
        self._optimizer.step()

        with torch.no_grad():
            tp_acc = (tp_logits.argmax(dim=-1) == tp_lbl).float().mean().item()
            ntp_acc = (ntp_logits.argmax(dim=-1) == ntp_lbl).float().mean().item()

        if self.logger is not None:
            self.logger.record("train/diversity_loss", float(loss.item()))
            self.logger.record("train/diversity_loss_tp", float(loss_tp.item()))
            self.logger.record("train/diversity_loss_ntp", float(loss_ntp.item()))
            self.logger.record("diversity/disc_tp_acc", tp_acc)
            self.logger.record("diversity/disc_ntp_acc", ntp_acc)
            self.logger.record("diversity/disc_tp_baseline", 1.0 / NUM_PERSONA_TP)
            self.logger.record("diversity/disc_ntp_baseline", 1.0 / NUM_PERSONA_NTP)

        if self.verbose > 0:
            print(f"[Diversity] Step {self.num_timesteps}: "
                  f"loss={loss.item():.4f}, "
                  f"tp_acc={tp_acc:.3f} (baseline {1.0/NUM_PERSONA_TP:.3f}), "
                  f"ntp_acc={ntp_acc:.3f} (baseline {1.0/NUM_PERSONA_NTP:.3f})")

        self._obs_buf.clear()
        self._tp_labels.clear()
        self._ntp_labels.clear()

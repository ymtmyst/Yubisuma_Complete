# rl/network.py - カスタムネットワーク（補助予測ヘッド付き）
"""
PPOのActor-Criticネットワークに補助ヘッドを追加。
- メイン: 方策(Actor) + 価値(Critic)
- 補助1: 相手リアクション予測 + 相手指本数予測 + 相手スキル予測
- 補助2: 終盤勝敗予測 (終盤局面の価値表現を補強)

補助タスクは共有特徴量からの勾配を通じて、
不完全情報下での相手モデリング能力と長期戦略認識を向上させる。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.config import (
    OBS_TOTAL, NETWORK_CONFIG, TOTAL_ACTIONS,
    NUM_TP_ACTIONS, NUM_NTP_ACTIONS,
    NUM_NTP_REACTIONS, NUM_THUMB_OPTIONS, NUM_TP_SKILLS,
    AUX_REACTION_WEIGHT, AUX_THUMBS_WEIGHT, AUX_SKILL_WEIGHT, AUX_LOSS_WEIGHT,
    AUX_LOOKAHEAD_WEIGHT,
    NUM_PERSONA_TP, NUM_PERSONA_NTP,
)


class YubisumaFeaturesExtractor(BaseFeaturesExtractor):
    """
    カスタム特徴抽出器。
    観測ベクトルを共有特徴表現に変換する。
    補助予測ヘッドもここに含む。
    """
    
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = NETWORK_CONFIG["feature_dim"]):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        # 共有特徴抽出ネットワーク
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.LayerNorm(384),
            nn.Linear(384, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )
        
        # 補助予測ヘッド
        aux_hidden = NETWORK_CONFIG["aux_layers"]
        
        # 相手リアクション予測 (カウンター/ブロック/なし)
        self.aux_reaction_head = nn.Sequential(
            nn.Linear(features_dim, aux_hidden[0]),
            nn.ReLU(),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.ReLU(),
            nn.Linear(aux_hidden[1], NUM_NTP_REACTIONS),
        )
        
        # 相手指本数予測 (0/1/2)
        self.aux_thumbs_head = nn.Sequential(
            nn.Linear(features_dim, aux_hidden[0]),
            nn.ReLU(),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.ReLU(),
            nn.Linear(aux_hidden[1], NUM_THUMB_OPTIONS),
        )
        
        # 相手スキル予測 (TPスキル26種)
        self.aux_skill_head = nn.Sequential(
            nn.Linear(features_dim, aux_hidden[0]),
            nn.ReLU(),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.ReLU(),
            nn.Linear(aux_hidden[1], NUM_TP_SKILLS),
        )

        # 終盤勝敗予測 (エピソード末尾LOOKAHEAD_N手前の観測→勝敗をBCEで学習)
        # PPOの価値学習を壊さない範囲で、終盤局面の特徴表現を補助する。
        self.aux_lookahead_head = nn.Sequential(
            nn.Linear(features_dim, aux_hidden[0]),
            nn.ReLU(),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.ReLU(),
            nn.Linear(aux_hidden[1], 1),
        )

        # DIAYN persona discriminator (Eysenbach et al. ICLR 2019)
        # 観測 (persona one-hot を zero-mask したもの) を shared_net に通した features から
        # persona ID を予測。学習が進めば policy が persona毎に区別可能な行動を取るよう分化する。
        # NOTE: callback 側で観測の末尾 OBS_PERSONA 次元 (persona one-hot) を 0 にして
        # shared_net に渡すことで、discriminator が trivial 解に陥らない設計。
        self.persona_disc_shared = nn.Sequential(
            nn.Linear(features_dim, aux_hidden[0]),
            nn.ReLU(),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.ReLU(),
        )
        self.persona_tp_head = nn.Linear(aux_hidden[1], NUM_PERSONA_TP)
        self.persona_ntp_head = nn.Linear(aux_hidden[1], NUM_PERSONA_NTP)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.shared_net(observations)

    def get_aux_predictions(self, features: torch.Tensor):
        """補助予測を取得（明示的呼び出し用）"""
        return {
            'reaction': self.aux_reaction_head(features),
            'thumbs': self.aux_thumbs_head(features),
            'skill': self.aux_skill_head(features),
            'lookahead': self.aux_lookahead_head(features),
        }

    def persona_predictions(self, features: torch.Tensor):
        """DIAYN discriminator: features -> (TP persona logits, NTP persona logits)."""
        h = self.persona_disc_shared(features)
        return self.persona_tp_head(h), self.persona_ntp_head(h)


class AuxiliaryLossComputer:
    """
    補助タスクの損失を計算するユーティリティクラス。
    コールバックから呼ばれる。
    """
    
    def __init__(self, feature_extractor, device='cuda'):
        self.feature_extractor = feature_extractor
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()

        # バッファ: エピソード内で相手の行動を記録
        self.obs_buffer = []
        self.reaction_targets = []
        self.thumbs_targets = []
        self.skill_targets = []
        self.is_agent_tp_buffer = []  # エージェントがTP時のデータかどうか

        # 終盤勝敗予測バッファ: エピソード末尾付近の観測→勝敗ペア
        self.lookahead_obs = []
        self.lookahead_targets = []  # 1.0=勝利, 0.0=敗北・引き分け
    
    def add_lookahead(self, obs: np.ndarray, outcome: float) -> None:
        """エピソード末尾付近の観測と最終勝敗を記録 (callbackから呼ばれる)"""
        self.lookahead_obs.append(obs.copy())
        self.lookahead_targets.append(outcome)

    def record_opponent_action(self, obs, opponent_action, is_agent_tp):
        """相手の行動を記録"""
        self.obs_buffer.append(obs.copy())
        self.is_agent_tp_buffer.append(is_agent_tp)
        
        if is_agent_tp:
            # エージェントがTP → 相手はNTP → リアクション+指を予測
            reaction = opponent_action.get('reaction')
            if reaction == "カウンター":
                self.reaction_targets.append(1)
            elif reaction == "ブロック":
                self.reaction_targets.append(2)
            else:
                self.reaction_targets.append(0)
            self.thumbs_targets.append(opponent_action.get('thumbs', 0))
            self.skill_targets.append(-1)  # NTPにはスキルなし
        else:
            # エージェントがNTP → 相手はTP → スキル+指を予測
            self.reaction_targets.append(-1)  # TPにはリアクションなし
            self.thumbs_targets.append(opponent_action.get('thumbs', 0))
            skill = opponent_action.get('skill')
            # スキルインデックスに変換
            from rl.config import TP_SKILL_OPTIONS
            skill_idx = -1
            for i, s in enumerate(TP_SKILL_OPTIONS):
                if s == skill:
                    skill_idx = i
                    break
            self.skill_targets.append(skill_idx)
    
    def compute_loss(self):
        """蓄積されたデータから補助損失を計算"""
        total_loss = torch.tensor(0.0, device=self.device)

        # --- 終盤勝敗予測損失 (BCE): obs_bufferとは独立して計算 ---
        if len(self.lookahead_obs) >= 4:
            la_obs = torch.FloatTensor(np.array(self.lookahead_obs)).to(self.device)
            la_tgt = torch.FloatTensor(self.lookahead_targets).to(self.device)
            la_features = self.feature_extractor.shared_net(la_obs)
            la_logits = self.feature_extractor.aux_lookahead_head(la_features).squeeze(-1)
            total_loss = total_loss + F.binary_cross_entropy_with_logits(la_logits, la_tgt) * AUX_LOOKAHEAD_WEIGHT

        # --- 相手行動予測損失 ---
        if len(self.obs_buffer) < 8:
            return total_loss

        obs_tensor = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        features = self.feature_extractor.shared_net(obs_tensor)
        preds = self.feature_extractor.get_aux_predictions(features)

        pred_loss = torch.tensor(0.0, device=self.device)

        # リアクション予測損失 (エージェントがTPの場合のみ)
        tp_indices = [i for i, is_tp in enumerate(self.is_agent_tp_buffer) if is_tp]
        if tp_indices:
            tp_idx = torch.LongTensor(tp_indices).to(self.device)
            reaction_tgt = torch.LongTensor(
                [self.reaction_targets[i] for i in tp_indices]
            ).to(self.device)
            reaction_logits = preds['reaction'][tp_idx]
            pred_loss = pred_loss + self.ce_loss(reaction_logits, reaction_tgt) * AUX_REACTION_WEIGHT

        # 指本数予測損失 (全データ)
        valid_thumbs = [i for i, t in enumerate(self.thumbs_targets) if t >= 0]
        if valid_thumbs:
            t_idx = torch.LongTensor(valid_thumbs).to(self.device)
            thumbs_tgt = torch.LongTensor(
                [self.thumbs_targets[i] for i in valid_thumbs]
            ).to(self.device)
            thumbs_logits = preds['thumbs'][t_idx]
            pred_loss = pred_loss + self.ce_loss(thumbs_logits, thumbs_tgt) * AUX_THUMBS_WEIGHT

        # スキル予測損失 (エージェントがNTPの場合のみ)
        ntp_indices = [i for i, is_tp in enumerate(self.is_agent_tp_buffer)
                       if not is_tp and self.skill_targets[i] >= 0]
        if ntp_indices:
            ntp_idx = torch.LongTensor(ntp_indices).to(self.device)
            skill_tgt = torch.LongTensor(
                [self.skill_targets[i] for i in ntp_indices]
            ).to(self.device)
            skill_logits = preds['skill'][ntp_idx]
            pred_loss = pred_loss + self.ce_loss(skill_logits, skill_tgt) * AUX_SKILL_WEIGHT

        # 個別重みの合計(0.4+0.3+0.3=1.0)に全体スケールAUX_LOSS_WEIGHT(0.3)を適用
        total_loss = total_loss + pred_loss * AUX_LOSS_WEIGHT

        return total_loss

    def clear_buffer(self):
        """バッファをクリア"""
        self.obs_buffer.clear()
        self.reaction_targets.clear()
        self.thumbs_targets.clear()
        self.skill_targets.clear()
        self.is_agent_tp_buffer.clear()
        self.lookahead_obs.clear()
        self.lookahead_targets.clear()

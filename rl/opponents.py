# rl/opponents.py - 対戦相手管理（リーグ）＋固定方策ラッパー
"""
リーグ方式の対戦相手管理。
- _FrozenPolicy: パラメータ固定の方策ラッパー（env.pyから使用）
- チェックポイントをstep番号で保存（再開時の上書き防止）
- 対戦時は過去のチェックポイントからランダムに選択
- 最新/ランダム/初期の重み付き選択で多様な相手と対戦
"""

import os
import random
import glob

from rl.config import LEAGUE_DIR, LEAGUE_CONFIG


class _FrozenPolicy:
    """パラメータ固定の方策ラッパー（サブプロセス内で使用）"""

    def __init__(self, model):
        self.model = model

    def predict(self, obs, action_masks=None, deterministic=False):
        import torch
        import numpy as np

        with torch.no_grad():
            device = next(self.model.policy.parameters()).device
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            if action_masks is not None:
                mask_tensor = torch.BoolTensor(action_masks).unsqueeze(0).to(device)
                dist = self.model.policy.get_distribution(
                    obs_tensor, action_masks=mask_tensor
                )
                action = dist.get_actions(deterministic=deterministic)
                return action.cpu().numpy().flatten(), None
            else:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                return np.array([int(action)]), None


def _create_frozen_policy(model):
    """固定方策を作成"""
    return _FrozenPolicy(model)


class LeagueManager:
    """リーグ方式の対戦相手管理"""

    def __init__(self, model_class=None):
        """
        Args:
            model_class: MaskablePPOクラス（遅延importのため）
        """
        self.model_class = model_class
        self.opponents = []   # (base_path, timestep) のリスト
        self.win_rates = {}   # timestep -> win_rate

        os.makedirs(LEAGUE_DIR, exist_ok=True)
        self._load_existing_opponents()

    def _load_existing_opponents(self):
        """既存のチェックポイントを読み込み（step_*.zip 形式）"""
        pattern = os.path.join(LEAGUE_DIR, "step_*.zip")
        paths = sorted(glob.glob(pattern))
        self.opponents = []
        for p in paths:
            basename = os.path.basename(p)  # step_0000012345.zip
            try:
                step = int(basename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            base_path = p[:-4]  # .zip を除いたベースパス
            self.opponents.append((base_path, step))

    def save_checkpoint(self, model, timestep):
        """現在のモデルをリーグに保存（step番号で命名→再開時の上書き防止）"""
        base_path = os.path.join(LEAGUE_DIR, f"step_{timestep:010d}")
        model.save(base_path)
        self.opponents.append((base_path, timestep))

        # 最大数を超えたら古いものを削除（最初と最新は保持）
        max_opps = LEAGUE_CONFIG["max_opponents"]
        if len(self.opponents) > max_opps:
            keep_first = 3
            keep_last = 5
            if len(self.opponents) > keep_first + keep_last:
                middle = self.opponents[keep_first:-keep_last]
                keep_middle = max_opps - keep_first - keep_last
                if len(middle) > keep_middle:
                    kept = sorted(random.sample(middle, keep_middle),
                                  key=lambda x: x[1])
                    to_remove = set(m[0] for m in middle) - set(m[0] for m in kept)
                    for p in to_remove:
                        zip_p = p + ".zip"
                        if os.path.exists(zip_p):
                            os.remove(zip_p)
                    self.opponents = (
                        self.opponents[:keep_first] + kept +
                        self.opponents[-keep_last:]
                    )

    def select_opponent(self):
        """対戦相手をリーグから選択し (base_path, timestep) を返す"""
        if not self.opponents:
            return None, None

        config = LEAGUE_CONFIG
        r = random.random()

        if r < config["recent_weight"] and len(self.opponents) >= 1:
            path, step = self.opponents[-1]
        elif (r < config["recent_weight"] + config["initial_weight"]
              and len(self.opponents) >= 2):
            path, step = self.opponents[0]
        else:
            path, step = random.choice(self.opponents)

        return path, step

    def get_opponent_policy(self):
        """対戦相手の方策を取得（evaluate.py等からの利用向け）"""
        path, step = self.select_opponent()
        if path is None or self.model_class is None:
            return None, None
        try:
            model = self.model_class.load(path)
            return _create_frozen_policy(model), step
        except Exception as e:
            print(f"[League] 対戦相手読み込みエラー: {e}")
            return None, None

    def update_win_rate(self, timestep, wins, total):
        """勝率を更新"""
        if total > 0:
            self.win_rates[timestep] = wins / total

    @property
    def num_opponents(self):
        return len(self.opponents)

    def get_stats(self):
        """リーグの統計情報"""
        return {
            "total_opponents": len(self.opponents),
            "timesteps": [s for _, s in self.opponents],
            "win_rates": dict(self.win_rates),
        }

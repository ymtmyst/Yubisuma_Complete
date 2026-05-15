# yubisuma_base.py - 完全ルール版

from yubisuma_constants import MAX_HANDS


class Player:
    """プレイヤーの状態を管理するクラス"""

    def __init__(self, name, key):
        self.name = name  # 表示名
        self.key = key    # 内部キー (KEY_PLAYER / KEY_COMPUTER)
        self.left_hand = True
        self.right_hand = True

        # === バフ ===
        self.guard_active = False       # ガード: 相手の一発上がりを1回無効化
        self.charge_active = False      # チャージ: 次の数字宣言で一発上がり
        self.quick_level = 0            # クイック: 0=なし, 2=一発上がり, 1=手を1つ降ろす

        # === デバフ ===
        self.cement = None              # セメント: None=制限なし, int=最低指本数
        self.lock_debuff = 0            # ロック: 0=なし, 2=次ターンで有効化, 1=有効(カウンター不可)
        self.skip_phases = 0            # スキップ: スキル封印フェーズ数

        # === 必殺スキル ===
        self.used_ultimate = False      # 必殺スキル使用済みフラグ

        # === ストック ===
        self.stock = []                 # ストックしたスキルのリスト
        self.choice_used_this_phase = set()  # このフェーズ中にチョイスで使用したスキル

        # === タイム ===
        self.time_active = False        # タイム効果: 相手が追加ターンを得た時に発動して自分のターンへ

        # === ドロップ ===
        self.drop_blocked_skills = set()  # ドロップで封じられたスキル

    def get_active_hands(self):
        """アクティブな手の数を取得"""
        hands = 0
        if self.left_hand:
            hands += 1
        if self.right_hand:
            hands += 1
        return hands

    def remove_hand(self):
        """手を1つ降ろす"""
        if self.right_hand:
            self.right_hand = False
        elif self.left_hand:
            self.left_hand = False
        
        # セメント上限の更新
        if self.cement is not None:
            self.cement = min(self.cement, self.get_active_hands())

    def remove_all_hands(self):
        """全ての手を降ろす（一発上がり）"""
        self.left_hand = False
        self.right_hand = False
        if self.cement is not None:
            self.cement = 0

    def reset_phase_state(self):
        """フェーズ開始時のリセット処理"""
        self.choice_used_this_phase = set()

    def get_swappable_state(self):
        """リバーシ用: 入れ替え対象の状態を取得
        対象: 手, ガード, チャージ, クイック, ロック, セメント
        除外: スキップ, タイム, ドロップ, ストック, 必殺使用済み, 追加ターン
        """
        return {
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            "guard_active": self.guard_active,
            "charge_active": self.charge_active,
            "quick_level": self.quick_level,
            "lock_debuff": self.lock_debuff,
            "cement": self.cement,
        }

    def set_swappable_state(self, state):
        """リバーシ用: 状態を設定"""
        self.left_hand = state["left_hand"]
        self.right_hand = state["right_hand"]
        self.guard_active = state["guard_active"]
        self.charge_active = state["charge_active"]
        self.quick_level = state["quick_level"]
        self.lock_debuff = state["lock_debuff"]
        self.cement = state["cement"]


def count_total_thumbs(p1_thumbs, p2_thumbs):
    """両者の指の合計を算出"""
    return p1_thumbs + p2_thumbs

# yubisuma_base.py - 完全ルール（新）版

from yubisuma_constants import MAX_HANDS


class Player:
    """プレイヤーの状態を管理するクラス"""

    def __init__(self, name, key):
        self.name = name  # 表示名
        self.key = key    # 内部キー (KEY_PLAYER / KEY_COMPUTER)
        self.left_hand = True
        self.right_hand = True

        # === バフ ===
        self.guard_active = False       # ガード: 「次の自分のフェーズのはじめまで継続」
        self.charge_active = False      # チャージ: 次の数字宣言で効果を2回分発動
        self.quick_level = 0            # クイック: 0=なし, 2=次のターン中で手2つ降ろし, 1=その次のターン中で手1つ降ろし
        self.mirror_ready = False       # ミラー（準備）: 相手ターン中にミラー（メイン）を宣言できる

        # === デバフ ===
        self.cement = None              # セメント: None=制限なし, int=最低指本数（ゲーム終了時まで継続）
        self.lock_pending = False       # ロック: 次の自分のターン中に有効化される（フラグ方式、累積なし）
        self.lock_active = False        # ロック: 現在のターンでロック状態か（相手ターン中スキル宣言不可）
        self.skip_phases = 0            # スキップ: スキル封印フェーズ数

        # === 必殺スキル ===
        self.used_ultimate = False      # 必殺スキル使用済みフラグ

        # === ストック（フィールド効果） ===
        self.stock = []                 # ストックしたスキルのリスト
        self.stock_alpha_used_this_phase = False  # チョイス/オール/ドロップの1フェーズ1回制限フラグ
        self.choice_used_this_phase = set()       # チョイスで選んだスキルの記録（重複防止用、互換性のため残置）

        # === タイム（フィールド効果） ===
        self.time_active = False        # タイム効果: 相手が連続行動しようとした時に発動

        # === ドロップ（デバフ） ===
        self.drop_blocked_skills = set()  # ドロップで封じられたスキル

        # === 勝利前提条件: スキル宣言フラグ ===
        self.has_declared_skill = False  # ゲーム中に少なくとも1回スキルを宣言したか

        # === ターンスコープのフラグ ===
        self.hands_lower_blocked_this_turn = False  # ガード発動時にON、ターン終了でリセット

    def get_active_hands(self):
        """アクティブな手の数を取得"""
        hands = 0
        if self.left_hand:
            hands += 1
        if self.right_hand:
            hands += 1
        return hands

    def remove_hand(self):
        """手を1つ降ろす（ターン中ブロックがかかっている場合は無効）"""
        if self.hands_lower_blocked_this_turn:
            return False
        if self.right_hand:
            self.right_hand = False
        elif self.left_hand:
            self.left_hand = False
        else:
            return False

        # セメント上限の更新
        if self.cement is not None:
            self.cement = min(self.cement, self.get_active_hands())
        return True

    def remove_two_hands(self):
        """手を2つ同時に降ろす（ガード発動チェックは呼び出し側で行う）"""
        if self.hands_lower_blocked_this_turn:
            return False
        before = self.get_active_hands()
        self.left_hand = False
        self.right_hand = False
        if self.cement is not None:
            self.cement = 0
        return before > 0

    def reset_phase_state(self):
        """フェーズ開始時のリセット処理"""
        self.choice_used_this_phase = set()
        self.stock_alpha_used_this_phase = False

    def reset_turn_state(self):
        """ターン開始時のリセット処理"""
        self.hands_lower_blocked_this_turn = False

    def get_swappable_state(self):
        """リバーシ用: 入れ替え対象の状態を取得
        新ルール分類:
        - バフ: ガード, クイック, チャージ, ミラー
        - デバフ: セメント, ロック, ドロップ
        - フィールド効果（対象外）: スキップ, ストック, タイム
        """
        return {
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            # バフ
            "guard_active": self.guard_active,
            "charge_active": self.charge_active,
            "quick_level": self.quick_level,
            "mirror_ready": self.mirror_ready,
            # デバフ
            "cement": self.cement,
            "lock_pending": self.lock_pending,
            "lock_active": self.lock_active,
            "drop_blocked_skills": set(self.drop_blocked_skills),
        }

    def set_swappable_state(self, state):
        """リバーシ用: 状態を設定"""
        self.left_hand = state["left_hand"]
        self.right_hand = state["right_hand"]
        self.guard_active = state["guard_active"]
        self.charge_active = state["charge_active"]
        self.quick_level = state["quick_level"]
        self.mirror_ready = state["mirror_ready"]
        self.cement = state["cement"]
        self.lock_pending = state["lock_pending"]
        self.lock_active = state["lock_active"]
        self.drop_blocked_skills = state["drop_blocked_skills"]


def count_total_thumbs(p1_thumbs, p2_thumbs):
    """両者の指の合計を算出"""
    return p1_thumbs + p2_thumbs

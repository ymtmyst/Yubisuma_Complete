# yubisuma_effects.py - 完全ルール版

from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER


class EffectManager:
    """ゲーム全体のエフェクト・状態管理"""

    def __init__(self):
        # 追加ターン
        self.additional_turns = {KEY_PLAYER: 0, KEY_COMPUTER: 0}

        # ターン履歴: [(player_key, skill_name), ...]
        self.turn_history = []

        # フェーズ管理
        self.phase_count = 0
        self.first_player_key = None      # 先手プレイヤーキー
        self.is_first_phase_done = {KEY_PLAYER: False, KEY_COMPUTER: False}

        # 現在のフェーズ内でのターン数
        self.turns_in_current_phase = 0

        # スキップ連鎖判定用: 直前ターンでスキップを宣言 or スキップ効果を発動したか
        self.last_turn_was_skip = {KEY_PLAYER: False, KEY_COMPUTER: False}

    # === 追加ターン ===
    def add_extra_turns(self, player_key, count=1):
        self.additional_turns[player_key] += count

    def has_extra_turn(self, player_key):
        return self.additional_turns[player_key] > 0

    def use_extra_turn(self, player_key):
        if self.additional_turns[player_key] > 0:
            self.additional_turns[player_key] -= 1
            return True
        return False

    # === ターン履歴 ===
    def record_turn(self, player_key, skill_name):
        """ターンの記録"""
        self.turn_history.append((player_key, skill_name))

    def get_previous_turn_skill(self):
        """1ターン前に宣言されたスキル名を取得"""
        if len(self.turn_history) >= 1:
            return self.turn_history[-1][1]
        return None

    # === 先手制限 ===
    def is_first_phase_restricted(self, player_key):
        """先手プレイヤーの開幕1フェーズ目かどうか"""
        return (
            player_key == self.first_player_key
            and not self.is_first_phase_done[player_key]
        )

    def mark_first_phase_done(self, player_key):
        """フェーズ完了を記録"""
        self.is_first_phase_done[player_key] = True

    # === ガード ===
    def try_block_instant_win(self, defender_player):
        """
        一発上がりをガードで防ぐ試行。
        defender_player: 一発上がりされる側（＝ガードを持っている側）のPlayerオブジェクト
        ガードが有効ならTrue（一発上がり無効化）、なければFalse
        """
        if defender_player.guard_active:
            defender_player.guard_active = False
            return True
        return False

    # === スキップ連鎖判定 ===
    def update_skip_chain(self, player_key, is_skip_effect):
        """
        スキップ連鎖状態を更新。
        is_skip_effect: スキップを宣言した or スキップの効果を発動した（コピー経由含む）
        Returns: 直前のターンがスキップだったか
        """
        was_skip = self.last_turn_was_skip[player_key]
        self.last_turn_was_skip[player_key] = is_skip_effect
        return was_skip

    # === リバーシ ===
    def swap_player_states(self, player1, player2):
        """2人のプレイヤーの状態を入れ替える
        対象: 手, ガード, チャージ, クイック, ロック, セメント
        除外: スキップ, タイム, ドロップ, ストック, 必殺使用済み, 追加ターン
        """
        state1 = player1.get_swappable_state()
        state2 = player2.get_swappable_state()
        player1.set_swappable_state(state2)
        player2.set_swappable_state(state1)

# yubisuma_effects.py - 完全ルール（新）版

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
        self.first_player_key = None

        # 現在のフェーズ内でのターン数
        self.turns_in_current_phase = 0

        # ガード追加ターン: 1フェーズ中に一度だけ発動するための制限フラグ
        # コピーでガードを2回発動しても追加ターンは1回のみ
        self.guard_extra_turn_used_this_phase = {KEY_PLAYER: False, KEY_COMPUTER: False}

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

    # === ガード ===
    def try_block_two_hand_drop(self, defender_player):
        """
        相手が手を同時に2つ降ろそうとした際、ガードによる無効化を試みる。
        ガードが有効ならガードを消費し、攻撃側のターン中手降ろし無効フラグを立てる。
        Returns: True=ガード発動（攻撃側のターン中手降ろし全無効化）, False=ガードなし
        """
        if defender_player.guard_active:
            defender_player.guard_active = False
            return True
        return False

    # === リバーシ ===
    def swap_player_states(self, player1, player2):
        """2人のプレイヤーの状態を入れ替える
        新ルール分類:
        - 対象（バフ）: ガード, クイック, チャージ, ミラー
        - 対象（デバフ）: セメント, ロック, ドロップ
        - 対象外（フィールド効果）: スキップ, ストック, タイム
        - その他対象外: 必殺使用済み, 追加ターン, has_declared_skill
        """
        state1 = player1.get_swappable_state()
        state2 = player2.get_swappable_state()
        player1.set_swappable_state(state2)
        player2.set_swappable_state(state1)

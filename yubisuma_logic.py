# yubisuma_logic.py - 完全ルール（新）版

import random
from yubisuma_constants import (
    SKILLS, MESSAGES, KEY_PLAYER, KEY_COMPUTER, PLAYER_NAMES,
    MAX_HANDS, MIN_HANDS, NORMAL_SKILLS, ANTI_COUNTER_SKILLS,
    REFERENCE_SKILLS, ULTIMATE_SKILLS, OPPONENT_TURN_SKILLS,
    REFERENCEABLE_SKILLS, TURN_PLAYER_SKILLS, GAME_CONFIG,
    STOCK_ALPHA_SKILLS,
)
from yubisuma_base import Player, count_total_thumbs
from yubisuma_effects import EffectManager


def computer_strategy(comp_hands, player_hands=None, is_guesser=False,
                      for_reaction=False, cement_min=None):
    """コンピューターのランダム戦略"""
    if for_reaction:
        return random.choice(["カウンター", "n"])
    if is_guesser:
        all_skills = list(NORMAL_SKILLS | ANTI_COUNTER_SKILLS | REFERENCE_SKILLS |
                          (ULTIMATE_SKILLS - OPPONENT_TURN_SKILLS))
        total_possible = comp_hands + (player_hands or 0)
        choices = list(range(total_possible + 1)) + all_skills
        return random.choice(choices)
    if cement_min is not None:
        return random.randint(min(cement_min, comp_hands), comp_hands)
    return random.randint(MIN_HANDS, comp_hands)


def get_valid_skills(game_state, player_key):
    """現在宣言可能なスキル一覧を取得"""
    player = game_state.get_player(player_key)
    effects = game_state.effects
    valid = []

    # スキップ中は数字のみ
    if player.skip_phases > 0:
        return []

    for skill in TURN_PLAYER_SKILLS:
        # 必殺スキル使用済みチェック
        if skill in ULTIMATE_SKILLS and player.used_ultimate:
            continue
        # リバーシが無効の場合
        if skill == "リバーシ" and not GAME_CONFIG["ENABLE_REVERSI"]:
            continue
        # ミラーが無効の場合
        if skill == "ミラー" and not GAME_CONFIG["ENABLE_MIRROR"]:
            continue
        # ブロックはターンプレイヤーが宣言するスキルではない
        if skill == "ブロック":
            continue
        # ドロップで封じられたスキル
        if skill in player.drop_blocked_skills:
            continue
        # 参照スキル（コピー/ストック）: 前ターンに参照可能なスキルがあるか
        if skill in ("コピー", "ストック"):
            prev = effects.get_previous_turn_skill()
            if prev is None or (isinstance(prev, str) and prev not in REFERENCEABLE_SKILLS):
                continue
            # ストックは数字（int）を対象外とする
            if skill == "ストック" and isinstance(prev, int):
                continue
        # ストック+α 系（チョイス/オール/ドロップ）: 1フェーズ1回制限
        if skill in STOCK_ALPHA_SKILLS:
            if player.stock_alpha_used_this_phase:
                continue
        # チョイス: ストックが空でないか
        if skill == "チョイス":
            available = [s for s in player.stock if s not in player.choice_used_this_phase]
            if not available:
                continue
        # オール: ストックが空でないか
        if skill == "オール":
            if not player.stock:
                continue
        # ドロップ: ストックが空でないか
        if skill == "ドロップ":
            if not player.stock:
                continue
        valid.append(skill)

    return valid


def get_player_guess_or_command(game_state, player_key):
    """プレイヤーの予想またはスキルの入力を取得"""
    player = game_state.get_player(player_key)
    valid_skills = get_valid_skills(game_state, player_key)

    while True:
        skill_list = "｜".join(valid_skills) if valid_skills else "なし"
        prompt = f"0以上の整数 or スキル（{skill_list}）を入力: "
        guess_input = input(prompt)

        if guess_input in valid_skills:
            return guess_input
        try:
            guess = int(guess_input)
            if guess >= MIN_HANDS:
                return guess
            print(MESSAGES["INVALID_INPUT"])
        except ValueError:
            print(MESSAGES["INVALID_INPUT"])


def get_player_thumbs(max_hands, cement_state=None):
    """プレイヤーの指の数を取得"""
    while True:
        if cement_state is not None:
            print(MESSAGES["CEMENT_NOTICE"].format(count=cement_state))
        try:
            thumbs = int(input(MESSAGES["THUMB_PROMPT"].format(max=max_hands) + ": "))
            if thumbs < MIN_HANDS or max_hands < thumbs:
                print(MESSAGES["INVALID_INPUT"] + f" ({MIN_HANDS}-{max_hands}の範囲)")
            elif cement_state is not None and thumbs < cement_state:
                print(MESSAGES["CEMENT_ERROR"].format(count=cement_state))
            else:
                return thumbs
        except ValueError:
            print(MESSAGES["NUMBER_REQUIRED"])


def get_player_reaction(game_state, player_key):
    """非ターンプレイヤーの反応（カウンター/ブロック/ミラー/なし）を取得"""
    player = game_state.get_player(player_key)

    # ロックデバフで相手ターン中スキル封じ
    lock_blocked = player.lock_active
    # ミラー（メイン）が宣言可能か
    mirror_available = (
        GAME_CONFIG["ENABLE_MIRROR"]
        and player.mirror_ready
        and not lock_blocked
    )

    while True:
        opts = []
        if not lock_blocked:
            opts.append("k:カウンター")
        if not player.used_ultimate:
            opts.append("b:ブロック")
        if mirror_available:
            opts.append("m:ミラー")
        opts.append("n:なし")
        if lock_blocked:
            print(MESSAGES["LOCK_NOTICE"])
        prompt = "相手ターン中スキルを宣言しますか？(" + "/".join(opts) + "): "

        choice = input(prompt).strip().lower()

        if choice == "n" or choice == "":
            return None
        if choice == "k" or choice == "カウンター":
            if lock_blocked:
                print("ロック効果によりカウンターは使用できません！")
                continue
            return "カウンター"
        if choice == "b" or choice == "ブロック":
            if player.used_ultimate:
                print(MESSAGES["ULTIMATE_USED"])
                continue
            return "ブロック"
        if choice == "m" or choice == "ミラー":
            if not mirror_available:
                print("ミラーは現在使用できません！")
                continue
            return "ミラー"
        print(MESSAGES["INVALID_INPUT"])


def get_choice_selection(player, reaction=None):
    """チョイスでストックからスキルを選択（カウンター確認後に選べる）"""
    available = [s for s in player.stock if s not in player.choice_used_this_phase]
    if not available:
        return None
    if reaction:
        print(f"  ※相手は「{reaction}」を宣言しています")
    else:
        print("  ※相手は相手ターン中スキルを宣言していません")
    print("ストック一覧:")
    for i, s in enumerate(available):
        print(f"  {i + 1}. {s}")
    while True:
        try:
            idx = int(input("選択 (番号): ")) - 1
            if 0 <= idx < len(available):
                return available[idx]
            print(MESSAGES["INVALID_INPUT"])
        except ValueError:
            print(MESSAGES["NUMBER_REQUIRED"])


def get_all_selection(player):
    """オールでストックから発動順序を選択"""
    if not player.stock:
        return []
    print("オール: ストック内の全スキルを発動します。発動順を選んでください。")
    print("ストック一覧:")
    for i, s in enumerate(player.stock):
        print(f"  {i + 1}. {s}")
    available = list(player.stock)
    order = []
    while available:
        line = input(f"次に発動するスキルの番号 (現状{','.join(order) or 'なし'}, 残り{len(available)}個): ").strip()
        if not line:
            print("オールは全てのストックを発動します。残りの順番も選んでください。")
            continue
        try:
            idx = int(line) - 1
            if 0 <= idx < len(available):
                order.append(available.pop(idx))
                print(f"  → 「{order[-1]}」を{len(order)}番目に追加")
            else:
                print(MESSAGES["INVALID_INPUT"])
        except ValueError:
            print(MESSAGES["NUMBER_REQUIRED"])
    return order


class GameState:
    """ゲーム全体の状態管理"""

    def __init__(self):
        self.player = Player(PLAYER_NAMES[KEY_PLAYER], KEY_PLAYER)
        self.computer = Player(PLAYER_NAMES[KEY_COMPUTER], KEY_COMPUTER)
        self.effects = EffectManager()
        self.current_player_key = None
        self.game_over = False
        self.winner = None

    def get_player(self, key):
        return self.player if key == KEY_PLAYER else self.computer

    def get_opponent(self, key):
        return self.computer if key == KEY_PLAYER else self.player

    def get_opponent_key(self, key):
        return KEY_COMPUTER if key == KEY_PLAYER else KEY_PLAYER

    def initialize_game(self):
        """ゲーム初期化"""
        self.current_player_key = random.choice([KEY_PLAYER, KEY_COMPUTER])
        self.effects.first_player_key = self.current_player_key
        cp = self.get_player(self.current_player_key)
        print(f"{cp.name}が先手です。")

    def display_state(self):
        """現在の状態を表示"""
        p = self.player
        c = self.computer

        p_display = "👍" * p.get_active_hands()
        c_display = "👍" * c.get_active_hands()

        for plr in [p, c]:
            parts = []
            if plr.guard_active:
                parts.append("🛡")
            if plr.charge_active:
                parts.append("⚡")
            if plr.quick_level > 0:
                parts.append(f"💨{plr.quick_level}")
            if plr.mirror_ready:
                parts.append("🪞")
            if plr.lock_active:
                parts.append("🔒")
            elif plr.lock_pending:
                parts.append("🔒…")
            if plr.skip_phases > 0:
                parts.append(f"⏭{plr.skip_phases}")
            if plr.time_active:
                parts.append("⏰")
            if plr.cement is not None:
                parts.append(f"🧱{plr.cement}")
            if plr.stock:
                parts.append(f"📦{len(plr.stock)}")
            suffix = " " + " ".join(parts) if parts else ""
            if plr == p:
                p_display += suffix
            else:
                c_display += suffix

        print(f"\n{p.name}の手: {p_display}")
        print(f"{c.name}の手: {c_display}")
        cp = self.get_player(self.current_player_key)
        print(f"\n{cp.name}のターンです")

    def check_victory(self):
        """勝利判定
        新ルール: 「スキルを宣言していないプレイヤーがいる間、ゲームに勝利することができない」
        """
        # 勝利前提条件: 両プレイヤーが少なくとも1回スキルを宣言済みか
        both_declared = self.player.has_declared_skill and self.computer.has_declared_skill

        if self.player.get_active_hands() == 0:
            if not both_declared:
                print(MESSAGES["VICTORY_BLOCKED"])
                return False
            print(MESSAGES["VICTORY"].format(name=self.player.name))
            self.game_over = True
            self.winner = KEY_PLAYER
            return True
        if self.computer.get_active_hands() == 0:
            if not both_declared:
                print(MESSAGES["VICTORY_BLOCKED"])
                return False
            print(MESSAGES["VICTORY"].format(name=self.computer.name))
            self.game_over = True
            self.winner = KEY_COMPUTER
            return True
        return False

    def on_phase_start(self, player_key):
        """フェーズ開始時の処理"""
        player = self.get_player(player_key)

        # ガード解除（自分のフェーズ開始時に解除）
        player.guard_active = False

        # フェーズ内チョイス制限 & ストック+α 1フェーズ1回制限リセット
        player.reset_phase_state()

        # ドロップ封印解除
        player.drop_blocked_skills = set()

        # ガード追加ターンの1フェーズ1回制限リセット
        self.effects.guard_extra_turn_used_this_phase[player_key] = False

        # フェーズカウント
        self.effects.turns_in_current_phase = 0

    def on_phase_end(self, player_key):
        """フェーズ終了時の処理"""
        player = self.get_player(player_key)

        # スキップカウンタ減少
        if player.skip_phases > 0:
            player.skip_phases -= 1

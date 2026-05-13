# yubisuma_gameplay.py - 完全ルール版: メインゲームループ

import random
from yubisuma_constants import (
    GAME_RULES, KEY_PLAYER, KEY_COMPUTER, MESSAGES,
    OPPONENT_TURN_SKILLS, ULTIMATE_SKILLS,
)
from yubisuma_logic import (
    GameState, get_player_guess_or_command, get_player_thumbs,
    get_player_reaction, computer_strategy, get_valid_skills,
)
from yubisuma_turn_handler import TurnHandler


def execute_turn(gs, tp_key):
    """
    1ターンを実行。
    全ての決定は同時公開：スキル宣言・指の数・カウンターは互いに見えない状態で決定する。
    """
    tp = gs.get_player(tp_key)
    ntp_key = gs.get_opponent_key(tp_key)
    ntp = gs.get_opponent(tp_key)
    effects = gs.effects
    is_human_turn = (tp_key == KEY_PLAYER)

    # === スキップ判定 ===
    if tp.skip_phases > 0:
        print(f"  スキップ効果により、{tp.name}はスキルを宣言できませんでした。")
        if tp.quick_level > 0:
            tp.quick_level = max(0, tp.quick_level - 1)
        return False

    # ==============================
    # Phase 1: TP（ターンプレイヤー）の決定
    #   - スキル/数字の宣言
    #   - 指の数
    # ==============================
    if is_human_turn:
        # TP = 人間: スキル選択 + 指の数を入力
        skill = get_player_guess_or_command(gs, tp_key)
        tp_cement = tp.cement
        tp_thumbs = get_player_thumbs(tp.get_active_hands(), tp_cement)
        # NTP(PC)の指はまだ未定
        ntp_thumbs = None
    else:
        # TP = PC: スキル + 指の数を内部で決定（表示しない）
        valid_skills = get_valid_skills(gs, tp_key)
        total_possible = tp.get_active_hands() + ntp.get_active_hands()
        choices = list(range(total_possible + 1))
        if valid_skills:
            choices.extend(valid_skills)
        skill = random.choice(choices)
        tp_cement = tp.cement
        tp_thumbs = computer_strategy(tp.get_active_hands(), cement_min=tp_cement)
        # NTP(人間)の指はまだ未定
        ntp_thumbs = None

    # ==============================
    # Phase 2: NTP（非ターンプレイヤー）の決定
    #   - カウンター/ブロック/なし
    #   - 指の数
    #   ※TPの宣言を知らない状態で同時に決定
    # ==============================
    if is_human_turn:
        # NTP = PC: カウンター + 指の数を内部で決定
        choices = ["n", "n"]
        if ntp.lock_debuff == 0:
            choices.append("カウンター")
        if not ntp.used_ultimate and not gs.effects.is_first_phase_restricted(ntp_key):
            choices.append("ブロック")
            
        comp_reaction = random.choice(choices)
        reaction = comp_reaction if comp_reaction != "n" else None
        
        ntp_cement = ntp.cement
        ntp_thumbs = computer_strategy(ntp.get_active_hands(), cement_min=ntp_cement)
    else:
        # NTP = 人間: カウンター + 指の数を入力
        # ※TPの宣言はまだ見えていない
        reaction = get_player_reaction(gs, ntp_key)
        ntp_cement = ntp.cement
        ntp_thumbs = get_player_thumbs(ntp.get_active_hands(), ntp_cement)

    # ==============================
    # Phase 3: 同時公開 → ターン解決
    # ==============================
    thumbs = {
        KEY_PLAYER: tp_thumbs if is_human_turn else ntp_thumbs,
        KEY_COMPUTER: ntp_thumbs if is_human_turn else tp_thumbs,
    }

    TurnHandler.resolve_turn(gs, tp_key, skill, thumbs, reaction)

    # 勝利判定
    return gs.check_victory()


def play_game():
    """メインゲームループ"""
    print(GAME_RULES)

    gs = GameState()
    gs.initialize_game()

    current_key = gs.current_player_key

    while not gs.game_over:
        # フェーズ開始
        gs.on_phase_start(current_key)

        # フェーズ内ループ（追加ターン含む）
        while True:
            gs.display_state()
            gs.effects.turns_in_current_phase += 1

            # ターン実行
            if execute_turn(gs, current_key):
                break  # 勝利

            # 追加ターンチェック
            if gs.effects.has_extra_turn(current_key):
                gs.effects.use_extra_turn(current_key)
                cp = gs.get_player(current_key)
                print(f"\n{cp.name}は追加ターンを行使します！")
                continue
            else:
                break

        if gs.game_over:
            break

        # フェーズ終了
        gs.on_phase_end(current_key)

        # === タイム効果チェック ===
        opponent_key = gs.get_opponent_key(current_key)
        opponent = gs.get_opponent(current_key)
        current_player = gs.get_player(current_key)

        next_key = opponent_key  # デフォルト: 相手のターン

        if current_player.time_active:
            # current_playerがタイムを使っている
            # 次は相手のターン → 1ターンのみ実行（追加ターン無視）→ 自分に戻る
            current_player.time_active = False

            gs.current_player_key = opponent_key
            gs.on_phase_start(opponent_key)
            gs.display_state()
            gs.effects.turns_in_current_phase += 1

            if execute_turn(gs, opponent_key):
                break

            # タイム効果: 相手の追加ターン（ガード等）を全て無視して自分に戻る
            if gs.effects.has_extra_turn(opponent_key):
                lost_turns = gs.effects.additional_turns[opponent_key]
                gs.effects.additional_turns[opponent_key] = 0
                print(f"  タイム効果により{opponent.name}の追加{lost_turns}ターンは無効化！")

            gs.on_phase_end(opponent_key)

            if gs.game_over:
                break

            next_key = current_key
            print(f"\n  タイム効果により{current_player.name}のターンに戻ります！")

        elif opponent.time_active:
            # opponent側がタイムを使っていた場合
            opponent.time_active = False

            if gs.effects.has_extra_turn(current_key):
                lost_turns = gs.effects.additional_turns[current_key]
                gs.effects.additional_turns[current_key] = 0
                print(f"  タイム効果により{current_player.name}の追加{lost_turns}ターンは無効化！")

            next_key = opponent_key
            print(f"\n  タイム効果により{opponent.name}のターンに戻ります！")

        current_key = next_key
        gs.current_player_key = current_key

    print("\nゲーム終了！")


if __name__ == "__main__":
    play_game()

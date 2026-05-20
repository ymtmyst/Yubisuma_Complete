# yubisuma_gameplay.py - 完全ルール（新）版: メインゲームループ

import random
from yubisuma_constants import (
    GAME_RULES, KEY_PLAYER, KEY_COMPUTER, MESSAGES,
    OPPONENT_TURN_SKILLS, ULTIMATE_SKILLS, GAME_CONFIG,
)
from yubisuma_logic import (
    GameState, get_player_guess_or_command, get_player_thumbs,
    get_player_reaction, computer_strategy, get_valid_skills,
    get_choice_selection, get_all_selection,
)
from yubisuma_turn_handler import TurnHandler


def _get_choice_data(gs, tp_key, skill, reaction):
    """チョイス/オール宣言時、リアクション確認後にストックから選択・順序指定
    skill: TP が宣言したスキル名
    reaction: NTP のリアクション (None/カウンター/ブロック/ミラー)
    """
    tp = gs.get_player(tp_key)
    is_human = (tp_key == KEY_PLAYER)

    if skill == "チョイス":
        if is_human:
            chosen = get_choice_selection(tp, reaction)
        else:
            available = [s for s in tp.stock if s not in tp.choice_used_this_phase]
            if not available:
                chosen = None
            elif reaction == "カウンター" and "フェイント" in available:
                chosen = "フェイント"
            else:
                chosen = random.choice(available)
        return {"choice": chosen}

    if skill == "オール":
        if is_human:
            order = get_all_selection(tp)
        else:
            # AI: ランダム順序で全ストック発動
            order = list(tp.stock)
            random.shuffle(order)
        return {"all_order": order}

    return None


def execute_turn(gs, tp_key):
    """
    1ターンを実行。
    全ての決定は同時公開：スキル宣言・指の数・カウンター/ミラーは互いに見えない状態で決定する。
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
    # Phase 1: TP の決定（スキル + 指）
    # ==============================
    if is_human_turn:
        skill = get_player_guess_or_command(gs, tp_key)
        tp_thumbs = get_player_thumbs(tp.get_active_hands(), tp.cement)
        ntp_thumbs = None
    else:
        valid_skills = get_valid_skills(gs, tp_key)
        total_possible = tp.get_active_hands() + ntp.get_active_hands()
        choices = list(range(total_possible + 1))
        if valid_skills:
            choices.extend(valid_skills)
        skill = random.choice(choices)
        tp_thumbs = computer_strategy(tp.get_active_hands(), cement_min=tp.cement)
        ntp_thumbs = None

    # ==============================
    # Phase 2: NTP の決定（リアクション + 指）
    # ==============================
    if is_human_turn:
        # NTP = PC
        choices = ["n", "n"]
        if not ntp.lock_active:
            choices.append("カウンター")
        if not ntp.used_ultimate:
            choices.append("ブロック")
        if GAME_CONFIG["ENABLE_MIRROR"] and ntp.mirror_ready and not ntp.lock_active:
            choices.append("ミラー")

        comp_reaction = random.choice(choices)
        reaction = comp_reaction if comp_reaction != "n" else None
        ntp_thumbs = computer_strategy(ntp.get_active_hands(), cement_min=ntp.cement)
    else:
        reaction = get_player_reaction(gs, ntp_key)
        ntp_thumbs = get_player_thumbs(ntp.get_active_hands(), ntp.cement)

    # ==============================
    # Phase 3: 同時公開 → ターン解決
    # ==============================
    thumbs = {
        KEY_PLAYER: tp_thumbs if is_human_turn else ntp_thumbs,
        KEY_COMPUTER: ntp_thumbs if is_human_turn else tp_thumbs,
    }

    # チョイス/オール: リアクション確認後にストックから選択・順序指定
    choice_data = None
    if skill in ("チョイス", "オール"):
        choice_data = _get_choice_data(gs, tp_key, skill, reaction)

    TurnHandler.resolve_turn(gs, tp_key, skill, thumbs, reaction, choice_data)

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

        # タイム+スキップ判定用: フェーズ開始時点でのスキップ状態を記録
        phase_was_skip = gs.get_player(current_key).skip_phases > 0

        # フェーズ内ループ（追加ターン含む）
        while True:
            gs.display_state()
            gs.effects.turns_in_current_phase += 1

            # ターン実行
            if execute_turn(gs, current_key):
                break  # 勝利

            # タイム効果チェック: current_keyが追加ターンを得ようとした && 相手がtime_active
            # 新ルール: 「次に相手が2回続けてスキルを宣言しようとした時、代わりに自分がスキルを宣言する」
            opp = gs.get_opponent(current_key)
            if opp.time_active and gs.effects.has_extra_turn(current_key):
                opp.time_active = False
                lost = gs.effects.additional_turns[current_key]
                gs.effects.additional_turns[current_key] = 0
                cp_name = gs.get_player(current_key).name
                print(f"  タイム発動！{cp_name}の追加{lost}ターンを無効化 → {opp.name}のターンへ！")
                break

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

        # タイム × スキップ: スキップ宣言者が連続行動しようとする時、タイム保持者にターンを戻す
        # 「スキップ: 通常の処理を行った後、本来はスキップ宣言側のターンになるところ、
        #   代わりに宣言された側がターンを行う」
        cp = gs.get_player(current_key)
        if phase_was_skip and cp.time_active:
            cp.time_active = False  # タイム消費
            next_key = current_key
            print(f"  タイム発動！{cp.name}の連続フェーズに割り込みます")
        else:
            next_key = gs.get_opponent_key(current_key)

        current_key = next_key
        gs.current_player_key = current_key

    print("\nゲーム終了！")


if __name__ == "__main__":
    play_game()

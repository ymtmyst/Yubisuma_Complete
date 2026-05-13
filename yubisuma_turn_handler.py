# yubisuma_turn_handler.py - 完全ルール版: ターン処理

from yubisuma_constants import (
    SKILLS, KEY_PLAYER, KEY_COMPUTER, NORMAL_SKILLS,
    ANTI_COUNTER_SKILLS, REFERENCE_SKILLS, ULTIMATE_SKILLS,
    REFERENCEABLE_SKILLS, GAME_CONFIG,
)
from yubisuma_base import count_total_thumbs


class TurnHandler:
    """ターン内のスキル解決処理"""

    @staticmethod
    def resolve_turn(gs, tp_key, skill, thumbs, reaction):
        """ターンを解決する。"""
        tp = gs.get_player(tp_key)
        ntp_key = gs.get_opponent_key(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects
        total = count_total_thumbs(thumbs[KEY_PLAYER], thumbs[KEY_COMPUTER])

        skill_name = "数字" if isinstance(skill, int) else skill

        # === 結果表示 ===
        print("\n=== 結果表示 ===")
        print(f"{tp.name}の宣言: {skill}")
        print(f"プレイヤー: {thumbs[KEY_PLAYER]}本")
        print(f"コンピューター: {thumbs[KEY_COMPUTER]}本")
        print(f"合計: {total}本")
        if reaction:
            print(f"{ntp.name}が{reaction}を宣言！")

        # === チャージ消費（数字宣言時に即時消費）===
        charge_was_active = False
        if isinstance(skill, int) and tp.charge_active:
            tp.charge_active = False
            charge_was_active = True
            print(f"  -> {tp.name}のチャージが消費されました")

        # === ターン履歴記録 ===
        effects.record_turn(tp_key, skill)

        # === クイックレベル記録（ターン終了時に減少）===
        quick_before = tp.quick_level

        # === チョイス特別処理（カウンター確認後に選択）===
        if skill_name == "チョイス":
            TurnHandler._resolve_choice_with_reaction(gs, tp_key, thumbs, total, reaction)
            # クイックレベル減少
            if quick_before > 0:
                tp.quick_level = max(0, quick_before - 1)
            # ロックデバフ減少（NTPがデバフ持ちの場合）
            if ntp.lock_debuff > 0:
                ntp.lock_debuff -= 1
            return

        # === スキップ連鎖判定 ===
        was_skip_before = effects.last_turn_was_skip[tp_key]
        effects.last_turn_was_skip[tp_key] = False  # いったん初期化（効果発動時に再セット）

        # === 必殺スキル宣言フラグ（効果の発動有無に関わらず宣言時に確定）===
        if skill_name in ULTIMATE_SKILLS:
            tp.used_ultimate = True

        # === 解決フェーズ ===

        # 1. 対カウンタースキル + カウンター
        if skill_name in ANTI_COUNTER_SKILLS and reaction == "カウンター":
            TurnHandler._resolve_anti_counter(gs, tp_key, skill_name)
        # 2. ブロック使用
        elif reaction == "ブロック":
            ntp.used_ultimate = True
            if skill_name == "スキップ":
                print(f"  ブロックはスキップに対しては無効！")
                TurnHandler._resolve_skill_effect(gs, tp_key, skill, thumbs, total, was_skip_before, charge_was_active)
            else:
                print(f"  {ntp.name}のブロックにより{tp.name}のスキル効果が無効化！")
        # 3. カウンター使用（対カウンタースキル以外）
        elif reaction == "カウンター":
            TurnHandler._resolve_counter(gs, tp_key, skill, thumbs, total, charge_was_active)
        # 4. 通常解決
        else:
            TurnHandler._resolve_skill_effect(gs, tp_key, skill, thumbs, total, was_skip_before, charge_was_active)

        # === クイックレベル減少（ターン終了時）===
        if quick_before > 0:
            tp.quick_level = max(0, quick_before - 1)

        # === ロックデバフ減少（NTPがデバフ持ちの場合、このターンで消費）===
        if ntp.lock_debuff > 0:
            ntp.lock_debuff -= 1

    @staticmethod
    def _resolve_anti_counter(gs, tp_key, skill_name):
        """対カウンタースキル + カウンターの解決"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        if skill_name == "フェイント":
            tp.remove_hand()
            gs.effects.add_extra_turns(tp_key, 1)
            print(f"  フェイント成功！{tp.name}が手を1つ降ろし、追加1ターンを得ました！")
        elif skill_name == "ロック":
            # ロックは相手へのデバフ
            ntp.lock_debuff = 2  # 次の関連ターンで1に、その次のターン終了で0に
            print(f"  ロック成功！次の{tp.name}のターン中、{ntp.name}はカウンター不可！")

    @staticmethod
    def _resolve_counter(gs, tp_key, skill, thumbs, total, charge_was_active):
        """カウンターの解決（対カウンタースキル以外）"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        if isinstance(skill, int):
            if total == skill:
                ntp.remove_hand()
                print(f"  カウンター成功！数字的中により{ntp.name}が手を1つ降ろします！")
            else:
                tp.remove_hand()
                print(f"  カウンター失敗！外れにより{tp.name}が手を1つ降ろします！")
        elif skill == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                if gs.effects.try_block_instant_win(tp):
                    print(f"  {tp.name}のガードによりカウンターフラッシュが無効化！")
                else:
                    ntp.remove_all_hands()
                    print(f"  カウンター成功！{ntp.name}がフラッシュ効果で一発上がり！")
            else:
                print(f"  カウンターしたがフラッシュ条件不成立（指の数が不一致）")
        else:
            print(f"  カウンターにより{skill}の効果は発動せず、何も起こりません")

    @staticmethod
    def _resolve_skill_effect(gs, tp_key, skill, thumbs, total, was_skip_before, charge_was_active):
        """スキル効果の通常解決"""
        tp = gs.get_player(tp_key)
        ntp_key = gs.get_opponent_key(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        # --- 数字 ---
        if isinstance(skill, int):
            if total == skill:
                if charge_was_active:
                    if effects.try_block_instant_win(ntp):
                        print(f"  {ntp.name}のガードによりチャージ一発上がりが無効化！")
                    else:
                        tp.remove_all_hands()
                        print(f"  チャージ効果！数字的中で{tp.name}が一発上がり！")
                else:
                    tp.remove_hand()
                    print(f"  的中！{tp.name}が手を1つ降ろします！")
            else:
                print("  予想が外れました。")
            return

        # --- フラッシュ ---
        if skill == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                if effects.try_block_instant_win(ntp):
                    print(f"  {ntp.name}のガードによりフラッシュが無効化！")
                else:
                    tp.remove_all_hands()
                    print(f"  フラッシュ成功！{tp.name}が一発上がり！")
            else:
                print("  フラッシュ失敗（指の数が不一致）")
            return

        # --- セメント ---
        if skill == "セメント":
            for key in [KEY_PLAYER, KEY_COMPUTER]:
                t = thumbs[key]
                plr = gs.get_player(key)
                if t > 0:
                    new_cement = min(t, plr.get_active_hands())
                    if plr.cement is None or new_cement > plr.cement:
                        plr.cement = new_cement
                    print(f"  {plr.name}に{new_cement}本以上のセメント制限を付与")
            return

        # --- ガード ---
        if skill == "ガード":
            tp.guard_active = True
            effects.add_extra_turns(tp_key, 1)
            print(f"  {tp.name}がガードを展開！一発上がりを1回無効化 + 追加1ターン")
            return

        # --- チャージ ---
        if skill == "チャージ":
            tp.charge_active = True
            print(f"  {tp.name}がチャージ！次の数字宣言で一発上がり！")
            return

        # --- クイック ---
        if skill == "クイック":
            if tp.quick_level == 2:
                if effects.try_block_instant_win(ntp):
                    print(f"  {ntp.name}のガードによりクイック一発上がりが無効化！")
                else:
                    tp.remove_all_hands()
                    print(f"  クイック効果発動！{tp.name}が一発上がり！")
                tp.quick_level = 0
            elif tp.quick_level == 1:
                tp.remove_hand()
                print(f"  クイック効果発動！{tp.name}が手を1つ降ろします！")
                tp.quick_level = 0
            else:
                tp.quick_level = 2
                print(f"  {tp.name}がクイックを宣言！次ターンで再宣言すると一発上がり！")
            return

        # --- スキップ ---
        if skill == "スキップ":
            if was_skip_before:
                ntp.skip_phases += 2
                print(f"  スキップ連鎖！{ntp.name}の次の2フェーズ中スキル封印！")
            else:
                ntp.skip_phases += 1
                print(f"  スキップ！{ntp.name}の次のフェーズ中スキル封印！")
            effects.last_turn_was_skip[tp_key] = True  # 効果が実際に発動した時のみ連鎖フラグを立てる
            return

        # --- フェイント（カウンターされなかった場合）---
        if skill == "フェイント":
            print(f"  フェイント不発（カウンターされなかった）")
            return

        # --- ロック（カウンターされなかった場合）---
        if skill == "ロック":
            print(f"  ロック不発（カウンターされなかった）")
            return

        # --- コピー ---
        if skill == "コピー":
            TurnHandler._resolve_copy(gs, tp_key, thumbs, total, was_skip_before)
            return

        # --- ストック ---
        if skill == "ストック":
            prev_skill = effects.turn_history[-2][1] if len(effects.turn_history) >= 2 else None
            if prev_skill is not None and not isinstance(prev_skill, int) and prev_skill in REFERENCEABLE_SKILLS:
                tp.stock.append(prev_skill)
                print(f"  {tp.name}が「{prev_skill}」をストック！(計{len(tp.stock)}個)")
            else:
                print(f"  ストック失敗（参照可能なスキルがありません）")
            return

        # --- ドロップ ---
        if skill == "ドロップ":
            ntp.drop_blocked_skills = set(tp.stock)
            effects.add_extra_turns(tp_key, 1)
            blocked_list = ", ".join(tp.stock) if tp.stock else "なし"
            print(f"  ドロップ！{ntp.name}の次フェーズ中 [{blocked_list}] を封印 + 追加1ターン")
            return

        # --- ブースト ---
        if skill == "ブースト":
            tp.used_ultimate = True
            effects.add_extra_turns(tp_key, 3)
            print(f"  ブースト！{tp.name}が追加3ターンを獲得！")
            return

        # --- リバーシ ---
        if skill == "リバーシ":
            tp.used_ultimate = True
            effects.swap_player_states(tp, ntp)
            print(f"  リバーシ！{tp.name}と{ntp.name}の状態が入れ替わりました！")
            return

        # --- タイム ---
        if skill == "タイム":
            tp.used_ultimate = True
            tp.time_active = True
            effects.add_extra_turns(tp_key, 1)
            print(f"  タイム！{ntp.name}の次スキル後、{tp.name}のターンになる + 追加1ターン")
            return

    @staticmethod
    def _resolve_copy(gs, tp_key, thumbs, total, was_skip_before=False):
        """コピーの解決"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        if len(effects.turn_history) >= 2:
            prev_skill = effects.turn_history[-2][1]
        else:
            print("  コピー失敗（参照可能なスキルがありません）")
            return

        if isinstance(prev_skill, str) and prev_skill not in REFERENCEABLE_SKILLS:
            print("  コピー失敗（参照不可能なスキル）")
            return

        print(f"  コピー！「{prev_skill}」の効果を発動します")
        TurnHandler._execute_copied_skill(gs, tp_key, prev_skill, thumbs, total, upgrade_hand_to_win=True, was_skip_before=was_skip_before)

    @staticmethod
    def _resolve_choice_with_reaction(gs, tp_key, thumbs, total, reaction):
        """チョイスの解決（カウンター確認後に選択）"""
        tp = gs.get_player(tp_key)
        ntp_key = gs.get_opponent_key(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        # ブロックされた場合: 効果無効（選択自体スキップ）
        if reaction == "ブロック":
            ntp.used_ultimate = True
            print(f"  {ntp.name}のブロックによりチョイスの効果が無効化！")
            return

        # プレイヤーがカウンター有無を見てからストックを選択
        if tp.key == KEY_PLAYER:
            from yubisuma_logic import get_choice_selection
            chosen = get_choice_selection(tp, reaction)
        else:
            available = [s for s in tp.stock if s not in tp.choice_used_this_phase]
            if available:
                import random
                # AIはカウンターされていたらフェイント優先
                if reaction == "カウンター" and "フェイント" in available:
                    chosen = "フェイント"
                else:
                    chosen = random.choice(available)
            else:
                chosen = None

        if chosen is None:
            print("  チョイス失敗（選択可能なスキルがありません）")
            return

        tp.choice_used_this_phase.add(chosen)
        print(f"  チョイス！「{chosen}」の効果を発動します")

        # スキップ連鎖判定（チョイスでスキップを選んだ場合も連鎖対象）
        was_skip_before = effects.last_turn_was_skip[tp_key]
        if chosen == "スキップ":
            effects.last_turn_was_skip[tp_key] = True

        # 選んだスキルにリアクションを適用
        if chosen in ANTI_COUNTER_SKILLS and reaction == "カウンター":
            # 対カウンタースキルをチョイス → 対カウンター効果発動
            TurnHandler._resolve_anti_counter(gs, tp_key, chosen)
        elif reaction == "カウンター":
            # カウンターだが対カウンタースキルでない → カウンター処理
            # チョイスで選んだスキルに対してカウンター適用
            # フラッシュの場合はカウンター側がフラッシュ発動
            if chosen == "フラッシュ":
                tp_thumbs = thumbs[tp.key]
                ntp_thumbs = thumbs[ntp.key]
                if tp_thumbs == ntp_thumbs:
                    if gs.effects.try_block_instant_win(tp):
                        print(f"    {tp.name}のガードによりカウンターフラッシュが無効化！")
                    else:
                        ntp.remove_all_hands()
                        print(f"    カウンター成功！{ntp.name}がフラッシュで一発上がり！")
                else:
                    print(f"    カウンターしたがフラッシュ条件不成立")
            else:
                print(f"    カウンターにより{chosen}の効果は発動せず、何も起こりません")
        else:
            # リアクションなし → 通常効果発動
            TurnHandler._execute_copied_skill(gs, tp_key, chosen, thumbs, total,
                                               upgrade_hand_to_win=False,
                                               was_skip_before=was_skip_before)

    @staticmethod
    def _execute_copied_skill(gs, tp_key, skill_name, thumbs, total,
                               upgrade_hand_to_win=False, was_skip_before=False):
        """コピー/チョイスで参照したスキルの効果を実行"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        if isinstance(skill_name, int):
            if total == skill_name:
                if upgrade_hand_to_win:
                    if effects.try_block_instant_win(ntp):
                        print(f"    {ntp.name}のガードにより無効化！")
                    else:
                        tp.remove_all_hands()
                        print(f"    数字的中で一発上がり！")
                else:
                    tp.remove_hand()
                    print(f"    的中！手を1つ降ろします！")
            else:
                print("    予想が外れました。")
            return

        if skill_name == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                if effects.try_block_instant_win(ntp):
                    print(f"    {ntp.name}のガードにより無効化！")
                else:
                    tp.remove_all_hands()
                    print(f"    フラッシュ効果で一発上がり！")
            else:
                print(f"    フラッシュ条件不成立")

        elif skill_name == "セメント":
            for key in [KEY_PLAYER, KEY_COMPUTER]:
                t = thumbs[key]
                plr = gs.get_player(key)
                if t > 0:
                    new_cement = min(t, plr.get_active_hands())
                    if plr.cement is None or new_cement > plr.cement:
                        plr.cement = new_cement
                    print(f"  {plr.name}に{new_cement}本セメント")

        elif skill_name == "ガード":
            tp.guard_active = True
            effects.add_extra_turns(tp_key, 1)
            print(f"    ガード展開 + 追加1ターン")

        elif skill_name == "チャージ":
            tp.charge_active = True
            print(f"    チャージ付与！")

        elif skill_name == "クイック":
            tp.quick_level = 2
            print(f"    クイックバフ設定！")

        elif skill_name == "スキップ":
            # スキップ連鎖判定（コピー/チョイス経由でも連鎖対象）
            # update_skip_chainは呼び出し元で行われている場合と行われていない場合がある
            # コピー経由の場合はここで連鎖判定
            if was_skip_before:
                ntp.skip_phases += 2
                print(f"    スキップ連鎖！{ntp.name}の次の2フェーズ中スキル封印！")
            else:
                ntp.skip_phases += 1
                print(f"    {ntp.name}の次フェーズスキル封印！")
            # コピー経由でスキップ効果が発動した場合、連鎖フラグを立てる
            effects.last_turn_was_skip[tp_key] = True

        elif skill_name == "フェイント":
            print(f"    フェイント効果（カウンター不在のため不発）")

        elif skill_name == "ロック":
            print(f"    ロック効果（カウンター不在のため不発）")

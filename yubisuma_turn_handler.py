# yubisuma_turn_handler.py - 完全ルール（新）版: ターン処理

from yubisuma_constants import (
    SKILLS, KEY_PLAYER, KEY_COMPUTER, NORMAL_SKILLS,
    ANTI_COUNTER_SKILLS, REFERENCE_SKILLS, ULTIMATE_SKILLS,
    REFERENCEABLE_SKILLS, GAME_CONFIG, STOCK_ALPHA_SKILLS,
)
from yubisuma_base import count_total_thumbs


class TurnHandler:
    """ターン内のスキル解決処理"""

    @staticmethod
    def resolve_turn(gs, tp_key, skill, thumbs, reaction, choice_data=None):
        """ターンを解決する。
        choice_data: チョイス/オール用の追加情報
            - チョイス: {"choice": <スキル名>}
            - オール: {"all_order": [スキル名のリスト]}
        """
        tp = gs.get_player(tp_key)
        ntp_key = gs.get_opponent_key(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects
        total = count_total_thumbs(thumbs[KEY_PLAYER], thumbs[KEY_COMPUTER])

        skill_name = "数字" if isinstance(skill, int) else skill

        # ターン開始時のリセット処理
        tp.reset_turn_state()
        ntp.reset_turn_state()

        # ロックpending→active 転送（TPのターン開始時に、TPの相手がpending状態なら有効化）
        if ntp.lock_pending:
            ntp.lock_active = True
            ntp.lock_pending = False

        # === 結果表示 ===
        print("\n=== 結果表示 ===")
        print(f"{tp.name}の宣言: {skill}")
        print(f"プレイヤー: {thumbs[KEY_PLAYER]}本")
        print(f"コンピューター: {thumbs[KEY_COMPUTER]}本")
        print(f"合計: {total}本")
        if reaction:
            print(f"{ntp.name}が{reaction}を宣言！")

        # === スキル宣言フラグ（勝利前提条件用）===
        tp.has_declared_skill = True

        # === チャージ消費（数字宣言時に即時消費）===
        charge_was_active = False
        if isinstance(skill, int) and tp.charge_active:
            tp.charge_active = False
            charge_was_active = True
            print(f"  -> {tp.name}のチャージが消費されました（効果を2回分発動）")

        # === ターン履歴記録 ===
        effects.record_turn(tp_key, skill)

        # === クイックレベル記録（ターン終了時に減少）===
        quick_before = tp.quick_level

        # === 必殺スキル宣言フラグ ===
        if skill_name in ULTIMATE_SKILLS:
            tp.used_ultimate = True

        # === ストック+α 系の1フェーズ1回制限フラグ ===
        if skill_name in STOCK_ALPHA_SKILLS:
            tp.stock_alpha_used_this_phase = True

        # === ミラー（メイン）反射処理 ===
        if reaction == "ミラー":
            ntp.mirror_ready = False  # ミラー（メイン）使用後、準備状態を消費
            TurnHandler._resolve_mirror_reflection(gs, tp_key, skill, thumbs, total, charge_was_active)
            TurnHandler._end_of_turn_cleanup(gs, tp_key, quick_before, skill_name)
            return

        # === 解決フェーズ ===

        # 1. 対カウンタースキル + カウンター
        if skill_name in ANTI_COUNTER_SKILLS and reaction == "カウンター":
            TurnHandler._resolve_anti_counter(gs, tp_key, skill_name)
        # 2. ブロック使用
        elif reaction == "ブロック":
            ntp.used_ultimate = True
            # スキップはブロック無効
            is_skip_effect = skill_name == "スキップ"
            if not is_skip_effect and skill_name == "コピー" and len(effects.turn_history) >= 2:
                copied_skill = effects.turn_history[-2][1]
                if copied_skill == "スキップ":
                    is_skip_effect = True
            if is_skip_effect:
                print(f"  ブロックはスキップに対しては無効！")
                TurnHandler._resolve_skill_effect(gs, tp_key, skill, thumbs, total, charge_was_active, choice_data)
            else:
                print(f"  {ntp.name}のブロックにより{tp.name}のスキル効果が無効化！")
        # 3. カウンター使用（対カウンタースキル以外）
        elif reaction == "カウンター":
            if skill_name == "コピー":
                # 参照スキルへのカウンター: 参照元のスキルに準拠
                TurnHandler._resolve_copy_countered(gs, tp_key, thumbs, total)
            elif skill_name in ("チョイス", "オール"):
                # 参照スキル（ドロップを除く）へのカウンター
                TurnHandler._resolve_stock_alpha_countered(gs, tp_key, thumbs, total, skill_name, choice_data)
            else:
                TurnHandler._resolve_counter(gs, tp_key, skill, thumbs, total, charge_was_active)
        # 4. 通常解決
        else:
            TurnHandler._resolve_skill_effect(gs, tp_key, skill, thumbs, total, charge_was_active, choice_data)

        TurnHandler._end_of_turn_cleanup(gs, tp_key, quick_before, skill_name)

    @staticmethod
    def _end_of_turn_cleanup(gs, tp_key, quick_before, skill_name):
        """ターン終了時の後処理"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        # クイックレベル減少
        if quick_before > 0:
            tp.quick_level = max(0, quick_before - 1)

        # ロック状態の解除（自分のターン終了でNTPのロックが切れる）
        if ntp.lock_active:
            ntp.lock_active = False

    # =========================
    # 対カウンタースキル + カウンター
    # =========================
    @staticmethod
    def _resolve_anti_counter(gs, tp_key, skill_name):
        """対カウンタースキル + カウンターの解決"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        if skill_name == "フェイント":
            if tp.remove_hand():
                gs.effects.add_extra_turns(tp_key, 1)
                print(f"  フェイント成功！{tp.name}が手を1つ降ろし、追加1ターンを得ました！")
            else:
                gs.effects.add_extra_turns(tp_key, 1)
                print(f"  フェイント発動（手は降ろせず）、追加1ターンを獲得")
        elif skill_name == "ロック":
            # ロックは相手へのデバフ（フラグ方式、累積なし）
            ntp.lock_pending = True
            print(f"  ロック成功！次の{tp.name}のターン中、{ntp.name}は相手ターン中スキル宣言不可！")

    # =========================
    # カウンター（対カウンタースキル以外）
    # =========================
    @staticmethod
    def _resolve_counter(gs, tp_key, skill, thumbs, total, charge_was_active):
        """カウンターの解決"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        if isinstance(skill, int):
            # チャージで2回分発動の場合、カウンターも2回分発動
            fire_count = 2 if charge_was_active else 1
            for i in range(fire_count):
                if total == skill:
                    ntp.remove_hand()
                    print(f"  カウンター成功！数字的中により{ntp.name}が手を1つ降ろします！({i+1}/{fire_count})")
                else:
                    tp.remove_hand()
                    print(f"  カウンター失敗！外れにより{tp.name}が手を1つ降ろします！({i+1}/{fire_count})")
        elif skill == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                # カウンターフラッシュ: ntp が手を2つ降ろす（2手同時降ろし）
                TurnHandler._attempt_two_hand_drop(gs, ntp, tp, "カウンターフラッシュ")
            else:
                print(f"  カウンターしたがフラッシュ条件不成立（指の数が不一致）")
        else:
            print(f"  カウンターにより{skill}の効果は発動せず、何も起こりません")

    # =========================
    # 通常スキル効果の解決
    # =========================
    @staticmethod
    def _resolve_skill_effect(gs, tp_key, skill, thumbs, total, charge_was_active, choice_data=None):
        """スキル効果の通常解決"""
        tp = gs.get_player(tp_key)
        ntp_key = gs.get_opponent_key(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        # --- 数字 ---
        if isinstance(skill, int):
            fire_count = 2 if charge_was_active else 1
            for i in range(fire_count):
                if total == skill:
                    if tp.remove_hand():
                        print(f"  的中！{tp.name}が手を1つ降ろします！({i+1}/{fire_count})")
                    else:
                        print(f"  的中したが、手を降ろせない状態です ({i+1}/{fire_count})")
                else:
                    print(f"  予想が外れました。({i+1}/{fire_count})")
                    break  # 外れたら2回目も外れる（同じ指数のため）
            return

        # --- フラッシュ ---
        if skill == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                TurnHandler._attempt_two_hand_drop(gs, tp, ntp, "フラッシュ")
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
            # 追加1ターンは1フェーズ中に一度だけ発動
            if not effects.guard_extra_turn_used_this_phase[tp_key]:
                effects.add_extra_turns(tp_key, 1)
                effects.guard_extra_turn_used_this_phase[tp_key] = True
                print(f"  {tp.name}がガードを展開！2手同時降ろしを無効化 + 追加1ターン")
            else:
                print(f"  {tp.name}がガードを展開！2手同時降ろしを無効化（追加ターンは本フェーズ中既に取得済み）")
            return

        # --- チャージ ---
        if skill == "チャージ":
            tp.charge_active = True
            print(f"  {tp.name}がチャージ！次の数字宣言で効果を2回分発動！")
            return

        # --- クイック ---
        if skill == "クイック":
            # quick_level==2: 「次の自分のターン中」に発動 → 手を2つ降ろす
            # quick_level==1: 「その次の自分のターン中」に発動 → 手を1つ降ろす
            # quick_level==0: 初回宣言 → quick_level=2 にセット
            if tp.quick_level == 2:
                TurnHandler._attempt_two_hand_drop(gs, tp, ntp, "クイック")
                tp.quick_level = 0
            elif tp.quick_level == 1:
                if tp.remove_hand():
                    print(f"  クイック効果発動！{tp.name}が手を1つ降ろします！")
                tp.quick_level = 0
            else:
                tp.quick_level = 2
                print(f"  {tp.name}がクイックを宣言！次の自分のターン中で再宣言すると手を2つ降ろします！")
            return

        # --- スキップ ---
        if skill == "スキップ":
            # 新ルール: チェーン効果削除、常に +1
            ntp.skip_phases += 1
            print(f"  スキップ！{ntp.name}の次のフェーズ中スキル封印！")
            return

        # --- ミラー（準備）---
        if skill == "ミラー":
            tp.mirror_ready = True
            print(f"  {tp.name}がミラーを宣言！次の相手ターン中にミラー（メイン）を発動可能！")
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
            TurnHandler._resolve_copy(gs, tp_key, thumbs, total)
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

        # --- チョイス ---
        if skill == "チョイス":
            chosen = choice_data.get("choice") if choice_data else None
            if chosen is None:
                print("  チョイス失敗（選択可能なスキルがありません）")
                return
            tp.choice_used_this_phase.add(chosen)
            print(f"  チョイス！「{chosen}」の効果を発動します")
            TurnHandler._execute_stocked_skill(gs, tp_key, chosen, thumbs, total)
            return

        # --- オール ---
        if skill == "オール":
            order = choice_data.get("all_order") if choice_data else list(tp.stock)
            print(f"  オール！ストック内の {len(order)} 個のスキルを順番に発動します")
            for idx, s in enumerate(order):
                print(f"    [{idx+1}/{len(order)}] {s} を発動")
                TurnHandler._execute_stocked_skill(gs, tp_key, s, thumbs, total)
            tp.stock = []  # 発動後ストック全消滅
            print(f"  オール完了！{tp.name}のストックを全て消費")
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
            effects.add_extra_turns(tp_key, 3)
            print(f"  ブースト！{tp.name}が追加3ターンを獲得！")
            return

        # --- リバーシ ---
        if skill == "リバーシ":
            effects.swap_player_states(tp, ntp)
            print(f"  リバーシ！{tp.name}と{ntp.name}の状態が入れ替わりました！")
            return

        # --- タイム ---
        if skill == "タイム":
            tp.time_active = True
            effects.add_extra_turns(tp_key, 1)
            print(f"  タイム！{ntp.name}が連続行動しようとした時、{tp.name}のターンになる + 追加1ターン")
            return

    # =========================
    # 2手同時降ろし試行（ガード判定込み）
    # =========================
    @staticmethod
    def _attempt_two_hand_drop(gs, dropper, opponent, source_name):
        """
        dropper が手を2つ同時に降ろそうとする。opponent がガードを持っていれば発動。
        ガード発動時: dropper の hands_lower_blocked_this_turn = True
        """
        # 既にこのターンで手降ろし無効化フラグが立っているなら無効
        if dropper.hands_lower_blocked_this_turn:
            print(f"  {source_name}は{opponent.name}のガードにより無効化されました（このターン中、手を降ろせません）")
            return

        # 2手降ろしの試行: dropper が現在2手持っていなければ「2手同時降ろし」にならない
        if dropper.get_active_hands() < 2:
            # 手が1つしかない場合は「同時に2つ」ではないのでガードのトリガーにならず通る
            if dropper.remove_hand():
                print(f"  {source_name}発動！{dropper.name}が手を1つ降ろします（ガード貫通）")
            return

        # 2手同時降ろしを試行 → ガードチェック
        if gs.effects.try_block_two_hand_drop(opponent):
            # ガード発動: そのターン中 dropper は手を降ろせない
            dropper.hands_lower_blocked_this_turn = True
            print(f"  {opponent.name}のガードにより{source_name}の2手同時降ろしが無効化！")
            print(f"  → このターン中、{dropper.name}は手を降ろせなくなります")
            # ガード発動時の追加効果: 「そのターンの終了後、自分のフェーズを終了する」
            # ガード使用者のフェーズを終了するため、追加ターンをクリア
            gs.effects.additional_turns[opponent.key] = 0
            print(f"  ガード発動効果: {opponent.name}のターン終了後、フェーズを終了します")
        else:
            dropper.remove_two_hands()
            print(f"  {source_name}成功！{dropper.name}が手を2つ降ろします！")

    # =========================
    # コピー（2回分発動）
    # =========================
    @staticmethod
    def _resolve_copy(gs, tp_key, thumbs, total):
        """コピーの解決: 参照元スキルを2回分発動"""
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

        print(f"  コピー！「{prev_skill}」の効果を2回分発動します")

        # 2回ループで発動
        for i in range(2):
            print(f"    [{i+1}/2] 「{prev_skill}」を発動")
            TurnHandler._execute_referenced_skill(gs, tp_key, prev_skill, thumbs, total)

    @staticmethod
    def _resolve_copy_countered(gs, tp_key, thumbs, total):
        """コピー宣言がカウンターされた場合: 参照元のスキルに準拠した処理"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        if len(effects.turn_history) < 2:
            print("  コピー失敗（参照可能なスキルがありません）")
            return

        prev_skill = effects.turn_history[-2][1]
        if isinstance(prev_skill, str) and prev_skill not in REFERENCEABLE_SKILLS:
            print("  コピー失敗（参照不可能なスキル）")
            return

        print(f"  コピー！参照元「{prev_skill}」に対してカウンター処理を2回分適用")

        # 参照元が数字: カウンター処理を2回分（同じ指数なので同じ結果）
        if isinstance(prev_skill, int):
            for i in range(2):
                if total == prev_skill:
                    ntp.remove_hand()
                    print(f"    [{i+1}/2] カウンター成功！数字的中により{ntp.name}が手を1つ降ろします")
                else:
                    tp.remove_hand()
                    print(f"    [{i+1}/2] カウンター失敗！外れにより{tp.name}が手を1つ降ろします")
            return

        # 参照元が対カウンタースキル: 効果が2回分発動
        if prev_skill == "フェイント":
            for i in range(2):
                if tp.remove_hand():
                    gs.effects.add_extra_turns(tp_key, 1)
                    print(f"    [{i+1}/2] フェイント成功！{tp.name}が手を1つ降ろし、追加1ターン獲得")
                else:
                    gs.effects.add_extra_turns(tp_key, 1)
                    print(f"    [{i+1}/2] フェイント発動（手は降ろせず）、追加1ターン獲得")
            return

        if prev_skill == "ロック":
            ntp.lock_pending = True  # 累積しないのでフラグセットは1回でOK
            print(f"    コピー(ロック)成功！次の{tp.name}のターン中、{ntp.name}は相手ターン中スキル宣言不可！")
            return

        # 参照元がフラッシュ: カウンターフラッシュを2回分（同じ指数なので2回ともガード判定）
        if prev_skill == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                for i in range(2):
                    print(f"    [{i+1}/2] カウンターフラッシュ試行")
                    TurnHandler._attempt_two_hand_drop(gs, ntp, tp, "カウンターフラッシュ")
            else:
                print(f"    カウンターしたがフラッシュ条件不成立")
            return

        # その他: 通常カウンター=効果なし
        print(f"    カウンターにより「{prev_skill}」の効果は発動せず、何も起こりません")

    @staticmethod
    def _resolve_stock_alpha_countered(gs, tp_key, thumbs, total, skill_name, choice_data):
        """チョイス/オールがカウンターされた場合: 各参照元スキルに準拠"""
        # 簡略化: チョイスの選択スキルにカウンターを適用
        # オールの場合は全スキルにカウンターを適用（複雑なので最初の1つだけにカウンター、残りは通る）
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        if skill_name == "チョイス":
            chosen = choice_data.get("choice") if choice_data else None
            if chosen is None:
                print("  チョイス失敗（選択可能なスキルがありません）")
                return
            tp.choice_used_this_phase.add(chosen)
            print(f"  チョイス！「{chosen}」をカウンター処理")
            # カウンターを適用（参照元のスキルに準拠）
            TurnHandler._apply_counter_to_skill(gs, tp_key, chosen, thumbs, total)
            return

        if skill_name == "オール":
            # 全スキルにカウンターを適用するのは複雑なので、簡略実装: 各スキルにカウンター適用
            order = choice_data.get("all_order") if choice_data else list(tp.stock)
            print(f"  オール！{len(order)}個のスキルをカウンター処理")
            for idx, s in enumerate(order):
                print(f"    [{idx+1}/{len(order)}] {s} にカウンター適用")
                TurnHandler._apply_counter_to_skill(gs, tp_key, s, thumbs, total)
            tp.stock = []
            return

    @staticmethod
    def _apply_counter_to_skill(gs, tp_key, skill_name, thumbs, total):
        """単一スキルにカウンター処理を適用（参照元スキルに準拠）"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)

        if skill_name == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                TurnHandler._attempt_two_hand_drop(gs, ntp, tp, "カウンターフラッシュ")
            else:
                print(f"    カウンターしたがフラッシュ条件不成立")
            return

        if skill_name == "フェイント":
            if tp.remove_hand():
                gs.effects.add_extra_turns(tp_key, 1)
                print(f"    フェイント成功（カウンター経由）：{tp.name}が手を1つ降ろし、追加1ターン")
            return

        if skill_name == "ロック":
            ntp.lock_pending = True
            print(f"    ロック成功（カウンター経由）")
            return

        # その他のスキル: 通常カウンター → 効果なし
        print(f"    「{skill_name}」はカウンターにより効果不発")

    # =========================
    # ストックされたスキル効果の発動（チョイス/オール用）
    # =========================
    @staticmethod
    def _execute_stocked_skill(gs, tp_key, skill_name, thumbs, total):
        """チョイス/オールで選択したスキルの効果を発動"""
        # ストックされたスキルはカウンター反応を受けない（チョイス/オール自体への反応はメインで処理済）
        TurnHandler._execute_referenced_skill(gs, tp_key, skill_name, thumbs, total)

    @staticmethod
    def _execute_referenced_skill(gs, tp_key, skill_name, thumbs, total):
        """参照スキル経由で発動されるスキルの効果（コピー/チョイス/オール共通）"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        if isinstance(skill_name, int):
            if total == skill_name:
                if tp.remove_hand():
                    print(f"      数字的中！{tp.name}が手を1つ降ろします")
                else:
                    print(f"      数字的中（手降ろし無効）")
            else:
                print("      予想が外れました")
            return

        if skill_name == "フラッシュ":
            tp_thumbs = thumbs[tp.key]
            ntp_thumbs = thumbs[ntp.key]
            if tp_thumbs == ntp_thumbs:
                TurnHandler._attempt_two_hand_drop(gs, tp, ntp, "フラッシュ")
            else:
                print(f"      フラッシュ条件不成立")

        elif skill_name == "セメント":
            for key in [KEY_PLAYER, KEY_COMPUTER]:
                t = thumbs[key]
                plr = gs.get_player(key)
                if t > 0:
                    new_cement = min(t, plr.get_active_hands())
                    if plr.cement is None or new_cement > plr.cement:
                        plr.cement = new_cement
                    print(f"      {plr.name}に{new_cement}本セメント")

        elif skill_name == "ガード":
            tp.guard_active = True
            # 追加1ターンは1フェーズ中に一度だけ発動
            if not effects.guard_extra_turn_used_this_phase[tp_key]:
                effects.add_extra_turns(tp_key, 1)
                effects.guard_extra_turn_used_this_phase[tp_key] = True
                print(f"      ガード展開 + 追加1ターン")
            else:
                print(f"      ガード展開（追加ターンは本フェーズ中既に取得済み）")

        elif skill_name == "チャージ":
            tp.charge_active = True
            print(f"      チャージ付与！")

        elif skill_name == "クイック":
            # 累積しない（quick_level = 2 を上書き）
            tp.quick_level = 2
            print(f"      クイックバフ設定！")

        elif skill_name == "スキップ":
            # 新ルール: スキップは累積する（参照スキルで複数回発動で +1 ずつ）
            ntp.skip_phases += 1
            print(f"      {ntp.name}の次のフェーズスキル封印（累計{ntp.skip_phases}）")

        elif skill_name == "ミラー":
            tp.mirror_ready = True
            print(f"      {tp.name}のミラー（準備）状態を付与")

        elif skill_name == "フェイント":
            # チョイス/オール経由でフェイント発動: 通常効果（手降ろし+追加ターン）
            if tp.remove_hand():
                gs.effects.add_extra_turns(tp_key, 1)
                print(f"      {tp.name}が手を1つ降ろし、追加1ターン")

        elif skill_name == "ロック":
            ntp.lock_pending = True
            print(f"      ロック付与！")

    # =========================
    # ミラー（メイン）反射処理
    # =========================
    @staticmethod
    def _resolve_mirror_reflection(gs, tp_key, skill, thumbs, total, charge_was_active):
        """ミラー（メイン）の反射処理: TPのスキル効果をNTPに反射"""
        tp = gs.get_player(tp_key)
        ntp = gs.get_opponent(tp_key)
        effects = gs.effects

        skill_name = "数字" if isinstance(skill, int) else skill
        print(f"  ミラー反射！「{skill_name}」の効果を{ntp.name}が発動します")

        # 数字: TP→NTPへ反射、NTPが手を降ろす（チャージ効果も反射）
        if isinstance(skill, int):
            fire_count = 2 if charge_was_active else 1
            for i in range(fire_count):
                if total == skill:
                    ntp.remove_hand()
                    print(f"    [{i+1}/{fire_count}] 数字的中！{ntp.name}が手を1つ降ろします")
                else:
                    print(f"    [{i+1}/{fire_count}] 予想が外れました")
                    break
            return

        # フラッシュ: NTPが手を2つ降ろす（TPがガードを持っていれば TP がガード防御）
        if skill == "フラッシュ":
            if thumbs[tp.key] == thumbs[ntp.key]:
                TurnHandler._attempt_two_hand_drop(gs, ntp, tp, "ミラーフラッシュ")
            else:
                print("    フラッシュ条件不成立")
            return

        # スキップ: TPのスキップ効果を反射 → TPが封印される
        if skill == "スキップ":
            tp.skip_phases += 1
            print(f"    ミラー反射！{tp.name}の次のフェーズが封印されました")
            return

        # セメント: 通常通り両者に適用（既存ルール通り）
        if skill == "セメント":
            for key in [KEY_PLAYER, KEY_COMPUTER]:
                t = thumbs[key]
                plr = gs.get_player(key)
                if t > 0:
                    new_cement = min(t, plr.get_active_hands())
                    if plr.cement is None or new_cement > plr.cement:
                        plr.cement = new_cement
                    print(f"    {plr.name}に{new_cement}本セメント")
            return

        # クイック: NTPに反射 → NTPがクイック効果を発動
        if skill == "クイック":
            if tp.quick_level == 2:
                TurnHandler._attempt_two_hand_drop(gs, ntp, tp, "ミラークイック")
                tp.quick_level = 0
            elif tp.quick_level == 1:
                ntp.remove_hand()
                tp.quick_level = 0
                print(f"    ミラークイック！{ntp.name}が手を1つ降ろします")
            else:
                # 初回宣言の場合、ミラーによりクイックバフがNTPに移る
                ntp.quick_level = 2
                print(f"    ミラー反射！{ntp.name}にクイックバフ付与")
            return

        # ガード/チャージ/ミラー（準備）: バフ系 → 反射してNTPに付与
        if skill == "ガード":
            ntp.guard_active = True
            print(f"    ミラー反射！{ntp.name}にガード付与")
            return
        if skill == "チャージ":
            ntp.charge_active = True
            print(f"    ミラー反射！{ntp.name}にチャージ付与")
            return
        if skill == "ミラー":
            ntp.mirror_ready = True
            print(f"    ミラー反射！{ntp.name}にミラー（準備）付与")
            return

        # ドロップ: 反射 → TPがドロップを受ける
        if skill == "ドロップ":
            tp.drop_blocked_skills = set(ntp.stock)
            print(f"    ミラー反射！{tp.name}は{ntp.name}のストックを使用不可")
            return

        # その他のスキル: ミラー対象外として扱う
        print(f"    「{skill_name}」はミラーで反射しきれず、効果が不発に")


# 互換性のためのモジュールレベル参照（旧コード対応）
def get_choice_selection(player, reaction=None):
    """互換性: yubisuma_logic.py からのインポートをサポート"""
    from yubisuma_logic import get_choice_selection as _gcs
    return _gcs(player, reaction)

# test_skills.py - 全スキルの単体テスト（修正版）
import sys
sys.path.insert(0, ".")

from yubisuma_constants import *
from yubisuma_base import Player, count_total_thumbs
from yubisuma_effects import EffectManager
from yubisuma_logic import GameState, get_valid_skills
from yubisuma_turn_handler import TurnHandler

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  OK: {name}")
    else:
        failed += 1
        print(f"  NG: {name} <- FAILED")

def make_gs():
    gs = GameState()
    gs.current_player_key = KEY_PLAYER
    gs.effects.first_player_key = KEY_PLAYER
    gs.effects.is_first_phase_done[KEY_PLAYER] = True
    gs.effects.is_first_phase_done[KEY_COMPUTER] = True
    return gs

# ===== 数字 =====
print("\n=== 数字テスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("数字的中", gs.player.get_active_hands() == 1)

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, 3, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("数字外れ", gs.player.get_active_hands() == 2)

# ===== フラッシュ =====
print("\n=== フラッシュテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "フラッシュ", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("フラッシュ成功->一発上がり", gs.player.get_active_hands() == 0)

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "フラッシュ", {KEY_PLAYER: 1, KEY_COMPUTER: 0}, None)
test("フラッシュ失敗", gs.player.get_active_hands() == 2)

# ===== ガード =====
print("\n=== ガードテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ガード", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("ガード->バフ付与", gs.player.guard_active == True)
test("ガード->追加ターン", gs.effects.additional_turns[KEY_PLAYER] == 1)

gs = make_gs()
gs.computer.guard_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, "フラッシュ", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("ガードでフラッシュ無効化", gs.player.get_active_hands() == 2)
test("ガード消費済み", gs.computer.guard_active == False)

# ===== チャージ =====
print("\n=== チャージテスト ===")
gs = make_gs()
gs.player.charge_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("チャージ+数字的中->一発上がり", gs.player.get_active_hands() == 0)
test("チャージ消費", gs.player.charge_active == False)

gs = make_gs()
gs.player.charge_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 3, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("チャージ+数字外れ->手変化なし", gs.player.get_active_hands() == 2)
test("チャージ消費(外れ)", gs.player.charge_active == False)

gs = make_gs()
gs.player.charge_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, "カウンター")
test("チャージ+カウンター(当)->チャージ消費", gs.player.charge_active == False)
test("チャージ+カウンター(当)->NTP手降ろし", gs.computer.get_active_hands() == 1)
test("チャージ+カウンター(当)->TP変化なし", gs.player.get_active_hands() == 2)

gs = make_gs()
gs.player.charge_active = True
gs.computer.guard_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("チャージ+ガード->一発上がり無効", gs.player.get_active_hands() == 2)

gs = make_gs()
gs.player.charge_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, "ブロック")
test("チャージ+ブロック->チャージ消費", gs.player.charge_active == False)
test("チャージ+ブロック->手変化なし", gs.player.get_active_hands() == 2)

# ===== クイック =====
print("\n=== クイックテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイック初回->レベル2", gs.player.quick_level == 2)

gs = make_gs()
gs.player.quick_level = 2
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイックLv2+再宣言->一発上がり", gs.player.get_active_hands() == 0)

gs = make_gs()
gs.player.quick_level = 1
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイックLv1+再宣言->手1つ降ろす", gs.player.get_active_hands() == 1)

gs = make_gs()
gs.player.quick_level = 2
gs.computer.guard_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイック+ガード->一発上がり無効", gs.player.get_active_hands() == 2)

# ===== セメント（Player管理）=====
print("\n=== セメントテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "セメント", {KEY_PLAYER: 1, KEY_COMPUTER: 2}, None)
test("セメント->P制限", gs.player.cement == 1)
test("セメント->C制限", gs.computer.cement == 2)

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "セメント", {KEY_PLAYER: 1, KEY_COMPUTER: 2}, "カウンター")
test("セメント+カウンター->無効", gs.player.cement is None)

# ===== フェイント =====
print("\n=== フェイントテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "フェイント", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("フェイント+カウンター->手降ろし", gs.player.get_active_hands() == 1)
test("フェイント+カウンター->追加ターン", gs.effects.additional_turns[KEY_PLAYER] == 1)

# ===== ロック（相手デバフ）=====
print("\n=== ロックテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ロック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
# ロック発動: ntp.lock_debuff=2 → ターン末尾で-1 → 1
test("ロック+カウンター->NTPデバフ=1", gs.computer.lock_debuff == 1)
# 実際のテスト: resolve_turn内でntp.lock_debuffが減少するので
# ロック発動直後のターン終了時: 2 が設定された後にntp.lock_debuff -= 1 で 1 になる

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ロック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ロック不発->変化なし", gs.computer.lock_debuff == 0)

# ===== スキップ =====
print("\n=== スキップテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("スキップ->1フェーズ封印", gs.computer.skip_phases == 1)

gs = make_gs()
gs.effects.last_turn_was_skip[KEY_PLAYER] = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("スキップ連鎖->2フェーズ封印", gs.computer.skip_phases == 2)

# スキップ連鎖（コピー経由でスキップ効果発動後にスキップ宣言）
print("\n=== スキップ連鎖（コピー経由）===")
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "スキップ")  # 前ターン: 相手がスキップ
# コピーでスキップを発動
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("コピー(スキップ)->封印発動", gs.computer.skip_phases == 1)
test("コピー(スキップ)->連鎖フラグON", gs.effects.last_turn_was_skip[KEY_PLAYER] == True)
# 次ターンでスキップ宣言→連鎖
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("コピー後スキップ->連鎖2フェーズ", gs.computer.skip_phases == 3)  # 1(コピー) + 2(連鎖)

# ===== カウンター(数字) =====
print("\n=== カウンター詳細テスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, "カウンター")
test("カウンター+数字当たり->NTP手降ろし", gs.computer.get_active_hands() == 1)
test("カウンター+数字当たり->TP変化なし", gs.player.get_active_hands() == 2)

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, 3, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, "カウンター")
test("カウンター+数字外れ->TP手降ろし", gs.player.get_active_hands() == 1)

# ===== ブースト =====
print("\n=== ブーストテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ブースト", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ブースト->追加3ターン", gs.effects.additional_turns[KEY_PLAYER] == 3)
test("ブースト->必殺使用済み", gs.player.used_ultimate == True)

# ===== ブロック =====
print("\n=== ブロックテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "フラッシュ", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, "ブロック")
test("ブロック->フラッシュ無効化", gs.player.get_active_hands() == 2)

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "ブロック")
test("ブロック+スキップ->スキップ有効", gs.computer.skip_phases == 1)

# ===== リバーシ（セメント含む入替）=====
print("\n=== リバーシテスト ===")
GAME_CONFIG["ENABLE_REVERSI"] = True  # テスト用ON
gs = make_gs()
gs.player.right_hand = False
gs.computer.charge_active = True
gs.computer.cement = 2
TurnHandler.resolve_turn(gs, KEY_PLAYER, "リバーシ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("リバーシ->手の数入替(P)", gs.player.get_active_hands() == 2)
test("リバーシ->手の数入替(C)", gs.computer.get_active_hands() == 1)
test("リバーシ->チャージ入替(P)", gs.player.charge_active == True)
test("リバーシ->セメント入替(P)", gs.player.cement == 2)
test("リバーシ->セメント入替(C)", gs.computer.cement is None)

# リバーシでスキップ/タイム/追加ターンは入替されない
gs = make_gs()
gs.player.skip_phases = 3
gs.player.time_active = True
gs.effects.additional_turns[KEY_PLAYER] = 2
TurnHandler.resolve_turn(gs, KEY_PLAYER, "リバーシ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("リバーシ->スキップ入替されない(P)", gs.player.skip_phases == 3)
test("リバーシ->タイム入替されない(P)", gs.player.time_active == True)
test("リバーシ->追加ターン入替されない", gs.effects.additional_turns[KEY_PLAYER] == 2)
GAME_CONFIG["ENABLE_REVERSI"] = False  # 戻す

# ===== タイム =====
print("\n=== タイムテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "タイム", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("タイム->フラグ", gs.player.time_active == True)
test("タイム->追加1ターン", gs.effects.additional_turns[KEY_PLAYER] == 1)

# ===== コピー =====
print("\n=== コピーテスト ===")
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "フラッシュ")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("コピー(フラッシュ)->一発上がり", gs.player.get_active_hands() == 0)

gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, 2)
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("コピー(数字2)+的中->一発上がり", gs.player.get_active_hands() == 0)

# ===== ストック =====
print("\n=== ストックテスト ===")
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "ガード")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ストック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ストック(ガード)->保存", "ガード" in gs.player.stock)

gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, 3)
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ストック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ストック(数字3)->失敗", 3 not in gs.player.stock)

# ===== ドロップ =====
print("\n=== ドロップテスト ===")
gs = make_gs()
gs.player.stock = ["ガード", "フラッシュ"]
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ドロップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ドロップ->相手封印", gs.computer.drop_blocked_skills == {"ガード", "フラッシュ"})
test("ドロップ->追加ターン", gs.effects.additional_turns[KEY_PLAYER] == 1)

# ===== 先手制限 =====
print("\n=== 先手制限テスト ===")
gs = make_gs()
gs.effects.is_first_phase_done[KEY_PLAYER] = False
valid = get_valid_skills(gs, KEY_PLAYER)
test("先手制限->フラッシュ不可", "フラッシュ" not in valid)
test("先手制限->ブースト不可", "ブースト" not in valid)
test("先手制限->ガードは可", "ガード" in valid)

# ===== リバーシOFF =====
print("\n=== リバーシOFF設定テスト ===")
gs = make_gs()
GAME_CONFIG["ENABLE_REVERSI"] = False
valid = get_valid_skills(gs, KEY_PLAYER)
test("リバーシOFF->リバーシ不可", "リバーシ" not in valid)

# ===== スキップのフェーズ封印動作テスト =====
print("\n=== スキップ封印動作テスト ===")
gs = make_gs()
gs.computer.skip_phases = 1
# on_phase_startでは減少しない（フェーズ中は封印有効）
gs.on_phase_start(KEY_COMPUTER)
test("フェーズ開始時->skip_phases維持", gs.computer.skip_phases == 1)
valid = get_valid_skills(gs, KEY_COMPUTER)
test("フェーズ中->スキル使用不可", valid == [])
# on_phase_endで減少
gs.on_phase_end(KEY_COMPUTER)
test("フェーズ終了時->skip_phases減少", gs.computer.skip_phases == 0)
# 次フェーズではスキル使用可能
gs.on_phase_start(KEY_COMPUTER)
valid = get_valid_skills(gs, KEY_COMPUTER)
test("次フェーズ->スキル使用可能", len(valid) > 0)

# ===== スキップ連鎖(コピーがカウンターで止められた場合) =====
print("\n=== スキップ連鎖(コピー被カウンター) ===")
gs = make_gs()
# Turn 1: スキップ宣言
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("T1: スキップ->連鎖フラグON", gs.effects.last_turn_was_skip[KEY_PLAYER] == True)
# Turn 2: コピー(スキップ)がカウンターで止められる
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("T2: コピー被カウンター->連鎖フラグOFF", gs.effects.last_turn_was_skip[KEY_PLAYER] == False)
# Turn 3: スキップ宣言 → 連鎖しない（コピーは宣言されたスキルではないのでフラグ不在）
prev_skip = gs.computer.skip_phases
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("T3: コピー被カウンター後スキップ->連鎖なし", gs.computer.skip_phases == prev_skip + 1)

# ===== スキップ連鎖(スキップが被カウンター後にコピー) =====
print("\n=== スキップ連鎖(スキップ被カウンター→コピー) ===")
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "スキップ")  # 1ターン前の履歴を設定
# Turn 1: スキップ被カウンター → 宣言したので連鎖フラグON（条件A）
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("T1: スキップ被カウンター->連鎖フラグON", gs.effects.last_turn_was_skip[KEY_PLAYER] == True)
# Turn 2: コピー(スキップ) → 直前にスキップ宣言済みなので連鎖して2フェーズ封印
prev_skip = gs.computer.skip_phases
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("T2: スキップ被カウンター後コピー(スキップ)->連鎖2フェーズ", gs.computer.skip_phases == prev_skip + 2)

# ===== サマリ =====
print(f"\n{'='*40}")
print(f"結果: {passed} passed, {failed} failed, {passed+failed} total")
if failed == 0:
    print("全テスト合格！")
else:
    print(f"!!! {failed} テスト失敗 !!!")

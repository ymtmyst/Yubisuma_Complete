# test_skills.py - 完全ルール（新）版: 全スキルの単体テスト
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
    """テスト用ゲーム状態（勝利前提条件は満たした状態で開始）"""
    gs = GameState()
    gs.current_player_key = KEY_PLAYER
    gs.effects.first_player_key = KEY_PLAYER
    # 勝利前提条件: 両プレイヤーがスキルを宣言した状態にする（テスト時の勝利判定簡略化）
    gs.player.has_declared_skill = True
    gs.computer.has_declared_skill = True
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
test("フラッシュ成功->手を2つ降ろす", gs.player.get_active_hands() == 0)

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

# 新仕様: ガード追加ターンは1フェーズ1回制限
gs = make_gs()
# ガード1回目宣言
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ガード", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
# ガード2回目宣言（同フェーズ内 = guard_extra_turn_used_this_phase が True のため追加ターン取得なし）
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ガード", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ガード2回宣言->追加ターンは1回のみ", gs.effects.additional_turns[KEY_PLAYER] == 1)

# ===== チャージ =====
print("\n=== チャージテスト ===")
gs = make_gs()
gs.player.charge_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("チャージ+数字的中->2回分発動で2手降ろし", gs.player.get_active_hands() == 0)
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
test("チャージ+カウンター(当)->NTPが2手降ろし", gs.computer.get_active_hands() == 0)
test("チャージ+カウンター(当)->TP変化なし", gs.player.get_active_hands() == 2)

# 新仕様: チャージ+数字はガードに防がれない（2手同時降ろしじゃない）
gs = make_gs()
gs.player.charge_active = True
gs.computer.guard_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, 2, {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("チャージ+ガード->ガード貫通で2手降ろし", gs.player.get_active_hands() == 0)
test("チャージ+数字->ガード未消費", gs.computer.guard_active == True)

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
test("クイックLv2+再宣言->2手降ろし", gs.player.get_active_hands() == 0)

gs = make_gs()
gs.player.quick_level = 1
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイックLv1+再宣言->手1つ降ろす", gs.player.get_active_hands() == 1)

gs = make_gs()
gs.player.quick_level = 2
gs.computer.guard_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイックLv2+ガード->2手降ろし無効", gs.player.get_active_hands() == 2)
test("クイックLv2+ガード->ガード消費", gs.computer.guard_active == False)

# 新仕様: 手が1つでクイックLv2 → ガード貫通
gs = make_gs()
gs.player.quick_level = 2
gs.player.right_hand = False  # 手は1つ
gs.computer.guard_active = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, "クイック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("クイックLv2+手1個+ガード->貫通", gs.player.get_active_hands() == 0)
test("クイックLv2+手1個+ガード->ガード未消費", gs.computer.guard_active == True)

# ===== セメント =====
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

# ===== ロック（新仕様: フラグ方式）=====
print("\n=== ロックテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ロック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("ロック+カウンター->NTP.lock_pending=True", gs.computer.lock_pending == True)
test("ロック+カウンター->NTP.lock_active=False(まだTPのターンが来てない)", gs.computer.lock_active == False)

gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ロック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ロック不発->変化なし", gs.computer.lock_pending == False)

# 新仕様: ロックは累積しない
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ロック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ロック", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("ロック2回->累積しない(pending=True、フラグなので変化なし)", gs.computer.lock_pending == True)

# ===== スキップ（新仕様: チェーン削除）=====
print("\n=== スキップテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("スキップ->1フェーズ封印", gs.computer.skip_phases == 1)

# 新仕様: スキップ連続宣言してもチェーン効果なし（毎回 +1 のみ）
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
TurnHandler.resolve_turn(gs, KEY_PLAYER, "スキップ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("スキップ2連続->累積するがチェーン2倍はなし(+1+1=2)", gs.computer.skip_phases == 2)

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

# ===== リバーシ（新仕様: ドロップを含む）=====
print("\n=== リバーシテスト ===")
GAME_CONFIG["ENABLE_REVERSI"] = True
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

# 新仕様: ドロップはリバーシで入れ替えられる
gs = make_gs()
gs.player.drop_blocked_skills = {"ガード"}
TurnHandler.resolve_turn(gs, KEY_PLAYER, "リバーシ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("リバーシ->ドロップ入替(P→C)", gs.computer.drop_blocked_skills == {"ガード"})
test("リバーシ->ドロップ入替(P側はクリア)", gs.player.drop_blocked_skills == set())

# フィールド効果（スキップ/タイム/ストック）はリバーシで入れ替えされない
gs = make_gs()
gs.player.skip_phases = 3
gs.player.time_active = True
gs.player.stock = ["ガード"]
gs.effects.additional_turns[KEY_PLAYER] = 2
TurnHandler.resolve_turn(gs, KEY_PLAYER, "リバーシ", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("リバーシ->スキップ入替されない(P)", gs.player.skip_phases == 3)
test("リバーシ->タイム入替されない(P)", gs.player.time_active == True)
test("リバーシ->ストック入替されない(P)", gs.player.stock == ["ガード"])
test("リバーシ->追加ターン入替されない", gs.effects.additional_turns[KEY_PLAYER] == 2)
GAME_CONFIG["ENABLE_REVERSI"] = False

# ===== タイム =====
print("\n=== タイムテスト ===")
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "タイム", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("タイム->フラグ", gs.player.time_active == True)
test("タイム->追加1ターン", gs.effects.additional_turns[KEY_PLAYER] == 1)

# ===== コピー（新仕様: 2回分発動）=====
print("\n=== コピーテスト ===")
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "フラッシュ")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("コピー(フラッシュ)->2手降ろし(2回発動するが手は2つしかないので)", gs.player.get_active_hands() == 0)

gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, 2)
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None)
test("コピー(数字2)+的中->2回数字発動で両手降ろし", gs.player.get_active_hands() == 0)

# 新仕様: コピー×フェイント = 手2つ降ろし + 追加2ターン
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "フェイント")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("コピー(フェイント)+カウンター->手2つ降ろし", gs.player.get_active_hands() == 0)
test("コピー(フェイント)+カウンター->追加2ターン", gs.effects.additional_turns[KEY_PLAYER] == 2)

# 新仕様: コピー×スキップ = 2フェーズスキップ
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "スキップ")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("コピー(スキップ)->2フェーズ封印", gs.computer.skip_phases == 2)

# 新仕様: コピー×ガード = 追加ターンは1回のみ
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "ガード")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("コピー(ガード)->追加1ターンのみ", gs.effects.additional_turns[KEY_PLAYER] == 1)
test("コピー(ガード)->ガードバフ付与", gs.player.guard_active == True)

# 新仕様: コピー×ロック = フラグなので累積しない
gs = make_gs()
gs.effects.record_turn(KEY_COMPUTER, "ロック")
TurnHandler.resolve_turn(gs, KEY_PLAYER, "コピー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, "カウンター")
test("コピー(ロック)->lock_pending=True(累積なし)", gs.computer.lock_pending == True)

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

# ===== オール（新スキル）=====
print("\n=== オールテスト ===")
gs = make_gs()
gs.player.stock = ["フラッシュ", "ガード"]
TurnHandler.resolve_turn(
    gs, KEY_PLAYER, "オール", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, None,
    choice_data={"all_order": ["フラッシュ", "ガード"]}
)
test("オール->ストック消滅", gs.player.stock == [])
test("オール->フラッシュ発動で2手降ろし", gs.player.get_active_hands() == 0)
# ガードも発動するが、もう手が0なので追加ターンだけ取得
test("オール->ガード追加ターン", gs.effects.additional_turns[KEY_PLAYER] == 1)

# ===== チョイス/オール/ドロップの1フェーズ1回制限 =====
print("\n=== ストック+α 1フェーズ1回制限テスト ===")
gs = make_gs()
gs.player.stock = ["ガード"]
TurnHandler.resolve_turn(
    gs, KEY_PLAYER, "チョイス", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None,
    choice_data={"choice": "ガード"}
)
test("チョイス1回目->stock_alpha_used_this_phase=True", gs.player.stock_alpha_used_this_phase == True)
valid = get_valid_skills(gs, KEY_PLAYER)
test("チョイス1回後->チョイス再宣言不可", "チョイス" not in valid)
test("チョイス1回後->オール宣言不可", "オール" not in valid)
test("チョイス1回後->ドロップ宣言不可", "ドロップ" not in valid)

# ===== 勝利前提条件: スキル未宣言の場合勝利しない =====
print("\n=== 勝利前提条件テスト ===")
gs = GameState()
gs.current_player_key = KEY_PLAYER
gs.computer.left_hand = False
gs.computer.right_hand = False  # 相手の手は全部0
# プレイヤーがまだスキル宣言していない状態
gs.player.has_declared_skill = False
gs.computer.has_declared_skill = False
result = gs.check_victory()
test("両者スキル未宣言->勝利不可", result == False and gs.game_over == False)

# 片方だけスキル宣言済み
gs = GameState()
gs.current_player_key = KEY_PLAYER
gs.computer.left_hand = False
gs.computer.right_hand = False
gs.player.has_declared_skill = True
gs.computer.has_declared_skill = False
result = gs.check_victory()
test("片方未宣言->勝利不可", result == False and gs.game_over == False)

# 両方宣言済み
gs = GameState()
gs.current_player_key = KEY_PLAYER
gs.computer.left_hand = False
gs.computer.right_hand = False
gs.player.has_declared_skill = True
gs.computer.has_declared_skill = True
result = gs.check_victory()
test("両者宣言済み->勝利可能", result == True and gs.game_over == True)

# ===== 先手1フェーズ目制限が廃止されているか確認 =====
print("\n=== 先手制限廃止確認テスト ===")
gs = make_gs()
valid = get_valid_skills(gs, KEY_PLAYER)
test("先手1フェーズ目でもフラッシュ宣言可", "フラッシュ" in valid)
test("先手1フェーズ目でもブースト宣言可", "ブースト" in valid)

# ===== リバーシOFF =====
print("\n=== リバーシOFF設定テスト ===")
gs = make_gs()
GAME_CONFIG["ENABLE_REVERSI"] = False
valid = get_valid_skills(gs, KEY_PLAYER)
test("リバーシOFF->リバーシ不可", "リバーシ" not in valid)

# ===== ミラーOFF =====
print("\n=== ミラーOFF設定テスト ===")
gs = make_gs()
GAME_CONFIG["ENABLE_MIRROR"] = False
valid = get_valid_skills(gs, KEY_PLAYER)
test("ミラーOFF->ミラー不可", "ミラー" not in valid)

# ===== ミラーON =====
print("\n=== ミラーON設定テスト ===")
gs = make_gs()
GAME_CONFIG["ENABLE_MIRROR"] = True
valid = get_valid_skills(gs, KEY_PLAYER)
test("ミラーON->ミラー宣言可", "ミラー" in valid)

# ミラー（準備）宣言
gs = make_gs()
TurnHandler.resolve_turn(gs, KEY_PLAYER, "ミラー", {KEY_PLAYER: 0, KEY_COMPUTER: 0}, None)
test("ミラー宣言->mirror_ready=True", gs.player.mirror_ready == True)

# ミラー（メイン）反射（フラッシュを反射）
gs = make_gs()
gs.computer.mirror_ready = True
TurnHandler.resolve_turn(gs, KEY_PLAYER, "フラッシュ", {KEY_PLAYER: 1, KEY_COMPUTER: 1}, "ミラー")
test("ミラー反射->NTPが2手降ろし(勝利)", gs.computer.get_active_hands() == 0)
test("ミラー反射->TPは変化なし", gs.player.get_active_hands() == 2)
test("ミラー反射->mirror_ready消費", gs.computer.mirror_ready == False)

GAME_CONFIG["ENABLE_MIRROR"] = False  # 戻す

# ===== スキップのフェーズ封印動作テスト =====
print("\n=== スキップ封印動作テスト ===")
gs = make_gs()
gs.computer.skip_phases = 1
gs.on_phase_start(KEY_COMPUTER)
test("フェーズ開始時->skip_phases維持", gs.computer.skip_phases == 1)
valid = get_valid_skills(gs, KEY_COMPUTER)
test("フェーズ中->スキル使用不可", valid == [])
gs.on_phase_end(KEY_COMPUTER)
test("フェーズ終了時->skip_phases減少", gs.computer.skip_phases == 0)
gs.on_phase_start(KEY_COMPUTER)
valid = get_valid_skills(gs, KEY_COMPUTER)
test("次フェーズ->スキル使用可能", len(valid) > 0)

# ===== サマリ =====
print(f"\n{'='*40}")
print(f"結果: {passed} passed, {failed} failed, {passed+failed} total")
if failed == 0:
    print("全テスト合格！")
else:
    print(f"!!! {failed} テスト失敗 !!!")

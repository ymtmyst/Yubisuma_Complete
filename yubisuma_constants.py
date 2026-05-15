# yubisuma_constants.py - 完全ルール版

# プレイヤーキー
KEY_PLAYER = "player"
KEY_COMPUTER = "computer"

# 表示用名称
PLAYER_NAMES = {KEY_PLAYER: "プレイヤー", KEY_COMPUTER: "コンピューター"}
PLAYER_TYPES = PLAYER_NAMES  # 互換性

# ゲーム設定
MAX_HANDS = 2
MIN_HANDS = 0

GAME_CONFIG = {
    "ENABLE_REVERSI": False,  # リバーシのオンオフ（デフォルトOFF: テストプレイ段階）
}

# ===== スキル定義 =====
SKILLS = {
    # 通常スキル
    "NUMBER": "数字",
    "FLASH": "フラッシュ",
    "CEMENT": "セメント",
    "GUARD": "ガード",
    "CHARGE": "チャージ",
    "QUICK": "クイック",
    "SKIP": "スキップ",
    # 相手ターン中スキル
    "COUNTER": "カウンター",
    # 対カウンタースキル
    "FEINT": "フェイント",
    "LOCK": "ロック",
    # 参照スキル
    "COPY": "コピー",
    "STOCK": "ストック",
    "CHOICE": "チョイス",
    "DROP": "ドロップ",
    # 必殺スキル
    "BOOST": "ブースト",
    "BLOCK": "ブロック",
    "REVERSI": "リバーシ",
    "TIME": "タイム",
}

# スキルカテゴリ
NORMAL_SKILLS = {"フラッシュ", "セメント", "ガード", "チャージ", "クイック", "スキップ"}
ANTI_COUNTER_SKILLS = {"フェイント", "ロック"}
REFERENCE_SKILLS = {"コピー", "ストック", "チョイス", "ドロップ"}
ULTIMATE_SKILLS = {"ブースト", "ブロック", "リバーシ", "タイム"}
OPPONENT_TURN_SKILLS = {"カウンター", "ブロック"}  # 相手ターン中に宣言するスキル

# 参照スキルが参照可能なカテゴリ（通常スキル＋対カウンタースキル、数字除く）
REFERENCEABLE_SKILLS = (NORMAL_SKILLS | ANTI_COUNTER_SKILLS)

# 一発上がりを発生させうるスキル（ガード判定用）
INSTANT_WIN_SOURCES = {"フラッシュ", "チャージ", "クイック", "コピー"}

# ターンプレイヤーが宣言可能なスキル一覧（数字以外）
TURN_PLAYER_SKILLS = sorted(
    NORMAL_SKILLS | ANTI_COUNTER_SKILLS | REFERENCE_SKILLS |
    (ULTIMATE_SKILLS - OPPONENT_TURN_SKILLS)
)

# メッセージ
MESSAGES = {
    "INVALID_INPUT": "無効な入力です。",
    "NUMBER_REQUIRED": "数字を入力してください。",
    "COUNTER_PROMPT": "相手ターン中スキルを宣言しますか？(k/b/n): ",
    "TURN_ANNOUNCEMENT": "{name}のターンです",
    "VICTORY": "\n{name}の勝利です！",
    "THUMB_PROMPT": "あなたの親指を何本立てますか？(0-{max})",
    "CEMENT_NOTICE": "※セメント効果により{count}本以上の指を立てる必要があります",
    "CEMENT_ERROR": "セメント効果により{count}本以上の指を立てる必要があります！",
    "SKIP_NOTICE": "※スキップ効果により全てのスキル(数字含む)が宣言できません",
    "FIRST_PHASE_NOTICE": "※先手1フェーズ目のためフラッシュ・必殺スキルは使用不可",
    "LOCK_NOTICE": "※ロック効果によりカウンターを宣言できません",
    "ULTIMATE_USED": "※必殺スキルは既に使用済みです",
}

# ゲーム説明文
GAME_RULES = """===== 指スマ 完全ルール版 =====
【通常スキル】
  数字      : 合計と一致で手を1つ降ろす
  フラッシュ  : 互いの指が同数で一発上がり
  セメント   : 上がった指を下げられなくする（永続）
  ガード    : 一発上がりを1回無効化 + 追加1ターン
  チャージ   : 次の数字宣言で一発上がり
  クイック   : 次ターンで再宣言→一発上がり / 次々ターン→手を1つ降ろす
  スキップ   : 相手の次フェーズのスキル封印（連鎖で2フェーズ）

【対カウンタースキル】（カウンターされた時に発動）
  フェイント  : 手を1つ降ろし追加1ターン
  ロック    : 次ターン中、相手のカウンターを封じる

【参照スキル】（1ターン前のスキルを参照）
  コピー    : 前ターンのスキル効果を発動（手降ろし→一発上がり）
  ストック   : 前ターンのスキルを保存
  チョイス   : ストックから選んで効果発動（カウンター確認後に選択可）
  ドロップ   : ストック内スキルを相手に使用不可 + 追加1ターン

【必殺スキル】（ゲーム中1回）
  ブースト   : 追加3ターン
  ブロック   : 相手のスキル効果を無効化（スキップ除く）※相手ターン中
  リバーシ   : 互いの状態を入れ替え（ストック/スキップ/タイム/ドロップ除外）
  タイム    : 相手が追加ターンを得た時に代わりに自分のターンになる + 追加1ターン（発動まで永続）

※先手は開幕1フェーズ目にフラッシュ・必殺スキルを使用不可
=================="""

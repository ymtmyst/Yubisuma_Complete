# rl/config.py - 設定定数・ハイパーパラメータ
"""
指スマ完全ルール版 PPO強化学習の設定。
RTX 4070 Ti (1枚) を想定。
"""

import os

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "rl_models")
LOG_DIR = os.path.join(BASE_DIR, "rl_logs")
ANALYSIS_DIR = os.path.join(BASE_DIR, "rl_analysis")
LEAGUE_DIR = os.path.join(MODEL_DIR, "league")
DB_PATH = os.path.join(BASE_DIR, "rl_analysis.db")
SNAPSHOT_PATH = os.path.join(BASE_DIR, "skill_snapshot.json")  # スキル分布スナップショット

# === 行動空間 ===
# TP (ターンプレイヤー) の行動
# スキル/数字の選択肢 (リバーシ除外、チョイスは展開)
TP_SKILL_OPTIONS = [
    # 数字 (0-4)
    0, 1, 2, 3, 4,
    # 通常スキル
    "フラッシュ", "セメント", "ガード", "チャージ", "クイック", "スキップ",
    # 対カウンタースキル
    "フェイント", "ロック",
    # 参照スキル
    "コピー", "ストック", "ドロップ",
    # 必殺スキル (ブロック/リバーシ除外)
    "ブースト", "タイム",
    # チョイス展開 (ストック可能な8スキルそれぞれ)
    "チョイス:フラッシュ", "チョイス:セメント", "チョイス:ガード",
    "チョイス:チャージ", "チョイス:クイック", "チョイス:スキップ",
    "チョイス:フェイント", "チョイス:ロック",
]
NUM_TP_SKILLS = len(TP_SKILL_OPTIONS)  # 26
NUM_THUMB_OPTIONS = 3  # 0, 1, 2
NUM_TP_ACTIONS = NUM_TP_SKILLS * NUM_THUMB_OPTIONS  # 78

# NTP (非ターンプレイヤー) の行動
NTP_REACTION_OPTIONS = ["なし", "カウンター", "ブロック"]
NUM_NTP_REACTIONS = len(NTP_REACTION_OPTIONS)  # 3
NUM_NTP_ACTIONS = NUM_NTP_REACTIONS * NUM_THUMB_OPTIONS  # 9

# 全行動空間
TOTAL_ACTIONS = NUM_TP_ACTIONS + NUM_NTP_ACTIONS  # 87

# ストック可能スキル一覧 (チョイス展開用)
STOCKABLE_SKILLS = [
    "フラッシュ", "セメント", "ガード", "チャージ",
    "クイック", "スキップ", "フェイント", "ロック",
]
NUM_STOCKABLE = len(STOCKABLE_SKILLS)  # 8

# === 観測空間 ===
# 自分の状態
OBS_SELF_BASE = 11        # hands(3) + buffs/debuffs(8)  ※cement_flagは冗長なので除外
OBS_SELF_STOCK = 8        # ストック内容 (one-hot)
OBS_SELF_CHOICE_USED = 8  # フェーズ内チョイス使用済み
OBS_SELF_DROP_BLOCKED = 8 # ドロップ封印
OBS_SELF_TOTAL = OBS_SELF_BASE + OBS_SELF_STOCK + OBS_SELF_CHOICE_USED + OBS_SELF_DROP_BLOCKED  # 35

# 相手の可視状態 (ストックは公開情報のため内容を完全開示)
OBS_OPP_BASE = 12          # hands(3) + buffs/debuffs(8) + stock_count(1)  ※cement_flagは冗長なので除外
OBS_OPP_DROP_BLOCKED = 8   # ドロップ封印
OBS_OPP_STOCK = 8          # ストック内容 (one-hot、公開情報)
OBS_OPP_CHOICE_USED = 8    # フェーズ内チョイス使用済み (公開情報)
OBS_OPP_TOTAL = OBS_OPP_BASE + OBS_OPP_DROP_BLOCKED + OBS_OPP_STOCK + OBS_OPP_CHOICE_USED  # 36

# グローバル状態
OBS_IS_TP = 1
OBS_FIRST_RESTRICTED = 2  # me + opp
OBS_PREV_SKILL = 20       # one-hot: 5数字 + 14スキル + None
OBS_EXTRA_TURNS = 2       # me + opp (各1)
OBS_SKIP_CHAIN = 2        # me + opp
OBS_PHASE_TURN = 1        # フェーズ内ターン数 (実務上8以下)
OBS_TURN_COUNT = 1        # 試合全体のターン進行度

# エンコーディング上限値
SKIP_PHASES_MAX = 20      # スキップ連鎖は理論上無制限なので十分大きく設定
PHASE_TURNS_MAX = 8       # 実務上8以下
OBS_GLOBAL_TOTAL = (OBS_IS_TP + OBS_FIRST_RESTRICTED + OBS_PREV_SKILL +
                    OBS_EXTRA_TURNS + OBS_SKIP_CHAIN + OBS_PHASE_TURN + OBS_TURN_COUNT)  # 29

# 合計観測次元
OBS_TOTAL = OBS_SELF_TOTAL + OBS_OPP_TOTAL + OBS_GLOBAL_TOTAL  # 100

# === PPOハイパーパラメータ (RTX 4070 Ti 向け) ===
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.03,         # 混合戦略を学習するため探索を促進 (序盤の希少スキル探索を確保)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
}

# ネットワーク構造
NETWORK_CONFIG = {
    "feature_dim": 256,       # 特徴抽出層の次元
    "policy_layers": [256, 128],
    "value_layers": [256, 128],
    "aux_layers": [128, 64],  # 補助予測ヘッド
}

# === 補助タスク ===
AUX_LOSS_WEIGHT = 0.3         # 補助損失の重み
AUX_REACTION_WEIGHT = 0.4     # リアクション予測の重み
AUX_THUMBS_WEIGHT = 0.3       # 指の本数予測の重み
AUX_SKILL_WEIGHT = 0.3        # スキル予測の重み
AUX_LOOKAHEAD_WEIGHT = 0.2    # 終盤勝敗予測ヘッドの重み (終盤局面の価値表現を補強)
LOOKAHEAD_N = 3               # エピソード末尾から何ステップ分を勝敗予測に使うか

# === 訓練設定 ===
TOTAL_TIMESTEPS = 5_000_000
CHECKPOINT_FREQ = 50_000      # チェックポイント保存頻度
OPPONENT_UPDATE_FREQ = 10_000  # 対戦相手更新頻度
EVAL_FREQ = 20_000            # 評価頻度
EVAL_EPISODES = 100            # 評価エピソード数

# === リーグ設定 ===
LEAGUE_CONFIG = {
    "max_opponents": 30,       # リーグ内最大対戦相手数
    "recent_weight": 0.7,      # 最新チェックポイントの選択確率
    "random_weight": 0.2,      # ランダム対戦相手の選択確率
    "initial_weight": 0.1,     # 初期（弱い）対戦相手の選択確率
    "win_rate_threshold": 0.6, # 新規対戦相手追加の閾値
}

# === 報酬設定 ===
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.0  # タイムアウト時
MAX_TURNS = 200     # 1エピソードの最大ターン数

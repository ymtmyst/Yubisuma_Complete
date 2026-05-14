# rl/observation.py - 観測空間エンコーディング
"""
ゲーム状態を固定長の観測ベクトルに変換する。
完全公開情報: 相手のストック内容を含む全状態を観測に含める (マルコフゲーム理論的に正しい設計)。
観測次元: 自分 35 + 相手 36 + グローバル 29 = 100次元
"""

import numpy as np
from rl.config import (
    OBS_TOTAL, STOCKABLE_SKILLS, TP_SKILL_OPTIONS, MAX_TURNS,
    SKIP_PHASES_MAX, PHASE_TURNS_MAX,
)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yubisuma_constants import (
    KEY_PLAYER, KEY_COMPUTER, NORMAL_SKILLS, ANTI_COUNTER_SKILLS,
)


LEGACY_OBS_TOTAL_V86 = 86


# 前ターンスキルのone-hotインデックス
# 0-4: 数字, 5-17: スキル名, 18: None
PREV_SKILL_LIST = [
    0, 1, 2, 3, 4,
    "フラッシュ", "セメント", "ガード", "チャージ", "クイック", "スキップ",
    "フェイント", "ロック",
    "コピー", "ストック", "チョイス", "ドロップ",
    "ブースト", "タイム",
]
PREV_SKILL_NONE_IDX = len(PREV_SKILL_LIST)  # 19
PREV_SKILL_DIM = PREV_SKILL_NONE_IDX + 1    # 20

# ストック可能スキルのインデックス
STOCKABLE_SKILL_IDX = {s: i for i, s in enumerate(STOCKABLE_SKILLS)}

# ドロップ封印対象はTP_SKILL_OPTIONSの文字列スキル
DROP_BLOCKABLE_SKILLS = STOCKABLE_SKILLS  # 同じ8種


def encode_observation(game_state, agent_key, turn_count=0):
    """
    ゲーム状態を観測ベクトルに変換。

    Args:
        game_state: GameStateオブジェクト
        agent_key: エージェントのプレイヤーキー
        turn_count: 現在のターン数 (試合全体の進行度として使用)

    Returns:
        np.ndarray: 観測ベクトル (shape: (OBS_TOTAL,), dtype: float32)
    """
    me = game_state.get_player(agent_key)
    opp_key = game_state.get_opponent_key(agent_key)
    opp = game_state.get_opponent(agent_key)
    effects = game_state.effects

    obs = []

    # === 自分の状態 (36次元) ===
    obs.extend(_encode_self_state(me, effects, agent_key))

    # === 相手の可視状態 (21次元) ===
    obs.extend(_encode_opponent_state(opp, effects, opp_key))

    # === グローバル状態 ===
    obs.extend(_encode_global_state(game_state, effects, agent_key, opp_key, turn_count))
    
    result = np.array(obs, dtype=np.float32)
    assert result.shape == (OBS_TOTAL,), f"Observation dim mismatch: {result.shape} != ({OBS_TOTAL},)"
    return result


def encode_observation_for_dim(game_state, agent_key, turn_count=0, obs_dim=OBS_TOTAL):
    """Encode observation for the expected model input width."""
    if obs_dim == OBS_TOTAL:
        return encode_observation(game_state, agent_key, turn_count)
    if obs_dim == LEGACY_OBS_TOTAL_V86:
        return _encode_observation_legacy_v86(game_state, agent_key, turn_count)
    raise ValueError(f"Unsupported observation dim: {obs_dim}")


def _encode_observation_legacy_v86(game_state, agent_key, turn_count=0):
    """
    Legacy 86-dim observation used by older checkpoints.

    Layout:
      self 36 = base12 + stock8 + choice8 + drop8
      opp  21 = base13 + drop8
      global 29 = unchanged
    """
    me = game_state.get_player(agent_key)
    opp_key = game_state.get_opponent_key(agent_key)
    opp = game_state.get_opponent(agent_key)
    effects = game_state.effects

    obs = []
    obs.extend(_encode_self_state_legacy_v86(me))
    obs.extend(_encode_opponent_state_legacy_v86(opp))
    obs.extend(_encode_global_state(game_state, effects, agent_key, opp_key, turn_count))

    result = np.array(obs, dtype=np.float32)
    assert result.shape == (LEGACY_OBS_TOTAL_V86,), (
        f"Legacy observation dim mismatch: {result.shape} != ({LEGACY_OBS_TOTAL_V86},)"
    )
    return result


def _encode_self_state(player, effects, player_key):
    """自分の完全な状態をエンコード (35次元)"""
    obs = []

    # 基本状態 (11次元)
    # cement: None=0.0, 1=0.5, 2=1.0 (通常プレイ中にcement=0は発生しない)
    obs.append(player.get_active_hands() / 2.0)
    obs.append(float(player.left_hand))
    obs.append(float(player.right_hand))
    obs.append(float(player.guard_active))
    obs.append(float(player.charge_active))
    obs.append(player.quick_level / 2.0)
    obs.append((player.cement or 0) / 2.0)
    obs.append(min(player.lock_debuff, 2) / 2.0)
    obs.append(min(player.skip_phases, SKIP_PHASES_MAX) / SKIP_PHASES_MAX)
    obs.append(float(player.used_ultimate))
    obs.append(float(player.time_active))
    
    # ストック内容 (8次元) - 各スキルの所持数を最大ストック数8で正規化→[0,1]に収める
    stock_vec = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.stock:
        if s in STOCKABLE_SKILL_IDX:
            stock_vec[STOCKABLE_SKILL_IDX[s]] += 1.0
    obs.extend([min(v, 8.0) / 8.0 for v in stock_vec])
    
    # チョイス使用済み (8次元)
    choice_used = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.choice_used_this_phase:
        if s in STOCKABLE_SKILL_IDX:
            choice_used[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(choice_used)
    
    # ドロップ封印 (8次元)
    drop_blocked = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.drop_blocked_skills:
        if s in STOCKABLE_SKILL_IDX:
            drop_blocked[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(drop_blocked)
    
    return obs  # 11 + 8 + 8 + 8 = 35


def _encode_self_state_legacy_v86(player):
    """Legacy self-state encoder for old 86-dim checkpoints."""
    obs = []
    obs.append(player.get_active_hands() / 2.0)
    obs.append(float(player.left_hand))
    obs.append(float(player.right_hand))
    obs.append(float(player.guard_active))
    obs.append(float(player.charge_active))
    obs.append(player.quick_level / 2.0)
    obs.append(float(player.cement is not None))
    obs.append((player.cement or 0) / 2.0)
    obs.append(min(player.lock_debuff, 2) / 2.0)
    obs.append(min(player.skip_phases, 2) / 2.0)
    obs.append(float(player.used_ultimate))
    obs.append(float(player.time_active))

    stock_vec = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.stock:
        if s in STOCKABLE_SKILL_IDX:
            stock_vec[STOCKABLE_SKILL_IDX[s]] += 1.0
    obs.extend([min(v, 8.0) / 8.0 for v in stock_vec])

    choice_used = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.choice_used_this_phase:
        if s in STOCKABLE_SKILL_IDX:
            choice_used[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(choice_used)

    drop_blocked = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.drop_blocked_skills:
        if s in STOCKABLE_SKILL_IDX:
            drop_blocked[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(drop_blocked)
    return obs


def _encode_opponent_state(player, effects, player_key):
    """相手の可視状態をエンコード (36次元) - ストックは公開情報のため内容を完全開示"""
    obs = []

    # 基本状態 (12次元)
    # cement: None=0.0, 1=0.5, 2=1.0 (通常プレイ中にcement=0は発生しない)
    obs.append(player.get_active_hands() / 2.0)
    obs.append(float(player.left_hand))
    obs.append(float(player.right_hand))
    obs.append(float(player.guard_active))
    obs.append(float(player.charge_active))
    obs.append(player.quick_level / 2.0)
    obs.append((player.cement or 0) / 2.0)
    obs.append(min(player.lock_debuff, 2) / 2.0)
    obs.append(min(player.skip_phases, SKIP_PHASES_MAX) / SKIP_PHASES_MAX)
    obs.append(float(player.used_ultimate))
    obs.append(float(player.time_active))
    obs.append(min(len(player.stock), 8) / 8.0)  # ストック件数

    # ドロップ封印 (8次元) - 公開情報
    drop_blocked = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.drop_blocked_skills:
        if s in STOCKABLE_SKILL_IDX:
            drop_blocked[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(drop_blocked)

    # ストック内容 (8次元) - 公開情報（完全ルール準拠）
    stock_vec = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.stock:
        if s in STOCKABLE_SKILL_IDX:
            stock_vec[STOCKABLE_SKILL_IDX[s]] += 1.0
    obs.extend([min(v, 8.0) / 8.0 for v in stock_vec])

    # チョイス使用済み (8次元) - 公開情報
    choice_used = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.choice_used_this_phase:
        if s in STOCKABLE_SKILL_IDX:
            choice_used[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(choice_used)

    return obs  # 12 + 8 + 8 + 8 = 36


def _encode_opponent_state_legacy_v86(player):
    """Legacy opponent-state encoder for old 86-dim checkpoints."""
    obs = []
    obs.append(player.get_active_hands() / 2.0)
    obs.append(float(player.left_hand))
    obs.append(float(player.right_hand))
    obs.append(float(player.guard_active))
    obs.append(float(player.charge_active))
    obs.append(player.quick_level / 2.0)
    obs.append(float(player.cement is not None))
    obs.append((player.cement or 0) / 2.0)
    obs.append(min(player.lock_debuff, 2) / 2.0)
    obs.append(min(player.skip_phases, 2) / 2.0)
    obs.append(float(player.used_ultimate))
    obs.append(float(player.time_active))
    obs.append(min(len(player.stock), 8) / 8.0)

    drop_blocked = [0.0] * len(STOCKABLE_SKILLS)
    for s in player.drop_blocked_skills:
        if s in STOCKABLE_SKILL_IDX:
            drop_blocked[STOCKABLE_SKILL_IDX[s]] = 1.0
    obs.extend(drop_blocked)
    return obs


def _encode_global_state(game_state, effects, agent_key, opp_key, turn_count=0):
    """グローバル状態をエンコード (29次元)"""
    obs = []

    # TPフラグ (1次元)
    is_tp = float(game_state.current_player_key == agent_key)
    obs.append(is_tp)

    # 先手制限 (2次元)
    obs.append(float(effects.is_first_phase_restricted(agent_key)))
    obs.append(float(effects.is_first_phase_restricted(opp_key)))

    # 前ターンスキル (20次元) - one-hot
    prev_skill_vec = [0.0] * PREV_SKILL_DIM
    prev_skill = effects.get_previous_turn_skill()
    if prev_skill is None:
        prev_skill_vec[PREV_SKILL_NONE_IDX] = 1.0
    else:
        for i, s in enumerate(PREV_SKILL_LIST):
            if s == prev_skill:
                prev_skill_vec[i] = 1.0
                break
    obs.extend(prev_skill_vec)

    # 追加ターン (2次元)
    obs.append(min(effects.additional_turns.get(agent_key, 0), 4) / 4.0)
    obs.append(min(effects.additional_turns.get(opp_key, 0), 4) / 4.0)

    # スキップ連鎖 (2次元)
    obs.append(float(effects.last_turn_was_skip.get(agent_key, False)))
    obs.append(float(effects.last_turn_was_skip.get(opp_key, False)))

    # フェーズ内ターン数 (1次元) - 実務上8以下
    obs.append(min(effects.turns_in_current_phase, PHASE_TURNS_MAX) / PHASE_TURNS_MAX)

    # 試合全体のターン進行度 (1次元) - セメント等のタイミング依存スキルの文脈を与える
    obs.append(min(turn_count, MAX_TURNS) / MAX_TURNS)

    return obs  # 1 + 2 + 20 + 2 + 2 + 1 + 1 = 29

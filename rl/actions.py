# rl/actions.py - 行動空間エンコーディング・デコーディング
"""
離散行動空間の設計:
  TP行動 = skill_idx * 3 + thumb_idx
  NTP行動 = NUM_TP_ACTIONS + reaction_idx * 3 + thumb_idx

MaskablePPOで状態に応じた動的マスキングを行う。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.config import (
    TP_SKILL_OPTIONS, NTP_REACTION_OPTIONS,
    NUM_TP_SKILLS, NUM_THUMB_OPTIONS, NUM_TP_ACTIONS,
    NUM_NTP_REACTIONS, NUM_NTP_ACTIONS, TOTAL_ACTIONS,
    STOCKABLE_SKILLS,
)
from yubisuma_constants import (
    NORMAL_SKILLS, ANTI_COUNTER_SKILLS, REFERENCE_SKILLS,
    ULTIMATE_SKILLS, OPPONENT_TURN_SKILLS, REFERENCEABLE_SKILLS,
    GAME_CONFIG, STOCK_ALPHA_SKILLS,
)


# === エンコード ===

CHOICE_SKILL = "チョイス"
CHOICE_PREFIX = CHOICE_SKILL + ":"


def encode_tp_action(skill_idx, thumb_count):
    """TPの行動をインデックスにエンコード"""
    return skill_idx * NUM_THUMB_OPTIONS + thumb_count


def encode_ntp_action(reaction_idx, thumb_count):
    """NTPの行動をインデックスにエンコード"""
    return NUM_TP_ACTIONS + reaction_idx * NUM_THUMB_OPTIONS + thumb_count


# === デコード ===

def decode_action(action_idx):
    """
    行動インデックスをデコード。
    
    Returns:
        dict: {
            'role': 'tp' or 'ntp',
            'skill': int/str (TP) or None (NTP),
            'reaction': str or None,
            'thumbs': int,
            'choice_target': str or None (チョイス展開時)
        }
    """
    if action_idx < NUM_TP_ACTIONS:
        # TP行動
        skill_idx = action_idx // NUM_THUMB_OPTIONS
        thumb_count = action_idx % NUM_THUMB_OPTIONS
        skill = TP_SKILL_OPTIONS[skill_idx]
        
        # チョイス展開の処理
        choice_target = None
        if isinstance(skill, str) and skill.startswith("チョイス:"):
            choice_target = skill.split(":")[1]
            skill = "チョイス"
        
        return {
            'role': 'tp',
            'skill': skill,
            'reaction': None,
            'thumbs': thumb_count,
            'choice_target': choice_target,
        }
    else:
        # NTP行動
        ntp_idx = action_idx - NUM_TP_ACTIONS
        reaction_idx = ntp_idx // NUM_THUMB_OPTIONS
        thumb_count = ntp_idx % NUM_THUMB_OPTIONS
        reaction_str = NTP_REACTION_OPTIONS[reaction_idx]
        reaction = reaction_str if reaction_str != "なし" else None
        
        return {
            'role': 'ntp',
            'skill': None,
            'reaction': reaction,
            'thumbs': thumb_count,
            'choice_target': None,
        }


# === マスク生成 ===

def get_action_mask(game_state, agent_key):
    """
    現在の状態に基づいて有効な行動のマスクを生成。
    
    Returns:
        np.ndarray: shape (TOTAL_ACTIONS,), dtype bool. True=有効
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    
    is_tp = (game_state.current_player_key == agent_key)
    me = game_state.get_player(agent_key)
    opp = game_state.get_opponent(agent_key)
    effects = game_state.effects

    pending_choice = getattr(game_state, "pending_choice", None)
    if pending_choice and pending_choice.get("chooser_key") == agent_key:
        _mask_choice_target_actions(mask, me)
        return mask
    
    if is_tp:
        _mask_tp_actions(mask, game_state, me, opp, agent_key, effects)
    else:
        _mask_ntp_actions(mask, game_state, me, opp, agent_key, effects)
    
    return mask


def _mask_tp_actions(mask, game_state, me, opp, agent_key, effects):
    """TP行動のマスクを設定"""
    max_hands = me.get_active_hands()
    
    # セメント制限
    cement_min = me.cement if me.cement is not None else 0
    
    # スキップ中は全TP行動を禁止（数字のみ許可…ではなく全禁止）
    # ルール: スキップ中はスキルを宣言できない → 数字も含めて行動不可
    # → 実際にはスキップ中のターンは自動スキップされるので、ここには到達しない
    if me.skip_phases > 0:
        # スキップ中でもTPとしてstepする場合の安全策
        # 数字0を指0本で宣言（ダミー行動）
        mask[encode_tp_action(0, 0)] = True
        return
    
    # 先手制限は新ルールで廃止
    first_restricted = False

    # 有効なスキル一覧を取得
    valid_skills = set()
    
    for idx, skill in enumerate(TP_SKILL_OPTIONS):
        skill_name = skill if isinstance(skill, str) else None
        
        # 数字 (0-4)
        if isinstance(skill, int):
            valid_skills.add(idx)
            continue
        
        # チョイス展開
        if skill_name.startswith(CHOICE_PREFIX):
            if me.stock_alpha_used_this_phase:
                continue
            target = skill_name.split(":")[1]
            # チョイスが宣言可能 + 対象がストックにある + フェーズ内未使用
            available = [s for s in me.stock if s not in me.choice_used_this_phase]
            if available and target == available[0]:
                # ドロップ封印チェック
                if CHOICE_SKILL not in me.drop_blocked_skills:
                    valid_skills.add(idx)
            continue
        
        # 通常スキルのチェック
        if not _is_skill_valid(skill_name, me, opp, effects, agent_key, first_restricted):
            continue
        
        valid_skills.add(idx)
    
    # マスク設定: 有効なスキル × 有効な指の本数
    for skill_idx in valid_skills:
        for thumb in range(NUM_THUMB_OPTIONS):
            if thumb > max_hands:
                continue
            if me.cement is not None and thumb < me.cement:
                continue
            mask[encode_tp_action(skill_idx, thumb)] = True


def _is_skill_valid(skill_name, me, opp, effects, agent_key, first_restricted):
    """個別スキルの宣言可能性を判定（新ルール: 先手制限廃止）"""
    # チョイス/オール/ドロップはいずれか1フェーズ1回のみ
    if skill_name in STOCK_ALPHA_SKILLS and me.stock_alpha_used_this_phase:
        return False

    # 必殺スキル使用済み
    if skill_name in ULTIMATE_SKILLS and me.used_ultimate:
        return False

    # リバーシ無効
    if skill_name == "リバーシ" and not GAME_CONFIG["ENABLE_REVERSI"]:
        return False

    # ミラー無効
    if skill_name == "ミラー" and not GAME_CONFIG.get("ENABLE_MIRROR", False):
        return False

    # ブロック/カウンターはTP用ではない
    if skill_name in ("ブロック", "カウンター", "ミラー（メイン）"):
        return False
    
    # ドロップ封印
    if skill_name in me.drop_blocked_skills:
        return False
    
    # 参照スキル (コピー/ストック): 前ターンに参照可能なスキルがあるか
    if skill_name in ("コピー", "ストック"):
        prev = effects.get_previous_turn_skill()
        if prev is None or (isinstance(prev, str) and prev not in REFERENCEABLE_SKILLS):
            return False
        if skill_name == "ストック" and isinstance(prev, int):
            return False
    
    # チョイス本体は展開済みなのでスキップ
    if skill_name == "チョイス":
        return False
    
    # ドロップ: ストックが空でないか
    if skill_name == "ドロップ":
        if not me.stock:
            return False

    # オール: ストックが空でないか
    if skill_name == "オール":
        if not me.stock:
            return False
    
    return True


def _mask_choice_target_actions(mask, player):
    """Mask target-selection actions for an already declared Choice."""
    available = [s for s in player.stock if s not in player.choice_used_this_phase]
    for idx, skill in enumerate(TP_SKILL_OPTIONS):
        if not isinstance(skill, str) or not skill.startswith(CHOICE_PREFIX):
            continue
        target = skill.split(":")[1]
        if target in available:
            # Thumb count was fixed when Choice was declared.
            mask[encode_tp_action(idx, 0)] = True


def _mask_ntp_actions(mask, game_state, me, opp, agent_key, effects):
    """NTP行動のマスクを設定"""
    max_hands = me.get_active_hands()
    cement_min = me.cement if me.cement is not None else 0
    
    # 新ルール: lock_active (フラグ方式) を使用
    lock_blocked = me.lock_active
    first_restricted = False  # 先手制限は新ルールで廃止
    mirror_available = (
        GAME_CONFIG.get("ENABLE_MIRROR", False)
        and me.mirror_ready
        and not lock_blocked
    )

    for react_idx, reaction in enumerate(NTP_REACTION_OPTIONS):
        # リアクションの有効性チェック
        if reaction == "カウンター" and lock_blocked:
            continue
        if reaction == "ブロック":
            if me.used_ultimate:
                continue
            if first_restricted:
                continue
        if reaction == "ミラー" and not mirror_available:
            continue
        
        # 指の本数
        for thumb in range(NUM_THUMB_OPTIONS):
            if thumb > max_hands:
                continue
            if me.cement is not None and thumb < me.cement:
                continue
            mask[encode_ntp_action(react_idx, thumb)] = True


# === ユーティリティ ===

def action_to_readable(action_idx):
    """行動を人間が読める文字列に変換"""
    decoded = decode_action(action_idx)
    if decoded['role'] == 'tp':
        skill_str = str(decoded['skill'])
        if decoded['choice_target']:
            skill_str = f"チョイス({decoded['choice_target']})"
        return f"TP: {skill_str} / 指{decoded['thumbs']}本"
    else:
        reaction_str = decoded['reaction'] or "なし"
        return f"NTP: {reaction_str} / 指{decoded['thumbs']}本"

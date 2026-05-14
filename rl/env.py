# rl/env.py - Gymnasium環境
"""
指スマ完全ルール版のGymnasium互換環境。
同時意思決定をSelf-Play形式で処理する。

各step()は1ゲームターンに対応:
- エージェントがTP時: エージェントがスキル+指を選択、対戦相手がNTP行動を内部生成
- エージェントがNTP時: 対戦相手がTP行動を内部生成、エージェントがリアクション+指を選択
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.config import (
    TOTAL_ACTIONS, OBS_TOTAL, REWARD_WIN, REWARD_LOSE,
    REWARD_DRAW, MAX_TURNS,
)
from rl.observation import encode_observation, encode_observation_for_dim
from rl.model_utils import load_maskable_ppo
from rl.actions import (
    decode_action, get_action_mask, encode_tp_action, encode_ntp_action,
    action_to_readable, NUM_TP_ACTIONS,
)
from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER
from yubisuma_base import Player, count_total_thumbs
from yubisuma_effects import EffectManager
from yubisuma_turn_handler import TurnHandler


class YubisumaEnv(gym.Env):
    """指スマ完全ルール版のRL環境"""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, opponent_policy=None, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_TOTAL,), dtype=np.float32
        )

        self.opponent_policy = opponent_policy
        self.render_mode = render_mode

        # ゲーム状態
        self.game_state = None
        self.agent_key = None
        self.opponent_key = None
        self.turn_count = 0

        # エピソード記録
        self.episode_turns = []

        # 最新の相手行動（補助タスク用）
        self.last_opponent_action = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = _create_game_state()
        self.turn_count = 0
        self.episode_turns = []
        self.last_opponent_action = None
        
        # エージェントのプレイヤー割り当て（ランダム）
        if self.np_random.random() < 0.5:
            self.agent_key = KEY_PLAYER
            self.opponent_key = KEY_COMPUTER
        else:
            self.agent_key = KEY_COMPUTER
            self.opponent_key = KEY_PLAYER
        
        # 先手をランダムに決定 (np_randomを使いGymシードを有効化)
        keys = [KEY_PLAYER, KEY_COMPUTER]
        self.game_state.current_player_key = keys[self.np_random.integers(2)]
        self.game_state.effects.first_player_key = self.game_state.current_player_key
        
        # フェーズ開始処理
        self._on_phase_start(self.game_state.current_player_key)
        
        # 相手が先手でスキップ中なら自動処理
        self._auto_advance()
        
        obs = encode_observation(self.game_state, self.agent_key, self.turn_count)
        info = self._get_info()

        return obs, info
    
    def step(self, action):
        decoded = decode_action(action)
        is_agent_tp = (self.game_state.current_player_key == self.agent_key)
        
        # エージェントの行動を解釈
        if is_agent_tp:
            agent_tp_action = decoded
            # 対戦相手のNTP行動を生成
            opp_ntp_action = self._get_opponent_ntp_action()
            self.last_opponent_action = opp_ntp_action
        else:
            agent_ntp_action = decoded
            # 対戦相手のTP行動を生成
            opp_tp_action = self._get_opponent_tp_action()
            self.last_opponent_action = opp_tp_action
        
        # ターン解決
        if is_agent_tp:
            tp_key = self.agent_key
            skill = agent_tp_action['skill']
            choice_target = agent_tp_action['choice_target']
            tp_thumbs = agent_tp_action['thumbs']
            reaction = opp_ntp_action['reaction']
            ntp_thumbs = opp_ntp_action['thumbs']
        else:
            tp_key = self.opponent_key
            skill = opp_tp_action['skill']
            choice_target = opp_tp_action.get('choice_target')
            tp_thumbs = opp_tp_action['thumbs']
            reaction = agent_ntp_action['reaction']
            ntp_thumbs = agent_ntp_action['thumbs']
        
        # 指の本数を構築
        if tp_key == KEY_PLAYER:
            thumbs = {KEY_PLAYER: tp_thumbs, KEY_COMPUTER: ntp_thumbs}
        else:
            thumbs = {KEY_PLAYER: ntp_thumbs, KEY_COMPUTER: tp_thumbs}
        
        # チョイスの特別処理: 選択対象を事前に設定
        if choice_target:
            tp = self.game_state.get_player(tp_key)
            tp._pending_choice_target = choice_target
        
        # ターン記録
        turn_record = {
            'turn': self.turn_count,
            'tp_key': tp_key,
            'skill': skill,
            'choice_target': choice_target,
            'thumbs': thumbs.copy(),
            'reaction': reaction,
            'agent_is_tp': is_agent_tp,
        }
        
        # ターン解決（内部print抑制）
        self._resolve_turn_silent(tp_key, skill, thumbs, reaction, choice_target)
        
        self.turn_count += 1
        self.episode_turns.append(turn_record)
        
        # 勝利判定
        terminated = self._check_victory()
        
        if not terminated:
            # ゲーム進行: フェーズ遷移、追加ターン、タイム効果を処理
            self._advance_game_state(tp_key)
            # スキップターン等を自動処理
            self._auto_advance()
        
        # タイムアウト判定
        truncated = False
        if not terminated and self.turn_count >= MAX_TURNS:
            truncated = True
        
        # 報酬
        reward = self._compute_reward(terminated, truncated)
        
        obs = encode_observation(self.game_state, self.agent_key, self.turn_count)
        info = self._get_info()

        # エピソード完了時にサマリをinfoに含める (AnalysisCallbackが主プロセスで収集)
        if terminated or truncated:
            info['episode_summary'] = self._get_episode_summary()

        return obs, reward, terminated, truncated, info
    
    def action_masks(self):
        """MaskablePPO用のアクションマスク"""
        return get_action_mask(self.game_state, self.agent_key)
    
    # === 内部メソッド ===
    
    def _resolve_turn_silent(self, tp_key, skill, thumbs, reaction, choice_target=None):
        """ターンを解決（stdout抑制）"""
        import io
        from contextlib import redirect_stdout
        
        # チョイスの選択を注入
        if choice_target:
            tp = self.game_state.get_player(tp_key)
            tp._pending_choice_target = choice_target
            
            # チョイスの選択をオーバーライドするためのモンキーパッチ
            original_resolve = TurnHandler._resolve_choice_with_reaction
            
            @staticmethod
            def patched_choice(gs, tp_key_inner, thumbs_inner, total, reaction_inner):
                tp_inner = gs.get_player(tp_key_inner)
                ntp = gs.get_opponent(tp_key_inner)
                effects = gs.effects
                
                chosen = getattr(tp_inner, '_pending_choice_target', None)
                if chosen is None:
                    # フォールバック
                    available = [s for s in tp_inner.stock if s not in tp_inner.choice_used_this_phase]
                    chosen = available[0] if available else None

                if reaction_inner == "ブロック":
                    ntp.used_ultimate = True
                    # スキップはブロック無効: スキップ以外のみ早期return
                    if chosen != "スキップ":
                        return

                if chosen is None:
                    return

                tp_inner.choice_used_this_phase.add(chosen)
                
                from yubisuma_constants import ANTI_COUNTER_SKILLS
                was_skip_before = effects.last_turn_was_skip[tp_key_inner]
                if chosen == "スキップ":
                    effects.last_turn_was_skip[tp_key_inner] = True
                
                if chosen in ANTI_COUNTER_SKILLS and reaction_inner == "カウンター":
                    TurnHandler._resolve_anti_counter(gs, tp_key_inner, chosen)
                elif reaction_inner == "カウンター":
                    if chosen == "フラッシュ":
                        tp_thumbs = thumbs_inner[tp_inner.key]
                        ntp_thumbs = thumbs_inner[ntp.key]
                        if tp_thumbs == ntp_thumbs:
                            if not gs.effects.try_block_instant_win(tp_inner):
                                ntp.remove_all_hands()
                    # その他のカウンター: 何も起こらない
                else:
                    TurnHandler._execute_copied_skill(
                        gs, tp_key_inner, chosen, thumbs_inner, total,
                        upgrade_hand_to_win=False, was_skip_before=was_skip_before
                    )
            
            TurnHandler._resolve_choice_with_reaction = patched_choice
            try:
                with redirect_stdout(io.StringIO()):
                    TurnHandler.resolve_turn(self.game_state, tp_key, skill, thumbs, reaction)
            finally:
                TurnHandler._resolve_choice_with_reaction = original_resolve
                if hasattr(tp, '_pending_choice_target'):
                    delattr(tp, '_pending_choice_target')
        else:
            with redirect_stdout(io.StringIO()):
                TurnHandler.resolve_turn(self.game_state, tp_key, skill, thumbs, reaction)
    
    def _check_victory(self):
        """勝利判定（stdout抑制）"""
        p = self.game_state.player
        c = self.game_state.computer
        if p.get_active_hands() == 0:
            self.game_state.game_over = True
            self.game_state.winner = KEY_PLAYER
            return True
        if c.get_active_hands() == 0:
            self.game_state.game_over = True
            self.game_state.winner = KEY_COMPUTER
            return True
        return False
    
    def _advance_game_state(self, last_tp_key):
        """ターン後のゲーム進行を処理"""
        gs = self.game_state
        effects = gs.effects

        # タイム復帰チェック（最優先: 強制ターン中の追加ターンを無効化してTime使用者に戻る）
        if hasattr(gs, '_time_return_to') and gs._time_return_to:
            if gs._time_return_to != last_tp_key:
                # 強制ターン終了: 追加ターンを破棄してTime使用者に戻る
                # ※ _time_return_toはクリアしない → _auto_advanceがスキップ中かどうか検知に使用
                next_key = gs._time_return_to
                if effects.has_extra_turn(last_tp_key):
                    effects.additional_turns[last_tp_key] = 0
                gs.on_phase_end(last_tp_key)
                gs.current_player_key = next_key
                self._on_phase_start(next_key)
                return
            else:
                # Time使用者が通常ターンを完了した（スキップなし）: クリアして通常進行
                gs._time_return_to = None

        # 追加ターンチェック
        if effects.has_extra_turn(last_tp_key):
            effects.use_extra_turn(last_tp_key)
            # 同じTPが続けてターンを行う（次のstep()で処理）
            return

        # フェーズ終了
        gs.on_phase_end(last_tp_key)

        # タイム効果チェック
        opp_key = gs.get_opponent_key(last_tp_key)
        current_player = gs.get_player(last_tp_key)
        opponent = gs.get_opponent(last_tp_key)

        next_key = opp_key

        if current_player.time_active:
            current_player.time_active = False
            # タイム効果: 相手に1ターンだけ渡して戻る
            gs.current_player_key = opp_key
            self._on_phase_start(opp_key)
            # タイム効果のフラグをセット（次のstep()後に復帰）
            gs._time_return_to = last_tp_key
            return

        if opponent.time_active:
            opponent.time_active = False
            if effects.has_extra_turn(last_tp_key):
                effects.additional_turns[last_tp_key] = 0
            next_key = opp_key

        # フェーズ遷移
        gs.current_player_key = next_key
        self._on_phase_start(next_key)
    
    def _on_phase_start(self, player_key):
        """フェーズ開始処理"""
        self.game_state.on_phase_start(player_key)
    
    def _auto_advance(self):
        """スキップターン等を自動処理"""
        gs = self.game_state
        current_key = gs.current_player_key
        player = gs.get_player(current_key)

        # スキップ中のターンを自動処理
        while player.skip_phases > 0 and not gs.game_over:
            # クイックレベル減少（スキップされた場合もターン経過として扱う）
            if player.quick_level > 0:
                player.quick_level = max(0, player.quick_level - 1)
            gs.on_phase_end(current_key)

            # タイム復帰中にTime使用者がスキップされた場合:
            # タイムは全処理に優先するため相手もスキップし、Time使用者に戻る
            if hasattr(gs, '_time_return_to') and gs._time_return_to == current_key:
                gs._time_return_to = None
                opp_key = gs.get_opponent_key(current_key)
                opp = gs.get_player(opp_key)
                if opp.quick_level > 0:
                    opp.quick_level = max(0, opp.quick_level - 1)
                gs.effects.mark_first_phase_done(opp_key)
                # Time使用者に戻る
                gs.current_player_key = current_key
                self._on_phase_start(current_key)
                player = gs.get_player(current_key)
                continue  # skip_phases = 0 になったのでループ終了へ

            # 相手にターンを渡す（通常スキップ）
            current_key = gs.get_opponent_key(current_key)
            gs.current_player_key = current_key
            self._on_phase_start(current_key)
            player = gs.get_player(current_key)
    
    def _get_opponent_tp_action(self):
        """対戦相手のTP行動を生成"""
        if self.opponent_policy is None:
            return self._random_tp_action()
        
        # 対戦相手の観測を作成（相手視点）
        opp_obs = encode_observation(self.game_state, self.opponent_key, self.turn_count)
        obs_dim = getattr(self.opponent_policy, "observation_dim", OBS_TOTAL)
        opp_obs = encode_observation_for_dim(self.game_state, self.opponent_key, self.turn_count, obs_dim=obs_dim)
        opp_mask = get_action_mask(self.game_state, self.opponent_key)

        action, _ = self.opponent_policy.predict(
            opp_obs, action_masks=opp_mask, deterministic=False
        )
        return decode_action(int(action))

    def _get_opponent_ntp_action(self):
        """対戦相手のNTP行動を生成"""
        if self.opponent_policy is None:
            return self._random_ntp_action()

        # 対戦相手の観測を作成（相手視点でNTP）
        opp_obs = encode_observation(self.game_state, self.opponent_key, self.turn_count)
        obs_dim = getattr(self.opponent_policy, "observation_dim", OBS_TOTAL)
        opp_obs = encode_observation_for_dim(self.game_state, self.opponent_key, self.turn_count, obs_dim=obs_dim)
        opp_mask = get_action_mask(self.game_state, self.opponent_key)
        
        action, _ = self.opponent_policy.predict(
            opp_obs, action_masks=opp_mask, deterministic=False
        )
        return decode_action(int(action))
    
    def _random_tp_action(self):
        """ランダムなTP行動を生成"""
        mask = get_action_mask(self.game_state, self.opponent_key)
        tp_mask = mask[:NUM_TP_ACTIONS]
        valid_indices = np.where(tp_mask)[0]
        if len(valid_indices) == 0:
            return decode_action(0)  # フォールバック
        chosen = random.choice(valid_indices)
        return decode_action(int(chosen))
    
    def _random_ntp_action(self):
        """ランダムなNTP行動を生成"""
        mask = get_action_mask(self.game_state, self.opponent_key)
        ntp_mask = mask[NUM_TP_ACTIONS:]
        valid_indices = np.where(ntp_mask)[0]
        if len(valid_indices) == 0:
            return decode_action(NUM_TP_ACTIONS)  # フォールバック
        chosen = random.choice(valid_indices) + NUM_TP_ACTIONS
        return decode_action(int(chosen))
    
    def _compute_reward(self, terminated, truncated):
        """報酬を計算"""
        if terminated:
            if self.game_state.winner == self.agent_key:
                return REWARD_WIN
            else:
                return REWARD_LOSE
        if truncated:
            return REWARD_DRAW
        return 0.0
    
    def _get_info(self):
        """追加情報"""
        info = {
            'action_mask': self.action_masks(),
            'turn_count': self.turn_count,
            'agent_is_tp': self.game_state.current_player_key == self.agent_key,
        }
        # 補助タスク用: 直前の相手行動
        if self.last_opponent_action is not None:
            info['opponent_action'] = self.last_opponent_action
        return info
    
    def _get_episode_summary(self):
        """エピソードの要約を返す"""
        return {
            'winner': self.game_state.winner,
            'agent_key': self.agent_key,
            'agent_won': self.game_state.winner == self.agent_key,
            'total_turns': self.turn_count,
            'turns': self.episode_turns,
        }

    def set_opponent_path(self, path):
        """対戦相手モデルをパスから読み込んで設定。
        SubprocVecEnvからenv_method経由で呼ばれるためサブプロセス内で実行される。
        device='cpu'を強制: 16プロセス×GPUロードによるVRAM枯渇を防ぐ。"""
        try:
            from rl.opponents import _FrozenPolicy
            model = load_maskable_ppo(path, device='cpu')
            self.opponent_policy = _FrozenPolicy(model)
        except Exception as e:
            print(f"[Env] 対戦相手読み込みエラー ({path}): {e}")


def _create_game_state():
    """ゲーム状態を新規作成（printなし）"""
    from yubisuma_logic import GameState
    gs = GameState()
    # initialize_gameのprint無し版
    gs.current_player_key = None
    gs.game_over = False
    gs.winner = None
    return gs

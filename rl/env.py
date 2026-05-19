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
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.config import (
    TOTAL_ACTIONS, OBS_TOTAL, REWARD_WIN, REWARD_LOSE,
    REWARD_DRAW, MAX_TURNS,
    NUM_PERSONA_TP, NUM_PERSONA_NTP,
)
from rl.observation import encode_observation, encode_observation_for_dim
from rl.model_utils import load_maskable_ppo
from rl.actions import (
    decode_action, get_action_mask, encode_tp_action, encode_ntp_action,
    action_to_readable, NUM_TP_ACTIONS,
    CHOICE_SKILL,
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

        # Persona (UVFA + DIAYN, ICLR 2019)
        # エピソード単位で固定。reset() でランダムサンプル。
        # 学習中は分化を促し、評価/推論時は外部から P_TP0=0 に固定して柔軟運用。
        self.agent_persona_tp = 0
        self.agent_persona_ntp = 0
        self.opp_persona_tp = 0
        self.opp_persona_ntp = 0
        # 外部から persona を強制したい時の override (None なら reset でランダム)
        self._forced_agent_persona_tp = None
        self._forced_agent_persona_ntp = None

        # エピソード記録
        self.episode_turns = []

        # 最新の相手行動（補助タスク用）
        self.last_opponent_action = None
        self.opponent_spec = {"kind": "random", "preset": None, "step": None}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_state = _create_game_state()
        self.game_state.pending_choice = None
        self.turn_count = 0
        self.episode_turns = []
        self.last_opponent_action = None

        # Persona の割り当て (override がなければランダム)
        if self._forced_agent_persona_tp is not None:
            self.agent_persona_tp = int(self._forced_agent_persona_tp)
        else:
            self.agent_persona_tp = int(self.np_random.integers(NUM_PERSONA_TP))
        if self._forced_agent_persona_ntp is not None:
            self.agent_persona_ntp = int(self._forced_agent_persona_ntp)
        else:
            self.agent_persona_ntp = int(self.np_random.integers(NUM_PERSONA_NTP))
        # 対戦相手 persona もランダム (リーグ側で固定したい場合は set_opponent_persona で上書き)
        self.opp_persona_tp = int(self.np_random.integers(NUM_PERSONA_TP))
        self.opp_persona_ntp = int(self.np_random.integers(NUM_PERSONA_NTP))

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

        obs = self._agent_obs()
        info = self._get_info()

        return obs, info

    def _agent_obs(self):
        """エージェントの persona を反映した観測ベクトル。"""
        return encode_observation(
            self.game_state, self.agent_key, self.turn_count,
            persona_tp=self.agent_persona_tp,
            persona_ntp=self.agent_persona_ntp,
        )

    def set_agent_persona(self, persona_tp=None, persona_ntp=None):
        """評価時等に persona を外部から強制設定する (None なら reset でランダムに戻す)。"""
        self._forced_agent_persona_tp = persona_tp
        self._forced_agent_persona_ntp = persona_ntp

    def set_opponent_persona(self, persona_tp, persona_ntp):
        """リーグ等が対戦相手 persona を上書きする用 (即時反映)。"""
        self.opp_persona_tp = int(persona_tp)
        self.opp_persona_ntp = int(persona_ntp)
    
    def step(self, action):
        if self._has_pending_choice_for(self.agent_key):
            return self._step_pending_choice(action)

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
        if choice_target and skill != CHOICE_SKILL:
            tp = self.game_state.get_player(tp_key)
            tp._pending_choice_target = choice_target
        
        # ターン記録
        turn_record = {
            'turn': self.turn_count,
            'tp_key': tp_key,
            'skill': skill,
            'choice_target': None,
            'thumbs': thumbs.copy(),
            'reaction': reaction,
            'agent_is_tp': is_agent_tp,
        }
        
        # ターン解決（内部print抑制）
        if skill == CHOICE_SKILL:
            self._start_pending_choice(tp_key, thumbs, reaction, is_agent_tp, turn_record)
            if is_agent_tp:
                obs = self._agent_obs()
                return obs, 0.0, False, False, self._get_info()
            choice_target = self._get_opponent_choice_target()
            self.game_state.pending_choice = None
            turn_record['choice_target'] = choice_target
            if self.last_opponent_action is not None:
                self.last_opponent_action['choice_target'] = choice_target

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

        obs = self._agent_obs()
        info = self._get_info()

        # エピソード完了時にサマリをinfoに含める (AnalysisCallbackが主プロセスで収集)
        if terminated or truncated:
            info['episode_summary'] = self._get_episode_summary()

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """MaskablePPO用のアクションマスク"""
        return get_action_mask(self.game_state, self.agent_key)
    
    # === 内部メソッド ===
    
    def _has_pending_choice_for(self, player_key):
        pending = getattr(self.game_state, "pending_choice", None)
        return bool(pending and pending.get("chooser_key") == player_key)

    def _start_pending_choice(self, tp_key, thumbs, reaction, agent_is_tp, turn_record):
        self.game_state.pending_choice = {
            "chooser_key": tp_key,
            "tp_key": tp_key,
            "thumbs": thumbs.copy(),
            "reaction": reaction,
            "agent_is_tp": agent_is_tp,
            "turn_record": turn_record,
        }

    def _step_pending_choice(self, action):
        pending = self.game_state.pending_choice
        decoded = decode_action(action)
        choice_target = decoded.get("choice_target")
        tp = self.game_state.get_player(pending["tp_key"])
        available = [s for s in tp.stock if s not in tp.choice_used_this_phase]
        if choice_target not in available:
            choice_target = available[0] if available else None

        turn_record = pending["turn_record"]
        turn_record["choice_target"] = choice_target
        self.game_state.pending_choice = None

        self._resolve_turn_silent(
            pending["tp_key"],
            CHOICE_SKILL,
            pending["thumbs"],
            pending["reaction"],
            choice_target,
        )
        self.last_opponent_action = None
        return self._finish_resolved_turn(pending["tp_key"], turn_record)

    def _get_opponent_choice_target(self):
        mask = get_action_mask(self.game_state, self.opponent_key)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            pending = self.game_state.pending_choice
            tp = self.game_state.get_player(pending["tp_key"])
            available = [s for s in tp.stock if s not in tp.choice_used_this_phase]
            return available[0] if available else None

        action = self._get_opponent_action_index()

        return decode_action(action).get("choice_target")

    def _finish_resolved_turn(self, tp_key, turn_record):
        self.turn_count += 1
        self.episode_turns.append(turn_record)

        terminated = self._check_victory()

        if not terminated:
            self._advance_game_state(tp_key)
            self._auto_advance()

        truncated = False
        if not terminated and self.turn_count >= MAX_TURNS:
            truncated = True

        reward = self._compute_reward(terminated, truncated)
        obs = self._agent_obs()
        info = self._get_info()
        if terminated or truncated:
            info['episode_summary'] = self._get_episode_summary()
        return obs, reward, terminated, truncated, info

    def _resolve_turn_silent(self, tp_key, skill, thumbs, reaction, choice_target=None):
        """ターンを解決（stdout抑制）。新ルールでは choice_data を直接渡せる"""
        import io
        from contextlib import redirect_stdout

        # choice_data: チョイス選択を resolve_turn に直接渡す
        choice_data = None
        if choice_target:
            if skill == "チョイス":
                choice_data = {"choice": choice_target}
            elif skill == "オール":
                # choice_target がリストなら順序、文字列なら単一スキル
                if isinstance(choice_target, list):
                    choice_data = {"all_order": choice_target}
                else:
                    choice_data = {"all_order": [choice_target]}

        with redirect_stdout(io.StringIO()):
            TurnHandler.resolve_turn(self.game_state, tp_key, skill, thumbs, reaction, choice_data)
    
    def _check_victory(self):
        """勝利判定（新ルール: 両者のスキル宣言後のみ勝利確定）"""
        p = self.game_state.player
        c = self.game_state.computer
        both_declared = p.has_declared_skill and c.has_declared_skill
        if p.get_active_hands() == 0:
            if not both_declared:
                return False
            self.game_state.game_over = True
            self.game_state.winner = KEY_PLAYER
            return True
        if c.get_active_hands() == 0:
            if not both_declared:
                return False
            self.game_state.game_over = True
            self.game_state.winner = KEY_COMPUTER
            return True
        return False
    
    def _advance_game_state(self, last_tp_key):
        """ターン後のゲーム進行を処理"""
        gs = self.game_state
        effects = gs.effects
        opp_key = gs.get_opponent_key(last_tp_key)
        opp = gs.get_player(opp_key)

        # タイム効果チェック: last_tpが追加ターンを得た && opp(time使用者)がtime_active
        if opp.time_active and effects.has_extra_turn(last_tp_key):
            opp.time_active = False
            effects.additional_turns[last_tp_key] = 0
            gs.on_phase_end(last_tp_key)
            gs.current_player_key = opp_key
            self._on_phase_start(opp_key)
            return

        # 追加ターンチェック
        if effects.has_extra_turn(last_tp_key):
            effects.use_extra_turn(last_tp_key)
            # 同じTPが続けてターンを行う（次のstep()で処理）
            return

        # フェーズ終了 → 通常遷移
        gs.on_phase_end(last_tp_key)
        gs.current_player_key = opp_key
        self._on_phase_start(opp_key)
    
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

            # タイム+スキップ: スキップされた側がtime_activeなら、フェーズを戻す
            # （スキップされても相手に連続行動を許さない）
            if gs.get_player(current_key).time_active:
                gs.current_player_key = current_key
                self._on_phase_start(current_key)
                player = gs.get_player(current_key)
                continue

            # 相手にターンを渡す（通常スキップ）
            current_key = gs.get_opponent_key(current_key)
            gs.current_player_key = current_key
            self._on_phase_start(current_key)
            player = gs.get_player(current_key)

    def _get_opponent_action_index(self):
        mask = get_action_mask(self.game_state, self.opponent_key)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0

        if self.opponent_policy is None:
            return int(self.np_random.choice(valid_indices))

        if hasattr(self.opponent_policy, "predict_action"):
            action = int(self.opponent_policy.predict_action(
                self.game_state,
                self.opponent_key,
                mask,
                rng=self.np_random,
            ))
        else:
            obs_dim = getattr(self.opponent_policy, "observation_dim", OBS_TOTAL) or OBS_TOTAL
            opp_obs = encode_observation_for_dim(
                self.game_state, self.opponent_key, self.turn_count, obs_dim=obs_dim,
                persona_tp=self.opp_persona_tp, persona_ntp=self.opp_persona_ntp,
            )
            action_arr, _ = self.opponent_policy.predict(
                opp_obs, action_masks=mask, deterministic=False
            )
            action = int(action_arr[0])

        if action < 0 or action >= len(mask) or not mask[action]:
            action = int(self.np_random.choice(valid_indices))
        return action
    
    def _get_opponent_tp_action(self):
        """対戦相手のTP行動を生成"""
        if self.opponent_policy is None:
            return self._random_tp_action()
        return decode_action(self._get_opponent_action_index())

    def _get_opponent_ntp_action(self):
        """対戦相手のNTP行動を生成"""
        if self.opponent_policy is None:
            return self._random_ntp_action()
        return decode_action(self._get_opponent_action_index())
    
    def _random_tp_action(self):
        """ランダムなTP行動を生成"""
        mask = get_action_mask(self.game_state, self.opponent_key)
        tp_mask = mask[:NUM_TP_ACTIONS]
        valid_indices = np.where(tp_mask)[0]
        if len(valid_indices) == 0:
            return decode_action(0)  # フォールバック
        chosen = self.np_random.choice(valid_indices)
        return decode_action(int(chosen))
    
    def _random_ntp_action(self):
        """ランダムなNTP行動を生成"""
        mask = get_action_mask(self.game_state, self.opponent_key)
        ntp_mask = mask[NUM_TP_ACTIONS:]
        valid_indices = np.where(ntp_mask)[0]
        if len(valid_indices) == 0:
            return decode_action(NUM_TP_ACTIONS)  # フォールバック
        chosen = self.np_random.choice(valid_indices) + NUM_TP_ACTIONS
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
            # DIAYN discriminator のラベル
            'agent_persona_tp': self.agent_persona_tp,
            'agent_persona_ntp': self.agent_persona_ntp,
        }
        # 補助タスク用: 直前の相手行動
        pending_choice = getattr(self.game_state, "pending_choice", None)
        if pending_choice is not None:
            info['pending_choice'] = {
                'chooser_key': pending_choice.get('chooser_key'),
                'reaction': pending_choice.get('reaction'),
            }
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
            'agent_persona_tp': self.agent_persona_tp,
            'agent_persona_ntp': self.agent_persona_ntp,
            'opponent_kind': self.opponent_spec.get("kind"),
            'opponent_preset': self.opponent_spec.get("preset"),
            'opponent_step': self.opponent_spec.get("step"),
        }

    def set_opponent_path(self, path):
        """対戦相手モデルをパスから読み込んで設定。
        SubprocVecEnvからenv_method経由で呼ばれるためサブプロセス内で実行される。
        device='cpu'を強制: 16プロセス×GPUロードによるVRAM枯渇を防ぐ。"""
        self.set_opponent_spec({"kind": "model", "path": path})

    def set_opponent_spec(self, spec):
        """Set opponent policy from a serializable league spec."""
        self.opponent_spec = dict(spec or {"kind": "random", "preset": None, "step": None})
        try:
            from rl.opponents import create_opponent_policy
            self.opponent_policy = create_opponent_policy(spec, device='cpu')
        except Exception as e:
            print(f"[Env] opponent load error ({spec}): {e}")
            try:
                from rl.opponents import RuleStrategyPolicy
                self.opponent_policy = RuleStrategyPolicy("balanced")
                self.opponent_spec = {"kind": "fallback_rule", "preset": "balanced", "step": None}
            except Exception:
                self.opponent_policy = None
                self.opponent_spec = {"kind": "random", "preset": None, "step": None}

    def get_search_teacher(self, samples_per_action=1, rollout_turns=32,
                           max_actions=24, temperature=0.35):
        """Return a counterfactual policy target from terminal random rollouts.

        The target is reward-preserving: it is derived from actual terminal
        outcomes, not from hand-written skill bonuses.
        """
        obs = self._agent_obs()
        mask = self.action_masks()
        legal = np.where(mask)[0]
        if len(legal) <= 1:
            return None

        if max_actions and len(legal) > max_actions:
            legal = self.np_random.choice(legal, size=max_actions, replace=False)
            legal = np.asarray(legal, dtype=np.int64)

        scores = []
        samples = max(1, int(samples_per_action))
        for action in legal:
            total = 0.0
            for _ in range(samples):
                total += self._rollout_action_value(int(action), int(rollout_turns))
            scores.append(total / samples)

        scores = np.asarray(scores, dtype=np.float32)
        target = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        scaled = scores / max(1e-6, float(temperature))
        scaled = scaled - float(np.max(scaled))
        probs = np.exp(scaled)
        prob_sum = float(probs.sum())
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / prob_sum
        target[legal] = probs.astype(np.float32)
        return {
            "obs": obs,
            "mask": mask.astype(np.bool_),
            "target": target,
        }

    def _rollout_action_value(self, action, rollout_turns):
        sim = self._clone_for_search()
        try:
            _, reward, terminated, truncated, _ = sim.step(action)
            done = terminated or truncated
            turns = 0
            while not done and turns < rollout_turns:
                mask = sim.action_masks()
                legal = np.where(mask)[0]
                if len(legal) == 0:
                    break
                next_action = int(sim.np_random.choice(legal))
                _, reward, terminated, truncated, _ = sim.step(next_action)
                done = terminated or truncated
                turns += 1
            return float(reward) if done else 0.0
        except Exception:
            return 0.0
        finally:
            sim.close()

    def _clone_for_search(self):
        sim = YubisumaEnv(opponent_policy=None, render_mode=None)
        sim.game_state = copy.deepcopy(self.game_state)
        sim.agent_key = self.agent_key
        sim.opponent_key = self.opponent_key
        sim.turn_count = self.turn_count
        sim.agent_persona_tp = self.agent_persona_tp
        sim.agent_persona_ntp = self.agent_persona_ntp
        sim.opp_persona_tp = self.opp_persona_tp
        sim.opp_persona_ntp = self.opp_persona_ntp
        sim.episode_turns = copy.deepcopy(self.episode_turns)
        sim.last_opponent_action = copy.deepcopy(self.last_opponent_action)
        sim.opponent_spec = copy.deepcopy(self.opponent_spec)
        return sim


def _create_game_state():
    """ゲーム状態を新規作成（printなし）"""
    from yubisuma_logic import GameState
    gs = GameState()
    # initialize_gameのprint無し版
    gs.current_player_key = None
    gs.game_over = False
    gs.winner = None
    gs.pending_choice = None
    return gs

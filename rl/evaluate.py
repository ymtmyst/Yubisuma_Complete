# rl/evaluate.py - 評価・分析スクリプト
"""
訓練済みモデルの評価と分析データの可視化。

使い方:
  python -m rl.evaluate --model PATH              # モデル評価
  python -m rl.evaluate --stats                    # 分析統計表示
  python -m rl.evaluate --model PATH --watch       # 1ゲームを詳細表示
  python -m rl.evaluate --policy                   # 方策softmax分布を分析
  python -m rl.evaluate --policy --latest          # 最新チェックポイントを自動選択
"""

import argparse
import glob as glob_module
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

from rl.config import MODEL_DIR, ANALYSIS_DIR, TOTAL_ACTIONS
from rl.env import YubisumaEnv
from rl.actions import decode_action, action_to_readable
from rl.analysis import AnalysisDB
from rl.model_utils import import_maskable_ppo, load_maskable_ppo
from rl.observation import encode_observation_for_dim


def _get_model_obs(model, env):
    """Build an observation vector matching the loaded model's expected width."""
    obs_dim = int(model.observation_space.shape[0])
    return encode_observation_for_dim(
        env.game_state,
        env.agent_key,
        env.turn_count,
        obs_dim=obs_dim,
        persona_tp=env.agent_persona_tp,
        persona_ntp=env.agent_persona_ntp,
    )


def find_latest_model():
    """最新のチェックポイントを更新時刻で自動検索"""
    top_level = glob_module.glob(os.path.join(MODEL_DIR, "*.zip"))
    if top_level:
        return max(top_level, key=os.path.getmtime)

    files = glob_module.glob(os.path.join(MODEL_DIR, "**", "*.zip"), recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_latest_analysis_db():
    """rl_analysis 配下の最新DBを更新時刻で自動検索"""
    files = glob_module.glob(os.path.join(ANALYSIS_DIR, "*.db"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _bar(value, width=32, max_val=1.0):
    """テキストプログレスバーを生成"""
    filled = int(value / max_val * width) if max_val > 0 else 0
    return "#" * filled + "." * (width - filled)
    return "#" * filled + "." * (width - filled)


def _skill_name(skill):
    """スキルの表示名を返す"""
    if isinstance(skill, str) and skill.startswith("チョイス:"):
        return f"チョイス:{skill.split(':')[1]}"
    return str(skill)


def _print_episode_logs(episode_logs):
    """エピソードの詳細ターンログを表示"""
    from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER

    print(f"\n\n{'='*72}")
    print(f"  直近エピソード詳細 ({len(episode_logs)}試合分)")
    print(f"{'='*72}")

    for log in episode_logs:
        sente_label = "先手" if log['agent_is_sente'] else "後手"
        result_str = "★勝利" if log['agent_won'] else "✗敗北"
        agent_disp = "player" if log['agent_key'] == KEY_PLAYER else "computer"
        fp_disp = "player" if log['first_player_key'] == KEY_PLAYER else "computer"

        print(f"\n{'─'*72}")
        print(
            f"  エピソード #{log['ep_num']}  "
            f"Agent={agent_disp}({sente_label})  "
            f"先手={fp_disp}  "
            f"{result_str}  ({log['total_turns']}ターン)"
        )
        print(f"{'─'*72}")

        for t in log['turns']:
            tp_who = "Agent★" if t['is_agent_tp'] else "相手  "
            skill_col = f"{t['skill']:<10}"
            react_col = f"{t['reaction']:<8}"
            # 残り手の表示
            ph = t['player_hands']
            ch = t['computer_hands']
            hands_str = f"P:{ph}本 C:{ch}本"
            # 手が0になった=決着
            decisive = " ← 決着!" if (ph == 0 or ch == 0) else ""

            print(
                f"  T{t['turn_num']:>2} [TP:{tp_who}] {skill_col} 指{t['tp_thumbs']}本"
                f" / [{react_col}] 指{t['ntp_thumbs']}本"
                f"  → {hands_str}{decisive}"
            )

        print(f"       → {result_str}")

    print(f"\n{'='*72}\n")


def analyze_policy(model_path, n_episodes=150, n_show_episodes=5):
    """
    ポリシーネットワークのsoftmax出力を集計して方策分布を表示。
    先手/後手別の勝率・方策の違いも表示。
    最後のn_show_episodes試合の詳細ターンログも出力。

    使い方 (訓練中でも別ターミナルから実行可):
      python -m rl.evaluate --policy --latest
      python -m rl.evaluate --policy --latest --episodes 300
    """
    import torch
    from collections import deque
    MaskablePPO = import_maskable_ppo()
    from rl.config import (
        TP_SKILL_OPTIONS, NTP_REACTION_OPTIONS,
        NUM_TP_SKILLS, NUM_THUMB_OPTIONS, NUM_TP_ACTIONS,
        NUM_NTP_REACTIONS,
    )
    from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER

    print(f"[Policy] モデル読み込み: {os.path.basename(model_path)}")
    model = load_maskable_ppo(model_path)
    policy = model.policy
    policy.eval()
    device = model.device

    env = YubisumaEnv(opponent_policy=None)

    # 全体集計
    tp_skill_probs = np.zeros(NUM_TP_SKILLS)
    tp_thumb_probs = np.zeros(NUM_THUMB_OPTIONS)
    ntp_react_probs = np.zeros(NUM_NTP_REACTIONS)
    ntp_thumb_probs = np.zeros(NUM_THUMB_OPTIONS)
    tp_values, ntp_values = [], []
    tp_steps = ntp_steps = 0

    # 先手/後手別集計
    sente_tp_skill = np.zeros(NUM_TP_SKILLS)
    gote_tp_skill = np.zeros(NUM_TP_SKILLS)
    sente_tp_steps = gote_tp_steps = 0
    sente_wins = sente_total = 0
    gote_wins = gote_total = 0

    # 直近エピソードログ
    episode_logs = deque(maxlen=n_show_episodes)

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent_key = env.agent_key
        first_player_key = env.game_state.effects.first_player_key
        agent_is_sente = (agent_key == first_player_key)

        turn_logs = []
        done = False

        while not done:
            mask = env.action_masks()
            is_tp = bool(mask[:NUM_TP_ACTIONS].any())
            current_tp_key = env.game_state.current_player_key  # stepの前に取得

            model_obs = _get_model_obs(model, env)
            obs_tensor = torch.FloatTensor(model_obs).unsqueeze(0).to(device)

            with torch.no_grad():
                dist = policy.get_distribution(
                    obs_tensor, action_masks=mask.reshape(1, -1)
                )
                probs = dist.distribution.probs.squeeze(0).cpu().numpy()
                value = policy.predict_values(obs_tensor).item()

            # TP/NTP 別に確率を累積
            if is_tp:
                mat = probs[:NUM_TP_ACTIONS].reshape(NUM_TP_SKILLS, NUM_THUMB_OPTIONS)
                tp_skill_probs += mat.sum(axis=1)
                tp_thumb_probs += mat.sum(axis=0)
                tp_values.append(value)
                tp_steps += 1
                # 先手/後手別
                if agent_is_sente:
                    sente_tp_skill += mat.sum(axis=1)
                    sente_tp_steps += 1
                else:
                    gote_tp_skill += mat.sum(axis=1)
                    gote_tp_steps += 1
            else:
                mat = probs[NUM_TP_ACTIONS:].reshape(NUM_NTP_REACTIONS, NUM_THUMB_OPTIONS)
                ntp_react_probs += mat.sum(axis=1)
                ntp_thumb_probs += mat.sum(axis=0)
                ntp_values.append(value)
                ntp_steps += 1

            action, _ = model.predict(model_obs, action_masks=mask, deterministic=False)
            decoded = decode_action(int(action))

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            # ターンログ記録（step後に取得）
            opp_action = info.get('opponent_action') or {}
            p_hands = env.game_state.player.get_active_hands()
            c_hands = env.game_state.computer.get_active_hands()

            if is_tp:
                skill_str = _skill_name(decoded['skill'])
                if decoded.get('choice_target'):
                    skill_str = f"チョイス({decoded['choice_target']})"
                tp_th = decoded['thumbs']
                ntp_react = opp_action.get('reaction') or 'なし'
                ntp_th = opp_action.get('thumbs', '?')
            else:
                opp_skill = opp_action.get('skill', '?')
                skill_str = _skill_name(opp_skill) if opp_skill != '?' else '?'
                if opp_action.get('choice_target'):
                    skill_str = f"チョイス({opp_action['choice_target']})"
                tp_th = opp_action.get('thumbs', '?')
                ntp_react = decoded['reaction'] or 'なし'
                ntp_th = decoded['thumbs']

            turn_logs.append({
                'turn_num': len(turn_logs) + 1,
                'tp_key': current_tp_key,
                'is_agent_tp': is_tp,
                'skill': skill_str,
                'tp_thumbs': tp_th,
                'reaction': ntp_react,
                'ntp_thumbs': ntp_th,
                'player_hands': p_hands,
                'computer_hands': c_hands,
            })

        agent_won = (reward > 0)

        # 先手/後手別勝率集計
        if agent_is_sente:
            sente_total += 1
            if agent_won:
                sente_wins += 1
        else:
            gote_total += 1
            if agent_won:
                gote_wins += 1

        episode_logs.append({
            'ep_num': ep + 1,
            'agent_key': agent_key,
            'first_player_key': first_player_key,
            'agent_is_sente': agent_is_sente,
            'turns': turn_logs,
            'agent_won': agent_won,
            'total_turns': env.turn_count,
        })

        if (ep + 1) % 50 == 0:
            print(f"  {ep+1}/{n_episodes} エピソード完了...")

    env.close()

    # 正規化
    sp = tp_skill_probs / tp_steps if tp_steps > 0 else tp_skill_probs
    tp_t = tp_thumb_probs / tp_steps if tp_steps > 0 else tp_thumb_probs
    nr = ntp_react_probs / ntp_steps if ntp_steps > 0 else ntp_react_probs
    nt_t = ntp_thumb_probs / ntp_steps if ntp_steps > 0 else ntp_thumb_probs
    sente_sp = sente_tp_skill / sente_tp_steps if sente_tp_steps > 0 else sente_tp_skill
    gote_sp = gote_tp_skill / gote_tp_steps if gote_tp_steps > 0 else gote_tp_skill

    W = 22  # バー幅

    print(f"\n{'='*72}")
    print(f"  方策分析  ({n_episodes}ep | TP:{tp_steps}steps / NTP:{ntp_steps}steps)")
    print(f"{'='*72}")

    # 勝率: 全体・先手・後手
    overall_wr = (sente_wins + gote_wins) / n_episodes
    sente_wr = sente_wins / sente_total if sente_total > 0 else 0.0
    gote_wr = gote_wins / gote_total if gote_total > 0 else 0.0
    print(f"\n【勝率 (対ランダム)】")
    print(f"  全体: {overall_wr:.1%}  ({sente_wins+gote_wins}/{n_episodes}ep)")
    print(f"  先手: {sente_wr:.1%}  ({sente_wins}/{sente_total}ep)  "
          f"{_bar(sente_wr, 20)}")
    print(f"  後手: {gote_wr:.1%}  ({gote_wins}/{gote_total}ep)  "
          f"{_bar(gote_wr, 20)}")

    # TP スキル分布: 先手/後手比較
    # チョイス:* の8エントリを合算して "チョイス(計)" として表示
    choice_ps = sum(
        sente_sp[i] for i, s in enumerate(TP_SKILL_OPTIONS)
        if isinstance(s, str) and s.startswith("チョイス:")
    )
    choice_pg = sum(
        gote_sp[i] for i, s in enumerate(TP_SKILL_OPTIONS)
        if isinstance(s, str) and s.startswith("チョイス:")
    )

    rows = []  # (name, ps, pg, always_show)
    for i, skill in enumerate(TP_SKILL_OPTIONS):
        if isinstance(skill, str) and skill.startswith("チョイス:"):
            continue  # 合算済みなのでスキップ
        name = _skill_name(skill)
        # ドロップ・チョイスは条件付きスキルなので頻度0でも必ず表示
        always = name == "ドロップ"
        rows.append((name, sente_sp[i], gote_sp[i], always))

    rows.append(("チョイス(計)", choice_ps, choice_pg, True))

    # 後手確率の降順でソート
    rows.sort(key=lambda x: x[2], reverse=True)
    max_bar = max((max(r[1], r[2]) for r in rows), default=0.001)
    max_bar = max(max_bar, 0.001)

    print(f"\n【TP: スキル選択確率 (先手 vs 後手)】  ※softmax出力の平均")
    print(f"  {'スキル':<18} {'先手':>6}  {'後手':>6}  先手分布")
    print(f"  {'-'*18} {'-'*6}  {'-'*6}  {'-'*W}")
    for name, ps, pg, always_show in rows:
        if not always_show and ps < 0.002 and pg < 0.002:
            continue
        flag = " ←封印" if ps < 0.001 and pg > 0.01 else ""
        print(f"  {name:<18} {ps:>5.1%}  {pg:>5.1%}  {_bar(ps, W, max_bar)}{flag}")

    # TP 指本数分布
    print(f"\n【TP: 指本数分布】")
    max_tt = tp_t.max() if tp_t.max() > 0 else 1.0
    for thumb, p in enumerate(tp_t):
        print(f"  {thumb}本  {p:>5.1%}  {_bar(p, 28, max_tt)}")

    # NTP リアクション分布
    print(f"\n【NTP: リアクション分布】")
    max_nr = nr.max() if nr.max() > 0 else 1.0
    for i, reaction in enumerate(NTP_REACTION_OPTIONS):
        print(f"  {reaction:<10} {nr[i]:>5.1%}  {_bar(nr[i], 28, max_nr)}")

    # NTP 指本数分布
    print(f"\n【NTP: 指本数分布】")
    max_nt = nt_t.max() if nt_t.max() > 0 else 1.0
    for thumb, p in enumerate(nt_t):
        print(f"  {thumb}本  {p:>5.1%}  {_bar(p, 28, max_nt)}")

    # 状態価値
    print(f"\n【状態価値 (Critic出力の平均)】")
    if tp_values:
        print(f"  TP時:  {np.mean(tp_values):+.3f}  (±{np.std(tp_values):.3f})")
    if ntp_values:
        print(f"  NTP時: {np.mean(ntp_values):+.3f}  (±{np.std(ntp_values):.3f})")

    print(f"\n{'='*72}")

    # エピソード詳細表示
    _print_episode_logs(list(episode_logs))


def evaluate_model(model_path, n_episodes=200, verbose=0):
    """モデルを評価"""
    MaskablePPO = import_maskable_ppo()
    
    print(f"[Eval] モデル読み込み: {model_path}")
    model = load_maskable_ppo(model_path)
    
    # ランダム相手との対戦
    env = YubisumaEnv(opponent_policy=None)
    
    wins = 0
    total_turns = []
    skill_counter = Counter()
    reaction_counter = Counter()
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            mask = env.action_masks()
            model_obs = _get_model_obs(model, env)
            action, _ = model.predict(model_obs, action_masks=mask, deterministic=False)
            
            decoded = decode_action(int(action))
            if decoded['role'] == 'tp':
                skill_counter[str(decoded['skill'])] += 1
            else:
                reaction_counter[decoded['reaction'] or 'なし'] += 1
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        
        if reward > 0:
            wins += 1
        total_turns.append(env.turn_count)
        
        if verbose > 0 and (ep + 1) % 50 == 0:
            print(f"  {ep+1}/{n_episodes}: 勝率={wins/(ep+1):.1%}")
    
    env.close()
    
    # 結果表示
    print(f"\n=== 評価結果 ({n_episodes}エピソード) ===")
    print(f"勝率: {wins/n_episodes:.1%} ({wins}/{n_episodes})")
    print(f"平均ターン数: {np.mean(total_turns):.1f} ± {np.std(total_turns):.1f}")
    
    print(f"\n--- スキル使用頻度 ---")
    total_skills = sum(skill_counter.values())
    for skill, count in skill_counter.most_common():
        print(f"  {skill}: {count} ({count/total_skills:.1%})")
    
    print(f"\n--- リアクション分布 ---")
    total_react = sum(reaction_counter.values())
    for reaction, count in reaction_counter.most_common():
        print(f"  {reaction}: {count} ({count/total_react:.1%})")
    
    return wins / n_episodes


def watch_game(model_path):
    """1ゲームを詳細表示"""
    MaskablePPO = import_maskable_ppo()
    
    print(f"[Watch] モデル読み込み: {model_path}")
    model = load_maskable_ppo(model_path)
    
    env = YubisumaEnv(opponent_policy=None)
    obs, info = env.reset()
    
    print(f"\n=== ゲーム開始 ===")
    print(f"エージェント: {env.agent_key}")
    print(f"先手: {env.game_state.current_player_key}")
    
    done = False
    turn = 0
    
    while not done:
        mask = env.action_masks()
        model_obs = _get_model_obs(model, env)
        action, _ = model.predict(model_obs, action_masks=mask, deterministic=False)
        
        decoded = decode_action(int(action))
        is_tp = info.get('agent_is_tp', False)
        
        print(f"\n--- ターン {turn+1} (Agent {'TP' if is_tp else 'NTP'}) ---")
        print(f"  行動: {action_to_readable(int(action))}")
        
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        # 相手行動
        opp_action = info.get('opponent_action')
        if opp_action:
            if opp_action['role'] == 'tp':
                opp_str = f"TP: {opp_action['skill']} / 指{opp_action['thumbs']}本"
            else:
                r = opp_action['reaction'] or 'なし'
                opp_str = f"NTP: {r} / 指{opp_action['thumbs']}本"
            print(f"  相手: {opp_str}")
        
        # 状態表示
        gs = env.game_state
        p = gs.player
        c = gs.computer
        print(f"  Player手: {p.get_active_hands()}, Computer手: {c.get_active_hands()}")
        
        done = terminated or truncated
        turn += 1
    
    if reward > 0:
        print(f"\n★ エージェントの勝利！ ({turn}ターン)")
    elif reward < 0:
        print(f"\n✗ エージェントの敗北 ({turn}ターン)")
    else:
        print(f"\n△ 引き分け ({turn}ターン)")
    
    env.close()


def show_stats(db_path=None, run_id=None):
    """分析統計を表示"""
    db = AnalysisDB(db_path=db_path, run_id=run_id)
    
    # 全体サマリー
    summary = db.get_summary()
    print("=== 全体統計 ===")
    print(f"総エピソード数: {summary['total_episodes']}")
    print(f"総合勝率: {summary['overall_win_rate']:.1%}")
    print(f"直近100戦: {summary['recent_100_win_rate']:.1%} "
          f"({summary['recent_100_wins']}/{summary['recent_100_games']})")
    
    # 勝率推移
    win_history = db.get_win_rate_history(window=200)
    if win_history:
        print(f"\n=== 勝率推移 ===")
        for entry in win_history[-10:]:
            bar = "#" * int(entry['win_rate'] * 40)
            print(f"  Step {entry['step']:>8}: {entry['win_rate']:.1%} {bar}")
    
    # スキル使用統計
    skill_stats = db.get_skill_usage_stats()
    if skill_stats:
        print(f"\n=== スキル使用統計 (直近500戦) ===")
        print(f"  {'スキル':<16} {'使用回数':>8} {'勝率':>8}")
        print(f"  {'-'*16} {'-'*8} {'-'*8}")
        for s in skill_stats:
            print(f"  {s['skill']:<16} {s['usage_count']:>8} "
                  f"{s['win_rate']:>7.1%}")
    
    # リアクション統計
    react_stats = db.get_reaction_stats()
    if react_stats:
        print(f"\n=== リアクション統計 ===")
        for r in react_stats:
            print(f"  {r['reaction']}: {r['count']} "
                  f"(勝率: {r['win_rate']:.1%})")

    opponent_stats = db.get_opponent_stats()
    if opponent_stats:
        print(f"\n=== Opponent breakdown ===")
        for o in opponent_stats:
            label = o['kind'] if o['preset'] == '-' else f"{o['kind']}/{o['preset']}"
            print(f"  {label}: {o['games']} games, "
                  f"win={o['win_rate']:.1%}, avg_turns={o['avg_turns']:.1f}")

    persona_stats = db.get_persona_stats()
    if persona_stats:
        print(f"\n=== Persona breakdown ===")
        for p in persona_stats[:12]:
            print(f"  TP{p['persona_tp']} NTP{p['persona_ntp']}: "
                  f"{p['games']} games, win={p['win_rate']:.1%}, "
                  f"avg_turns={p['avg_turns']:.1f}")
    
    # 指分布
    thumb_stats = db.get_thumb_distribution()
    if thumb_stats:
        print(f"\n=== 指本数分布 ===")
        total = sum(t['count'] for t in thumb_stats)
        for t in thumb_stats:
            pct = t['count'] / total * 100 if total > 0 else 0
            bar = "#" * int(pct)
            print(f"  {t['thumbs']}本: {t['count']:>6} ({pct:>5.1f}%) {bar}")
    
    # エピソード長統計
    length_stats = db.get_episode_length_stats()
    if length_stats:
        print(f"\n=== エピソード長 ===")
        print(f"  平均: {length_stats['mean']:.1f} ± {length_stats['std']:.1f}")
        print(f"  中央値: {length_stats['median']:.1f}")
        print(f"  勝利時平均: {length_stats['win_mean']:.1f}")
        print(f"  敗北時平均: {length_stats['loss_mean']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="指スマAI 評価・分析")
    parser.add_argument("--model", type=str, help="モデルパス (.zip)")
    parser.add_argument("--latest", action="store_true",
                        help="最新チェックポイントを自動選択 (訓練中でも使用可)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="評価/分析エピソード数")
    parser.add_argument("--watch", action="store_true",
                        help="1ゲームを詳細表示")
    parser.add_argument("--stats", action="store_true",
                        help="訓練統計を表示")
    parser.add_argument("--db-path", type=str, default=None,
                        help="分析DBパス (--stats 用)")
    parser.add_argument("--latest-db", action="store_true",
                        help="rl_analysis 配下の最新DBを使用 (--stats 用)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="分析DB内のrun_idで絞り込み (--stats 用)")
    parser.add_argument("--policy", action="store_true",
                        help="方策のsoftmax確率分布を分析 (--latest と組み合わせ推奨)")
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    # --latest: 最新チェックポイントをモデルパスに設定
    if args.latest and not args.model:
        args.model = find_latest_model()
        if args.model:
            print(f"[Auto] 最新モデル: {os.path.basename(args.model)}")
        else:
            print("[Error] モデルが見つかりません。先に訓練を実行してください。")
            print(f"       検索先: {MODEL_DIR}")
            return

    if args.stats:
        db_path = args.db_path
        if args.latest_db and not db_path:
            db_path = find_latest_analysis_db()
            if db_path:
                print(f"[Auto] 最新DB: {os.path.basename(db_path)}")
            else:
                print(f"[Info] rl_analysis 配下にDBが見つかりません。既定DBを使用します。")
        show_stats(db_path=db_path, run_id=args.run_id)
    elif args.policy:
        model_path = args.model or find_latest_model()
        if not model_path:
            print("[Error] --model PATH または --latest を指定してください")
            return
        analyze_policy(model_path, args.episodes)
    elif args.model:
        if args.watch:
            watch_game(args.model)
        else:
            evaluate_model(args.model, args.episodes, args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

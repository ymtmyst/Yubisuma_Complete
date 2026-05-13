# rl/analysis.py - SQLite分析データベース
"""
訓練中のゲームデータをSQLiteに記録し、
スキル使用頻度・勝率・行動パターン等の分析を可能にする。
"""

import sqlite3
import json
import os
from datetime import datetime

from rl.config import DB_PATH


class AnalysisDB:
    """分析用SQLiteデータベース"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """テーブル作成"""
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    training_step INTEGER,
                    opponent_generation INTEGER,
                    agent_key TEXT,
                    winner TEXT,
                    agent_won INTEGER,
                    total_turns INTEGER,
                    duration_ms REAL
                );
                
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    tp_key TEXT NOT NULL,
                    skill TEXT,
                    choice_target TEXT,
                    player_thumbs INTEGER,
                    computer_thumbs INTEGER,
                    reaction TEXT,
                    agent_is_tp INTEGER,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id)
                );
                
                CREATE TABLE IF NOT EXISTS training_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    win_rate REAL,
                    avg_episode_length REAL,
                    policy_loss REAL,
                    value_loss REAL,
                    entropy REAL,
                    aux_reaction_loss REAL,
                    aux_thumbs_loss REAL,
                    aux_skill_loss REAL,
                    league_size INTEGER,
                    opponent_generation INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS skill_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    skill_name TEXT NOT NULL,
                    usage_count INTEGER,
                    success_count INTEGER,
                    win_correlation REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_episodes_step ON episodes(training_step);
                CREATE INDEX IF NOT EXISTS idx_turns_episode ON turns(episode_id);
                CREATE INDEX IF NOT EXISTS idx_training_step ON training_stats(step);
            """)
    
    def _connect(self):
        return sqlite3.connect(self.db_path)
    
    def record_episode(self, episode_data, training_step=None,
                       opponent_generation=None):
        """エピソードを記録"""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO episodes
                (timestamp, training_step, opponent_generation, agent_key,
                 winner, agent_won, total_turns)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                training_step,
                opponent_generation,
                episode_data.get('agent_key'),
                episode_data.get('winner'),
                int(episode_data.get('agent_won', False)),
                episode_data.get('total_turns', 0),
            ))
            episode_id = cursor.lastrowid
            
            # ターンデータ
            turns = episode_data.get('turns', [])
            for turn in turns:
                thumbs = turn.get('thumbs', {})
                conn.execute("""
                    INSERT INTO turns
                    (episode_id, turn_number, tp_key, skill, choice_target,
                     player_thumbs, computer_thumbs, reaction, agent_is_tp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    turn.get('turn', 0),
                    turn.get('tp_key'),
                    str(turn.get('skill', '')),
                    turn.get('choice_target'),
                    thumbs.get('player', 0),
                    thumbs.get('computer', 0),
                    turn.get('reaction'),
                    int(turn.get('agent_is_tp', False)),
                ))
            
            return episode_id
    
    def record_training_stats(self, step, stats):
        """訓練統計を記録"""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO training_stats
                (timestamp, step, win_rate, avg_episode_length,
                 policy_loss, value_loss, entropy,
                 aux_reaction_loss, aux_thumbs_loss, aux_skill_loss,
                 league_size, opponent_generation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                step,
                stats.get('win_rate'),
                stats.get('avg_episode_length'),
                stats.get('policy_loss'),
                stats.get('value_loss'),
                stats.get('entropy'),
                stats.get('aux_reaction_loss'),
                stats.get('aux_thumbs_loss'),
                stats.get('aux_skill_loss'),
                stats.get('league_size'),
                stats.get('opponent_generation'),
            ))
    
    # === 分析クエリ ===
    
    def get_win_rate_history(self, window=100):
        """勝率の推移を取得"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT training_step, agent_won
                FROM episodes
                WHERE training_step IS NOT NULL
                ORDER BY training_step
            """).fetchall()
        
        if not rows:
            return []
        
        results = []
        for i in range(0, len(rows), window):
            chunk = rows[i:i+window]
            step = chunk[-1][0]
            wins = sum(r[1] for r in chunk)
            results.append({
                'step': step,
                'win_rate': wins / len(chunk),
                'games': len(chunk),
            })
        return results
    
    def get_skill_usage_stats(self, last_n_episodes=500):
        """スキル使用統計を取得"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT t.skill, COUNT(*) as cnt,
                       COUNT(DISTINCT CASE WHEN e.agent_won = 1
                             THEN t.episode_id ELSE NULL END) as win_episodes,
                       COUNT(DISTINCT t.episode_id) as episodes
                FROM turns t
                JOIN episodes e ON t.episode_id = e.id
                WHERE t.agent_is_tp = 1
                AND t.episode_id > (SELECT MAX(id) - ? FROM episodes)
                GROUP BY t.skill
                ORDER BY cnt DESC
            """, (last_n_episodes,)).fetchall()

        results = []
        for skill, count, win_episodes, episodes in rows:
            results.append({
                'skill': skill,
                'usage_count': count,
                'win_episode_count': win_episodes,
                'episode_count': episodes,
                # 「そのスキルを使ったエピソードのうち勝利した割合」= [0,1]に収まる
                'win_rate': win_episodes / episodes if episodes > 0 else 0,
            })
        return results
    
    def get_reaction_stats(self, last_n_episodes=500):
        """リアクション使用統計を取得"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT t.reaction, COUNT(*) as cnt,
                       SUM(CASE WHEN e.agent_won = 1 THEN 1 ELSE 0 END) as wins
                FROM turns t
                JOIN episodes e ON t.episode_id = e.id
                WHERE t.agent_is_tp = 0
                AND t.episode_id > (SELECT MAX(id) - ? FROM episodes)
                GROUP BY t.reaction
                ORDER BY cnt DESC
            """, (last_n_episodes,)).fetchall()
        
        results = []
        for reaction, count, wins in rows:
            results.append({
                'reaction': reaction or 'なし',
                'count': count,
                'win_count': wins,
                'win_rate': wins / count if count > 0 else 0,
            })
        return results
    
    def get_thumb_distribution(self, last_n_episodes=500):
        """指本数の分布を取得"""
        with self._connect() as conn:
            # エージェントの指分布
            rows = conn.execute("""
                SELECT 
                    CASE WHEN t.agent_is_tp = 1 
                         THEN CASE WHEN t.tp_key = 'player' 
                              THEN t.player_thumbs ELSE t.computer_thumbs END
                         ELSE CASE WHEN t.tp_key = 'player'
                              THEN t.computer_thumbs ELSE t.player_thumbs END
                    END as agent_thumbs,
                    COUNT(*) as cnt
                FROM turns t
                WHERE t.episode_id > (SELECT MAX(id) - ? FROM episodes)
                GROUP BY agent_thumbs
                ORDER BY agent_thumbs
            """, (last_n_episodes,)).fetchall()
        
        return [{'thumbs': t, 'count': c} for t, c in rows]
    
    def get_episode_length_stats(self, last_n_episodes=500):
        """エピソード長の統計"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT total_turns, agent_won
                FROM episodes
                WHERE id > (SELECT MAX(id) - ? FROM episodes)
            """, (last_n_episodes,)).fetchall()
        
        if not rows:
            return {}
        
        lengths = [r[0] for r in rows]
        wins = [r[0] for r in rows if r[1] == 1]
        losses = [r[0] for r in rows if r[1] == 0]
        
        import statistics
        return {
            'mean': statistics.mean(lengths) if lengths else 0,
            'median': statistics.median(lengths) if lengths else 0,
            'std': statistics.stdev(lengths) if len(lengths) > 1 else 0,
            'win_mean': statistics.mean(wins) if wins else 0,
            'loss_mean': statistics.mean(losses) if losses else 0,
        }
    
    def get_summary(self):
        """全体の要約"""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE agent_won = 1"
            ).fetchone()[0]
            recent = conn.execute("""
                SELECT COUNT(*), SUM(agent_won) FROM episodes
                WHERE id > (SELECT MAX(id) - 100 FROM episodes)
            """).fetchone()
        
        return {
            'total_episodes': total,
            'total_wins': wins,
            'overall_win_rate': wins / total if total > 0 else 0,
            'recent_100_games': recent[0] or 0,
            'recent_100_wins': recent[1] or 0,
            'recent_100_win_rate': (
                (recent[1] or 0) / recent[0] if recent[0] and recent[0] > 0 else 0
            ),
        }

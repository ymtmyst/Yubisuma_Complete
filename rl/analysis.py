# rl/analysis.py - SQLite analysis database
"""
Stores training episodes and summary statistics for later inspection.

New training runs should pass a dedicated db_path and run_id so metrics from
different experiments do not get mixed together.
"""

import os
import sqlite3
from datetime import datetime

from rl.config import DB_PATH


class AnalysisDB:
    """SQLite helper for RL training analysis."""

    def __init__(self, db_path=None, run_id=None):
        self.db_path = db_path or DB_PATH
        self.run_id = run_id
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    training_step INTEGER,
                    run_id TEXT,
                    opponent_generation INTEGER,
                    agent_key TEXT,
                    winner TEXT,
                    agent_won INTEGER,
                    total_turns INTEGER,
                    duration_ms REAL,
                    agent_persona_tp INTEGER,
                    agent_persona_ntp INTEGER,
                    opponent_kind TEXT,
                    opponent_preset TEXT,
                    opponent_step INTEGER
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
                    run_id TEXT,
                    step INTEGER NOT NULL,
                    win_rate REAL,
                    avg_episode_length REAL,
                    policy_loss REAL,
                    value_loss REAL,
                    entropy REAL,
                    aux_reaction_loss REAL,
                    aux_thumbs_loss REAL,
                    aux_skill_loss REAL,
                    aux_loss REAL,
                    search_teacher_loss REAL,
                    diversity_loss REAL,
                    diversity_disc_tp_acc REAL,
                    diversity_disc_ntp_acc REAL,
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
            self._ensure_column(conn, "episodes", "run_id", "TEXT")
            self._ensure_column(conn, "episodes", "agent_persona_tp", "INTEGER")
            self._ensure_column(conn, "episodes", "agent_persona_ntp", "INTEGER")
            self._ensure_column(conn, "episodes", "opponent_kind", "TEXT")
            self._ensure_column(conn, "episodes", "opponent_preset", "TEXT")
            self._ensure_column(conn, "episodes", "opponent_step", "INTEGER")
            self._ensure_column(conn, "training_stats", "run_id", "TEXT")
            self._ensure_column(conn, "training_stats", "aux_loss", "REAL")
            self._ensure_column(conn, "training_stats", "search_teacher_loss", "REAL")
            self._ensure_column(conn, "training_stats", "diversity_loss", "REAL")
            self._ensure_column(conn, "training_stats", "diversity_disc_tp_acc", "REAL")
            self._ensure_column(conn, "training_stats", "diversity_disc_ntp_acc", "REAL")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_run ON training_stats(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_opponent ON episodes(opponent_kind, opponent_preset)")

    def _ensure_column(self, conn, table, column, column_type):
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _recent_episode_cutoff_sql(self):
        if self.run_id:
            return """
                COALESCE(
                    (SELECT MAX(id) - ? FROM episodes WHERE run_id = ?), 0
                )
            """
        return "COALESCE((SELECT MAX(id) - ? FROM episodes), 0)"

    def record_episode(self, episode_data, training_step=None,
                       opponent_generation=None):
        """Record one finished episode and all of its turns."""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO episodes
                (timestamp, training_step, run_id, opponent_generation, agent_key,
                 winner, agent_won, total_turns, agent_persona_tp, agent_persona_ntp,
                 opponent_kind, opponent_preset, opponent_step)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                training_step,
                self.run_id,
                opponent_generation,
                episode_data.get("agent_key"),
                episode_data.get("winner"),
                int(episode_data.get("agent_won", False)),
                episode_data.get("total_turns", 0),
                episode_data.get("agent_persona_tp"),
                episode_data.get("agent_persona_ntp"),
                episode_data.get("opponent_kind"),
                episode_data.get("opponent_preset"),
                episode_data.get("opponent_step"),
            ))
            episode_id = cursor.lastrowid

            for turn in episode_data.get("turns", []):
                thumbs = turn.get("thumbs", {})
                conn.execute("""
                    INSERT INTO turns
                    (episode_id, turn_number, tp_key, skill, choice_target,
                     player_thumbs, computer_thumbs, reaction, agent_is_tp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    turn.get("turn", 0),
                    turn.get("tp_key"),
                    str(turn.get("skill", "")),
                    turn.get("choice_target"),
                    thumbs.get("player", 0),
                    thumbs.get("computer", 0),
                    turn.get("reaction"),
                    int(turn.get("agent_is_tp", False)),
                ))

            return episode_id

    def record_training_stats(self, step, stats):
        """Record aggregate training statistics."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO training_stats
                (timestamp, run_id, step, win_rate, avg_episode_length,
                 policy_loss, value_loss, entropy,
                 aux_reaction_loss, aux_thumbs_loss, aux_skill_loss,
                 aux_loss, search_teacher_loss, diversity_loss,
                 diversity_disc_tp_acc, diversity_disc_ntp_acc,
                 league_size, opponent_generation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                self.run_id,
                step,
                stats.get("win_rate"),
                stats.get("avg_episode_length"),
                stats.get("policy_loss"),
                stats.get("value_loss"),
                stats.get("entropy"),
                stats.get("aux_reaction_loss"),
                stats.get("aux_thumbs_loss"),
                stats.get("aux_skill_loss"),
                stats.get("aux_loss"),
                stats.get("search_teacher_loss"),
                stats.get("diversity_loss"),
                stats.get("diversity_disc_tp_acc"),
                stats.get("diversity_disc_ntp_acc"),
                stats.get("league_size"),
                stats.get("opponent_generation"),
            ))

    def get_win_rate_history(self, window=100):
        """Return win-rate history grouped by episode chunks."""
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute("""
                    SELECT training_step, agent_won
                    FROM episodes
                    WHERE training_step IS NOT NULL AND run_id = ?
                    ORDER BY training_step
                """, (self.run_id,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT training_step, agent_won
                    FROM episodes
                    WHERE training_step IS NOT NULL
                    ORDER BY training_step
                """).fetchall()

        results = []
        for i in range(0, len(rows), window):
            chunk = rows[i:i + window]
            if not chunk:
                continue
            wins = sum(r[1] for r in chunk)
            results.append({
                "step": chunk[-1][0],
                "win_rate": wins / len(chunk),
                "games": len(chunk),
            })
        return results

    def get_skill_usage_stats(self, last_n_episodes=500):
        """Return TP skill usage in recent episodes."""
        cutoff = self._recent_episode_cutoff_sql()
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute(f"""
                    SELECT t.skill, COUNT(*) as cnt,
                           COUNT(DISTINCT CASE WHEN e.agent_won = 1
                                 THEN t.episode_id ELSE NULL END) as win_episodes,
                           COUNT(DISTINCT t.episode_id) as episodes
                    FROM turns t
                    JOIN episodes e ON t.episode_id = e.id
                    WHERE t.agent_is_tp = 1
                    AND e.run_id = ?
                    AND t.episode_id > {cutoff}
                    GROUP BY t.skill
                    ORDER BY cnt DESC
                """, (self.run_id, last_n_episodes, self.run_id)).fetchall()
            else:
                rows = conn.execute(f"""
                    SELECT t.skill, COUNT(*) as cnt,
                           COUNT(DISTINCT CASE WHEN e.agent_won = 1
                                 THEN t.episode_id ELSE NULL END) as win_episodes,
                           COUNT(DISTINCT t.episode_id) as episodes
                    FROM turns t
                    JOIN episodes e ON t.episode_id = e.id
                    WHERE t.agent_is_tp = 1
                    AND t.episode_id > {cutoff}
                    GROUP BY t.skill
                    ORDER BY cnt DESC
                """, (last_n_episodes,)).fetchall()

        return [{
            "skill": skill,
            "usage_count": count,
            "win_episode_count": win_episodes,
            "episode_count": episodes,
            "win_rate": win_episodes / episodes if episodes > 0 else 0,
        } for skill, count, win_episodes, episodes in rows]

    def get_skill_opportunity_stats(self, last_n_episodes=500):
        """Return agent TP skill usage relative to legal opportunities.

        This replays recent episodes and counts each skill once per agent TP
        decision where at least one action for that skill was legal. Choice
        target-selection actions are collapsed to the Choice skill.
        """
        episodes = self._get_recent_episodes(last_n_episodes)
        if not episodes:
            return []

        episode_ids = [ep["id"] for ep in episodes]
        placeholders = ",".join("?" for _ in episode_ids)
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT episode_id, turn_number, tp_key, skill, choice_target,
                       player_thumbs, computer_thumbs, reaction, agent_is_tp
                FROM turns
                WHERE episode_id IN ({placeholders})
                ORDER BY episode_id, turn_number
            """, episode_ids).fetchall()

        turns_by_episode = {ep_id: [] for ep_id in episode_ids}
        for row in rows:
            turns_by_episode[row[0]].append(row)

        from collections import defaultdict
        from rl.actions import decode_action, get_action_mask, NUM_TP_ACTIONS
        from rl.env import YubisumaEnv, _create_game_state
        from yubisuma_constants import KEY_PLAYER, KEY_COMPUTER

        opportunity = defaultdict(int)
        usage = defaultdict(int)
        win_episodes = defaultdict(set)
        used_episodes = defaultdict(set)

        def normalize_skill(skill):
            if skill is None:
                return None
            if isinstance(skill, int):
                return str(skill)
            text = str(skill)
            if text.startswith("チョイス:"):
                return "チョイス"
            return text

        def parse_skill(skill):
            if skill is None:
                return None
            text = str(skill)
            if text.isdigit():
                return int(text)
            return text

        def to_int(value):
            if isinstance(value, bytes):
                return int.from_bytes(value, byteorder="little", signed=True)
            return int(value or 0)

        for ep in episodes:
            turns = turns_by_episode.get(ep["id"], [])
            if not turns:
                continue

            env = YubisumaEnv(opponent_policy=None)
            env.game_state = _create_game_state()
            env.game_state.current_player_key = turns[0][2]
            env.game_state.effects.first_player_key = turns[0][2]
            env.agent_key = ep["agent_key"]
            env.opponent_key = (
                KEY_COMPUTER if ep["agent_key"] == KEY_PLAYER else KEY_PLAYER
            )
            env._on_phase_start(env.game_state.current_player_key)

            try:
                for turn in turns:
                    (_, _, tp_key, skill, choice_target,
                     player_thumbs, computer_thumbs, reaction, agent_is_tp) = turn

                    if int(agent_is_tp):
                        mask = get_action_mask(env.game_state, ep["agent_key"])
                        legal_skills = set()
                        for action_idx, ok in enumerate(mask[:NUM_TP_ACTIONS]):
                            if not ok:
                                continue
                            decoded = decode_action(int(action_idx))
                            legal_skills.add(normalize_skill(decoded.get("skill")))
                        for legal in legal_skills:
                            if legal is not None:
                                opportunity[legal] += 1

                        used = normalize_skill(skill)
                        if used is not None:
                            usage[used] += 1
                            used_episodes[used].add(ep["id"])
                            if ep["agent_won"]:
                                win_episodes[used].add(ep["id"])

                    thumbs = {
                        KEY_PLAYER: to_int(player_thumbs),
                        KEY_COMPUTER: to_int(computer_thumbs),
                    }
                    env._resolve_turn_silent(
                        tp_key,
                        parse_skill(skill),
                        thumbs,
                        reaction,
                        choice_target,
                    )
                    env._finish_resolved_turn(tp_key, {
                        "turn": int(turn[1]),
                        "tp_key": tp_key,
                        "skill": parse_skill(skill),
                        "choice_target": choice_target,
                        "thumbs": thumbs,
                        "reaction": reaction,
                        "agent_is_tp": bool(agent_is_tp),
                    })
                    if env.game_state.game_over:
                        break
            finally:
                env.close()

        all_skills = sorted(set(opportunity) | set(usage), key=lambda s: (-usage[s], s))
        return [{
            "skill": skill,
            "usage_count": usage[skill],
            "opportunity_count": opportunity[skill],
            "opportunity_usage_rate": (
                usage[skill] / opportunity[skill] if opportunity[skill] > 0 else 0.0
            ),
            "win_episode_count": len(win_episodes[skill]),
            "episode_count": len(used_episodes[skill]),
            "win_rate": (
                len(win_episodes[skill]) / len(used_episodes[skill])
                if used_episodes[skill] else 0.0
            ),
        } for skill in all_skills]

    def _get_recent_episodes(self, last_n_episodes=500):
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute("""
                    SELECT id, agent_key, agent_won
                    FROM episodes
                    WHERE run_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (self.run_id, last_n_episodes)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, agent_key, agent_won
                    FROM episodes
                    ORDER BY id DESC
                    LIMIT ?
                """, (last_n_episodes,)).fetchall()

        rows.reverse()
        return [
            {"id": row[0], "agent_key": row[1], "agent_won": bool(row[2])}
            for row in rows
        ]

    def get_reaction_stats(self, last_n_episodes=500):
        """Return NTP reaction usage in recent episodes."""
        cutoff = self._recent_episode_cutoff_sql()
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute(f"""
                    SELECT t.reaction, COUNT(*) as cnt,
                           SUM(CASE WHEN e.agent_won = 1 THEN 1 ELSE 0 END) as wins
                    FROM turns t
                    JOIN episodes e ON t.episode_id = e.id
                    WHERE t.agent_is_tp = 0
                    AND e.run_id = ?
                    AND t.episode_id > {cutoff}
                    GROUP BY t.reaction
                    ORDER BY cnt DESC
                """, (self.run_id, last_n_episodes, self.run_id)).fetchall()
            else:
                rows = conn.execute(f"""
                    SELECT t.reaction, COUNT(*) as cnt,
                           SUM(CASE WHEN e.agent_won = 1 THEN 1 ELSE 0 END) as wins
                    FROM turns t
                    JOIN episodes e ON t.episode_id = e.id
                    WHERE t.agent_is_tp = 0
                    AND t.episode_id > {cutoff}
                    GROUP BY t.reaction
                    ORDER BY cnt DESC
                """, (last_n_episodes,)).fetchall()

        return [{
            "reaction": reaction or "なし",
            "count": count,
            "win_count": wins,
            "win_rate": wins / count if count > 0 else 0,
        } for reaction, count, wins in rows]

    def get_thumb_distribution(self, last_n_episodes=500):
        """Return agent thumb-count distribution in recent episodes."""
        cutoff = self._recent_episode_cutoff_sql()
        base_select = """
            SELECT
                CASE WHEN t.agent_is_tp = 1
                     THEN CASE WHEN t.tp_key = 'player'
                          THEN t.player_thumbs ELSE t.computer_thumbs END
                     ELSE CASE WHEN t.tp_key = 'player'
                          THEN t.computer_thumbs ELSE t.player_thumbs END
                END as agent_thumbs,
                COUNT(*) as cnt
            FROM turns t
        """

        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute(f"""
                    {base_select}
                    JOIN episodes e ON t.episode_id = e.id
                    WHERE e.run_id = ?
                    AND t.episode_id > {cutoff}
                    GROUP BY agent_thumbs
                    ORDER BY agent_thumbs
                """, (self.run_id, last_n_episodes, self.run_id)).fetchall()
            else:
                rows = conn.execute(f"""
                    {base_select}
                    WHERE t.episode_id > {cutoff}
                    GROUP BY agent_thumbs
                    ORDER BY agent_thumbs
                """, (last_n_episodes,)).fetchall()

        def to_int(value):
            if isinstance(value, bytes):
                return int.from_bytes(value, byteorder="little", signed=True)
            return int(value or 0)

        return [{"thumbs": to_int(t), "count": c} for t, c in rows]

    def get_episode_length_stats(self, last_n_episodes=500):
        """Return episode length statistics."""
        cutoff = self._recent_episode_cutoff_sql()
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute(f"""
                    SELECT total_turns, agent_won
                    FROM episodes
                    WHERE run_id = ?
                    AND id > {cutoff}
                """, (self.run_id, last_n_episodes, self.run_id)).fetchall()
            else:
                rows = conn.execute(f"""
                    SELECT total_turns, agent_won
                    FROM episodes
                    WHERE id > {cutoff}
                """, (last_n_episodes,)).fetchall()

        if not rows:
            return {}

        import statistics
        lengths = [r[0] for r in rows]
        wins = [r[0] for r in rows if r[1] == 1]
        losses = [r[0] for r in rows if r[1] == 0]
        return {
            "mean": statistics.mean(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            "win_mean": statistics.mean(wins) if wins else 0,
            "loss_mean": statistics.mean(losses) if losses else 0,
        }

    def get_opponent_stats(self, last_n_episodes=500):
        """Return win rate grouped by opponent kind/preset."""
        cutoff = self._recent_episode_cutoff_sql()
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute(f"""
                    SELECT COALESCE(opponent_kind, 'unknown') as kind,
                           COALESCE(opponent_preset, '') as preset,
                           COUNT(*) as games,
                           SUM(agent_won) as wins,
                           AVG(total_turns) as avg_turns
                    FROM episodes
                    WHERE run_id = ?
                    AND id > {cutoff}
                    GROUP BY kind, preset
                    ORDER BY games DESC, kind, preset
                """, (self.run_id, last_n_episodes, self.run_id)).fetchall()
            else:
                rows = conn.execute(f"""
                    SELECT COALESCE(opponent_kind, 'unknown') as kind,
                           COALESCE(opponent_preset, '') as preset,
                           COUNT(*) as games,
                           SUM(agent_won) as wins,
                           AVG(total_turns) as avg_turns
                    FROM episodes
                    WHERE id > {cutoff}
                    GROUP BY kind, preset
                    ORDER BY games DESC, kind, preset
                """, (last_n_episodes,)).fetchall()
        return [{
            "kind": kind,
            "preset": preset or "-",
            "games": games,
            "wins": wins or 0,
            "win_rate": (wins or 0) / games if games else 0.0,
            "avg_turns": avg_turns or 0.0,
        } for kind, preset, games, wins, avg_turns in rows]

    def get_persona_stats(self, last_n_episodes=500):
        """Return win rate grouped by sampled agent persona."""
        cutoff = self._recent_episode_cutoff_sql()
        with self._connect() as conn:
            if self.run_id:
                rows = conn.execute(f"""
                    SELECT agent_persona_tp, agent_persona_ntp,
                           COUNT(*) as games,
                           SUM(agent_won) as wins,
                           AVG(total_turns) as avg_turns
                    FROM episodes
                    WHERE run_id = ?
                    AND id > {cutoff}
                    GROUP BY agent_persona_tp, agent_persona_ntp
                    ORDER BY games DESC, agent_persona_tp, agent_persona_ntp
                """, (self.run_id, last_n_episodes, self.run_id)).fetchall()
            else:
                rows = conn.execute(f"""
                    SELECT agent_persona_tp, agent_persona_ntp,
                           COUNT(*) as games,
                           SUM(agent_won) as wins,
                           AVG(total_turns) as avg_turns
                    FROM episodes
                    WHERE id > {cutoff}
                    GROUP BY agent_persona_tp, agent_persona_ntp
                    ORDER BY games DESC, agent_persona_tp, agent_persona_ntp
                """, (last_n_episodes,)).fetchall()
        return [{
            "persona_tp": tp,
            "persona_ntp": ntp,
            "games": games,
            "wins": wins or 0,
            "win_rate": (wins or 0) / games if games else 0.0,
            "avg_turns": avg_turns or 0.0,
        } for tp, ntp, games, wins, avg_turns in rows]

    def get_summary(self):
        """Return summary statistics."""
        with self._connect() as conn:
            if self.run_id:
                total = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE run_id = ?",
                    (self.run_id,),
                ).fetchone()[0]
                wins = conn.execute("""
                    SELECT COUNT(*) FROM episodes
                    WHERE agent_won = 1 AND run_id = ?
                """, (self.run_id,)).fetchone()[0]
                recent = conn.execute("""
                    SELECT COUNT(*), SUM(agent_won) FROM episodes
                    WHERE run_id = ?
                    AND id > COALESCE(
                        (SELECT MAX(id) - 100 FROM episodes WHERE run_id = ?), 0
                    )
                """, (self.run_id, self.run_id)).fetchone()
            else:
                total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                wins = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE agent_won = 1"
                ).fetchone()[0]
                recent = conn.execute("""
                    SELECT COUNT(*), SUM(agent_won) FROM episodes
                    WHERE id > COALESCE((SELECT MAX(id) - 100 FROM episodes), 0)
                """).fetchone()

        return {
            "total_episodes": total,
            "total_wins": wins,
            "overall_win_rate": wins / total if total > 0 else 0,
            "recent_100_games": recent[0] or 0,
            "recent_100_wins": recent[1] or 0,
            "recent_100_win_rate": (
                (recent[1] or 0) / recent[0] if recent[0] and recent[0] > 0 else 0
            ),
        }

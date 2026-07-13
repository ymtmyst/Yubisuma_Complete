"""ブラウザで遊べるユビスマ完全ルール版。

外部パッケージなしで起動できる小さなHTTPサーバー。ゲームの解決は既存の
GameState / TurnHandlerを利用し、ここでは画面向けの入出力だけを担当する。
"""

from __future__ import annotations

import argparse
import io
import json
import random
import threading
import uuid
import webbrowser
from contextlib import redirect_stdout
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from yubisuma_constants import GAME_CONFIG, KEY_COMPUTER, KEY_PLAYER, ULTIMATE_SKILLS
from yubisuma_logic import GameState, computer_strategy, get_valid_skills
from yubisuma_turn_handler import TurnHandler


WEB_DIR = Path(__file__).with_name("web")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and line.strip() != "=== 結果表示 ==="]


class WebGame:
    """1ブラウザセッション分の、入力待ち状態を持つゲーム進行役。"""

    def __init__(self) -> None:
        self.gs = GameState()
        self.mode = "attack"
        self.phase_was_skip = False
        self.pending_ai: dict | None = None
        self.pending_choice: dict | None = None
        self.log: list[str] = []
        with redirect_stdout(io.StringIO()) as out:
            self.gs.initialize_game()
        self.log.extend(_lines(out.getvalue()))
        self._begin_phase(self.gs.current_player_key)

    def _begin_phase(self, key: str) -> None:
        self.gs.current_player_key = key
        self.gs.effects.phase_count += 1
        self.gs.on_phase_start(key)
        player = self.gs.get_player(key)
        self.phase_was_skip = player.skip_phases > 0
        if self.phase_was_skip:
            self.log.append(f"{player.name}はスキップ効果でこのフェーズを休みます")
            self._finish_phase(key)
            return
        self._prepare_prompt()

    def _prepare_prompt(self) -> None:
        if self.gs.game_over:
            self.mode = "gameover"
            return
        key = self.gs.current_player_key
        if key == KEY_PLAYER:
            self.mode = "attack"
            self.pending_ai = None
        else:
            self.mode = "defense"
            comp = self.gs.computer
            player = self.gs.player
            valid = get_valid_skills(self.gs, KEY_COMPUTER)
            total_possible = comp.get_active_hands() + player.get_active_hands()
            action = random.choice(list(range(total_possible + 1)) + valid)
            thumbs = computer_strategy(comp.get_active_hands(), cement_min=comp.cement)
            choice_data = None
            if action == "チョイス":
                available = [s for s in comp.stock if s not in comp.choice_used_this_phase]
                choice_data = {"choice": random.choice(available)} if available else {"choice": None}
            elif action == "オール":
                order = list(comp.stock)
                random.shuffle(order)
                choice_data = {"all_order": order}
            self.pending_ai = {"action": action, "thumbs": thumbs, "choice_data": choice_data}

    def _finish_phase(self, key: str) -> None:
        self.gs.on_phase_end(key)
        current = self.gs.get_player(key)
        if self.phase_was_skip and current.time_active:
            current.time_active = False
            next_key = key
            self.log.append(f"タイム発動！ {current.name}が続けて行動します")
        else:
            next_key = self.gs.get_opponent_key(key)
        self._begin_phase(next_key)

    def _after_turn(self, key: str) -> None:
        opponent = self.gs.get_opponent(key)
        if opponent.time_active and self.gs.effects.has_extra_turn(key):
            opponent.time_active = False
            lost = self.gs.effects.additional_turns[key]
            self.gs.effects.additional_turns[key] = 0
            self.log.append(f"タイム発動！ 追加{lost}ターンを止めました")
            self._finish_phase(key)
        elif self.gs.effects.has_extra_turn(key):
            self.gs.effects.use_extra_turn(key)
            self.log.append(f"{self.gs.get_player(key).name}の追加ターン！")
            self._prepare_prompt()
        else:
            self._finish_phase(key)

    def _resolve(self, key: str, action, thumbs: dict, reaction, choice_data=None) -> None:
        self.gs.effects.turns_in_current_phase += 1
        with redirect_stdout(io.StringIO()) as out:
            TurnHandler.resolve_turn(self.gs, key, action, thumbs, reaction, choice_data)
            won = self.gs.check_victory()
        self.log.extend(_lines(out.getvalue()))
        if won:
            self.mode = "gameover"
            self.log.append(f"ゲーム終了 — {self.gs.get_player(self.gs.winner).name}の勝利！")
        else:
            self._after_turn(key)

    @staticmethod
    def _parse_action(value):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return value

    def submit(self, data: dict) -> None:
        if self.gs.game_over:
            raise ValueError("ゲームは終了しています")

        thumbs_value = int(data.get("thumbs", -1))
        if self.mode == "attack":
            action = self._parse_action(data.get("action"))
            legal = list(range(self.gs.player.get_active_hands() + self.gs.computer.get_active_hands() + 1))
            legal += get_valid_skills(self.gs, KEY_PLAYER)
            if action not in legal:
                raise ValueError("現在は選べない宣言です")
            self._validate_thumbs(self.gs.player, thumbs_value)
            reaction = self._computer_reaction()
            comp_thumbs = computer_strategy(self.gs.computer.get_active_hands(), cement_min=self.gs.computer.cement)
            thumbs = {KEY_PLAYER: thumbs_value, KEY_COMPUTER: comp_thumbs}
            if action in ("チョイス", "オール"):
                self.pending_choice = {"action": action, "thumbs": thumbs, "reaction": reaction}
                self.mode = "choice"
                return
            self._resolve(KEY_PLAYER, action, thumbs, reaction)
            return

        if self.mode == "choice":
            if not self.pending_choice:
                raise ValueError("選択待ちの宣言がありません")
            pending = self.pending_choice
            stock = list(self.gs.player.stock)
            if pending["action"] == "チョイス":
                choice = data.get("choice")
                if choice not in stock:
                    raise ValueError("ストックから1つ選んでください")
                choice_data = {"choice": choice}
            else:
                order = data.get("order")
                if not isinstance(order, list) or sorted(order) != sorted(stock):
                    raise ValueError("すべてのストックの順番を指定してください")
                choice_data = {"all_order": order}
            self.pending_choice = None
            self._resolve(KEY_PLAYER, pending["action"], pending["thumbs"], pending["reaction"], choice_data)
            return

        if self.mode == "defense":
            self._validate_thumbs(self.gs.player, thumbs_value)
            reaction = data.get("reaction") or None
            if reaction not in self._legal_reactions():
                raise ValueError("現在は選べないリアクションです")
            pending = self.pending_ai
            if not pending:
                raise ValueError("コンピューターの宣言がありません")
            thumbs = {KEY_PLAYER: thumbs_value, KEY_COMPUTER: pending["thumbs"]}
            self.pending_ai = None
            self._resolve(KEY_COMPUTER, pending["action"], thumbs, reaction, pending["choice_data"])
            return

        raise ValueError("現在は入力を受け付けていません")

    @staticmethod
    def _validate_thumbs(player, value: int) -> None:
        minimum = player.cement or 0
        if not minimum <= value <= player.get_active_hands():
            raise ValueError(f"親指は{minimum}〜{player.get_active_hands()}本から選んでください")

    def _legal_reactions(self) -> list:
        player = self.gs.player
        result = [None]
        if not player.lock_active:
            result.append("カウンター")
        if not player.used_ultimate:
            result.append("ブロック")
        if GAME_CONFIG["ENABLE_MIRROR"] and player.mirror_ready and not player.lock_active:
            result.append("ミラー")
        return result

    def _computer_reaction(self):
        comp = self.gs.computer
        choices = [None, None]
        if not comp.lock_active:
            choices.append("カウンター")
        if not comp.used_ultimate:
            choices.append("ブロック")
        if GAME_CONFIG["ENABLE_MIRROR"] and comp.mirror_ready and not comp.lock_active:
            choices.append("ミラー")
        return random.choice(choices)

    @staticmethod
    def _player_data(player) -> dict:
        effects = []
        if player.guard_active: effects.append({"icon": "shield", "label": "ガード"})
        if player.charge_active: effects.append({"icon": "bolt", "label": "チャージ"})
        if player.quick_level: effects.append({"icon": "wind", "label": f"クイック {player.quick_level}"})
        if player.mirror_ready: effects.append({"icon": "mirror", "label": "ミラー"})
        if player.lock_active or player.lock_pending: effects.append({"icon": "lock", "label": "ロック"})
        if player.skip_phases: effects.append({"icon": "skip", "label": f"スキップ {player.skip_phases}"})
        if player.time_active: effects.append({"icon": "clock", "label": "タイム"})
        if player.cement is not None: effects.append({"icon": "brick", "label": f"セメント {player.cement}"})
        return {
            "name": player.name,
            "hands": player.get_active_hands(),
            "left": player.left_hand,
            "right": player.right_hand,
            "effects": effects,
            "stock": list(player.stock),
            "ultimate_used": player.used_ultimate,
            "declared": player.has_declared_skill,
        }

    def state(self) -> dict:
        total_hands = self.gs.player.get_active_hands() + self.gs.computer.get_active_hands()
        valid = get_valid_skills(self.gs, KEY_PLAYER) if self.mode == "attack" else []
        choice = None
        if self.mode == "choice" and self.pending_choice:
            choice = {
                "type": self.pending_choice["action"],
                "stock": list(self.gs.player.stock),
                "reaction": self.pending_choice["reaction"],
            }
        return {
            "mode": self.mode,
            "turn": self.gs.current_player_key,
            "phase": self.gs.effects.phase_count,
            "player": self._player_data(self.gs.player),
            "computer": self._player_data(self.gs.computer),
            "numbers": list(range(total_hands + 1)),
            "skills": valid,
            "reactions": [r or "なし" for r in self._legal_reactions()] if self.mode == "defense" else [],
            "thumb_min": self.gs.player.cement or 0,
            "choice": choice,
            "winner": self.gs.winner,
            "log": self.log[-30:],
        }


class GameHandler(SimpleHTTPRequestHandler):
    sessions: dict[str, WebGame] = {}
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _session_id(self) -> str | None:
        cookie = self.headers.get("Cookie", "")
        for item in cookie.split(";"):
            key, _, value = item.strip().partition("=")
            if key == "yubisuma_session":
                return value
        return None

    def _game(self, create=True) -> tuple[str, WebGame | None]:
        sid = self._session_id()
        with self.lock:
            game = self.sessions.get(sid) if sid else None
            if game is None and create:
                sid = uuid.uuid4().hex
                game = WebGame()
                self.sessions[sid] = game
        return sid, game

    def _json(self, payload, status=HTTPStatus.OK, sid=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if sid:
            self.send_header("Set-Cookie", f"yubisuma_session={sid}; Path=/; SameSite=Lax")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/api/state":
            sid, game = self._game()
            self._json(game.state(), sid=sid)
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
            sid, game = self._game()
            if path == "/api/new":
                game = WebGame()
                with self.lock:
                    self.sessions[sid] = game
            elif path == "/api/action":
                game.submit(data)
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json(game.state(), sid=sid)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="ユビスマのブラウザ版を起動します")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), GameHandler)
    url = f"http://{args.host}:{args.port}"
    # Windowsでstdoutがcp1252等でも起動を妨げないASCIIメッセージにする。
    print(f"YUBISUMA started: {url}")
    print("Press Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Visual browser game vs the trained AI (correct rules).

A small stdlib HTTP server that lets you play Yubisuma Complete (mirror/
reversi OFF) against the current value-net + LP-search agent, using the
CORRECT rules engine (complete_solver.transition — the one fixed and tested
in 2026-07-13). The AI's moves come from the same SearchAgent used in
evaluation; it samples from the LP equilibrium mixture, so it plays the real
deployed policy.

Design for accessibility: the human picks in two easy steps — WHAT to declare
(a number or a skill), then HOW MANY thumbs to raise. Simultaneity is kept
honest: when you declare, the AI's reaction is committed without seeing your
choice; when the AI declares, it commits first and you react before the
reveal.

Run:  python -m complete_ai.play_server   (opens the browser automatically)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from complete_solver.actions import (
    RulesConfig,
    TPAction,
    NTPAction,
    legal_ntp_actions,
    legal_tp_actions,
)
from complete_solver.state import State, initial_state
from complete_solver.transition import transition
from complete_solver.packed_engine import pack_state

from .agents import SearchAgent
from .batched_search import BatchedSearcher
from .generation_loop import load_model

CONFIG = RulesConfig(enable_mirror=False, enable_reversi=False)


def _bundle_root() -> Path:
    """Root for bundled assets — works in dev and in a PyInstaller build."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else Path(__file__).resolve().parent


WEB_DIR = _bundle_root() / "webplay" if getattr(sys, "_MEIPASS", None) \
    else Path(__file__).with_name("webplay")


def _default_model_path() -> str:
    """Find the model in dev (models/…) or beside the exe / in the bundle."""
    candidates = [
        Path("models/value_latest.pt"),
        _bundle_root() / "models" / "value_latest.pt",
        Path(sys.argv[0]).resolve().parent / "models" / "value_latest.pt",
    ]
    for c in candidates:
        if Path(c).is_file():
            return str(c)
    return "models/value_latest.pt"

# One-line child-friendly descriptions + emoji per skill.
SKILL_INFO = {
    "フラッシュ": ("✨", "相手と同じ指の数なら手を2つ降ろせる大技"),
    "セメント": ("🧱", "上げた指を固めて下げられなくする"),
    "ガード": ("🛡️", "相手の2つ降ろしを1回防ぎ、もう1ターン動ける"),
    "チャージ": ("⚡", "次の数字が2回分になる"),
    "クイック": ("💨", "続けて使うと手を降ろせる"),
    "スキップ": ("⏭️", "相手の次の番を飛ばす"),
    "フェイント": ("🎭", "相手がカウンターした時だけ、手を1つ降ろして追加ターン"),
    "ロック": ("🔒", "相手がカウンターした時だけ、相手の反応を封じる"),
    "コピー": ("📋", "直前に出たスキルを2回分マネする"),
    "ストック": ("📦", "直前のスキルをためる(2つで強力)"),
    "チョイス": ("🎯", "ためたスキルから1つ選んで使う"),
    "オール": ("🎆", "ためたスキルを全部いっきに出す"),
    "ドロップ": ("🗑️", "相手が使えないスキルを作り、追加ターン"),
    "ブースト": ("🚀", "必殺技:追加3ターン(1ゲーム1回)"),
    "タイム": ("⏳", "必殺技:相手の連続行動を奪う(1ゲーム1回)"),
}
REACTION_INFO = {
    "なし": ("🙅", "反応しない"),
    "カウンター": ("↩️", "相手のスキルを跳ね返す"),
    "ブロック": ("🚫", "必殺技:相手のスキルを無効化(1ゲーム1回)"),
}


def _player_view(p) -> dict:
    buffs = []
    if p.guard_active:
        buffs.append("🛡️ガード")
    if p.charge_active:
        buffs.append("⚡チャージ")
    if p.quick_level:
        buffs.append(f"💨クイック{p.quick_level}")
    if p.lock_pending or p.lock_active:
        buffs.append("🔒ロック中")
    if p.time_active:
        buffs.append("⏳タイム")
    if p.skip_phases:
        buffs.append(f"⏭️スキップ{p.skip_phases}")
    return {
        "hands": p.hands,
        "cement": p.cement,
        "buffs": buffs,
        "stock": sorted(p.stock),
        "ultimate_used": p.used_ultimate,
    }


class WebGame:
    # Difficulty = how much random handicap the AI plays with (0 = full
    # LP-equilibrium strength). Easy adds frequent random moves for kids.
    DIFFICULTY_EPSILON = {"easy": 0.45, "normal": 0.12, "hard": 0.0}

    def __init__(self, searcher: BatchedSearcher, human_first: bool = True,
                 difficulty: str = "normal"):
        self.searcher = searcher
        eps = self.DIFFICULTY_EPSILON.get(difficulty, 0.12)
        self.agent = SearchAgent(searcher, np.random.default_rng(), epsilon=eps)
        self.difficulty = difficulty
        self.state = initial_state()
        self.human_is_mover = human_first
        self.committed_ai_tp: TPAction | None = None
        self.entries: list[dict] = []          # structured exchange log
        self.last_exchange: dict | None = None  # most recent, shown big
        self.phase = {"human": 0, "ai": 0}
        self.turn = {"human": 0, "ai": 0}
        self.last_mover: str | None = None
        self.over = False
        self.human_won: bool | None = None

    # ── declaration / reaction → emoji+label ───────────────────────────────
    @staticmethod
    def _tp_badge(tp: TPAction) -> dict:
        if isinstance(tp.skill, int):
            return {"emoji": "🔢", "name": f"数字{tp.skill}", "thumb": tp.thumb}
        if tp.skill == "チョイス":
            emoji = SKILL_INFO.get(tp.choice, ("🎯", ""))[0]
            return {"emoji": emoji, "name": f"選ぶ→{tp.choice}", "thumb": tp.thumb}
        emoji = SKILL_INFO.get(tp.skill, ("❓", ""))[0]
        return {"emoji": emoji, "name": tp.skill, "thumb": tp.thumb}

    @staticmethod
    def _ntp_badge(ntp: NTPAction) -> dict:
        emoji = REACTION_INFO.get(ntp.reaction, ("❓", ""))[0]
        return {"emoji": emoji, "name": ntp.reaction, "thumb": ntp.thumb}

    def _bump_phase_turn(self, mover: str) -> None:
        if mover != self.last_mover:
            self.phase[mover] += 1
            self.turn[mover] = 1
        else:
            self.turn[mover] += 1
        self.last_mover = mover

    # ── perspective helpers ────────────────────────────────────────────────
    def _human(self):
        return self.state.me if self.human_is_mover else self.state.opp

    def _ai(self):
        return self.state.opp if self.human_is_mover else self.state.me

    def _decode_ai_tp(self) -> TPAction:
        lane0, lane1 = pack_state(self.state)
        code = self.agent.tp_action(int(lane0), int(lane1))
        from complete_solver.packed_engine import code_to_tp_action
        return code_to_tp_action(code)

    def _ai_ntp(self) -> NTPAction:
        lane0, lane1 = pack_state(self.state)
        code = self.agent.ntp_action(int(lane0), int(lane1))
        from complete_solver.packed_engine import code_to_ntp_action
        return code_to_ntp_action(code)

    # ── options for the human ──────────────────────────────────────────────
    def _human_tp_options(self) -> dict:
        # Names/values only — the front-end owns emoji, descriptions and the
        # kid/adult text (skills_data.js), so toggles need no server round trip.
        actions = legal_tp_actions(self.state, CONFIG)
        numbers = sorted({a.skill for a in actions if isinstance(a.skill, int)})
        skills = []
        for a in actions:
            if isinstance(a.skill, str) and a.skill != "チョイス" and a.skill not in skills:
                skills.append(a.skill)
        choices = sorted({a.choice for a in actions if a.skill == "チョイス" and a.choice})
        thumbs = sorted({a.thumb for a in actions})
        return {"numbers": numbers, "skills": skills, "choices": choices,
                "thumbs": thumbs, "max_hands": self._human().hands}

    def _human_ntp_options(self) -> dict:
        actions = legal_ntp_actions(self.state, CONFIG)
        reactions = [r for r in ["なし", "カウンター", "ブロック"]
                     if any(a.reaction == r for a in actions)]
        thumbs = sorted({a.thumb for a in actions})
        return {"reactions": reactions, "thumbs": thumbs,
                "max_hands": self._human().hands}

    # ── view ───────────────────────────────────────────────────────────────
    def view(self) -> dict:
        v = {
            "human": _player_view(self._human()),
            "ai": _player_view(self._ai()),
            "last_exchange": self.last_exchange,
            "entries": self.entries[-30:],
            "over": self.over,
        }
        if self.over:
            v["phase"] = "over"
            v["human_won"] = self.human_won
            v["ai_advantage"] = -1.0 if self.human_won else 1.0
            return v
        # AI confidence meter: the game value from the AI's perspective in
        # [-1, 1] (positive = AI ahead). One extra (cached) search.
        lane0, lane1 = pack_state(self.state)
        val, _, _, _, _ = self.searcher.solve(int(lane0), int(lane1))
        v["ai_advantage"] = float(val if not self.human_is_mover else -val)
        if self.human_is_mover:
            v["phase"] = "declare"
            v["options"] = self._human_tp_options()
        else:
            # AI commits its declaration now (hidden); human reacts.
            if self.committed_ai_tp is None:
                self.committed_ai_tp = self._decode_ai_tp()
            v["phase"] = "react"
            v["options"] = self._human_ntp_options()
        return v

    # ── action resolution ──────────────────────────────────────────────────
    def act(self, payload: dict) -> None:
        if self.over:
            return
        if payload.get("surrender"):
            self.over = True
            self.human_won = False
            return
        mover = "human" if self.human_is_mover else "ai"
        if self.human_is_mover:
            tp = self._build_tp(payload)
            ntp = self._ai_ntp()
        else:
            tp = self.committed_ai_tp
            ntp = self._build_ntp(payload)

        self._bump_phase_turn(mover)
        entry = {
            "side": mover,
            "phase": self.phase[mover],
            "turn": self.turn[mover],
            "decl": self._tp_badge(tp),
            "react": self._ntp_badge(ntp),
        }
        self.entries.append(entry)
        self.last_exchange = entry

        result = transition(self.state, tp, ntp, CONFIG)

        if result.terminal_reward is not None:
            mover_won = result.terminal_reward > 0
            self.human_won = (mover_won == self.human_is_mover)
            self.over = True
            return

        self.state = result.next_state
        if not result.same_turn_player:
            self.human_is_mover = not self.human_is_mover
        self.committed_ai_tp = None

        # If it's now the AI's continuous turn (AI is mover), auto-play AI
        # declarations until the turn returns to the human or the game ends.
        self._autoplay_ai_solo()

    def _autoplay_ai_solo(self) -> None:
        guard = 0
        while (not self.over) and (not self.human_is_mover) and guard < 40:
            guard += 1
            # AI is mover but the human still reacts — so we only auto-play
            # when there is genuinely no human decision. There always is a
            # reaction, so we stop here and let the human react.
            return


class Handler(BaseHTTPRequestHandler):
    games: dict[str, WebGame] = {}
    searcher: BatchedSearcher = None  # set at startup

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        f = WEB_DIR / path.lstrip("/")
        if f.is_file() and WEB_DIR in f.resolve().parents:
            ctype = {"html": "text/html", "css": "text/css",
                     "js": "application/javascript"}.get(f.suffix[1:], "text/plain")
            self._send(HTTPStatus.OK, f.read_bytes(), ctype)
        else:
            self._send(HTTPStatus.NOT_FOUND, "not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or "{}")
        if self.path == "/api/new":
            gid = uuid.uuid4().hex
            self.games[gid] = WebGame(
                self.searcher,
                human_first=payload.get("human_first", True),
                difficulty=payload.get("difficulty", "normal"),
            )
            self._send(HTTPStatus.OK,
                       json.dumps({"game_id": gid, "view": self.games[gid].view()}))
        elif self.path == "/api/act":
            gid = payload.get("game_id")
            game = self.games.get(gid)
            if game is None:
                self._send(HTTPStatus.NOT_FOUND, json.dumps({"error": "no game"}))
                return
            game.act(payload.get("action", {}))
            self._send(HTTPStatus.OK, json.dumps({"view": game.view()}))
        else:
            self._send(HTTPStatus.NOT_FOUND, json.dumps({"error": "unknown"}))


# TP/NTP builders on WebGame (defined here to keep the class readable).
def _build_tp(self: WebGame, payload: dict) -> TPAction:
    kind = payload.get("kind")
    thumb = int(payload.get("thumb", 0))
    if kind == "number":
        return TPAction(int(payload["value"]), thumb)
    if kind == "choice":
        return TPAction("チョイス", thumb, choice=payload["target"])
    return TPAction(payload["name"], thumb)


def _build_ntp(self: WebGame, payload: dict) -> NTPAction:
    return NTPAction(payload.get("name", "なし"), int(payload.get("thumb", 0)))


def _tp_label(self: WebGame, tp: TPAction) -> str:
    if isinstance(tp.skill, int):
        return f"数字{tp.skill}(親指{tp.thumb})"
    if tp.skill == "チョイス":
        return f"チョイス→{tp.choice}(親指{tp.thumb})"
    return f"{tp.skill}(親指{tp.thumb})"


WebGame._build_tp = _build_tp
WebGame._build_ntp = _build_ntp
WebGame._tp_label = _tp_label


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Japanese prints on cmd
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=_default_model_path())
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("AIを読み込み中...", flush=True)
    model = load_model(Path(args.model), device)
    Handler.searcher = BatchedSearcher(model, device, prune_stock=True)
    # Warm up the compiled kernels / model once.
    Handler.searcher.solve(*[int(x) for x in pack_state(initial_state())])
    print("準備完了。", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"対戦画面: {url}  (終了は Ctrl+C)", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了しました。")


if __name__ == "__main__":
    main()

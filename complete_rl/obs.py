"""Observation encoding for the Complete RL environment.

The public game state is encoded as a flat float32 array so that standard
neural networks can consume it directly.  All values are normalised to [0, 1].

Layout
------
[0  : N_PLAYER)                    me   (current turn player)
[N_PLAYER : 2*N_PLAYER)            opp  (non-turn player)
[2*N_PLAYER : 2*N_PLAYER+N_STATE)  state-level features
[2*N_PLAYER+N_STATE : OBS_SIZE)    opponent reaction history (most recent first)
"""

from __future__ import annotations

import numpy as np

from complete_solver.constants import (
    ANTI_COUNTER_SKILLS,
    NORMAL_SKILLS,
    REACTIONS,
    REFERENCE_SKILLS,
    REFERENCEABLE_SKILLS,
    ULTIMATE_TP_SKILLS,
)
from complete_solver.state import PlayerState, State

# ── Fixed skill orderings ─────────────────────────────────────────────────
# Always encode all 9 referenceable skills (MIRROR_PREP is 0 when mirror is OFF).
_REFERENCEABLE: tuple[str, ...] = tuple(sorted(REFERENCEABLE_SKILLS))

# All possible values of State.previous_skill:
#   None | integer total 0-4 | skill name (17 skills)
_PREV_NUMS: list[int] = list(range(5))
_PREV_SKILLS: list[str] = sorted(
    NORMAL_SKILLS | ANTI_COUNTER_SKILLS | REFERENCE_SKILLS | ULTIMATE_TP_SKILLS
)
_PREV_ALL: list = [None] + _PREV_NUMS + _PREV_SKILLS   # length 23
_PREV_INDEX: dict = {v: i for i, v in enumerate(_PREV_ALL)}

# ── Reaction history encoding ─────────────────────────────────────────────
# Track last N_REACTION_HISTORY opponent reactions as per-slot one-hot vectors.
# Slots are ordered most-recent first; all zeros = no reaction yet.
_REACTION_INDEX: dict[str, int] = {r: i for i, r in enumerate(REACTIONS)}
N_REACTIONS: int = len(REACTIONS)         # 4  (なし・カウンター・ブロック・ミラー)
N_REACTION_HISTORY: int = 4              # number of past reactions to encode
N_REACTION_FEATURES: int = N_REACTION_HISTORY * N_REACTIONS   # 16

# ── Dimension constants ───────────────────────────────────────────────────
N_REFERENCEABLE: int = len(_REFERENCEABLE)       # 9
N_PREV: int = len(_PREV_ALL)                     # 23

# Per-player: 13 scalar/bool features + 3 × N_REFERENCEABLE bitmasks
N_PLAYER: int = 13 + N_REFERENCEABLE * 3         # 40

# State-level: previous_skill one-hot + 4 scalar/bool features
N_STATE: int = N_PREV + 4                        # 27

OBS_SIZE: int = N_PLAYER * 2 + N_STATE + N_REACTION_FEATURES  # 123


# ── Encoding helpers ──────────────────────────────────────────────────────

def _encode_player(ps: PlayerState, buf: np.ndarray, offset: int) -> int:
    """Write per-player features into *buf* starting at *offset*.

    Returns the next free index.
    """
    buf[offset]      = ps.hands / 2.0
    buf[offset + 1]  = ps.cement / 2.0
    buf[offset + 2]  = float(ps.guard_active)
    buf[offset + 3]  = float(ps.charge_active)
    buf[offset + 4]  = ps.quick_level / 2.0
    buf[offset + 5]  = float(ps.mirror_ready)
    buf[offset + 6]  = float(ps.lock_pending)
    buf[offset + 7]  = float(ps.lock_active)
    buf[offset + 8]  = min(ps.skip_phases, 2) / 2.0
    buf[offset + 9]  = float(ps.used_ultimate)
    buf[offset + 10] = float(ps.time_active)
    buf[offset + 11] = float(ps.has_declared_skill)
    buf[offset + 12] = float(ps.stock_alpha_used_this_phase)

    base = offset + 13
    for i, skill in enumerate(_REFERENCEABLE):
        buf[base + i] = float(skill in ps.stock)
    base += N_REFERENCEABLE

    for i, skill in enumerate(_REFERENCEABLE):
        buf[base + i] = float(skill in ps.drop_blocked_skills)
    base += N_REFERENCEABLE

    for i, skill in enumerate(_REFERENCEABLE):
        buf[base + i] = float(skill in ps.choice_used_this_phase)

    return offset + N_PLAYER


def encode_state(
    state: State,
    reaction_history: tuple[str, ...] = (),
) -> np.ndarray:
    """Encode a public game state into a flat float32 observation vector.

    Parameters
    ----------
    state:
        Current game state.
    reaction_history:
        Most-recent-first tuple of the opponent's last N_REACTION_HISTORY
        reaction strings (e.g., ``("なし", "カウンター", "なし", "なし")``).
        Shorter or empty tuples are zero-padded at the end (= oldest slots).
    """
    buf = np.zeros(OBS_SIZE, dtype=np.float32)

    idx = _encode_player(state.me, buf, 0)
    idx = _encode_player(state.opp, buf, idx)

    # previous_skill one-hot
    prev_idx = _PREV_INDEX.get(state.previous_skill, 0)  # unknown → None slot
    buf[idx + prev_idx] = 1.0
    idx += N_PREV

    buf[idx]     = min(state.me_extra_turns, 3) / 3.0
    buf[idx + 1] = min(state.opp_extra_turns, 3) / 3.0
    buf[idx + 2] = float(state.me_guard_extra_used_this_phase)
    buf[idx + 3] = float(state.opp_guard_extra_used_this_phase)
    idx += 4

    # Opponent reaction history: N_REACTION_HISTORY slots × N_REACTIONS one-hot
    for slot, reaction in enumerate(reaction_history[:N_REACTION_HISTORY]):
        r_idx = _REACTION_INDEX.get(reaction)
        if r_idx is not None:
            buf[idx + slot * N_REACTIONS + r_idx] = 1.0

    return buf

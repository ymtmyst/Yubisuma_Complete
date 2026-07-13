"""Value-network feature extraction (N3).

Two implementations with identical output, differentially tested:
- ``features_from_state``  — reference, from a solver ``State``.
- ``features_from_lanes``  — Numba batch version, from packed int64 lanes
                             (used for high-throughput dataset generation).

Layout (FEATURE_SIZE = 103):
  per player (39 dims × 2, me then opp):
    0  hands/2          1  cement/2       2  guard        3  charge
    4  quick/2          5  lock_pending   6  lock_active  7  min(skip,3)/3
    8  used_ultimate    9  time_active   10  declared    11  stock_alpha_used
    12..19  stock one-hot (packed skill ids 0..7)
    20      |stock|/4 (capped at 4)
    21      has anti-counter stocked (feint/lock)
    22      has normal skill stocked
    23..30  choice_used one-hot
    31..38  drop_blocked one-hot
  state level (25 dims):
    78..98  previous_skill one-hot (packed prev codes 0..20)
    99      min(me_extra_turns, 7)/7
    100     min(opp_extra_turns, 7)/7
    101     me_guard_extra_used
    102     opp_guard_extra_used

Domain notes baked in (user knowledge, 2026-07-12): stock strength peaks at
2 held and the anti-counter/normal composition matters, hence the explicit
count and composition flags on top of the raw one-hot.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from complete_solver.packed_engine import pack_state
from complete_solver.state import State

N_PLAYER = 39
N_PREV = 21
FEATURE_SIZE = N_PLAYER * 2 + N_PREV + 4  # 103

_ANTI_COUNTER_MASK = (1 << 6) | (1 << 7)  # feint, lock (packed skill ids)
_NORMAL_MASK = 0x3F  # ids 0..5


@njit(cache=True, inline="always")
def _player_features(bits, out, base):
    out[base + 0] = (bits & 3) / 2.0
    out[base + 1] = (bits >> 2 & 3) / 2.0
    out[base + 2] = float(bits >> 4 & 1)
    out[base + 3] = float(bits >> 5 & 1)
    out[base + 4] = (bits >> 6 & 3) / 2.0
    out[base + 5] = float(bits >> 8 & 1)
    out[base + 6] = float(bits >> 9 & 1)
    skip = bits >> 10 & 15
    if skip > 3:
        skip = 3
    out[base + 7] = skip / 3.0
    out[base + 8] = float(bits >> 14 & 1)
    out[base + 9] = float(bits >> 15 & 1)
    out[base + 10] = float(bits >> 16 & 1)
    out[base + 11] = float(bits >> 17 & 1)
    stock = bits >> 18 & 255
    count = 0
    for i in range(8):
        bit = stock >> i & 1
        out[base + 12 + i] = float(bit)
        count += bit
    if count > 4:
        count = 4
    out[base + 20] = count / 4.0
    out[base + 21] = 1.0 if (stock & _ANTI_COUNTER_MASK) != 0 else 0.0
    out[base + 22] = 1.0 if (stock & _NORMAL_MASK) != 0 else 0.0
    choice_used = bits >> 26 & 255
    for i in range(8):
        out[base + 23 + i] = float(choice_used >> i & 1)
    drop_blocked = bits >> 34 & 255
    for i in range(8):
        out[base + 31 + i] = float(drop_blocked >> i & 1)


@njit(cache=True)
def _features_from_lanes_into(lane0, lane1, out):
    player_mask = (np.int64(1) << 42) - 1
    _player_features(lane0 & player_mask, out, 0)
    _player_features(lane1 & player_mask, out, N_PLAYER)
    base = 2 * N_PLAYER
    for i in range(N_PREV):
        out[base + i] = 0.0
    prev = lane0 >> 42 & 31
    out[base + prev] = 1.0
    me_extra = lane0 >> 47 & 15
    if me_extra > 7:
        me_extra = 7
    opp_extra = lane1 >> 42 & 15
    if opp_extra > 7:
        opp_extra = 7
    out[base + N_PREV + 0] = me_extra / 7.0
    out[base + N_PREV + 1] = opp_extra / 7.0
    out[base + N_PREV + 2] = float(lane0 >> 51 & 1)
    out[base + N_PREV + 3] = float(lane0 >> 52 & 1)


@njit(cache=True)
def features_from_lanes(keys0, keys1):
    """Batch feature extraction: (n,) int64 lanes → (n, FEATURE_SIZE) f32."""
    n = keys0.shape[0]
    out = np.empty((n, FEATURE_SIZE), dtype=np.float32)
    row = np.empty(FEATURE_SIZE, dtype=np.float32)
    for i in range(n):
        _features_from_lanes_into(keys0[i], keys1[i], row)
        out[i] = row
    return out


def features_from_state(state: State) -> np.ndarray:
    """Reference single-state extraction (goes through the packed layout so
    the packed and reference paths cannot drift apart silently)."""
    lane0, lane1 = pack_state(state)
    out = np.empty(FEATURE_SIZE, dtype=np.float32)
    _features_from_lanes_into(np.int64(lane0), np.int64(lane1), out)
    return out

"""Experimental "broken stock" env-var toggles (game-design exploration).

Three independent toggles, all OFF by default (env var unset ⇒ base game is
byte-identical). This module is a leaf: it imports nothing from the rest of
the package, specifically so ``actions.py``, ``transition.py`` and
``packed_engine.py`` can all import these constants without creating an
import cycle (``packed_engine`` already imports ``TPAction``/``NTPAction``
from ``actions``, so ``actions`` cannot import ``packed_engine`` back).

  YS_STOCK_FREECHOICE      — STOCK becomes a TARGETED action: the mover may
                             stock ANY stockable skill (id 0..7) not already
                             held, regardless of ``previous_skill``, on any
                             turn (including turn 1). ``max_stock_size``
                             still applies.
  YS_STOCK_UNLIMITED_ALPHA — CHOICE/ALL/DROP lose the once-per-phase gate
                             (``stock_alpha_used_this_phase``); they may be
                             declared repeatedly within one phase.
  YS_STOCK_FREETEMPO      — declaring STOCK (base or targeted) grants the
                             mover +1 extra turn, so STOCK does not pass
                             initiative.

Read ONCE at import time, exactly like the existing YS_COUNTER_PIERCE
toggle (see ``packed_engine._CP_MASK``): numba bakes module globals into
compiled ``@njit(cache=True)`` code at JIT time, so flipping the env var has
NO EFFECT on an already-compiled/cached function. Any change to these values
requires deleting the on-disk numba cache (``complete_solver/__pycache__``,
``complete_ai/__pycache__`` — the ``*.nbi``/``*.nbc`` files) before the next
run; see the golden tests under ``complete_solver/tests/test_stock_toggles*``
for the fresh-subprocess + cache-clear pattern this requires.
"""

from __future__ import annotations

import os


def _flag(name: str) -> bool:
    raw = os.environ.get(name, "0")
    try:
        return int(raw) != 0
    except ValueError:
        return raw.strip().lower() not in ("", "0", "false")


STOCK_FREECHOICE: bool = _flag("YS_STOCK_FREECHOICE")
STOCK_UNLIMITED_ALPHA: bool = _flag("YS_STOCK_UNLIMITED_ALPHA")
STOCK_FREETEMPO: bool = _flag("YS_STOCK_FREETEMPO")

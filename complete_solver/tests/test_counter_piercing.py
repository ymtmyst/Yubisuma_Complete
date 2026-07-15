"""Golden tests for the counter-piercing toggle (YS_COUNTER_PIERCE).

Counter-piercing is an experimental, OFF-by-default rule change: for a
configurable bitmask of skill ids, that skill's effect fires under
reaction COUNTER exactly as it would under reaction NONE (BLOCK is
unaffected). This experiment targets STOCK (id 9, mask bit 1<<9=512) and
CEMENT (id 1, mask bit 1<<1=2) only.

Each scenario runs in a FRESH subprocess with the on-disk numba cache
cleared first. This is not optional: numba's ``cache=True`` keys its
on-disk cache on the compiled function's bytecode, not on the runtime
value of the module-level ``_CP_MASK`` global it closes over. Reusing a
process (or a stale on-disk cache) across different mask values can
silently keep an old mask baked into the compiled ``step``. See
``complete_solver/packed_engine.py`` (the ``_CP_MASK`` docstring) and
``_cp_worker.py`` (the actual checks) for details.

The worker subprocess itself:
  1. asserts the mask it reads back from both engines matches what this
     test asked for (proves the cache-clear + reimport actually worked);
  2. runs hand-picked golden scenarios for STOCK/CEMENT declared under
     COUNTER (and under BLOCK, which must be unaffected), checking
     reference (transition.py) and packed (packed_engine.step) agree;
  3. asserts CHOICE/ALL/DROP under COUNTER are unaffected (fixed
     assertions independent of the mask);
  4. reuses the project's reference-vs-packed differential playout
     harness (test_packed_engine.run_playouts) as a fuzz cross-check
     under the active mask.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CACHE_DIRS = [
    os.path.join(REPO_ROOT, "complete_solver", "__pycache__"),
    os.path.join(REPO_ROOT, "complete_ai", "__pycache__"),
]

STOCK_MASK = 1 << 9  # 512
CEMENT_MASK = 1 << 1  # 2


def _clear_numba_cache() -> None:
    """Delete every .nbi/.nbc numba cache artifact under this repo.

    Required before every mask change (including reverting to 0) — see
    module docstring: a stale on-disk cache can silently keep an old mask
    baked into the compiled njit functions.
    """
    for cache_dir in CACHE_DIRS:
        for pattern in ("*.nbi", "*.nbc"):
            for path in glob.glob(os.path.join(cache_dir, pattern)):
                os.remove(path)


def _run_worker(mask: int) -> None:
    """Run the golden-check worker in a fresh subprocess under *mask*.

    IMPORTANT hygiene: this clears the on-disk numba cache both BEFORE the
    subprocess runs (so it compiles fresh under *mask*, not a stale cache
    from a previous mask) AND AFTER it returns (success or failure). The
    "after" clear is not optional — the subprocess's compile under a
    non-zero mask writes .nbi/.nbc files to the SAME on-disk cache the
    main test process (and every other process on this machine) will read
    from. Left in place, those files would silently make the NEXT
    process's first call to `packed_engine.step` load mask-512 (or
    mask-2) machine code even though that process's own YS_COUNTER_PIERCE
    is unset/0 — i.e. this test file would quietly break the "toggle off
    ⇒ base game byte-identical" guarantee for everything that runs after
    it in the same `unittest discover` session (or on the same machine
    before anyone else clears the cache). Confirmed by reproduction: a
    prior version of this file without the trailing clear caused
    test_packed_engine's differential tests to observe pierced STOCK
    behavior purely from cache leakage, with the mask genuinely 0 in the
    process the whole time.
    """
    _clear_numba_cache()
    try:
        env = dict(os.environ)
        env["YS_COUNTER_PIERCE"] = str(mask)
        proc = subprocess.run(
            [sys.executable, "-m", "complete_solver.tests._cp_worker", str(mask)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        _clear_numba_cache()
    if proc.returncode != 0 or "OK" not in proc.stdout:
        raise AssertionError(
            f"counter-piercing worker failed for mask={mask} "
            f"(YS_COUNTER_PIERCE={mask})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


class TestCounterPiercingToggle(unittest.TestCase):
    def test_mask_zero_is_a_true_no_op(self) -> None:
        """With the mask forced to 0 (equivalent to the env var unset),
        STOCK/CEMENT under COUNTER must behave exactly like the base game
        — proves the new dispatch branch is inert when off, even after a
        fresh recompile."""
        _run_worker(0)

    def test_stock_bit_pierces_counter(self) -> None:
        """YS_COUNTER_PIERCE=512 (STOCK, id 9): declaring STOCK when the
        previous skill is stockable and the opponent COUNTERs now stocks
        it (baseline: not stocked). BLOCK still blocks. CHOICE/ALL/DROP
        (different ids) are unaffected. Reference and packed engine
        agree."""
        _run_worker(STOCK_MASK)

    def test_cement_bit_pierces_counter(self) -> None:
        """YS_COUNTER_PIERCE=2 (CEMENT, id 1): declaring CEMENT under
        COUNTER now applies cement (baseline: no cement). BLOCK still
        blocks. Reference and packed engine agree."""
        _run_worker(CEMENT_MASK)


if __name__ == "__main__":
    unittest.main()

"""Golden tests for the three "broken stock" experiment toggles:
YS_STOCK_FREECHOICE, YS_STOCK_UNLIMITED_ALPHA, YS_STOCK_FREETEMPO.

All default OFF; the base game must be byte-identical when unset (proven by
the untouched complete_solver/complete_ai suites — see WORK_LOG for the run
that established this). Each scenario here runs in a FRESH subprocess with
the on-disk numba cache cleared first: numba's ``cache=True`` keys its
on-disk cache on the compiled function's bytecode, not on the runtime value
of the module-level toggle globals it closes over (see
``complete_solver/toggles.py`` and ``packed_engine._CP_MASK`` for the same
caveat on the pre-existing YS_COUNTER_PIERCE toggle). Reusing a process (or a
stale on-disk cache) across different toggle values can silently keep an old
combination baked into the compiled ``step``/``legal_tp_codes``.

See ``_stock_worker.py`` for the individual-toggle golden checks (each also
cross-checks reference vs packed and fuzzes the differential playout
harness), and ``_stock_smoke_worker.py`` for the composition smoke checks
(a run_selfplay/play_match pass must complete without error).
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


def _clear_numba_cache() -> None:
    """Delete every .nbi/.nbc numba cache artifact under this repo.

    Required before AND after every toggle change (including reverting to
    all-off) — see module docstring and _cp_worker.py's identical warning:
    leaving a non-default-compiled cache in place silently breaks the
    "toggle off => base game byte-identical" guarantee for every OTHER
    process (including subsequent tests in this same session) that runs on
    this machine afterwards.
    """
    for cache_dir in CACHE_DIRS:
        for pattern in ("*.nbi", "*.nbc"):
            for path in glob.glob(os.path.join(cache_dir, pattern)):
                os.remove(path)


def _run_worker(freechoice: bool, unlimited_alpha: bool, freetempo: bool) -> None:
    _clear_numba_cache()
    try:
        env = dict(os.environ)
        env["YS_STOCK_FREECHOICE"] = "1" if freechoice else "0"
        env["YS_STOCK_UNLIMITED_ALPHA"] = "1" if unlimited_alpha else "0"
        env["YS_STOCK_FREETEMPO"] = "1" if freetempo else "0"
        # Never let an ambient COUNTER_PIERCE leak into these single-toggle
        # scenarios (composition with it is tested separately below).
        env.pop("YS_COUNTER_PIERCE", None)
        proc = subprocess.run(
            [
                sys.executable, "-m", "complete_solver.tests._stock_worker",
                "1" if freechoice else "0",
                "1" if unlimited_alpha else "0",
                "1" if freetempo else "0",
            ],
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
            f"stock-toggle worker failed for freechoice={freechoice} "
            f"unlimited_alpha={unlimited_alpha} freetempo={freetempo}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def _run_smoke(env_overrides: dict[str, str]) -> None:
    _clear_numba_cache()
    try:
        env = dict(os.environ)
        for key in (
            "YS_STOCK_FREECHOICE", "YS_STOCK_UNLIMITED_ALPHA",
            "YS_STOCK_FREETEMPO", "YS_COUNTER_PIERCE",
        ):
            env.pop(key, None)
        env.update(env_overrides)
        proc = subprocess.run(
            [sys.executable, "-m", "complete_solver.tests._stock_smoke_worker"],
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
            f"composition smoke worker failed for env={env_overrides}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


class TestStockTogglesAllOff(unittest.TestCase):
    def test_all_off_is_a_true_no_op(self) -> None:
        """With every flag forced to 0 (equivalent to unset), STOCK/CHOICE/
        ALL/DROP must behave exactly like the base game — proves the new
        dispatch branches are inert when off, even after a fresh recompile."""
        _run_worker(False, False, False)


class TestStockFreechoiceToggle(unittest.TestCase):
    def test_freechoice_targets_any_unheld_skill(self) -> None:
        """YS_STOCK_FREECHOICE=1: STOCK-target(FEINT) is legal from a fresh
        initial state (previous_skill=None) and stocks FEINT (baseline:
        illegal). Cannot target an already-held skill. reference==packed."""
        _run_worker(True, False, False)


class TestStockUnlimitedAlphaToggle(unittest.TestCase):
    def test_unlimited_alpha_allows_repeat_choice(self) -> None:
        """YS_STOCK_UNLIMITED_ALPHA=1: after one CHOICE this phase, a second
        CHOICE/ALL/DROP is still legal (baseline: illegal, once-per-phase
        gate). reference==packed."""
        _run_worker(False, True, False)


class TestStockFreetempoToggle(unittest.TestCase):
    def test_freetempo_grants_extra_turn(self) -> None:
        """YS_STOCK_FREETEMPO=1: declaring STOCK keeps the mover's turn
        (baseline: STOCK passes initiative). Also checked in combination with
        FREECHOICE (targeted STOCK). reference==packed."""
        _run_worker(False, False, True)


class TestStockToggleCombinations(unittest.TestCase):
    def test_freechoice_plus_unlimited_alpha(self) -> None:
        _run_worker(True, True, False)

    def test_freechoice_plus_freetempo(self) -> None:
        _run_worker(True, False, True)

    def test_all_three_together(self) -> None:
        _run_worker(True, True, True)


class TestStockToggleCompositionSmoke(unittest.TestCase):
    """run_selfplay/play_match must complete without error under the two
    target configurations from the design brief."""

    def test_p1_freechoice_unlimited_alpha_counter_pierce(self) -> None:
        _run_smoke({
            "YS_STOCK_FREECHOICE": "1",
            "YS_STOCK_UNLIMITED_ALPHA": "1",
            "YS_COUNTER_PIERCE": "512",
        })

    def test_p2_freechoice_unlimited_alpha_freetempo(self) -> None:
        _run_smoke({
            "YS_STOCK_FREECHOICE": "1",
            "YS_STOCK_UNLIMITED_ALPHA": "1",
            "YS_STOCK_FREETEMPO": "1",
        })


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash
# Phase 2 counter-piercing experiment driver (ONE mask, serial).
#
# Cache discipline (a stale pierced cache silently corrupts everything):
#   1. clear the on-disk numba cache,
#   2. export YS_COUNTER_PIERCE=<mask>,
#   3. in a FRESH python process, ASSERT both engines' _CP_MASK == <mask>
#      (proves the recompile baked the mask in) BEFORE any training,
#   4. train (warm-started from the adopted graph-vi model, graph-vi teacher,
#      new --tag namespace so nothing existing is overwritten),
#   5. measure the resulting model's action distribution UNDER THE SAME MASK
#      (same cache, valid — same mask; the measurement re-asserts the mask).
#
# Usage: run_cp_experiment.sh <mask> <tag> <train_games> <measure_games>
set -euo pipefail

MASK="$1"; TAG="$2"; TRAIN_GAMES="$3"; MEAS_GAMES="$4"
ROOT="/c/Users/tktho/Documents/★Download/yubisuma/Complete"
cd "$ROOT"

export PYTHONIOENCODING=utf-8

echo "[$(date)] clearing numba cache"
find complete_solver/__pycache__ complete_ai/__pycache__ -type f \
     \( -iname '*.nbi' -o -iname '*.nbc' \) -delete 2>/dev/null || true

export YS_COUNTER_PIERCE="$MASK"
echo "[$(date)] YS_COUNTER_PIERCE=$YS_COUNTER_PIERCE ; asserting mask baked in"
# NOTE: use `from ... import _CP_MASK` — `import complete_solver.transition`
# resolves to the transition() FUNCTION (complete_solver/__init__ re-exports
# it), not the submodule, so `t._CP_MASK` would spuriously fail.
python -c "from complete_solver.packed_engine import _CP_MASK as pmask; from complete_solver.transition import _CP_MASK as tmask; assert pmask==$MASK and tmask==$MASK, ('MASK MISMATCH', pmask, tmask); print('mask assert OK:', $MASK)"

echo "[$(date)] TRAIN tag=$TAG mask=$MASK games=$TRAIN_GAMES (graph-vi, warm from value_gvi_latest.pt)"
python -m complete_ai.generation_loop \
    --teacher graph-vi \
    --start-model models/value_gvi_latest.pt \
    --tag "$TAG" \
    --generations 12 \
    --games "$TRAIN_GAMES"

echo "[$(date)] MEASURE model=models/value_${TAG}_latest.pt mask=$MASK games=$MEAS_GAMES"
python -m scratchpad.cp_policy_dist \
    --model "models/value_${TAG}_latest.pt" \
    --expect-mask "$MASK" \
    --games "$MEAS_GAMES" \
    --out "data/${TAG}_dist.json"

echo "[$(date)] EXPERIMENT_DONE tag=$TAG mask=$MASK"

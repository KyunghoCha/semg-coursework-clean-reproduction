#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/semg-coursework-matplotlib}"
SEMG_PYTHON="${SEMG_PYTHON:-python3}"
phase="${1:-test}"
case "$phase" in
  test) "$SEMG_PYTHON" -m pytest -q ;;
  prepare)
    : "${SEMG_RAW:?Set SEMG_RAW to the directory containing the original 64 CSV files}"
    "$SEMG_PYTHON" scripts/prepare.py "$SEMG_RAW"
    "$SEMG_PYTHON" scripts/audit_splits.py ;;
  baseline|improve) "$SEMG_PYTHON" scripts/run_experiments.py "$phase" ;;
  verify) "$SEMG_PYTHON" scripts/verify_results.py ;;
  analyze)
    if [[ -n "${SEMG_RAW:-}" ]]; then
      "$SEMG_PYTHON" scripts/analyze.py --raw "$SEMG_RAW"
    else
      "$SEMG_PYTHON" scripts/analyze.py
    fi ;;
  *) echo "Usage: bash run.sh {test|prepare|baseline|improve|verify|analyze}" >&2; exit 2 ;;
esac

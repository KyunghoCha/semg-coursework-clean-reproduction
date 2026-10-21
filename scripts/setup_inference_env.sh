#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_output="${SEMG_RUN_DIR:-$PWD}"
task_python="${SEMG_PYTHON:-python3}"
task_env="$task_output/private/inference_env"
if [[ -e "$task_env" ]]; then echo 'Preserve the existing inference environment.' >&2; exit 2; fi
mkdir -p "$task_output/private/cost_benchmark"
"$task_python" -m venv --system-site-packages "$task_env"
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' "$task_env/bin/python" -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements-gpu-inference.txt
"$task_env/bin/python" -m pip freeze > "$task_output/private/cost_benchmark/inference_environment_freeze.txt"

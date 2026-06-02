#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY_BIN=".venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

DEFAULT_MODELS=(
  "gpt-5-mini"
  "gpt-5.4-mini"
  "gpt-5.4"
  "gpt-5.5"
)

SYNTHETIC_SAFE_MODE=1
FORWARD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --full-data)
      SYNTHETIC_SAFE_MODE=0
      ;;
    -h|--help)
      echo "Usage: ./scripts/run_llm_model_benchmark.sh [--full-data] [benchmark args]"
      echo "  --full-data   Use real recommender snapshots instead of synthetic-safe benchmark payloads."
      echo ""
      PYTHONUNBUFFERED=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib "$PY_BIN" main.py benchmark-llm --help
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$arg")
      ;;
  esac
done

echo "[run] LLM model benchmark starting..."
echo "[run] root: $ROOT_DIR"
echo "[run] python: $PY_BIN"
echo "[run] models: ${DEFAULT_MODELS[*]}"
echo "[run] tip: pass extra CLI options after this script to override defaults."

BENCHMARK_MODE_ARGS=()
if [[ "$SYNTHETIC_SAFE_MODE" -eq 1 ]]; then
  BENCHMARK_MODE_ARGS+=(--synthetic-safe-mode)
fi

PYTHONUNBUFFERED=1 \
PYTHONPATH=src \
MPLCONFIGDIR=/tmp/matplotlib \
"$PY_BIN" main.py benchmark-llm \
  "${BENCHMARK_MODE_ARGS[@]}" \
  --no-template-fallback \
  --use-explainer-moe \
  --compliance-rules-path src/explainer/compliance_rules.txt \
  --enable-understanding-eval \
  --use-llm-user-simulator \
  --use-llm-evaluator \
  --max-understanding-samples 40 \
  --synthetic-repeat 5 \
  --models "${DEFAULT_MODELS[@]}" \
  --out-dir reports/e2e/llm_model_benchmark \
  "${FORWARD_ARGS[@]}"

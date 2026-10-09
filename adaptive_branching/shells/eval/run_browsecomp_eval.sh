#!/usr/bin/env bash
set -euo pipefail

# Evaluate a served checkpoint on BrowseComp or GAIA-text.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export AGENT_TOOLS_CONFIG="${AGENT_TOOLS_CONFIG:-${REPO_ROOT}/adaptive_branching/config/eval_tools.yaml}"
export AGENT_JUDGE_CONFIG="${AGENT_JUDGE_CONFIG:-${REPO_ROOT}/adaptive_branching/config/judge.yaml}"
export AGENT_SEARCH_PROVIDER="${AGENT_SEARCH_PROVIDER:-serper}"
export SERPER_ENDPOINT="${SERPER_ENDPOINT:-https://google.serper.dev/search}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/run_browsecomp_eval.py" --help
fi

for name in MODEL BASE_URL DATA OUT LLM_API_KEY BROWSER_LLM_URL LLM_JUDGE_URL LLM_JUDGE_KEY; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: missing required environment variable ${name}" >&2
    exit 1
  fi
done
if [[ "${AGENT_SEARCH_PROVIDER}" == "serper" && -z "${SERPER_API_KEY:-}" ]]; then
  echo "ERROR: missing required environment variable SERPER_API_KEY" >&2
  exit 1
fi
for config in "${AGENT_TOOLS_CONFIG}" "${AGENT_JUDGE_CONFIG}"; do
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: config file not found: ${config}" >&2
    exit 1
  fi
done
if [[ ! -s "${DATA}" ]]; then
  echo "ERROR: data file is missing or empty: ${DATA}" >&2
  exit 1
fi

exec "${PYTHON_BIN:-python3}" -u "${SCRIPT_DIR}/run_browsecomp_eval.py" \
  --model "${MODEL}" --base-url "${BASE_URL}" --data "${DATA}" --out "${OUT}" \
  --history-mode "${HISTORY_MODE:-keep5}" --concurrency "${CONCURRENCY:-128}" \
  --max-turns "${MAX_TURNS:-200}" --max-tokens "${MAX_TOKENS:-22000}" \
  --max-seq-len "${MAX_SEQ_LEN:-262144}" \
  --context-reserve-tokens "${CONTEXT_RESERVE_TOKENS:-32768}" \
  --keep-tool-results "${KEEP_TOOL_RESULTS:-5}" \
  --temperature "${TEMPERATURE:-1}" --top-p "${TOP_P:-1}" --top-k "${TOP_K:--1}" \
  --samples-per-task "${SAMPLES_PER_TASK:-1}" "$@"

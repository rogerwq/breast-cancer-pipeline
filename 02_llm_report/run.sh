#!/bin/bash
set -e

echo "=== Job 02: LLM Patient Report Generation ==="
mkdir -p outputs

OLLAMA_MODEL="${PARAM_OLLAMA_MODEL:-phi3:mini}"
OLLAMA_HOST="${PARAM_OLLAMA_HOST:-http://host.docker.internal:11434}"
MAX_TOKENS="${PARAM_MAX_TOKENS:-700}"

echo "Ollama model : $OLLAMA_MODEL"
echo "Ollama host  : $OLLAMA_HOST"
echo "Max tokens   : $MAX_TOKENS"

python llm_report.py \
    --ollama_model  "$OLLAMA_MODEL" \
    --ollama_host   "$OLLAMA_HOST" \
    --max_tokens    "$MAX_TOKENS" \
    --prediction    "cnn_prediction.json"

echo "=== Job 02 complete ==="

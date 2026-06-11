#!/bin/bash
set -e

echo "=== Job 02: LLM Patient Report Generation ==="
mkdir -p outputs

OLLAMA_MODEL="${PARAM_OLLAMA_MODEL:-mistral}"
OLLAMA_HOST="http://localhost:11434"
MAX_TOKENS="${PARAM_MAX_TOKENS:-700}"

echo "Ollama model : $OLLAMA_MODEL"
echo "Max tokens   : $MAX_TOKENS"

# Start Ollama daemon and wait until ready
ollama serve &
OLLAMA_PID=$!
until curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do sleep 1; done

python llm_report.py \
    --ollama_model  "$OLLAMA_MODEL" \
    --ollama_host   "$OLLAMA_HOST" \
    --max_tokens    "$MAX_TOKENS" \
    --prediction    "cnn_prediction.json"

kill $OLLAMA_PID
echo "=== Job 02 complete ==="

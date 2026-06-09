#!/bin/bash
set -e

echo "=== Job 03: Final Results ==="
mkdir -p outputs

python display_results.py \
    --report "patient_report.txt" \
    --scores "evaluation_scores.json"

echo "=== Pipeline complete ==="

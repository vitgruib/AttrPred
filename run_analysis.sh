#!/bin/bash
# Run probes + all analyses after embeddings exist. Ridge is primary; --mlp adds MLP secondary.
cd "$(dirname "$0")"
source env.sh
PY=~/.venvs/attrpred/bin/python
echo "=== ANALYSIS eval START ==="
$PY src/evaluate.py --mlp && echo "=== ANALYSIS eval OK ===" || { echo "=== ANALYSIS eval FAIL ==="; exit 1; }
echo "=== ANALYSIS invariance START ==="
$PY src/invariance.py && echo "=== ANALYSIS invariance OK ===" || echo "=== ANALYSIS invariance FAIL ==="
echo "=== ANALYSIS bias START ==="
$PY src/bias.py && echo "=== ANALYSIS bias OK ===" || echo "=== ANALYSIS bias FAIL ==="
echo "=== ANALYSIS ALL_DONE ==="

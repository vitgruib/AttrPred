#!/bin/bash
# Run all embedding extractions sequentially, logging per-family markers.
# torch families (facenet/adaface/fairface/clip) and onnx families (arcface/cosface)
# both benefit from the CUDA libs on LD_LIBRARY_PATH.
cd "$(dirname "$0")"
source env.sh
PY=~/.venvs/attrpred/bin/python
for fam in facenet arcface cosface adaface fairface geometric clip; do
  echo "=== EXTRACT_START $fam ==="
  if $PY src/extract.py "$fam" > "logs_extract_${fam}.txt" 2>&1; then
    echo "=== EXTRACT_OK $fam ==="
  else
    echo "=== EXTRACT_FAIL $fam (see logs_extract_${fam}.txt) ==="
    tail -5 "logs_extract_${fam}.txt"
  fi
done
echo "=== EXTRACT_ALL_DONE ==="

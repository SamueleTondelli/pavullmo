#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
DOCUMENTS_DIR=${1:-"$PROJECT_ROOT/artifacts/documents"}
MIXTURE=${2:-balanced}
for size in 4096 8192 16384; do
  uv run --project "$PROJECT_ROOT" python "$PROJECT_ROOT/src/dataset/tokenizer/train_tokenizer.py" --documents-dir "$DOCUMENTS_DIR" --mixture "$MIXTURE" --vocab-size "$size" --byte-budget 1073741824 --model-prefix "$PROJECT_ROOT/artifacts/tokenizers/${size}"
done

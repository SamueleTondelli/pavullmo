#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
for size in 4096 8192 16384; do
  python "$PROJECT_ROOT/dataset/tokenizer/train_tokenizer.py" --vocab-size "$size" --byte-budget 1073741824 --model-prefix "$PROJECT_ROOT/tmp/tokenizers/${size}"
done

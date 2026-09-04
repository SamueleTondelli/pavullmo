#!/bin/sh

python train_tokenizer.py --vocab-size 4096 --byte-budget 1073741824 --model-prefix 4k
python train_tokenizer.py --vocab-size 8192 --byte-budget 1073741824 --model-prefix 8k
python train_tokenizer.py --vocab-size 16384 --byte-budget 1073741824 --model-prefix 16k

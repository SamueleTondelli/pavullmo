#!/bin/sh
python src/pavullmo/evaluate_base.py --model models/sweep_20260812T163136Z-66771_arch_10m_dataset_42m_tokenizer_4k_vocab_4096_lr_1.25e-3_batch_8_dropout_0.0_weight_decay_0.0.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok4k_90M.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok4k_195M.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok4k_421m.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok4k_907M.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok4k_1B9.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok4k_4b.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/sweep_20260813T085338Z-3943_arch_20m_dataset_42m_tokenizer_4k_vocab_4096_lr_1e-3_batch_8_dropout_0.0_weight_decay_0.0.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/sweep_20260813T094702Z-11515_arch_35m_dataset_42m_tokenizer_4k_vocab_4096_lr_0.75e-3_batch_8_dropout_0.0_weight_decay_0.0.pt --csv src/pavullmo/scaling_loss.csv

python src/pavullmo/evaluate_base.py --model models/sweep_20260812T163136Z-66771_arch_10m_dataset_42m_tokenizer_16k_vocab_16384_lr_1.25e-3_batch_8_dropout_0.0_weight_decay_0.0.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok16k_90M.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok16k_195M.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok16k_421m.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok16k_907M.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok16k_1B9.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/10m_tok16k_4b.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/sweep_20260813T085338Z-3943_arch_20m_dataset_42m_tokenizer_16k_vocab_16384_lr_1e-3_batch_8_dropout_0.0_weight_decay_0.0.pt --csv src/pavullmo/scaling_loss.csv
python src/pavullmo/evaluate_base.py --model models/sweep_20260813T094702Z-11515_arch_35m_dataset_42m_tokenizer_16k_vocab_16384_lr_0.75e-3_batch_8_dropout_0.0_weight_decay_0.0.pt --csv src/pavullmo/scaling_loss.csv

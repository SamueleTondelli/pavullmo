# PavuLLMo

## Setup
After setting up the uv project/venv and the modal cli, create the local tokenized dataset with

```bash
  python dataset/build_dataset.py
```

3 diffferent dataset sizes will be created: 10M, 100M and 1B tokens.

Then create the modal volume with

```bash
  python dataset/create_modal_volume.py
```

To start a pre-train run on modal

```bash
  EXPERIMENT_NAME=<experiment_name> DATASET_VARIANT=<dataset_size> modal run src/pavullmo/modal_pretrain_base.py
```

All hyperparameters can be configured as enviroment variables. BATCH_SIZE=32 seems to be the maximum batch size before the memory explodes.

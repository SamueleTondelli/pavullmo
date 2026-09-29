"""Checkpoint restoration retains history while honoring a new training phase."""

import ast
import copy
import json
import random
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pavullmo import pretrain_base, pretrain_base_muon
from src.pavullmo.training_checkpoint import (
    EpochBatchSampler,
    capture_training_state,
    restore_training_state,
)


ROOT = Path(__file__).resolve().parents[1]
MODULES = (pretrain_base, pretrain_base_muon)


class TrainingCheckpointTests(unittest.TestCase):
    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                self.assert_state_equal(expected[name], actual[name])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def build_training_objects(self, module, total_steps, *, changed=False):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(0.1 if changed else 0.35),
            torch.nn.Linear(4, 1),
        )
        adam_options = {
            "lr": 0.02 if changed else 0.01,
            "weight_decay": 0.07 if changed else 0.01,
            "betas": (0.8, 0.9) if changed else (0.9, 0.999),
            "eps": 1e-6 if changed else 1e-8,
        }
        if module is pretrain_base:
            optimizers = {"adamw": torch.optim.AdamW(model.parameters(), **adam_options)}
        else:
            optimizers = {
                "adamw": torch.optim.AdamW([model[0].bias, *model[2].parameters()], **adam_options),
                "muon": torch.optim.Muon(
                    [model[0].weight], lr=0.03 if changed else 0.01,
                    weight_decay=0.08 if changed else 0.05,
                    momentum=0.8 if changed else 0.95, nesterov=not changed,
                    eps=1e-6 if changed else 1e-7, ns_steps=3 if changed else 5,
                    adjust_lr_fn="original" if changed else "match_rms_adamw",
                ),
            }
        schedulers = {name: module.build_scheduler(optimizer, total_steps)
                      for name, optimizer in optimizers.items()}
        return model, optimizers, schedulers

    def update(self, model, optimizers, schedulers, iterator):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for _ in range(2):
            inputs, targets = next(iterator)
            loss = (model(inputs) - targets).square().mean()
            (loss / 2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        for optimizer in optimizers.values():
            optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()

    def save_checkpoint(self, module, model, optimizers, schedulers, folder):
        state = capture_training_state(
            model, optimizers, schedulers,
            global_step=2, total_steps=8, tokens_seen=32, clipped_steps=2,
            optimizer_steps_per_epoch=6,
            dataset_metadata={"train": {"token_count": 100}},
            data_generator=torch.Generator().manual_seed(42),
            validation_generator=torch.Generator().manual_seed(42),
        )
        hyperparameters = {"EMBED_DIM": 4, "ATTN_HEADS": 1, "DROPOUT": 0.35}
        checkpoint = module.save_final_model(
            model, Path(folder), experiment_name="first", dataset_variant="old",
            global_step=2, train_loss=1.0, val_loss=1.1,
            hyperparameters=hyperparameters, training_state=state,
        )
        return checkpoint, state, hyperparameters

    def test_restores_history_but_keeps_new_options_schedule_and_rng(self):
        for module in MODULES:
            for schedule in ("cosine", "wsd"):
                with self.subTest(module=module.__name__, schedule=schedule), \
                     tempfile.TemporaryDirectory() as folder, \
                     patch.object(module, "WARMUP_STEPS", 1):
                    torch.manual_seed(42)
                    model, optimizers, schedulers = self.build_training_objects(module, 8)
                    dataset = TensorDataset(
                        torch.arange(96, dtype=torch.float32).reshape(24, 4) / 96,
                        torch.arange(24, dtype=torch.float32).reshape(24, 1) / 24,
                    )
                    iterator = iter(DataLoader(dataset, batch_size=2))
                    for _ in range(2):
                        self.update(model, optimizers, schedulers, iterator)
                    checkpoint, state, hyperparameters = self.save_checkpoint(
                        module, model, optimizers, schedulers, folder
                    )
                    expected_weights = copy.deepcopy(model.state_dict())
                    expected_history = {
                        name: copy.deepcopy(optimizer.state_dict()["state"])
                        for name, optimizer in optimizers.items()
                    }
                    # New seed and settings describe this phase. Restoration must
                    # not reinstate the old RNG, schedule counters, or group options.
                    random.seed(99)
                    torch.manual_seed(99)
                    with patch.object(module, "WARMUP_STEPS", 2), \
                         patch.object(module, "LR_SCHEDULER", schedule), \
                         patch.object(module, "WSD_DECAY_FRACTION", 0.6):
                        model, optimizers, schedulers = self.build_training_objects(module, 5, changed=True)
                        new_groups = {name: copy.deepcopy(optimizer.state_dict()["param_groups"])
                                      for name, optimizer in optimizers.items()}
                        new_schedulers = {name: copy.deepcopy(scheduler.state_dict())
                                          for name, scheduler in schedulers.items()}
                        python_rng = random.getstate()
                        torch_rng = torch.get_rng_state()
                        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
                        progress = restore_training_state(
                            checkpoint, model, optimizers,
                            hyperparameters={
                                **hyperparameters, "DROPOUT": 0.1, "LR": 0.02,
                                "WEIGHT_DECAY": 0.07, "WARMUP_STEPS": 2,
                                "LR_SCHEDULER": schedule, "WSD_DECAY_FRACTION": 0.6,
                                "DATASET_VARIANT": "new", "DATASET_PREFIX": "new_token_stream",
                                "BATCH_SIZE": 1, "GRAD_ACCUM_STEPS": 3,
                                "SEQ_LEN": 8, "ROPE_BASE": 20000, "MAX_STEPS": 5,
                                "SEED": 99, "Z_LOSS_COEFFICIENT": 1e-4,
                                "INITIALIZATION": "gpt_scaled",
                            },
                        )
                        self.assertEqual(progress, {"global_step": 2})
                        self.assert_state_equal(expected_weights, model.state_dict())
                        self.assertEqual(random.getstate(), python_rng)
                        self.assert_state_equal(torch_rng, torch.get_rng_state())
                        if cuda_rng:
                            self.assert_state_equal(cuda_rng, torch.cuda.get_rng_state_all())
                        for name, parameter in model.named_parameters():
                            self.assert_state_equal(state["gradients"][name], parameter.grad)
                        for name, optimizer in optimizers.items():
                            self.assert_state_equal(expected_history[name], optimizer.state_dict()["state"])
                            self.assert_state_equal(new_groups[name], optimizer.state_dict()["param_groups"])
                            self.assert_state_equal(new_schedulers[name], schedulers[name].state_dict())
                            self.assertEqual(schedulers[name].last_epoch, 0)
                        # Stored gradients were already applied; the normal update
                        # clears them and advances the retained Adam step counters.
                        self.update(model, optimizers, schedulers, iter(DataLoader(dataset, batch_size=2)))
                        for parameter_state in optimizers["adamw"].state.values():
                            self.assertEqual(parameter_state["step"].item(), 3)
                        for scheduler in schedulers.values():
                            self.assertEqual(scheduler.last_epoch, 1)

    def test_rejects_incompatible_architecture_optimizer_and_model_only_files(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder, \
                 patch.object(module, "WARMUP_STEPS", 0):
                model, optimizers, schedulers = self.build_training_objects(module, 8)
                checkpoint, state, hyperparameters = self.save_checkpoint(
                    module, model, optimizers, schedulers, folder
                )
                for name in ("EMBED_DIM", "ATTN_HEADS"):
                    with self.assertRaisesRegex(ValueError, name):
                        restore_training_state(checkpoint, model, optimizers,
                                               hyperparameters={**hyperparameters, name: 8})
                with self.assertRaisesRegex(ValueError, "optimizer types"):
                    restore_training_state(checkpoint, model, {"different": optimizers["adamw"]},
                                           hyperparameters=hyperparameters)
                wrong_groups = {"adamw": torch.optim.AdamW(list(reversed(list(model.parameters()))))}
                if "muon" in optimizers:
                    wrong_groups["muon"] = optimizers["muon"]
                with self.assertRaisesRegex(ValueError, "parameter groups"):
                    restore_training_state(checkpoint, model, wrong_groups,
                                           hyperparameters=hyperparameters)
                model_only = module.save_final_model(
                    model, Path(folder), experiment_name="weights", dataset_variant="old",
                    global_step=2, train_loss=1.0, val_loss=1.0, hyperparameters=hyperparameters,
                )
                saved = torch.load(model_only, weights_only=True)
                self.assertEqual(saved["format_version"], 1)
                self.assertNotIn("training_state", saved)
                with self.assertRaisesRegex(ValueError, "SAVE_TRAINING_STATE=true"):
                    restore_training_state(model_only, model, optimizers, hyperparameters=hyperparameters)

    def test_epoch_sampler_cursor_ignores_worker_prefetch(self):
        dataset = TensorDataset(torch.arange(25))
        sampler = EpochBatchSampler(len(dataset), 2, 42)
        epoch_zero = list(sampler)
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=2,
            generator=torch.Generator().manual_seed(42), multiprocessing_context="spawn",
        )
        iterator = iter(loader)
        self.assertEqual(next(iterator)[0].tolist(), epoch_zero[0])
        del iterator
        sampler.set_epoch(0, 1)
        self.assertEqual(list(sampler), epoch_zero[1:])
        sampler.set_epoch(1)
        self.assertNotEqual(list(sampler), epoch_zero)
        self.assertEqual(len(list(sampler)), 12)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "requires BF16 CUDA")
    def test_actual_cuda_loops_start_new_phase_with_new_data_and_hyperparameters(self):
        # The independent reference restores buffers directly onto parameters,
        # leaving newly built optimizer options untouched. Both runs then use the
        # actual BF16 loops and fresh schedules on the newly selected token stream.
        for module in MODULES:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                for name, count, offset in (
                    ("train_old", 37, 0), ("validation", 37, 0),
                    ("train_alt_new", 99, 5), ("validation_alt", 49, 5),
                ):
                    directory = root / "datasets" / name
                    directory.mkdir(parents=True)
                    tokens = ((torch.arange(count) + offset) % 16).to(torch.uint16)
                    (directory / "tokens.bin").write_bytes(tokens.numpy().tobytes())
                    (directory / "metadata.json").write_text(json.dumps({
                        "storage": {"dtype": "uint16", "endianness": "little", "layout": "flat_token_stream"},
                        "token_count": count, "tokenizer": {"vocab_size": 16},
                        "shards": [{"file": "tokens.bin", "tokens": count}],
                    }))
                settings = {
                    "VOCAB_SIZE": 16, "N_BLOCKS": 1, "EMBED_DIM": 16,
                    "ATTN_HEADS": 4, "FFN_DIM": 32, "SEQ_LEN": 4,
                    "DROPOUT": 0.2, "BATCH_SIZE": 2, "GRAD_ACCUM_STEPS": 2,
                    "EPOCHS": 2, "MAX_STEPS": 4, "WARMUP_STEPS": 0,
                    "LR": 0.001, "MIN_LR": 0.0, "LR_SCHEDULER": "cosine",
                    "DATASET_DIR": root / "datasets", "DATASET_PREFIX": "",
                    "DATASET_VARIANT": "old", "NUM_WORKERS": 0, "PIN_MEMORY": False,
                    "VALIDATION_INTERVAL": 1, "VALIDATION_STEPS": 1,
                    "DIAGNOSTICS_INTERVAL": 1, "COMPILE_MODEL": False,
                    "MODEL_OUTPUT_DIR": root / "models", "LOG_DIR": root / "logs",
                    "RUNS_CSV": root / "runs.csv", "SAVE_TRAINING_STATE": True,
                    "RESUME_CHECKPOINT": "", "EXPERIMENT_NAME": "first",
                }
                for name, value in settings.items():
                    stack.enter_context(patch.object(module, name, value))
                with redirect_stdout(StringIO()):
                    module.main()
                original = torch.load(root / "models/first.pt", map_location="cpu", weights_only=True)

                def restore_reference(checkpoint_path, model, optimizers, *, hyperparameters):
                    model.load_state_dict(original["model_state_dict"])
                    for name, optimizer in optimizers.items():
                        saved_optimizer = original["training_state"]["optimizer_state_dicts"][name]
                        for group, saved_group in zip(optimizer.param_groups, saved_optimizer["param_groups"]):
                            for parameter, saved_id in zip(group["params"], saved_group["params"]):
                                optimizer.state[parameter] = {
                                    key: value.to(parameter.device).clone() if isinstance(value, torch.Tensor)
                                    else copy.deepcopy(value)
                                    for key, value in saved_optimizer["state"][saved_id].items()
                                }
                    for name, parameter in model.named_parameters():
                        gradient = original["training_state"]["gradients"][name]
                        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
                    return {"global_step": original["global_step"]}

                new_settings = {
                    "EPOCHS": 1, "MAX_STEPS": 3, "LR": 0.002, "MIN_LR": 0.0002,
                    "WEIGHT_DECAY": 0.09, "WARMUP_STEPS": 1, "LR_SCHEDULER": "wsd",
                    "WSD_DECAY_FRACTION": 0.67, "ADAM_BETA1": 0.8, "ADAM_BETA2": 0.9,
                    "ADAM_EPS": 1e-6, "NORM_LR_MULTIPLIER": 0.5,
                    "SEQ_LEN": 8, "DROPOUT": 0.1, "BATCH_SIZE": 1,
                    "GRAD_ACCUM_STEPS": 2, "SEED": 101, "ROPE_BASE": 20000.0,
                    "DATASET_PREFIX": "alt", "DATASET_VARIANT": "new",
                    "Z_LOSS_COEFFICIENT": 1e-4, "MAX_GRAD_NORM": 0.5,
                    "RESUME_CHECKPOINT": str(root / "models/first.pt"),
                    "EXPERIMENT_NAME": "continued",
                }
                if module is pretrain_base_muon:
                    new_settings.update({
                        "MUON_LR_MULTIPLIER": 2.0, "MUON_WEIGHT_DECAY": 0.08,
                        "MUON_MOMENTUM": 0.8, "MUON_NESTEROV": False,
                        "MUON_EPS": 1e-6, "MUON_NS_ITERS": 3,
                        "MUON_ADJUST_LR_FN": "original",
                    })
                for name, value in new_settings.items():
                    stack.enter_context(patch.object(module, name, value))
                with redirect_stdout(StringIO()):
                    module.main()
                    with patch.object(module, "restore_training_state", side_effect=restore_reference), \
                         patch.object(module, "EXPERIMENT_NAME", "reference"):
                        module.main()
                continued = torch.load(root / "models/continued.pt", map_location="cpu", weights_only=True)
                reference = torch.load(root / "models/reference.pt", map_location="cpu", weights_only=True)
                self.assert_state_equal(reference["model_state_dict"], continued["model_state_dict"])
                self.assert_state_equal(reference["training_state"], continued["training_state"])
                self.assertEqual(continued["global_step"], 3)
                self.assertEqual(continued["training_state"]["tokens_seen"], 3 * 1 * 2 * 8)
                self.assertEqual(continued["training_state"]["dataset_metadata"]["train"]["token_count"], 99)
                adam = continued["training_state"]["optimizer_state_dicts"]["adamw"]
                for parameter_state in adam["state"].values():
                    self.assertEqual(parameter_state["step"].item(), 7)
                self.assertEqual(adam["param_groups"][0]["initial_lr"], 0.002)
                self.assertEqual(adam["param_groups"][2]["initial_lr"], 0.001)
                self.assertEqual(adam["param_groups"][0]["weight_decay"], 0.09)
                self.assertEqual(adam["param_groups"][1]["weight_decay"], 0.0)
                if module is pretrain_base_muon:
                    muon = continued["training_state"]["optimizer_state_dicts"]["muon"]
                    self.assertEqual(muon["param_groups"][0]["initial_lr"], 0.004)
                    self.assertEqual(muon["param_groups"][0]["momentum"], 0.8)
                    self.assertEqual(muon["param_groups"][0]["weight_decay"], 0.08)
                self.assertIn("LR=0.002", continued["config"])
                self.assertIn("DATASET_VARIANT=new", continued["config"])

    def test_modal_wrappers_forward_checkpoint_controls(self):
        for filename in ("modal_pretrain_base.py", "modal_pretrain_base_muon.py"):
            tree = ast.parse((ROOT / "src/pavullmo" / filename).read_text())
            setting = next(node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "PRETRAIN_ENVIRONMENT_VARIABLES"
                                   for target in node.targets))
            names = ast.literal_eval(setting.value)
            self.assertIn("SAVE_TRAINING_STATE", names)
            self.assertIn("RESUME_CHECKPOINT", names)


if __name__ == "__main__":
    unittest.main()

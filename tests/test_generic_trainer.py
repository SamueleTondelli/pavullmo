"""Shared setup with independent root configs and caller-controlled updates."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch._functorch.config as aot_config
import torch.nn.functional as F
from pydantic import Field

from config import Config, TrainerConfig
from components import GenericTrainer
from model import DecoderTransformer


class ObjectiveConfig(Config):
    coefficient: float = 0.5


class OtherStageConfig(TrainerConfig):
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)


class GenericTrainerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.device_patch = patch("components.generic_trainer._select_device", return_value=torch.device("cpu"))
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)
        self.addCleanup(self.directory.cleanup)

    def trainer(self):
        config = OtherStageConfig.from_dict({
            "name": "other-stage", "model": {"vocab_size": 32, "n_blocks": 1,
                "embed_dim": 16, "attn_heads": 4, "ffn_dim": 32, "seq_len": 8},
            "optimizer": {"adamw": {"fused": False}, "schedule": {"warmup_steps": 0}},
            "runtime": {"output_dir": str(self.root), "compile_model": False,
                        "dataset_dir": "/not-needed"},
            "logging": {"console": False}, "objective": {"coefficient": 0.25},
        })
        trainer = GenericTrainer(config, total_steps=3, seed=42)
        self.addCleanup(trainer.close)
        return trainer, config

    def test_setup_preserves_independent_typed_root_and_user_settings(self):
        trainer, config = self.trainer()
        self.assertIs(type(trainer.config), OtherStageConfig)
        self.assertIsInstance(trainer.config.objective, ObjectiveConfig)
        self.assertEqual(trainer.config.to_dict(), config.to_dict())
        self.assertIsNot(trainer.config, config)
        self.assertNotIn("training", trainer.config.to_dict())
        self.assertNotIn("data", trainer.config.to_dict())
        self.assertEqual(OtherStageConfig.load(trainer.run_dir / "config.toml"), config)
        self.assertEqual(trainer.checkpoints.directory, trainer.run_dir / "checkpoints")
        self.assertEqual(trainer.optimizer.total_steps, 3)

    def test_tensor_and_mapping_forwards_accumulate_without_stepping(self):
        trainer, _ = self.trainer()
        reference = DecoderTransformer(trainer.config.model)
        reference.load_state_dict(trainer.model.state_dict())
        inputs = torch.tensor([[1, 5, 6]])
        labels = torch.tensor([[5, 6, 2]])
        mask = torch.tensor([[1.0, 0.0, 2.0]])
        weight = trainer.config.objective.coefficient

        def objective(outputs, *, labels, mask, weight):
            losses = F.cross_entropy(outputs.flatten(0, 1), labels.flatten(), reduction="none")
            return weight * (losses * mask.flatten()).sum() / mask.sum()

        expected = objective(reference(inputs), labels=labels, mask=mask, weight=weight * 2)
        expected.backward()
        original_weights = copy.deepcopy(trainer.model.state_dict())
        optimizer_state = copy.deepcopy(trainer.optimizer.optimizers["adamw"].state_dict())
        scheduler_state = copy.deepcopy(trainer.optimizer.schedulers["adamw"].state_dict())
        kwargs = {"labels": labels, "mask": mask, "weight": weight}
        first = trainer.forward_backward(inputs, objective, loss_kwargs=kwargs)
        second = trainer.forward_backward({"x": inputs}, objective, loss_kwargs=kwargs)
        self.assertFalse(first.requires_grad)
        torch.testing.assert_close(first + second, expected.detach())
        for name, parameter in trainer.model.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)
            torch.testing.assert_close(parameter, original_weights[name], rtol=0, atol=0)
        self.assertEqual(trainer.optimizer.optimizers["adamw"].state_dict(), optimizer_state)
        self.assertEqual(trainer.optimizer.schedulers["adamw"].state_dict(), scheduler_state)
        self.assertIs(kwargs["labels"], labels)
        self.assertIs(kwargs["mask"], mask)

    def test_scalar_loss_without_extra_arguments_and_invalid_return(self):
        trainer, _ = self.trainer()
        inputs = torch.tensor([[1, 5, 6]])
        value = trainer.forward_backward(inputs, lambda outputs: outputs.square().mean())
        self.assertEqual(value.ndim, 0)
        self.assertTrue(any(parameter.grad is not None for parameter in trainer.model.parameters()))
        trainer.model.zero_grad(set_to_none=True)
        for loss_fn in (lambda outputs: outputs, lambda outputs: 1.0):
            with self.subTest(loss_fn=loss_fn), self.assertRaisesRegex(ValueError, "scalar tensor"):
                trainer.forward_backward(inputs, loss_fn)
            self.assertTrue(all(parameter.grad is None for parameter in trainer.model.parameters()))

    def test_activation_memory_budget_covers_forward_backward_and_is_restored(self):
        trainer, config = self.trainer()
        trainer.config = config.with_overrides({
            "runtime.compile_model": True, "runtime.act_checkpoint_mem_budget": 0.25,
        })
        observed = []
        forward_hook = trainer.model.register_forward_pre_hook(
            lambda *_: observed.append(("forward", aot_config.activation_memory_budget))
        )
        gradient_hook = trainer.model.embeddings.weight.register_hook(
            lambda _: observed.append(("backward", aot_config.activation_memory_budget))
        )
        self.addCleanup(forward_hook.remove)
        self.addCleanup(gradient_hook.remove)
        inputs = torch.tensor([[1, 5, 6]])
        with aot_config.patch(activation_memory_budget=0.75):
            trainer.forward_backward(inputs, lambda outputs: outputs.square().mean())
            self.assertEqual(observed, [("forward", 0.25), ("backward", 0.25)])
            self.assertEqual(aot_config.activation_memory_budget, 0.75)
            with self.assertRaisesRegex(ValueError, "scalar tensor"):
                trainer.forward_backward(inputs, lambda outputs: outputs)
            self.assertEqual(aot_config.activation_memory_budget, 0.75)


if __name__ == "__main__":
    unittest.main()

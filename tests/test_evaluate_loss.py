"""Loss evaluation shares saved-model loading and fixed validation spans."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from components.data import TokenBlockDataset
from config import ModelConfig
from config.serialization import dump_toml
from model import DecoderTransformer
from src.evaluation import evaluate_loss


def validation_artifact(root):
    # Three selected spans, crossing shard boundaries, with a partial last batch.
    root.mkdir()
    tokens = torch.arange(31, dtype=torch.int64).remainder(16)
    shards = []
    for index, chunk in enumerate(tokens.split(11)):
        name = f"shard_{index}.bin"
        (root / name).write_bytes(chunk.to(torch.uint16).numpy().tobytes())
        shards.append({"file": name, "tokens": chunk.numel()})
    (root / "metadata.json").write_text(json.dumps({
        "storage": {"dtype": "uint16", "endianness": "little", "layout": "flat_token_stream"},
        "token_count": tokens.numel(), "shards": shards,
        "fixed_evaluation_sample": {
            "sequence_length": 4, "block_count": 3,
            "blocks": [{"start_token": start} for start in (8, 20, 25)],
        },
    }))
    return root


class EvaluationTests(unittest.TestCase):
    def test_current_and_legacy_configs_evaluate_complete_sample_on_cpu(self):
        for kind in ("full", "weights_only", "legacy"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = ModelConfig(vocab_size=16, n_blocks=1, embed_dim=8,
                                     attn_heads=2, ffn_dim=16, seq_len=4)
                model = DecoderTransformer(config).eval()
                text = dump_toml({"model": config.to_dict()})
                if kind == "legacy":
                    torch.save({"model_state_dict": model.state_dict()}, root / "old.pt")
                    (root / "config.toml").write_text(text)
                else:
                    (root / "checkpoints").mkdir()
                    state = {"model": model.state_dict()}
                    if kind == "full":
                        state["unused_optimizer"] = {"state": {}}
                    torch.save({"format_version": 1, "config_toml": text,
                                "full_state": kind == "full", "trainer_state": state},
                               root / "checkpoints/step_00000002.pt")
                    # Embedded model settings suffice: no full training config/datasets needed.
                    (root / "config.toml").write_text("invalid: edited config")
                dataset_path = validation_artifact(root / "validation")
                dataset = TokenBlockDataset(dataset_path, 4)
                inputs, targets = zip(*(dataset[i] for i in range(len(dataset))))
                with torch.no_grad():
                    logits = model(torch.stack(inputs))
                    expected = F.cross_entropy(logits.flatten(0, 1), torch.stack(targets).flatten()).item()
                with patch("sys.argv", ["evaluate_loss", "--model", str(root), "--dataset", str(dataset_path)]), \
                     patch.object(evaluate_loss, "BATCH_SIZE", 2), \
                     patch.object(evaluate_loss, "NUM_WORKERS", 0), \
                     patch.object(evaluate_loss, "PIN_MEMORY", False), \
                     patch.object(evaluate_loss.torch.cuda, "is_available", return_value=False), \
                     patch.object(evaluate_loss.torch, "autocast") as autocast, \
                     patch("sys.stdout", new_callable=io.StringIO) as output:
                    loader = evaluate_loss.build_loader(dataset, shuffle=False)
                    self.assertEqual([x.shape[0] for x, _ in loader], [2, 1])
                    self.assertEqual(torch.cat([x for x, _ in loader]).tolist(), torch.stack(inputs).tolist())
                    evaluate_loss.main()
                    measured = float(output.getvalue().split("Final loss: ")[1])
                    self.assertAlmostEqual(measured, expected, places=5)
                    autocast.assert_not_called()

    def test_checkpoint_selection_matches_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            for step in (2, 10):
                config = ModelConfig(vocab_size=16, n_blocks=1, embed_dim=8,
                                     attn_heads=2, ffn_dim=16, seq_len=4, rope_base=100.0 + step)
                model = DecoderTransformer(config)
                torch.save({"format_version": 1, "full_state": False,
                            "config_toml": dump_toml({"model": config.to_dict()}),
                            "trainer_state": {"model": model.state_dict()}},
                           checkpoints / f"step_{step:08d}.pt")
            for selection in (Path("checkpoints/step_00000002.pt"), checkpoints / "step_00000002.pt"):
                loaded, vocab, context = evaluate_loss.load_model(root, torch.device("cpu"), checkpoint=selection)
                self.assertEqual(loaded.config.rope_base, 102.0)
                self.assertEqual((vocab, context), (16, 4))

    def test_cli_and_model_mode(self):
        args = evaluate_loss.parse_args(["--model", "saved", "--dataset", "validation",
                                         "--checkpoint", "checkpoints/step_00000002.pt"])
        self.assertEqual(args.model, Path("saved"))
        self.assertEqual(args.checkpoint, Path("checkpoints/step_00000002.pt"))
        model = torch.nn.Embedding(4, 4)
        batches = [(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))]
        for training in (True, False):
            model.train(training)
            self.assertGreater(evaluate_loss.validation_loss(model, batches, torch.device("cpu")), 0)
            self.assertEqual(model.training, training)
        with self.assertRaisesRegex(RuntimeError, "did not produce any tokens"):
            evaluate_loss.validation_loss(model, [], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

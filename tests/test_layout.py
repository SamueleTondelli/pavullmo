"""The dataset builder and training reader share the artifact directory contract."""
from pathlib import Path
import tempfile
import unittest

from dataset import build_dataset, create_modal_volume
from config import PretrainingConfig
from components.data import TokenBlockDataset

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_default_artifact_paths(self):
        expected = ROOT / 'artifacts'
        self.assertEqual(build_dataset.DEFAULT_OUTPUT_DIR, ROOT / 'dataset/ds')
        self.assertEqual(build_dataset.DEFAULT_TOKENIZER, ROOT / 'src/tokenizer/tokenizer.model')
        self.assertEqual(create_modal_volume.DEFAULT_DATASET_DIR, ROOT / 'dataset/ds')
        config = PretrainingConfig(name='layout')
        self.assertEqual(ROOT / config.runtime.dataset_dir, expected / 'datasets')
        self.assertEqual(ROOT / config.runtime.output_dir, expected / 'runs')
        self.assertEqual(ROOT / config.runtime.runs_csv, expected / 'results/pretrain_runs.csv')

    def test_builder_reader_cross_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            builder = build_dataset.ArtifactBuilder(Path(folder), 'validation', 7, False)
            builder.writer.write(list(range(40)))
            builder.finish({})
            data = TokenBlockDataset(Path(folder) / 'validation', 8)
            self.assertEqual(len(data), 4)
            inputs, targets = data[1]
            self.assertEqual(inputs.tolist(), list(range(8, 16)))
            self.assertEqual(targets.tolist(), list(range(9, 17)))


if __name__ == '__main__':
    unittest.main()

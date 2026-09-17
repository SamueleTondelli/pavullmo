"""Checks for local artifact paths and Modal's shallow remote module location."""
import ast
import importlib
import csv
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RepositoryInterfaceTests(unittest.TestCase):
    def test_modal_paths_work_locally_and_under_root(self):
        modal = SimpleNamespace(is_local=lambda: True)

        wrappers = [
            ('src/pavullmo/modal_pretrain_base.py', 'LOCAL_RUNS_CSV', 'pretrain_runs.csv'),
            ('src/pavullmo/modal_pretrain_base_muon.py', 'LOCAL_RUNS_CSV', 'pretrain_runs.csv'),
            ('src/scaling/modal_evaluate_base.py', 'LOCAL_EVALUATIONS_CSV', 'scaling_loss.csv'),
        ]
        for relative, setting, filename in wrappers:
            path = ROOT / relative
            # Execute the real module preamble without constructing cloud resources.
            tree = ast.parse(path.read_text())
            preamble = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'image'
                    for target in node.targets
                ):
                    break
                preamble.append(node)
            code = compile(ast.Module(body=preamble, type_ignores=[]), str(path), 'exec')
            for local in [True, False]:
                with self.subTest(wrapper=relative, local=local), patch.dict('sys.modules', modal=modal), patch.object(modal, 'is_local', return_value=local):
                    namespace = {'__file__': str(path if local else Path('/root') / path.name)}
                    exec(code, namespace)
                    expected = ROOT / 'tmp/results' / filename if local else Path('/outputs') / filename
                    self.assertEqual(namespace[setting], expected)

    def test_scaling_evaluation_resolves_validation_prefix(self):
        module = importlib.import_module('src.scaling.evaluate_base')
        weight = module.torch.nn.Parameter(module.torch.ones(2, 2))
        model = SimpleNamespace(
            parameters=lambda: iter([weight]),
            embeddings=SimpleNamespace(weight=weight),
        )
        for prefix in [None, '', 'production_16k']:
            settings = {'SEQ_LEN': 16, 'DATASET_VARIANT': 'smoke'}
            if prefix is not None:
                settings['DATASET_PREFIX'] = prefix
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
                root = Path(folder)
                train = root / (f'train_{prefix}_smoke' if prefix else 'train_smoke')
                train.mkdir()
                (train / 'metadata.json').write_text(json.dumps({'token_count': 100}))
                output = root / 'evaluation.csv'
                with patch('sys.argv', ['evaluate_base', '--model', 'unused.pt', '--csv', str(output)]), \
                     patch.object(module, 'DATASET_DIR', root), \
                     patch.object(module, 'load_model', return_value=(model, settings)), \
                     patch.object(module, 'TokenBlockDataset') as dataset, \
                     patch.object(module, 'build_loader', return_value=[]), \
                     patch.object(module, 'validation_loss', return_value=0.5), \
                     patch.object(module.torch.cuda, 'is_available', return_value=False):
                    module.main()
                    directory = f'validation_{prefix}' if prefix else 'validation'
                    dataset.assert_called_once_with(root / directory, 16)
                with output.open() as file:
                    row = next(csv.DictReader(file))
                self.assertEqual(row['tokenizer'], prefix or '')
                self.assertEqual(row['train_tokens'], '100')
                self.assertEqual(row['validation_loss'], '0.5')


if __name__ == '__main__':
    unittest.main()

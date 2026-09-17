"""CPU checks for fixed source-stratified validation artifacts and readers."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.dataset.production_pipeline import TEST_MIX, build_artifact, fixed_validation_sample
from src.dataset.build_dataset import ArtifactBuilder


class FixedValidationTests(unittest.TestCase):
    def test_exact_reproducible_source_safe_selection(self):
        sizes = dict(web=4001, wiki=2003, books=2007, edu_pdf=2009)
        sample = fixed_validation_sample(sizes, TEST_MIX, block_count=110, sequence_length=8, seed=42)
        self.assertEqual(sample, fixed_validation_sample(sizes, TEST_MIX, block_count=110, sequence_length=8, seed=42))
        self.assertNotEqual(sample['blocks'], fixed_validation_sample(sizes, TEST_MIX, block_count=110, sequence_length=8, seed=43)['blocks'])
        self.assertEqual(sample['blocks_by_source'], dict(web=44, wiki=22, books=22, edu_pdf=22))
        self.assertEqual(len({b['start_token'] for b in sample['blocks']}), 110)
        offset = 0
        for source, count in sizes.items():
            selected = [b['start_token'] for b in sample['blocks'] if b['source'] == source]
            self.assertTrue(all(offset <= start and start + 8 < offset + count for start in selected))
            offset += count
        with self.assertRaises(ValueError):
            fixed_validation_sample(sizes, TEST_MIX, block_count=112, sequence_length=8, seed=42)
        with self.assertRaises(ValueError):
            fixed_validation_sample(sizes, TEST_MIX, block_count=11000, sequence_length=8, seed=42)

    def test_production_builder_embeds_sample(self):
        def chunks(source, **kwargs):
            for i in range(100):
                yield f'{source.key}-{i}', f'{source.key} text {i}'
        with tempfile.TemporaryDirectory() as tmp, \
             patch('src.dataset.production_pipeline.iter_partition_chunks', side_effect=chunks), \
             patch('src.dataset.production_pipeline.encode_document', return_value=list(range(11))), \
             patch('src.dataset.production_pipeline.tokenizer_metadata', return_value={'vocab_size': 32}):
            build_artifact(name='validation', weights=TEST_MIX, total_tokens=1000,
                           partition='validation', heldout_text_hashes=set(), tokenizer=None,
                           tokenizer_path=Path(tmp)/'tokenizer.model', tokenizer_digest='test',
                           output_dir=Path(tmp), shard_tokens=17, overwrite=False,
                           holdout_permille=20, seed=42, buffer_size=10, documents_dir=None,
                           validation_sample_blocks=10, evaluation_sequence_length=8)
            metadata=json.loads((Path(tmp)/'validation/metadata.json').read_text())
            self.assertEqual(metadata['token_count'], 1000)
            self.assertEqual(metadata['fixed_evaluation_sample']['blocks_by_source'], dict(web=4,wiki=2,books=2,edu_pdf=2))
            self.assertEqual(sum(s['tokens'] for s in metadata['shards']),1000)

    def test_both_readers_preserve_spans_partial_batch_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            sizes=dict(web=400,wiki=200,books=200,edu_pdf=200)
            sample=fixed_validation_sample(sizes,TEST_MIX,block_count=10,sequence_length=8,seed=42)
            builder=ArtifactBuilder(Path(tmp),'validation',17,False)
            builder.writer.write(list(range(1000)))
            builder.finish({'fixed_evaluation_sample':sample})
            for name in ['src.pavullmo.pretrain_base','src.pavullmo.pretrain_base_muon']:
                module=importlib.import_module(name)
                with self.subTest(reader=name), patch.multiple(module,BATCH_SIZE=4,VALIDATION_STEPS=3,NUM_WORKERS=0,PIN_MEMORY=False):
                    dataset=module.TokenBlockDataset(Path(tmp)/'validation',8)
                    self.assertEqual(len(dataset),10)
                    for i,block in enumerate(sample['blocks']):
                        x,y=dataset[i];start=block['start_token']
                        self.assertEqual(x.tolist(),list(range(start,start+8)))
                        self.assertEqual(y.tolist(),list(range(start+1,start+9)))
                    loader=module.build_loader(dataset,shuffle=False)
                    first=[x.tolist() for x,y in loader]
                    self.assertEqual([len(x) for x in first],[4,4,2])
                    self.assertEqual(first,[x.tolist() for x,y in loader])
                    with patch.object(module,'VALIDATION_STEPS',2),self.assertRaisesRegex(ValueError,'complete fixed'):
                        module.build_loader(dataset,shuffle=False)
                    with self.assertRaises(ValueError):module.build_loader(dataset,shuffle=True)
                    with self.assertRaisesRegex(ValueError,'SEQ_LEN'):module.TokenBlockDataset(Path(tmp)/'validation',4)
            path=Path(tmp)/'validation/metadata.json'
            metadata=json.loads(path.read_text());del metadata['fixed_evaluation_sample'];path.write_text(json.dumps(metadata))
            for name in ['src.pavullmo.pretrain_base','src.pavullmo.pretrain_base_muon']:
                dataset=importlib.import_module(name).TokenBlockDataset(Path(tmp)/'validation',8)
                self.assertEqual(len(dataset),124)
                self.assertEqual(dataset[0][0].tolist(),list(range(8)))


if __name__ == '__main__':
    unittest.main()

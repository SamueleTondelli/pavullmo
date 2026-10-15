"""The shared epoch sampler remains restartable with spawned worker prefetch."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from components.data import EpochBatchSampler


class EpochSamplerTests(unittest.TestCase):
    def test_cursor_ignores_spawned_worker_prefetch(self):
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


if __name__ == "__main__":
    unittest.main()

"""PyTorch implementation of the training algorithm for action-value prediction."""

import torch
from torch.utils.data import Dataset
import os
import grain.python as pygrain
import bagz
from itertools import cycle

from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src.data_loader import _process_fen
from searchless_chess.src.data_loader import _TRANSFORMATION_BY_POLICY

class ConvertToTorch(pygrain.MapTransform):
    def map(self, element):
        return tuple(torch.from_numpy(arr) for arr in element)

        

def load_datasource(config: config_lib.DataConfig):
    data_path = os.path.join(
        os.getcwd(),
        config.dataset_path,
    )
    bag_source = bagz.BagDataSource(data_path)
    sampler = pygrain.IndexSampler(
        num_records=len(bag_source),
        shard_options=pygrain.NoSharding(),
        shuffle=config.shuffle,
        seed=config.seed,
        num_epochs=None,
    )

    transformations = (
        _TRANSFORMATION_BY_POLICY[config.policy](num_return_buckets=config.num_return_buckets),
        pygrain.Batch(batch_size=config.batch_size),
        ConvertToTorch(),
    )

    return pygrain.DataLoader(
        data_source=bag_source,
        sampler=sampler,
        operations=transformations,
        worker_count=config.worker_count,
        read_options=pygrain.ReadOptions(
            prefetch_buffer_size=10,
        )
    )
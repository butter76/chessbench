"""PyTorch implementation of the training algorithm for action-value prediction."""

import torch
import os
import grain.python as pygrain
import bagz

from typing import Any, Sequence
from searchless_chess.src import config as config_lib
from searchless_chess.src.data_loader import _TRANSFORMATION_BY_POLICY
from concurrent.futures import ThreadPoolExecutor

class ConvertToTorch(pygrain.MapTransform):
    def map(self, element):
        return tuple(torch.from_numpy(arr) for arr in element)

        

class _InterleavedShardedSequence(Sequence[bytes]):
    """Shards a base sequence across `world_size` by interleaving indices.

    For base length N, ranks 0..(r-1) get ceil(N/world_size) items, others get floor.
    Local index k maps to global index k*world_size + rank.
    """

    def __init__(self, base: Sequence[bytes], world_size: int, rank: int) -> None:
        self._base = base
        self._world_size = int(world_size)
        self._rank = int(rank)
        total = len(base)
        q, r = divmod(total, self._world_size)
        self._length = q + (1 if self._rank < r else 0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> bytes:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError('Index out of range for sharded sequence')
        global_index = index * self._world_size + self._rank
        return self._base[global_index]


def load_datasource(
    config: config_lib.DataConfig,
    *,
    world_size: int | None = None,
    rank: int | None = None,
):
    data_path = os.path.join(
        os.getcwd(),
        config.dataset_path,
    )
    bag_source = bagz.BagDataSource(data_path)
    # If distributed, shard the data source so each rank sees a disjoint view.
    if world_size is not None and rank is not None and world_size > 1:
        bag_source = _InterleavedShardedSequence(bag_source, world_size=world_size, rank=rank)
    sampler = pygrain.IndexSampler(
        num_records=len(bag_source),
        shard_options=pygrain.NoSharding(),
        shuffle=config.shuffle,
        seed=(config.seed if rank is None else (config.seed + int(rank))),
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

class PrefetchIterator:
    """Single-batch lookahead prefetch wrapper.

    Fetches and prepares the next batch in a background thread while the
    current batch is being used by the training step.
    """

    def __init__(self, loader: Any, device: str) -> None:
        self._device = device
        self._it = iter(loader)
        self._executor = ThreadPoolExecutor(max_workers=1)
        # Kick off first fetch+prepare
        self._next_future = self._executor.submit(self._fetch_and_prepare)

    def _prepare_batch(self, batch: Any) -> Any:
        # Expected tuple for lc0_data policy
        (
            x,
            legal_actions,
            policy,
            soft_policy,
            hard_policy,
            hardest_policy,
            hl,
            dhl,
            wdl,
            value_prob,
            draw_prob,
            plies_left,
        ) = batch

        # Cast dtypes on CPU first
        x = x.to(torch.long, copy=False)
        legal_actions = legal_actions.to(torch.float32, copy=False)
        policy = policy.to(torch.float32, copy=False)
        soft_policy = soft_policy.to(torch.float32, copy=False)
        hard_policy = hard_policy.to(torch.float32, copy=False)
        hardest_policy = hardest_policy.to(torch.float32, copy=False)
        hl = hl.to(torch.float32, copy=False)
        dhl = dhl.to(torch.float32, copy=False)
        wdl = wdl.to(torch.long, copy=False)
        value_prob = value_prob.to(torch.float32, copy=False)
        draw_prob = draw_prob.to(torch.float32, copy=False)
        # plies_left is not used downstream; keep dtype as-is

        # Move to device (blocking for simplicity/reliability)
        x = x.to(self._device)
        legal_actions = legal_actions.to(self._device)
        policy = policy.to(self._device)
        soft_policy = soft_policy.to(self._device)
        hard_policy = hard_policy.to(self._device)
        hardest_policy = hardest_policy.to(self._device)
        hl = hl.to(self._device)
        dhl = dhl.to(self._device)
        wdl = wdl.to(self._device)
        value_prob = value_prob.to(self._device)
        draw_prob = draw_prob.to(self._device)
        # Keep plies_left on CPU to avoid extra H2D for unused tensor

        return (
            x,
            legal_actions,
            policy,
            soft_policy,
            hard_policy,
            hardest_policy,
            hl,
            dhl,
            wdl,
            value_prob,
            draw_prob,
            plies_left,
        )

    def _fetch_and_prepare(self) -> Any:
        batch = next(self._it)
        return self._prepare_batch(batch)

    def __iter__(self):
        return self

    def __next__(self):
        ready = self._next_future.result()
        # Immediately schedule the next one
        self._next_future = self._executor.submit(self._fetch_and_prepare)
        return ready
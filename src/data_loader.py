# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements a PyGrain DataLoader for chess data."""

import abc
import os

import grain.python as pygrain
import jax
import numpy as np

from searchless_chess.src import bagz
from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src import utils

from searchless_chess.src.engines import engine
import torch
import chess
import random

NUM_BINS = 81


def _process_prob(
    win_prob: float,
) -> np.ndarray:
  bin_width = 1.0 / NUM_BINS
  sigma = bin_width * 0.75
  bin_centers = np.arange(bin_width / 2, 1.0, bin_width)
  

  diffs = win_prob - bin_centers
  probs = np.exp(-0.5 * (diffs / sigma)**2)
  probs = probs / probs.sum(keepdims=True)
  return probs

def _process_fen(fen: str, move: str | None) -> np.ndarray:
  return tokenizer.tokenize(fen, move).astype(np.int32)


def _process_move(move: str) -> np.ndarray:
  return np.asarray([utils.MOVE_TO_ACTION[move]], dtype=np.int32)


def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
  return utils.compute_return_buckets_from_returns(
      returns=np.asarray([win_prob]),
      bins_edges=return_buckets_edges,
  )


class ConvertToSequence(pygrain.MapTransform, abc.ABC):
  """Base class for converting chess data to a sequence of integers."""

  def __init__(self, num_return_buckets: int) -> None:
    super().__init__()
    self._return_buckets_edges, _ = utils.get_uniform_buckets_edges_values(
        num_return_buckets,
    )


class ConvertBehavioralCloningDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  def map(
      self, element: bytes
  ):
    fen, move = constants.CODERS['behavioral_cloning'].decode(element)
    state = _process_fen(fen)
    action = _process_move(move)
    sequence = np.concatenate([state, action])
    return torch.from_numpy(state), torch.from_numpy(action)


class ConvertStateValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  def map(
      self, element: bytes
  ):
    fen, win_prob = constants.CODERS['state_value'].decode(element)
    state = _process_fen(fen)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, return_bucket])
    return state, np.array([win_prob])


class ConvertActionValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  def map(
      self, element: bytes
  ):
    fen, move, win_prob = constants.CODERS['action_value'].decode(element)
    state = _process_fen(fen)
    action = _process_move(move)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, action, return_bucket])
    return state, np.array([win_prob])

class ConvertActionValuesDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""
  def map(
    self, element: bytes
  ):
  
    fen, move_values = constants.CODERS['action_values'].decode(element)
    legal_actions = np.zeros((64, 64))
    actions = np.zeros((64, 64))

    ## Validation
    # assert len(move_values) == len(engine.get_ordered_legal_moves(chess.Board(fen)))
    assert len(move_values) != 0

    value_prob = 0.0
    for move, win_prob in move_values:
      # Dropping underpromotions for now
      if "=" in move:
        if move[4:] not in ["=Q", "-q"]:
          continue
      s1 = utils._parse_square(move[0:2])
      s2 = utils._parse_square(move[2:4])
      legal_actions[s1, s2] = 1
      actions[s1, s2] = win_prob
      if win_prob > value_prob:
        value_prob = win_prob


    probs = _process_prob(value_prob)
    move, win_prob = random.choice(move_values)

    action_probs = _process_prob(win_prob)

    state = _process_fen(fen, move)

    return state, legal_actions, actions, action_probs, np.array([win_prob]), probs, np.array([value_prob])

_TRANSFORMATION_BY_POLICY = {
    'behavioral_cloning': ConvertBehavioralCloningDataToSequence,
    'action_value': ConvertActionValueDataToSequence,
    'action_values': ConvertActionValuesDataToSequence,
    'state_value': ConvertStateValueDataToSequence,
}

# Follows the base_constants.DataLoaderBuilder protocol.
def build_data_loader(config: config_lib.DataConfig) -> pygrain.DataLoader:
  """Returns a data loader for chess from the config."""
  data_source = bagz.BagDataSource(
      os.path.join(
          os.getcwd(),
          f'../data/{config.split}/{config.policy}_data.bag',
      ),
  )

  if config.num_records is not None:
    num_records = config.num_records
    if len(data_source) < num_records:
      raise ValueError(
          f'[Process {jax.process_index()}]: The number of records requested'
          f' ({num_records}) is larger than the dataset ({len(data_source)}).'
      )
  else:
    num_records = len(data_source)

  sampler = pygrain.IndexSampler(
      num_records=num_records,
      shard_options=pygrain.NoSharding(),
      shuffle=config.shuffle,
      num_epochs=None,
      seed=config.seed,
  )
  transformations = (
      _TRANSFORMATION_BY_POLICY[config.policy](
          num_return_buckets=config.num_return_buckets
      ),
      pygrain.Batch(config.batch_size, drop_remainder=True),
  )
  return pygrain.DataLoader(
      data_source=data_source,
      sampler=sampler,
      operations=transformations,
      worker_count=config.worker_count,
      read_options=None, 
  )
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
import math
from scipy.stats import norm

NUM_BINS = 81
NUM_AV = 50

def _process_prob(
    win_prob: float,
) -> np.ndarray:
  bin_width = 1.0 / NUM_BINS
  sigma = bin_width * 0.75
  bin_starts = np.arange(0.0, 1.0, bin_width)
  bin_ends = bin_starts + bin_width

  probs = norm.cdf(bin_ends, loc=win_prob, scale=sigma) - norm.cdf(bin_starts, loc=win_prob, scale=sigma)
  # Normalize the probabilities
  probs = probs / probs.sum(keepdims=True)
  return probs

def _process_fen(fen: str) -> np.ndarray:
  return tokenizer.tokenize(fen).astype(np.int32)


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
    state = _process_fen(fen)
    # legal_actions = np.zeros((77, 77))
    # actions = np.zeros((77, 77))
    # policy = np.zeros((77, 77))
    # weights = np.zeros((77, 77))
    move_embd = np.zeros((NUM_AV, 77))

    ## Validation
    # assert len(move_values) == len(engine.get_ordered_legal_moves(chess.Board(fen)))
    assert len(move_values) != 0

    value_prob = max(win_prob for _, win_prob in move_values)
    weights = []
    # set_policy = False
    for move, win_prob in move_values:
      # s1 = utils._parse_square(move[0:2])
      # if move[4:] in ['R', 'r']:
      #   s2 = 64
      # elif move[4:] in ['B', 'b']:
      #   s2 = 65
      # elif move[4:] in ['N', 'n']:
      #   s2 = 66
      # else:
      #   assert move[4:] in ['Q', 'q', '']
      #   s2 = utils._parse_square(move[2:4])
      # legal_actions[s1, s2] = 1
      # actions[s1, s2] = win_prob
      # if win_prob == value_prob and not set_policy:
      #   policy[s1, s2] = 1
      #   set_policy = True
      
      if win_prob >= value_prob * 0.95:
        weights.append(1)
      else:
        weights.append(1 / (1 + math.e ** (4 - 4 * (win_prob / (value_prob + 0.01)))))


    probs = _process_prob(value_prob)
    next_state = np.zeros((NUM_AV, 77))
    action_probs = np.zeros((NUM_AV, NUM_BINS), dtype=np.float32)
    win_prob = np.zeros((NUM_AV, 1), dtype=np.float32)
    if len(move_values) > NUM_AV:
      random_moves = move_values[: (NUM_AV // 3)]
      random_moves = random_moves + random.choices(move_values[(NUM_AV // 3):], k=NUM_AV - len(random_moves))
      random.shuffle(random_moves)
    else:
      random_moves = move_values[:]
      random_moves = random_moves + random.choices(move_values, k=NUM_AV - len(random_moves))
      random.shuffle(random_moves)
    for i, (move, prob) in enumerate(random_moves):

      action_prob = _process_prob(prob)


      s1 = utils._parse_square(move[0:2])
      if move[4:] in ['R', 'r']:
        s2 = 64
      elif move[4:] in ['B', 'b']:
        s2 = 65
      elif move[4:] in ['N', 'n']:
        s2 = 66
      else:
        assert move[4:] in ['Q', 'q', '']
        s2 = utils._parse_square(move[2:4])

      move_embd[i, s1] = 1
      move_embd[i, s2] = 2

      board = chess.Board(fen)
      board.push(chess.Move.from_uci(move))

      next_fen = board.fen()
      next_state[i, :] = _process_fen(next_fen)
      win_prob[i, 0] = prob
      action_probs[i, :] = action_prob
    return state, next_state, probs, np.array([value_prob]), action_probs, win_prob, move_embd

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
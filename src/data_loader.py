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
S = tokenizer.SEQUENCE_LENGTH

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
  return tokenizer.tokenize(fen, useRule50=True).astype(np.int32)


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
    stm = fen.split(' ')[1]
    flip = stm == 'b'
    state = _process_fen(fen)
    legal_actions = np.zeros((S, S))
    actions = np.zeros((S, S))
    policy = np.zeros((S, S))
    weights = np.zeros((S, S))

    ## Validation
    # assert len(move_values) == len(engine.get_ordered_legal_moves(chess.Board(fen)))
    assert len(move_values) != 0

    value_prob = max(win_prob for _, win_prob in move_values)
    for move, win_prob in move_values:
      s1 = utils._parse_square(move[0:2], flip)
      if move[4:] in ['R', 'r']:
        s2 = 64
      elif move[4:] in ['B', 'b']:
        s2 = 65
      elif move[4:] in ['N', 'n']:
        s2 = 66
      else:
        assert move[4:] in ['Q', 'q', '']
        s2 = utils._parse_square(move[2:4], flip)
      legal_actions[s1, s2] = 1
      actions[s1, s2] = win_prob
      if win_prob == value_prob:
        policy[s1, s2] = 1
      
      if win_prob >= value_prob * 0.95:
        weights[s1, s2] = 1
      else:
        weights[s1, s2] = 1 / (1 + math.e ** (4 - 4 * (win_prob / (value_prob + 0.01))))


    probs = _process_prob(value_prob)
    policy = policy / policy.sum()

    return state, legal_actions, actions, probs, np.array([value_prob]), policy, weights
  

class ConvertLeelaDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""
  def map(
    self, element: bytes
  ):
    fen, leela_policy, result, root_q, root_d, played_q, played_d, plies_left, _move = constants.CODERS['lc0_data'].decode(element)
    
    state = _process_fen(fen)
    Q = (root_q + 1) / 2
    D = root_d

    # board = chess.Board(fen)
    # board.generate_legal_moves()
    # assert len(leela_policy) == len(list(board.legal_moves))

    legal_actions = np.zeros((S, S))
    policy = np.zeros((S, S))
    soft_policy = np.zeros((S, S))
    hard_policy = np.zeros((S, S))
    hardest_policy = np.zeros((S, S))
    flip = fen.split(' ')[1] == 'b'

    max_p = max(p for _, p in leela_policy)
    assert max_p > 0, f'max_p is 0, board:{fen}'
    # First pass to get raw policy values
    for move, p in leela_policy:
        s1, s2 = utils.move_to_indices(chess.Move.from_uci(move), flip)
        policy[s1, s2] = p
        assert legal_actions[s1, s2] == 0, f'{move} is already in the legal actions, board:{fen}'
        legal_actions[s1, s2] = 1
        if p >= max_p * 0.98:
            hardest_policy[s1, s2] = 1

    # Compute temperature-adjusted policies
    # Flatten for numerical stability
    flat_policy = policy.reshape(-1)

    hardest_policy = hardest_policy / hardest_policy.sum()

    if _move != 'CDB':
    
      # Hard policy (temp=0.25)
      # Add small epsilon to avoid log(0)
      eps = 1e-8
      logits = np.where(flat_policy > 0, np.log(flat_policy + eps), -np.inf) / 0.25
      exp_logits = np.exp(logits - np.max(logits))
      hard_policy = (exp_logits / exp_logits.sum()).reshape(S, S)

      # Soft policy (temp=4)
      logits = np.where(flat_policy > 0, np.log(flat_policy + eps), -np.inf) / 4
      exp_logits = np.exp(logits - np.max(logits))
      soft_policy = (exp_logits / exp_logits.sum()).reshape(S, S)
    else:
      policy = np.zeros((S, S))
      hard_policy = hardest_policy.copy()

    wdl = np.array([int(-result + 1)])
    probs = _process_prob(Q)
    draw_probs = _process_prob(D)

    # if _move == 'CDB':
    #   draw_probs = np.zeros(NUM_BINS)

    return state, legal_actions, policy, soft_policy, hard_policy, hardest_policy, probs, draw_probs, wdl, np.array([Q]), np.array([D]), np.array([plies_left])


class ConvertLeelaDataToSequenceWithU(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""
  def map(
    self, element: bytes
  ):
    fen, policy, result, root_q, root_d, played_q, played_d, plies_left, _, U_moves = constants.CODERS['lc0_data_with_U'].decode(element)
    state = _process_fen(fen)

    legal_actions = np.zeros((S, S))
    U_values = np.zeros((S, S))
    Q_values = np.zeros((S, S))
    D_values = np.zeros((S, S))
    flip = fen.split(' ')[1] == 'b'

    for move, u_val, q_val, d_val in U_moves:
      s1, s2 = utils.move_to_indices(chess.Move.from_uci(move), flip)
      U_values[s1, s2] = (u_val * 4 + 1e-6) ** 0.5
      legal_actions[s1, s2] = 1
      Q_values[s1, s2] = q_val
      D_values[s1, s2] = d_val

    return state, legal_actions, U_values, Q_values, D_values


_TRANSFORMATION_BY_POLICY = {
    'behavioral_cloning': ConvertBehavioralCloningDataToSequence,
    'action_value': ConvertActionValueDataToSequence,
    'action_values': ConvertActionValuesDataToSequence,
    'state_value': ConvertStateValueDataToSequence,
    'lc0_data': ConvertLeelaDataToSequence,
    'lc0_data_with_U': ConvertLeelaDataToSequenceWithU,
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
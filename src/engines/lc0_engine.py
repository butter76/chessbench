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

"""Implements a Leela Chess Zero engine."""

from enum import Enum
import os

import chess.engine

from searchless_chess.src.engines import engine

class Lc0Network(str, Enum):
  T1D = 't1d',
  T1S = 't1s',
  T3D = 't3d',
  T82 = 't82',
  BT4 = 'bt4',
  BT3 = 'bt3',
  LS15 = 'ls15',

class Lc0Engine(engine.Engine):
  """Leela Chess Zero with the biggest available network.

  WARNING: This can only be used with CUDA (i.e. Nvidia GPUs) for now.
  """

  def __init__(
      self,
      limit: chess.engine.Limit,
      network: Lc0Network = Lc0Network.T1D,
  ) -> None:
    self._limit = limit
    bin_path = os.path.join(
        os.getcwd(),
        '../lc0/build/release/lc0',
    )
    # We use the biggest available network.
    if network == Lc0Network.T1D:
      network_path = '../lc0/build/release/t1-512x15x8h-distilled-swa-3395000.pb'
    elif network == Lc0Network.T1S:
      network_path = '../lc0/build/release/t1-256x10-distilled-swa-2432500.pb'
    elif network == Lc0Network.T3D:
      network_path = '../lc0/build/release/t3-512x15x16h-distill-swa-2767500.pb'
    elif network == Lc0Network.T82:
      network_path = '../lc0/build/release/768x15x24h-t82-swa-7464000.pb'
    elif network == Lc0Network.BT4:
      network_path = '../lc0/build/release/BT4-1740.pb'
    elif network == Lc0Network.BT3:
      network_path = '../lc0/build/release/BT3-768x15x24h-swa-2790000.pb'
    elif network == Lc0Network.LS15:
      network_path = '../lc0/build/release/LS15-20x256SE-jj-9-75000000.pb'
    else:
      raise ValueError(f'Network {network} not found.')
    weights_path = os.path.join(
        os.getcwd(),
        network_path,
    )
    options = [f'--weights={weights_path}']
    self._raw_engine = chess.engine.SimpleEngine.popen_uci(
        command=[bin_path] + options,
    )
    self._raw_engine.configure({'Threads': 1, 'MinibatchSize': 1})

  def __del__(self) -> None:
    self._raw_engine.close()

  @property
  def limit(self) -> chess.engine.Limit:
    return self._limit

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns various analysis results from the Lc0 engine."""
    outcome = board.outcome()
    if outcome is not None:
      # The game has now ended.
      if outcome.winner is None:
        score = chess.engine.Cp(0)
      elif outcome.winner == board.turn:
        score = -chess.engine.Mate(moves=0)
      else:
        score = chess.engine.Mate(moves=0)
      return {'score': chess.engine.PovScore(score, turn=board.turn)}
    return self._raw_engine.analyse(board, limit=self._limit)

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from the Lc0 engine."""
    best_move = self._raw_engine.play(board, limit=self._limit).move
    if best_move is None:
      raise ValueError('No best move found, something went wrong.')
    return best_move


class AllMovesLc0Engine(Lc0Engine):
  """A version of Lc0 that evaluates all moves individually."""

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from Lc0."""
    scores = []
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    for move in sorted_legal_moves:
      board.push(move)
      results = super().analyse(board)
      board.pop()
      scores.append((move, -results['score'].relative))
    return {'scores': scores}

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from Lc0."""
    scores = self.analyse(board)['scores']
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0]

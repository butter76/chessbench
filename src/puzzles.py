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

"""Evaluates engines on the puzzles dataset from lichess."""

from collections.abc import Sequence
import io
import os

from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import pandas as pd

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines.my_engine import MoveSelectionStrategy, MyTransformerEngine


_NUM_PUZZLES = flags.DEFINE_integer(
    name='num_puzzles',
    default=None,
    help='The number of puzzles to evaluate.',
    required=True,
)

_STRATEGY = flags.DEFINE_enum(
    name='strategy',
    default='value',
    enum_values=[
        'value',
        'avs',
        'avs2',
        'policy',
        'policy_split',
        'opt_policy_split',
        'negamax',
        'alpha_beta',
    ],
    help='The move selection strategy to use for my_engine.',
)


def evaluate_puzzle_from_pandas_row(
    puzzle: pd.Series,
    engine: engine_lib.Engine,
) -> bool:
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
  if game is None:
    raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
  board = game.end().board()
  return evaluate_puzzle_from_board(
      board=board,
      moves=puzzle['Moves'].split(' '),
      engine=engine,
  )


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine: engine_lib.Engine,
) -> bool:
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  for move_idx, move in enumerate(moves):
    # According to https://database.lichess.org/#puzzles, the FEN is the
    # position before the opponent makes their move. The position to present to
    # the player is after applying the first move to that FEN. The second move
    # is the beginning of the solution.
    if move_idx % 2 == 1:
      predicted_move = engine.play(board=board).uci()
      # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
      # check if the `predicted_move` results in a checkmate if it differs from
      # the solution.
      if move != predicted_move:
        board.push(chess.Move.from_uci(predicted_move))
        return board.is_checkmate()
    board.push(chess.Move.from_uci(move))
  return True


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  puzzles_path = os.path.join(
      os.getcwd(),
      '../data/puzzles.csv',
  )
  puzzles = pd.read_csv(puzzles_path, nrows=_NUM_PUZZLES.value)

  for strategy in [MoveSelectionStrategy.VALUE, MoveSelectionStrategy.AVS, MoveSelectionStrategy.AVS2, MoveSelectionStrategy.POLICY, MoveSelectionStrategy.OPT_POLICY_SPLIT]:
    engine = MyTransformerEngine(
        '../checkpoints/p1-standard/checkpoint_300000.pt',
        chess.engine.Limit(nodes=1),
        strategy=strategy,
    )
    with open(f'puzzles-{strategy}.txt', 'w') as f:
        num_correct = 0
        for puzzle_id, puzzle in puzzles.iterrows():
            correct = evaluate_puzzle_from_pandas_row(
                puzzle=puzzle,
                engine=engine,
            )
            num_correct += correct
            f.write(str({'puzzle_id': puzzle_id, 'correct': correct, 'rating': puzzle['Rating']}) + '\n')
        print(f'{strategy}: {num_correct / len(puzzles):.2%}')


if __name__ == '__main__':
  app.run(main)

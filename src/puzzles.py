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
import numpy as np
import pandas as pd
import bagz
from tqdm import tqdm

from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines.lc0_engine import AllMovesLc0Engine, Lc0Engine
from searchless_chess.src.engines.stockfish_engine import StockfishEngine
from searchless_chess.src.engines.my_engine import MoveSelectionStrategy, MyTransformerEngine
from searchless_chess.src.constants import CODERS
import searchless_chess.src.utils as utils
from apache_beam import coders

lichess_coder = coders.TupleCoder([
    coders.StrUtf8Coder(),
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
])


_NUM_PUZZLES = flags.DEFINE_integer(
    name='num_puzzles',
    default=10000,
    help='The number of puzzles to evaluate.',
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
  board = chess.Board(puzzle['FEN'])
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

def validate_chessbench_policy(engine: engine_lib.Engine):
    """Sample positions from validation set and compare MyEngine policy with dataset policy."""
    print("Validating MyEngine policy against dataset policy...")

    # Create validation dataloader

    bag_source = bagz.BagDataSource('../data/output/validation.bag')
    
    # Track metrics
    total_positions = 0
    total_top1_match = 0.0
    
    # Sample positions
    sample_size = min(10000, 1_000_000)

    val_iter = iter(bag_source)

    explore = False
    for i in range(1000000):
        element = next(val_iter)
        if (i % 100 == 0):
            explore = True
        if not explore:
            continue

        fen, move_values = CODERS['action_values'].decode(element)
        policy = np.zeros((77, 77))
        value_prob = max(win_prob for _, win_prob in move_values)
        if value_prob > 0.75 or value_prob < 0.25 or 0.47 < value_prob < 0.53:
            continue
        for move, win_prob in move_values:
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
            policy[s1, s2] = win_prob
            if win_prob == value_prob:
                policy[s1, s2] = 1

        board = chess.Board(fen)
        best_move = engine.play(board).uci()
        best_s1 = utils._parse_square(best_move[0:2])
        if best_move[4:] in ['R', 'r']:
            best_s2 = 64
        elif best_move[4:] in ['B', 'b']:
            best_s2 = 65
        elif best_move[4:] in ['N', 'n']:
            best_s2 = 66
        else:
            best_s2 = utils._parse_square(best_move[2:4])
        total_positions += 1
        total_top1_match += policy[best_s1, best_s2]
        explore = False

    
    # Report final results
    print(f"Chessbench Top-1 move match rate: {total_top1_match/total_positions:.4f}")


def validate_lichess_policy(engine: engine_lib.Engine):
    """Sample positions from validation set and compare MyEngine policy with dataset policy."""
    print("Validating MyEngine policy against dataset policy...")

    # Create validation dataloader
    bag_source = bagz.BagDataSource('../data/lichess_data.bag')

    val_iter = iter(bag_source)

    # Track metrics
    total_positions = 0
    total_top1_match = 0.0

    explore = False
    for i in range(1000000):
        if (i % 100 == 0):
            explore = True
        if not explore:
            continue

        element = next(val_iter)
        fen, move, win_prob = lichess_coder.decode(element)
        if not (0.564 < win_prob < 0.65):
            continue
        board = chess.Board(fen)
        best_move = engine.play(board).uci()
        if move == best_move:
            total_top1_match += 1
        total_positions += 1
        explore = False

    print(f"Lichess Top-1 move match rate: {total_top1_match/total_positions:.4f}")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # checkpoint_path = '../checkpoints/p1/checkpoint_480000.pt'
    checkpoint_path = '../checkpoints/p1-standard-take2/checkpoint_300000.pt'
    # engine = MyTransformerEngine(checkpoint_path=checkpoint_path, limit=chess.engine.Limit(nodes=1), strategy=MoveSelectionStrategy.POLICY)
    # engine = AllMovesLc0Engine(chess.engine.Limit(nodes=1))
    # engine = Lc0Engine(chess.engine.Limit(nodes=800))
    # engine = StockfishEngine(chess.engine.Limit(nodes=1000_000))
    # validate_lichess_policy(engine)
    # validate_chessbench_policy(engine)


    puzzles_path = os.path.join(
        os.getcwd(),
        '../data/high_rated_puzzles.csv',
    )
    puzzles = pd.read_csv(puzzles_path, nrows=_NUM_PUZZLES.value).sample(frac=1)

    for strategy in [MoveSelectionStrategy.ALPHA_BETA_NODE]:
        engine = MyTransformerEngine(
            checkpoint_path,
            chess.engine.Limit(nodes=1),
            strategy=strategy,
            search_depth=600,
        )

        with open(f'puzzles-{strategy}.txt', 'w') as f:
            num_correct = 0
            pbar = tqdm(puzzles.iterrows(), total=len(puzzles), desc=f"Evaluating puzzles ({strategy})")
            num_iterations = 0
            for puzzle_id, puzzle in pbar:
                correct = evaluate_puzzle_from_pandas_row(
                    puzzle=puzzle,
                    engine=engine,
                )
                num_correct += correct
                num_iterations += 1
                f.write(str({'puzzle_id': puzzle_id, 'correct': correct, 'rating': puzzle['Rating']}) + '\n')
                pbar.set_postfix({
                    'accuracy': f'{num_correct / num_iterations:.2%}',
                    'avg_nodes': f'{engine.metrics["num_nodes"] / engine.metrics["num_searches"]:.2f}',
                })
            print(f'{strategy}: {num_correct / len(puzzles):.2%}')
            print(f'{strategy}: {engine.metrics}')


if __name__ == '__main__':
  app.run(main)

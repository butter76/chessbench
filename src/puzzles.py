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
from searchless_chess.src.engines.strategy import MoveSelectionStrategy
from searchless_chess.src.engines.my_engine import MyTransformerEngine
from searchless_chess.src.constants import CODERS
import searchless_chess.src.utils as utils
from apache_beam import coders

lichess_coder = coders.TupleCoder([
    coders.StrUtf8Coder(),
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
])

_STRATEGY = flags.DEFINE_enum(
    name='strategy',
    default='pvs',
    enum_values=[
        'value',
        'avs',
        'avs2',
        'policy',
        'policy_split',
        'opt_policy_split',
        'negamax',
        'alpha_beta',
        'alpha_beta_node',
        'mcts',
        'mtdf',
        'pvs',
    ],
    help='The move selection strategy to use for my_engine.',
)

_TYPE = flags.DEFINE_enum(
    name='type',
    default='tactical',
    enum_values=[
        'positional',
        'tactical',
    ],
    help='The type of puzzles to evaluate.',
)

_DEPTH = flags.DEFINE_float(
    name='depth',
    default=6.73,
    help='The search depth to use for search-based strategies.',
)

_NETWORK = flags.DEFINE_enum(
    name='network',
    default=None,
    enum_values=['t1d', 't1s', 't3d', 't82', 'bt4', 'bt3', 'ls15'],
    help='The network to use for Lc0Engine.',
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

def validate_chessbench(engine: engine_lib.Engine):
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


def validate_lichess(engine: engine_lib.Engine):
    """Sample positions from validation set and compare MyEngine policy with dataset policy."""
    print("Validating MyEngine policy against dataset policy...")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    strategy = _STRATEGY.value
    MY_ENGINE = False
    if _NETWORK.value is not None:
        if strategy == 'value':
            # DEPTH 1 SEARCH
            engine = AllMovesLc0Engine(chess.engine.Limit(nodes=1), network=_NETWORK.value)
        else:
            # 400 NODE MCTS
            engine = Lc0Engine(chess.engine.Limit(nodes=400), network=_NETWORK.value)
    else:
        MY_ENGINE = True
        # checkpoint_path = '../checkpoints/p1/checkpoint_480000.pt'
        checkpoint_path = '../checkpoints/p1-standard-take2/checkpoint_300000.pt'
        engine = MyTransformerEngine(
            checkpoint_path,
            chess.engine.Limit(nodes=1),
            strategy=strategy,
            search_depth=_DEPTH.value if strategy != 'mcts' else 400,
        )

    if _TYPE.value == 'tactical':
        # Evaluates on 10_000 random puzzles from Lichess' most difficult puzzles
        # Puzzles are many moves long, and every move is an only move or a checkmate
        # This is a good proxy for the tactical accuracy needed to evaluate subtrees

        puzzles_path = os.path.join(
            os.getcwd(),
            '../data/high_rated_puzzles.csv',
        )
        puzzles = pd.read_csv(puzzles_path, nrows=10000).sample(frac=1)

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
                stats = {
                    'accuracy': f'{num_correct / num_iterations:.2%}',
                }
                if MY_ENGINE:
                    stats['nodes'] = f'{engine.metrics["num_nodes"] / engine.metrics["num_searches"]:.2f}'
                    stats['perplexity'] = f'{engine.metrics["policy_perplexity"] / max(1, engine.metrics["num_nodes"]):.2f}'
                    stats['depth'] = f'{engine.metrics["depth"] / engine.metrics["num_searches"]:.2f}'
                pbar.set_postfix(stats)
            print(f'{strategy}: {num_correct / len(puzzles):.2%}')

    elif _TYPE.value == 'positional':
        # Evaluates on 10_000 random positions from lichess that were analyzed by Stockfish
        # for at least 1 billion nodes on positions that are not obviously drawn or won
        # which was estimated to be positions between +0.70 and +1.40 evaluation
        # This is a good proxy for the positional accuracy needed in a real game
        # However, it is not possible to reach 100% accuracy on this dataset, as the
        # Stockfish evaluation is not perfect

        # Create validation dataloader
        bag_source = bagz.BagDataSource('../data/lichess_data.bag')

        val_iter = iter(bag_source)

        # Track metrics
        total_positions = 0
        total_top1_match = 0.0

        explore = False
        pbar = tqdm(range(10000), desc=f"Evaluating positions ({strategy})")
        
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

            pbar.update(1)

            stats = {
                'accuracy': f'{total_top1_match / total_positions:.2%}' if total_positions > 0 else '0.00%',
            }
            if MY_ENGINE:
                stats['nodes'] = f'{engine.metrics["num_nodes"] / engine.metrics["num_searches"]:.2f}'
                stats['perplexity'] = f'{engine.metrics["policy_perplexity"] / max(1, engine.metrics["num_nodes"]):.2f}'
                stats['depth'] = f'{engine.metrics["depth"] / engine.metrics["num_searches"]:.2f}'
            
            pbar.set_postfix(stats)

        print(f"Lichess Top-1 move match rate: {total_top1_match/total_positions:.4f}")


if __name__ == '__main__':
  app.run(main)

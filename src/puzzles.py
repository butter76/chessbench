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
import multiprocessing as mp
import time

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
from searchless_chess.src.engines.remote_engine import RemoteInferenceEngine
from searchless_chess.src.engines.inference_server import start_inference_server
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

_NUM_PROCESSES = flags.DEFINE_integer(
    name='num_processes',
    default=None,
    help='Number of processes to use for multiprocessing. Defaults to CPU count.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    name='batch_size',
    default=32,
    help='Maximum batch size for inference server.',
)

_BATCH_TIMEOUT_MS = flags.DEFINE_integer(
    name='batch_timeout_ms',
    default=10,
    help='Batch timeout in milliseconds for inference server.',
)

# Global variables for worker processes
worker_engine = None
request_queue = None
response_queue = None

def init_worker(strategy, network, depth, req_queue, resp_queue, checkpoint_path=None):
    """Initialize worker process with an engine instance."""
    global worker_engine, request_queue, response_queue
    request_queue = req_queue
    response_queue = resp_queue
    
    if network is not None:
        if strategy == 'value':
            # DEPTH 1 SEARCH
            worker_engine = AllMovesLc0Engine(chess.engine.Limit(nodes=1), network=network)
        else:
            # 400 NODE MCTS
            worker_engine = Lc0Engine(chess.engine.Limit(nodes=400), network=network)
    else:
        # Use remote inference engine for transformer models
        worker_engine = RemoteInferenceEngine(
            request_queue=request_queue,
            response_queue=response_queue,
            limit=chess.engine.Limit(nodes=1),
            strategy=strategy,
            search_depth=depth if strategy != 'mcts' else 400,
            worker_id=os.getpid(),
        )

def create_engine(strategy, network, depth, checkpoint_path=None):
    """Create an engine instance with the given parameters."""
    if network is not None:
        if strategy == 'value':
            # DEPTH 1 SEARCH
            engine = AllMovesLc0Engine(chess.engine.Limit(nodes=1), network=network)
        else:
            # 400 NODE MCTS
            engine = Lc0Engine(chess.engine.Limit(nodes=400), network=network)
    else:
        if checkpoint_path is None:
            checkpoint_path = '../checkpoints/p1-standard-take2/checkpoint_300000.pt'
        engine = MyTransformerEngine(
            checkpoint_path,
            chess.engine.Limit(nodes=1),
            strategy=strategy,
            search_depth=depth if strategy != 'mcts' else 400,
        )
    return engine

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

def worker_evaluate_puzzle(puzzle_data):
    """Worker function for evaluating a single puzzle using the global engine."""
    global worker_engine
    puzzle_id, puzzle = puzzle_data
    try:
        correct = evaluate_puzzle_from_pandas_row(puzzle, worker_engine)
        metrics = {}
        if hasattr(worker_engine, 'metrics'):
            metrics = worker_engine.metrics.copy()
        return {
            'puzzle_id': puzzle_id,
            'correct': correct,
            'rating': puzzle['Rating'],
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error evaluating puzzle {puzzle_id}: {e}")
        return {
            'puzzle_id': puzzle_id,
            'correct': False,
            'rating': puzzle['Rating'],
            'metrics': {}
        }

def worker_evaluate_position(position_data):
    """Worker function for evaluating a single position using the global engine."""
    global worker_engine
    fen, expected_move, win_prob = position_data
    try:
        board = chess.Board(fen)
        predicted_move = worker_engine.play(board).uci()
        correct = (expected_move == predicted_move)
        metrics = {}
        if hasattr(worker_engine, 'metrics'):
            metrics = worker_engine.metrics.copy()
        return {
            'correct': correct,
            'win_prob': win_prob,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error evaluating position {fen}: {e}")
        return {
            'correct': False,
            'win_prob': win_prob,
            'metrics': {}
        }

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
    network = _NETWORK.value
    depth = _DEPTH.value
    num_processes = _NUM_PROCESSES.value or mp.cpu_count()
    batch_size = _BATCH_SIZE.value
    batch_timeout_ms = _BATCH_TIMEOUT_MS.value
    
    MY_ENGINE = (network is None)
    checkpoint_path = '../checkpoints/p1-standard-take2/checkpoint_300000.pt' if MY_ENGINE else None

    print(f"Using {num_processes} processes for evaluation")
    
    # Initialize inference server for transformer models
    inference_server_process = None
    req_queue = None
    resp_queue = None
    
    if MY_ENGINE:
        print(f"Starting inference server with batch_size={batch_size}, timeout={batch_timeout_ms}ms")
        req_queue = mp.Queue()
        resp_queue = mp.Queue()
        
        inference_server_process = mp.Process(
            target=start_inference_server,
            args=(checkpoint_path, req_queue, resp_queue, batch_size, batch_timeout_ms)
        )
        inference_server_process.start()
        
        # Give the server time to initialize
        time.sleep(2)
        print("Inference server started")

    try:
        if _TYPE.value == 'tactical':
            # Evaluates on 10_000 random puzzles from Lichess' most difficult puzzles
            # Puzzles are many moves long, and every move is an only move or a checkmate
            # This is a good proxy for the tactical accuracy needed to evaluate subtrees

            puzzles_path = os.path.join(
                os.getcwd(),
                '../data/high_rated_puzzles.csv',
            )
            puzzles = pd.read_csv(puzzles_path, nrows=10000).sample(frac=1)

            # Prepare data for multiprocessing
            puzzle_data = [(puzzle_id, puzzle) for puzzle_id, puzzle in puzzles.iterrows()]
            
            with open(f'puzzles-{strategy}.txt', 'w') as f:
                num_correct = 0
                num_iterations = 0
                
                # Use multiprocessing Pool with initializer
                with mp.Pool(
                    processes=num_processes, 
                    initializer=init_worker,
                    initargs=(strategy, network, depth, req_queue, resp_queue, checkpoint_path)
                ) as pool:
                    # Use imap for streaming results with progress bar
                    pbar = tqdm(
                        pool.imap(worker_evaluate_puzzle, puzzle_data),
                        total=len(puzzle_data),
                        desc=f"Evaluating puzzles ({strategy})"
                    )
                    
                    # Aggregate metrics
                    total_metrics = {
                        'num_nodes': 0,
                        'num_searches': 0,
                        'policy_perplexity': 0,
                        'depth': 0,
                    }
                    
                    for result in pbar:
                        num_correct += result['correct']
                        num_iterations += 1
                        f.write(str(result) + '\n')
                        
                        # Aggregate metrics if available
                        if result['metrics']:
                            for key in total_metrics:
                                if key in result['metrics']:
                                    total_metrics[key] += result['metrics'][key]
                        
                        stats = {
                            'accuracy': f'{num_correct / num_iterations:.2%}',
                        }
                        if MY_ENGINE and total_metrics['num_searches'] > 0:
                            stats['nodes'] = f'{total_metrics["num_nodes"] / total_metrics["num_searches"]:.2f}'
                            stats['perplexity'] = f'{total_metrics["policy_perplexity"] / max(1, total_metrics["num_nodes"]):.2f}'
                            stats['depth'] = f'{total_metrics["depth"] / total_metrics["num_searches"]:.2f}'
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

            # Collect positions to evaluate
            positions_to_evaluate = []
            explore = False
            
            print("Collecting positions to evaluate...")
            for i in range(1000000):
                if len(positions_to_evaluate) >= 10000:
                    break
                    
                if (i % 100 == 0):
                    explore = True
                if not explore:
                    continue

                element = next(val_iter)
                fen, move, win_prob = lichess_coder.decode(element)
                if not (0.564 < win_prob < 0.65):
                    continue
                    
                positions_to_evaluate.append((fen, move, win_prob))
                explore = False

            print(f"Evaluating {len(positions_to_evaluate)} positions...")
            
            # Track metrics
            total_positions = 0
            total_top1_match = 0.0
            total_metrics = {
                'num_nodes': 0,
                'num_searches': 0,
                'policy_perplexity': 0,
                'depth': 0,
            }
            
            # Use multiprocessing Pool with initializer
            with mp.Pool(
                processes=num_processes,
                initializer=init_worker,
                initargs=(strategy, network, depth, req_queue, resp_queue, checkpoint_path)
            ) as pool:
                pbar = tqdm(
                    pool.imap(worker_evaluate_position, positions_to_evaluate),
                    total=len(positions_to_evaluate),
                    desc=f"Evaluating positions ({strategy})"
                )
                
                for result in pbar:
                    if result['correct']:
                        total_top1_match += 1
                    total_positions += 1
                    
                    # Aggregate metrics if available
                    if result['metrics']:
                        for key in total_metrics:
                            if key in result['metrics']:
                                total_metrics[key] += result['metrics'][key]

                    stats = {
                        'accuracy': f'{total_top1_match / total_positions:.2%}' if total_positions > 0 else '0.00%',
                    }
                    if MY_ENGINE and total_metrics['num_searches'] > 0:
                        stats['nodes'] = f'{total_metrics["num_nodes"] / total_metrics["num_searches"]:.2f}'
                        stats['perplexity'] = f'{total_metrics["policy_perplexity"] / max(1, total_metrics["num_nodes"]):.2f}'
                        stats['depth'] = f'{total_metrics["depth"] / total_metrics["num_searches"]:.2f}'
                    
                    pbar.set_postfix(stats)

            print(f"Lichess Top-1 move match rate: {total_top1_match/total_positions:.4f}")

    finally:
        # Clean up inference server
        if inference_server_process is not None:
            print("Shutting down inference server...")
            # Send shutdown signal
            req_queue.put(None)
            inference_server_process.join(timeout=5)
            if inference_server_process.is_alive():
                print("Force terminating inference server...")
                inference_server_process.terminate()
                inference_server_process.join()
            print("Inference server shut down")


if __name__ == '__main__':
  app.run(main)

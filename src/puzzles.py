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
import signal
import math

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
from searchless_chess.src.engines.uci_engine import UCIEngine, AllMovesUCIEngine
from searchless_chess.src.engines.stockfish_engine import StockfishEngine
from searchless_chess.src.engines.strategy import MoveSelectionStrategy
from searchless_chess.src.engines.my_engine import MyTransformerEngine
from searchless_chess.src.constants import CODERS
import searchless_chess.src.utils as utils
from apache_beam import coders

class WorkerTimeoutError(Exception):
    """Raised when worker evaluation takes longer than the allowed timeout."""
    pass

def worker_timeout_handler(signum, frame):
    """Signal handler for worker timeout."""
    raise WorkerTimeoutError("Worker evaluation timed out after 10 minutes")

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
        'soft_policy',
        'hard_policy',
        'hardest_policy',
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
        'blunderbase',
        'fortressbase',
    ],
    help='The type of puzzles to evaluate.',
)

_DEPTH = flags.DEFINE_float(
    name='depth',
    default=6.73,
    help='The search depth to use for search-based strategies.',
)

_NUM_NODES = flags.DEFINE_integer(
    name='num_nodes',
    default=400,
    help='The number of nodes to use.',
)

_NETWORK = flags.DEFINE_enum(
    name='network',
    default=None,
    enum_values=['t1d', 't1s', 't3d', 't82', 'bt4', 'bt3', 'ls15', 'bt5'],
    help='The network to use for Lc0Engine.',
)

_CHECKPOINT = flags.DEFINE_string(
    name='checkpoint',
    default=None,
    help='The checkpoint to use for MyTransformerEngine.',
)

_UCI_ENGINE_BIN = flags.DEFINE_string(
    name='uci_engine_bin',
    default=None,
    help='Path to a UCI engine binary to evaluate (e.g., C++ search engine).',
)


_NUM_PROCESSES = flags.DEFINE_integer(
    name='num_processes',
    default=8,
    help='Number of processes to use for multiprocessing.',
)

_NAME = flags.DEFINE_string(
    name='name',
    default=None,
    help='The name of the experiment.',
)

# Global variable to store the engine in each worker process
worker_engine = None
worker_num_nodes = None

def init_worker(strategy, network, depth, num_nodes, checkpoint_path, uci_engine_bin):
    """Initialize worker process with an engine instance."""
    global worker_engine
    global worker_num_nodes
    worker_num_nodes = num_nodes
    if network is not None:
        if strategy == 'value':
            # DEPTH 1 SEARCH
            worker_engine = AllMovesLc0Engine(chess.engine.Limit(nodes=1), network=network)
        else:
            # NODE MCTS
            worker_engine = Lc0Engine(chess.engine.Limit(nodes=num_nodes), network=network)
    else:
        # Prefer external UCI engine if provided
        if uci_engine_bin is not None:
            if strategy == 'value':
                # Evaluate all moves at minimal cost
                worker_engine = AllMovesUCIEngine(
                    engine_path=uci_engine_bin,
                    limit=chess.engine.Limit(nodes=1),
                )
            else:
                worker_engine = UCIEngine(
                    engine_path=uci_engine_bin,
                    limit=chess.engine.Limit(nodes=num_nodes),
                )
        else:
            worker_engine = MyTransformerEngine(
                checkpoint_path,
                chess.engine.Limit(nodes=1),
                strategy=strategy,
                search_depth=depth,
                num_nodes=num_nodes,
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

def worker_evaluate_puzzle(puzzle_data):
    """Worker function for evaluating a single puzzle using the global engine."""
    global worker_engine
    puzzle_id, puzzle = puzzle_data
    
    # Set up timeout (10 minutes = 600 seconds)
    old_handler = signal.signal(signal.SIGALRM, worker_timeout_handler)
    signal.alarm(600)  # 10 minutes timeout
    
    try:
        correct = evaluate_puzzle_from_pandas_row(puzzle, worker_engine)
        metrics = {}
        if hasattr(worker_engine, 'metrics'):
            metrics = worker_engine.metrics.copy()
        result = {
            'puzzle_id': puzzle_id,
            'correct': correct,
            'rating': puzzle['Rating'],
            'metrics': metrics
        }
    except (Exception, WorkerTimeoutError) as e:
        if isinstance(e, WorkerTimeoutError):
            print(f"Puzzle {puzzle_id} evaluation timed out after 10 minutes")
        else:
            print(f"Error evaluating puzzle {puzzle_id}: {e}")
        result = {
            'puzzle_id': puzzle_id,
            'correct': False,
            'rating': puzzle['Rating'],
            'metrics': {}
        }
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    return result

def worker_evaluate_position(position_data):
    """Worker function for evaluating a single position using the global engine."""
    global worker_engine
    fen, expected_move, win_prob = position_data
    
    # Set up timeout (10 minutes = 600 seconds)
    old_handler = signal.signal(signal.SIGALRM, worker_timeout_handler)
    signal.alarm(600)  # 10 minutes timeout
    
    try:
        board = chess.Board(fen)
        predicted_move = worker_engine.play(board).uci()
        correct = (expected_move == predicted_move)
        metrics = {}
        if hasattr(worker_engine, 'metrics'):
            metrics = worker_engine.metrics.copy()
        result = {
            'correct': correct,
            'win_prob': win_prob,
            'metrics': metrics
        }
    except (Exception, WorkerTimeoutError) as e:
        if isinstance(e, WorkerTimeoutError):
            print(f"Position evaluation timed out after 10 minutes for FEN: {fen}")
        else:
            print(f"Error evaluating position {fen}: {e}")
        result = {
            'correct': False,
            'win_prob': win_prob,
            'metrics': {}
        }
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    return result


def _ucinewgame_if_possible(raw_engine):
    """Sends 'ucinewgame' to the engine if supported to avoid tree reuse."""
    try:
        # python-chess exposes the low-level protocol for SimpleEngine
        raw_engine.protocol.send_line('ucinewgame')
    except Exception:
        pass


def _track_last_bestmove_change_nodes(board: chess.Board) -> tuple[chess.Move | None, int]:
    """Run a fixed-nodes analysis and return (final_best_move, last_change_nodes).

    Tracks the first move of the PV and records the node count at the last time
    that first PV move changed during the search.
    """
    global worker_engine, worker_num_nodes
    if worker_engine is None:
        raise RuntimeError('Worker engine not initialized')

    # Access underlying SimpleEngine to stream info
    if not hasattr(worker_engine, '_raw_engine'):
        raise RuntimeError('Blunderbase requires a UCI engine (Lc0/Stockfish/generic UCI).')
    raw_engine = getattr(worker_engine, '_raw_engine')

    # Avoid tree reuse between positions
    _ucinewgame_if_possible(raw_engine)

    best_move_so_far: chess.Move | None = None
    last_change_nodes: int = 0

    limit = chess.engine.Limit(nodes=worker_num_nodes or 0)
    try:
        with raw_engine.analysis(board, limit=limit, info=chess.engine.INFO_ALL) as analysis:
            for info in analysis:
                pv = info.get('pv')
                if pv:
                    current = pv[0]
                    if best_move_so_far is None or current != best_move_so_far:
                        best_move_so_far = current
                        # Some engines might not report nodes every tick
                        nodes_val = info.get('nodes')
                        if isinstance(nodes_val, int):
                            last_change_nodes = nodes_val
    except Exception:
        # Fallback: try a non-streaming best move if streaming failed
        try:
            result_move = raw_engine.play(board, limit=limit).move
            best_move_so_far = result_move
        except Exception:
            best_move_so_far = None

    return best_move_so_far, last_change_nodes


def worker_evaluate_blunderbase(entry: tuple[str, str]) -> dict:
    """Evaluate a single Blunderbase entry.

    Args:
        entry: Tuple of (fen, expected_move_san)

    Returns:
        dict with fields: 'fen', 'result' ('FAILED' or int nodes), 'bestmove_uci'
    """
    fen, expected_san = entry
    board = chess.Board(fen)

    # Parse expected move from SAN into UCI for comparison
    try:
        expected_move = board.parse_san(expected_san)
    except Exception:
        # If SAN parsing fails, mark as failed
        return {'fen': fen, 'result': 'FAILED', 'bestmove_uci': None}

    best_move, last_change_nodes = _track_last_bestmove_change_nodes(board)

    if best_move is not None and best_move == expected_move:
        return {
            'fen': fen,
            'result': last_change_nodes,
            'bestmove_uci': best_move.uci(),
        }
    else:
        return {
            'fen': fen,
            'result': 'FAILED',
            'bestmove_uci': best_move.uci() if best_move is not None else None,
        }


def _final_centipawns_at_nodes(board: chess.Board) -> int | None:
    """Run fixed-nodes analysis and return final CP score from the engine's POV.

    Returns None if score is unavailable.
    """
    global worker_engine, worker_num_nodes
    if worker_engine is None:
        raise RuntimeError('Worker engine not initialized')
    if not hasattr(worker_engine, '_raw_engine'):
        raise RuntimeError('Fortressbase requires a UCI engine (Lc0/Stockfish/generic UCI).')
    raw_engine = getattr(worker_engine, '_raw_engine')
    _ucinewgame_if_possible(raw_engine)

    limit = chess.engine.Limit(nodes=worker_num_nodes or 0)
    try:
        # Non-streaming final analysis; python-chess returns final info including 'score'
        result = raw_engine.analyse(board, limit=limit)
        score = result.get('score')
        if score is None:
            return None
        # Convert to centipawns from the side to move perspective
        rel = score.relative
        if rel.is_mate():
            # Map mates to large CP magnitude keeping sign
            sign = 1 if rel.mate() and rel.mate() > 0 else -1
            return 100000 * sign
        return int(rel.cp)
    except Exception:
        return None


def worker_evaluate_fortressbase(entry: tuple[str, str]) -> dict:
    """Evaluate a single Fortressbase entry.

    Args:
        entry: (fen, result_label) where result_label is 'Win' or 'Draw'

    Returns:
        dict with fields: 'fen', 'result' (float), 'cp'
    """
    fen, result_label = entry
    board = chess.Board(fen)
    cp = _final_centipawns_at_nodes(board)
    if cp is None:
        return {'fen': fen, 'result': float('nan'), 'cp': None}

    # decisive probability
    dp = math.atan(abs(cp) / 90.0) / 1.5637541897

    # If fortress is a win, return 1 - dp; if draw, return 0
    label = (result_label or '').strip().lower()
    if label == 'win':
        value = 1.0 - dp
    else:
        value = dp

    return {'fen': fen, 'result': float(value), 'cp': int(cp)}

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
    num_nodes = _NUM_NODES.value
    num_processes = _NUM_PROCESSES.value or mp.cpu_count()
    name = _NAME.value if _NAME.value is not None else strategy
    
    MY_ENGINE = (network is None)
    checkpoint_path = '../checkpoints/r1/r1.pt' if _CHECKPOINT.value is None else _CHECKPOINT.value

    print(f"Using {num_processes} processes for evaluation")

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
        
        with open(f'puzzles-{name}.txt', 'w') as f:
            num_correct = 0
            num_iterations = 0
            
            # Use multiprocessing Pool with initializer
            with mp.Pool(
                processes=num_processes, 
                initializer=init_worker,
                initargs=(strategy, network, depth, num_nodes, checkpoint_path, _UCI_ENGINE_BIN.value)
            ) as pool:
                # Use imap for streaming results with progress bar
                pbar = tqdm(
                    pool.imap(worker_evaluate_puzzle, puzzle_data),
                    total=len(puzzle_data),
                    desc=f"Evaluating puzzles ({name})"
                )
                
                # Aggregate metrics
                total_metrics = {
                    'num_nodes': 0,
                    'num_searches': 0,
                    'bf': 0,
                    'depth': 0,
                    'pv': 0,
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
                        stats['bf'] = f'{total_metrics["bf"] / total_metrics["num_searches"]:.3f}'
                        stats['depth'] = f'{total_metrics["depth"] / total_metrics["num_searches"]:.2f}'
                        stats['pv'] = f'{total_metrics["pv"] / total_metrics["num_searches"]:.2f}'
                    pbar.set_postfix(stats)
                    
            print(f'{name}: {num_correct / len(puzzles):.2%}')

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
            'bf': 0,
            'depth': 0,
            'pv': 0,
        }
        
        # Use multiprocessing Pool with initializer
        with mp.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(strategy, network, depth, num_nodes, checkpoint_path, _UCI_ENGINE_BIN.value)
        ) as pool:
            pbar = tqdm(
                pool.imap(worker_evaluate_position, positions_to_evaluate),
                total=len(positions_to_evaluate),
                desc=f"Evaluating positions ({name})"
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
                    stats['bf'] = f'{total_metrics["bf"] / total_metrics["num_searches"]:.3f}'
                    stats['depth'] = f'{total_metrics["depth"] / total_metrics["num_searches"]:.2f}'
                    stats['pv'] = f'{total_metrics["pv"] / total_metrics["num_searches"]:.2f}'
                pbar.set_postfix(stats)

        print(f"Lichess Top-1 move match rate: {total_top1_match/total_positions:.4f}")

    elif _TYPE.value == 'blunderbase':
        # Evaluate on the Blunderbase dataset, computing the node count at the
        # last time the PV's first move changed, and reporting it only if the
        # final best move matches the provided correct move; otherwise FAILED.

        # Load dataset
        blunderbase_path = os.path.join(
            os.getcwd(),
            '../data/Blunderbase.csv',
        )
        df = pd.read_csv(blunderbase_path)

        # Column names as in the CSV header
        fen_col = 'FEN of the position'
        correct_san_col = 'Correct move that Lc0 should have played'

        # Filter valid rows
        valid = df[[fen_col, correct_san_col]].dropna()

        # Prepare entries for multiprocessing
        entries: list[tuple[str, str]] = []
        for _, row in valid.iterrows():
            fen = str(row[fen_col]).strip()
            san = str(row[correct_san_col]).strip()
            if fen and san and fen != 'nan' and san != 'nan':
                entries.append((fen, san))

        print(f"Evaluating {len(entries)} Blunderbase positions...")

        # Run with multiprocessing and initialized engines
        with mp.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(strategy, network, depth, num_nodes, checkpoint_path, _UCI_ENGINE_BIN.value)
        ) as pool:
            pbar = tqdm(
                pool.imap(worker_evaluate_blunderbase, entries),
                total=len(entries),
                desc=f"Evaluating Blunderbase ({name})"
            )

            out_path = f'blunderbase-{name}.txt'
            correct_count = 0
            sum_log_nodes = 0.0
            with open(out_path, 'w') as f:
                for res in pbar:
                    # res['result'] is either int nodes or 'FAILED'
                    is_ok = isinstance(res['result'], int)
                    if is_ok:
                        correct_count += 1
                    # Accumulate avg log(nodes): use last-change nodes if OK, else max nodes
                    nodes_for_avg = res['result'] if is_ok else num_nodes * math.e
                    try:
                        nodes_int = int(nodes_for_avg)
                    except Exception:
                        nodes_int = num_nodes
                    if nodes_int <= 0:
                        nodes_int = 1
                    sum_log_nodes += np.log(nodes_int)

                    f.write(str(res) + '\n')
                    f.flush()
                    pbar.set_postfix({
                        'ok': f"{correct_count} / {len(entries)}",
                    })

        print(f"Results written to {out_path}")
        # Report final metrics
        total_examples = len(entries)
        avg_log_nodes = (sum_log_nodes / total_examples) if total_examples > 0 else float('nan')
        success_pct = (correct_count / total_examples) if total_examples > 0 else 0.0
        print(f"Average log(nodes): {avg_log_nodes:.6f}")
        print(f"Success (not FAILED): {success_pct:.2%}")

    elif _TYPE.value == 'fortressbase':
        # Evaluate on the Fortressbase dataset. For each entry, run fixed-nodes
        # analysis, convert final CP to decisive probability via
        # atan(|cp|/90)/1.5637541897, then return (1 - dp) if the fortress is a
        # win, or 0 if it's a draw.

        fortress_path = os.path.join(
            os.getcwd(),
            '../data/Fortressbase.csv',
        )
        df = pd.read_csv(fortress_path)

        fen_col = 'FEN of the position'
        result_col = 'Result'
        valid = df[[fen_col, result_col]].dropna()

        entries: list[tuple[str, str]] = []
        for _, row in valid.iterrows():
            fen = str(row[fen_col]).strip()
            label = str(row[result_col]).strip()
            if fen and label and fen != 'nan' and label != 'nan':
                entries.append((fen, label))

        print(f"Evaluating {len(entries)} Fortressbase positions...")

        with mp.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(strategy, network, depth, num_nodes, checkpoint_path, _UCI_ENGINE_BIN.value)
        ) as pool:
            pbar = tqdm(
                pool.imap(worker_evaluate_fortressbase, entries),
                total=len(entries),
                desc=f"Evaluating Fortressbase ({name})"
            )

            out_path = f'fortressbase-{name}.txt'
            sum_values = 0.0
            count_values = 0
            with open(out_path, 'w') as f:
                for res in pbar:
                    f.write(str(res) + '\n')
                    f.flush()
                    val = res['result']
                    if isinstance(val, float) and not math.isnan(val):
                        sum_values += val
                        count_values += 1
                    pbar.set_postfix({
                        'avg': f"{(sum_values / max(1, count_values)):.4f}",
                    })

        print(f"Results written to {out_path}")
        if count_values > 0:
            print(f"Average fortress score: {sum_values / count_values:.6f}")
        else:
            print("Average fortress score: NaN (no valid results)")


if __name__ == '__main__':
  app.run(main)

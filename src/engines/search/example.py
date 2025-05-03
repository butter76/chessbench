"""Example demonstrating the parallel search infrastructure."""

import chess
import numpy as np
import time
import argparse
import pprint

from searchless_chess.src.engines.search.utils import SearchManager
from searchless_chess.src.engines.search.alphabeta_worker import AlphaBetaWorker


def dummy_model_predict_fn(tokenized_boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dummy model predict function for demonstration.
    
    Args:
        tokenized_boards: Batch of tokenized board positions
        
    Returns:
        (policy_logits, values) tuple
    """
    batch_size = tokenized_boards.shape[0]
    
    # Generate random policy logits (shape: [batch_size, 68 * 68])
    policy_logits = np.random.normal(size=(batch_size, 68, 68))
    
    # Generate random values between -1 and 1 (shape: [batch_size])
    values = np.random.uniform(-1, 1, size=batch_size)
    
    # Simulate GPU computation time (adjust to test different scenarios)
    time.sleep(0.005 * batch_size)  # Scale with batch size for realism
    
    return policy_logits, values


def run_example(search_time: float = 10.0, verbose: bool = True):
    """Run a simple example of parallel search with AlphaBeta search.
    
    Args:
        search_time: Time to run search in seconds
        verbose: Whether to print detailed results
    """
    # Create initial board
    board = chess.Board()
    
    # Create search manager with our dummy model
    search_manager = SearchManager(
        model_predict_fn=dummy_model_predict_fn,
        max_batch_size=48,
        timeout=0.01,
        game_log_file="../data/self-play/gen0-selfplay.bag"
    )
    
    # Set metrics logging interval based on verbosity
    search_manager.log_metrics_interval = 2.0 if verbose else float('inf')
    
    # Create AlphaBeta workers with different parameters
    alpha_beta_worker1 = AlphaBetaWorker(
        initial_board=board,
        evaluation_queue=search_manager.evaluation_queue,
        max_depth=3
    )
    search_manager.add_worker(alpha_beta_worker1)
    
    alpha_beta_worker2 = AlphaBetaWorker(
        initial_board=board,
        evaluation_queue=search_manager.evaluation_queue,
        max_depth=4
    )
    search_manager.add_worker(alpha_beta_worker2)

    alpha_beta_worker3 = AlphaBetaWorker(
        initial_board=board,
        evaluation_queue=search_manager.evaluation_queue,
        max_depth=5
    )
    search_manager.add_worker(alpha_beta_worker3)
    
    # Start the search
    print(f"Starting parallel search with AlphaBeta algorithm (running for {search_time} seconds)...")
    start_time = time.time()
    search_manager.start()
    
    # Wait for search time
    time.sleep(search_time)
    
    # Stop all search components (will print final metrics)
    search_manager.stop()
    
    # Get final metrics for reporting
    final_metrics = search_manager.get_metrics()
    
    # Print results
    print(f"\n=== Search Results ===")
    print(f"Total search time: {time.time() - start_time:.2f} seconds")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"  Total positions evaluated: {final_metrics['positions_evaluated']}")
    print(f"  Average evaluation rate: {final_metrics['avg_positions_per_second']:.1f} positions/sec")
    print(f"  Peak evaluation rate: {final_metrics['positions_per_second']:.1f} positions/sec")
    print(f"  GPU utilization: {final_metrics['gpu_utilization']*100:.1f}%")
    print(f"  Average batch size: {final_metrics['avg_batch_size']:.2f}")
            
    # Print detailed worker stats if verbose
    if verbose:
        print("\nDetailed Worker Stats:")
        for search_id, worker_metrics in final_metrics['worker_metrics'].items():
            print(f"  Worker {search_id[:6]} ({worker_metrics['algorithm_type']}):")
            print(f"    Positions submitted: {worker_metrics['positions_submitted']}")
            print(f"    Results processed: {worker_metrics['results_processed']}")
            
        print("\nBatch Size Stats:")
        print(f"  Min: {final_metrics['min_batch_size']}")
        print(f"  Max: {final_metrics['max_batch_size']}")
        print(f"  Avg: {final_metrics['avg_batch_size']:.2f}")
    
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel search example")
    parser.add_argument("--time", type=float, default=10.0, 
                      help="Search time in seconds")
    parser.add_argument("--verbose", action="store_true", 
                      help="Print detailed metrics")
    
    args = parser.parse_args()
    
    metrics = run_example(search_time=args.time, verbose=args.verbose) 
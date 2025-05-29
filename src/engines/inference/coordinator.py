"""Coordinates multiple search processes with shared inference server."""

import multiprocessing as mp
import time
import os
from typing import List, Optional, Dict, Any, Callable
import chess

# Fix CUDA multiprocessing issue by setting spawn method
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method was already set, continue with existing method
        pass

from .server import InferenceServer
from .manager import InferenceManager
from searchless_chess.src.engines.search.base import SearchAlgorithm, SearchResult
from searchless_chess.src.engines.strategy import MoveSelectionStrategy


def _search_worker(
    boards: List[chess.Board],
    algorithm_name: str,
    search_kwargs: Dict[str, Any],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int
):
    """
    Worker function that runs in a separate process.
    Performs search on assigned boards using remote inference.
    """
    try:
        # Import here to avoid circular imports in multiprocessing
        from searchless_chess.src.engines.search import (
            ValueSearch, PolicySearch, AVSSearch, NegamaxSearch,
            AlphaBetaSearch, PVSSearch, MTDFSearch, MCTSSearch
        )
        
        # Create search algorithm instance
        algorithm_map = {
            'value': ValueSearch(),
            'policy': PolicySearch('policy'),
            'opt_policy_split': PolicySearch('opt_policy_split'),
            'avs': AVSSearch('avs'),
            'avs2': AVSSearch('avs2'),
            'negamax': NegamaxSearch(),
            'alpha_beta': AlphaBetaSearch(),
            'pvs': PVSSearch(),
            'alpha_beta_node': PVSSearch(),  # Legacy alias
            'mtdf': MTDFSearch(),
            'mcts': MCTSSearch(),
        }
        
        if algorithm_name not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algorithm = algorithm_map[algorithm_name]
        
        # Create inference functions that use the shared queues
        def remote_inference_func(board: chess.Board) -> Dict[str, Any]:
            import uuid
            import time
            from .server import InferenceRequest, InferenceResponse
            
            request_id = str(uuid.uuid4())
            request = InferenceRequest(request_id, board.fen())
            
            try:
                request_queue.put(request, timeout=1.0)
            except:
                raise RuntimeError("Inference request queue is full")
            
            # Wait for our specific response
            timeout_time = time.time() + 10.0
            while time.time() < timeout_time:
                try:
                    response = response_queue.get(timeout=0.1)
                    if response.request_id == request_id:
                        if response.error:
                            raise RuntimeError(f"Inference failed: {response.error}")
                        return response.output
                    else:
                        # Not our response, put it back (this is not ideal but works)
                        response_queue.put(response)
                except:
                    continue
            
            raise TimeoutError("Inference request timed out")
        
        def remote_batch_inference_func(boards: List[chess.Board]) -> List[Dict[str, Any]]:
            # For now, just do sequential requests
            # TODO: Could optimize this further with true batch requests
            return [remote_inference_func(board) for board in boards]
        
        # Process each board
        results = []
        for i, board in enumerate(boards):
            try:
                result = algorithm.search(
                    board=board,
                    inference_func=remote_inference_func,
                    batch_inference_func=remote_batch_inference_func,
                    **search_kwargs
                )
                results.append((i, result, None))  # (index, result, error)
                
            except Exception as e:
                results.append((i, None, str(e)))
        
        # Send results back
        result_queue.put((worker_id, results))
        
    except Exception as e:
        # Send error for entire worker
        import traceback
        error_msg = f"Worker {worker_id} failed: {e}\n{traceback.format_exc()}"
        result_queue.put((worker_id, [(i, None, error_msg) for i in range(len(boards))]))


class BatchSearchCoordinator:
    """
    Coordinates multiple search processes working on different boards
    while sharing a single inference server for optimal GPU utilization.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        max_batch_size: int = 64,
        batch_timeout_ms: float = 10.0,
        num_workers: int = None,
        device: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.num_workers = num_workers or mp.cpu_count()
        self.device = device
        
        # Process management
        self.inference_server = None
        self.request_queue = None
        self.response_queue = None
        self.is_running = False
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def start(self):
        """Start the inference server."""
        if self.is_running:
            return
        
        print("BatchSearchCoordinator: Starting inference server...")
        
        # Create shared queues for inference server
        self.request_queue = mp.Queue(maxsize=1000)
        self.response_queue = mp.Queue(maxsize=1000)
        
        # Start inference server process
        self.inference_server = InferenceServer(
            checkpoint_path=self.checkpoint_path,
            request_queue=self.request_queue,
            response_queue=self.response_queue,
            max_batch_size=self.max_batch_size,
            batch_timeout_ms=self.batch_timeout_ms,
            device=self.device
        )
        self.inference_server.start()
        
        # Wait a moment for server to initialize
        time.sleep(2.0)
        self.is_running = True
        
        print("BatchSearchCoordinator: Inference server started")
    
    def shutdown(self):
        """Shutdown the inference server and cleanup."""
        if not self.is_running:
            return
        
        print("BatchSearchCoordinator: Shutting down...")
        
        if self.inference_server:
            self.inference_server.shutdown()
            self.inference_server = None
        
        self.request_queue = None
        self.response_queue = None
        self.is_running = False
        
        print("BatchSearchCoordinator: Shutdown complete")
    
    def batch_search(
        self,
        boards: List[chess.Board],
        algorithm: str,
        depth: float = 2.0,
        **search_kwargs
    ) -> List[SearchResult]:
        """
        Perform search on multiple boards using multiple worker processes.
        
        Args:
            boards: List of chess boards to analyze
            algorithm: Search algorithm name (e.g., 'pvs', 'value', 'mcts')
            depth: Search depth
            **search_kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects in the same order as input boards
        """
        if not self.is_running:
            raise RuntimeError("BatchSearchCoordinator not started. Use start() or context manager.")
        
        if not boards:
            return []
        
        num_boards = len(boards)
        print(f"BatchSearchCoordinator: Processing {num_boards} boards with {self.num_workers} workers")
        
        # Split boards among workers
        boards_per_worker = max(1, num_boards // self.num_workers)
        worker_boards = []
        for i in range(0, num_boards, boards_per_worker):
            worker_boards.append(boards[i:i + boards_per_worker])
        
        # If we have more workers than chunks, adjust
        if len(worker_boards) > self.num_workers:
            # Redistribute
            actual_workers = len(worker_boards)
        else:
            actual_workers = len(worker_boards)
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Start worker processes
        workers = []
        for worker_id, board_chunk in enumerate(worker_boards):
            worker = mp.Process(
                target=_search_worker,
                args=(
                    board_chunk,
                    algorithm,
                    {'depth': depth, **search_kwargs},
                    self.request_queue,
                    self.response_queue,
                    result_queue,
                    worker_id
                )
            )
            worker.start()
            workers.append(worker)
        
        # Collect results from all workers
        all_results = [None] * num_boards
        workers_completed = 0
        
        while workers_completed < actual_workers:
            try:
                worker_id, worker_results = result_queue.get(timeout=30.0)
                workers_completed += 1
                
                # Map worker results back to original board positions
                board_offset = worker_id * boards_per_worker
                for local_idx, result, error in worker_results:
                    global_idx = board_offset + local_idx
                    if global_idx < num_boards:
                        if error:
                            # Create error result
                            all_results[global_idx] = SearchResult(
                                move=None,
                                score=0.0,
                                metadata={'error': error}
                            )
                        else:
                            all_results[global_idx] = result
                
            except mp.TimeoutError:
                print("Warning: Worker timeout, terminating remaining processes")
                break
        
        # Clean up worker processes
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=1.0)
        
        # Fill any missing results with default values
        for i in range(num_boards):
            if all_results[i] is None:
                # Fallback: just pick the first legal move
                legal_moves = list(boards[i].legal_moves)
                fallback_move = legal_moves[0] if legal_moves else None
                all_results[i] = SearchResult(
                    move=fallback_move,
                    score=0.0,
                    metadata={'error': 'Worker failed, using fallback'}
                )
        
        print(f"BatchSearchCoordinator: Completed processing {num_boards} boards")
        return all_results 
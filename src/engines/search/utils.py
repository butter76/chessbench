"""Parallel search infrastructure for massively parallel game playing.

This module implements a massively parallel search environment where multiple
search algorithms run in parallel, submit positions for batch GPU evaluation,
and process the results to generate new positions for evaluation.
"""

import concurrent.futures
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import uuid
from searchless_chess.src.engines.utils.nnutils import get_policy
import numpy as np
import chess
import collections
import datetime

from searchless_chess.src import tokenizer
from searchless_chess.src.engines.utils.node import Node
from searchless_chess.src import bagz


class GameLogger:
    """Logger for recording game data to a .bag file."""
    
    def __init__(self, output_file: str):
        """Initialize game logger.
        
        Args:
            output_file: Path to output .bag file
        """
        self.output_file = output_file
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self.bag_writer = None
        
    def start(self):
        """Start the game logger thread."""
        self.running = True
        self.bag_writer = bagz.BagWriter(self.output_file)
        self.thread = threading.Thread(target=self._logging_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the game logger thread."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.bag_writer:
            self.bag_writer.close()
            
    def transmit_game(self, encoded_data: bytes):
        """Add encoded game data to the logging queue.
        
        Args:
            encoded_data: Encoded game data bytes
        """
        self.queue.put(encoded_data)
        
    def _logging_loop(self):
        """Main logging loop."""
        while self.running:
            try:
                # Get encoded data (blocking with timeout)
                encoded_data = self.queue.get(block=True, timeout=0.1)
                
                # Write to bag file
                self.bag_writer.write(encoded_data)
                
            except queue.Empty:
                # Continue if queue is empty
                continue


class Position:
    """Represents a position submitted for evaluation."""
    
    def __init__(self, board: chess.Board, search_id: str, node_id: str):
        """Initialize a position for evaluation.
        
        Args:
            board: Chess board position
            search_id: ID of the search worker that submitted this position
            node_id: ID of the node in the search tree
        """
        self.board = board.copy()
        self.search_id = search_id
        self.node_id = node_id
        # Eagerly tokenize the board during initialization
        # This distributes tokenization work across search worker threads
        self.tokenized = tokenizer.tokenize(self.board.fen()).astype(np.int32)
        
    def tokenize(self):
        """Return the tokenized board for neural network input."""
        return self.tokenized


class EvaluationResult:
    """Evaluation result returned from GPU."""
    
    def __init__(self, search_id: str, node_id: str, raw_policy: np.ndarray, value: float):
        """Initialize evaluation result.
        
        Args:
            search_id: ID of the search worker that submitted the position
            node_id: ID of the node in the search tree
            policy: Raw policy from neural network
            value: Value evaluation [-1, 1]
        """
        self.search_id = search_id
        self.node_id = node_id
        self.raw_policy = raw_policy
        self.value = value


class EvaluationQueue:
    """Buffer for positions awaiting evaluation."""
    
    def __init__(self, max_batch_size: int = 64, timeout: float = 0.01):
        """Initialize evaluation queue.
        
        Args:
            max_batch_size: Maximum batch size for GPU evaluation
            timeout: Time to wait for batch to fill before sending partial batch
        """
        self.queue = queue.Queue()
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        
    def add_position(self, position: Position):
        """Add a position to the evaluation queue.
        
        Args:
            position: Position to evaluate
        """
        self.queue.put(position)
        
    def get_batch(self) -> List[Position]:
        """Get a batch of positions for evaluation.
        
        Returns:
            List of positions to evaluate as a batch
        """
        batch = []
        try:
            # Get at least one position (blocking)
            batch.append(self.queue.get(block=True, timeout=self.timeout))
            
            # Try to fill the batch up to max_batch_size (non-blocking)
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get(block=False))
                except queue.Empty:
                    break
                    
        except queue.Empty:
            # Return empty batch if queue is empty
            pass
            
        return batch


class ResultDistributor:
    """Distributes evaluation results back to search workers."""
    
    def __init__(self):
        """Initialize result distributor."""
        self.result_queues = {}
        
    def register_search(self, search_id: str) -> queue.Queue:
        """Register a search worker to receive results.
        
        Args:
            search_id: Unique ID for the search worker
            
        Returns:
            Queue to receive evaluation results
        """
        result_queue = queue.Queue()
        self.result_queues[search_id] = result_queue
        return result_queue
        
    def unregister_search(self, search_id: str):
        """Unregister a search worker.
        
        Args:
            search_id: ID of the search worker to unregister
        """
        if search_id in self.result_queues:
            del self.result_queues[search_id]
            
    def distribute_result(self, result: EvaluationResult):
        """Distribute an evaluation result to the appropriate search worker.
        
        Args:
            result: Evaluation result to distribute
        """
        if result.search_id in self.result_queues:
            self.result_queues[result.search_id].put(result)


class GPUEvaluator:
    """GPU-based neural network evaluator for positions."""
    
    def __init__(self, 
                 model_predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                 evaluation_queue: EvaluationQueue,
                 result_distributor: ResultDistributor):
        """Initialize GPU evaluator.
        
        Args:
            model_predict_fn: Function that takes batched tokenized positions and returns
                             (policy_logits, value) tuples
            evaluation_queue: Queue of positions to evaluate
            result_distributor: Distributor to send results back to search workers
        """
        self.model_predict_fn = model_predict_fn
        self.evaluation_queue = evaluation_queue
        self.result_distributor = result_distributor
        self.running = False
        self.thread = None
        
        # Performance metrics
        self.total_positions_evaluated = 0
        self.total_batches_evaluated = 0
        self.total_evaluation_time = 0.0
        self.batch_sizes = []  # Track all batch sizes
        
    def start(self):
        """Start the GPU evaluator thread."""
        self.running = True
        self.thread = threading.Thread(target=self._evaluation_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the GPU evaluator thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _evaluation_loop(self):
        """Main evaluation loop."""
        while self.running:
            # Get a batch of positions
            batch = self.evaluation_queue.get_batch()
            
            if not batch:
                # Sleep briefly if no positions to evaluate
                time.sleep(0.001)
                continue
                
            # Update batch size metrics
            self.batch_sizes.append(len(batch))
            self.total_batches_evaluated += 1
                
            # Prepare batch for evaluation
            tokenized_batch = np.stack([pos.tokenize() for pos in batch])
            
            # Measure evaluation time
            start_time = time.time()
            
            # Evaluate batch on GPU
            policy_batch, values_batch = self.model_predict_fn(tokenized_batch)
            
            # Update timing metrics
            eval_time = time.time() - start_time
            self.total_evaluation_time += eval_time
            
            # Process and distribute results
            for i, position in enumerate(batch):
                
                # Create and distribute result
                result = EvaluationResult(
                    search_id=position.search_id,
                    node_id=position.node_id,
                    raw_policy=policy_batch[i],
                    value=float(values_batch[i])
                )
                
                self.result_distributor.distribute_result(result)
            
            # Update counter
            self.total_positions_evaluated += len(batch)


class SearchWorker:
    """Base class for search algorithm workers."""
    
    def __init__(self, evaluation_queue: EvaluationQueue, game_logger: Optional[GameLogger] = None):
        """Initialize search worker.
        
        Args:
            evaluation_queue: Queue for submitting positions for evaluation
            game_logger: Optional logger for recording games
        """
        self.evaluation_queue = evaluation_queue
        self.ack_queue = EvaluationQueue(1)
        self.search_id = str(uuid.uuid4())
        self.result_queue = None
        self.running = False
        self.thread = None
        self.game_logger = game_logger
        
    def register(self, result_distributor: ResultDistributor):
        """Register with result distributor to receive evaluation results.
        
        Args:
            result_distributor: Result distributor to register with
        """
        self.result_queue = result_distributor.register_search(self.search_id)
        
    def submit_position(self, board: chess.Board, node_id: str):
        """Submit a position for evaluation.
        
        Args:
            board: Chess board position to evaluate
            node_id: ID of the node in the search tree
        """
        position = Position(board=board, search_id=self.search_id, node_id=node_id)
        self.evaluation_queue.add_position(position)
        self.ack_queue.add_position(position)
        
    def process_result(self, result: EvaluationResult):
        """Process an evaluation result (implemented by subclasses).
        
        Args:
            result: Evaluation result to process
        """
        raise NotImplementedError("Subclasses must implement process_result")
        
    def search_step(self):
        """Perform one step of search (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement search_step")
        
    def transmit_game(self, encoded_data: bytes):
        """Transmit encoded game data to the game logger.
        
        Args:
            encoded_data: Encoded game data bytes
        """
        if self.game_logger:
            self.game_logger.transmit_game(encoded_data)
        
    def start(self):
        """Start the search worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._search_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the search worker thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _search_loop(self):
        """Main search loop."""
        while self.running:
            # Perform one step of search first
            self.search_step()
            
            # Block until a result is available
            result = self.result_queue.get()
            pos = self.ack_queue.get_batch()[0]
            policy, policy_map = get_policy(pos.board, result.raw_policy)
            result.policy = policy
            result.policy_map = policy_map
            result.board = pos.board
            self.process_result(result)


class SearchMetrics:
    """Tracks performance metrics for the parallel search system."""
    
    def __init__(self):
        """Initialize search metrics tracker."""
        self.start_time = time.time()
        
        # Position metrics
        self.positions_submitted = 0
        self.positions_evaluated = 0
        
        # Batch metrics
        self.batches_processed = 0
        self.min_batch_size = float('inf')
        self.max_batch_size = 0
        self.avg_batch_size = 0.0
        
        # Timing metrics
        self.total_gpu_time = 0.0
        
        # Thread metrics
        self.worker_metrics = {}  # search_id -> metrics dict
        
        # Cache recent metrics for rate calculations
        self.recent_positions = collections.deque(maxlen=100)  # (timestamp, count)
        
    def record_position_submitted(self):
        """Record a position submitted for evaluation."""
        self.positions_submitted += 1
        self.recent_positions.append((time.time(), 1))
        
    def record_batch_evaluated(self, batch_size: int, eval_time: float):
        """Record a batch evaluation.
        
        Args:
            batch_size: Size of the batch
            eval_time: Time taken to evaluate the batch
        """
        self.batches_processed += 1
        self.positions_evaluated += batch_size
        self.total_gpu_time += eval_time
        
        # Update batch size stats
        if batch_size < self.min_batch_size:
            self.min_batch_size = batch_size
        if batch_size > self.max_batch_size:
            self.max_batch_size = batch_size
            
        # Update average batch size
        self.avg_batch_size = ((self.avg_batch_size * (self.batches_processed - 1)) + batch_size) / self.batches_processed
        
    def register_worker(self, search_id: str, algorithm_type: str):
        """Register a new search worker.
        
        Args:
            search_id: ID of the search worker
            algorithm_type: Type of search algorithm
        """
        self.worker_metrics[search_id] = {
            'algorithm_type': algorithm_type,
            'positions_submitted': 0,
            'results_processed': 0,
        }
        
    def record_worker_submitted_position(self, search_id: str):
        """Record a position submitted by a worker.
        
        Args:
            search_id: ID of the search worker
        """
        if search_id in self.worker_metrics:
            self.worker_metrics[search_id]['positions_submitted'] += 1
            
    def record_worker_processed_result(self, search_id: str):
        """Record a result processed by a worker.
        
        Args:
            search_id: ID of the search worker
        """
        if search_id in self.worker_metrics:
            self.worker_metrics[search_id]['results_processed'] += 1
            
    def get_positions_per_second(self) -> float:
        """Calculate the recent positions evaluated per second.
        
        Returns:
            Positions evaluated per second
        """
        if not self.recent_positions:
            return 0.0
            
        now = time.time()
        # Filter to positions evaluated in the last 5 seconds
        recent = [p for p in self.recent_positions if now - p[0] < 5.0]
        
        if not recent:
            return 0.0
            
        total_positions = sum(count for _, count in recent)
        earliest_time = min(t for t, _ in recent)
        time_span = now - earliest_time
        
        if time_span < 0.001:  # Avoid division by near-zero
            return 0.0
            
        return total_positions / time_span
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.
        
        Returns:
            Dictionary of metrics
        """
        elapsed_time = time.time() - self.start_time
        
        return {
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(datetime.timedelta(seconds=int(elapsed_time))),
            
            'positions_submitted': self.positions_submitted,
            'positions_evaluated': self.positions_evaluated,
            'positions_per_second': self.get_positions_per_second(),
            'avg_positions_per_second': self.positions_evaluated / elapsed_time if elapsed_time > 0 else 0,
            
            'batches_processed': self.batches_processed,
            'min_batch_size': self.min_batch_size if self.min_batch_size != float('inf') else 0,
            'max_batch_size': self.max_batch_size,
            'avg_batch_size': self.avg_batch_size,
            
            'gpu_utilization': (self.total_gpu_time / elapsed_time) if elapsed_time > 0 else 0,
            'worker_metrics': self.worker_metrics,
        }


class SearchManager:
    """Manages parallel search workers and GPU evaluation."""
    
    def __init__(self, 
                 model_predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                 max_batch_size: int = 64,
                 timeout: float = 0.01,
                 game_log_file: Optional[str] = None):
        """Initialize search manager.
        
        Args:
            model_predict_fn: Function for neural network evaluation
            max_batch_size: Maximum batch size for GPU evaluation
            timeout: Time to wait for batch to fill before sending partial batch
            game_log_file: Optional path to file for logging games
        """
        self.evaluation_queue = EvaluationQueue(max_batch_size=max_batch_size, timeout=timeout)
        self.result_distributor = ResultDistributor()
        self.gpu_evaluator = GPUEvaluator(
            model_predict_fn=model_predict_fn,
            evaluation_queue=self.evaluation_queue,
            result_distributor=self.result_distributor
        )
        self.search_workers = []
        
        # Set up game logger if a log file is provided
        self.game_logger = None
        if game_log_file:
            self.game_logger = GameLogger(game_log_file)
        
        # Set up metrics tracking
        self.metrics = SearchMetrics()
        
        # Monitor thread for periodic logging
        self.monitor_thread = None
        self.monitor_running = False
        self.log_metrics_interval = 5.0  # seconds
        
    def add_worker(self, worker: SearchWorker):
        """Add a search worker.
        
        Args:
            worker: Search worker to add
        """
        # Set the game logger for the worker
        worker.game_logger = self.game_logger
        
        worker.register(self.result_distributor)
        self.search_workers.append(worker)
        
        # Register worker in metrics system
        algorithm_type = type(worker).__name__
        self.metrics.register_worker(worker.search_id, algorithm_type)
        
        # Wrap worker's submit_position method to track metrics
        original_submit = worker.submit_position
        
        def submit_with_metrics(board, node_id):
            self.metrics.record_position_submitted()
            self.metrics.record_worker_submitted_position(worker.search_id)
            return original_submit(board, node_id)
            
        worker.submit_position = submit_with_metrics
        
        # Wrap worker's process_result method to track metrics
        original_process = worker.process_result
        
        def process_with_metrics(result):
            self.metrics.record_worker_processed_result(worker.search_id)
            return original_process(result)
            
        worker.process_result = process_with_metrics
        
    def start(self):
        """Start all components."""
        # Reset metrics
        self.metrics = SearchMetrics()
        
        # Start game logger if available
        if self.game_logger:
            self.game_logger.start()
        
        # Start GPU evaluator
        self.gpu_evaluator.start()
        
        # Start workers
        for worker in self.search_workers:
            worker.start()
            
        # Start metrics monitor
        self.start_metrics_monitor()
            
    def stop(self):
        """Stop all components."""
        # Stop metrics monitor
        self.stop_metrics_monitor()
        
        # Stop workers
        for worker in self.search_workers:
            worker.stop()
            
        # Stop GPU evaluator
        self.gpu_evaluator.stop()
        
        # Stop game logger if available
        if self.game_logger:
            self.game_logger.stop()
        
        # Final metrics summary
        self.log_metrics_summary()
        
    def wait_for_workers(self, timeout: Optional[float] = None):
        """Wait for all search workers to complete.
        
        Args:
            timeout: Maximum time to wait (seconds), or None to wait indefinitely
        """
        start_time = time.time()
        while timeout is None or (time.time() - start_time) < timeout:
            all_done = True
            if all_done:
                break
            time.sleep(0.01)
            
    def start_metrics_monitor(self):
        """Start the metrics monitoring thread."""
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._metrics_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_metrics_monitor(self):
        """Stop the metrics monitoring thread."""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _metrics_monitor_loop(self):
        """Monitor loop that periodically logs metrics."""
        last_log_time = time.time()
        
        while self.monitor_running:
            now = time.time()
            
            # Update metrics from GPU evaluator
            self.metrics.record_batch_evaluated(
                batch_size=self.gpu_evaluator.total_positions_evaluated,
                eval_time=self.gpu_evaluator.total_evaluation_time
            )
            
            # Log periodically
            if now - last_log_time >= self.log_metrics_interval:
                self.log_metrics_summary()
                last_log_time = now
                
            time.sleep(0.1)
            
    def log_metrics_summary(self):
        """Log a summary of the current metrics."""
        metrics = self.metrics.get_summary()
        
        print(f"\n--- Search Metrics Summary (Elapsed: {metrics['elapsed_time_formatted']}) ---")
        print(f"Positions: {metrics['positions_evaluated']} evaluated, {metrics['positions_per_second']:.1f} pos/sec")
        print(f"Batches: {metrics['batches_processed']} processed, avg size: {metrics['avg_batch_size']:.1f}")
        print(f"GPU Utilization: {metrics['gpu_utilization']*100:.1f}%")
        
        # Worker-specific stats
        print("\nWorker Stats:")
        for search_id, worker_metrics in metrics['worker_metrics'].items():
            worker_type = worker_metrics['algorithm_type']
            positions = worker_metrics['positions_submitted']
            results = worker_metrics['results_processed']
            print(f"  {worker_type} (ID: {search_id[:6]}): {positions} submitted, {results} processed")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics.
        
        Returns:
            Dictionary of current metrics
        """
        return self.metrics.get_summary()

import uuid
import time
import multiprocessing as mp
from queue import Empty
from typing import List, Union, Optional, cast, Dict
import os
import torch
import numpy as np
import chess
import chess.engine

from searchless_chess.src.engines import engine
from searchless_chess.src.engines.strategy import MoveSelectionStrategy
from searchless_chess.src.engines.search import (
    ValueSearch, PolicySearch, AVSSearch, NegamaxSearch, 
    AlphaBetaSearch, PVSSearch, MTDFSearch, MCTSSearch
)


class RemoteInferenceEngine(engine.Engine):
    """
    Lightweight engine that sends inference requests to a remote inference server
    and maintains all search algorithm logic locally.
    """
    
    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        limit: chess.engine.Limit,
        strategy: Union[MoveSelectionStrategy, str] = MoveSelectionStrategy.VALUE,
        search_depth: int | float = 2,
        search_ordering_strategy: Union[MoveSelectionStrategy, str, None] = MoveSelectionStrategy.AVS,
        worker_id: Optional[int] = None,
    ) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._limit = limit
        self.search_depth = search_depth
        self.worker_id = worker_id or os.getpid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store pending requests to match responses
        self.pending_requests = {}
        
        # Convert string to enum if needed
        if isinstance(strategy, str):
            strategy = MoveSelectionStrategy(strategy)
        self.strategy = strategy

        # Convert ordering strategy string to enum if needed
        if isinstance(search_ordering_strategy, str):
            search_ordering_strategy = MoveSelectionStrategy(search_ordering_strategy)
        self.search_ordering_strategy = search_ordering_strategy

        # Initialize search algorithms (same as MyTransformerEngine)
        self.search_algorithms = {
            MoveSelectionStrategy.VALUE: ValueSearch(),
            MoveSelectionStrategy.AVS: AVSSearch("avs"),
            MoveSelectionStrategy.AVS2: AVSSearch("avs2"),
            MoveSelectionStrategy.POLICY: PolicySearch("policy"),
            MoveSelectionStrategy.OPT_POLICY_SPLIT: PolicySearch("opt_policy_split"),
            MoveSelectionStrategy.NEGAMAX: NegamaxSearch(self.search_ordering_strategy),
            MoveSelectionStrategy.ALPHA_BETA: AlphaBetaSearch(self.search_ordering_strategy),
            MoveSelectionStrategy.ALPHA_BETA_NODE: PVSSearch(),  # Legacy name for PVS
            MoveSelectionStrategy.PVS: PVSSearch(),
            MoveSelectionStrategy.MTDF: MTDFSearch(),
            MoveSelectionStrategy.MCTS: MCTSSearch(),
        }
        
        # Initialize metrics tracking
        self.metrics = {
            'num_nodes': 0,
            'num_searches': 0,
            'policy_perplexity': 0,
            'depth': 0,
        }

    def _send_inference_request(self, fens: List[str], timeout: float = 30.0) -> Dict[str, torch.Tensor]:
        """Send inference request to server and wait for response."""
        request_id = str(uuid.uuid4())
        
        # Send request
        request_data = {
            'request_id': request_id,
            'fens': fens,
            'worker_id': self.worker_id,
        }
        
        self.request_queue.put(request_data)
        self.pending_requests[request_id] = time.time()
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.request_id == request_id:
                    # Clean up
                    del self.pending_requests[request_id]
                    
                    if response.error:
                        raise RuntimeError(f"Inference server error: {response.error}")
                    
                    # Move tensors back to appropriate device if needed
                    if response.results is not None:
                        results = {}
                        for key, tensor in response.results.items():
                            results[key] = tensor.to(self.device)
                        return results
                    else:
                        raise RuntimeError("Received empty results from inference server")
                else:
                    # Put back response that's not for us
                    self.response_queue.put(response)
                    
            except Empty:
                continue
                
        # Timeout
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]
        raise TimeoutError(f"Inference request {request_id} timed out after {timeout}s")

    def analyse_shallow(self, board: chess.Board) -> engine.AnalysisResult:
        """Send single board for analysis."""
        fens = [board.fen()]
        results = self._send_inference_request(fens)
        return results  # Return the dictionary directly

    def analyse_batch(self, boards: List[chess.Board]) -> Dict[str, torch.Tensor]:
        """Send multiple boards for batch analysis."""
        fens = [board.fen() for board in boards]
        return self._send_inference_request(fens)

    def play(self, board: chess.Board) -> chess.Move:
        """Play a move using the configured search strategy."""
        # Get the appropriate search algorithm
        search_algorithm = self.search_algorithms.get(self.strategy)
        if search_algorithm is None:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Create inference functions for the search algorithm
        def inference_func(board: chess.Board):
            return self.analyse_shallow(board)
        
        def batch_inference_func(boards):
            # Convert FEN strings back to boards and analyze them
            board_objs = []
            for fen in boards:
                board_obj = chess.Board(fen)
                board_objs.append(board_obj)
            return self.analyse_batch(board_objs)
        
        # Configure search parameters based on strategy
        search_kwargs = {
            'num_nodes': int(self.search_depth) if self.strategy in [MoveSelectionStrategy.MTDF, MoveSelectionStrategy.PVS] else None,
            'num_rollouts': int(self.search_depth) if self.strategy == MoveSelectionStrategy.MCTS else None,
        }
        
        # Perform the search
        result = search_algorithm.search(
            board=board,
            inference_func=inference_func,
            batch_inference_func=batch_inference_func,
            depth=self.search_depth,
            **{k: v for k, v in search_kwargs.items() if v is not None}
        )
        
        # Update metrics if search algorithm provides them
        if hasattr(search_algorithm, 'metrics'):
            for key, value in search_algorithm.metrics.items():
                if key in self.metrics:
                    self.metrics[key] += value
        
        return result.move 
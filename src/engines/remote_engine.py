import uuid
import time
import multiprocessing as mp
from queue import Empty
from typing import List, Union, Optional, cast, Dict
import os
import numpy as np
import chess
import chess.engine

from searchless_chess.src.engines import engine
from searchless_chess.src.engines.strategy import MoveSelectionStrategy
from searchless_chess.src.engines.search import (
    ValueSearch, PolicySearch, AVSSearch, NegamaxSearch, 
    AlphaBetaSearch, PVSSearch, MTDFSearch, MCTSSearch
)


class ModelOutputWrapper:
    """Wrapper to make numpy arrays behave like torch tensors for search algorithms."""
    
    def __init__(self, numpy_dict: Dict[str, np.ndarray]):
        self.data = numpy_dict
    
    def __getitem__(self, key: str):
        return ModelOutputTensorWrapper(self.data[key])
    
    def __contains__(self, key: str):
        return key in self.data
    
    def get(self, key: str, default=None):
        if key in self.data:
            return ModelOutputTensorWrapper(self.data[key])
        return default


class ModelOutputTensorWrapper:
    """Wrapper to make numpy arrays behave like torch tensors for search algorithms."""
    
    def __init__(self, numpy_array: np.ndarray):
        self._array = numpy_array
    
    def item(self):
        """Return scalar value like torch tensor."""
        return self._array.item()
    
    def clone(self):
        """Return copy like torch tensor."""
        return ModelOutputTensorWrapper(self._array.copy())
    
    def cpu(self):
        """Already on CPU, return self."""
        return self
    
    def numpy(self):
        """Return numpy array."""
        return self._array
    
    def float(self):
        """Convert to float like torch tensor."""
        return ModelOutputTensorWrapper(self._array.astype(np.float32))
    
    def __getitem__(self, idx):
        """Support indexing like torch tensor."""
        result = self._array[idx]
        if np.isscalar(result):
            return result
        return ModelOutputTensorWrapper(result)
    
    def view(self, *shape):
        """Reshape like torch tensor."""
        return ModelOutputTensorWrapper(self._array.reshape(shape))
    
    def softmax(self, dim=-1):
        """Apply softmax like torch tensor."""
        # Subtract max for numerical stability
        shifted = self._array - np.max(self._array, axis=dim, keepdims=True)
        exp_vals = np.exp(shifted)
        return ModelOutputTensorWrapper(exp_vals / np.sum(exp_vals, axis=dim, keepdims=True))
    
    @property
    def shape(self):
        return self._array.shape
    
    def __gt__(self, other):
        """Support comparison operations."""
        if isinstance(other, (int, float)):
            return self._array > other
        return self._array > other._array
    
    def __setitem__(self, idx, value):
        """Support assignment like torch tensor."""
        if isinstance(value, ModelOutputTensorWrapper):
            self._array[idx] = value._array
        else:
            self._array[idx] = value
    
    def __neg__(self):
        """Support negation."""
        return ModelOutputTensorWrapper(-self._array)
    
    def __add__(self, other):
        """Support addition."""
        if isinstance(other, ModelOutputTensorWrapper):
            return ModelOutputTensorWrapper(self._array + other._array)
        return ModelOutputTensorWrapper(self._array + other)
    
    def __sub__(self, other):
        """Support subtraction."""
        if isinstance(other, ModelOutputTensorWrapper):
            return ModelOutputTensorWrapper(self._array - other._array)
        return ModelOutputTensorWrapper(self._array - other)
    
    def __mul__(self, other):
        """Support multiplication."""
        if isinstance(other, ModelOutputTensorWrapper):
            return ModelOutputTensorWrapper(self._array * other._array)
        return ModelOutputTensorWrapper(self._array * other)
    
    def sum(self, axis=None, keepdims=False):
        """Support sum operation like torch tensor."""
        return ModelOutputTensorWrapper(np.sum(self._array, axis=axis, keepdims=keepdims))
    
    def max(self, axis=None, keepdims=False):
        """Support max operation like torch tensor."""
        return ModelOutputTensorWrapper(np.max(self._array, axis=axis, keepdims=keepdims))
    
    def min(self, axis=None, keepdims=False):
        """Support min operation like torch tensor."""
        return ModelOutputTensorWrapper(np.min(self._array, axis=axis, keepdims=keepdims))
    
    def __getattr__(self, name):
        """Fallback to numpy array attributes."""
        return getattr(self._array, name)


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
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics to zero."""
        self.metrics = {
            'num_nodes': 0,
            'num_searches': 0,
            'policy_perplexity': 0,
            'depth': 0,
        }

    def _send_inference_request(self, fens: List[str], timeout: float = 30.0) -> Dict[str, np.ndarray]:
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
                    
                    if response.results is not None:
                        # Return numpy arrays directly - no tensor operations needed
                        return response.results
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
        # Track inference call
        self.metrics['num_nodes'] += 1
        
        fens = [board.fen()]
        results = self._send_inference_request(fens)
        
        # Calculate policy perplexity if policy is available
        if 'policy' in results:
            policy_array = results['policy'][0]  # Get first (and only) batch element
            # Create legal mask for the board
            legal_mask = np.zeros_like(policy_array, dtype=bool)
            
            # Import here to avoid circular imports
            from searchless_chess.src import utils
            
            for move in board.legal_moves:
                s1, s2 = utils.move_to_indices(move, flip=board.turn == chess.BLACK)
                legal_mask[s1, s2] = True
            
            # Calculate perplexity over legal moves
            if np.any(legal_mask):
                legal_logits = policy_array[legal_mask]
                # Subtract max for numerical stability
                shifted = legal_logits - np.max(legal_logits)
                exp_vals = np.exp(shifted)
                probs = exp_vals / np.sum(exp_vals)
                
                # Calculate entropy and perplexity
                log_probs = np.log(probs + 1e-12)
                entropy = -np.sum(probs * log_probs)
                perplexity = np.exp(entropy)
                self.metrics['policy_perplexity'] += perplexity
        
        # Wrap numpy arrays to behave like torch tensors for search algorithm compatibility
        return ModelOutputWrapper(results)

    def analyse_batch(self, boards: List[chess.Board]) -> ModelOutputWrapper:
        """Send multiple boards for batch analysis."""
        # Track inference calls (one per board in batch)
        self.metrics['num_nodes'] += len(boards)
        
        fens = [board.fen() for board in boards]
        results = self._send_inference_request(fens)
        return ModelOutputWrapper(results)

    def play(self, board: chess.Board) -> chess.Move:
        """Play a move using the configured search strategy."""
        # Track search call
        self.metrics['num_searches'] += 1
        self.metrics['depth'] = self.search_depth
        
        # Get the appropriate search algorithm
        search_algorithm = self.search_algorithms.get(self.strategy)
        if search_algorithm is None:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Reset search algorithm metrics before search
        if hasattr(search_algorithm, 'metrics'):
            search_algorithm.metrics = {
                'num_nodes': 0,
                'depth': 0,
            }

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
        
        # Update metrics from search algorithm (these add to our own inference metrics)
        if hasattr(search_algorithm, 'metrics'):
            for key, value in search_algorithm.metrics.items():
                if key in self.metrics and key != 'num_nodes':  # Don't double count num_nodes
                    self.metrics[key] += value
                elif key == 'depth':  # Update depth from search algorithm
                    self.metrics[key] = value
        
        return result.move 
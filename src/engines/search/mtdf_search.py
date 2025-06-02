import chess
from typing import Optional, Dict
from collections import defaultdict

from .base import SearchAlgorithm, SearchResult
from .pvs_search import PVSSearch, NULL_EPS, NodeType
from searchless_chess.src.engines.utils.node import Node, TTEntry


class MTDFSearch(PVSSearch):
    """MTD(f) search algorithm that uses iterative zero-window searches."""
    
    def __init__(self):
        super().__init__()
        self.name = "mtdf"
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform MTD(f) search with iterative deepening.
        """
        num_nodes = kwargs.get('num_nodes', 400)
        
        # Store inference function for use in node creation
        self.inference_func = inference_func
        self.tt_hits = 0  # Reset TT hit counter
        
        # Create root node
        root = self._create_node(board, inference_func)
        history = defaultdict(int)
        tt = defaultdict(lambda: None)
        
        start_depth = 2.0
        node_count = self.metrics['num_nodes']
        current_depth = start_depth
        best_score = None
        best_move = None
        
        # Iterative deepening
        while self.metrics['num_nodes'] - node_count < num_nodes * 0.95 and current_depth < 20:
            f = best_score - NULL_EPS / 2 if best_score is not None else None
            score, move = self._mtdf(root, current_depth, history, tt, f)
            
            if move is not None:
                best_score = score
                best_move = move
            
            current_depth += 0.2
        
        self.metrics['depth'] = current_depth
        
        if best_move is None:
            # Fallback - just pick the first move from policy
            if root.policy:
                best_move = root.policy[0][0]
            else:
                best_move = list(board.legal_moves)[0]
        
        return SearchResult(
            move=best_move,
            score=best_score if best_score is not None else 0.0,
            metadata={
                'depth': current_depth,
                'nodes': self.metrics['num_nodes'],
                'tt_hits': self.tt_hits,
                'tt_entries': len([entry for entry in tt.values() if entry is not None])
            }
        )
    
    def _mtdf(self, root: Node, depth: float, history: Dict[str, int], tt: Dict[str, TTEntry], f=None) -> tuple[float, Optional[chess.Move]]:
        """
        Performs MTD(f) search with policy-based move ordering and depth extension.
        Returns score relative to the current player and the best move found.
        
        MTD(f) uses iterative zero-window searches to find the exact minimax value.
        It's more efficient than regular alpha-beta because it uses narrower search windows.
        """
        # Initial guess - use the static evaluation of the root if not provided
        if f is None:
            f = root.value
        best_move = None
        max_iterations = 15  # Prevent infinite loops
        
        upperbound = 1.0   # Maximum possible score (win)
        lowerbound = -1.0  # Minimum possible score (loss)
        
        for iteration in range(max_iterations):
            # Choose search window based on current bounds
            if abs(f - lowerbound) < NULL_EPS:
                # Test if true value is > f (upper bound search)
                alpha = f
                beta = f + NULL_EPS
            else:
                # Test if true value is < f (lower bound search)  
                alpha = f - NULL_EPS
                beta = f
                
            # Perform zero-window search using PVS
            score, move = self._pvs(root, depth, alpha, beta, history, tt)
            
            if move is not None:
                best_move = move
                
            # Update our bounds and f based on the result
            if score <= alpha:
                # True value is <= alpha, so update upper bound
                upperbound = min(upperbound, score)
                f = score
            elif score >= beta:
                # True value is >= beta, so update lower bound
                lowerbound = max(lowerbound, score)
                f = score
            else:
                # We have an exact value (alpha < score < beta)
                f = score
                break
                
            # Check for convergence - bounds have essentially met
            if upperbound <= lowerbound + NULL_EPS:
                f = (upperbound + lowerbound) / 2.0
                break
                
            # Safety check against getting stuck
            if abs(f) >= 1.0 - NULL_EPS:
                break
        
        return f, best_move 
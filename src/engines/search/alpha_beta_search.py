import chess
from typing import Optional, Tuple

from .base import SearchAlgorithm, SearchResult
from .negamax_search import NegamaxSearch
from searchless_chess.src.engines.strategy import MoveSelectionStrategy


class AlphaBetaSearch(NegamaxSearch):
    """Search algorithm that performs alpha-beta search (negamax variant) with move ordering."""
    
    def __init__(self, ordering_strategy: Optional[MoveSelectionStrategy] = MoveSelectionStrategy.AVS):
        super().__init__(ordering_strategy)
        self.name = "alpha_beta"
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform alpha-beta search and return the best move and score.
        """
        self.metrics['depth'] = depth
        score, move = self._alpha_beta(board, int(depth), -1.0, 1.0, inference_func)
        
        return SearchResult(
            move=move,
            score=score,
            metadata={
                'depth': depth,
                'nodes': self.metrics['num_nodes'],
                'ordering_strategy': str(self.ordering_strategy) if self.ordering_strategy else None
            }
        )
    
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, inference_func) -> Tuple[float, Optional[chess.Move]]:
        """
        Performs alpha-beta search (negamax variant) with move ordering.
        Returns score relative to the current player and the best move found.
        alpha: Lower bound (best score maximizing player can guarantee).
        beta: Upper bound (best score minimizing player can guarantee).
        """
        # Check for game over conditions first
        if board.is_checkmate():
            # Player whose turn it is got checkmated. Worst score.
            return -1.0, None
        if board.is_stalemate() or board.is_insufficient_material():
            # Draw
            return 0.0, None
            
        if depth == 0:
            # At leaf node, use the static evaluator
            return self._static_evaluate(board, inference_func), None

        max_eval = -float('inf')  # Represents the best score found *so far* for the maximizing player
        best_move = None

        ordered_moves, policy_weight = self._get_ordered_moves(board, self.ordering_strategy, inference_func)

        for move, move_weight in zip(ordered_moves, policy_weight):
            board.push(move)
            # Recursive call: Negate and swap alpha/beta bounds
            score, _ = self._alpha_beta(board, depth - 1, -beta, -alpha, inference_func)
            score = -score  # Negate score back to current player's perspective
            board.pop()

            if score > max_eval:
                max_eval = score
                best_move = move  # Update best move if this path is better

            # Update alpha (the lower bound for the current maximizing player)
            alpha = max(alpha, max_eval)

            # Pruning: If the current maximizing player can guarantee a score >= beta,
            # the minimizing parent node (which provided beta) will not choose this path.
            if alpha >= beta:
                break  # Beta cut-off

        return max_eval, best_move 
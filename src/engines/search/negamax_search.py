import chess
from typing import Optional, Tuple

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src.engines.strategy import MoveSelectionStrategy


class NegamaxSearch(SearchAlgorithm):
    """Search algorithm that performs plain negamax search with move ordering."""
    
    def __init__(self, ordering_strategy: Optional[MoveSelectionStrategy] = MoveSelectionStrategy.AVS):
        super().__init__("negamax")
        self.ordering_strategy = ordering_strategy
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform negamax search and return the best move and score.
        """
        self.metrics['depth'] = depth
        score, move = self._negamax(board, int(depth), inference_func)
        
        return SearchResult(
            move=move,
            score=score,
            metadata={
                'depth': depth,
                'nodes': self.metrics['num_nodes'],
                'ordering_strategy': str(self.ordering_strategy) if self.ordering_strategy else None
            }
        )
    
    def _negamax(self, board: chess.Board, depth: int, inference_func) -> Tuple[float, Optional[chess.Move]]:
        """
        Performs plain negamax search with optional move ordering.
        Returns a tuple of (score relative to the current player to move, best move found).
        """
        if board.is_checkmate():
            # Player whose turn it is got checkmated. Worst score.
            return -1.0, None
        if board.is_stalemate() or board.is_insufficient_material():
            # Draw
            return 0.0, None
            
        if depth == 0:
            # At leaf node, use the static evaluator
            return self._static_evaluate(board, inference_func), None

        max_eval = -float('inf')  # Best score found so far for the current player
        best_move = None

        ordered_moves, _ = self._get_ordered_moves(board, self.ordering_strategy, inference_func)

        for move in ordered_moves:
            board.push(move)
            # Score returned by recursive call is relative to the next player. Negate it.
            score, _ = self._negamax(board, depth - 1, inference_func)
            score = -score
            board.pop()
            if score > max_eval:
                max_eval = score
                best_move = move
            
        return max_eval, best_move
    
    def _static_evaluate(self, board: chess.Board, inference_func) -> float:
        """
        Evaluates a position using the shallow analysis model.
        Returns score relative to the current player [-1 (loss), 0 (draw), 1 (win)].
        """
        self.metrics['num_nodes'] += 1
        output = inference_func(board)
        # Assuming model value v (0-1) is score for the current player
        # (1 = current player wins, 0 = current player loses)
        model_val = output['value'][0, 0].item() 
        # Convert to score relative to current player: score = 2 * model_val - 1
        # 1 -> 1, 0.5 -> 0, 0 -> -1
        score = 2.0 * model_val - 1.0
        return score
    
    def _get_ordered_moves(self, board: chess.Board, ordering_strategy: Optional[MoveSelectionStrategy], inference_func) -> Tuple[list[chess.Move], list[float]]:
        """
        Gets legal moves ordered by the specified strategy head's output,
        or by a default heuristic if strategy is None.
        """
        self.metrics['num_nodes'] += 1

        if not ordering_strategy or ordering_strategy not in [
            MoveSelectionStrategy.AVS, MoveSelectionStrategy.AVS2,
            MoveSelectionStrategy.POLICY, MoveSelectionStrategy.OPT_POLICY_SPLIT]:
            # Return moves in default order
            return list(board.legal_moves), [1.0] * len(list(board.legal_moves))
        
        output = inference_func(board)
        
        # Get policy values and make legal moves have valid probabilities
        policies = output['policy'].clone()
        is_legal = output['legal'].clone() > 0
        policies[~is_legal] = float('-inf')
        policies = policies.view(1, -1).softmax(dim=-1).view(1, is_legal.shape[1], -1)
        
        move_scores = []
        from searchless_chess.src import utils
        
        for move in board.legal_moves:
            s1, s2 = utils.move_to_indices(move, flip=board.turn == chess.BLACK)
            score = policies[0, s1, s2].item()
            move_scores.append(score)
            
        # Sort moves based on scores (descending)
        sorted_data = sorted(zip(move_scores, board.legal_moves), key=lambda pair: pair[0], reverse=True)
        sorted_moves = [move for _, move in sorted_data]
        policy_weights = [score for score, _ in sorted_data]
        
        return sorted_moves, policy_weights 
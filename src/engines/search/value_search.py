import chess
import torch
from typing import cast

from .base import SearchAlgorithm, SearchResult


class ValueSearch(SearchAlgorithm):
    """Search algorithm that evaluates all legal moves and picks the best value."""
    
    def __init__(self):
        super().__init__("value")
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Evaluate all legal moves using the value head and return the move with the best value.
        """
        self.metrics['num_nodes'] += len(list(board.legal_moves))
        
        # Get value predictions for all legal moves
        if batch_inference_func:
            # Use batch inference for efficiency
            x = []
            moves = list(board.legal_moves)
            for move in moves:
                board.push(move)
                x.append(board.fen())
                board.pop()
            
            outputs = batch_inference_func(x)
            values = outputs['value'][:, 0].clone()
        else:
            # Fallback to individual inference calls
            values = []
            moves = list(board.legal_moves)
            for move in moves:
                board.push(move)
                output = inference_func(board)
                value = output['value'][0, 0].item()
                values.append(value)
                board.pop()
            values = torch.tensor(values)
        
        # Check for immediate checkmates and stalemates
        for i, move in enumerate(moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return SearchResult(
                    move=move,
                    score=1.0,  # Checkmate is best possible
                    metadata={'type': 'checkmate', 'nodes': self.metrics['num_nodes']}
                )
            if board.is_stalemate():
                values[i] = 0.5  # Neutral for stalemate
            board.pop()
        
        # Find move with minimum value (best for opponent, so worst for us)
        best_ix = cast(int, torch.argmin(values).item())
        best_move = moves[best_ix]
        best_value = values[best_ix].item()
        
        return SearchResult(
            move=best_move,
            score=1.0 - best_value,  # Convert to current player's perspective
            metadata={
                'value': best_value,
                'nodes': self.metrics['num_nodes']
            }
        ) 
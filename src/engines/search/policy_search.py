import chess
import torch
from typing import cast

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src import utils


class PolicySearch(SearchAlgorithm):
    """Search algorithm that uses the policy head to select moves."""
    
    def __init__(self, policy_type: str = "policy"):
        super().__init__(f"policy_{policy_type}")
        self.policy_type = policy_type
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Use the policy head to evaluate moves and return the move with highest policy probability.
        """
        self.metrics['num_nodes'] += 1
        
        # Get policy predictions from the model
        output = inference_func(board)
        policy = output[self.policy_type][0, :, :].clone()
        
        move_values = []
        moves = list(board.legal_moves)
        
        for i, move in enumerate(moves):
            # Check for immediate checkmate
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return SearchResult(
                    move=move,
                    score=1.0,  # Checkmate is best possible
                    metadata={'type': 'checkmate', 'policy_score': 1.0, 'nodes': self.metrics['num_nodes']}
                )
            board.pop()
            
            # Get policy score for this move
            s1, s2 = utils.move_to_indices(move, flip=board.turn == chess.BLACK)
            policy_score = policy[s1, s2].item()
            move_values.append((policy_score, i))
        
        # Find move with highest policy probability
        best_policy_score, best_idx = max(move_values)
        best_move = moves[best_idx]
        
        return SearchResult(
            move=best_move,
            score=best_policy_score,
            metadata={
                'policy_score': best_policy_score,
                'policy_type': self.policy_type,
                'nodes': self.metrics['num_nodes']
            }
        ) 
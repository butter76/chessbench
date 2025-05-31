import chess

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src import utils


class AVSSearch(SearchAlgorithm):
    """Search algorithm that uses the AVS (Action Value Score) head to select moves."""
    
    def __init__(self, avs_type: str = "avs"):
        super().__init__(f"avs_{avs_type}")
        self.avs_type = avs_type
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Use the AVS head to evaluate moves and return the move with the highest AVS score.
        """
        self.metrics['num_nodes'] += 1
        
        # Get AVS predictions from the model
        output = inference_func(board)
        avs = output[self.avs_type][0, :, :].clone()
        
        move_values = []
        moves = list(board.legal_moves)
        
        for i, move in enumerate(moves):
            board.push(move)
            
            # Check for immediate game ending conditions
            if board.is_checkmate():
                board.pop()
                return SearchResult(
                    move=move,
                    score=1.0,  # Checkmate is best possible
                    metadata={'type': 'checkmate', 'avs_score': 1.0, 'nodes': self.metrics['num_nodes']}
                )
            
            if board.is_stalemate() or board.is_insufficient_material():
                # Stalemate/insufficient material results in draw
                best_res = 0.5
                board.pop()
            else:
                board.pop()
                # Get AVS score for this move
                s1, s2 = utils.move_to_indices(move, flip=board.turn == chess.BLACK)
                best_res = avs[s1, s2].item()
            
            move_values.append((best_res, i))
        
        # Find move with highest AVS score
        best_avs_score, best_idx = max(move_values)
        best_move = moves[best_idx]
        
        return SearchResult(
            move=best_move,
            score=best_avs_score,
            metadata={
                'avs_score': best_avs_score,
                'avs_type': self.avs_type,
                'nodes': self.metrics['num_nodes']
            }
        ) 
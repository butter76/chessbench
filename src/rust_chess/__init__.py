"""
Rust-powered Chess Engine Python bindings
"""
import numpy as np
try:
    from .chess_bindings import PyChessEngine, __version__
except ImportError:
    raise ImportError(
        "Failed to import Rust chess bindings. "
        "Make sure to build the Rust package first using maturin."
    )

__all__ = ["ChessEngine", "__version__"]


class ChessEngine:
    """
    Python wrapper for the Rust chess engine
    """
    
    def __init__(self, fen=None):
        """
        Initialize a new chess engine
        
        Args:
            fen: Optional FEN string to start from a specific position
        """
        if fen is not None:
            self._engine = PyChessEngine.from_fen(fen)
        else:
            self._engine = PyChessEngine()
    
    def get_fen(self):
        """Get the current position as FEN string"""
        return self._engine.get_fen()
    
    def get_legal_moves(self):
        """Get a list of legal moves in UCI format"""
        return self._engine.get_legal_moves()
    
    def make_move(self, move):
        """
        Make a move on the board
        
        Args:
            move: Move in UCI format (e.g., "e2e4")
            
        Raises:
            ValueError: If the move is invalid or illegal
        """
        self._engine.make_move(move)
    
    def make_array(self):
        """
        Create a numpy array from Rust
        
        Returns:
            numpy.ndarray: A sample array created in Rust
        """
        return self._engine.make_array()
        
    def register_on_move_callback(self, callback):
        """
        Register a callback to be called when a move is made
        
        Args:
            callback: A function that takes a dictionary with move information
                      The dictionary contains:
                      - 'move': The move in UCI format
        """
        self._engine.register_on_move_callback(callback)
        
    def register_on_game_end_callback(self, callback):
        """
        Register a callback to be called when the game ends
        
        Args:
            callback: A function that takes a dictionary with game end information
                      The dictionary contains:
                      - 'result': The game result (e.g., "1-0", "0-1", "1/2-1/2")
                      - 'reason': The reason for the game ending (e.g., "checkmate", "stalemate")
                      - 'final_position': The final position as FEN
        """
        self._engine.register_on_game_end_callback(callback)
        
    def register_on_eval_callback(self, callback):
        """
        Register a callback to be called when a position is evaluated
        
        Args:
            callback: A function that takes a dictionary with evaluation information
                      The dictionary contains:
                      - 'score': The evaluation score
                      - 'depth': The search depth
                      - 'position': The evaluated position as FEN
        """
        self._engine.register_on_eval_callback(callback)
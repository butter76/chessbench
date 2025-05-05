"""
Rust-powered Chess Engine Python bindings
"""
import numpy as np
try:
    from .chess_bindings import ThreadManager, __version__
except ImportError:
    raise ImportError(
        "Failed to import Rust chess bindings. "
        "Make sure to build the Rust package first using maturin."
    )

__all__ = ["ChessEngine", "__version__"]


class ChessEngine:
    """
    Python wrapper for the Rust chess engine using the new ThreadManager API
    """
    
    def __init__(self, fen=None):
        """
        Initialize a new chess engine
        
        Args:
            fen: Optional FEN string to start from a specific position
        """
        self._manager = ThreadManager()
        self._engine_id = None
        self._on_move_callback = None
        self._on_eval_callback = None
        self._on_fen_callback = None
        
        # Create the engine
        self._engine_id = self._manager.create_thread(
            fen=fen,
            on_move_callback=self._handle_move_callback,
            on_fen_callback=self._handle_fen_callback,
            on_eval_callback=self._handle_eval_callback
        )
        
    def _handle_move_callback(self, data):
        """Internal callback handler for move events"""
        if self._on_move_callback:
            return self._on_move_callback(data)
        return None
        
    def _handle_fen_callback(self, data):
        """Internal callback handler for FEN events"""
        if self._on_fen_callback:
            return self._on_fen_callback(data)
        return None
        
    def _handle_eval_callback(self, data):
        """Internal callback handler for evaluation events"""
        if self._on_eval_callback:
            return self._on_eval_callback(data)
        return None
    
    def start(self):
        """Start the search thread"""
        if self._engine_id is not None:
            self._manager.start_thread(self._engine_id)
    
    def stop(self):
        """Stop the search thread"""
        if self._engine_id is not None:
            self._manager.stop_thread(self._engine_id)
    
    def is_running(self):
        """Check if the search thread is running"""
        if self._engine_id is not None:
            return self._manager.is_running(self._engine_id)
        return False
    
    def is_evaluating(self):
        """Check if the search thread is evaluating a position"""
        if self._engine_id is not None:
            return self._manager.is_evaluating(self._engine_id)
        return False
    
    def is_waiting(self):
        """Check if the search thread is waiting for input"""
        if self._engine_id is not None:
            return self._manager.is_waiting(self._engine_id)
        return False
    
    def set_position(self, fen):
        """
        Set a new position using FEN notation
        
        Args:
            fen: FEN string representing the position
        """
        if self._engine_id is not None:
            self._manager.receive_fen(self._engine_id, fen)
    
    def make_move(self, uci_move):
        """
        Make a move on the board
        
        Args:
            move: Move in UCI format (e.g., "e2e4")
        """
        if self._engine_id is not None:
            self._manager.receive_move(self._engine_id, uci_move)
    
    def submit_eval(self, value, policy=None):
        """
        Submit an evaluation for the current position
        
        Args:
            value: Evaluation value
            policy: Optional move policy as list of (move, probability) tuples
        """
        if self._engine_id is not None:
            policy = policy or []
            self._manager.receive_eval(self._engine_id, value, policy)
    
    def register_on_move_callback(self, callback):
        """
        Register a callback to be called when a move is made
        
        Args:
            callback: A function that takes a dictionary with move information
        """
        self._on_move_callback = callback
    
    def register_on_fen_callback(self, callback):
        """
        Register a callback to be called when a new position is set
        
        Args:
            callback: A function that takes a dictionary with position information
        """
        self._on_fen_callback = callback
    
    def register_on_eval_callback(self, callback):
        """
        Register a callback to be called when a position is evaluated
        
        Args:
            callback: A function that takes a dictionary with evaluation information
        """
        self._on_eval_callback = callback
        
    def __del__(self):
        """Clean up resources when the engine is deleted"""
        if self._engine_id is not None:
            try:
                self._manager.stop_thread(self._engine_id)
            except:
                pass
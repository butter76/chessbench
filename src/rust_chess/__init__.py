"""
Rust-powered Chess Engine Python bindings
"""
import numpy as np
try:
    from .chess_bindings import PyChessEngine, PyEngineManager, __version__
except ImportError:
    raise ImportError(
        "Failed to import Rust chess bindings. "
        "Make sure to build the Rust package first using maturin."
    )

__all__ = ["ChessEngine", "EngineManager", "__version__"]


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


class EngineManager:
    """
    Manager for multiple chess engines running in separate threads
    """
    
    def __init__(self, max_concurrent=None):
        """
        Initialize a new engine manager
        
        Args:
            max_concurrent: Maximum number of engines that can run concurrently.
                           If None, defaults to the number of CPU cores.
        """
        self._manager = PyEngineManager(max_concurrent)
        self._engines = {}
    
    def create_engine(self, fen=None, on_move=None, on_game_end=None, on_eval=None):
        """
        Create a new engine instance
        
        Args:
            fen: Optional FEN string for starting position
            on_move: Callback for move events
            on_game_end: Callback for game end events
            on_eval: Callback for evaluation events
            
        Returns:
            int: Engine ID that can be used to reference this engine later
        """
        engine_id = self._manager.create_engine(fen, on_move, on_game_end, on_eval)
        self._engines[engine_id] = engine_id
        return engine_id
    
    def make_move(self, engine_id, move):
        """
        Make a move on the specified engine
        
        Args:
            engine_id: ID of the engine
            move: Move in UCI format (e.g., "e2e4")
            
        Raises:
            ValueError: If the engine ID is invalid or the move is illegal
        """
        return self._manager.make_move(engine_id, move)
    
    def get_fen(self, engine_id):
        """
        Get the current position from the specified engine
        
        Args:
            engine_id: ID of the engine
            
        Returns:
            str: FEN string representing the position
        """
        return self._manager.get_fen(engine_id)
    
    def get_legal_moves(self, engine_id):
        """
        Get legal moves from the specified engine
        
        Args:
            engine_id: ID of the engine
            
        Returns:
            list: Legal moves in UCI format
        """
        return self._manager.get_legal_moves(engine_id)
    
    def notify_eval(self, engine_id, score, depth):
        """
        Trigger the evaluation callback on the specified engine
        
        Args:
            engine_id: ID of the engine
            score: Evaluation score
            depth: Search depth
        """
        return self._manager.notify_eval(engine_id, score, depth)
    
    def notify_game_end(self, engine_id, result, reason):
        """
        Trigger the game end callback on the specified engine
        
        Args:
            engine_id: ID of the engine
            result: Game result (e.g., "1-0")
            reason: Reason for game end (e.g., "checkmate")
        """
        return self._manager.notify_game_end(engine_id, result, reason)
    
    def stop_engine(self, engine_id):
        """
        Stop the specified engine
        
        Args:
            engine_id: ID of the engine to stop
        """
        if engine_id in self._engines:
            self._manager.stop_engine(engine_id)
            del self._engines[engine_id]
    
    def stop_all(self):
        """Stop all engines managed by this manager"""
        for engine_id in list(self._engines.keys()):
            self.stop_engine(engine_id)
    
    def active_engines(self):
        """
        Get the number of active engines
        
        Returns:
            int: Number of active engines
        """
        return self._manager.active_engines()
    
    def __del__(self):
        """Clean up resources when the manager is deleted"""
        self.stop_all()
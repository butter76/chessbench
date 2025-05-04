"""
Example usage of the Rust chess engine
"""

from rust_chess import ChessEngine

def on_move(move_info):
    """Callback function for move events"""
    print(f"\nMove callback triggered: {move_info['move']}")
    return {"status": "received", "message": "Move processed by Python"}

def on_game_end(end_info):
    """Callback function for game end events"""
    print(f"\nGame end callback triggered:")
    print(f"  Result: {end_info['result']}")
    print(f"  Reason: {end_info['reason']}")
    print(f"  Final position: {end_info['final_position']}")
    return {"status": "received", "message": "Game end processed by Python"}

def on_eval(eval_info):
    """Callback function for evaluation events"""
    print(f"\nEvaluation callback triggered:")
    print(f"  Score: {eval_info['score']}")
    print(f"  Depth: {eval_info['depth']}")
    print(f"  Position: {eval_info['position']}")
    return {"status": "received", "message": "Evaluation processed by Python"}

def main():
    # Create a new chess engine
    engine = ChessEngine()
    
    # Register callbacks
    engine.register_on_move_callback(on_move)
    engine.register_on_game_end_callback(on_game_end)
    engine.register_on_eval_callback(on_eval)
    
    # Print the starting position
    print("Starting position (FEN):", engine.get_fen())
    
    # Get legal moves
    legal_moves = engine.get_legal_moves()
    print(f"Legal moves ({len(legal_moves)}):", legal_moves)
    
    # Make a move (this will trigger the on_move callback)
    move = "e2e4"  # King's pawn opening
    print(f"Making move: {move}")
    engine.make_move(move)
    
    # Get the new position and legal moves
    print("New position (FEN):", engine.get_fen())
    legal_moves = engine.get_legal_moves()
    print(f"Legal moves ({len(legal_moves)}):", legal_moves)
    
    # Try a specific position from FEN
    sicilian_fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    print("\nCreating position from FEN:", sicilian_fen)
    engine = ChessEngine(fen=sicilian_fen)
    
    # Register callbacks again for the new engine instance
    engine.register_on_move_callback(on_move)
    engine.register_on_game_end_callback(on_game_end)
    engine.register_on_eval_callback(on_eval)
    
    # Get legal moves for this position
    legal_moves = engine.get_legal_moves()
    print(f"Legal moves ({len(legal_moves)}):", legal_moves)
    
    # Make another move (this will trigger the on_move callback again)
    move = "d2d4"  # Open the center
    print(f"Making move: {move}")
    engine.make_move(move)
    
    print("\nNote: The game end and evaluation callbacks would be triggered ")
    print("by the Rust code when those events occur in the chess engine.")
    print("In this example, we're only seeing the move callbacks being triggered.")

    # Get a numpy array from Rust
    array = engine.make_array()
    print("\nNumpy array:", array)
    print(array.shape)
    print(array.dtype)

if __name__ == "__main__":
    main() 
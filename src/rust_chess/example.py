"""
Example usage of the Rust chess engine
"""

import time
from rust_chess import ChessEngine, EngineManager

def on_move(move_info):
    """Callback function for move events"""
    print(f"\nMove callback triggered: {move_info['move']}")
    # time.sleep(0.5)  # Simulate some processing work
    return {"status": "received", "message": "Move processed by Python"}

def on_game_end(end_info):
    """Callback function for game end events"""
    print(f"\nGame end callback triggered:")
    print(f"  Result: {end_info['result']}")
    print(f"  Reason: {end_info['reason']}")
    print(f"  Final position: {end_info['final_position']}")
    time.sleep(0.5)  # Simulate some processing work
    return {"status": "received", "message": "Game end processed by Python"}

def on_eval(eval_info):
    """Callback function for evaluation events"""
    print(f"\nEvaluation callback triggered:")
    print(f"  Score: {eval_info['score']}")
    print(f"  Depth: {eval_info['depth']}")
    print(f"  Position: {eval_info['position']}")
    time.sleep(0.5)  # Simulate some processing work
    return {"status": "received", "message": "Evaluation processed by Python"}

def standard_engine_example():
    """Example using the standard single-threaded engine"""
    print("\n=== Standard Engine Example ===")
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

def threaded_engine_example():
    """Example using the multithreaded engine manager"""
    print("\n=== Threaded Engine Manager Example ===")
    
    # Create an engine manager with 2 concurrent engines
    manager = EngineManager(max_concurrent=2)
    print(f"Created engine manager with 2 concurrent engines")
    
    # Create multiple engine instances
    positions = [
        None,  # Default starting position
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian Defense
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # King's Pawn Opening
    ]
    
    engine_ids = []
    for i, fen in enumerate(positions):
        name = f"Engine {i+1}" + (f" (Default)" if fen is None else f" (Custom)")
        print(f"Creating {name}")
        engine_id = manager.create_engine(
            fen=fen,
            on_move=on_move,
            on_game_end=on_game_end,
            on_eval=on_eval
        )
        engine_ids.append(engine_id)
        print(f"  Engine ID: {engine_id}")
        print(f"  Position: {manager.get_fen(engine_id)}")
    
    print(f"\nActive engines: {manager.active_engines()}")
    
    # Make moves on all engines concurrently
    print("\nMaking moves on all engines concurrently")
    # This demonstrates that even though callbacks take time (sleep 0.5s),
    # they don't block other engines
    start_time = time.time()
    
    for i, engine_id in enumerate(engine_ids):
        moves = [
            "Nf3" if i != 0 else "d2d4",  # Different first moves
            # "g1f3",  # Knight to f3
        ]
        
        for move in moves:
            print(f"Engine {i+1}: Making move {move}")
            manager.make_move(engine_id, move)
            # No sleep here - the engines should process concurrently
    
    elapsed = time.time() - start_time
    print(f"\nAll moves completed in {elapsed:.2f} seconds")
    
    # If this were truly sequential, it would take:
    # 3 engines × 2 moves × 0.5s sleep = 3 seconds minimum
    # Concurrent execution should be faster
    
    # Trigger evaluation callbacks
    print("\nTriggering evaluation callbacks")
    for i, engine_id in enumerate(engine_ids):
        score = 0.1 * (i + 1)  # Different scores for each engine
        manager.notify_eval(engine_id, score, 10)
    
    # Trigger game end callbacks
    print("\nTriggering game end callbacks")
    results = ["1-0", "0-1", "1/2-1/2"]
    reasons = ["checkmate", "resignation", "draw by agreement"]
    
    for i, engine_id in enumerate(engine_ids):
        result = results[i % len(results)]
        reason = reasons[i % len(reasons)]
        manager.notify_game_end(engine_id, result, reason)

    time.sleep(5)
    
    # Stop all engines
    print("\nStopping all engines")
    manager.stop_all()
    print(f"Active engines after stopping: {manager.active_engines()}")

def main():
    # standard_engine_example()
    threaded_engine_example()

if __name__ == "__main__":
    main() 
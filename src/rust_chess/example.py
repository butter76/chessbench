"""
Example usage of the Rust chess engine with the new SearchThread implementation
"""

import time
from rust_chess import ChessEngine

def single_engine_example():
    """Example using a single chess engine"""
    print("\n=== Chess Engine Example with SearchThread ===")
    
    # Create a new chess engine with starting position
    engine = ChessEngine()


    def on_move(move_info):
        """Callback function for move events"""
        print(f"\nMove callback triggered: {move_info}")
        time.sleep(0.5)  # Simulate some processing work
        return {"status": "received", "message": "Move processed by Python"}

    def on_eval(eval_info):
        """Callback function for evaluation events"""
        print(f"\nEvaluation callback triggered:")
        print(f"  Position: {eval_info}")
        time.sleep(0.5)  # Simulate some processing work
        # Don't call submit_eval from within the callback!
        print("\nReceived evaluation request, will reply separately")
        return {"status": "received", "message": "Evaluation processed by Python"}

    def on_fen(fen_info):
        """Callback function for FEN position updates"""
        print(f"\nFEN callback triggered:")
        print(f"  New position: {fen_info}")
        time.sleep(0.5)  # Simulate some processing work
        return {"status": "received", "message": "Position update processed by Python"}
    
    # Register callbacks
    engine.register_on_move_callback(on_move)
    engine.register_on_eval_callback(on_eval)
    engine.register_on_fen_callback(on_fen)
    
    # Set a new position
    sicilian_defense = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    print(f"\nSetting new position: Sicilian Defense")
    engine.set_position(sicilian_defense)

    # Start the engine
    print("Starting engine...")
    engine.start()
    
    # Give time for callbacks to process
    time.sleep(1)
    
    # Submit an evaluation separately (not from within the callback)
    print("\nSubmitting evaluation response")
    engine.submit_eval(0.5, [("d2d4", 0.8), ("c2c4", 0.2)])
    
    # Give more time for processing
    time.sleep(1)
    
    # Check engine status
    print(f"\nEngine status:")
    print(f"  Running: {engine.is_running()}")
    print(f"  Evaluating: {engine.is_evaluating()}")
    print(f"  Waiting: {engine.is_waiting()}")
    
    # Stop the engine
    print("\nStopping engine...")
    engine.stop()

def multiple_engine_example():
    """Example showing how to use multiple engines simultaneously"""
    print("\n=== Multiple Chess Engines Example ===")
    
    # Create engines with different starting positions
    starting_positions = [
        None,  # Default starting position
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian Defense
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # King's Pawn Opening
    ]
    
    engines = []
    for i, pos in enumerate(starting_positions):
        name = f"Engine {i+1}" + (" (Default)" if pos is None else " (Custom)")
        print(f"Creating {name}")
        engine = ChessEngine(fen=pos)
        
        # Create a closure that stores engine index but doesn't call submit_eval
        def make_callback(idx):
            def callback(data):
                print(f"Engine {idx+1} evaluation request: {data}")
                return {"status": "received"}
            return callback
        
        engine.register_on_eval_callback(make_callback(i))
        engines.append(engine)
    
    # Start all engines
    print("\nStarting all engines...")
    for i, engine in enumerate(engines):
        engine.start()
        print(f"Engine {i+1} started: {engine.is_running()}")
    
    # Make different moves on each engine
    moves = [
        "e2e4",  # King's pawn
        "g1f3",  # Knight to f3
        "d2d4",  # Queen's pawn
    ]
    
    print("\nMaking moves on all engines...")
    for i, engine in enumerate(engines):
        move = moves[i % len(moves)]
        print(f"Engine {i+1}: Making move {move}")
        engine.make_move(move)
    
    # Wait for evaluation requests to be processed
    time.sleep(1)
    
    # Submit evaluations to all engines separately (not from callbacks)
    print("\nSubmitting evaluations...")
    for i, engine in enumerate(engines):
        value = 0.1 * (i + 1)
        engine.submit_eval(value)
    
    # Give time for callbacks to process
    time.sleep(2)
    
    # Stop all engines
    print("\nStopping all engines...")
    for i, engine in enumerate(engines):
        engine.stop()
        print(f"Engine {i+1} stopped: not running = {not engine.is_running()}")

def main():
    single_engine_example()
    # multiple_engine_example()

if __name__ == "__main__":
    main() 
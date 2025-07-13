#!/usr/bin/env python3
"""
Test script to demonstrate search tree logging functionality.
"""

import chess
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.append('src')

from searchless_chess.src.engines.my_engine import MyTransformerEngine
from searchless_chess.src.engines.strategy import MoveSelectionStrategy
import chess.engine

def test_search_logging():
    """Test the search tree logging functionality."""
    
    # Check if checkpoint exists
    checkpoint_path = "../checkpoints/p2-other/post_train_checkpoint_6000.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint path in the script.")
        return
    
    # Create engine with verbose logging enabled
    engine = MyTransformerEngine(
        checkpoint_path=checkpoint_path,
        limit=chess.engine.Limit(time=1.0),
        strategy=MoveSelectionStrategy.PVS,  # Use PVS search
        search_depth=2.0,
        num_nodes=50,  # Small number for testing
        verbose=True  # Enable logging
    )
    
    # Test positions
    test_positions = [
        "2q5/5P1k/5Qpp/p2p4/3P4/P2b2PB/5K1P/1r6 w - - 1 44",  # Starting position
    ]
    
    
    for i, fen in enumerate(test_positions, 1):
        print(f"\n--- Test Position {i} ---", file=sys.stderr)
        print(f"FEN: {fen}", file=sys.stderr)
        
        try:
            board = chess.Board(fen)
            print(f"Board:\n{board}", file=sys.stderr)
            
            # Perform search - this will output JSON logs to stdout
            move = engine.play(board)
            
            print(f"Best move: {move}", file=sys.stderr)
            print(f"Move in SAN: {board.san(move)}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error with position {i}: {e}", file=sys.stderr)
        
        print("-" * 40, file=sys.stderr)
    
    print("\nSearch tree logging test completed!", file=sys.stderr)

if __name__ == "__main__":
    test_search_logging() 
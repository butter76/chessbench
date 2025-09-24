#!/usr/bin/env python3
"""
Script to analyze underpromotion policy probabilities for a given FEN position.
"""

import argparse
import chess
from pyparsing import cast
import torch
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path so we can import modules
sys.path.append('src')

from searchless_chess.src.models.transformer import ChessTransformer
from searchless_chess.src.engines.utils.nnutils import get_policy
from searchless_chess.src import tokenizer


def load_model(checkpoint_path: str, device: str = None):
    """Load model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    
    # Create model
    model = ChessTransformer(config=model_config).to(device)

    
    # Load state dict - handle compiled models
    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        # Remove the _orig_mod prefix for compiled models
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model config: {model_config}")
    
    return model, device


def create_inference_func(model, device):
    """Create inference function for the model."""
    def inference_func(board: chess.Board):
        with torch.no_grad():
            # Tokenize the board
            tokens = tokenizer.tokenize(board.fen())
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Run inference
            output = model(x)
            
            return output
    
    return inference_func


def analyze_underpromotions(fen: str, checkpoint_path: str, device: str = None):
    """Analyze underpromotion policy probabilities for a given FEN."""
    # Load model
    model, device = load_model(checkpoint_path, device)
    inference_func = create_inference_func(model, device)
    
    # Parse FEN
    try:
        board = chess.Board(fen)
    except ValueError as e:
        print(f"Invalid FEN: {e}")
        return
    
    print(f"Analyzing position: {fen}")
    print(f"Board:\n{board}")
    print()
    
    # Run inference
    output = inference_func(board)
    
    # Get policy probabilities and U, Q, D values
    policies = output['hardest_policy'].float().cpu().numpy()
    U = output['U'].float().cpu().numpy()
    Q = output['Q'].float().cpu().numpy() 
    D = output['D'].float().cpu().numpy()

    print("Value: ", output['value'].item() * 2 - 1)
    print("MI1: ", output['hl'])
    print("Draw: ", output['draw'].item())
    
    # Get policy with metadata including U, Q, D
    policy_results, policy_map, perplexity = get_policy(board, policies[0], U[0], Q[0], D[0])
    
    # Find underpromotions
    all_moves = []
    for move in board.legal_moves:
        all_moves.append(move)
    
    print(f"Found {len(all_moves)} moves:")
    print("-" * 100)
    
    # Sort underpromotions by probability
    move_data = []
    for move in all_moves:
        prob = policy_map.get(move, 0.0)
        
        # Find the move in policy_results to get metadata
        move_metadata = None
        for policy_move, _, metadata in policy_results:
            if policy_move == move:
                move_metadata = metadata
                break
        
        if move_metadata:
            u_val = move_metadata.get('U', 0.0)
            q_val = move_metadata.get('Q', 0.0)
            d_val = move_metadata.get('D', 0.0)
        else:
            u_val = q_val = d_val = 0.0
            
        move_data.append((move, prob, u_val, q_val, d_val))
    
    # Sort by probability (descending)
    move_data.sort(key=lambda x: x[1], reverse=True)
    
    total_prob = sum(prob for _, prob, _, _, _ in move_data)
    
    print(f"{'Move':<12} {'Piece':<8} {'Probability':<12} {'Percentage':<12} {'U':<10} {'Q':<10} {'D':<10}")
    print("-" * 100)
    
    for move, prob, u_val, q_val, d_val in move_data:
        percentage = prob * 100
        print(f"{move.uci():<12} {prob:<12.6f} {percentage:<12.2f}% {u_val:<10.4f} {q_val:<10.4f} {d_val:<10.4f}")
    
    print("-" * 100)
    print(f"Total move probability: {total_prob:.6f} ({total_prob * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze policy probabilities")
    parser.add_argument("fen", help="FEN string of the position to analyze")
    parser.add_argument("--checkpoint", default="../checkpoints/r1/r1.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--device", default=None, 
                       help="Device to use (cuda/cpu, default: auto)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    analyze_underpromotions(args.fen, args.checkpoint, args.device)


if __name__ == "__main__":
    main() 
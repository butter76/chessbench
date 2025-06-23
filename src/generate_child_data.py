"""Generate child data by loading positions, computing children, and running inference.

This script:
1. Loads a trained model checkpoint
2. Loads chess positions from the dataset (100 at a time by default)
3. For each position, generates all legal non-terminal child positions
4. Runs the neural network on batches of ~2000 child positions to get values/policies
5. Saves the results for further analysis

Usage:
    python generate_child_data.py [checkpoint_path]
    
Example:
    python generate_child_data.py ../checkpoints/p2-dhl/checkpoint_300000.pt
"""

import os
from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')
import chess
import chess.engine
from tqdm import tqdm
import numpy as np
import bagz

from searchless_chess.src import config as config_lib
from searchless_chess.src.dataset import load_datasource
from searchless_chess.src.models.transformer import ChessTransformer
from searchless_chess.src import tokenizer
from searchless_chess.src.constants import CODERS
from searchless_chess.src.data_loader import NUM_BINS


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> ChessTransformer:
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint['model_config']
    
    # Create model that matches the checkpoint
    model = ChessTransformer(config=model_config).to(device)
    
    if checkpoint.get('compiled', False):
        model = torch.compile(model)
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Model config: {model_config}")
    
    return model

def generate_child_positions(fen: str) -> List[Tuple[str, chess.Move]]:
    """Generate all non-terminal child positions from a given FEN."""
    board = chess.Board(fen)
    
    children = []
    
    # Generate all legal moves
    for move in board.legal_moves:
        # Make a copy and apply the move
        child_board = board.copy()
        child_board.push(move)
        
        # Skip if position is terminal (checkmate, stalemate, or draw)
        if not child_board.is_checkmate() and not child_board.is_stalemate() and not child_board.is_insufficient_material() and not child_board.is_fifty_moves():
            children.append((child_board.fen(), move))
        else:
            if child_board.is_checkmate():
                children.append(("-1", move))
            else:
                children.append(("0", move))
    
    return children


def batch_tokenize_positions(fens) -> torch.Tensor:
    """Tokenize a batch of FEN strings."""
    encoded_positions = []
    
    for _, policy, result, root_q, root_d, played_q, played_d, plies_left, move, child_fens in fens:
        for child_fen, _ in child_fens:
            if len(child_fen) <= 2:
                continue
            
            # Use the tokenizer module's tokenize function
            encoded = tokenizer.tokenize(child_fen)
            # Convert numpy array to torch tensor
            encoded_tensor = torch.from_numpy(encoded).long()
            encoded_positions.append(encoded_tensor)
    
    # Stack into a batch tensor
    return torch.stack(encoded_positions)

def main():
    """Main function to generate child data."""
    import sys
    
    # Configuration - make checkpoint path configurable
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "../checkpoints/p2-dhl-2x/checkpoint_300000.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Load model
    print("Loading model...")
    model = load_checkpoint(checkpoint_path, device)
    
    bag_source = bagz.BagDataSource('../processed_data/processed_lc0_data_202308*.bag')
    
    # Process data
    print("Processing positions and generating child data...")
    
    total_children = 0
    total_positions = 0
    
    # Create output directory
    output_dir = "../data/child_data"
    os.makedirs(output_dir, exist_ok=True)
    MAX_POSITIONS = 200_000_000

    with bagz.BagWriter('../data/child_data/child_data.bag') as writer:

        fens = []
    
        for batch_idx, batch_data in enumerate(tqdm(bag_source, desc="Processing batches")):
            
            # Process the batch
            fen, policy, result, root_q, root_d, played_q, played_d, plies_left, move = CODERS['lc0_data'].decode(batch_data)

            child_fens = generate_child_positions(fen)

            fens.append((fen, policy, result, root_q, root_d, played_q, played_d, plies_left, move, child_fens))
            
            total_positions += 1

            if (total_positions % 100_000) == 0:
                print(f"Processed {total_positions} positions")


            if len(fens) >= 100 or total_positions >= MAX_POSITIONS:
                encoded_positions = batch_tokenize_positions(fens)
                
                with torch.inference_mode():
                    encoded_positions = encoded_positions.to(device)
                    output = model(encoded_positions)

                    hl = output['hl']
                    bin_centers = torch.tensor([(2 * i + 1) / (2 * NUM_BINS) for i in range(NUM_BINS)], device=device)

                    hl_probs = torch.softmax(hl, dim=-1)
                    hl_mean = torch.sum(hl_probs * bin_centers, dim=-1)
                    hl_variance = torch.sum(hl_probs * bin_centers**2, dim=-1) - hl_mean**2
                    wdl_variance = hl_variance

                    value = output['value']
                    draw = output['draw']
                    
                wdl_variance = wdl_variance.cpu().numpy()
                value = value.cpu().numpy()
                draw = draw.cpu().numpy()
                
                idx = 0
                for fen, policy, result, root_q, root_d, played_q, played_d, plies_left, move, child_fens in fens:
                    Us = []
                    for child_fen, child_move in child_fens:
                        if len(child_fen) <= 2:
                            U = 0
                            Q = 0 if child_fen == "-1" else 0.5
                            D = 1 if child_fen == "0" else 0
                        else:
                            U = wdl_variance[idx]
                            Q = value[idx]
                            D = draw[idx]
                            idx += 1
                        Us.append((child_move.uci(), U, Q, D))
                    output = CODERS['lc0_data_with_U'].encode((fen, policy, result, root_q, root_d, played_q, played_d, plies_left, move, Us))

                    writer.write(output)

                fens = []
            
            
            # Stop after processing the specified number of records
            if total_positions >= MAX_POSITIONS:
                break
    
    print(f"\nCompleted processing!")
    print(f"Total positions processed: {total_positions}")
    print(f"Total children generated: {total_children}")
    print(f"Average children per position: {total_children / total_positions:.2f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

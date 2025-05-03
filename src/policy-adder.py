#!/usr/bin/env python3

"""
Add policy vectors to action_values bag files.

This script reads in .bag files containing action_values elements, uses a teacher model
to generate policy vectors for each board position, and writes the modified elements
to new .bag files.
"""

import argparse
import os
import glob
import sys
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
import numpy as np
import chess
from tqdm import tqdm

from apache_beam import coders
from searchless_chess.src import bagz
from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src import dataset
from searchless_chess.src import utils
from searchless_chess.src import tokenizer
from searchless_chess.src.models.teacher import ChessTransformer, TransformerConfig

class ActionValuesWithPolicyData:
    """Data with action values and policy vectors."""
    def __init__(self, fen: str, move_values: List[Tuple[str, float]], 
                 policy_vectors: List[Tuple[int, int, float]], 
                 teacher_value: float):
        self.fen = fen
        self.move_values = move_values
        self.policy_vectors = policy_vectors  # (from_square, to_square, probability)
        self.teacher_value = teacher_value    # Teacher's evaluation of the position

# Add new coder for policy vectors
POLICY_VECTOR_CODER = coders.TupleCoder([
    coders.BigIntegerCoder(),  # from_square
    coders.BigIntegerCoder(),  # to_square
    coders.FloatCoder(),  # probability
    coders.FloatCoder(),  # probability_2 (opt_policy_split)
])

# Add to the CODERS dictionary for our new data type
constants.CODERS['action_values_with_policy'] = coders.TupleCoder([
    constants.CODERS['fen'],
    coders.IterableCoder(
        coders.TupleCoder([constants.CODERS['move'], constants.CODERS['win_prob']])
    ),
    coders.IterableCoder(POLICY_VECTOR_CODER),
    coders.FloatCoder(),  # teacher_value
])

def encode_action_values_with_policy(data: ActionValuesWithPolicyData) -> bytes:
    """Encode ActionValuesWithPolicyData to bytes."""
    return constants.CODERS['action_values_with_policy'].encode(
        (data.fen, data.move_values, data.policy_vectors, data.teacher_value)
    )

def load_teacher_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> ChessTransformer:
    """Load teacher model from checkpoint."""
    print(f"Loading teacher model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config']
        
        # Create model with the saved configuration
        model = ChessTransformer(config=model_config).to(device)
        
        # Compile model
        if checkpoint.get('compiled', False):
            model = torch.compile(model)
        
        # Load state dict
        model.load_state_dict(checkpoint['model'])
        
        # Set model to evaluation mode
        model.eval()
            
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def process_bag_file(
    input_path: str,
    output_path: str,
    teacher_model: ChessTransformer,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Process a bag file, adding policy vectors to each element."""
    print(f"Processing {input_path} -> {output_path}")

    # Configure data loader for input bag file
    config = config_lib.DataConfig(
        batch_size=batch_size,
        shuffle=False,
        worker_count=0,
        num_return_buckets=128,
        policy='action_values',
        split='train',
        dataset_path=input_path,
    )
    
    try:
        data_loader = dataset.load_datasource(config)
    except Exception as e:
        print(f"Error loading data source: {e}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read all original elements from the bag file
    try:
        source = bagz.BagReader(input_path)
        total_elements = len(source)
        print(f"Total elements in source file: {total_elements}")
    except Exception as e:
        print(f"Error reading source file: {e}")
        return

    # Create BagWriter for output
    try:
        with bagz.BagWriter(output_path) as writer:
            batch_counter = 0
            processed_elements = 0
            
            # Use tqdm for progress tracking
            with tqdm(total=total_elements, desc="Processing") as pbar:
                for x, legal_actions in data_loader:
                    batch_counter += 1
                    batch_size_actual = x.size(0)
                    
                    x = x.to(torch.long).to(device)
                    legal_actions = legal_actions.to(torch.float32).to(device)
                    
                    # Generate policy vectors with teacher model
                    with torch.inference_mode(), autocast(device, dtype=torch.bfloat16):
                        lesson = teacher_model(x)
                    
                    policy = lesson['policy'].clone()
                    policy[~legal_actions.bool()] = -float('inf')
                    policy = F.softmax(policy.view(-1, 68 * 68), dim=-1).view(-1, 68, 68)
                    
                    # Also process opt_policy_split 
                    opt_policy_split = lesson['opt_policy_split'].clone()
                    opt_policy_split[~legal_actions.bool()] = -float('inf')
                    opt_policy_split = F.softmax(opt_policy_split.view(-1, 68 * 68), dim=-1).view(-1, 68, 68)
                    
                    # For each item in the batch
                    for i in range(batch_size_actual):
                        element_idx = (batch_counter - 1) * batch_size + i
                        if element_idx >= total_elements:
                            break
                        
                        # Get original data from bag file
                        original_data = source[element_idx]
                        fen, move_values = constants.CODERS['action_values'].decode(original_data)
                        
                        # Extract policy vectors in one batch operation
                        legal_move_positions = torch.nonzero(legal_actions[i].bool())
                        s1s = legal_move_positions[:, 0]
                        s2s = legal_move_positions[:, 1]
                        probs = policy[i, s1s, s2s]
                        probs2 = opt_policy_split[i, s1s, s2s]  
                        policy_vectors = list(zip(s1s.tolist(), s2s.tolist(), probs.tolist(), probs2.tolist()))
                        
                        # Get teacher's value for this position
                        teacher_value = lesson['value'][i].item()
                        
                        # Create new data with policy vectors
                        new_data = ActionValuesWithPolicyData(
                            fen=fen,
                            move_values=move_values,
                            policy_vectors=policy_vectors,
                            teacher_value=teacher_value
                        )
                        
                        # Write to output bag file
                        writer.write(encode_action_values_with_policy(new_data))
                    else:
                        # Update progress bar
                        processed_elements += batch_size_actual
                        pbar.update(batch_size_actual)  # Update by batch, not elements
                        continue
                    break
            
            print(f"Completed processing {input_path}. Processed {processed_elements} elements.")
    except Exception as e:
        print(f"Error processing file: {e}")
        # If output file was created but processing failed, remove it
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Removed partial output file {output_path}")

def inspect_bag_file(bag_file_path: str, num_samples: int = 10):
    """Inspect the first num_samples entries in a bag file."""
    try:
        source = bagz.BagReader(bag_file_path)
        total_elements = len(source)
        print(f"\nInspecting {bag_file_path} - Total elements: {total_elements}")
        
        for i in range(min(num_samples, total_elements)):
            data = source[i + 44525]
            fen, move_values, policy_vectors, teacher_value = constants.CODERS['action_values_with_policy'].decode(data)
            
            print(f"\nEntry {i+1}:")
            print(f"  FEN: {fen}")
            print(f"  Teacher value: {teacher_value:.4f}")
            print(f"  Move values: {len(move_values)} moves")
            if len(move_values) > 0:
                best_move = max(move_values, key=lambda x: x[1])
                print(f"    Best move: {best_move[0]} with value {best_move[1]:.4f}")
            
            print(f"  Policy vectors: {len(policy_vectors)} positions")
            if len(policy_vectors) > 0:
                top_policy_moves = sorted(policy_vectors, key=lambda x: x[2], reverse=True)[:3]
                print("    Top 3 policy moves:")
                for idx, (s1, s2, prob, prob2) in enumerate(top_policy_moves):
                    print(f"      {idx+1}. From square {s1} to {s2} with policy: {prob:.4f}, opt_policy_split: {prob2:.4f}")
    
    except Exception as e:
        print(f"Error inspecting bag file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Add policy vectors to action_values bag files')
    parser.add_argument('--input-pattern', type=str, required=True, 
                        help='Glob pattern for input bag files (e.g., "../data/new-*.bag")')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed bag files')
    parser.add_argument('--teacher-model', type=str, required=True,
                        help='Path to teacher model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for processing')
    args = parser.parse_args()

    # Get list of input files matching pattern
    input_files = glob.glob(args.input_pattern)
    
    if not input_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load teacher model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = load_teacher_model(args.teacher_model, device)
    
    # Process each input file
    for file_idx, input_path in enumerate(input_files):
        # Create output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, f"policy_{filename}")
        
        # Process the file
        print(f"[{file_idx+1}/{len(input_files)}] Processing file {filename}")
        process_bag_file(
            input_path=input_path,
            output_path=output_path,
            teacher_model=teacher_model,
            batch_size=args.batch_size,
            device=device
        )
    
    print("All files processed successfully.")

if __name__ == "__main__":
    main()
    # inspect_bag_file("./data/policy/policy_new-00012-of-00024.bag", num_samples=10)


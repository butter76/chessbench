#!/usr/bin/env python3
"""Read and display child data from the .bag file in human-readable format.

This script reads the output from generate_child_data.py and displays
the records in a human-readable format, including:
- Chess position (FEN)
- Original LC0 data
- Neural network uncertainty values (U) for each child move

Usage:
    python read_child_data.py [bag_file_path] [--num-records N]
    
Example:
    python read_child_data.py ../data/child_data/child_data.bag --num-records 5
"""

import sys
import os
import argparse
import chess
from typing import List, Tuple

# Add the parent directory to sys.path so we can import from searchless_chess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchless_chess.src import bagz
from searchless_chess.src.constants import CODERS


def format_policy(policy: List[Tuple[str, float]], max_moves: int = 5) -> str:
    """Format the policy moves for display."""
    if not policy:
        return "No policy moves"
    
    # Sort by probability (descending)
    sorted_policy = sorted(policy, key=lambda x: x[1], reverse=True)
    
    lines = []
    for i, (move, prob) in enumerate(sorted_policy[:max_moves]):
        lines.append(f"    {move}: {prob:.4f}")
    
    if len(sorted_policy) > max_moves:
        lines.append(f"    ... ({len(sorted_policy) - max_moves} more moves)")
    
    return "\n".join(lines)


def format_U_values(U_values: List[Tuple[str, float]], max_moves: int = 10) -> str:
    """Format the uncertainty values for display."""
    if not U_values:
        return "No U values"
    
    # Sort by U value (descending) to show highest uncertainty first
    sorted_U = sorted(U_values, key=lambda x: x[1], reverse=True)
    
    lines = []
    for i, (move, U) in enumerate(sorted_U[:max_moves]):
        lines.append(f"    {move}: {U:.6f}")
    
    if len(sorted_U) > max_moves:
        lines.append(f"    ... ({len(sorted_U) - max_moves} more moves)")
    
    return "\n".join(lines)


def format_position_info(fen: str) -> str:
    """Format additional chess position information."""
    try:
        board = chess.Board(fen)
        lines = []
        lines.append(f"  Side to move: {'White' if board.turn else 'Black'}")
        lines.append(f"  Legal moves: {len(list(board.legal_moves))}")
        lines.append(f"  In check: {board.is_check()}")
        
        # Show the board position
        lines.append("  Board:")
        board_str = str(board)
        for line in board_str.split('\n'):
            lines.append(f"    {line}")
            
        return "\n".join(lines)
    except Exception as e:
        return f"  Error parsing position: {e}"


def print_record(record_idx: int, record_data: tuple, verbose: bool = False) -> None:
    """Print a single record in human-readable format."""
    (fen, policy, result, root_q, root_d, played_q, played_d, 
     plies_left, move, U_values) = record_data
    
    print(f"\n{'='*80}")
    print(f"RECORD #{record_idx + 1}")
    print(f"{'='*80}")
    
    print(f"Position (FEN): {fen}")
    
    if verbose:
        print(format_position_info(fen))
    
    print(f"\nOriginal LC0 Data:")
    print(f"  Result: {result:.4f}")
    print(f"  Root Q: {root_q:.4f}, Root D: {root_d:.4f}")
    print(f"  Played Q: {played_q:.4f}, Played D: {played_d:.4f}")
    print(f"  Plies left: {plies_left}")
    print(f"  Move played: {move}")
    
    print(f"\nPolicy (top moves):")
    print(format_policy(policy))
    
    print(f"\nUncertainty Values (U) for child positions:")
    print(format_U_values(U_values))
    
    # Summary statistics
    if U_values:
        U_vals = [u for _, u in U_values if u > 0]  # Filter out terminal positions (U=0)
        if U_vals:
            print(f"\nU-value statistics for {len(U_vals)} non-terminal children:")
            print(f"  Min: {min(U_vals):.6f}")
            print(f"  Max: {max(U_vals):.6f}")
            print(f"  Mean: {sum(U_vals)/len(U_vals):.6f}")
        
        terminal_count = sum(1 for _, u in U_values if u == 0)
        if terminal_count > 0:
            print(f"  Terminal positions: {terminal_count}")


def main():
    parser = argparse.ArgumentParser(description='Read and display child data from .bag file')
    parser.add_argument('bag_file', nargs='?', 
                       default='../data/child_data/child_data.bag',
                       help='Path to the .bag file (default: ../data/child_data/child_data.bag)')
    parser.add_argument('--num-records', '-n', type=int, default=3,
                       help='Number of records to display (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show additional position information including board display')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Starting record index (default: 0)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.bag_file):
        print(f"Error: File '{args.bag_file}' not found.")
        print("Make sure you've run generate_child_data.py first to create the output file.")
        sys.exit(1)
    
    print(f"Reading child data from: {args.bag_file}")
    
    try:
        # Open the bag file
        reader = bagz.BagReader(args.bag_file)
        total_records = len(reader)
        
        print(f"Total records in file: {total_records}")
        
        if total_records == 0:
            print("No records found in the file.")
            return
        
        # Validate start index
        if args.start_from >= total_records:
            print(f"Error: start-from index {args.start_from} is >= total records {total_records}")
            return
        
        # Determine how many records to read
        end_idx = min(args.start_from + args.num_records, total_records)
        
        print(f"Displaying records {args.start_from} to {end_idx - 1}:")
        
        # Read and display records
        for i in range(args.start_from, end_idx):
            try:
                # Read the raw bytes
                raw_data = reader[i]
                
                # Decode using the appropriate coder
                decoded_data = CODERS['lc0_data_with_U'].decode(raw_data)
                
                # Print the record
                print_record(i, decoded_data, verbose=args.verbose)
                
            except Exception as e:
                print(f"\nError reading record {i}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"Displayed {end_idx - args.start_from} records out of {total_records} total")
        
    except Exception as e:
        print(f"Error reading bag file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
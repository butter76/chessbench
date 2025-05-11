#!/usr/bin/env python3
"""
LC0 Training Data Conversion Example

This script demonstrates how to use the LC0 training data conversion utilities
to process LC0 training data files in V6 format.
"""

import os
import argparse
import logging
from typing import List, Dict, Any
import chess  # For validating legal moves
import struct  # For unpacking 64-bit integers

from searchless_chess.convert.lc0_coder import LC0TrainingDataCoder
from searchless_chess.convert.chunk_parser import read_chunks, ChunkReader, get_chunk_files
from searchless_chess.convert.utils import planes_to_fen, get_uci_move_from_idx, get_input_format_name
from searchless_chess.src.constants import LC0DataRecord


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def is_potential_castling(uci_move, side_to_move):
    """Check if a move could be a castling move based on its UCI representation.
    
    Args:
        uci_move: Move in UCI format (e.g., 'e1h1')
        side_to_move: 'w' for white, 'b' for black
        
    Returns:
        bool: True if the move could be a castling move
    """
    # LC0 represents castling as the king moving to the rook's square
    
    # For white
    if side_to_move == 'w':
        # King-side castling in LC0: e1h1 (king to rook's square)
        if uci_move == 'e1h1':
            return True
        # Queen-side castling in LC0: e1a1 (king to rook's square)
        if uci_move == 'e1a1':
            return True
        
        # Standard UCI notation as fallback
        if uci_move == 'e1g1' or uci_move == 'e1c1':
            return True
    # For black
    else:
        # King-side castling in LC0: e8h8 (king to rook's square)
        if uci_move == 'e8h8':
            return True
        # Queen-side castling in LC0: e8a8 (king to rook's square)
        if uci_move == 'e8a8':
            return True
        
        # Standard UCI notation as fallback
        if uci_move == 'e8g8' or uci_move == 'e8c8':
            return True
    
    # More general check for Chess960 castling
    # King starts on e file (in standard chess) and moves to a or h file
    if side_to_move == 'w' and uci_move.startswith('e1'):
        dst_file = uci_move[2]
        if dst_file in 'ah':  # King moves to a rook's square
            return True
    if side_to_move == 'b' and uci_move.startswith('e8'):
        dst_file = uci_move[2]
        if dst_file in 'ah':  # King moves to a rook's square
            return True
    
    # King moves two or more squares horizontally (possible Chess960 castling)
    if len(uci_move) == 4:
        from_file = uci_move[0]
        from_rank = uci_move[1]
        to_file = uci_move[2]
        to_rank = uci_move[3]
        
        # King moving on the same rank
        if from_rank == to_rank:
            # Verify it's the correct rank for the side to move
            if (side_to_move == 'w' and from_rank == '1') or (side_to_move == 'b' and from_rank == '8'):
                # Check if king is moving two or more squares horizontally
                file_diff = abs(ord(from_file) - ord(to_file))
                if file_diff >= 2:
                    return True
    
    return False


def is_promotion_move(uci_move, side_to_move):
    """Check if a move could be a promotion move, including LC0 knight promotions without suffix.
    
    Args:
        uci_move: Move in UCI format (e.g., 'a7a8q')
        side_to_move: 'w' for white, 'b' for black
        
    Returns:
        bool: True if the move could be a promotion move
    """
    # Standard promotion moves end with a promotion piece character
    if len(uci_move) == 5 and uci_move[4] in 'qrbn':
        return True
    
    # Check for LC0-style knight promotions (missing 'n' suffix)
    # For white: pawn move from rank 7 to rank 8
    if side_to_move == 'w' and len(uci_move) == 4:
        if uci_move[1] == '7' and uci_move[3] == '8':
            # Verify that the file hasn't changed more than one space (diagonal capture)
            file_diff = abs(ord(uci_move[0]) - ord(uci_move[2]))
            return file_diff <= 1
    
    # For black: pawn move from rank 2 to rank 1
    if side_to_move == 'b' and len(uci_move) == 4:
        if uci_move[1] == '2' and uci_move[3] == '1':
            # Verify that the file hasn't changed more than one space (diagonal capture)
            file_diff = abs(ord(uci_move[0]) - ord(uci_move[2]))
            return file_diff <= 1
    
    return False


def check_knight_promotion(lc0_move, python_chess_moves):
    """Check if an LC0 move without 'n' suffix maps to a knight promotion in python-chess.
    
    Args:
        lc0_move: UCI move from LC0
        python_chess_moves: List of legal moves from python-chess
        
    Returns:
        bool: True if adding 'n' to lc0_move is in python_chess_moves
    """
    if len(lc0_move) == 4:  # Standard move length without promotion indicator
        knight_promotion = lc0_move + 'n'
        return knight_promotion in python_chess_moves
    return False


def uci_to_lc0_notation(uci_move, board):
    """Convert standard UCI notation to LC0 notation.
    
    Args:
        uci_move: Move in standard UCI format (e.g., 'e1g1' for castling)
        board: python-chess Board object for context
    
    Returns:
        str: Move in LC0 notation
    """
    # Check if it's a valid UCI move
    move = chess.Move.from_uci(uci_move)
    
    # Handle castling
    if board.is_castling(move):
        from_square = move.from_square
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        
        # Determine rook square based on castling type
        to_file = chess.square_file(move.to_square)
        
        # Check if kingside castling (to g-file, which is index 6)
        if to_file == 6:  # G file
            if from_rank == 0:  # First rank (white)
                return chess.square_name(from_square) + 'h1'
            else:  # Eighth rank (black)
                return chess.square_name(from_square) + 'h8'
        else:  # Queenside castling (to c-file, which is index 2)
            if from_rank == 0:  # First rank (white)
                return chess.square_name(from_square) + 'a1'
            else:  # Eighth rank (black)
                return chess.square_name(from_square) + 'a8'
    
    # Handle knight promotions (remove the 'n' suffix)
    if move.promotion == chess.KNIGHT:
        return uci_move[:-1]  # Remove the last character ('n')
    
    # If neither of the above, return the move as is
    return uci_move


def lc0_to_uci_notation(lc0_move, board):
    """Convert LC0 notation to standard UCI notation.
    
    Args:
        lc0_move: Move in LC0 format (e.g., 'e1h1' for kingside castling)
        board: python-chess Board object for context
    
    Returns:
        str: Move in standard UCI notation, or None if not convertible
    """
    # Handle potential knight promotion (add 'n' suffix)
    if len(lc0_move) == 4:
        # Check if it's a pawn move to the last rank
        from_file, from_rank = lc0_move[0], int(lc0_move[1])
        to_file, to_rank = lc0_move[2], int(lc0_move[3])
        
        is_white = board.turn == chess.WHITE
        is_to_last_rank = (is_white and to_rank == 8) or (not is_white and to_rank == 1)
        
        if is_to_last_rank:
            # Get the piece at from_square
            from_square = chess.parse_square(lc0_move[:2])
            piece = board.piece_at(from_square)
            
            if piece and piece.piece_type == chess.PAWN:
                # It's a pawn to last rank - try with knight promotion
                knight_promotion = lc0_move + 'n'
                move = chess.Move.from_uci(knight_promotion)
                if move in board.legal_moves:
                    return knight_promotion
    
    # Handle castling
    if (lc0_move in ['e1h1', 'e1a1', 'e8h8', 'e8a8'] or
        (len(lc0_move) == 4 and (lc0_move[0] + lc0_move[2]) in ['ea', 'eh'])):
        
        # King's starting position
        from_square = chess.parse_square(lc0_move[:2])
        
        # Check piece at from_square
        piece = board.piece_at(from_square)
        if not piece or piece.piece_type != chess.KING:
            return lc0_move  # Not a king, return as is
        
        # Determine standard UCI castling move
        if lc0_move[2] == 'h':  # Kingside castling
            if lc0_move[3] == '1':  # White
                standard_uci = 'e1g1'
            else:  # Black
                standard_uci = 'e8g8'
            
            # Verify it's a legal move
            move = chess.Move.from_uci(standard_uci)
            if move in board.legal_moves:
                return standard_uci
            else:
                raise ValueError(f"Invalid move: {standard_uci}")
        
        elif lc0_move[2] == 'a':  # Queenside castling
            if lc0_move[3] == '1':  # White
                standard_uci = 'e1c1'
            else:  # Black
                standard_uci = 'e8c8'
            
            # Verify it's a legal move
            move = chess.Move.from_uci(standard_uci)
            if move in board.legal_moves:
                return standard_uci
            else:
                raise ValueError(f"Invalid move: {standard_uci}")
    
    # Check if the move as-is is valid in UCI notation
    move = chess.Move.from_uci(lc0_move)
    if move in board.legal_moves:
        return lc0_move
    
    # This could be an en passant move, so silently return the move as is
    return lc0_move


def find_en_passant_move(board, lc0_moves, python_chess_moves):
    """Find the en passant square by analyzing move discrepancies.
    
    Args:
        board: python-chess Board object
        lc0_moves: List of moves in LC0 notation
        python_chess_moves: List of moves in UCI notation from python-chess
    
    Returns:
        str: En passant move in UCI notation (e.g., 'e5f6') or None if not found
    """
    # Convert all python-chess moves to LC0 notation
    python_chess_in_lc0 = [uci_to_lc0_notation(move, board) for move in python_chess_moves]
    
    # Find moves that are in python-chess but not in LC0's legal move list
    missing_in_lc0 = set(lc0_moves) - set(python_chess_in_lc0)
    
    # Check each missing move to see if it's an en passant capture
    if missing_in_lc0:
        for move_str in missing_in_lc0:
            if len(move_str) != 4:
                continue
            # Convert to a chess.Move object
            move = chess.Move.from_uci(move_str)
            
            # Get the starting square
            from_square = move.from_square
            
            # Check if the piece is a pawn
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Check if the pawn is on the 5th rank (rank 4 in zero-based indexing)
                # For white pawns, 5th rank is rank 4 (squares 32-39)
                # For black pawns, 5th rank is rank 3 (squares 24-31)
                rank = chess.square_rank(from_square)
                if (piece.color == chess.WHITE and rank == 4) or (piece.color == chess.BLACK and rank == 3):
                    # This could be an en passant capture
                    return move_str
    
    return None


def print_plane_as_grid(plane_int, label=None):
    """Print a plane as an 8x8 grid.
    
    Args:
        plane_int: 64-bit integer representing a bitboard plane
        label: Optional label to print above the grid
    """
    if label:
        print(f"  {label}:")
    
    print("    a b c d e f g h")
    print("    ---------------")
    for rank in range(7, -1, -1):  # 7 to 0 (8th rank to 1st rank)
        row_str = f"  {rank+1}|"
        for file in range(8):  # 0 to 7 (a to h)
            # Calculate bit position using the flipped board representation
            bit_pos = rank * 8 + (7 - file)  # Using flipped files (horizontal mirror)
            bit_value = (plane_int >> bit_pos) & 1
            row_str += "■ " if bit_value else ". "
        print(row_str)
    print()


def print_aux_planes(planes_array, start_idx=104, count=8):
    """Print a range of auxiliary planes as 8x8 grids.
    
    Args:
        planes_array: List of 64-bit integers representing planes
        start_idx: Starting index of auxiliary planes (default 104)
        count: Number of planes to print
    """
    print("  Auxiliary Planes:")
    for i in range(start_idx, min(start_idx + count, len(planes_array))):
        plane_int = planes_array[i]
        # Count the number of bits set (to identify planes with single bits that could be en passant)
        bits_set = bin(plane_int).count('1')
        label = f"Plane {i} - {bits_set} bit(s) set"
        
        # Add special labels for known planes
        if i == 104:
            label += " (Queenside Castling Rooks)"
        elif i == 105:
            label += " (Kingside Castling Rooks)"
        elif i == 106:
            label += " (Possible Side to Move or En Passant)"
        elif i == 107:
            label += " (Rule50 Counter)"
        
        print_plane_as_grid(plane_int, label)

def simple_convert_example(input_path: str):
    """Demonstrate simple usage by reading and decoding a single file.
    
    Args:
        input_path: Path to the chunk file
        
    Returns:
        List of processed records as LC0DataRecord objects
    """
    coder = LC0TrainingDataCoder(support_v7=False)
    last_board = None
    output_records = []
    
    for i, record_bytes in enumerate(read_chunks(input_path)):
        record = coder.decode(record_bytes)

        # Print basic information about the record
        version = record['version']
        input_format = record['input_format']

        if version != 6:
            raise Exception(f"Unsupported version: {version}")
        
        if input_format != 1:
            raise Exception(f"Unsupported input format: {input_format}")
        
        # Get the FEN
        planes = record['planes']
        planes_len = len(planes)

        if planes_len < 832:
            raise Exception(f"Insufficient planes: {planes_len} bytes")

        # Get the probabilities distribution
        probs = record['probabilities']
        
        root_q = record['root_q']
        best_q = record['best_q']
        root_d = record['root_d']
        best_d = record['best_d']
        played_q = record['played_q'] 
        played_d = record['played_d']

        result = record['result']
        rule50_count = record['rule50_count']
        
        castling_us_ooo = record['castling_us_ooo']
        castling_us_oo = record['castling_us_oo']
        castling_them_ooo = record['castling_them_ooo']
        castling_them_oo = record['castling_them_oo']
        plies_left = record['plies_left']
        played_idx = record['played_idx']

        # Convert board planes to FEN
        fen = planes_to_fen(
            planes, 
            input_format=input_format,
            castling_us_ooo=castling_us_ooo,
            castling_us_oo=castling_us_oo,
            castling_them_ooo=castling_them_ooo,
            castling_them_oo=castling_them_oo,
            rule50_count=rule50_count
        )

        if i == 0:
            if fen != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0":
                # This is a non-standard FEN, so we skip it
                return []

        board = chess.Board(fen)

        # Get all legal moves in UCI format, except for possibly en passant moves
        python_chess_legal_moves = [move.uci() for move in board.legal_moves]

        # Extract LC0 moves for comparison
        # Find all legal moves (indices where probability is not -1)
        legal_move_indices = [idx for idx, prob in enumerate(probs) if prob != -1]
        
        # Create a dictionary mapping UCI moves to their LC0 indices
        lc0_uci_moves = {get_uci_move_from_idx(idx): idx for idx in legal_move_indices}

        en_passant_move = find_en_passant_move(board, lc0_uci_moves.keys(), python_chess_legal_moves)
        en_passant_sq = None
        if en_passant_move:
            en_passant_sq = en_passant_move[2:4]

        if en_passant_sq is not None:
            fen = " ".join(fen.split(" ")[:3]) + " " + en_passant_sq + " " + " ".join(fen.split(" ")[4:])
            board = chess.Board(fen)
            board.generate_legal_moves()
            python_chess_legal_moves = [move.uci() for move in board.legal_moves]

        played_move = lc0_to_uci_notation(get_uci_move_from_idx(played_idx), board)
        if last_board is not None:
            if last_board.fen() != board.fen():
                raise Exception("Board mismatch")
            
        assert len(python_chess_legal_moves) == len(lc0_uci_moves)

        # Create policy list of (move, probability) tuples
        policy = [(move, probs[lc0_uci_moves[uci_to_lc0_notation(move, board)]]) for move in python_chess_legal_moves]

        for move, prob in policy:
            # Make sure this UCI move interpreted by python-chess maps to a legal move as interpreted by Leela
            assert prob != -1
        
        # Create LC0DataRecord instead of dictionary
        lc0_record = LC0DataRecord(
            fen=board.fen(),
            policy=policy,
            result=result,
            root_q=root_q,
            root_d=root_d,
            played_q=played_q,
            played_d=played_d,
            plies_left=plies_left,
            move=played_move,
        )
        
        output_records.append(lc0_record)
        
        last_board = board
        last_board.push(chess.Move.from_uci(played_move))
        last_board.apply_mirror()
    
    return output_records


def simple_example(input_path: str):
    """Demonstrate simple usage by reading and decoding a single file.
    
    Args:
        input_path: Path to the chunk file
    """
    # Create the coder without V7 support
    coder = LC0TrainingDataCoder(support_v7=False)
    
    # Read and process the first 5 records from the file
    for i, record_bytes in enumerate(read_chunks(input_path)):
        
        # Decode the record
        record = coder.decode(record_bytes)
        
        # Print basic information about the record
        version = record.get('version', 'unknown')
        input_format = record.get('input_format', 'unknown')
        input_format_name = get_input_format_name(input_format)
        root_q = record.get('root_q', 'N/A')
        best_q = record.get('best_q', 'N/A')
        root_d = record.get('root_d', 'N/A')
        best_d = record.get('best_d', 'N/A')
        
        # Try to get played_q and played_d (might not be available in V6)
        played_q = record.get('played_q', 'N/A') 
        played_d = record.get('played_d', 'N/A')
        
        # Additional fields requested
        visits = record.get('visits', 'N/A')
        plies_left = record.get('plies_left', 'N/A')
        played_idx = record.get('played_idx', 'N/A')
        
        # Get castling rights from record
        castling_us_ooo = record.get('castling_us_ooo', None)
        castling_us_oo = record.get('castling_us_oo', None)
        castling_them_ooo = record.get('castling_them_ooo', None)
        castling_them_oo = record.get('castling_them_oo', None)
        
        # Get side to move as 'w' or 'b'
        side_to_move = 'b' if record.get('us_opp', False) else 'w'
        
        # Get rule50 counter
        rule50_count = record.get('rule50_count', None)
        
        # Get board planes information
        planes = record.get('planes', b'')
        planes_len = len(planes)
        
        # Convert board planes to FEN if available
        fen = "N/A"
        if planes_len >= 832:  # At least 104 planes (minimum required)
            fen = planes_to_fen(
                planes, 
                input_format=input_format,
                castling_us_ooo=castling_us_ooo,
                castling_us_oo=castling_us_oo,
                castling_them_ooo=castling_them_ooo,
                castling_them_oo=castling_them_oo,
                side_to_move=side_to_move,
                rule50_count=rule50_count
            )
        
        # Convert played_idx to UCI notation if available
        played_move = "N/A"
        if played_idx != 'N/A':
            played_move = get_uci_move_from_idx(played_idx)
        
        # Get the probabilities distribution
        probs = record.get('probabilities', [])
        
        # Find all legal moves (indices where probability is not -1)
        legal_move_indices = [idx for idx, prob in enumerate(probs) if prob != -1]
        
        # Convert legal move indices to UCI notation
        legal_moves_uci = []
        for idx in legal_move_indices:
            move_uci = get_uci_move_from_idx(idx)
            legal_moves_uci.append((idx, move_uci))
        
        # Use python-chess to validate legal moves
        python_chess_legal_moves = []
        legal_moves_comparison = {}
        
        board = None
        en_passant_move = None
        
        if fen != "N/A" and not fen.startswith("Error:"):
            # Create a chess board from the FEN
            board = chess.Board(fen)
            
            # Get all legal moves in UCI format
            python_chess_legal_moves = [move.uci() for move in board.legal_moves]
            
            # Extract LC0 moves for comparison
            lc0_uci_moves = [uci for _, uci in legal_moves_uci]
            
            # Create bidirectional mappings between UCI and LC0 notation
            lc0_to_standard = {}
            standard_to_lc0 = {}
            
            # Map from LC0 notation to standard UCI
            for lc0_move in lc0_uci_moves:
                std_move = lc0_to_uci_notation(lc0_move, board)
                if std_move != lc0_move:  # Only add if there's a difference
                    lc0_to_standard[lc0_move] = std_move
            
            # Map from standard UCI to LC0 notation
            for std_move in python_chess_legal_moves:
                lc0_move = uci_to_lc0_notation(std_move, board)
                if lc0_move != std_move:  # Only add if there's a difference
                    standard_to_lc0[std_move] = lc0_move
            
            # Check for the en passant square
            en_passant_move = find_en_passant_move(board, lc0_uci_moves, python_chess_legal_moves)
            
            # Transform python-chess moves to LC0 notation for comparison
            python_chess_as_lc0 = [uci_to_lc0_notation(move, board) for move in python_chess_legal_moves]
            
            # Common moves in both notations
            common_moves_lc0 = set(lc0_uci_moves) & set(python_chess_as_lc0)
            
            # Moves only in python-chess (after converting to LC0 notation)
            only_in_python_chess_lc0 = set(python_chess_as_lc0) - set(lc0_uci_moves)
            
            # Moves only in LC0
            only_in_lc0 = set(lc0_uci_moves) - set(python_chess_as_lc0)
            
            # Initialize a dictionary to store matches found for moves exclusive to LC0
            matched_explanations = {}
            
            # Check each move exclusive to LC0
            for move in only_in_lc0:
                if is_potential_castling(move, side_to_move):
                    matched_explanations[move] = "castling"
                elif is_promotion_move(move, side_to_move):
                    if check_knight_promotion(move, python_chess_legal_moves):
                        matched_explanations[move] = "knight promotion"
                elif move == en_passant_move:
                    matched_explanations[move] = "en passant"
            
            # Identify problematic moves (unexplained discrepancies)
            problematic_moves = [move for move in only_in_lc0 if move not in matched_explanations]
            
            legal_moves_comparison = {
                "common_count": len(common_moves_lc0),
                "only_in_python_chess": list(only_in_python_chess_lc0),
                "only_in_lc0": list(only_in_lc0),
                "matched_explanations": matched_explanations,
                "problematic_moves": problematic_moves,
                "lc0_to_standard": lc0_to_standard,
                "standard_to_lc0": standard_to_lc0,
                "en_passant_move": en_passant_move
            }
        
        # Get the first 10 elements of the probability distribution
        first_10_probs = probs[:10] if len(probs) >= 10 else probs
        
        # If we have win/draw/loss probabilities, show them
        if 'wdl' in record:
            wdl = record['wdl']
            wdl_str = f"WDL: {wdl[0]:.2f}/{wdl[1]:.2f}/{wdl[2]:.2f}"
        else:
            wdl_str = "WDL: N/A"
        
        # Print summary
        print(f"Record {i+1}:")
        print(f"  Version: {version}")
        print(f"  Input Format: {input_format} ({input_format_name})")
        print(f"  FEN: {fen}")
        print(f"  Root Q: {root_q}")
        print(f"  Best Q: {best_q}")
        print(f"  Root D: {root_d}")
        print(f"  Best D: {best_d}")
        print(f"  Played Q: {played_q}")
        print(f"  Played D: {played_d}")
        print(f"  Visits: {visits}")
        print(f"  Plies Left: {plies_left}")
        print(f"  Played Idx: {played_idx}")
        print(f"  Played Move (UCI): {played_move}")
        print(f"  {wdl_str}")
        print(f"  First 10 probabilities: {first_10_probs}")
        print(f"  Planes: {planes_len} bytes")
        
        # Extract and print planes
        if planes_len >= 832:  # At least 104 planes
            planes_array = []
            for j in range(0, len(planes), 8):
                # Convert 8 bytes to a 64-bit integer using little-endian format
                plane = struct.unpack('<Q', planes[j:j+8])[0]
                planes_array.append(plane)
            
            # Print auxiliary planes to analyze en passant information
            print_aux_planes(planes_array)
        
        # Print castling rights info directly from the record
        castling_info = []
        if castling_us_ooo is not None:
            castling_info.append(f"us_ooo: {castling_us_ooo}")
        if castling_us_oo is not None:
            castling_info.append(f"us_oo: {castling_us_oo}")
        if castling_them_ooo is not None:
            castling_info.append(f"them_ooo: {castling_them_ooo}")
        if castling_them_oo is not None:
            castling_info.append(f"them_oo: {castling_them_oo}")
        
        if castling_info:
            print(f"  Castling: {', '.join(castling_info)}")
        
        # Print rule50 counter directly from the record
        if rule50_count is not None:
            print(f"  Rule50 count: {rule50_count}")
        
        # Print legal move indices
        print(f"  Legal move count from LC0: {len(legal_move_indices)}")
        if legal_moves_uci:
            # Format for better readability
            legal_moves_str = "\n    ".join([", ".join([f"{idx}:{uci}" for idx, uci in legal_moves_uci[i:i+5]]) 
                                           for i in range(0, len(legal_moves_uci), 5)])
            print(f"  Legal moves (idx:UCI) from LC0:\n    {legal_moves_str}")
        
        if en_passant_move:
            print(f"  EN PASSANT MOVE: {en_passant_move}")

        # Print legal moves comparison with python-chess
        if python_chess_legal_moves:
            print(f"  Legal move count from python-chess: {len(python_chess_legal_moves)}")
            
            if legal_moves_comparison:
                print(f"  Moves in both LC0 and python-chess (after notation conversion): {legal_moves_comparison['common_count']}")
                
                if legal_moves_comparison['only_in_python_chess']:
                    print(f"  Moves only in python-chess (in LC0 notation): {', '.join(legal_moves_comparison['only_in_python_chess'])}")
                
                if legal_moves_comparison['only_in_lc0']:
                    print(f"  Moves only in LC0: {', '.join(legal_moves_comparison['only_in_lc0'])}")
                
                # Print notation conversion tables if they exist
                if legal_moves_comparison['lc0_to_standard']:
                    print("  LC0 to standard UCI notation mapping:")
                    for lc0, std in legal_moves_comparison['lc0_to_standard'].items():
                        print(f"    {lc0} → {std}")
                
                if legal_moves_comparison['standard_to_lc0']:
                    print("  Standard UCI to LC0 notation mapping:")
                    for std, lc0 in legal_moves_comparison['standard_to_lc0'].items():
                        print(f"    {std} → {lc0}")
                
                # Print matched explanations for moves exclusive to LC0
                if legal_moves_comparison['matched_explanations']:
                    print("  Explanation for moves only in LC0:")
                    for move, explanation in legal_moves_comparison['matched_explanations'].items():
                        print(f"    {move}: {explanation}")
                
                # Print problematic moves
                if legal_moves_comparison['problematic_moves']:
                    print(f"  Unexplained moves in LC0: {', '.join(legal_moves_comparison['problematic_moves'])}")
                    raise Exception("Unexplained moves in LC0")
        
        # Check for V7/V7B specific fields
        if any(k.startswith('v7_') for k in record):
            v7_fields = [k for k in record if k.startswith('v7_')]
            print(f"  V7 fields: {len(v7_fields)} fields present")
            
            # Example of accessing a V7 field
            if 'v7_pol_kld' in record:
                print(f"  Policy KLD: {record['v7_pol_kld']}")
        
        print()


def batch_processing_example(input_dir: str, pattern: str = "*.gz", max_files: int = 3):
    """Demonstrate batch processing of multiple files.
    
    Args:
        input_dir: Directory containing chunk files
        pattern: File pattern to match
        max_files: Maximum number of files to process
    """
    # Get list of files
    files = get_chunk_files(input_dir, pattern, recursive=True)
    if not files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return
    
    # Limit number of files
    files = files[:max_files]
    print(f"Processing {len(files)} files:")
    for f in files:
        print(f"  {f}")
    print()
    
    # Create a ChunkReader for batch processing
    reader = ChunkReader(files, sample_rate=0.01)  # Sample 1% of positions
    
    # Count records and gather statistics
    total_records = 0
    v6_count = 0
    v7_count = 0
    v7b_count = 0
    
    coder = LC0TrainingDataCoder(support_v7=False)
    
    for record_bytes in reader:
        # Track total count
        total_records += 1
        
        # Determine version
        version_bytes = record_bytes[:4]
        if version_bytes == coder.V6_VERSION:
            v6_count += 1
        elif version_bytes == coder.V7_VERSION:
            v7_count += 1
        elif version_bytes == coder.V7B_VERSION:
            v7b_count += 1
        
        # Only process a limited number of records
        if total_records >= 1000:
            break
    
    # Print statistics
    print(f"Processed {total_records} records:")
    print(f"  V6: {v6_count} records ({v6_count/total_records*100:.1f}%)")
    print(f"  V7: {v7_count} records ({v7_count/total_records*100:.1f}%)")
    print(f"  V7B: {v7b_count} records ({v7b_count/total_records*100:.1f}%)")


def encode_example(input_path: str):
    """Demonstrate encoding functionality by reading, modifying, and re-encoding a record.
    
    Args:
        input_path: Path to the chunk file
    """
    # Create the coder
    coder = LC0TrainingDataCoder(support_v7=False)
    
    # Read a single record
    for record_bytes in read_chunks(input_path):
        # Decode the record
        record = coder.decode(record_bytes)
        
        # Print original information
        print("Original record:")
        print(f"  Version: {record.get('version')}")
        print(f"  Best Q: {record.get('best_q')}")
        
        # Modify the record
        record['best_q'] = 0.0  # Change evaluation to neutral
        
        # Re-encode the record
        encoded = coder.encode(record)
        
        # Decode again to verify
        decoded = coder.decode(encoded)
        
        # Print modified information
        print("Modified record:")
        print(f"  Version: {decoded.get('version')}")
        print(f"  Best Q: {decoded.get('best_q')}")
        
        # Verify size is the same
        print(f"Original size: {len(record_bytes)} bytes")
        print(f"Encoded size: {len(encoded)} bytes")
        
        break  # Only process one record


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LC0 Training Data Conversion Example')
    
    parser.add_argument('--input', type=str, help='Input chunk file')
    parser.add_argument('--input_dir', type=str, help='Directory containing input chunk files')
    parser.add_argument('--pattern', type=str, default='*.gz', 
                        help='File pattern for input files')
    parser.add_argument('--example', type=str, choices=['simple', 'batch', 'encode'], 
                        default='simple', help='Example to run')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()
    
    if args.example == 'simple':
        if not args.input:
            print("Error: --input is required for simple example")
            return
        simple_convert_example(args.input)
    
    elif args.example == 'batch':
        if not args.input_dir:
            print("Error: --input_dir is required for batch example")
            return
        batch_processing_example(args.input_dir, args.pattern)
    
    elif args.example == 'encode':
        if not args.input:
            print("Error: --input is required for encode example")
            return
        encode_example(args.input)


if __name__ == "__main__":
    main() 
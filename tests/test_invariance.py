import numpy as np
import chess

from searchless_chess.src.tokenizer import tokenize
from searchless_chess.src.utils import move_to_indices


def mirror_and_swap_fen(fen: str) -> str:
    board = chess.Board(fen)
    mirrored = board.mirror()
    return mirrored.fen()


def collect_move_index_pairs(board: chess.Board) -> set[tuple[int, int]]:
    flip = board.turn == chess.BLACK
    index_pairs: set[tuple[int, int]] = set()
    for move in board.legal_moves:
        s1, s2 = move_to_indices(move, flip)
        index_pairs.add((s1, s2))
    return index_pairs


def generate_positions() -> list[str]:
    fens: list[str] = []

    # 1) Starting position
    start = chess.Board()
    fens.append(start.fen())

    # 2) Position with castling rights affected (white short castle available)
    b2 = chess.Board()
    for uci in [
        "e2e4", "c7c5", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1"
    ]:
        b2.push_uci(uci)
    fens.append(b2.fen())

    # 3) Position with en passant available
    b3 = chess.Board()
    for uci in [
        "e2e4", "h7h6", "e4e5", "d7d5"
    ]:
        b3.push_uci(uci)
    # White to move with en passant possible on d6
    fens.append(b3.fen())

    # 4) Position with promotions in the future (advanced pawns) - still a legal midgame position
    b4 = chess.Board()
    for uci in [
        "a2a4", "h7h5", "a4a5", "h5h4", "a5a6", "h4h3", "a6b7", "h3g2"
    ]:
        b4.push_uci(uci)
    fens.append(b4.fen())

    # 5) Pawn can both promote directly or via a capture (white to move)
    # Board:
    # 8: ....k..n
    # 7: ......P.
    # 1: ....K...
    fens.append("4k2n/6P1/8/8/8/8/8/4K3 w - - 0 1")

    # 6) Two pawns can promote (white to move)
    # 8: ....k...
    # 7: P.....P.
    # 1: ....K...
    fens.append("4k3/P6P/8/8/8/8/8/4K3 w - - 0 1")


    fens.append("rnbqbbnr/PPPPPPPP/8/8/8/8/8/K6k w - - 0 1")

    return fens


def test_tokenizer_mirror_color_swap_invariance():
    for fen in generate_positions():
        tokens_original = tokenize(fen)
        tokens_mirrored = tokenize(mirror_and_swap_fen(fen))
        assert tokens_original.shape == tokens_mirrored.shape
        assert np.array_equal(tokens_original, tokens_mirrored)


def test_move_to_indices_mirror_color_swap_invariance():
    for fen in generate_positions():
        board = chess.Board(fen)
        board_m = chess.Board(mirror_and_swap_fen(fen))

        indices_original = collect_move_index_pairs(board)
        indices_mirrored = collect_move_index_pairs(board_m)

        assert indices_original == indices_mirrored



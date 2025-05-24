import chess
from searchless_chess.src.utils import move_to_indices
import torch
import numpy as np
def get_policy(board: chess.Board, output: torch.Tensor):
    if len(list(board.legal_moves)) == 0:
        return [], {}
    is_legal = np.zeros((68, 68), dtype=bool)
    for move in board.legal_moves:
        s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
        is_legal[s1, s2] = True
    output[~is_legal] = float('-inf')

    flat_output = output.reshape(-1)
    exp_output = np.exp(flat_output - np.max(flat_output))  # Subtract max for numerical stability
    softmax_output = exp_output / exp_output.sum()
    output = softmax_output.reshape(68, 68)

    result = []
    policy_map = {}
    for move in board.legal_moves:
        s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
        policy = output[s1, s2].item()
        result.append((move, policy))
        policy_map[move] = policy
    # Sort by policy in descending order
    result.sort(key=lambda x: x[1], reverse=True)
    return result, policy_map

def reduced_fen(board: chess.Board) -> str:
    return ' '.join(board.fen().split(' ')[:4])
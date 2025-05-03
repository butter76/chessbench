import chess
from searchless_chess.src.utils import move_to_indices
import torch

def get_policy(board: chess.Board, output: torch.Tensor):
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
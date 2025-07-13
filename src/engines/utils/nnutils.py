import chess
from searchless_chess.src.utils import move_to_indices
import torch
import numpy as np
def get_policy(board: chess.Board, output: torch.Tensor, U: torch.Tensor, Q: torch.Tensor, D: torch.Tensor):
    if len(list(board.legal_moves)) == 0:
        return [], {}
    is_legal = np.zeros((68, 68), dtype=bool)
    for move in board.legal_moves:
        s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
        assert is_legal[s1, s2] == False, f'{move} is already in the legal actions, board:{board.fen()}'
        is_legal[s1, s2] = True
    output[~is_legal] = float('-inf')

    flat_output = output.reshape(-1)
    exp_output = np.exp(flat_output - np.max(flat_output))  # Subtract max for numerical stability
    softmax_output = exp_output / exp_output.sum()
    
    # Compute perplexity
    # Perplexity = exp(-sum(p * log(p))) where p are the probabilities
    log_probs = np.log(softmax_output + 1e-12)  # Add small epsilon to avoid log(0)
    entropy = -np.sum(softmax_output * log_probs)
    perplexity = np.exp(entropy)
    
    output = softmax_output.reshape(68, 68)

    result = []
    policy_map = {}
    for move in board.legal_moves:
        s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
        policy = output[s1, s2].item()
        U_val = 1 / (1 + np.exp(-U[s1, s2].item()))
        Q_val = 1 / (1 + np.exp(-Q[s1, s2].item()))
        D_val = 1 / (1 + np.exp(-D[s1, s2].item()))
        Q_val = (Q_val * 2 - 1)
        result.append((move, policy, {'U': U_val, 'Q': Q_val, 'D': D_val}))
        policy_map[move] = policy
    # Sort by policy in descending order
    result.sort(key=lambda x: x[1], reverse=True)
    return result, policy_map, np.mean(perplexity)

def reduced_fen(board: chess.Board) -> str:
    return ' '.join(board.fen().split(' ')[:4])
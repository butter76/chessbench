from collections.abc import Callable, Sequence
from typing import cast, Union
from enum import Enum, auto

import chess
import chess.engine
import numpy as np

from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src.engines import engine
from searchless_chess.src.models.transformer import ChessTransformer
from searchless_chess.src.engines.utils.node import Node
from searchless_chess.src.utils import move_to_indices
import torch
from torch.amp.autocast_mode import autocast

from searchless_chess.src.engines.utils.nnutils import get_policy
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")

class MoveSelectionStrategy(str, Enum):
    VALUE = "value"
    AVS = "avs"
    AVS2 = "avs2"
    POLICY = "policy"
    OPT_POLICY_SPLIT = "opt_policy_split"
    NEGAMAX = "negamax"
    ALPHA_BETA = "alpha_beta"

class MyTransformerEngine(engine.Engine):
    def __init__(
        self,
        checkpoint_path: str,
        limit: chess.engine.Limit,
        strategy: Union[MoveSelectionStrategy, str] = MoveSelectionStrategy.VALUE,
        search_depth: int = 2,
        search_ordering_strategy: Union[MoveSelectionStrategy, str, None] = MoveSelectionStrategy.AVS,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._limit = limit
        self.search_depth = search_depth
        
        # Convert string to enum if needed
        if isinstance(strategy, str):
            strategy = MoveSelectionStrategy(strategy)
        self.strategy = strategy

        # Convert ordering strategy string to enum if needed
        if isinstance(search_ordering_strategy, str):
            search_ordering_strategy = MoveSelectionStrategy(search_ordering_strategy)
        self.search_ordering_strategy = search_ordering_strategy

        checkpoint = torch.load(checkpoint_path)
        model_config =  checkpoint['model_config']

        # Create model that matches the checkpoint
        self.model = ChessTransformer(
            config=model_config,
        ).to(self.device)

        if checkpoint['compiled']:
            self.model = cast(ChessTransformer, torch.compile(self.model))

        self.model.load_state_dict(checkpoint['model'])

    def _get_ordered_moves(self, board: chess.Board, ordering_strategy: MoveSelectionStrategy | None) -> list[chess.Move]:
        """
        Gets legal moves ordered by the specified strategy head's output,
        or by a default heuristic (checkmates/captures first) if strategy is None.
        """

        # Use the specified policy/AVS head for ordering if provided
        assert ordering_strategy and ordering_strategy in [
            MoveSelectionStrategy.AVS, MoveSelectionStrategy.AVS2,
            MoveSelectionStrategy.POLICY, MoveSelectionStrategy.OPT_POLICY_SPLIT]
        
        output_tensor = self.analyse_shallow(board)[ordering_strategy.value][0, :, :].clone()
        move_scores = []
        
        for move in board.legal_moves:
            s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
            score = output_tensor[s1, s2].item()
            move_scores.append(score)
            
        # Sort moves based on scores (descending)
        sorted_moves = [move for _, move in sorted(zip(move_scores, board.legal_moves), key=lambda pair: pair[0], reverse=True)]
        return sorted_moves

    def analyse_shallow(self, board: chess.Board) -> engine.AnalysisResult:
        x = np.array([tokenizer.tokenize(board.fen())])
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        return output

    def analyse(self, board: chess.Board) -> engine.AnalysisResult:
        x = []
        for move in board.legal_moves:
            board.push(move)
            x.append(tokenizer.tokenize(board.fen()))
            board.pop()
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        return output
    
    def batch_analyze(self, boards: list[tuple[Node, chess.Move]], construct_node, postprocess_child):
        # Input is a list of [parent_node, move], output is evaluation of the move added to the parent node
        # We need to tokenize the boards and moves, and then pass them to the model
        N = len(boards)
        x = []
        child_boards = []
        for parent_node, move in boards:
            child_board = parent_node.board.copy().push(move)
            x.append(tokenizer.tokenize(child_board.fen()))
            child_boards.append(child_board)
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        values = output['value'][:, 0] * 2.0 - 1.0 # Move from [0, 1] to [-1, 1]
        is_legal = output['legal'] > 0 # Rather than manually checking if the move is legal, just use the model's prediction of legality
        policies = output['policy']
        policies[~is_legal] = float('-inf')
        policies = torch.nn.functional.softmax(policies.view(N, -1), dim=-1).view(N, is_legal.shape[1], -1)

        values = values.cpu().numpy()
        policies = policies.cpu().numpy()
        
        for i, (parent_node, move) in enumerate(boards):
            child_board = child_boards[i]
            value = values[i]
            policy = get_policy(child_board, policies[i])
            # Construct the child node, which is likely a subclass of Node
            child = construct_node(child_board, parent=parent_node, value=value, policy=policy)
            parent_node.add_child(child)
            # Postprocess the child node, which is likely a subclass of Node
            postprocess_child(child)
        return

    def _static_evaluate(self, board: chess.Board) -> float:
        """
        Evaluates a position using the shallow analysis model.
        Returns score relative to the current player [-1 (loss), 0 (draw), 1 (win)].
        """
        x = np.array([tokenizer.tokenize(board.fen())])
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        # Assuming model value v (0-1) is score for the current player
        # (1 = current player wins, 0 = current player loses)
        model_val = output['value'][0, 0].item() 
        # Convert to score relative to current player: score = 2 * model_val - 1
        # 1 -> 1, 0.5 -> 0, 0 -> -1
        score = 2.0 * model_val - 1.0
        return score

    def negamax(self, board: chess.Board, depth: int) -> tuple[float, chess.Move | None]:
        """
        Performs plain negamax search with optional move ordering.
        Returns a tuple of (score relative to the current player to move, best move found).
        """
        if board.is_checkmate():
            # Player whose turn it is got checkmated. Worst score.
            return -1.0, None
        if board.is_stalemate() or board.is_insufficient_material():
            # Draw
            return 0.0, None
            
        if depth == 0:
            # At leaf node, use the static evaluator
            return self._static_evaluate(board), None

        max_eval = -float('inf') # Best score found so far for the current player
        best_move = None

        ordered_moves = self._get_ordered_moves(board, self.search_ordering_strategy)

        for move in ordered_moves:
            board.push(move)
            # Score returned by recursive call is relative to the next player. Negate it.
            score, _ = self.negamax(board, depth - 1)
            score = -score
            board.pop()
            if score > max_eval:
                max_eval = score
                best_move = move
            
        return max_eval, best_move

    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> tuple[float, chess.Move | None]:
        """
        Performs alpha-beta search (negamax variant) with move ordering.
        Returns score relative to the current player and the best move found.
        alpha: Lower bound (best score maximizing player can guarantee).
        beta: Upper bound (best score minimizing player can guarantee).
        """
        # Check for game over conditions first
        if board.is_checkmate():
            # Player whose turn it is got checkmated. Worst score.
            return -1.0, None
        if board.is_stalemate() or board.is_insufficient_material():
            # Draw
            return 0.0, None
            
        if depth == 0:
            # At leaf node, use the static evaluator
            return self._static_evaluate(board), None

        max_eval = -float('inf') # Represents the best score found *so far* for the maximizing player (alpha) at this node
        best_move = None

        ordered_moves = self._get_ordered_moves(board, self.search_ordering_strategy)

        for move in ordered_moves:
            board.push(move)
            # Recursive call: Negate and swap alpha/beta bounds
            score, _ = self.alpha_beta(board, depth - 1, -beta, -alpha)
            score = -score # Negate score back to current player's perspective
            board.pop()

            if score > max_eval:
                max_eval = score
                best_move = move # Update best move if this path is better

            # Update alpha (the lower bound for the current maximizing player)
            alpha = max(alpha, max_eval)

            # Pruning: If the current maximizing player can guarantee a score >= beta,
            # the minimizing parent node (which provided beta) will not choose this path.
            if alpha >= beta:
                break # Beta cut-off

        return max_eval, best_move

    def play(self, board: chess.Board) -> chess.Move:
        self.model.eval()

        if self.strategy == MoveSelectionStrategy.VALUE:
            # print(board.fen())
            value = self.analyse(board)['value']
            value = value[:, 0].clone()
            # print(board.fen())
            for i, (move, av) in enumerate(zip(board.legal_moves, value)):
                # print(move, av.item())
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if board.is_stalemate():
                    value[i] = 0.5
                board.pop()
            best_ix = cast(int, torch.argmin(value).item())
            best_move = list(board.legal_moves)[best_ix]
            best_value = value[best_ix].item()
            # print(f"Best Move: {best_move} with value {best_value}")
        elif self.strategy == MoveSelectionStrategy.AVS or self.strategy == MoveSelectionStrategy.AVS2:
            move_values = []
            avs = self.analyse_shallow(board)[self.strategy][0, :, :].clone()
            for (i, move) in enumerate(board.legal_moves):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if board.is_stalemate() or board.is_insufficient_material():
                    best_res = 0.5
                    board.pop()
                else:
                    board.pop()
                    s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
                    best_res = avs[s1, s2].item()
                move_values.append((best_res, i))
            (best_value, best_idx) = max(move_values)
            best_move = list(board.legal_moves)[best_idx]
        elif self.strategy == MoveSelectionStrategy.POLICY or \
             self.strategy == MoveSelectionStrategy.OPT_POLICY_SPLIT:
            move_values = []
            policy = self.analyse_shallow(board)[self.strategy][0, :, :].clone()
            for (i, move) in enumerate(board.legal_moves):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                board.pop()
                
                s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
                best_res = policy[s1, s2].item()

                move_values.append((best_res, i))
            (best_value, best_idx) = max(move_values)
            best_move = list(board.legal_moves)[best_idx]
        elif self.strategy == MoveSelectionStrategy.NEGAMAX:
            best_value, best_move = self.negamax(board, self.search_depth)

        elif self.strategy == MoveSelectionStrategy.ALPHA_BETA:
            best_value, best_move = self.alpha_beta(board, self.search_depth, -float('inf'), float('inf'))

        else:
             raise ValueError(f"Unknown strategy: {self.strategy}")

        return best_move




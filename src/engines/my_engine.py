from collections.abc import Callable, Sequence
from typing import cast, Union
from enum import Enum, auto

import chess
import chess.engine
import numpy as np

from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine
from searchless_chess.src.models.transformer import ChessTransformer

import torch
from torch.amp.autocast_mode import autocast
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")

def _parse_square(square: str):
  return chess.square_mirror(chess.parse_square(square))

class MoveSelectionStrategy(str, Enum):
    VALUE = "value"
    AVS = "avs"
    AVS2 = "avs2"
    POLICY = "policy"
    POLICY_SPLIT = "policy_split"
    OPT_POLICY_SPLIT = "opt_policy_split"

class MyTransformerEngine(engine.Engine):
    def __init__(
        self,
        checkpoint_path: str,
        limit: chess.engine.Limit,
        strategy: Union[MoveSelectionStrategy, str] = MoveSelectionStrategy.VALUE,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._limit = limit
        
        # Convert string to enum if needed
        if isinstance(strategy, str):
            strategy = MoveSelectionStrategy(strategy)
        self.strategy = strategy

        checkpoint = torch.load(checkpoint_path)
        model_config =  checkpoint['model_config']

        # Create model that matches the checkpoint
        self.model = ChessTransformer(
            config=model_config,
        ).to(self.device)

        if checkpoint['compiled']:
            self.model = cast(ChessTransformer, torch.compile(self.model))

        self.model.load_state_dict(checkpoint['model'])

    def analyse_shallow(self, board: chess.Board) -> engine.AnalysisResult:
        x = np.array([tokenizer.tokenize(board.fen())])
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        return output

    def analyse(self, board: chess.Board) -> engine.AnalysisResult:
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        x = []
        for move in sorted_legal_moves:
            board.push(move)
            x.append(tokenizer.tokenize(board.fen()))
            board.pop()
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        return output        

    def play(self, board: chess.Board) -> chess.Move:
        self.model.eval()
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        if self.strategy == MoveSelectionStrategy.VALUE:
            # print(board.fen())
            value = self.analyse(board)['value']
            value = value[:, 0].clone()
            # print(board.fen())
            for i, (move, av) in enumerate(zip(sorted_legal_moves, value)):
                # print(move, av.item())
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if board.is_stalemate():
                    value[i] = 0.5
                board.pop()
            best_ix = cast(int, torch.argmin(value).item())
            best_move = sorted_legal_moves[best_ix]
            best_value = value[best_ix].item()
            # print(f"Best Move: {best_move} with value {best_value}")
        elif self.strategy == MoveSelectionStrategy.AVS or self.strategy == MoveSelectionStrategy.AVS2:
            move_values = []
            avs = self.analyse_shallow(board)[self.strategy][0, :, :].clone()
            for (i, move) in enumerate(sorted_legal_moves):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if board.is_stalemate():
                    best_res = 0.5
                    board.pop()
                else:
                    board.pop()
                    move = move.uci()
                    s1 = _parse_square(move[0:2])
                    if move[4:] in ['R', 'r']:
                        s2 = 64
                    elif move[4:] in ['B', 'b']:
                        s2 = 65
                    elif move[4:] in ['N', 'n']:
                        s2 = 66
                    else:
                        assert move[4:] in ['Q', 'q', '']
                        s2 = utils._parse_square(move[2:4])
                    best_res = avs[s1, s2].item()
                move_values.append((best_res, i))
            (best_value, best_idx) = max(move_values)
            best_move = sorted_legal_moves[best_idx]
        else:
            move_values = []
            avs = self.analyse_shallow(board)[self.strategy][0, :, :].clone()
            for (i, move) in enumerate(sorted_legal_moves):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if False:
                    best_res = 0.5
                    board.pop()
                else:
                    board.pop()
                    move = move.uci()
                    s1 = _parse_square(move[0:2])
                    if move[4:] in ['R', 'r']:
                        s2 = 64
                    elif move[4:] in ['B', 'b']:
                        s2 = 65
                    elif move[4:] in ['N', 'n']:
                        s2 = 66
                    else:
                        assert move[4:] in ['Q', 'q', '']
                        s2 = utils._parse_square(move[2:4])
                    best_res = avs[s1, s2].item()
                move_values.append((best_res, i))
            (best_value, best_idx) = max(move_values)
            best_move = sorted_legal_moves[best_idx]
        return best_move




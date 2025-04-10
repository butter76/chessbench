from collections.abc import Callable, Sequence
from typing import cast

import chess
import chess.engine
import haiku as hk
import jax
import jax.nn as jnn
import numpy as np
import scipy.special

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

class MyTransformerEngine(engine.Engine):
    def __init__(
        self,
        checkpoint_path: str,
        limit: chess.engine.Limit,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._limit = limit


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
        if False:
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
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition() or board.is_stalemate():
                    value[i] = 0.5
                board.pop()
            best_ix = cast(int, torch.argmin(value).item())
            best_move = sorted_legal_moves[best_ix]
            best_value = value[best_ix].item()
            # print(f"Best Move: {best_move} with value {best_value}")
        elif False:
            move_values = []
            avs = self.analyse_shallow(board)['avs'][0, :, :].clone()
            for (i, move) in enumerate(sorted_legal_moves):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition() or board.is_stalemate():
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
        elif True:
            move_values = []
            avs = self.analyse_shallow(board)['policy'][0, :, :].clone()
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
        else:
            move_values = []
            avs = self.analyse(board)['avs']
            for i, (move, av) in enumerate(zip(sorted_legal_moves, avs)):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition() or board.is_stalemate():
                    best_res = 0.5
                else:
                    av = av.clone()
                    legal_moves = torch.zeros((72, 72)).to(self.device)
                    for next_move in engine.get_ordered_legal_moves(board):
                        board.push(next_move)
                        if board.is_checkmate():
                            board.pop()
                            best_res = 1
                            break
                        next_move = next_move.uci()
                        s1 = _parse_square(next_move[0:2])
                        if next_move[4:] in ['R', 'r']:
                            s2 = 64
                        elif next_move[4:] in ['B', 'b']:
                            s2 = 65
                        elif next_move[4:] in ['N', 'n']:
                            s2 = 66
                        else:
                            assert next_move[4:] in ['Q', 'q', '']
                            s2 = utils._parse_square(next_move[2:4])
                        if board.is_stalemate():
                            av[s1, s2] = 0.5
                        legal_moves[s1, s2] = 1
                        board.pop()
                    else:
                        best_res = torch.max(av[legal_moves == 1]).item()
                move_values.append((1 - best_res, i))
                board.pop()
            (best_value, best_idx) = max(move_values)
            best_move = sorted_legal_moves[best_idx]
            
                

        return best_move




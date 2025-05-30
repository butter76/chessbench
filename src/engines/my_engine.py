from collections.abc import Callable, Sequence
import math
from typing import cast, Union, Optional
from enum import Enum, auto

import chess
import chess.engine
import numpy as np
from collections import defaultdict

from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src.engines import engine
from searchless_chess.src.engines.strategy import MoveSelectionStrategy
from searchless_chess.src.models.transformer import ChessTransformer
from searchless_chess.src.engines.utils.node import Node, MCTSNode
from searchless_chess.src.utils import move_to_indices
import torch
from torch.amp.autocast_mode import autocast

from searchless_chess.src.engines.utils.nnutils import get_policy, reduced_fen
from searchless_chess.src.engines.search import (
    ValueSearch, PolicySearch, AVSSearch, NegamaxSearch, 
    AlphaBetaSearch, PVSSearch, MTDFSearch, MCTSSearch
)
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")

class NodeType(Enum):
    PV_NODE = auto()
    CUT_NODE = auto()
    ALL_NODE = auto()

NULL_EPS = 0.0001

class MyTransformerEngine(engine.Engine):
    def __init__(
        self,
        checkpoint_path: str,
        limit: chess.engine.Limit,
        strategy: Union[MoveSelectionStrategy, str] = MoveSelectionStrategy.VALUE,
        search_depth: int | float = 2,
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

        # Initialize search algorithms
        self.search_algorithms = {
            MoveSelectionStrategy.VALUE: ValueSearch(),
            MoveSelectionStrategy.AVS: AVSSearch("avs"),
            MoveSelectionStrategy.AVS2: AVSSearch("avs2"),
            MoveSelectionStrategy.POLICY: PolicySearch("policy"),
            MoveSelectionStrategy.OPT_POLICY_SPLIT: PolicySearch("opt_policy_split"),
            MoveSelectionStrategy.NEGAMAX: NegamaxSearch(self.search_ordering_strategy),
            MoveSelectionStrategy.ALPHA_BETA: AlphaBetaSearch(self.search_ordering_strategy),
            MoveSelectionStrategy.ALPHA_BETA_NODE: PVSSearch(),  # Legacy name for PVS
            MoveSelectionStrategy.PVS: PVSSearch(),
            MoveSelectionStrategy.MTDF: MTDFSearch(),
            MoveSelectionStrategy.MCTS: MCTSSearch(),
        }

    def analyse_shallow(self, board: chess.Board) -> engine.AnalysisResult:
        x = np.array([tokenizer.tokenize(board.fen())])
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        return output

    def play(self, board: chess.Board) -> chess.Move:
        self.model.eval()

        # Get the appropriate search algorithm
        search_algorithm = self.search_algorithms.get(self.strategy)
        if search_algorithm is None:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Create inference functions for the search algorithm
        def inference_func(board: chess.Board):
            return self.analyse_shallow(board)
        
        def batch_inference_func(boards):
            # Convert FEN strings back to boards and analyze them
            board_objs = []
            for fen in boards:
                board_obj = chess.Board(fen)
                board_objs.append(board_obj)
            return self.analyse_batch(board_objs)
        
        # Configure search parameters based on strategy
        search_kwargs = {
            'num_nodes': 400 if self.strategy in [MoveSelectionStrategy.MTDF, MoveSelectionStrategy.PVS] else None,
            'num_rollouts': int(self.search_depth) if self.strategy == MoveSelectionStrategy.MCTS else None,
        }
        
        # Perform the search
        result = search_algorithm.search(
            board=board,
            inference_func=inference_func,
            batch_inference_func=batch_inference_func,
            depth=self.search_depth,
            **{k: v for k, v in search_kwargs.items() if v is not None}
        )
        
        
        return result.move
    
    def analyse_batch(self, boards):
        """Analyze multiple boards in a batch."""
        x = []
        for board in boards:
            x.append(tokenizer.tokenize(board.fen()))
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            output = self.model(x)
        return output




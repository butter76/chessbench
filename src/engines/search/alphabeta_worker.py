"""Minimax search worker for parallel search infrastructure."""

from ctypes import util
import time
import uuid
import chess
from typing import Dict, List, Optional, Tuple, Any

from pyparsing import C

from searchless_chess.src.engines.search import utils
from apache_beam import coders
from searchless_chess.src.constants import CODERS

ACTION_VALUE_CODER = coders.TupleCoder((
    CODERS['move'],
    CODERS['win_prob'],
))

SELF_PLAY_CODER = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
    coders.FloatCoder(), # The terminal reward
    CODERS['count'], # The "moves left" counter
    coders.IterableCoder(ACTION_VALUE_CODER), # The action values
))

class AlphaBetaWorker(utils.SearchWorker):
    """Alpha-Beta search worker that uses parallel evaluation."""
    
    def __init__(self, evaluation_queue: utils.EvaluationQueue, 
                 game_logger: utils.GameLogger,
                 search_manager: utils.SearchManager,
                 max_depth: int = 4):
        """Initialize alpha-beta worker.
        
        Args:
            initial_board: Initial chess board position
            evaluation_queue: Queue for submitting positions for evaluation
            max_depth: Maximum depth of the search tree
        """
        super().__init__(evaluation_queue, game_logger, search_manager)
        self.max_depth = max_depth
        self.reset()

    def transcribe_game(self):
        terminal_reward = self.get_outcome(self.root.board)
        if terminal_reward is None:
            raise ValueError("Game is not terminal")
        log = []
        move_count = 0
        while self.root.parent is not None:
            last_move = self.root.board.peek()
            self.root = self.root.parent
            move_count += 1
            terminal_reward *= -1
            log.append(SELF_PLAY_CODER.encode((
                self.root.board.fen(),
                last_move.uci(),
                self.root.value,
                terminal_reward,
                move_count,
                [(move.uci(), self.root.children[i].value) for i, move in enumerate(self.root.moves)]
            )))
        return log
        
    def reset(self):
        # Get a board from the opening book if available, otherwise use the initial board
        board = self.fetch_board()
        self.root = utils.Node(board, None, -1, None, False)
        self.transposition_table = {}
        self.increment_tt(self.root.board)

    def increment_tt(self, board: chess.Board):
        h = self.get_filtered_fen(board)
        if h in self.transposition_table:
            self.transposition_table[h] += 1
        else:
            self.transposition_table[h] = 1

    def get_tt_count(self, board: chess.Board):
        h = self.get_filtered_fen(board)
        return self.transposition_table.get(h, 0)

    def get_filtered_fen(self, board: chess.Board):
        fen = board.fen()
        fen = fen.split(" ")
        return " ".join(fen[:4])

    def get_outcome(self, board: chess.Board):
        if board.is_checkmate():
            return -1
        elif board.is_stalemate():
            return 0
        elif board.is_insufficient_material():
            return 0
        elif board.is_fifty_moves():
            return 0
        elif self.get_tt_count(board) == 2:
            # This board would create a 3rd repetition, so it's a draw
            return 0
        return None
        
    def search_step(self):
        """Perform a single step of the search."""

        if self.root.is_fully_expanded():
            self.root = self.root.get_best_child()

            if self.get_outcome(self.root.board) is not None:
                game_log = self.transcribe_game()
                # Send each encoded position to the game logger
                for encoded_position in game_log:
                    self.transmit_game(encoded_position)
                self.reset()
            else:
                self.increment_tt(self.root.board)
            
        index = len(self.root.children)
        move = self.root.moves[index]

        new_board = self.root.board.copy()
        new_board.push(move)
        outcome = self.get_outcome(new_board)

        if outcome is not None:
            child = utils.Node(new_board, self.root, outcome, None, True)
            self.root.add_child(child)
        else: 
            self.submit_position(new_board, str(uuid.uuid4()))
            return
        
        # Must always call submit_position to avoid deadlocks, therefore we try again
        return self.search_step()
    
    def process_result(self, result: utils.EvaluationResult):
        child = utils.Node(result.board, self.root, result.value, result.policy, False)
        self.root.add_child(child)

        while child.parent is not None:
            if -1 * child.value > child.parent.value:
                child.parent.value = -1 * child.value
                child = child.parent
            else:
                break
    

        



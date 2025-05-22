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
from searchless_chess.src.models.transformer import ChessTransformer
from searchless_chess.src.engines.utils.node import Node, MCTSNode
from searchless_chess.src.utils import move_to_indices
import torch
from torch.amp.autocast_mode import autocast

from searchless_chess.src.engines.utils.nnutils import get_policy, reduced_fen
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
    ALPHA_BETA_NODE = "alpha_beta_node"
    MCTS = "mcts"

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

        self.metrics = {
            'num_nodes': 0,
            'num_searches': 0,
            'policy_perplexity': 0,
            'tt_hits': 0,
        }

    def _get_ordered_moves(self, board: chess.Board, ordering_strategy: MoveSelectionStrategy | None) -> tuple[list[chess.Move], list[float]]:
        """
        Gets legal moves ordered by the specified strategy head's output,
        or by a default heuristic (checkmates/captures first) if strategy is None.
        """
        self.metrics['num_nodes'] += 1

        # Use the specified policy/AVS head for ordering if provided
        assert ordering_strategy and ordering_strategy in [
            MoveSelectionStrategy.AVS, MoveSelectionStrategy.AVS2,
            MoveSelectionStrategy.POLICY, MoveSelectionStrategy.OPT_POLICY_SPLIT]
        
        output = self.analyse_shallow(board)
        values = output['value'][:, 0] * 2.0 - 1.0 # Move from [0, 1] to [-1, 1]
        is_legal = output['legal'].clone() > 0 # Rather than manually checking if the move is legal, just use the model's prediction of legality
        policies = output['policy'].clone()
        policies[~is_legal] = float('-inf')
        policies = torch.nn.functional.softmax(policies.view(1, -1), dim=-1).view(1, is_legal.shape[1], -1)
        move_scores = []
        
        for move in board.legal_moves:
            s1, s2 = move_to_indices(move, flip=board.turn == chess.BLACK)
            score = policies[0, s1, s2].item()
            move_scores.append(score)
            
        # Sort moves based on scores (descending)
        s = sorted(zip(move_scores, board.legal_moves), key=lambda pair: pair[0], reverse=True)
        sorted_moves = [move for _, move in s]
        policy_weight = [score for score, _ in s]
        return sorted_moves, policy_weight

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

        values = values.float().cpu().numpy()
        policies = policies.float().cpu().numpy()
        
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
        self.metrics['num_nodes'] += 1
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

        ordered_moves, _ = self._get_ordered_moves(board, self.search_ordering_strategy)

        for i, move in enumerate(ordered_moves):
            board.push(move)
            # Score returned by recursive call is relative to the next player. Negate it.
            score, _ = self.negamax(board, depth - 1)
            score = -score
            board.pop()
            if score > max_eval:
                max_eval = score
                best_move = move
            # if i == 1 and depth == 1:
            #     break
            
            
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

        ordered_moves, policy_weight = self._get_ordered_moves(board, self.search_ordering_strategy)

        for move, move_weight in zip(ordered_moves, policy_weight):
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

    def _create_node(self, board: chess.Board, parent: Optional[Node] = None, node_class: type[Node] = Node, tt: dict[str, Node | None] | None = None) -> Node:
        terminal_value = None
        if board.is_checkmate():
            terminal_value = -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
            terminal_value = 0.0

        if terminal_value is not None:
            return node_class(board=board, parent=parent, value=terminal_value, terminal=True)
        

        if tt is not None and reduced_fen(board) in tt:
            self.metrics['tt_hits'] += 1
            return tt[reduced_fen(board)]
        
        self.metrics['num_nodes'] += 1
        output = self.analyse_shallow(board)
        value = output['value'][:, 0] * 2.0 - 1.0 # Move from [0, 1] to [-1, 1]
        policy = output['policy'].clone()

        values = value.cpu().float().numpy()
        policies = policy.cpu().float().numpy()

        policy, _ = get_policy(board, policies[0])
        new_node = node_class(board=board, parent=parent, value=values[0], policy=policy)

        if tt is not None:
            tt[reduced_fen(board)] = new_node

        return new_node


    
    def alpha_beta_policy_node(self, node: Node, depth: float, alpha: float, beta: float, history: dict[str, int], tt: dict[str, Node | None] | None, rec_depth: int = 0) -> tuple[float, Optional[chess.Move]]:
        """
        Node-based alpha-beta search with policy-based move ordering and depth extension.
        Returns score relative to the current player and the best move found.
        
        Args:
            node: Current node in the search tree
            depth: Remaining search depth (can be fractional with policy extension)
            alpha: Lower bound score
            beta: Upper bound score
            rec_depth: Current recursion depth to prevent infinite loops
            history: Dictionary of previous positions and their counts
            tt: Transposition table to store and retrieve previously evaluated nodes
        Returns:
            Tuple of (best score, best move)
        """
        board = node.board

        if node.is_terminal():
            return node.value, None
        
        if history[reduced_fen(board)] >= 1:
            # This position has been seen before, so we can return a draw
            return 0.0, None
        
        # Leaf node evaluation
        if depth <= 0.0 and node.is_leaf():
            return node.value, None
        
        # Safety check against excessive recursion
        if rec_depth > 50:
            # If we've recursed too deep, likely a forced three-fold repetition
            print(node.print_lineage())
            raise ValueError("Excessive recursion")
            return node.value, None
        
        max_eval = -float('inf')
        best_move = None

        history[reduced_fen(board)] += 1
        
        total_move_weight = 0
        for i, (move, move_weight) in enumerate(node.policy):
            assert move_weight > 0.0

            # TODO: Mystery constant of 0.1
            new_depth = depth + math.log(move_weight + 1e-6) - 0.1

            if new_depth <= 0 and i >= len(node.children):
                if total_move_weight > 0.85 and i >= 2:
                    # Drop the low probability moves until the depth is high enough to explore them
                    # But don't drop the first two moves
                    # And don't drop moves that have already been expanded
                    break
                
            
            if i >= len(node.children):
                # Create a new child node
                child_board = board.copy()
                child_board.push(move)
                child_node = self._create_node(child_board, parent=node, tt=tt)
                node.add_child(child_node)
            else:
                child_node = node.children[i]

            # Recursive call with negated bounds
            score, _ = self.alpha_beta_policy_node(
                child_node, 
                new_depth, 
                -beta, 
                -alpha,
                history,
                tt,
                rec_depth + 1
            )
            score = -score  # Negate for current player's perspective
            
            if score > max_eval:
                max_eval = score
                best_move = move
                
            # Update alpha for pruning
            alpha = max(alpha, max_eval)
            
            # Beta cutoff
            if alpha >= beta:
                break

            total_move_weight += move_weight

        history[reduced_fen(board)] -= 1
        
        return max_eval, best_move
    
    def mcts(self, root: MCTSNode, num_rollouts: int) -> tuple[float, chess.Move]:
        """
        Performs Monte Carlo Tree Search with policy-based move ordering.
        Returns score relative to the current player and the best move found.
        """
        for _ in range(num_rollouts):
            node = root
            history = defaultdict(int)
            while not node.is_terminal():
                history[reduced_fen(node.board)] += 1
                node, new_node = self._choose_node(node)
                if new_node:
                    break


            Q = node.get_value()
            if history[reduced_fen(node.board)] >= 1:
                # This position has been seen before, so we can return a draw
                Q = 0.0

            node = node.parent
            while node is not None:
                Q = -Q
                node.avg_in(Q)
                node = node.parent
        
        best_move = None
        best_N = 0
        for i, child in enumerate(root.children):
            if child.N > best_N:
                best_N = child.N
                best_move = root.policy[i][0]

        return root.get_value(), best_move
            


    def _choose_node(self, node: MCTSNode) -> tuple[MCTSNode, bool]:
        """
        Chooses a child node to expand based on the UCT formula.
        """
        c_puct = 2.1
        best_q = -float('inf')
        best_idx = -1
        total_policy = 0.0
        for i, (_, p) in enumerate(node.policy):
            
            if i < len(node.children):
                child = node.children[i]
                q = -1 * child.get_value()
                n = child.N
                total_policy += p
            else:
                # First Play Urgency (FPU)
                q = node.get_value() - 0.9 * ((total_policy) ** 0.5)
                n = 0
            u = c_puct * p * math.sqrt(node.N) / (1 + n)
            if q + u > best_q:
                best_q = q + u
                best_idx = i


        if best_idx >= len(node.children):
            assert best_idx == len(node.children)
            # Create a new child node
            child_board = node.board.copy()
            child_board.push(node.policy[best_idx][0])
            child_node = self._create_node(child_board, parent=node, node_class=MCTSNode)
            node.add_child(child_node)
            return child_node, True
        return node.children[best_idx], False

    


    def play(self, board: chess.Board) -> chess.Move:
        self.model.eval()
        self.metrics['num_searches'] += 1

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
                    self.metrics['policy_perplexity'] += 1
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
            best_value, best_move = self.alpha_beta(board, self.search_depth, -1.0, 1.0)
        elif self.strategy == MoveSelectionStrategy.ALPHA_BETA_NODE:
            root_node = self._create_node(board)
            history = defaultdict(int)
            tt = defaultdict(lambda: None)
            best_value, best_move = self.alpha_beta_policy_node(root_node, self.search_depth, -1.0, 1.0, history, tt)
        elif self.strategy == MoveSelectionStrategy.MCTS:
            root_node = self._create_node(board, node_class=MCTSNode)
            best_value, best_move = self.mcts(root_node, self.search_depth)

        else:
             raise ValueError(f"Unknown strategy: {self.strategy}")

        return best_move




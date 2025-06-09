import chess
import math
from typing import Optional, Tuple, Dict
from collections import defaultdict

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src.engines.utils.node import Node, TTEntry, NodeType, MMMCTSNode
from searchless_chess.src.engines.utils.nnutils import reduced_fen

class MMMCTSSearch(SearchAlgorithm):
    """Monte Carlo Tree Search with mini-max."""
    
    def __init__(self):
        super().__init__("mmmcts")
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform MMMCTS search with depth limit.
        """

        num_nodes = kwargs.get('num_nodes', 400)

        # Store inference function for use in node creation
        self.inference_func = inference_func
        self.tt_hits = 0  # Reset TT hit counter
        
        # Create root node
        root = self._create_mmmcts_node(board, inference_func)
        history = defaultdict(int)
        tt = defaultdict(lambda: None)
        
        start_depth = 2.0
        node_count = self.metrics['num_nodes']
        current_depth = start_depth
        best_score = None
        best_move = None
        
        # Iterative deepening
        while self.metrics['num_nodes'] - node_count < num_nodes * 0.95 and current_depth < 20:
            score, move = self._mmmcts_root(root, current_depth, history, tt)

            if move is not None:
                best_score = score
                best_move = move

            current_depth += 0.2

        self.metrics['depth'] = current_depth

        if best_move is None:
            # Fallback - just pick the first move from policy
            if root.policy:
                best_move = root.policy[0][0]
            else:
                best_move = list(board.legal_moves)[0]

        return SearchResult(
            move=best_move,
            score=best_score if best_score is not None else 0.0,
            metadata={
                'depth': current_depth,
                'nodes': self.metrics['num_nodes'],
                'tt_hits': self.tt_hits,
                'tt_entries': len([entry for entry in tt.values() if entry is not None])
            }
        )
    
    def _mmmcts_root(self, root: Node, depth: float, history: Dict[str, int], tt: Dict[str, TTEntry]) -> tuple[float, Optional[chess.Move]]:
        """Root level MMMCTS call."""
        return self._mmmcts(root, depth, history, tt, 0)
    
    def _mmmcts(self, node: Node, depth: float,
             history: Dict[str, int], tt: Dict[str, TTEntry], 
             rec_depth: int = 0) -> tuple[float, Optional[chess.Move]]:
        """
        MMMCTS with policy-based move ordering and depth extension.
        """
        board = node.board
        position_key = reduced_fen(board)

        if depth > node.depth:
            node.depth = depth

        if node.is_terminal():
            return node.value, None
        
        if history[position_key] >= 1:
            # This position has been seen before, so we can return a draw
            node.Q = 0.0
            return 0.0, None
        
        # Query transposition table
        if position_key in tt and tt[position_key] is not None:
            tt_score = tt[position_key].query(-1, 1, depth)
            if tt_score is not None:
                self.tt_hits += 1
                node.Q = tt_score
                return tt_score, None
        
        # Leaf node evaluation
        if depth <= 1.0 and node.is_leaf():
            return node.value, None
        
        if rec_depth > 50:
            return node.value, None
        
        history[position_key] += 1
        
        total_policy_weight = 0.0
        if node.is_leaf():
            # Quickly expand the first 85% of the policy, and at least 2
            for i, (move, move_weight, child_node) in enumerate(node.policy):
                if total_policy_weight >= 0.85 and i >= 2:
                    break
                if child_node is None:
                    child_board = board.copy()
                    child_board.push(move)
                    child_node = self._create_mmmcts_node(child_board, inference_func=self.inference_func, parent=node, tt=tt)
                    node.add_child(child_node, move)

                    # Attempt a 0 depth search to get TT hits or leaf node evaluations
                    self._mmmcts(child_node, 0, history, tt, rec_depth + 1)

                total_policy_weight += move_weight

            node.update()


        N = node.child_N()
        while math.log(N) < depth - 0.1:
            c_puct = 2.1
            fpu_factor = 0.9
            best_q = -float('inf')
            best_idx = -1
            total_policy = 0.0

            for i, (move, move_weight, child_node) in enumerate(node.policy):
                if child_node is not None:
                    q = -1 * child_node.get_Q()
                    n = child_node.get_N()
                    total_policy += move_weight
                else:
                    # First Play Urgency (FPU)
                    q = node.get_Q() - fpu_factor * ((total_policy) ** 0.5)
                    n = 0

                u = c_puct * move_weight * math.sqrt(N) / (1 + n)
                if q + u > best_q:
                    best_q = q + u
                    best_idx = i

            move, move_weight, child_node = node.policy[best_idx]
            if child_node is None:
                child_board = board.copy()
                child_board.push(move)
                child_node = self._create_mmmcts_node(child_board, inference_func=self.inference_func, parent=node, tt=tt)
                node.add_child(child_node, move)

                self._mmmcts(child_node, 0, history, tt, rec_depth + 1)
            else:
                self._mmmcts(child_node, child_node.depth + 0.1, history, tt, rec_depth + 1)

            node.update()
            N = node.child_N()
        
        history[position_key] += -1

        # Store result in transposition table
        if position_key in tt and tt[position_key] is not None:
            # Exact score
            tt[position_key].store_exact(node.get_Q(), depth)

        return node.get_Q(), node.best_child()
    

    def _create_mmmcts_node(self, board: chess.Board, inference_func, parent = None, tt: Dict[str, TTEntry] = None) -> MMMCTSNode:
        """Create an MCTS node with static evaluation and policy."""
        # Check for terminal conditions
        terminal_value = None
        if board.is_checkmate():
            terminal_value = -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            terminal_value = 0.0

        if terminal_value is not None:
            return MMMCTSNode(board=board, parent=parent, value=terminal_value, terminal=True)
        
        position_key = reduced_fen(board)
        if tt is not None and position_key in tt and tt[position_key] is not None:
            tt_entry = tt[position_key]
            return MMMCTSNode(board=board, parent=parent, value=tt_entry.static_value, policy=tt_entry.policy, U=tt_entry.U)
        
        self.metrics['num_nodes'] += 1


        output = self.inference_func(board)
        value = output['value'][0, 0].item() * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
        
        # Get policy for move ordering
        from searchless_chess.src.engines.utils.nnutils import get_policy
        policies = output['policy'].float().cpu().numpy()
        policy, _, perplexity = get_policy(board, policies[0])
        
        wdl = output['wdl'].float().cpu().numpy()
        
        # Apply softmax to WDL
        import numpy as np
        wdl_exp = np.exp(wdl[0] - np.max(wdl[0]))  # Subtract max for numerical stability
        wdl_softmax = wdl_exp / np.sum(wdl_exp)
        
        W, D, L = wdl_softmax[0], wdl_softmax[1], wdl_softmax[2]
        
        wdl_variance = (max(1 - D - value ** 2, 0.0) ** 0.5)

        new_node = MMMCTSNode(board=board, parent=parent, value=value, policy=policy, U=wdl_variance)

        # Store static evaluation in transposition table
        if tt is not None:
            tt[position_key] = TTEntry(static_value=value, policy=policy, U=wdl_variance)

        return new_node

import chess
import math
from typing import Optional, Tuple
from collections import defaultdict

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src.engines.utils.node import MCTSNode
from searchless_chess.src.engines.utils.nnutils import reduced_fen


class MCTSSearch(SearchAlgorithm):
    """Monte Carlo Tree Search with policy-based move ordering."""
    
    def __init__(self):
        super().__init__("mcts")
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform MCTS search with the specified number of rollouts.
        """
        num_rollouts = kwargs.get('num_rollouts', int(depth))
        
        # Create root node
        root = self._create_mcts_node(board, inference_func)
        
        # Perform MCTS rollouts
        for _ in range(num_rollouts):
            node = root
            history = defaultdict(int)
            
            # Selection and expansion
            while not node.is_terminal():
                history[reduced_fen(node.board)] += 1
                node, new_node = self._choose_node(node, inference_func)
                if new_node:
                    break

            # Evaluation
            Q = node.get_value()
            if history[reduced_fen(node.board)] >= 1:
                # This position has been seen before, so we can return a draw
                Q = 0.0

            # Backpropagation
            current = node.parent
            while current is not None:
                Q = -Q
                current.avg_in(Q)
                current = current.parent
        
        # Select best move based on visit count
        best_move = None
        best_N = 0
        for move, _, child in root.policy:
            if child is not None and child.N > best_N:
                best_N = child.N
                best_move = move

        return SearchResult(
            move=best_move,
            score=root.get_value(),
            metadata={
                'rollouts': num_rollouts,
                'nodes': self.metrics['num_nodes'],
                'root_visits': root.N
            }
        )
    
    def _choose_node(self, node: MCTSNode, inference_func) -> Tuple[MCTSNode, bool]:
        """
        Chooses a child node to expand based on the UCT formula.
        Returns (selected_node, is_new_node).
        """
        c_puct = 2.1
        fpu_factor = 0.9
        best_q = -float('inf')
        best_idx = -1
        total_policy = 0.0
        
        for i, (move, p, child) in enumerate(node.policy):
            if child is not None:
                q = -1 * child.get_value()
                n = child.N
                total_policy += p
            else:
                # First Play Urgency (FPU)
                q = node.get_value() - fpu_factor * ((total_policy) ** 0.5)
                n = 0
            
            u = c_puct * p * math.sqrt(node.N) / (1 + n)
            if q + u > best_q:
                best_q = q + u
                best_idx = i

        move, p, child = node.policy[best_idx]
        if child is None:
            # Create a new child node
            child_board = node.board.copy()
            child_board.push(move)
            child_node = self._create_mcts_node(child_board, inference_func, parent=node)
            node.add_child(child_node, move)
            return child_node, True
        
        return child, False
    
    def _create_mcts_node(self, board: chess.Board, inference_func, parent: Optional[MCTSNode] = None) -> MCTSNode:
        """Create an MCTS node with static evaluation and policy."""
        # Check for terminal conditions
        terminal_value = None
        if board.is_checkmate():
            terminal_value = -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            terminal_value = 0.0

        if terminal_value is not None:
            return MCTSNode(board=board, parent=parent, value=terminal_value, terminal=True)
        
        self.metrics['num_nodes'] += 1
        
        # Get model evaluation
        if inference_func:
            output = inference_func(board)
            value = output['value'][0, 0].item() * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
            
            # Get policy for move ordering
            from searchless_chess.src.engines.utils.nnutils import get_policy
            policies = output['policy'].float().cpu().numpy()
            policy, _, _ = get_policy(board, policies[0])
        else:
            # Fallback values
            value = 0.0
            policy = [(move, 1.0/len(list(board.legal_moves))) for move in board.legal_moves]
        
        return MCTSNode(board=board, parent=parent, value=value, policy=policy) 
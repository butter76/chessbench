from typing import Optional, List, Tuple, cast
import chess
from collections.abc import Sequence
from searchless_chess.src.engines.utils.nnutils import reduced_fen
class Node:
    """
    Represents a node in a chess search tree.
    Used for algorithms like negamax and alpha-beta pruning.
    """
    
    def __init__(self, 
                 board: chess.Board,
                 parent: Optional['Node'] = None,
                 value: float = 0.0,
                 policy: Optional[List[Tuple[chess.Move, float]]] = None,
                 terminal: bool = False):
        """
        Initialize a new Node in the search tree.
        
        Args:
            board: The chess board state at this node
            parent: The parent node (None if this is the root)
            value: Static evaluation of this position
            policy: List of (move, probability) tuples in decreasing probability order
            terminal: Whether this node is a terminal state (checkmate, stalemate, etc.)
        """
        self.board = board.copy()
        self.moves = list(self.board.generate_legal_moves())
        self.parent = parent
        self.value = value
        self.policy = policy if policy is not None else []
        self.terminal = terminal
        self.children: List['Node'] = []

        
    def is_root(self) -> bool:
        """Check if this node is the root of the tree (has no parent)."""
        return self.parent is None
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no expanded children)."""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Check if this node is a terminal state (checkmate, stalemate, etc.)."""
        return self.terminal
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible moves from this position have been expanded.
        This compares the number of children to the length of the policy list.
        """
        return len(self.children) == len(self.policy)
    
    def add_child(self, child: 'Node') -> None:
        """Add a child node to this node."""
        self.children.append(child)
        
    def get_best_child(self, negamax: bool = True) -> Optional['Node']:
        """Get the child with the highest value (from opponent's perspective)."""
        if not self.children:
            return None
        # For negamax/alphabeta, we want the minimum value from opponent's perspective
        return min(self.children, key=lambda child: child.value) if negamax else max(self.children, key=lambda child: child.value)
    
    def get_highest_policy_move(self) -> Optional[chess.Move]:
        """
        Get the move with the highest policy value.
        Returns None if no policy is available.
        """
        if len(self.policy) == 0:
            return None
        return self.policy[0][0]  # Return the move from the highest policy tuple
    
    def get_move_to_child(self, child: 'Node') -> Optional[chess.Move]:
        """Find the move that leads to a specific child node."""
        if child not in self.children:
            return None
            
        # Compare FENs to find the move that leads to this child
        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)
            if test_board.fen() == child.board.fen():
                return move
        return None
    
    def __str__(self) -> str:
        """String representation of the node."""
        status = "Terminal" if self.terminal else "Internal"
        value_str = f"{self.value:.4f}"
        move_count = len(self.policy)
        child_count = len(self.children)
        return f"Node({self.board.fen()}, Value={value_str}, Moves={move_count}, Children={child_count}, policy: {self.policy})"
        
    def print_lineage(self) -> str:
        """
        Print this node and all its ancestors in a tree-like structure.
        Each node is indented based on its depth in the tree.
        """
        # Collect all ancestors from root to current
        ancestors = []
        current = self
        while current:
            ancestors.insert(0, current)
            current = current.parent
            
        # Build the display string
        result = []
        for i, node in enumerate(ancestors):
            indent = "-" * (i + 1) if i > 0 else ""
            result.append(f"{indent}{node.__str__()}")
            
        return "\n".join(result)

class MCTSNode(Node):
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, value: float = 0.0, policy: Optional[List[Tuple[chess.Move, float]]] = None, terminal: bool = False):
        super().__init__(board, parent, value, policy, terminal)
        self.Q = value
        self.N = 1

    def get_value(self) -> float:
        """Get the value of this node."""
        return self.Q / self.N
    
    def avg_in(self, Q: float):
        self.Q += Q
        self.N += 1
    
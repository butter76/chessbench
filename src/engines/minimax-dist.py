import chess
from typing import Dict, List, Optional, Tuple
import math
import random
import numpy as np
from typing import Callable, TypeAlias

class Node:
    """Represents a node in the Monte Carlo Search Tree with distribution-based minimax values."""

    def __init__(self, parent: Optional['Node'], move: Optional[chess.Move], board: chess.Board, prior_probability: float = 0.0):
        """
        Initializes a Node.

        Args:
            parent: The parent node. None for the root node.
            move: The chess move that led from the parent to this node. None for the root node.
            board: The chess board state for this node.
            prior_probability: The prior probability (P) of selecting the move leading to this node.
        """
        self.parent: Optional['Node'] = parent
        self.move: Optional[chess.Move] = move
        self.board: chess.Board = board  # Assumes board state is correctly passed

        self.children: Dict[chess.Move, 'Node'] = {}
        self.visit_count: int = 0
        
        # Distribution represented as a CDF with 81 bins from -1 to 1
        # Initialize with a uniform distribution
        self.num_bins = 81
        self.bin_edges = np.linspace(-1.0, 1.0, self.num_bins)
        self.cdf = np.linspace(0.0, 1.0, self.num_bins)  # Uniform CDF
        
        # We still keep a single value for PUCT calculation and display
        self.minimax_value: float = 0.0
        
        self.prior_probability: float = prior_probability # P(s, a) where s=parent, a=move

        self.is_terminal: bool = False
        self.terminal_value: Optional[float] = None # Outcome if terminal (1=win, -1=loss, 0=draw for player *at this node*)
        self.legal_moves: Optional[List[chess.Move]] = None # Populated during expansion

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (i.e., has not been expanded)."""
        return not self.children

    def get_value(self) -> float:
        """
        Returns the minimax value of this node.
        Represents the predicted outcome from the perspective of the player whose turn it is at this node.

        Returns:
            The minimax value, or 0.0 if the node has not been visited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.minimax_value

    def set_distribution_from_value(self, value: float, certainty: float = 0.9):
        """
        Sets the node's distribution based on a single value and certainty.
        
        Args:
            value: The evaluation value (-1 to 1)
            certainty: How certain we are about this value (0 to 1)
        """
        # Find the bin index for this value
        bin_idx = np.searchsorted(self.bin_edges, value)
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        # Create a CDF that represents certainty around this value
        # Higher certainty = steeper CDF around the value
        x = np.linspace(-1.0, 1.0, self.num_bins)
        scale = 0.2 * (1.0 - certainty)  # Scale controls the steepness
        self.cdf = 1.0 / (1.0 + np.exp(-(x - value) / max(scale, 0.01)))
        
        # Update the minimax value
        self.minimax_value = value

    def update(self, value: float, cdf: Optional[np.ndarray] = None):
        """
        Updates the node's statistics during the backpropagation phase.
        For distribution-based minimax, we update based on the children's distributions.

        Args:
            value: The value to backpropagate (used for leaf nodes without children).
            cdf: The CDF to use (if None, will create one from value)
        """
        self.visit_count += 1
        
        # If this is a leaf node or terminal node, use the provided value/distribution
        if self.is_leaf() or self.is_terminal:
            if cdf is not None:
                self.cdf = cdf.copy()
            else:
                self.set_distribution_from_value(value)
            self.minimax_value = value
            return
            
        # Otherwise, compute minimax distribution from children
        if self.children:
            visited_children = [child for child in self.children.values() if child.visit_count > 0]
            if visited_children:
                # For each child, get the negated CDF (opponent's perspective)
                child_cdfs = []
                for child in visited_children:
                    # Negate the CDF by flipping it horizontally and vertically
                    neg_cdf = 1.0 - child.cdf[::-1]
                    child_cdfs.append(neg_cdf)
                
                # Combine CDFs using the product rule (assuming independence)
                # This is equivalent to taking the minimum in the deterministic case
                if len(child_cdfs) == 1:
                    self.cdf = child_cdfs[0]
                else:
                    # Normalize to avoid numerical issues with multiple products
                    combined_cdf = child_cdfs[0]
                    for i in range(1, len(child_cdfs)):
                        combined_cdf = combined_cdf * child_cdfs[i]
                        # Renormalize to avoid numerical underflow
                        if np.max(combined_cdf) > 0:
                            combined_cdf = combined_cdf / np.max(combined_cdf)
                    
                    # Ensure the CDF is monotonically increasing
                    for i in range(1, len(combined_cdf)):
                        combined_cdf[i] = max(combined_cdf[i], combined_cdf[i-1])
                    
                    # Normalize to [0,1]
                    if np.max(combined_cdf) > 0:
                        combined_cdf = combined_cdf / np.max(combined_cdf)
                    
                    self.cdf = combined_cdf
                
                # Update the minimax value based on the expected value of the distribution
                # This is an approximation for PUCT calculation
                pdf = np.diff(np.concatenate(([0], self.cdf)))
                self.minimax_value = np.sum(pdf * self.bin_edges)
            else:
                # If no children have been visited yet, use the provided value
                self.set_distribution_from_value(value)
                self.minimax_value = value

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the node."""
        move_str = self.move.uci() if self.move else "ROOT"
        return f"Node(move={move_str}, N={self.visit_count}, V={self.minimax_value:.3f}, P={self.prior_probability:.3f})"

# Define the type hint for the neural network evaluation function
# It takes a board and returns a tuple: (policy_dict, value)
# policy_dict maps chess.Move to float (probability)
# value is a float representing the board evaluation [-1, 1]
EvaluateNN: TypeAlias = Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]]

class MinimaxDistMCTS:
    """Manages the Monte Carlo Tree Search process with distribution-based minimax values."""
    def __init__(self, initial_board: chess.Board, nn_evaluate_func: EvaluateNN, c_puct: float = 1.41, fpu_reduction: float = 0.5):
        """
        Initializes the MCTS search.

        Args:
            initial_board: The starting board state.
            nn_evaluate_func: A function that takes a chess.Board and returns
                              a tuple (policy_dict, value).
            c_puct: The exploration constant (Cp in the PUCT formula).
            fpu_reduction: Value subtracted from parent's value for FPU when selecting unvisited nodes.
        """
        self.nn_evaluate = nn_evaluate_func
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        # Create the root node - no parent, no move leading to it
        self.root = Node(parent=None, move=None, board=initial_board.copy(), prior_probability=0.0)

    def search(self, num_simulations: int):
        """Runs the MCTS search for a given number of simulations."""
        for _ in range(num_simulations):
            leaf_node = self._select(self.root)
            # Value returned by expand is from the perspective of the leaf_node
            value, cdf = self._expand(leaf_node)
            self._backup(leaf_node, value, cdf)

    def _select(self, node: Node) -> Node:
        """Selects a leaf node starting from the given node using PUCT.

        A leaf node is one that has not been expanded yet (node.children is empty).
        Descends the tree by selecting the child with the highest PUCT score until a leaf is reached.
        """
        current_node = node
        while not current_node.is_leaf():
            best_score = -float('inf')
            best_child = None

            # Ensure legal moves are available (should be populated during expansion)
            if current_node.legal_moves is None:
                # This should ideally not happen if the node was expanded correctly.
                current_node.legal_moves = list(current_node.board.legal_moves)
                if not current_node.legal_moves:
                    # If no legal moves, it's effectively terminal, treat as leaf
                    break

            # Iterate through existing children to find the best one according to PUCT
            for move in current_node.children:
                child_node = current_node.children[move]
                score = self._calculate_puct(current_node, child_node)

                if score > best_score:
                    best_score = score
                    best_child = child_node

            if best_child is None:
                # This case should not be hit if the node is not a leaf,
                # as a non-leaf node must have children.
                # If it happens, it might indicate an issue in expansion or selection logic.
                # For robustness, we can treat it as a leaf.
                break

            current_node = best_child # Move down the tree

        # Return the node that is determined to be a leaf in this path
        return current_node

    def _expand(self, node: Node) -> Tuple[float, np.ndarray]:
        """Expands a leaf node: computes legal moves, checks terminal state, evaluates with NN.

        Args:
            node: The leaf node to expand.

        Returns:
            A tuple (value, cdf) where:
            - value: The evaluation of the node (either terminal value or NN value),
                    from the perspective of the player whose turn it is at this node.
            - cdf: The CDF representing the distribution of possible values.
        """
        # Generate legal moves if not already done (e.g., for root node initially)
        if node.legal_moves is None:
            node.legal_moves = list(node.board.legal_moves)

        # Check for terminal state
        if node.board.is_game_over(claim_draw=False): # Simplified draw check
            node.is_terminal = True
            result = node.board.result(claim_draw=False)
            if result == '1-0': # White won
                node.terminal_value = 1.0
            elif result == '0-1': # Black won
                node.terminal_value = -1.0
            else: # Draw
                node.terminal_value = 0.0

            # Adjust terminal value based on whose turn it is at the node
            # If it's White's turn and White won (1.0), value is 1.0
            # If it's Black's turn and White won (1.0), value is -1.0
            if node.board.turn == chess.BLACK:
                node.terminal_value *= -1.0

            # For terminal nodes, create a very certain distribution
            node.set_distribution_from_value(node.terminal_value, certainty=0.99)
            node.minimax_value = node.terminal_value
            return node.terminal_value, node.cdf

        # If not terminal, expand using the neural network
        # The NN value is from the perspective of the current player at node.board
        policy_dict, value = self.nn_evaluate(node.board)

        # Create a distribution based on the NN value
        # The certainty could be based on game phase or other heuristics
        certainty = 0.7  # Medium certainty for NN evaluations
        node.set_distribution_from_value(value, certainty)

        # Populate children based on legal moves and policy network output
        for move in node.legal_moves:
            move_prob = policy_dict.get(move, 0.0) # Use get for safety, default to 0 if move not in policy
            # Create the child board state
            child_board = node.board.copy()
            try:
                child_board.push(move)
                 # Create the child node and add it to the parent's children
                node.children[move] = Node(parent=node, move=move, board=child_board, prior_probability=move_prob)
            except Exception as e:
                print(f"Error pushing move {move.uci()} on board {node.board.fen()}: {e}")
                # Decide how to handle invalid moves predicted by policy, e.g., skip
                continue

        # Set the initial minimax value to the NN evaluation
        node.minimax_value = value
        
        # Return the value and CDF from the NN (perspective of the current node's player)
        return value, node.cdf

    def _backup(self, node: Node, value: float, cdf: np.ndarray):
        """Backpropagates the value and distribution up the tree from the expanded node.

        Args:
            node: The node from where the backup starts (the expanded leaf).
            value: The evaluation value obtained from expanding 'node' (from node's perspective).
            cdf: The CDF representing the distribution of possible values.
        """
        current_node = node
        current_value = value
        current_cdf = cdf.copy()
        
        # Value and CDF need to be flipped for the parent during backup
        while current_node is not None:
            # Update the node with the value and CDF
            # Value perspective matches the node being updated
            current_node.update(current_value, current_cdf)
            
            # Flip the value and CDF for the parent (opponent's perspective)
            current_value = -current_value
            # Negate the CDF by flipping it horizontally and vertically
            current_cdf = 1.0 - current_cdf[::-1]
            
            current_node = current_node.parent

    def _calculate_puct(self, parent_node: Node, child_node: Node) -> float:
        """Calculates the PUCT score for an *existing* child node during selection.

        PUCT = Q(parent, child) + U(parent, child)
        Q is from the parent's perspective.
        U is the exploration bonus.
        """
        # Get minimax value from child's perspective, then negate for parent's perspective
        minimax_value = -child_node.minimax_value

        # Get prior probability P(parent, child)
        prior_p = child_node.prior_probability

        # Get visit counts
        parent_visits = parent_node.visit_count
        child_visits = child_node.visit_count

        # Calculate exploration bonus U(parent, child)
        if parent_visits == 0:
             # Should not happen if root is handled correctly, but as fallback:
             exploration_u = self.c_puct * prior_p # Base exploration on prior if parent has no visits
        else:
             exploration_u = self.c_puct * prior_p * (math.sqrt(parent_visits) / (1 + child_visits))

        # PUCT score = Q + U (Q is already from parent's perspective)
        return minimax_value + exploration_u

        # Note on FPU: For unvisited nodes, we could implement First Play Urgency by
        # using a different value than 0 for the minimax_value of unvisited nodes.
        # This could be: parent_node.minimax_value - self.fpu_reduction

    def get_best_move(self, temperature: float = 0.0) -> Optional[chess.Move]:
        """Selects the best move from the root node's children based on visit counts and minimax values.

        Args:
            temperature: Controls the randomness of selection.
                         0: deterministic (choose based on visits and minimax value).
                         >0: sample based on visit counts raised to 1/temperature.

        Returns:
            The selected chess.Move, or None if the root has no children.
        """
        if not self.root.children:
            # If root hasn't been expanded or has no legal moves initially.
            print("Warning: Root node has no children, cannot select best move.")
            if self.root.legal_moves is None:
                self.root.legal_moves = list(self.root.board.legal_moves)
            return random.choice(self.root.legal_moves) if self.root.legal_moves else None

        available_children = self.root.children

        if temperature == 0.0:
            # Deterministic: Choose the move based on visit count and minimax value
            # First check for terminal positions (wins/losses/draws)
            terminal_wins = []
            for move, child in available_children.items():
                if child.is_terminal and child.terminal_value == 1.0:
                    terminal_wins.append((move, child))
            
            # If we have winning moves, choose the one with shortest path to mate (highest visit count)
            if terminal_wins:
                return max(terminal_wins, key=lambda x: x[1].visit_count)[0]
                
            # Otherwise use visit count as primary criterion, minimax value as secondary
            best_move = max(available_children.keys(), 
                           key=lambda move: (available_children[move].visit_count, 
                                            available_children[move].minimax_value))
            return best_move
        else:
            # Probabilistic sampling based on visit counts raised to 1/temperature
            moves = list(available_children.keys())
            visit_counts = [available_children[m].visit_count for m in moves]

            if not visit_counts or all(v == 0 for v in visit_counts):
                 # If no visits recorded, or during early search, sample uniformly
                 if not moves: return None
                 return random.choice(moves)

            # Apply temperature
            try:
                powered_visits = [v**(1.0 / temperature) for v in visit_counts]
                total_power = sum(powered_visits)
            except OverflowError:
                # Handle potential overflow with large visit counts and low temperature
                # Fallback to choosing the max visit count move
                max_visits = -1
                best_move = None
                for i, m in enumerate(moves):
                    if visit_counts[i] > max_visits:
                        max_visits = visit_counts[i]
                        best_move = m
                return best_move

            if total_power == 0:
                # Avoid division by zero if all powered visits are zero (e.g., high temp on low visits)
                return random.choice(moves)

            probabilities = [p / total_power for p in powered_visits]

            # Sample a move based on the calculated probabilities
            try:
                chosen_move = random.choices(moves, weights=probabilities, k=1)[0]
                return chosen_move
            except ValueError:
                # Handle potential issue if probabilities don't sum to 1 (shouldn't happen if total_power > 0)
                print("Warning: Error during weighted random choice. Falling back to max visit count.")
                best_move = max(available_children.keys(), key=lambda move: available_children[move].visit_count)
                return best_move

# --- Placeholder for MCTS class later ---

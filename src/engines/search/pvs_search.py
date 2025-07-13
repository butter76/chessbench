import chess
import math
import json
import sys
from typing import Optional, Dict
from collections import defaultdict
from enum import Enum, auto

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src.engines.utils.node import Node, TTEntry
from searchless_chess.src.engines.utils.nnutils import reduced_fen
import searchless_chess.src.data_loader as data_loader
from searchless_chess.src.engines.utils.nnutils import get_policy
import numpy as np


class NodeType(Enum):
    PV_NODE = auto()
    CUT_NODE = auto()
    ALL_NODE = auto()


NULL_EPS = 0.0001


class PVSSearch(SearchAlgorithm):
    """Principal Variation Search with policy-based move ordering and depth extension."""
    
    def __init__(self, verbose=False):
        super().__init__("pvs")
        self.tt_hits = 0  # Track transposition table hits
        self.verbose = verbose
        self.node_counter = 0  # Unique node ID counter
        self.logged_nodes = set()  # Track which nodes have been logged
        
        if self.verbose:
            print("=== PVS SEARCH TREE LOG START ===", file=sys.stderr)

    def reset_metrics(self):
        """Reset engine metrics."""
        self.metrics = {
            'num_nodes': 0,
            'num_searches': 0,
            'bf': 0,
            'depth': 0,
            'parent_nodes': 0,
            'pv': 0,
        }
        if self.verbose:
            self.node_counter = 0
            self.logged_nodes.clear()
    
    def _generate_node_id(self, board: chess.Board) -> str:
        """Generate a unique node ID based on position and counter."""
        self.node_counter += 1
        return f"node_{self.node_counter}_{reduced_fen(board)}"
    
    def _log_node_expansion(self, node: Node, node_id: str):
        """Log node expansion with all potential children."""
        if not self.verbose or node_id in self.logged_nodes:
            return
            
        self.logged_nodes.add(node_id)
        
        # Get parent ID
        parent_id = None
        if node.parent is not None:
            parent_id = getattr(node.parent, '_log_id', None)
        
        # Get all potential children from policy
        potential_children = []
        for i, (move, prob, child_node, metadata) in enumerate(node.policy):
            child_info = {
                'move': move.uci(),
                'move_san': node.board.san(move),
                'probability': float(prob),
                'U': float(metadata.get('U', 0.0)),
                'Q': float(metadata.get('Q', 0.0)),
                'D': float(metadata.get('D', 0.0))
            }
            potential_children.append(child_info)
        
        # Create log entry
        log_entry = {
            'event': 'node_expansion',
            'node_id': node_id,
            'parent_id': parent_id,
            'fen': node.board.fen(),
            'value': float(node.value),
            'U': float(node.U),
            'expval': float(node.expval),
            'expoppval': float(node.expoppval),
            'is_terminal': node.is_terminal(),
            'potential_children': potential_children,
            'num_potential_children': len(potential_children),
            'timestamp': self.node_counter
        }
        
        # Output as JSON line
        print(json.dumps(log_entry), file=sys.stdout)
        sys.stdout.flush()



    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform PVS search with iterative deepening.
        """
        num_nodes = kwargs.get('num_nodes', 400)
        
        # Store inference function for use in node creation
        self.inference_func = inference_func
        self.tt_hits = 0  # Reset TT hit counter
        
        # Create root node
        root = self._create_node(board, inference_func)
        history = defaultdict(int)
        tt = defaultdict(lambda: None)
        
        start_depth = 2.0
        node_count = self.metrics['num_nodes']
        current_depth = start_depth
        best_score = None
        best_move = None
        
        # Iterative deepening
        while self.metrics['num_nodes'] - node_count < num_nodes * 0.95 and current_depth < 20:
            f = best_score - NULL_EPS / 2 if best_score is not None else None
            score, move = self._pvs_root(root, current_depth, history, tt, f)
            
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

        self.metrics['bf'] = self.metrics['num_nodes'] / max(1, self.metrics['parent_nodes'])

        pv = root
        self.metrics['pv'] += 1
        while not pv.is_terminal() and not pv.is_leaf():
            self.metrics['pv'] += 1
            pv = pv.policy[0][2]
        
        if self.verbose:
            print("=== PVS SEARCH TREE LOG END ===", file=sys.stderr)
        
        return SearchResult(
            move=best_move,
            score=best_score if best_score is not None else 0.0,
            metadata={
                'depth': current_depth,
                'nodes': self.metrics['num_nodes'],
                'bf': self.metrics['bf'],
                'tt_hits': self.tt_hits,
                'tt_entries': len([entry for entry in tt.values() if entry is not None]),
                'pv': self.metrics['pv']
            }
        )
    
    def _pvs_root(self, root: Node, depth: float, history: Dict[str, int], tt: Dict[str, TTEntry], f=None) -> tuple[float, Optional[chess.Move]]:
        """Root level PVS call."""
        return self._pvs(root, depth, -1.0, 1.0, history, tt, NodeType.PV_NODE, 0)
    
    def _pvs(self, node: Node, depth: float, alpha: float, beta: float, 
             history: Dict[str, int], tt: Dict[str, TTEntry], 
             node_type: NodeType = NodeType.PV_NODE, rec_depth: int = 0, soft_create: bool = False) -> tuple[float, Optional[chess.Move]]:
        """
        PVS with policy-based move ordering and depth extension.
        """
        board = node.board
        position_key = reduced_fen(board)

        if node.is_terminal():
            return node.value, None
        
        if history[position_key] >= 1:
            # This position has been seen before, so we can return a draw
            return 0.0, None
        
        # Query transposition table
        if position_key in tt and tt[position_key] is not None:
            tt_score = tt[position_key].query(alpha, beta, depth)
            if tt_score is not None:
                self.tt_hits += 1
                return tt_score, None

        # Multiply likelihood with the variance of this node
        depth_reduction = -2 * math.log(node.U + 1e-6)

        node.sort_and_normalize()
        
        # Leaf node evaluation
        total_move_weight = 0
        weight_divisor = 1.0
        unexpanded_count = 0
        for i, (move, prob, child_node, metadata) in enumerate(node.policy):
            if child_node is None:
                if (total_move_weight > 0.80 and i >= 2) or (total_move_weight > 0.95 and i >= 1):
                    weight_divisor -= prob
                else:
                    unexpanded_count += 1
            
            total_move_weight += prob

        if node.is_leaf() and depth <= math.log(unexpanded_count + 1e-6) + depth_reduction:
            return node.value, None
        
        # Safety check against excessive recursion
        if rec_depth > 50:
            return node.value, None

        if node.is_leaf():
            self.metrics['parent_nodes'] += 1
        
        max_eval = -float('inf')
        history[position_key] += 1
        
        total_move_weight = 0
        best_move_depth = None
        best_move = None
        original_alpha = alpha
        
        for i, (move, move_weight, child_node, metadata) in enumerate(node.policy):
            assert move_weight > 0.0

            # Compute new depth with policy extension
            new_depth = depth + math.log(move_weight + 1e-6) - math.log(weight_divisor + 1e-6) - 0.1

            if best_move_depth is None:
                best_move_depth = new_depth

            # NEW: Using the uncertainty of the child node here
            if child_node is None:
                depth_reduction = -2 * math.log(metadata['U'] + 1e-6)
            
            # Skip low probability moves if depth is too low
            if new_depth <= depth_reduction and child_node is None:
                if (total_move_weight > 0.80 and i >= 2) or (total_move_weight > 0.95 and i >= 1):
                    total_move_weight += move_weight
                    continue
            
            # Create child node if needed
            if child_node is None:
                child_board = board.copy()
                child_board.push(move)
                child_node = self._create_node(child_board, inference_func=self.inference_func, parent=node, tt=tt)
                node.add_child(child_node, move)

                self._backpropagate_policy_updates(child_node, move=move)

            child_re_searches = 0
            RE_SEARCH_DEPTH = 0.2
            
            while True:
                # Use full window for first move, null window for others
                search_alpha = -beta if i == 0 else -alpha - NULL_EPS
                search_beta = -alpha
                
                # Determine child node type
                if i == 0 and node_type == NodeType.PV_NODE:
                    child_node_type = NodeType.PV_NODE
                elif node_type == NodeType.CUT_NODE:
                    child_node_type = NodeType.ALL_NODE
                else:
                    child_node_type = NodeType.CUT_NODE
                
                score, _ = self._pvs(
                    child_node, 
                    new_depth, 
                    search_alpha,
                    search_beta,
                    history,
                    tt,
                    child_node_type,
                    rec_depth + 1
                )
                score = -score  # Negate for current player's perspective

                if i > 0 and score > alpha:
                    # This move improved alpha
                    
                    if new_depth < best_move_depth:
                        # Re-search with deeper depth
                        new_depth += RE_SEARCH_DEPTH
                        child_re_searches += 1
                        continue
                    else:
                        if node_type == NodeType.PV_NODE:
                            # Re-search with full window
                            score, _ = self._pvs(
                                child_node, 
                                new_depth, 
                                -beta, 
                                -alpha,
                                history,
                                tt,
                                NodeType.PV_NODE,
                                rec_depth + 1
                            )
                        else:
                            # Re-search with full window
                            score, _ = self._pvs(
                                child_node, 
                                new_depth, 
                                -beta, 
                                -alpha,
                                history,
                                tt,
                                child_node_type,
                                rec_depth + 1
                            )
                        score = -score
                break

            # Update policy if re-searches occurred
            if child_re_searches > 0:
                if node_type == NodeType.CUT_NODE and score > alpha:
                    # This is a CUT node and the move is best, so have to update the policy quickly
                    new_policy = node.policy[i][1] * math.exp(child_re_searches * RE_SEARCH_DEPTH)
                else:
                    # This is a PV or ALL node, so we can afford to do a slower update
                    new_policy = node.policy[i][1] + 0.1
                    if not (score > alpha):
                        # This is not the best move, so clip the policy
                        clip = max(node.policy[0][1] * 0.98, node.policy[i][1])
                        new_policy = min(new_policy, clip)
                
                node.policy[i] = (move, new_policy, child_node, node.policy[i][3])
                best_move_depth = max(best_move_depth, new_depth)

            if score > max_eval:
                max_eval = score
                best_move = move
            
            # Update alpha for pruning
            alpha = max(alpha, max_eval)
            
            # Beta cutoff
            if alpha >= beta:
                break

            total_move_weight += move_weight

        history[position_key] -= 1
        
        # Store result in transposition table
        if position_key in tt and tt[position_key] is not None:
            if max_eval <= original_alpha:
                # Fail low - upper bound
                tt[position_key].store_upper_bound(max_eval, depth)
            elif max_eval >= beta:
                # Fail high - lower bound
                tt[position_key].store_lower_bound(max_eval, depth)
            else:
                # Exact score
                tt[position_key].store_exact(max_eval, depth)
        
        return max_eval, best_move
    
    def _backpropagate_policy_updates(self, new_node: Node, move: Optional[chess.Move] = None):
        """
        Backpropagate policy updates from a newly created node up to the root.
        
        Args:
            new_node: The newly created node
            value: The value of the new node
            wdl_stdev: The uncertainty (standard deviation) of the new node
        """
        
        parent = new_node.parent
        if parent is None:
            return
        
        # Compute backup values
        # backup_other_turn and backup_same_turn range analysis:
        # Given value ∈ [-1, 1] and wdl_stdev ∈ [0, 1]:
        backup_other_turn = (new_node.expoppval - parent.expval) / (math.exp(1) - 1)
        backup_same_turn = (new_node.expval - parent.expoppval) / (math.exp(1) - 1)
        
        
        # Safety check for backup values
        if backup_other_turn > 1 or backup_other_turn < -1:
            assert False, f"backup_other_turn={backup_other_turn} is outside expected range [-1, 1]. expoppval={new_node.expoppval}, expval={parent.expval}"
        
        if backup_same_turn > 1 or backup_same_turn < -1:
            assert False, f"backup_same_turn={backup_same_turn} is outside expected range [-1, 1]. expval={new_node.expval}, expoppval={parent.expoppval}"

        
        
        # Find the path from parent to root and update policies

        # Find the policy value from parent to the newly created node
        parent_to_node_policy = 1.0  # Default value
        found_policy_entry = False
        for policy_move, prob, _, _ in parent.policy:
            if policy_move == move:
                parent_to_node_policy = prob
                found_policy_entry = True
                break
        
        if not found_policy_entry:
            assert found_policy_entry, "Could not find policy entry from parent to new node"
        
        depth_factor = parent_to_node_policy 
        
        current_node = parent
        distance_from_current = 1  # Distance from the newly created node

        while current_node is not None and current_node.parent is not None:
            distance_from_current += 1
            
            # Find the move that leads from current_node to its parent
            parent_node = current_node.parent
            move_to_parent = None
            policy_value = 1  # Default policy value
            policy_index = -1
            
            for i, (policy_move, policy_prob, policy_child, _) in enumerate(parent_node.policy):
                if policy_child is current_node:
                    move_to_parent = policy_move
                    policy_value = policy_prob
                    policy_index = i
                    break
            
            if move_to_parent is not None:
                value_effect = .95 #scalar 0 to 1. 0 is no updating, 1 is full updating.

                if distance_from_current % 2 == 0:  # Even distance from current node #=1 in the other-same swapped version
                    # Update with backup_same_turn
                    policy_update = depth_factor * backup_same_turn*value_effect * policy_value
                    new_policy_prob = policy_value + policy_update
                else:  # Odd distance from current node
                    # Update with backup_other_turn
                    policy_update = depth_factor * backup_other_turn*value_effect * policy_value
                    new_policy_prob = policy_value + policy_update
                
                # Update the policy entry
                parent_node.policy[policy_index] = (move_to_parent, new_policy_prob, current_node, parent_node.policy[policy_index][3])
                

                # Update depth_factor recursively
                depth_factor = depth_factor * policy_value

                #we will normalize only right before we call on a node!
                #parent_node.normalize()
            
            current_node = parent_node
        
        return
    
    def _create_node(self, board: chess.Board, inference_func=None, parent: Optional[Node] = None, tt: Dict[str, TTEntry] = None, soft_create: bool = False) -> Node | None:
        """Create a node with static evaluation and policy."""
        position_key = reduced_fen(board)
        
        # Check for terminal conditions
        terminal_value = None
        if board.is_checkmate():
            terminal_value = -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            terminal_value = 0.0

        if terminal_value is not None:
            new_node = Node(board=board, parent=parent, value=terminal_value, terminal=True, expval = math.exp(terminal_value/2+1/2), expoppval = math.exp(-terminal_value/2+1/2))
            # Assign node ID for logging
            if self.verbose:
                node_id = self._generate_node_id(board)
                new_node._log_id = node_id
                self._log_node_expansion(new_node, node_id)
            return new_node

        # Check transposition table for cached evaluation
        if tt is not None and position_key in tt and tt[position_key] is not None:
            tt_entry = tt[position_key]
            new_node = Node(board=board, parent=parent, value=tt_entry.static_value, policy=tt_entry.policy, U=tt_entry.U, expval=tt_entry.expval, expoppval=tt_entry.expoppval)
            # Assign node ID for logging
            if self.verbose:
                node_id = self._generate_node_id(board)
                new_node._log_id = node_id
                self._log_node_expansion(new_node, node_id)
            return new_node
        
        if soft_create:
            return None
        
        self.metrics['num_nodes'] += 1
        
        output = inference_func(board)
        value = output['value'][0, 0].item() * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
        
        policies = output['hardest_policy'].float().cpu().numpy()
        U = output['U'].float().cpu().numpy()
        Q = output['Q'].float().cpu().numpy()
        D = output['D'].float().cpu().numpy()
        policy, _, perplexity = get_policy(board, policies[0], U[0], Q[0], D[0])

        hl_logits = output['hl'].float().cpu().numpy()[0]  # Shape: (81,)
        hl_probs = np.exp(hl_logits - np.max(hl_logits))  # Softmax numerically stable
        hl_probs = hl_probs / np.sum(hl_probs)
        
        # Bin centers for NUM_BINS evenly spaced intervals in [0, 1]
        bin_centers = np.array([(2 * i + 1) / (2 * data_loader.NUM_BINS) for i in range(data_loader.NUM_BINS)])
        
        # Compute variance: E[X^2] - E[X]^2
        hl_mean = np.sum(hl_probs * bin_centers)
        hl_variance = np.sum(hl_probs * bin_centers**2) - hl_mean**2
        
        wdl_variance = math.sqrt(max(0, hl_variance * 4))

        #E(exp(value)) we compute
        expval = np.sum(hl_probs * np.exp(bin_centers))
        expoppval = np.sum(hl_probs * np.exp(1-bin_centers))
        
        new_node = Node(board=board, parent=parent, value=value, policy=policy, U=wdl_variance, expval=expval, expoppval=expoppval)

        # Store static evaluation in transposition table
        if tt is not None:
            tt[position_key] = TTEntry(static_value=value, policy=policy, U=wdl_variance, expval=expval, expoppval=expoppval)

        # Assign node ID for logging
        if self.verbose:
            node_id = self._generate_node_id(board)
            new_node._log_id = node_id
            self._log_node_expansion(new_node, node_id)

        return new_node 
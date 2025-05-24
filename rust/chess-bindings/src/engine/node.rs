use shakmaty::{Chess, Position, Move, Board, Setup, Color};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Represents a node in a chess search tree.
/// Used for algorithms like negamax and alpha-beta pruning.
#[derive(Debug, Clone)]
pub struct Node {
    /// The chess board state at this node
    pub board: Chess,
    /// Reference to the parent node (None if this is the root)
    pub parent: Option<Weak<Mutex<Node>>>,
    /// Static evaluation of this position
    pub value: f32,
    /// List of (move, probability) tuples in decreasing probability order
    pub policy: Vec<(Move, f32)>,
    /// Whether this node is a terminal state (checkmate, stalemate, etc.)
    pub terminal: bool,
    /// Child nodes
    pub children: Vec<Arc<Mutex<Node>>>,
    /// Legal moves from this position
    pub moves: Vec<Move>,
}

impl Node {
    /// Initialize a new Node in the search tree.
    pub fn new(
        board: Chess,
        parent: Option<Weak<Mutex<Node>>>,
        value: f32,
        policy: Option<Vec<(Move, f32)>>,
        terminal: bool,
    ) -> Self {
        let moves = board.legal_moves();
        let policy = match policy {
            Some(p) => p,
            _ => moves.iter().map(|m| (m.clone(), 0.0)).collect(),
        };

        Node {
            board,
            parent,
            value,
            policy,
            terminal,
            children: Vec::new(),
            moves: moves.into_iter().collect(),
        }
    }

    /// Check if this node is the root of the tree (has no parent).
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Check if this node is a leaf node (has no expanded children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Check if all possible moves from this position have been expanded.
    /// This compares the number of children to the length of the policy list.
    pub fn is_fully_expanded(&self) -> bool {
        self.children.len() == self.policy.len()
    }

    /// Add a child node to this node.
    pub fn add_child(&mut self, child: Arc<Mutex<Node>>) {
        self.children.push(child);
    }

    /// Get the child with the highest value (from opponent's perspective).
    pub fn get_best_child(&self, negamax: bool) -> Option<Arc<Mutex<Node>>> {
        if self.children.is_empty() {
            return None;
        }

        let mut best_child = None;
        let mut best_value = if negamax { f32::MAX } else { f32::MIN };

        for child in &self.children {
            let child_value = child.lock().unwrap().value;
            if (negamax && child_value < best_value) || (!negamax && child_value > best_value) {
                best_value = child_value;
                best_child = Some(Arc::clone(child));
            }
        }

        best_child
    }

    /// Get the move with the highest policy value.
    /// Returns None if no policy is available.
    pub fn get_highest_policy_move(&self) -> Option<Move> {
        if self.policy.is_empty() {
            return None;
        }
        Some(self.policy[0].0.clone())
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.terminal { "Terminal" } else { "Internal" };
        let value_str = format!("{:.4}", self.value);
        let move_count = self.policy.len();
        let child_count = self.children.len();
        
        write!(f, "Node({}, Value={}, Moves={}, Children={})", 
               status, value_str, move_count, child_count)
    }
}

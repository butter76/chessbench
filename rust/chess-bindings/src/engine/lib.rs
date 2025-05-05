use shakmaty::{Chess, Position, Move, CastlingMode, EnPassantMode, uci::UciMove, fen::Fen};
use std::sync::{Arc, Mutex, Condvar};
use std::cell::RefCell;
use crate::engine::Node;
use crate::engine::utils;
use crate::error::ChessError;
use crate::engine::thread::SearchThread;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use crate::engine::thread::EvalRequest;

use std::future::Future;

/// A simple Rust chess engine implementation
pub struct AlphaBetaEngine  {
    pub root: Arc<Mutex<Node>>,
    pub on_move_callback: Option<Py<PyAny>>,
    pub on_eval_callback: Option<Py<PyAny>>,
    pub on_fen_callback: Option<Py<PyAny>>,
    depth: u32,
    tt_table: HashMap<String, u32>,
    running: bool,
    evaluating: bool,
    waiting: bool,
    eval_request: Arc<Mutex<Option<EvalRequest>>>,
    eval_notify_mutex: Arc<Mutex<bool>>,       // Signals whether evaluation is complete
    eval_notify_condvar: Arc<Condvar>,         // Condition variable for notification
}

impl AlphaBetaEngine {
    pub fn new(fen: Option<String>, depth: u32) -> Result<Self, ChessError> {
        let board = utils::new_board_from_fen_opt(fen)?;
        let root = Arc::new(Mutex::new(Node::new(board, None, -1.0, None, false)));
        
        Ok(Self {
            root,
            depth,
            tt_table: HashMap::new(),
            running: false,
            evaluating: false,
            waiting: false,
            eval_request: Arc::new(Mutex::new(None)),
            eval_notify_mutex: Arc::new(Mutex::new(false)),
            eval_notify_condvar: Arc::new(Condvar::new()),
            on_move_callback: None,
            on_eval_callback: None,
            on_fen_callback: None,
        })
    }
    
    /// Get a filtered FEN string for transposition table lookups.
    /// Only includes the first 4 parts of the FEN (position, active color, castling, en passant)
    /// and excludes halfmove clock and fullmove number.
    pub fn get_filtered_fen(&self, board: Chess) -> String {
        let fen = Fen::from_position(board, EnPassantMode::Legal).to_string();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        parts.iter().take(4).map(|s| *s).collect::<Vec<&str>>().join(" ")
    }

    pub fn get_tt_count(&self, board: Chess) -> u32 {
        let fen = self.get_filtered_fen(board);
        *self.tt_table.get(&fen).unwrap_or(&0)
    }

    pub fn increment_tt(&mut self, board: Chess) {
        let fen = self.get_filtered_fen(board);
        *self.tt_table.entry(fen).or_insert(0) += 1;
    }

    pub fn get_outcome(&self, board: Chess) -> Option<f32> {
        if board.is_checkmate() {
            return Some(-1.0);
        }
        if board.is_stalemate() {
            return Some(0.0);
        }
        if board.is_insufficient_material() {
            return Some(0.0);
        }
        if board.halfmoves() >= 100 {
            return Some(0.0);
        }
        if self.get_tt_count(board) == 2 {
            return Some(0.0);
        }
        return None;
    }

    pub fn submit_position(&mut self, board: Chess) {
        let fen = Fen::from_position(board.clone(), EnPassantMode::Legal).to_string();
        let mut eval_request = self.eval_request.lock().unwrap();
        eval_request.replace(EvalRequest {
            query: fen.clone(),
            value: 0.0,
            policy: vec![],
        });

        drop(eval_request);

        self.waiting = true;

        Python::with_gil(|py| {
            if let Some(callback) = &self.on_eval_callback {
                let locals = PyDict::new(py);
                locals.set_item("fen", fen);

                let _ = callback.call1(py, (locals,));
            }
        });

        // Do NOT block here waiting for a response.  The evaluation will arrive
        // asynchronously via `receive_eval`, which will attach the evaluated node
        // and clear the `waiting` flag.  Blocking here would keep the global `engine`
        // mutex locked and cause a dead-lock.
    }

    pub fn search_step(&mut self) -> Result<(), ChessError> {
        self.evaluating = true;
        while self.is_running() {
            let root = self.root.lock().unwrap();
            
            if root.is_fully_expanded() {
                let best_child_arc = root.get_best_child(true).unwrap();
                let best_child = best_child_arc.lock().unwrap();
                let outcome = self.get_outcome(best_child.board.clone());

                drop(best_child);
                drop(root);

                self.root = Arc::new(Mutex::new(best_child_arc.lock().unwrap().clone()));

                match outcome {
                    Some(outcome) => {
                        self.running = false;
                        break;
                    }
                    _ => {
                        self.increment_tt(best_child_arc.lock().unwrap().board.clone());
                    }
                };
                continue;
            }

            let index = root.children.len();
            let m = root.moves[index].clone();

            let new_board = root.board.clone().play(&m)?;

            let outcome = self.get_outcome(new_board.clone());

            drop(root);

            match outcome {
                Some(outcome) => {
                    let child = Arc::new(Mutex::new(Node::new(new_board, Some(Arc::downgrade(&self.root)), outcome, None, true)));
                    self.root.lock().unwrap().add_child(child);
                }
                _ => {
                    self.submit_position(new_board);
                    return Ok(());
                }
            }
        }
        self.evaluating = false;
        Ok(())
    }
    
    
}

impl SearchThread for AlphaBetaEngine {

    fn receive_move(&mut self, uci_move: &str) -> Result<(), ChessError> {
        // TODO: Implement
        Ok(())
    }

    fn receive_eval(&mut self, eval: crate::engine::thread::EvalResponse) -> Result<(), ChessError> {
        {
            let mut eval_request = self.eval_request.lock().unwrap();
            if let Some(request) = eval_request.as_mut() {
                request.value = eval.value;
                request.policy = eval.policy.clone();
            }
        }
        // Add evaluated position as a child of the current root.
        // We recreate the board from the stored FEN so that we do not rely on
        // any mutable state that might have changed concurrently.
        if let Some(request) = &*self.eval_request.lock().unwrap() {
            // Safely reconstruct the board
            if let Ok(board) = utils::new_board_from_fen(&request.query) {
                let child = Arc::new(Mutex::new(Node::new(
                    board,
                    Some(Arc::downgrade(&self.root)),
                    request.value,
                    None,
                    false,
                )));
                self.root.lock().unwrap().add_child(child);
            }
        }

        // Reset waiting flag because evaluation has been supplied.
        self.waiting = false;
        
        // Notify waiting search logic (if any) using Condvar.
        let mut notified = self.eval_notify_mutex.lock().unwrap();
        *notified = true;
        self.eval_notify_condvar.notify_one();
        
        Ok(())
    }

    fn receive_fen(&mut self, fen: &str) -> Result<(), ChessError> {
        let board = utils::new_board_from_fen(fen)?;
        self.root = Arc::new(Mutex::new(Node::new(board, None, -1.0, None, false)));
        self.tt_table.clear();
        Ok(())
    }

    fn register_on_move_callback(&mut self, callback: Py<PyAny>) -> Result<(), ChessError> {
        self.on_move_callback = Some(callback);
        Ok(())
    }

    fn register_on_eval_callback(&mut self, callback: Py<PyAny>) -> Result<(), ChessError> {
        self.on_eval_callback = Some(callback);
        Ok(())
    }

    fn register_on_fen_callback(&mut self, callback: Py<PyAny>) -> Result<(), ChessError> {
        self.on_fen_callback = Some(callback);
        Ok(())
    }

    fn start(&mut self) -> Result<(), ChessError> {
        self.running = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), ChessError> {
        self.running = false;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn is_evaluating(&self) -> bool {
        self.evaluating
    }

    fn is_waiting(&self) -> bool {
        self.waiting
    }

    fn step(&mut self) -> Result<(), ChessError> {
        self.search_step()
    }
}
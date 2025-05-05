use crate::error::ChessError;
use pyo3::prelude::*;
use std::future::Future;
#[derive(Debug, Clone)]
pub struct EvalResponse {
    pub value: f32,
    pub policy: Vec<(String, f32)>,
}

// A request to evaluate a position
#[derive(Default)]
pub struct EvalRequest {
    pub query: String,
    pub value: f32,
    pub policy: Vec<(String, f32)>,
}

pub trait SearchThread: Send + Sync {

    fn receive_move(&mut self, uci_move: &str) -> Result<(), ChessError>;
    fn receive_eval(&mut self, eval: EvalResponse) -> Result<(), ChessError>;
    fn receive_fen(&mut self, fen: &str) -> Result<(), ChessError>;

    fn register_on_move_callback(&mut self, callback: Py<PyAny>) -> Result<(), ChessError>;
    fn register_on_eval_callback(&mut self, callback: Py<PyAny>) -> Result<(), ChessError>;
    fn register_on_fen_callback(&mut self, callback: Py<PyAny>) -> Result<(), ChessError>;

    fn start(&mut self) -> Result<(), ChessError>;
    fn stop(&mut self) -> Result<(), ChessError>;
    fn is_running(&self) -> bool;
    fn is_evaluating(&self) -> bool;
    fn is_waiting(&self) -> bool;
}


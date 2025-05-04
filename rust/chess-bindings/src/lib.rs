use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{IntoPyArray, PyArray1};

// Import error module and ChessEngine
mod error;
mod engine;
use crate::engine::ChessEngine;

/// A Python wrapper for the Rust ChessEngine
#[pyclass(unsendable)]
struct PyChessEngine {
    engine: ChessEngine,
    running: bool,
    on_move_callback: Option<Py<PyAny>>,
    on_game_end_callback: Option<Py<PyAny>>,
    on_eval_callback: Option<Py<PyAny>>,
}

#[pymethods]
impl PyChessEngine {
    /// Create a new chess engine with default starting position
    #[new]
    fn new() -> Self {
        Self {
            engine: ChessEngine::new(),
            running: false,
            on_move_callback: None,
            on_game_end_callback: None,
            on_eval_callback: None,
        }
    }

    /// Create a chess engine from a FEN string
    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        let engine = ChessEngine::from_fen(fen)?;
        Ok(Self { 
            engine,
            running: false,
            on_move_callback: None,
            on_game_end_callback: None,
            on_eval_callback: None,
        })
    }

    /// Register a callback function that will be called when a move is made
    fn register_on_move_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        self.on_move_callback = Some(callback);
        Ok(())
    }

    /// Register a callback function that will be called when the game ends
    fn register_on_game_end_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        self.on_game_end_callback = Some(callback);
        Ok(())
    }

    /// Register a callback function that will be called when a position is evaluated
    fn register_on_eval_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        self.on_eval_callback = Some(callback);
        Ok(())
    }

    /// Get the current position as FEN string
    fn get_fen(&self) -> String {
        self.engine.get_fen()
    }
    
    /// Get legal moves for the current position in UCI format
    fn get_legal_moves(&self) -> Vec<String> {
        self.engine.get_legal_moves()
            .into_iter()
            .map(|m| m.to_string())
            .collect()
    }
    
    /// Make a move on the board using UCI format
    fn make_move(&mut self, uci_move: &str) -> PyResult<()> {
        let chess_move = self.engine.convert_uci_to_move(uci_move)?;
        self.engine.make_move(&chess_move)?;
        self.notify_move(uci_move)
    }

    /// Start the engine
    fn start(&mut self) -> PyResult<()> {
        self.running = true;
        Ok(())
    }

    /// Stop the engine
    fn stop(&mut self) -> PyResult<()> {
        self.running = false;
        Ok(())
    }

    fn make_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let rust_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rust_vec.into_pyarray(py)
    }
}

// Private implementations not exposed to Python
impl PyChessEngine {
    /// Private method to notify the Python callback about a move
    fn notify_move(&self, uci_move: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            if let Some(callback) = &self.on_move_callback {
                // Create a dictionary with move information
                let locals = PyDict::new(py);
                locals.set_item("move", uci_move)?;
                
                // Call the callback
                let _ = callback.call1(py, (locals,));
            }
            Ok(())
        })
    }

    /// Private method to notify the Python callback about game end
    fn notify_game_end(&self, result: &str, reason: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            if let Some(callback) = &self.on_game_end_callback {
                // Create a dictionary with game end information
                let locals = PyDict::new(py);
                locals.set_item("result", result)?;
                locals.set_item("reason", reason)?;
                locals.set_item("final_position", self.engine.get_fen())?;
                
                // Call the callback
                let _ = callback.call1(py, (locals,));
            }
            Ok(())
        })
    }

    /// Private method to notify the Python callback about position evaluation
    fn notify_eval(&self, eval_score: f32, depth: i32) -> PyResult<()> {
        Python::with_gil(|py| {
            if let Some(callback) = &self.on_eval_callback {
                // Create a dictionary with evaluation information
                let locals = PyDict::new(py);
                locals.set_item("score", eval_score)?;
                locals.set_item("depth", depth)?;
                locals.set_item("position", self.engine.get_fen())?;
                
                // Call the callback
                let _ = callback.call1(py, (locals,));
            }
            Ok(())
        })
    }
}

/// Chess engine Python module
#[pymodule]
fn chess_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChessEngine>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
} 
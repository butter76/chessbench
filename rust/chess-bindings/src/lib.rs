use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{IntoPyArray, PyArray1};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use parking_lot::Mutex;
use async_channel::{bounded, Sender, Receiver};
use once_cell::sync::OnceCell;
use std::num::NonZeroUsize;

// Import error module and ChessEngine
mod error;
mod engine;
use crate::engine::ChessEngine;

// Global runtime accessible via once_cell
static RUNTIME: OnceCell<Runtime> = OnceCell::new();

// Initialize runtime at module load
fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .worker_threads(std::thread::available_parallelism().map(NonZeroUsize::get).unwrap_or(4))
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime")
    })
}

// Commands that can be sent to an engine task
enum EngineCommand {
    MakeMove(String),
    GetFen,
    GetLegalMoves,
    EvaluatePosition(f32, i32), // score, depth
    GameEnd(String, String),    // result, reason
    Stop,
}

// Responses from engine tasks
enum EngineResponse {
    MoveMade(Result<(), PyErr>),
    Fen(String),
    LegalMoves(Vec<String>),
    Stopped,
    Error(PyErr),
}

// A handle to an engine task
struct EngineHandle {
    id: usize,
    command_tx: Sender<EngineCommand>,
    response_rx: Receiver<EngineResponse>,
    _task: JoinHandle<()>,
}

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

// Engine Manager to handle multiple engine instances
#[pyclass]
struct PyEngineManager {
    engines: Arc<Mutex<HashMap<usize, EngineHandle>>>,
    semaphore: Arc<Semaphore>,
    next_id: AtomicUsize,
}

#[pymethods]
impl PyEngineManager {
    #[new]
    fn new(max_concurrent: Option<usize>) -> Self {
        // Initialize the tokio runtime if not already done
        let _ = get_runtime();
        
        let max = max_concurrent.unwrap_or_else(|| {
            std::thread::available_parallelism().map(NonZeroUsize::get).unwrap_or(4)
        });
        
        Self {
            engines: Arc::new(Mutex::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(max)),
            next_id: AtomicUsize::new(1),
        }
    }
    
    /// Create a new engine instance and return its ID
    fn create_engine(
        &self, 
        fen: Option<String>,
        on_move_callback: Option<Py<PyAny>>,
        on_game_end_callback: Option<Py<PyAny>>,
        on_eval_callback: Option<Py<PyAny>>
    ) -> PyResult<usize> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        
        // Create channels for communication
        let (command_tx, command_rx) = bounded(32);
        let (response_tx, response_rx) = bounded(32);
        
        // Create the engine
        let mut engine = match fen {
            Some(fen_str) => PyChessEngine::from_fen(&fen_str)?,
            _ => PyChessEngine::new(),
        };
        
        // Register callbacks
        if let Some(callback) = on_move_callback {
            engine.register_on_move_callback(callback)?;
        }
        
        if let Some(callback) = on_game_end_callback {
            engine.register_on_game_end_callback(callback)?;
        }
        
        if let Some(callback) = on_eval_callback {
            engine.register_on_eval_callback(callback)?;
        }
        
        // Wrap engine in thread-safe containers
        let engine = Arc::new(Mutex::new(engine));
        let semaphore = self.semaphore.clone();
        
        // Spawn the engine task
        let task = get_runtime().spawn(async move {
            // Acquire permit or wait
            let permit = match semaphore.acquire().await {
                Ok(p) => p,
                Err(_) => return, // Runtime is shutting down
            };
            
            // Run the engine task
            run_engine_task(engine, command_rx, response_tx).await;
            
            // Release permit when done
            drop(permit);
        });
        
        // Store the handle
        let handle = EngineHandle {
            id,
            command_tx,
            response_rx,
            _task: task,
        };
        
        self.engines.lock().insert(id, handle);
        
        Ok(id)
    }
    
    /// Make a move on the specified engine instance
    fn make_move(&self, engine_id: usize, uci_move: String) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        // Send the command
        let tx = handle.command_tx.clone();
        let rx = handle.response_rx.clone();
        
        // Drop the lock before awaiting
        drop(engines);
        
        // Execute the command in the runtime
        let result = get_runtime().spawn(async move {
            // Send command
            tx.send(EngineCommand::MakeMove(uci_move)).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send command"))?;
            
            // Wait for response
            match rx.recv().await {
                Ok(EngineResponse::MoveMade(result)) => result,
                Ok(EngineResponse::Error(err)) => Err(err),
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Unexpected response")),
                Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine task has stopped")),
            }
        });
        
        Ok(())
    }
    
    /// Get the FEN string from a specific engine
    fn get_fen(&self, engine_id: usize) -> PyResult<String> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        let rx = handle.response_rx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(EngineCommand::GetFen).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send command"))?;
            
            match rx.recv().await {
                Ok(EngineResponse::Fen(fen)) => Ok(fen),
                Ok(EngineResponse::Error(err)) => Err(err),
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Unexpected response")),
                Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine task has stopped")),
            }
        })
    }
    
    /// Get legal moves from a specific engine
    fn get_legal_moves(&self, engine_id: usize) -> PyResult<Vec<String>> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        let rx = handle.response_rx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(EngineCommand::GetLegalMoves).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send command"))?;
            
            match rx.recv().await {
                Ok(EngineResponse::LegalMoves(moves)) => Ok(moves),
                Ok(EngineResponse::Error(err)) => Err(err),
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Unexpected response")),
                Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine task has stopped")),
            }
        })
    }
    
    /// Stop a specific engine
    fn stop_engine(&self, engine_id: usize) -> PyResult<()> {
        let mut engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        
        // Remove from map
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(EngineCommand::Stop).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send stop command"))?;
            
            Ok(())
        })
    }
    
    /// Trigger the evaluation callback
    fn notify_eval(&self, engine_id: usize, score: f32, depth: i32) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(EngineCommand::EvaluatePosition(score, depth)).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send eval command"))
        })
    }
    
    /// Trigger the game end callback
    fn notify_game_end(&self, engine_id: usize, result: String, reason: String) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(EngineCommand::GameEnd(result, reason)).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send game end command"))
        })
    }

    /// Get the number of active engines
    fn active_engines(&self) -> usize {
        self.engines.lock().len()
    }
}

// Implementation of the engine task
async fn run_engine_task(
    engine: Arc<Mutex<PyChessEngine>>,
    command_rx: Receiver<EngineCommand>,
    response_tx: Sender<EngineResponse>,
) {
    while let Ok(command) = command_rx.recv().await {
        match command {
            EngineCommand::MakeMove(uci_move) => {
                let result = tokio::task::spawn_blocking({
                    let engine = engine.clone();
                    move || {
                        let mut engine = engine.lock();
                        engine.make_move(&uci_move)
                    }
                }).await.unwrap_or_else(|e| {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Task panicked: {}", e)
                    ))
                });
                
                let _ = response_tx.send(EngineResponse::MoveMade(result)).await;
            },
            
            EngineCommand::GetFen => {
                let fen = tokio::task::spawn_blocking({
                    let engine = engine.clone();
                    move || {
                        let engine = engine.lock();
                        engine.get_fen()
                    }
                }).await.unwrap_or_else(|_| String::from("error"));
                
                let _ = response_tx.send(EngineResponse::Fen(fen)).await;
            },
            
            EngineCommand::GetLegalMoves => {
                let moves = tokio::task::spawn_blocking({
                    let engine = engine.clone();
                    move || {
                        let engine = engine.lock();
                        engine.get_legal_moves()
                    }
                }).await.unwrap_or_else(|_| Vec::new());
                
                let _ = response_tx.send(EngineResponse::LegalMoves(moves)).await;
            },
            
            EngineCommand::EvaluatePosition(score, depth) => {
                let _ = tokio::task::spawn_blocking({
                    let engine = engine.clone();
                    move || {
                        let engine = engine.lock();
                        let _ = engine.notify_eval(score, depth);
                    }
                }).await;
                
                // No response needed for notifications
            },
            
            EngineCommand::GameEnd(result, reason) => {
                let _ = tokio::task::spawn_blocking({
                    let engine = engine.clone();
                    move || {
                        let engine = engine.lock();
                        let _ = engine.notify_game_end(&result, &reason);
                    }
                }).await;
                
                // No response needed for notifications
            },
            
            EngineCommand::Stop => {
                let _ = response_tx.send(EngineResponse::Stopped).await;
                break;
            },
        }
    }
}

/// Chess engine Python module
#[pymodule]
fn chess_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChessEngine>()?;
    m.add_class::<PyEngineManager>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
} 
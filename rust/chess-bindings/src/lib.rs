use pyo3::prelude::*;
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
use crate::engine::AlphaBetaEngine;
use crate::engine::SearchThread;
use crate::engine::{EvalRequest, EvalResponse};
use crate::error::ChessError;
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
enum SearchThreadCommand {
    IsRunning,
    IsEvaluating,
    IsWaiting,
    ReceiveMove(String),
    ReceiveEval(EvalResponse),
    ReceiveFen(String),
    Start,
    Stop,
}

// Responses from engine tasks
enum ThreadResponse {
    Running(bool),
    Evaluating(bool),
    Waiting(bool),
    Stopped,
    Error(PyErr),
}

// A handle to an engine task
struct ThreadHandle {
    id: usize,
    command_tx: Sender<SearchThreadCommand>,
    response_rx: Receiver<ThreadResponse>,
    _task: JoinHandle<()>,
}

// Engine Manager to handle multiple engine instances
#[pyclass]
struct ThreadManager {
    engines: Arc<Mutex<HashMap<usize, ThreadHandle>>>,
    next_id: AtomicUsize,
}

#[pymethods]
impl ThreadManager {
    #[new]
    fn new() -> Self {
        // Initialize the tokio runtime if not already done
        let _ = get_runtime();
        
        Self {
            engines: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicUsize::new(1),
        }
    }
    
    /// Create a new engine instance and return its ID
    fn create_thread(
        &self, 
        fen: Option<String>,
        on_move_callback: Option<Py<PyAny>>,
        on_fen_callback: Option<Py<PyAny>>,
        on_eval_callback: Option<Py<PyAny>>
    ) -> PyResult<usize> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        
        // Create channels for communication
        let (command_tx, command_rx) = bounded(32);
        let (response_tx, response_rx) = bounded(32);
        
        // Create the engine
        let mut engine = AlphaBetaEngine::new(fen, 1)?;
        
        // Register callbacks
        if let Some(callback) = on_move_callback {
            engine.register_on_move_callback(callback)?;
        }
        
        if let Some(callback) = on_fen_callback {
            engine.register_on_fen_callback(callback)?;
        }
        
        if let Some(callback) = on_eval_callback {
            engine.register_on_eval_callback(callback)?;
        }
        
        // Wrap engine in thread-safe containers
        let engine: Arc<Mutex<Box<dyn SearchThread>>> = Arc::new(Mutex::new(Box::new(engine)));
        
        // Spawn the engine task
        let task = get_runtime().spawn(async move {
            // Run the engine task directly without semaphore
            run_engine_task(engine, command_rx, response_tx).await;
        });
        
        // Store the handle
        let handle = ThreadHandle {
            id,
            command_tx,
            response_rx,
            _task: task,
        };
        
        self.engines.lock().insert(id, handle);
        
        Ok(id)
    }
    
    /// Start a specific thread
    fn start_thread(&self, engine_id: usize) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        
        // Release the lock
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::Start).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send start command"))?;
            
            Ok(())
        })
    }
    
    /// Stop a specific thread
    fn stop_thread(&self, engine_id: usize) -> PyResult<()> {
        let mut engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        
        // Remove from map
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::Stop).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send stop command"))?;
            
            Ok(())
        })
    }

    /// Get the number of active engines
    fn active_engines(&self) -> usize {
        self.engines.lock().len()
    }

    /// Check if the thread is running
    fn is_running(&self, engine_id: usize) -> PyResult<bool> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        let rx = handle.response_rx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::IsRunning).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send command"))?;
            
            match rx.recv().await {
                Ok(ThreadResponse::Running(is_running)) => Ok(is_running),
                Ok(ThreadResponse::Error(err)) => Err(err),
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Unexpected response")),
                Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine task has stopped")),
            }
        })
    }
    
    /// Check if the thread is evaluating
    fn is_evaluating(&self, engine_id: usize) -> PyResult<bool> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        let rx = handle.response_rx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::IsEvaluating).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send command"))?;
            
            match rx.recv().await {
                Ok(ThreadResponse::Evaluating(is_evaluating)) => Ok(is_evaluating),
                Ok(ThreadResponse::Error(err)) => Err(err),
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Unexpected response")),
                Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine task has stopped")),
            }
        })
    }
    
    /// Check if the thread is waiting
    fn is_waiting(&self, engine_id: usize) -> PyResult<bool> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        let rx = handle.response_rx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::IsWaiting).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send command"))?;
            
            match rx.recv().await {
                Ok(ThreadResponse::Waiting(is_waiting)) => Ok(is_waiting),
                Ok(ThreadResponse::Error(err)) => Err(err),
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Unexpected response")),
                Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine task has stopped")),
            }
        })
    }
    
    /// Send a move to the thread
    fn receive_move(&self, engine_id: usize, uci_move: String) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::ReceiveMove(uci_move)).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send move command"))
        })
    }
    
    /// Send an evaluation to the thread
    fn receive_eval(&self, engine_id: usize, value: f32, policy: Vec<(String, f32)>) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        
        drop(engines);
        
        let eval = EvalResponse {
            value,
            policy,
        };
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::ReceiveEval(eval)).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send eval command"))
        })
    }
    
    /// Send a FEN string to the thread
    fn receive_fen(&self, engine_id: usize, fen: String) -> PyResult<()> {
        let engines = self.engines.lock();
        let handle = engines.get(&engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Engine ID {} not found", engine_id))
        })?;
        
        let tx = handle.command_tx.clone();
        
        drop(engines);
        
        get_runtime().block_on(async move {
            tx.send(SearchThreadCommand::ReceiveFen(fen)).await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to send fen command"))
        })
    }
}

// Implementation of the engine task
async fn run_engine_task(
    engine: Arc<Mutex<Box<dyn SearchThread>>>,
    command_rx: Receiver<SearchThreadCommand>,
    response_tx: Sender<ThreadResponse>,
) {
    while let Ok(command) = command_rx.recv().await {
        match command {
            SearchThreadCommand::IsRunning => {
                let engine_clone = engine.clone();
                let is_running = tokio::task::spawn_blocking(move || {
                    let engine = engine_clone.lock();
                    engine.is_running()
                }).await.unwrap_or(false); // Default to false on error
                
                let _ = response_tx.send(ThreadResponse::Running(is_running)).await;
            },
            
            SearchThreadCommand::IsEvaluating => {
                let engine_clone = engine.clone();
                let is_evaluating = tokio::task::spawn_blocking(move || {
                    let engine = engine_clone.lock();
                    engine.is_evaluating()
                }).await.unwrap_or(false); // Default to false on error
                
                let _ = response_tx.send(ThreadResponse::Evaluating(is_evaluating)).await;
            },
            
            SearchThreadCommand::IsWaiting => {
                let engine_clone = engine.clone();
                let is_waiting = tokio::task::spawn_blocking(move || {
                    let engine = engine_clone.lock();
                    engine.is_waiting()
                }).await.unwrap_or(false); // Default to false on error
                
                let _ = response_tx.send(ThreadResponse::Waiting(is_waiting)).await;
            },
            
            SearchThreadCommand::ReceiveMove(uci_move) => {
                let engine_clone = engine.clone();
                let _ = tokio::spawn(async move {
                    let mut engine = engine_clone.lock();
                    engine.receive_move(&uci_move)
                });
            },
            
            SearchThreadCommand::ReceiveEval(eval) => {
                let engine_clone = engine.clone();
                let _ = tokio::spawn(async move {
                    let mut engine = engine_clone.lock();
                    engine.receive_eval(eval)
                });
            },
            
            SearchThreadCommand::ReceiveFen(fen) => {
                let engine_clone = engine.clone();
                let _ = tokio::spawn(async move {
                    let mut engine = engine_clone.lock();
                    engine.receive_fen(&fen)
                });
            },
            
            SearchThreadCommand::Start => {
                let engine_clone = engine.clone();
                let _ = tokio::spawn(async move {
                    let mut engine = engine_clone.lock();
                    engine.start()
                });
            },
            
            SearchThreadCommand::Stop => {
                let engine_clone = engine.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut engine = engine_clone.lock();
                    engine.stop()
                }).await.unwrap_or_else(|e| {
                    Err(ChessError::RuntimeError(
                        format!("Task panicked: {}", e)
                    ))
                });
                
                if let Err(err) = result {
                    let _ = response_tx.send(ThreadResponse::Error(
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
                    )).await;
                } else {
                    let _ = response_tx.send(ThreadResponse::Stopped).await;
                }
            },
        }
    }
}

/// Chess engine Python module
#[pymodule]
fn chess_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ThreadManager>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
} 
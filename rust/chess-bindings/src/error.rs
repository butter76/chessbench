use shakmaty::{Chess, PlayError, PositionError, uci::{ParseUciMoveError, IllegalUciMoveError}};
use shakmaty::fen::ParseFenError;
use thiserror::Error;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Custom error type for chess operations
#[derive(Error, Debug)]
pub enum ChessError {
    #[error("Position error: {0}")]
    PositionError(#[from] PositionError<Chess>),

    #[error("FEN parsing error: {0}")]
    FenParseError(#[from] ParseFenError),
    
    #[error("Play error: {0}")]
    PlayError(#[from] PlayError<Chess>),

    #[error("UCI parsing error: {0}")]
    UciParseError(#[from] ParseUciMoveError),

    #[error("UCI move error: {0}")]
    UciMoveError(#[from] IllegalUciMoveError),
} 

// Implement automatic conversion from ChessError to PyErr
impl From<ChessError> for PyErr {
    fn from(err: ChessError) -> PyErr {
        // Convert ChessError to PyValueError with the error message
        PyValueError::new_err(err.to_string())
    }
}
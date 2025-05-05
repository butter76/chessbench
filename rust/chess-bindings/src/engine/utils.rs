use shakmaty::{Chess, fen::Fen, CastlingMode};
use crate::error::ChessError;

pub fn new_board(fen: Option<String>) -> Result<Chess, ChessError> {
    Ok(Chess::default())
}

/// Create a chess engine from FEN string
pub fn new_board_from_fen(fen: &str) -> Result<Chess, ChessError> {
    let fen = Fen::from_ascii(fen.as_bytes())?;
    let position = fen.into_position::<Chess>(CastlingMode::Standard)?;
    Ok(position)
}

pub fn new_board_from_fen_opt(fen: Option<String>) -> Result<Chess, ChessError> {
    match fen {
        Some(fen) => new_board_from_fen(&fen),
        _ => new_board(None),
    }
}

use shakmaty::{Chess, Position, Move, CastlingMode, EnPassantMode, uci::UciMove, fen::Fen};

use crate::error::ChessError;

/// A simple Rust chess engine implementation
pub struct ChessEngine {
    position: Chess,
}

impl ChessEngine {
    /// Create a new chess engine with the starting position
    pub fn new() -> Self {
        Self {
            position: Chess::default(),
        }
    }

    /// Create a chess engine from FEN string
    pub fn from_fen(fen: &str) -> Result<Self, ChessError> {
        let fen = Fen::from_ascii(fen.as_bytes())?;
        let position = fen.into_position::<Chess>(CastlingMode::Standard)?;
        Ok(Self { position })
    }

    /// Get legal moves for the current position
    pub fn get_legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves().to_vec()
    }

    /// Make a move on the board
    pub fn make_move(&mut self, chess_move: &Move) -> Result<(), ChessError> {
        self.position = self.position.clone().play(chess_move)?;
        Ok(())
    }

    pub fn convert_uci_to_move(&self, uci_move: &str) -> Result<Move, ChessError> {
        let chess_move = UciMove::from_ascii(uci_move.as_bytes())?;
        Ok(chess_move.to_move(&self.position)?)
    }

    /// Get current FEN representation
    pub fn get_fen(&self) -> String {
        Fen::from_position(self.position.clone(), EnPassantMode::Legal).to_string()
    }
}
#pragma once

#include <array>
#include <cctype>
#include <cstdint>

#include "chess.hpp"

namespace engine_tokenizer {

// Token indices must match Python order in src/tokenizer.py
// [ '0','1','2','3','4','5','6','7','8','9',
//   'p','b','n','r','c','k','q',
//   'P','B','N','R','C','Q','K',
//   'x','.' ]
inline std::uint8_t charToTokenIndex(char c) {
    if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
    switch (c) {
        case 'p': return 10;
        case 'b': return 11;
        case 'n': return 12;
        case 'r': return 13;
        case 'c': return 14;
        case 'k': return 15;
        case 'q': return 16;
        case 'P': return 17;
        case 'B': return 18;
        case 'N': return 19;
        case 'R': return 20;
        case 'C': return 21;
        case 'Q': return 22;
        case 'K': return 23;
        case 'x': return 24;
        case '.': return 25;
        default:  return 25;  // fallback to '.' for safety
    }
}

// Returns a 68-length tokenization analogous to Python's tokenizer.tokenize(fen, useRule50=False)
inline std::array<std::uint8_t, 68> tokenizeBoard(const chess::Board &board) {
    // Build a 64-char board in FEN rank order: a8..h8, a7..h7, ..., a1..h1
    std::array<char, 64> fenOrder{};
    for (int i = 0; i < 64; ++i) fenOrder[i] = '.';

    for (int fenIndex = 0; fenIndex < 64; ++fenIndex) {
        int fileIndex = fenIndex % 8;              // a..h => 0..7
        int rankIndex = 7 - (fenIndex / 8);        // 7..0 maps to rank 8..1
        chess::Square sq(fileIndex + rankIndex * 8);
        const auto piece = board.at(sq);
        char c = static_cast<std::string>(piece)[0];
        fenOrder[fenIndex] = c;  // '.' for empty, else piece char
    }

    // Replace rooks with castling rights by 'C'/'c' (like Python does for KQkq on a/h files)
    const auto rights = board.castlingRights();

    auto set_castling_marker = [&](chess::Color color, chess::Board::CastlingRights::Side side) {
        if (!rights.has(color, side)) return;
        const auto rook_file = rights.getRookFile(color, side);
        if (rook_file == chess::File::NO_FILE) return;

        int fileIndex = static_cast<int>(rook_file);
        int rankIndex = (color == chess::Color::WHITE) ? 0 : 7;  // rank 1 or 8
        int fenIdx    = (7 - rankIndex) * 8 + fileIndex;        // convert to FEN-order index

        char &cell = fenOrder[fenIdx];
        if (color == chess::Color::WHITE) {
            if (cell == 'R') cell = 'C';
        } else {
            if (cell == 'r') cell = 'c';
        }
    };

    set_castling_marker(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE);
    set_castling_marker(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    set_castling_marker(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE);
    set_castling_marker(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);

    // If black to move, flip ranks and swap case
    const bool flip = (board.sideToMove() == chess::Color::BLACK);
    std::array<char, 64> oriented{};

    if (flip) {
        for (int r = 0; r < 8; ++r) {
            // destination chunk r gets source chunk from (7 - r)
            int srcStart = (7 - r) * 8;
            int dstStart = r * 8;
            for (int f = 0; f < 8; ++f) {
                char c = fenOrder[srcStart + f];
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    c = std::islower(static_cast<unsigned char>(c)) ? static_cast<char>(std::toupper(static_cast<unsigned char>(c)))
                                                                    : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
                oriented[dstStart + f] = c;
            }
        }
    } else {
        for (int i = 0; i < 64; ++i) oriented[i] = fenOrder[i];
    }

    // Mark en-passant square with 'x' (after orientation), if present
    const auto epSq = board.enpassantSq();
    if (epSq != chess::Square::NO_SQ) {
        int fileIndex = static_cast<int>(epSq.file());
        int rankIndex = static_cast<int>(epSq.rank());
        int idx = flip ? (rankIndex * 8 + fileIndex) : ((7 - rankIndex) * 8 + fileIndex);
        if (idx >= 0 && idx < 64 && oriented[idx] == '.') oriented[idx] = 'x';
    }

    // Build final 68-length token array: 64 board + '.' + '0..'
    std::array<std::uint8_t, 68> tokens{};
    int t = 0;
    for (int i = 0; i < 64; ++i) tokens[t++] = charToTokenIndex(oriented[i]);
    tokens[t++] = charToTokenIndex('.');           // sentinel
    tokens[t++] = charToTokenIndex('0');           // useRule50 is false => always '0'
    tokens[t++] = charToTokenIndex('.');
    tokens[t++] = charToTokenIndex('.');

    return tokens;
}

}  // namespace engine_tokenizer



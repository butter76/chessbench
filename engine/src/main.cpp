#include "chess.hpp"
#include <iostream>

int main() {
    // Start from the standard initial position
    chess::Board board;

    // Generate all legal moves from the current position
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    std::cout << "Legal moves from the initial position: " << moves.size() << "\n";

    // Show a sample of the legal moves in UCI format
    std::size_t shown = 0;
    for (const auto &mv : moves) {
        std::cout << chess::uci::moveToUci(mv) << '\n';
        if (++shown >= 20) {
            break; // keep output short
        }
    }

    return 0;
}


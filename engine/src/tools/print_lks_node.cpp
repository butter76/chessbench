#include "chess.hpp"
#include "search/lks_search.hpp"
#include "time/fixed_time.hpp"
#include "options.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: print_lks_node <FEN...>\n";
        std::cerr << "Example: print_lks_node 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'\n";
        return 1;
    }

    // Join all args as a FEN string to allow spaces without quoting
    std::ostringstream oss;
    for (int i = 1; i < argc; ++i) {
        if (i > 1) oss << ' ';
        oss << argv[i];
    }
    const std::string fen = oss.str();

    try {
        engine::Options options;
        engine::FixedTime time_handler(50);
        engine::LksSearch search(options, &time_handler);

        // Set board to FEN
        search.getBoard().setFen(fen);

        // Create node (await)
        engine::LKSNode node = cppcoro::sync_wait(search.create_node(search.getBoard()));

        // Print
        std::cout << "FEN: " << fen << '\n';
        std::cout << "terminal: " << (node.terminal ? "true" : "false") << '\n';
        std::cout << "value: " << node.value << '\n';
        std::cout << "U: " << node.U << '\n';
        std::cout << "policy_size: " << node.policy.size() << '\n';
        const std::size_t max_show = 20;
        std::size_t shown = std::min(max_show, node.policy.size());
        for (std::size_t i = 0; i < shown; ++i) {
            const auto &e = node.policy[i];
            std::cout << i << ": "
                      << chess::uci::moveToUci(e.move) << "  p=" << e.policy
                      << "  U=" << e.U << "  Q=" << e.Q << '\n';
        }
        if (node.policy.size() > shown) {
            std::cout << "... (" << (node.policy.size() - shown) << " more)" << '\n';
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 2;
    }

    return 0;
}



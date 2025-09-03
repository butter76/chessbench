#include "chess.hpp"
#include "options.hpp"
#include "search/search_algo.hpp"
#include "search/random_search.hpp"
#include "search/fixed_depth_search.hpp"
#include "search/lks_search.hpp"
#include "time/uci_time.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_map>
#include <cctype>
#include <thread>

namespace {

struct PositionCache {
    bool has_base = false;
    bool base_is_startpos = false;
    std::string base_fen;
    std::vector<std::string> moves;
};

PositionCache g_pos_cache;

void send_id() {
    std::cout << "id name CatGPT" << '\n';
    std::cout << "id author Nikhil Reddy (butter)" << '\n';
    // Advertise configurable options
    std::cout << "option name Network type string default ./p2.plan" << '\n';
    // Threads option (default to hardware concurrency if available)
    unsigned int hc = std::thread::hardware_concurrency();
    if (hc == 0u) hc = 32u;
    std::cout << "option name Threads type spin default " << hc << " min 1 max 512" << '\n';
    // Syzygy tablebase root path
    std::cout << "option name SyzygyPath type string default ../syzygy_tables/3-4-5/" << '\n';
}

std::vector<std::string> split(const std::string &line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}

void apply_moves(engine::SearchAlgo &search, const std::vector<std::string> &moves_tokens, std::size_t start_index) {
    for (std::size_t i = start_index; i < moves_tokens.size(); ++i) {
        const std::string &uci_move = moves_tokens[i];
        search.makemove(uci_move);
    }
}

void set_position(engine::SearchAlgo &search, const std::string &cmd) {
    // cmd starts with "position"
    // Supported:
    // - position startpos [moves <m1> <m2> ...]
    // - position fen <FEN (6 fields)> [moves <m1> <m2> ...]
    const std::vector<std::string> tokens = split(cmd);
    if (tokens.size() < 2) return;

    std::size_t idx = 1; // token after "position"

    // Parse base
    bool new_is_startpos = false;
    std::string new_fen;
    if (tokens[idx] == "startpos") {
        new_is_startpos = true;
        ++idx;
    } else if (tokens[idx] == "fen") {
        ++idx;
        int fields_needed = 6;
        while (idx < tokens.size() && tokens[idx] != "moves" && fields_needed > 0) {
            if (!new_fen.empty()) new_fen.push_back(' ');
            new_fen += tokens[idx++];
            --fields_needed;
        }
    }

    // Parse move list (if any)
    std::vector<std::string> new_moves;
    if (idx < tokens.size() && tokens[idx] == "moves") {
        ++idx;
        for (std::size_t i = idx; i < tokens.size(); ++i) new_moves.push_back(tokens[i]);
    }

    // Determine if we can incrementally apply only the suffix
    const bool base_matches = g_pos_cache.has_base &&
        ((new_is_startpos && g_pos_cache.base_is_startpos) ||
         (!new_is_startpos && !g_pos_cache.base_is_startpos && new_fen == g_pos_cache.base_fen));

    bool applied_incremental = false;
    if (base_matches && new_moves.size() >= g_pos_cache.moves.size()) {
        // Check prefix equality
        bool prefix_ok = true;
        for (std::size_t i = 0; i < g_pos_cache.moves.size(); ++i) {
            if (g_pos_cache.moves[i] != new_moves[i]) { prefix_ok = false; break; }
        }
        if (prefix_ok) {
            // Apply only the remaining moves
            if (new_moves.size() > g_pos_cache.moves.size()) {
                // Build token list to reuse apply_moves
                std::vector<std::string> suffix_tokens;
                suffix_tokens.reserve(new_moves.size() - g_pos_cache.moves.size());
                for (std::size_t i = g_pos_cache.moves.size(); i < new_moves.size(); ++i) {
                    suffix_tokens.push_back(new_moves[i]);
                }
                // Reuse apply_moves signature: needs tokens with moves only and start_index 0
                apply_moves(search, suffix_tokens, 0);
            }
            applied_incremental = true;
        }
    }

    if (!applied_incremental) {
        // Reinitialize to base and apply all moves
        search.reset();
        if (!new_fen.empty()) {
            search.getBoard().setFen(new_fen);
        }
        if (!new_moves.empty()) {
            // Build tokens array containing exactly the moves
            apply_moves(search, new_moves, 0);
        }
    }

    // Update cache
    g_pos_cache.has_base = true;
    g_pos_cache.base_is_startpos = new_is_startpos;
    g_pos_cache.base_fen = new_fen;
    g_pos_cache.moves = std::move(new_moves);
}

engine::Limits parse_go_limits(const std::string &cmd) {
    engine::Limits limits;
    const std::vector<std::string> tokens = split(cmd);
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        const std::string &t = tokens[i];
        auto read_ull = [&](unsigned long long &out) {
            if (i + 1 < tokens.size()) {
                try { out = std::stoull(tokens[++i]); } catch (...) {}
            }
        };
        auto is_keyword = [&](const std::string &s) {
            return s == "searchmoves" || s == "ponder" || s == "wtime" || s == "btime" ||
                   s == "winc" || s == "binc" || s == "movestogo" || s == "depth" ||
                   s == "nodes" || s == "mate" || s == "movetime" || s == "infinite";
        };

        if (t == "searchmoves") {
            // collect UCI moves until next keyword or end
            while (i + 1 < tokens.size() && !is_keyword(tokens[i + 1])) {
                limits.searchmoves.push_back(tokens[++i]);
            }
        } else if (t == "ponder") {
            // ignored for now
        } else if (t == "wtime") {
            read_ull(limits.wtime_ms);
        } else if (t == "btime") {
            read_ull(limits.btime_ms);
        } else if (t == "winc") {
            read_ull(limits.winc_ms);
        } else if (t == "binc") {
            read_ull(limits.binc_ms);
        } else if (t == "movestogo") {
            unsigned long long tmp = 0; read_ull(tmp); limits.movestogo = static_cast<int>(tmp);
        } else if (t == "nodes") {
            read_ull(limits.nodes);
        } else if (t == "movetime") {
            read_ull(limits.movetime_ms);
        } else if (t == "infinite") {
            limits.infinite = true;
        } else if (t == "depth") {
            unsigned long long tmp = 0; read_ull(tmp); limits.depth = static_cast<int>(tmp);
        } else {
            // mate, etc. ignored in this minimal engine
        }
    }
    return limits;
}

} // namespace

int main(int argc, char **argv) {
    engine::Options options;

    // Parse command-line options: --option key=value or -o key=value
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto set_kv = [&](const std::string &kv) {
            auto pos = kv.find('=');
            if (pos == std::string::npos) {
                options.set(kv, "");
            } else {
                options.set(kv.substr(0, pos), kv.substr(pos + 1));
            }
        };
        if (arg == "--option" || arg == "-o") {
            if (i + 1 < argc) {
                set_kv(argv[++i]);
            }
        } else if (arg.rfind("--option=", 0) == 0) {
            set_kv(arg.substr(std::string("--option=").size()));
        } else if (arg.rfind("-o=", 0) == 0) {
            set_kv(arg.substr(3));
        }
    }

    // Instantiate search with UCI time handler (obeys go movetime/wtime/btime, etc.)
    engine::UciTimeHandler time_handler;
    engine::LksSearch search(options, &time_handler);
    std::jthread search_thread; // background search thread

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "uci") {
            send_id();
            std::cout << "uciok" << '\n' << std::flush;
        } else if (line == "isready") {
            // Initialize TensorRT now that options could have been set
            search.initialize_trt();
            std::cout << "readyok" << '\n' << std::flush;
        } else if (line.rfind("setoption", 0) == 0) {
            // setoption name <id> [value <x>]
            const std::vector<std::string> tokens = split(line);
            std::size_t name_index = std::string::npos;
            std::size_t value_index = std::string::npos;
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "name" && name_index == std::string::npos) name_index = i + 1;
                if (tokens[i] == "value" && value_index == std::string::npos) value_index = i + 1;
            }
            if (name_index != std::string::npos) {
                std::string name;
                if (value_index == std::string::npos) {
                    // name = tokens[name_index..end)
                    for (std::size_t i = name_index; i < tokens.size(); ++i) {
                        if (!name.empty()) name.push_back(' ');
                        name += tokens[i];
                    }
                    options.set(name, "");
                } else {
                    // name = tokens[name_index..value_index-1]
                    for (std::size_t i = name_index; i < value_index && i < tokens.size(); ++i) {
                        if (!name.empty()) name.push_back(' ');
                        name += tokens[i];
                    }
                    std::string value;
                    for (std::size_t i = value_index; i < tokens.size(); ++i) {
                        if (!value.empty()) value.push_back(' ');
                        value += tokens[i];
                    }
                    options.set(name, value);
                }
            }
        } else if (line == "ucinewgame") {
            if (search_thread.joinable()) search_thread.join();
            search.reset();
            g_pos_cache = PositionCache{};
        } else if (line.rfind("position", 0) == 0) {
            if (search_thread.joinable()) search_thread.join();
            set_position(search, line);
        } else if (line.rfind("go", 0) == 0) {
            const engine::Limits limits = parse_go_limits(line);
            if (search_thread.joinable()) search_thread.join();
            // Run search on a separate thread so main loop remains responsive
            search_thread = std::jthread([&search, limits](std::stop_token){
                const std::string best = search.searchBestMove(limits);
                std::cout << "bestmove " << best << '\n' << std::flush;
            });
        } else if (line == "stop") {
            search.stop();
            if (search_thread.joinable()) search_thread.join();
        } else if (line == "quit") {
            search.stop();
            if (search_thread.joinable()) search_thread.join();
            break;
        } else if (line == "options") {
            // Non-standard debug helper: print all options as key=value (keys are stored lowercase)
            options.forEach([](const std::string &k, const std::string &v) {
                std::cout << k << '=' << v << '\n';
            });
        } else if (line == "print") {
            // helper for debugging
            std::cout << search.getBoard() << '\n' << std::flush;
        }
    }

    return 0;
}


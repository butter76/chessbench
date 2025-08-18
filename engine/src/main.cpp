#include "chess.hpp"
#include "search/search_algo.hpp"
#include "search/random_search.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_map>
#include <cctype>

namespace {

class Options {
public:
    void set(const std::string &key, const std::string &value) {
        storage_[normalizeKey(key)] = value;
    }

    std::string get(const std::string &key, const std::string &defaultValue) const {
        const std::string nk = normalizeKey(key);
        auto it = storage_.find(nk);
        return it == storage_.end() ? defaultValue : it->second;
    }

    void clear() { storage_.clear(); }

private:
    static std::string normalizeKey(const std::string &in) {
        std::string out;
        out.reserve(in.size());
        for (char c : in) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        return out;
    }

    std::unordered_map<std::string, std::string> storage_;
};

struct PositionCache {
    bool has_base = false;
    bool base_is_startpos = false;
    std::string base_fen;
    std::vector<std::string> moves;
};

PositionCache g_pos_cache;

void reset(chess::Board &board) {
    board = chess::Board();
}

void send_id() {
    std::cout << "id name SearchlessRandom" << '\n';
    std::cout << "id author Searchless" << '\n';
}

std::vector<std::string> split(const std::string &line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}

void apply_moves(chess::Board &board, const std::vector<std::string> &moves_tokens, std::size_t start_index) {
    for (std::size_t i = start_index; i < moves_tokens.size(); ++i) {
        const std::string &uci_move = moves_tokens[i];
        const chess::Move move = chess::uci::uciToMove(board, uci_move);
        if (move != chess::Move::NO_MOVE) {
            board.makeMove(move);
        } else {
            // Fallback: try to match against generated legal moves
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board);
            bool applied = false;
            for (const auto &m : legal) {
                if (chess::uci::moveToUci(m) == uci_move) {
                    board.makeMove(m);
                    applied = true;
                    break;
                }
            }
            (void)applied; // ignore if not applied; invalid move is skipped
        }
    }
}

void set_position(chess::Board &board, const std::string &cmd) {
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
                apply_moves(board, suffix_tokens, 0);
            }
            applied_incremental = true;
        }
    }

    if (!applied_incremental) {
        // Reinitialize to base and apply all moves
        if (new_is_startpos) {
            reset(board);
        } else if (!new_fen.empty()) {
            board.setFen(new_fen);
        }
        if (!new_moves.empty()) {
            // Build tokens array containing exactly the moves
            apply_moves(board, new_moves, 0);
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
        } else {
            // depth, mate, etc. ignored in this minimal engine
        }
    }
    return limits;
}

} // namespace

int main() {
    chess::Board board; // default startpos
    Options options;

    // Instantiate search
    engine::RandomSearch search;

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "uci") {
            send_id();
            std::cout << "uciok" << '\n' << std::flush;
        } else if (line == "isready") {
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
            reset(board);
        } else if (line.rfind("position", 0) == 0) {
            set_position(board, line);
        } else if (line.rfind("go", 0) == 0) {
            const engine::Limits limits = parse_go_limits(line);
            const std::string best = search.searchBestMove(board, limits);
            std::cout << "bestmove " << best << '\n' << std::flush;
        } else if (line == "stop") {
            // no background search; nothing to do
        } else if (line == "quit") {
            break;
        } else if (line == "print") {
            // helper for debugging
            std::cout << board << '\n' << std::flush;
        }
    }

    return 0;
}


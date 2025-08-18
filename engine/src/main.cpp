#include "chess.hpp"
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
    if (tokens[idx] == "startpos") {
        reset(board);
        ++idx;
    } else if (tokens[idx] == "fen") {
        // collect next up-to-6 fields as fen (some FENs include a move clock field)
        std::string fen;
        ++idx;
        // standard FEN has 6 fields
        int fields_needed = 6;
        while (idx < tokens.size() && tokens[idx] != "moves" && fields_needed > 0) {
            if (!fen.empty()) fen.push_back(' ');
            fen += tokens[idx++];
            --fields_needed;
        }
        if (!fen.empty()) {
            board.setFen(fen);
        }
    }

    // apply moves if provided
    if (idx < tokens.size() && tokens[idx] == "moves") {
        ++idx;
        apply_moves(board, tokens, idx);
    }
}

std::string pick_random_legal_move_uci(chess::Board &board, std::mt19937_64 &rng) {
    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);
    if (legal.empty()) return "0000"; // no legal move
    std::uniform_int_distribution<std::size_t> dist(0, legal.size() - 1);
    const chess::Move mv = legal[dist(rng)];
    const std::string uci = chess::uci::moveToUci(mv);
    board.makeMove(mv);
    return uci;
}

} // namespace

int main() {
    chess::Board board; // default startpos
    Options options;

    // RNG for random moves
    std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));

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
            // instantly play a random legal move
            const std::string best = pick_random_legal_move_uci(board, rng);
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


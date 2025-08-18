#pragma once

#include <string>
#include <unordered_map>
#include <cctype>
#include <functional>

namespace engine {

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

    void forEach(const std::function<void(const std::string&, const std::string&)> &fn) const {
        for (const auto &kv : storage_) fn(kv.first, kv.second);
    }

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

} // namespace engine



#pragma once

#include <unistd.h>

#include <algorithm>
#include <cmath>

#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "incgraph/util.h"

namespace incgraph {

inline std::vector<float> gather_rows(const FMatrix &m, const uint32_t *ids, size_t n) {
    std::vector<float> out(n * m.d);
    for (size_t i = 0; i < n; ++i) std::copy_n(m.row(ids[i]), m.d, out.data() + i * m.d);
    return out;
}

struct Step {
    std::string op;
    size_t off = 0, n = 0;
};
struct Runbook {
    std::string name, dir;
    JValue meta;
    std::vector<Step> steps;
    std::vector<uint32_t> ids;
    std::vector<size_t> search_steps;
    const uint32_t *ids_of(const Step &s) const { return ids.data() + s.off; }
};

inline Runbook load_runbook(const Dataset &ds, const std::string &name) {
    Runbook rb;
    rb.name = name;
    rb.dir = ds.dir + "/runbooks/" + name;
    rb.meta = read_json(rb.dir + "/steps.json");
    rb.ids = read_u32(rb.dir + "/ids.u32");
    for (auto &s : rb.meta["steps"].a) {
        Step st;
        st.op = s["op"].str();
        if (const JValue *o = s.find("off")) st.off = static_cast<size_t>(o->i64());
        if (const JValue *o = s.find("n")) st.n = static_cast<size_t>(o->i64());
        if (st.op == "search") rb.search_steps.push_back(rb.steps.size());
        rb.steps.push_back(st);
    }
    return rb;
}

inline std::set<size_t> pick_steps(const std::vector<size_t> &search_steps, const std::string &spec) {
    std::set<size_t> out;
    if (spec.empty() || spec == "all") {
        out.insert(search_steps.begin(), search_steps.end());
        return out;
    }
    if (spec == "none") return out;
    size_t p = 0;
    while (p <= spec.size()) {
        size_t q = spec.find(',', p);
        if (q == std::string::npos) q = spec.size();
        if (q > p) {
            long o = std::stol(spec.substr(p, q - p));
            if (o < 0) o += static_cast<long>(search_steps.size());
            if (o < 0 || o >= static_cast<long>(search_steps.size())) throw std::out_of_range("search ordinal " + spec);
            out.insert(search_steps[o]);
        }
        p = q + 1;
    }
    return out;
}

inline std::vector<uint32_t> live_after(const Runbook &rb, size_t n_base, size_t upto) {
    std::vector<uint8_t> live(n_base, 0);
    for (size_t i = 0; i < upto; ++i) {
        const Step &s = rb.steps[i];
        if (s.op == "build" || s.op == "insert")
            for (size_t j = 0; j < s.n; ++j) live[rb.ids[s.off + j]] = 1;
        else if (s.op == "delete")
            for (size_t j = 0; j < s.n; ++j) live[rb.ids[s.off + j]] = 0;
    }
    std::vector<uint32_t> out;
    for (size_t i = 0; i < n_base; ++i)
        if (live[i]) out.push_back(static_cast<uint32_t>(i));
    return out;
}

inline Truth runbook_gt(const Runbook &rb, size_t step, const FMatrix &base, const FMatrix &queries, size_t k,
                        Metric metric, int threads) {
    const std::string dir = rb.dir + "/gt";
    const std::string path = dir + "/step" + std::to_string(step) + "-nq" + std::to_string(queries.n) + "-k" +
                             std::to_string(k) + ".bin";
    if (file_exists(path)) return read_truth(path);
    make_dirs(dir);
    std::vector<uint32_t> live = live_after(rb, base.n, step);
    Truth t = knn(base, queries, k, metric, &live, threads);
    const std::string tmp = path + ".tmp" + std::to_string(::getpid());
    write_truth(tmp, t);
    std::rename(tmp.c_str(), path.c_str());
    return t;
}

inline void write_edges(const std::string &path, const std::vector<std::pair<uint32_t, uint32_t>> &e) {
    std::vector<uint32_t> flat;
    flat.reserve(e.size() * 2);
    for (auto &x : e) {
        flat.push_back(x.first);
        flat.push_back(x.second);
    }
    write_u32(path, flat);
}
inline std::vector<std::pair<uint32_t, uint32_t>> read_edges(const std::string &path) {
    std::vector<uint32_t> flat = read_u32(path);
    std::vector<std::pair<uint32_t, uint32_t>> e(flat.size() / 2);
    for (size_t i = 0; i < e.size(); ++i) e[i] = {flat[2 * i], flat[2 * i + 1]};
    return e;
}

class LineWriter {
  public:
    explicit LineWriter(const std::string &path) {
        make_dirs(dirname_of(path));
        f_ = std::fopen(path.c_str(), "w");
        if (!f_) throw std::runtime_error("cannot write " + path);
    }
    ~LineWriter() {
        if (f_) std::fclose(f_);
    }
    void line(const std::string &s, bool echo = true) {
        std::fputs(s.c_str(), f_);
        std::fputc('\n', f_);
        std::fflush(f_);
        if (echo) {
            std::puts(s.c_str());
            std::fflush(stdout);
        }
    }

  private:
    FILE *f_ = nullptr;
};

inline std::string hex64(uint64_t h) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h));
    return b;
}

}

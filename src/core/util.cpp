#include "incgraph/util.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace incgraph {

namespace {
inline float dotf(const float *a, const float *b, size_t d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}
inline float l2f(const float *a, const float *b, size_t d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}
[[noreturn]] void fail(const std::string &msg) { throw std::runtime_error(msg); }
}

std::pair<size_t, size_t> fbin_shape(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    uint32_t n, d;
    in.read(reinterpret_cast<char *>(&n), 4);
    in.read(reinterpret_cast<char *>(&d), 4);
    if (!in) fail("bad fbin header: " + path);
    return {n, d};
}

FMatrix read_fbin(const std::string &path, size_t start, size_t count) {
    auto [n, d] = fbin_shape(path);
    if (start > n) fail("read_fbin: start beyond end of " + path);
    count = std::min(count, n - start);
    FMatrix m;
    m.n = count;
    m.d = d;
    m.v.resize(count * d);
    std::ifstream in(path, std::ios::binary);
    in.seekg(8 + static_cast<std::streamoff>(start) * d * 4);
    in.read(reinterpret_cast<char *>(m.v.data()), static_cast<std::streamsize>(count * d * 4));
    if (!in) fail("truncated fbin: " + path);
    return m;
}

void write_fbin(const std::string &path, const float *data, size_t n, size_t d) {
    std::ofstream out(path, std::ios::binary);
    if (!out) fail("cannot write " + path);
    uint32_t nn = static_cast<uint32_t>(n), dd = static_cast<uint32_t>(d);
    out.write(reinterpret_cast<const char *>(&nn), 4);
    out.write(reinterpret_cast<const char *>(&dd), 4);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n * d * 4));
    if (!out) fail("write failed: " + path);
}

std::vector<uint32_t> read_u32(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    uint64_t n;
    in.read(reinterpret_cast<char *>(&n), 8);
    std::vector<uint32_t> v(n);
    in.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(n * 4));
    if (!in) fail("truncated u32 file: " + path);
    return v;
}

void write_u32(const std::string &path, const std::vector<uint32_t> &v) {
    std::ofstream out(path, std::ios::binary);
    if (!out) fail("cannot write " + path);
    uint64_t n = v.size();
    out.write(reinterpret_cast<const char *>(&n), 8);
    out.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(n * 4));
    if (!out) fail("write failed: " + path);
}

Truth read_truth(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    int32_t n, k;
    in.read(reinterpret_cast<char *>(&n), 4);
    in.read(reinterpret_cast<char *>(&k), 4);
    Truth t;
    t.n = n;
    t.k = k;
    t.ids.resize(t.n * t.k);
    t.dists.resize(t.n * t.k);
    in.read(reinterpret_cast<char *>(t.ids.data()), static_cast<std::streamsize>(t.ids.size() * 4));
    in.read(reinterpret_cast<char *>(t.dists.data()), static_cast<std::streamsize>(t.dists.size() * 4));
    if (!in) fail("truncated truthset: " + path);
    return t;
}

void write_truth(const std::string &path, const Truth &t) {
    std::ofstream out(path, std::ios::binary);
    if (!out) fail("cannot write " + path);
    int32_t n = static_cast<int32_t>(t.n), k = static_cast<int32_t>(t.k);
    out.write(reinterpret_cast<const char *>(&n), 4);
    out.write(reinterpret_cast<const char *>(&k), 4);
    out.write(reinterpret_cast<const char *>(t.ids.data()), static_cast<std::streamsize>(t.ids.size() * 4));
    out.write(reinterpret_cast<const char *>(t.dists.data()), static_cast<std::streamsize>(t.dists.size() * 4));
    if (!out) fail("write failed: " + path);
}

bool file_exists(const std::string &path) { return std::filesystem::exists(path); }
void make_dirs(const std::string &path) {
    if (!path.empty()) std::filesystem::create_directories(path);
}
std::string dirname_of(const std::string &path) { return std::filesystem::path(path).parent_path().string(); }

const JValue *JValue::find(const std::string &k) const {
    if (t != Obj) return nullptr;
    for (auto &kv : o)
        if (kv.first == k) return &kv.second;
    return nullptr;
}
const JValue &JValue::operator[](const std::string &k) const {
    const JValue *v = find(k);
    if (!v) fail("json: missing key '" + k + "'");
    return *v;
}
double JValue::num() const {
    if (t == Num) return n;
    if (t == Bool) return b ? 1 : 0;
    fail("json: not a number");
}
const std::string &JValue::str() const {
    if (t != Str) fail("json: not a string");
    return s;
}

namespace {
struct JParser {
    const std::string &x;
    size_t p = 0;
    void ws() {
        while (p < x.size() && std::isspace(static_cast<unsigned char>(x[p]))) ++p;
    }
    char peek() {
        ws();
        if (p >= x.size()) fail("json: unexpected end");
        return x[p];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("json: expected '") + c + "' at " + std::to_string(p));
        ++p;
    }
    std::string parse_str() {
        expect('"');
        std::string out;
        while (p < x.size() && x[p] != '"') {
            char c = x[p++];
            if (c == '\\') {
                char e = x[p++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        unsigned cp = std::stoul(x.substr(p, 4), nullptr, 16);
                        p += 4;
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: out += e;
                }
            } else {
                out += c;
            }
        }
        ++p;
        return out;
    }
    JValue parse() {
        JValue v;
        char c = peek();
        if (c == '{') {
            ++p;
            v.t = JValue::Obj;
            if (peek() == '}') {
                ++p;
                return v;
            }
            while (true) {
                std::string k = parse_str();
                expect(':');
                v.o.emplace_back(k, parse());
                if (peek() == ',') {
                    ++p;
                    continue;
                }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            ++p;
            v.t = JValue::Arr;
            if (peek() == ']') {
                ++p;
                return v;
            }
            while (true) {
                v.a.push_back(parse());
                if (peek() == ',') {
                    ++p;
                    continue;
                }
                expect(']');
                return v;
            }
        }
        if (c == '"') {
            v.t = JValue::Str;
            v.s = parse_str();
            return v;
        }
        if (x.compare(p, 4, "true") == 0) {
            p += 4;
            v.t = JValue::Bool;
            v.b = true;
            return v;
        }
        if (x.compare(p, 5, "false") == 0) {
            p += 5;
            v.t = JValue::Bool;
            return v;
        }
        if (x.compare(p, 4, "null") == 0) {
            p += 4;
            return v;
        }
        size_t used = 0;
        v.t = JValue::Num;
        v.n = std::stod(x.substr(p, 40), &used);
        p += used;
        return v;
    }
};
}

JValue parse_json(const std::string &text) {
    JParser ps{text};
    return ps.parse();
}

JValue read_json(const std::string &path) {
    std::ifstream in(path);
    if (!in) fail("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parse_json(ss.str());
}

std::string json_escape(const std::string &s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if (c == '\n') {
            o += "\\n";
        } else {
            o += c;
        }
    }
    return o + "\"";
}

std::string fmt_num(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    if (std::fabs(v) < 1e15 && v == std::floor(v))
        std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(v));
    else
        std::snprintf(buf, sizeof buf, "%.7g", v);
    return buf;
}

void JObj::key(const std::string &k) {
    if (!body_.empty()) body_ += ",";
    body_ += json_escape(k) + ":";
}
JObj &JObj::add(const std::string &k, double v) {
    key(k);
    body_ += fmt_num(v);
    return *this;
}
JObj &JObj::add(const std::string &k, int64_t v) {
    key(k);
    body_ += std::to_string(v);
    return *this;
}
JObj &JObj::add(const std::string &k, const std::string &v) {
    key(k);
    body_ += json_escape(v);
    return *this;
}
JObj &JObj::raw(const std::string &k, const std::string &json) {
    key(k);
    body_ += json;
    return *this;
}
JObj &JObj::merge(const std::string &fields) {
    if (fields.empty()) return *this;
    if (!body_.empty()) body_ += ",";
    body_ += fields;
    return *this;
}

Args::Args(int argc, char **argv) {
    for (int i = 0; i < argc; ++i) cmdline += (i ? " " : "") + std::string(argv[i]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-p") {
            if (i + 1 >= argc) fail("-p needs key=value");
            std::string s = argv[++i];
            auto eq = s.find('=');
            if (eq == std::string::npos) fail("-p needs key=value, got " + s);
            kv[s.substr(0, eq)] = s.substr(eq + 1);
        } else if (a.rfind("--", 0) == 0) {
            auto eq = a.find('=');
            if (eq != std::string::npos) {
                m_[a.substr(0, eq)] = a.substr(eq + 1);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0 &&
                       std::string(argv[i + 1]) != "-p") {
                m_[a] = argv[++i];
            } else {
                m_[a] = "1";
            }
        } else {
            fail("unexpected argument: " + a);
        }
    }
}
std::string Args::str(const std::string &k, const std::string &def) const {
    auto it = m_.find(k);
    return it == m_.end() ? def : it->second;
}
std::string Args::req(const std::string &k) const {
    auto it = m_.find(k);
    if (it == m_.end()) fail("missing required argument " + k);
    return it->second;
}
long Args::i(const std::string &k, long def) const { return has(k) ? std::stol(str(k, "")) : def; }
double Args::f(const std::string &k, double def) const { return has(k) ? std::stod(str(k, "")) : def; }
std::vector<long> Args::ints(const std::string &k, const std::string &def) const {
    std::vector<long> out;
    std::stringstream ss(str(k, def));
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stol(tok));
    return out;
}

uint64_t Rng::next() {
    uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t Rng::below(uint64_t n) {
    if (n == 0) return 0;
    const uint64_t lim = UINT64_MAX - UINT64_MAX % n;
    uint64_t x;
    do x = next();
    while (x >= lim);
    return x % n;
}
double Rng::uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
std::vector<uint32_t> Rng::permutation(size_t n) {
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint32_t>(i);
    shuffle(v);
    return v;
}
std::vector<uint32_t> Rng::sample(size_t n, size_t k) {
    k = std::min(k, n);
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint32_t>(i);
    for (size_t i = 0; i < k; ++i) std::swap(v[i], v[i + below(n - i)]);
    v.resize(k);
    std::sort(v.begin(), v.end());
    return v;
}

Dataset load_dataset(const std::string &name_or_dir) {
    std::string dir = name_or_dir;
    if (!file_exists(dir + "/dataset.json")) dir = "data/" + name_or_dir;
    if (!file_exists(dir + "/dataset.json")) fail("dataset not found: " + name_or_dir);
    JValue j = read_json(dir + "/dataset.json");
    Dataset ds;
    ds.dir = dir;
    ds.name = j["name"].str();
    ds.dim = static_cast<size_t>(j["dim"].i64());
    const std::string &m = j["metric"].str();
    if (m == "ip")
        ds.metric = Metric::IP;
    else if (m == "l2")
        ds.metric = Metric::L2;
    else
        fail("unknown metric " + m);
    const JValue &f = j["files"];
    ds.base_file = dir + "/" + f["base"].str();
    ds.query_file = dir + "/" + f["query"].str();
    ds.learn_file = dir + "/" + f["learn"].str();
    for (auto [key, out] : {std::make_pair("base_groups", &ds.base_groups),
                            std::make_pair("query_groups", &ds.query_groups)}) {
        if (const JValue *g = j.find(key))
            for (auto &kv : g->o)
                out->push_back({kv.first, static_cast<size_t>(kv.second[0].i64()),
                                static_cast<size_t>(kv.second[1].i64())});
    }
    return ds;
}

FMatrix test_queries(const Dataset &ds, size_t nq, std::vector<Range> *groups) {
    if (groups) groups->clear();
    if (ds.query_groups.empty()) return read_fbin(ds.query_file, 0, nq);
    FMatrix all = read_fbin(ds.query_file);
    const size_t per = nq / ds.query_groups.size();
    FMatrix out;
    out.d = all.d;
    for (auto &g : ds.query_groups) {
        size_t m = std::min(per, g.hi - g.lo);
        if (groups) groups->push_back({g.name, out.n, out.n + m});
        out.v.insert(out.v.end(), all.row(g.lo), all.row(g.lo) + m * all.d);
        out.n += m;
    }
    return out;
}

Truth knn(const FMatrix &base, const FMatrix &q, size_t k, Metric m, const std::vector<uint32_t> *ids,
          int threads) {
    if (base.d != q.d) fail("knn: dimension mismatch");
    const size_t d = q.d, nb = ids ? ids->size() : base.n;
    if (k > nb) fail("knn: k larger than the number of base points");
    Truth t;
    t.n = q.n;
    t.k = k;
    t.ids.resize(q.n * k);
    t.dists.resize(q.n * k);
    constexpr size_t QB = 32;
    const int64_t nblk = static_cast<int64_t>((q.n + QB - 1) / QB);
    const int nt = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int64_t b = 0; b < nblk; ++b) {
        const size_t q0 = static_cast<size_t>(b) * QB, q1 = std::min(q.n, q0 + QB), nqb = q1 - q0;
        std::vector<std::vector<std::pair<float, uint32_t>>> heaps(nqb);
        for (auto &h : heaps) h.reserve(k + 1);
        for (size_t j = 0; j < nb; ++j) {
            const uint32_t id = ids ? (*ids)[j] : static_cast<uint32_t>(j);
            const float *x = base.row(id);
            for (size_t r = 0; r < nqb; ++r) {
                const float dd = m == Metric::IP ? -dotf(q.row(q0 + r), x, d) : l2f(q.row(q0 + r), x, d);
                auto &h = heaps[r];
                if (h.size() < k) {
                    h.emplace_back(dd, id);
                    std::push_heap(h.begin(), h.end());
                } else if (std::make_pair(dd, id) < h.front()) {
                    std::pop_heap(h.begin(), h.end());
                    h.back() = {dd, id};
                    std::push_heap(h.begin(), h.end());
                }
            }
        }
        for (size_t r = 0; r < nqb; ++r) {
            auto &h = heaps[r];
            std::sort_heap(h.begin(), h.end());
            for (size_t i = 0; i < k; ++i) {
                t.ids[(q0 + r) * k + i] = h[i].second;
                t.dists[(q0 + r) * k + i] = h[i].first;
            }
        }
    }
    return t;
}

std::vector<int> kmeans_assign(const float *x, size_t n, size_t d, int k, uint64_t seed, size_t sample, int niter) {
    Rng rng(seed);
    std::vector<uint32_t> smp = rng.sample(n, std::min(sample, n));
    const size_t ns = smp.size();
    if (ns < static_cast<size_t>(k)) fail("kmeans: fewer points than clusters");
    std::vector<float> C(static_cast<size_t>(k) * d);
    std::vector<uint32_t> init = rng.sample(ns, k);
    for (int c = 0; c < k; ++c) std::memcpy(&C[c * d], x + static_cast<size_t>(smp[init[c]]) * d, d * 4);
    auto nearest = [&](const float *v) {
        int best = 0;
        float bd = std::numeric_limits<float>::max();
        for (int c = 0; c < k; ++c) {
            float dd = l2f(v, &C[c * d], d);
            if (dd < bd) {
                bd = dd;
                best = c;
            }
        }
        return best;
    };
    std::vector<int> as(ns);
    std::vector<double> sum(static_cast<size_t>(k) * d);
    std::vector<size_t> cnt(k);
    for (int it = 0; it < niter; ++it) {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(ns); ++i) as[i] = nearest(x + static_cast<size_t>(smp[i]) * d);
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < ns; ++i) {
            const float *v = x + static_cast<size_t>(smp[i]) * d;
            double *s = &sum[as[i] * d];
            for (size_t j = 0; j < d; ++j) s[j] += v[j];
            ++cnt[as[i]];
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c] == 0) {
                std::memcpy(&C[c * d], x + static_cast<size_t>(smp[rng.below(ns)]) * d, d * 4);
                continue;
            }
            for (size_t j = 0; j < d; ++j) C[c * d + j] = static_cast<float>(sum[c * d + j] / cnt[c]);
        }
    }
    std::vector<int> out(n);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) out[i] = nearest(x + static_cast<size_t>(i) * d);
    return out;
}

double recall_at(const uint32_t *res, size_t nq, size_t k_res, const Truth &gt, size_t k, size_t gt_row0) {
    if (nq == 0) return 0;
    if (gt.k < k) fail("recall_at: ground truth has fewer than k neighbors");
    size_t hit = 0;
    for (size_t i = 0; i < nq; ++i) {
        const uint32_t *g = gt.row(gt_row0 + i);
        for (size_t j = 0; j < k; ++j) {
            const uint32_t r = res[i * k_res + j];
            for (size_t l = 0; l < k; ++l)
                if (g[l] == r) {
                    ++hit;
                    break;
                }
        }
    }
    return static_cast<double>(hit) / static_cast<double>(nq * k);
}

std::string join(const std::vector<std::string> &v, const std::string &sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
    return o;
}

}

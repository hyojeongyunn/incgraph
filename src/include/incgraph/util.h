#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "incgraph/index.h"

namespace incgraph {

struct FMatrix {
    size_t n = 0, d = 0;
    std::vector<float> v;
    const float *row(size_t i) const { return v.data() + i * d; }
    float *row(size_t i) { return v.data() + i * d; }
};
FMatrix read_fbin(const std::string &path, size_t start = 0, size_t count = SIZE_MAX);
void write_fbin(const std::string &path, const float *data, size_t n, size_t d);
std::pair<size_t, size_t> fbin_shape(const std::string &path);

std::vector<uint32_t> read_u32(const std::string &path);
void write_u32(const std::string &path, const std::vector<uint32_t> &v);

struct Truth {
    size_t n = 0, k = 0;
    std::vector<uint32_t> ids;
    std::vector<float> dists;
    const uint32_t *row(size_t i) const { return ids.data() + i * k; }
};
Truth read_truth(const std::string &path);
void write_truth(const std::string &path, const Truth &t);

bool file_exists(const std::string &path);
void make_dirs(const std::string &path);
std::string dirname_of(const std::string &path);

struct JValue {
    enum Type { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<JValue> a;
    std::vector<std::pair<std::string, JValue>> o;

    const JValue *find(const std::string &k) const;
    const JValue &operator[](const std::string &k) const;
    const JValue &operator[](size_t i) const { return a.at(i); }
    size_t size() const { return t == Arr ? a.size() : o.size(); }
    double num() const;
    int64_t i64() const { return static_cast<int64_t>(num()); }
    const std::string &str() const;
};
JValue parse_json(const std::string &text);
JValue read_json(const std::string &path);

class JObj {
  public:
    JObj &add(const std::string &k, double v);
    JObj &add(const std::string &k, int64_t v);
    JObj &add(const std::string &k, uint64_t v) { return add(k, static_cast<int64_t>(v)); }
    JObj &add(const std::string &k, int v) { return add(k, static_cast<int64_t>(v)); }
    JObj &add(const std::string &k, uint32_t v) { return add(k, static_cast<int64_t>(v)); }
    JObj &add(const std::string &k, const std::string &v);
    JObj &add(const std::string &k, const char *v) { return add(k, std::string(v)); }
    JObj &raw(const std::string &k, const std::string &json);
    JObj &merge(const std::string &fields);
    std::string fields() const { return body_; }
    std::string str() const { return "{" + body_ + "}"; }

  private:
    void key(const std::string &k);
    std::string body_;
};
std::string json_escape(const std::string &s);
std::string fmt_num(double v);

class Args {
  public:
    Args(int argc, char **argv);
    bool has(const std::string &k) const { return m_.count(k) > 0; }
    std::string str(const std::string &k, const std::string &def) const;
    std::string req(const std::string &k) const;
    long i(const std::string &k, long def) const;
    double f(const std::string &k, double def) const;
    std::vector<long> ints(const std::string &k, const std::string &def) const;
    std::map<std::string, std::string> kv;
    std::string cmdline;

  private:
    std::map<std::string, std::string> m_;
};

class Rng {
  public:
    explicit Rng(uint64_t seed) : s_(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
    uint64_t next();
    uint64_t below(uint64_t n);
    double uniform();
    template <typename T>
    void shuffle(std::vector<T> &v) {
        for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[below(i)]);
    }
    std::vector<uint32_t> permutation(size_t n);
    std::vector<uint32_t> sample(size_t n, size_t k);

  private:
    uint64_t s_;
};

struct Range {
    std::string name;
    size_t lo, hi;
};
struct Dataset {
    std::string name, dir;
    Metric metric = Metric::IP;
    size_t dim = 0;
    std::string base_file, query_file, learn_file;
    std::vector<Range> base_groups, query_groups;
};
Dataset load_dataset(const std::string &name_or_dir);
FMatrix test_queries(const Dataset &ds, size_t nq, std::vector<Range> *groups);

Truth knn(const FMatrix &base, const FMatrix &queries, size_t k, Metric m, const std::vector<uint32_t> *ids,
          int threads);
std::vector<int> kmeans_assign(const float *x, size_t n, size_t d, int k, uint64_t seed, size_t sample = 200000,
                               int niter = 20);

double recall_at(const uint32_t *res, size_t nq, size_t k_res, const Truth &gt, size_t k, size_t gt_row0 = 0);

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::string join(const std::vector<std::string> &v, const std::string &sep);

}

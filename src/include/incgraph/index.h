#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace incgraph {

enum class Metric : int { L2 = 0, IP = 1 };

constexpr uint32_t ENTRY_ID = 0xFFFFFFFFu;

struct Params {
    uint32_t R = 64;
    uint32_t L = 100;
    float alpha = 1.2f;
    uint32_t max_candidates = 750;
    float slack = 1.3f;
    uint32_t R_extra = 0;
};

struct SearchStats {
    uint64_t dist_cmps = 0;
    uint32_t hops = 0;
};

struct Candidate {
    float dist;
    uint32_t id;
    bool expanded;
};

class CandidatePool {
  public:
    void reset(size_t capacity) {
        v_.clear();
        cap_ = capacity;
        cur_ = 0;
    }
    void insert(uint32_t id, float dist);
    bool has_unexpanded();
    uint32_t pop_closest_unexpanded();
    size_t size() const { return v_.size(); }
    const Candidate &operator[](size_t i) const { return v_[i]; }

  private:
    std::vector<Candidate> v_;
    size_t cap_ = 0;
    size_t cur_ = 0;
};

class Index {
  public:
    Index(size_t dim, Metric metric, size_t capacity, const Params &params);
    ~Index();
    Index(const Index &) = delete;
    Index &operator=(const Index &) = delete;

    void set_entry_point(const float *vec, bool reconnect);
    void build(const float *data, const uint32_t *labels, size_t n, int num_threads);
    void insert(const float *data, const uint32_t *labels, size_t n, int num_threads, float alpha_override);
    void insert_one(const float *vec, uint32_t label, float alpha_override, std::vector<uint32_t> *trace_labels,
                    std::vector<float> *trace_dists);
    size_t lazy_delete(const uint32_t *labels, size_t n);
    size_t consolidate(int num_threads);

    void search(const float *q, uint32_t k, uint32_t L, uint32_t *out_labels, float *out_dists, SearchStats *stats,
                const std::vector<uint32_t> *entry_labels = nullptr, int group = -1) const;
    void batch_search(const float *queries, size_t nq, uint32_t k, uint32_t L, uint32_t *out_labels,
                      float *out_dists, uint64_t *out_cmps, uint32_t *out_hops, int num_threads,
                      int group = -1) const;

    void set_group(const std::vector<uint32_t> &labels, uint8_t g);
    uint8_t group_of(uint32_t label) const;
    void search_trace(const float *q, uint32_t L, std::vector<uint32_t> &labels, std::vector<float> &dists,
                      SearchStats *stats) const;

    std::vector<uint32_t> neighbors(uint32_t label) const;
    std::vector<uint32_t> extra_neighbors(uint32_t label) const;
    void set_neighbors(uint32_t label, const std::vector<uint32_t> &nbr_labels);
    bool add_edge(uint32_t from, uint32_t to, bool allow_overflow);
    bool remove_edge(uint32_t from, uint32_t to);
    bool add_extra_edge(uint32_t from, uint32_t to);
    bool remove_extra_edge(uint32_t from, uint32_t to);
    size_t clear_extra_edges();
    size_t set_R_extra(uint32_t r_extra);
    std::vector<uint32_t> robust_prune(uint32_t label, const std::vector<uint32_t> &cand_labels, float alpha,
                                       uint32_t R) const;
    std::vector<uint32_t> in_neighbors(uint32_t label, bool include_extra) const;

    size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }
    const Params &params() const { return params_; }
    size_t capacity() const { return capacity_; }
    size_t size() const { return label_to_slot_.size(); }
    size_t num_deleted() const { return num_deleted_; }
    bool contains(uint32_t label) const { return label_to_slot_.count(label) > 0; }
    std::vector<uint32_t> labels() const;
    const float *vector(uint32_t label) const;
    float distance(uint32_t a, uint32_t b) const;
    float distance_to(const float *q, uint32_t label) const;
    std::pair<size_t, size_t> reachability(bool include_extra) const;
    void degree_stats(double &avg, uint32_t &max_deg, double &avg_extra, size_t &n_extra_edges) const;

    void save(const std::string &path) const;
    static std::unique_ptr<Index> load(const std::string &path);
    void save_graph(const std::string &path) const;
    static std::unique_ptr<Index> load_graph(const std::string &path, const float *base, size_t n_base);
    uint64_t graph_hash() const;
    std::vector<std::pair<uint32_t, uint32_t>> extra_edge_list() const;

  private:
    struct Scratch;
    enum : uint8_t { FREE = 0, ACTIVE = 1, DELETED = 2 };
    static constexpr uint32_t kEntry = 0;
    static constexpr size_t kNumLocks = 1 << 16;

    float dist_q(const float *q, uint32_t slot) const;
    float dist_ss(uint32_t a, uint32_t b) const;
    const float *vec_of(uint32_t slot) const { return data_.data() + (size_t)slot * dim_; }
    float *vec_of(uint32_t slot) { return data_.data() + (size_t)slot * dim_; }
    std::mutex &lock_of(uint32_t slot) const { return locks_[slot % kNumLocks]; }
    Scratch &scratch() const;

    uint32_t slot_of(uint32_t label) const;
    uint32_t label_of(uint32_t slot) const { return slot == kEntry ? ENTRY_ID : slot_label_[slot]; }
    uint32_t slot_of_any(uint32_t label) const { return label == ENTRY_ID ? kEntry : slot_of(label); }
    bool usable(uint32_t slot) const { return slot == kEntry || state_[slot] == ACTIVE; }

    void ensure_capacity(size_t needed_slots);
    uint32_t allocate_slot();
    void compute_medoid(const float *data, size_t n, std::vector<float> &out) const;

    void greedy_search(const float *q, uint32_t L, Scratch &s, bool lock_nbrs,
                       std::vector<std::pair<float, uint32_t>> *expanded, SearchStats *stats,
                       const std::vector<uint32_t> *seed_slots, int group = -1) const;
    void prune(uint32_t p, std::vector<std::pair<float, uint32_t>> &cands, float alpha, uint32_t R,
               std::vector<uint32_t> &out, Scratch &s) const;
    void insert_slot(uint32_t slot, float alpha, Scratch &s, std::vector<std::pair<float, uint32_t>> *trace);
    void inter_insert(uint32_t n, uint32_t p, float alpha, Scratch &s);

    size_t dim_;
    Metric metric_;
    size_t capacity_;
    Params params_;

    std::vector<float> data_;
    std::vector<std::vector<uint32_t>> nbrs_;
    std::vector<std::vector<uint32_t>> extra_;
    std::vector<uint8_t> state_;
    std::vector<uint8_t> group_;
    std::vector<uint32_t> slot_label_;
    std::unordered_map<uint32_t, uint32_t> label_to_slot_;
    std::vector<uint32_t> free_slots_;
    uint32_t next_slot_ = 1;
    size_t num_deleted_ = 0;
    bool entry_set_ = false;

    mutable std::unique_ptr<std::mutex[]> locks_;
    mutable std::vector<std::unique_ptr<Scratch>> scratch_;
};

}

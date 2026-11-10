#include "incgraph/index.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace incgraph {

namespace {

inline float l2sq(const float *a, const float *b, size_t d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

inline float dot(const float *a, const float *b, size_t d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

std::string label_str(uint32_t label) {
    std::ostringstream os;
    os << label;
    return os.str();
}

template <typename T>
void write_pod(std::ofstream &out, const T &v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
template <typename T>
void read_pod(std::ifstream &in, T &v) {
    in.read(reinterpret_cast<char *>(&v), sizeof(T));
    if (!in) throw std::runtime_error("incgraph: unexpected end of file");
}

}

void CandidatePool::insert(uint32_t id, float dist) {
    if (cap_ == 0) return;
    if (v_.size() == cap_ && dist >= v_.back().dist) return;
    auto it = std::lower_bound(v_.begin(), v_.end(), dist,
                               [](const Candidate &c, float d) { return c.dist < d; });
    size_t pos = static_cast<size_t>(it - v_.begin());
    if (v_.size() == cap_) v_.pop_back();
    v_.insert(v_.begin() + pos, Candidate{dist, id, false});
    if (pos < cur_) cur_ = pos;
}

bool CandidatePool::has_unexpanded() {
    while (cur_ < v_.size() && v_[cur_].expanded) ++cur_;
    return cur_ < v_.size();
}

uint32_t CandidatePool::pop_closest_unexpanded() {
    v_[cur_].expanded = true;
    return v_[cur_].id;
}

struct Index::Scratch {
    std::vector<uint32_t> visited;
    uint32_t epoch = 0;
    CandidatePool pool;
    std::vector<uint32_t> nbr_buf;
    std::vector<std::pair<float, uint32_t>> expanded;
    std::vector<std::pair<float, uint32_t>> cands;
    std::vector<float> occlude;
    std::vector<uint32_t> pruned;

    void new_epoch(size_t n) {
        if (visited.size() < n) {
            visited.assign(n, 0);
            epoch = 0;
        }
        if (++epoch == 0) {
            std::fill(visited.begin(), visited.end(), 0);
            epoch = 1;
        }
    }
    bool test_and_set(uint32_t id) {
        if (visited[id] == epoch) return true;
        visited[id] = epoch;
        return false;
    }
};

Index::Scratch &Index::scratch() const {
    int t = omp_get_thread_num();
    if (t < 0 || static_cast<size_t>(t) >= scratch_.size())
        throw std::runtime_error("incgraph: thread id exceeds scratch pool");
    return *scratch_[t];
}

Index::Index(size_t dim, Metric metric, size_t capacity, const Params &params)
    : dim_(dim), metric_(metric), capacity_(0), params_(params) {
    if (dim == 0) throw std::invalid_argument("incgraph: dim must be > 0");
    if (params_.R == 0 || params_.L == 0) throw std::invalid_argument("incgraph: R and L must be > 0");
    if (params_.alpha < 1.f) throw std::invalid_argument("incgraph: alpha must be >= 1");
    if (params_.slack < 1.f) params_.slack = 1.f;
    locks_.reset(new std::mutex[kNumLocks]);
    size_t nt = static_cast<size_t>(std::max(omp_get_max_threads(), omp_get_num_procs()));
    scratch_.resize(nt);
    for (auto &s : scratch_) s.reset(new Scratch());
    data_.assign(dim_, 0.f);
    nbrs_.resize(1);
    extra_.resize(1);
    state_.assign(1, ACTIVE);
    group_.assign(1, 0);
    slot_label_.assign(1, ENTRY_ID);
    ensure_capacity(std::max<size_t>(capacity, 1));
}

Index::~Index() = default;

void Index::ensure_capacity(size_t needed_slots) {
    if (needed_slots <= capacity_) return;
    size_t newcap = std::max(needed_slots, capacity_ == 0 ? needed_slots : capacity_ * 2);
    data_.resize((newcap + 1) * dim_, 0.f);
    nbrs_.resize(newcap + 1);
    extra_.resize(newcap + 1);
    state_.resize(newcap + 1, FREE);
    group_.resize(newcap + 1, 0);
    slot_label_.resize(newcap + 1, 0);
    capacity_ = newcap;
}

uint32_t Index::allocate_slot() {
    if (!free_slots_.empty()) {
        uint32_t s = free_slots_.back();
        free_slots_.pop_back();
        return s;
    }
    ensure_capacity(next_slot_);
    return next_slot_++;
}

uint32_t Index::slot_of(uint32_t label) const {
    auto it = label_to_slot_.find(label);
    if (it == label_to_slot_.end())
        throw std::out_of_range("incgraph: label " + label_str(label) + " is not an active point");
    return it->second;
}

float Index::dist_q(const float *q, uint32_t slot) const {
    const float *x = vec_of(slot);
    return metric_ == Metric::L2 ? l2sq(q, x, dim_) : -dot(q, x, dim_);
}

float Index::dist_ss(uint32_t a, uint32_t b) const { return dist_q(vec_of(a), b); }

float Index::distance(uint32_t a, uint32_t b) const { return dist_ss(slot_of_any(a), slot_of_any(b)); }

float Index::distance_to(const float *q, uint32_t label) const { return dist_q(q, slot_of_any(label)); }

const float *Index::vector(uint32_t label) const { return vec_of(slot_of_any(label)); }

void Index::compute_medoid(const float *data, size_t n, std::vector<float> &out) const {
    out.assign(dim_, 0.f);
    if (n == 0) return;
    std::vector<double> c(dim_, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < dim_; ++j) c[j] += data[i * dim_ + j];
    std::vector<float> cf(dim_);
    for (size_t j = 0; j < dim_; ++j) cf[j] = static_cast<float>(c[j] / static_cast<double>(n));
    std::vector<float> d(n);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) d[i] = l2sq(data + i * dim_, cf.data(), dim_);
    size_t best = static_cast<size_t>(std::min_element(d.begin(), d.end()) - d.begin());
    std::copy(data + best * dim_, data + (best + 1) * dim_, out.begin());
}

void Index::greedy_search(const float *q, uint32_t L, Scratch &s, bool lock_nbrs,
                          std::vector<std::pair<float, uint32_t>> *expanded, SearchStats *stats,
                          const std::vector<uint32_t> *seed_slots, int group) const {
    s.new_epoch(capacity_ + 1);
    s.pool.reset(L);
    uint64_t cmps = 0;
    uint32_t hops = 0;

    auto visit = [&](uint32_t id) {
        if (s.test_and_set(id)) return;
        if (id != kEntry && state_[id] == FREE) return;
        if (group >= 0 && id != kEntry && group_[id] != group) return;
        s.pool.insert(id, dist_q(q, id));
        ++cmps;
    };
    visit(kEntry);
    if (seed_slots)
        for (uint32_t sid : *seed_slots) visit(sid);

    while (s.pool.has_unexpanded()) {
        uint32_t cur = s.pool.pop_closest_unexpanded();
        ++hops;
        if (expanded) expanded->emplace_back(dist_q(q, cur), cur);
        if (lock_nbrs) {
            {
                std::lock_guard<std::mutex> g(lock_of(cur));
                s.nbr_buf = nbrs_[cur];
                s.nbr_buf.insert(s.nbr_buf.end(), extra_[cur].begin(), extra_[cur].end());
            }
            for (uint32_t n : s.nbr_buf) visit(n);
        } else {
            for (uint32_t n : nbrs_[cur]) visit(n);
            for (uint32_t n : extra_[cur]) visit(n);
        }
    }
    if (stats) {
        stats->dist_cmps += cmps;
        stats->hops += hops;
    }
}

void Index::search(const float *q, uint32_t k, uint32_t L, uint32_t *out_labels, float *out_dists,
                   SearchStats *stats, const std::vector<uint32_t> *entry_labels, int group) const {
    Scratch &s = scratch();
    std::vector<uint32_t> seeds;
    if (entry_labels) {
        for (uint32_t lab : *entry_labels) {
            auto it = label_to_slot_.find(lab);
            if (it != label_to_slot_.end()) seeds.push_back(it->second);
        }
    }
    greedy_search(q, std::max(L, k), s, false, nullptr, stats, entry_labels ? &seeds : nullptr, group);
    uint32_t found = 0;
    for (size_t i = 0; i < s.pool.size() && found < k; ++i) {
        uint32_t id = s.pool[i].id;
        if (id == kEntry || state_[id] != ACTIVE) continue;
        out_labels[found] = slot_label_[id];
        out_dists[found] = s.pool[i].dist;
        ++found;
    }
    for (; found < k; ++found) {
        out_labels[found] = ENTRY_ID;
        out_dists[found] = std::numeric_limits<float>::infinity();
    }
}

void Index::batch_search(const float *queries, size_t nq, uint32_t k, uint32_t L, uint32_t *out_labels,
                         float *out_dists, uint64_t *out_cmps, uint32_t *out_hops, int num_threads,
                         int group) const {
    int nt = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
    for (int64_t i = 0; i < static_cast<int64_t>(nq); ++i) {
        SearchStats st;
        search(queries + i * dim_, k, L, out_labels + i * k, out_dists + i * k, &st, nullptr, group);
        if (out_cmps) out_cmps[i] = st.dist_cmps;
        if (out_hops) out_hops[i] = st.hops;
    }
}

void Index::set_group(const std::vector<uint32_t> &labels, uint8_t g) {
    for (uint32_t l : labels) group_[slot_of(l)] = g;
}

uint8_t Index::group_of(uint32_t label) const { return group_[slot_of_any(label)]; }

void Index::search_trace(const float *q, uint32_t L, std::vector<uint32_t> &labels, std::vector<float> &dists,
                         SearchStats *stats) const {
    Scratch &s = scratch();
    s.expanded.clear();
    greedy_search(q, L, s, false, &s.expanded, stats, nullptr);
    labels.clear();
    dists.clear();
    for (auto &e : s.expanded) {
        labels.push_back(label_of(e.second));
        dists.push_back(e.first);
    }
}

void Index::prune(uint32_t p, std::vector<std::pair<float, uint32_t>> &cands, float alpha, uint32_t R,
                  std::vector<uint32_t> &out, Scratch &s) const {
    out.clear();
    if (cands.empty()) return;
    if (cands.size() > params_.max_candidates) cands.resize(params_.max_candidates);
    std::vector<float> &occ = s.occlude;
    occ.assign(cands.size(), 0.f);
    float cur_alpha = 1.f;
    while (cur_alpha <= alpha && out.size() < R) {
        const float eps = cur_alpha + 0.01f;
        for (size_t i = 0; i < cands.size() && out.size() < R; ++i) {
            if (occ[i] > cur_alpha) continue;
            occ[i] = std::numeric_limits<float>::max();
            const uint32_t id = cands[i].second;
            if (id != p) out.push_back(id);
            for (size_t j = i + 1; j < cands.size(); ++j) {
                if (occ[j] > alpha) continue;
                const float djk = dist_ss(cands[j].second, id);
                if (metric_ == Metric::L2) {
                    occ[j] = (djk == 0.f) ? std::numeric_limits<float>::max() : std::max(occ[j], cands[j].first / djk);
                } else {
                    const float x = -cands[j].first;
                    const float y = -djk;
                    if (y > cur_alpha * x) occ[j] = std::max(occ[j], eps);
                }
            }
        }
        cur_alpha *= 1.2f;
    }
}

std::vector<uint32_t> Index::robust_prune(uint32_t label, const std::vector<uint32_t> &cand_labels, float alpha,
                                          uint32_t R) const {
    const uint32_t p = slot_of_any(label);
    Scratch &s = scratch();
    s.new_epoch(capacity_ + 1);
    s.cands.clear();
    for (uint32_t lab : cand_labels) {
        auto it = lab == ENTRY_ID ? label_to_slot_.end() : label_to_slot_.find(lab);
        uint32_t c = lab == ENTRY_ID ? kEntry : (it == label_to_slot_.end() ? p : it->second);
        if (c == p || s.test_and_set(c)) continue;
        s.cands.emplace_back(dist_ss(p, c), c);
    }
    std::sort(s.cands.begin(), s.cands.end());
    std::vector<uint32_t> out;
    prune(p, s.cands, alpha, R, out, s);
    for (auto &x : out) x = label_of(x);
    return out;
}

void Index::insert_slot(uint32_t slot, float alpha, Scratch &s, std::vector<std::pair<float, uint32_t>> *trace) {
    const float *v = vec_of(slot);
    s.expanded.clear();
    greedy_search(v, params_.L, s, true, &s.expanded, nullptr, nullptr);
    s.cands.clear();
    for (auto &e : s.expanded)
        if (e.second != slot && usable(e.second)) s.cands.push_back(e);
    if (trace) {
        trace->clear();
        for (auto &e : s.expanded)
            if (e.second != slot) trace->push_back(e);
    }
    std::sort(s.cands.begin(), s.cands.end());
    prune(slot, s.cands, alpha, params_.R, s.pruned, s);
    std::vector<uint32_t> out = s.pruned;
    {
        std::lock_guard<std::mutex> g(lock_of(slot));
        nbrs_[slot] = out;
    }
    for (uint32_t n : out) inter_insert(n, slot, params_.alpha, s);
}

void Index::inter_insert(uint32_t n, uint32_t p, float alpha, Scratch &s) {
    const size_t slack_cap = static_cast<size_t>(params_.slack * static_cast<float>(params_.R));
    std::vector<uint32_t> copy;
    {
        std::lock_guard<std::mutex> g(lock_of(n));
        auto &lst = nbrs_[n];
        if (std::find(lst.begin(), lst.end(), p) != lst.end()) return;
        if (lst.size() < slack_cap) {
            lst.push_back(p);
            return;
        }
        copy = lst;
    }
    copy.push_back(p);
    s.cands.clear();
    for (uint32_t c : copy)
        if (c != n && usable(c)) s.cands.emplace_back(dist_ss(n, c), c);
    std::sort(s.cands.begin(), s.cands.end());
    prune(n, s.cands, alpha, params_.R, s.pruned, s);
    std::lock_guard<std::mutex> g(lock_of(n));
    nbrs_[n] = s.pruned;
}

void Index::set_entry_point(const float *vec, bool reconnect) {
    std::memcpy(vec_of(kEntry), vec, dim_ * sizeof(float));
    entry_set_ = true;
    if (reconnect && size() > 0) {
        Scratch &s = scratch();
        s.expanded.clear();
        greedy_search(vec, params_.L, s, false, &s.expanded, nullptr, nullptr);
        s.cands.clear();
        for (auto &e : s.expanded)
            if (e.second != kEntry && usable(e.second)) s.cands.push_back(e);
        std::sort(s.cands.begin(), s.cands.end());
        prune(kEntry, s.cands, params_.alpha, params_.R, s.pruned, s);
        nbrs_[kEntry] = s.pruned;
    }
}

void Index::build(const float *data, const uint32_t *labels, size_t n, int num_threads) {
    if (!label_to_slot_.empty() || num_deleted_ > 0)
        throw std::runtime_error("incgraph: build() requires an empty index (use insert() to add points)");
    std::vector<float> med;
    compute_medoid(data, n, med);
    set_entry_point(med.data(), false);
    const size_t warm = std::min<size_t>(n, 1000);
    insert(data, labels, warm, 1, -1.f);
    insert(data + warm * dim_, labels + warm, n - warm, num_threads, -1.f);
    int nt = num_threads > 0 ? num_threads : omp_get_max_threads();
    std::vector<uint32_t> over;
    for (uint32_t sl = 0; sl < next_slot_; ++sl)
        if (usable(sl) && nbrs_[sl].size() > params_.R) over.push_back(sl);
#pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
    for (int64_t i = 0; i < static_cast<int64_t>(over.size()); ++i) {
        uint32_t sl = over[i];
        Scratch &s = scratch();
        s.cands.clear();
        for (uint32_t c : nbrs_[sl])
            if (c != sl && usable(c)) s.cands.emplace_back(dist_ss(sl, c), c);
        std::sort(s.cands.begin(), s.cands.end());
        prune(sl, s.cands, params_.alpha, params_.R, s.pruned, s);
        nbrs_[sl] = s.pruned;
    }
}

void Index::insert(const float *data, const uint32_t *labels, size_t n, int num_threads, float alpha_override) {
    if (n == 0) return;
    {
        std::unordered_map<uint32_t, int> seen;
        seen.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) {
            if (labels[i] == ENTRY_ID) throw std::invalid_argument("incgraph: label 0xFFFFFFFF is reserved");
            if (label_to_slot_.count(labels[i]) || seen.count(labels[i]))
                throw std::invalid_argument("incgraph: duplicate label " + label_str(labels[i]));
            seen[labels[i]] = 1;
        }
    }
    if (!entry_set_) {
        std::vector<float> med;
        compute_medoid(data, n, med);
        set_entry_point(med.data(), false);
    }
    std::vector<uint32_t> slots(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t sl = allocate_slot();
        std::memcpy(vec_of(sl), data + i * dim_, dim_ * sizeof(float));
        nbrs_[sl].clear();
        extra_[sl].clear();
        slot_label_[sl] = labels[i];
        label_to_slot_[labels[i]] = sl;
        state_[sl] = ACTIVE;
        group_[sl] = 0;
        slots[i] = sl;
    }
    const float alpha = alpha_override > 0.f ? alpha_override : params_.alpha;
    int nt = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) insert_slot(slots[i], alpha, scratch(), nullptr);
}

void Index::insert_one(const float *vec, uint32_t label, float alpha_override, std::vector<uint32_t> *trace_labels,
                       std::vector<float> *trace_dists) {
    if (label == ENTRY_ID) throw std::invalid_argument("incgraph: label 0xFFFFFFFF is reserved");
    if (label_to_slot_.count(label)) throw std::invalid_argument("incgraph: duplicate label " + label_str(label));
    if (!entry_set_) set_entry_point(vec, false);
    uint32_t sl = allocate_slot();
    std::memcpy(vec_of(sl), vec, dim_ * sizeof(float));
    nbrs_[sl].clear();
    extra_[sl].clear();
    slot_label_[sl] = label;
    label_to_slot_[label] = sl;
    state_[sl] = ACTIVE;
    std::vector<std::pair<float, uint32_t>> trace;
    insert_slot(sl, alpha_override > 0.f ? alpha_override : params_.alpha, scratch(), &trace);
    if (trace_labels) {
        trace_labels->clear();
        for (auto &e : trace) trace_labels->push_back(label_of(e.second));
    }
    if (trace_dists) {
        trace_dists->clear();
        for (auto &e : trace) trace_dists->push_back(e.first);
    }
}

size_t Index::lazy_delete(const uint32_t *labels, size_t n) {
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        auto it = label_to_slot_.find(labels[i]);
        if (it == label_to_slot_.end()) continue;
        state_[it->second] = DELETED;
        label_to_slot_.erase(it);
        ++num_deleted_;
        ++cnt;
    }
    return cnt;
}

size_t Index::consolidate(int num_threads) {
    if (num_deleted_ == 0) return 0;
    std::vector<uint32_t> todo;
    todo.reserve(label_to_slot_.size() + 1);
    for (uint32_t sl = 0; sl < next_slot_; ++sl)
        if (usable(sl)) todo.push_back(sl);
    const uint32_t R = params_.R;
    int nt = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
    for (int64_t i = 0; i < static_cast<int64_t>(todo.size()); ++i) {
        const uint32_t p = todo[i];
        Scratch &s = scratch();
        auto &ex = extra_[p];
        if (!ex.empty())
            ex.erase(std::remove_if(ex.begin(), ex.end(), [&](uint32_t x) { return state_[x] == DELETED; }),
                     ex.end());
        auto &lst = nbrs_[p];
        bool has_del = false;
        for (uint32_t n : lst)
            if (state_[n] == DELETED) {
                has_del = true;
                break;
            }
        if (!has_del) continue;
        s.new_epoch(capacity_ + 1);
        s.cands.clear();
        auto add = [&](uint32_t c) {
            if (c == p || !usable(c) || s.test_and_set(c)) return;
            s.cands.emplace_back(dist_ss(p, c), c);
        };
        for (uint32_t n : lst) {
            if (state_[n] == DELETED) {
                for (uint32_t m : nbrs_[n]) add(m);
            } else {
                add(n);
            }
        }
        std::sort(s.cands.begin(), s.cands.end());
        if (s.cands.size() <= R) {
            lst.clear();
            for (auto &c : s.cands) lst.push_back(c.second);
        } else {
            prune(p, s.cands, params_.alpha, R, s.pruned, s);
            lst = s.pruned;
        }
    }
    size_t freed = 0;
    for (uint32_t sl = 1; sl < next_slot_; ++sl) {
        if (state_[sl] != DELETED) continue;
        std::vector<uint32_t>().swap(nbrs_[sl]);
        std::vector<uint32_t>().swap(extra_[sl]);
        state_[sl] = FREE;
        free_slots_.push_back(sl);
        ++freed;
    }
    num_deleted_ = 0;
    return freed;
}

std::vector<uint32_t> Index::neighbors(uint32_t label) const {
    std::vector<uint32_t> out;
    for (uint32_t n : nbrs_[slot_of_any(label)]) out.push_back(label_of(n));
    return out;
}

std::vector<uint32_t> Index::extra_neighbors(uint32_t label) const {
    std::vector<uint32_t> out;
    for (uint32_t n : extra_[slot_of_any(label)]) out.push_back(label_of(n));
    return out;
}

void Index::set_neighbors(uint32_t label, const std::vector<uint32_t> &nbr_labels) {
    const uint32_t p = slot_of_any(label);
    std::vector<uint32_t> lst;
    for (uint32_t lab : nbr_labels) {
        uint32_t c = slot_of_any(lab);
        if (c == p || std::find(lst.begin(), lst.end(), c) != lst.end()) continue;
        lst.push_back(c);
    }
    nbrs_[p] = std::move(lst);
}

bool Index::add_edge(uint32_t from, uint32_t to, bool allow_overflow) {
    const uint32_t a = slot_of_any(from), b = slot_of_any(to);
    if (a == b) return false;
    auto &lst = nbrs_[a];
    if (std::find(lst.begin(), lst.end(), b) != lst.end()) return false;
    if (!allow_overflow && lst.size() >= params_.R) return false;
    lst.push_back(b);
    return true;
}

bool Index::remove_edge(uint32_t from, uint32_t to) {
    const uint32_t a = slot_of_any(from), b = slot_of_any(to);
    auto &lst = nbrs_[a];
    auto it = std::find(lst.begin(), lst.end(), b);
    if (it == lst.end()) return false;
    lst.erase(it);
    return true;
}

bool Index::add_extra_edge(uint32_t from, uint32_t to) {
    const uint32_t a = slot_of_any(from), b = slot_of_any(to);
    if (a == b || params_.R_extra == 0) return false;
    auto &ex = extra_[a];
    if (ex.size() >= params_.R_extra) return false;
    if (std::find(ex.begin(), ex.end(), b) != ex.end()) return false;
    if (std::find(nbrs_[a].begin(), nbrs_[a].end(), b) != nbrs_[a].end()) return false;
    ex.push_back(b);
    return true;
}

bool Index::remove_extra_edge(uint32_t from, uint32_t to) {
    const uint32_t a = slot_of_any(from), b = slot_of_any(to);
    auto &ex = extra_[a];
    auto it = std::find(ex.begin(), ex.end(), b);
    if (it == ex.end()) return false;
    ex.erase(it);
    return true;
}

size_t Index::set_R_extra(uint32_t r_extra) {
    size_t dropped = 0;
    for (auto &ex : extra_) {
        if (ex.size() > r_extra) {
            dropped += ex.size() - r_extra;
            ex.resize(r_extra);
        }
    }
    params_.R_extra = r_extra;
    return dropped;
}

size_t Index::clear_extra_edges() {
    size_t cnt = 0;
    for (auto &ex : extra_) {
        cnt += ex.size();
        std::vector<uint32_t>().swap(ex);
    }
    return cnt;
}

std::vector<uint32_t> Index::in_neighbors(uint32_t label, bool include_extra) const {
    const uint32_t t = slot_of_any(label);
    std::vector<uint32_t> out;
    for (uint32_t sl = 0; sl < next_slot_; ++sl) {
        if (!usable(sl)) continue;
        const auto &l = nbrs_[sl];
        bool hit = std::find(l.begin(), l.end(), t) != l.end();
        if (!hit && include_extra) hit = std::find(extra_[sl].begin(), extra_[sl].end(), t) != extra_[sl].end();
        if (hit) out.push_back(label_of(sl));
    }
    return out;
}

std::vector<uint32_t> Index::labels() const {
    std::vector<uint32_t> out;
    out.reserve(label_to_slot_.size());
    for (uint32_t sl = 1; sl < next_slot_; ++sl)
        if (state_[sl] == ACTIVE) out.push_back(slot_label_[sl]);
    return out;
}

std::pair<size_t, size_t> Index::reachability(bool include_extra) const {
    std::vector<uint8_t> seen(next_slot_, 0);
    std::vector<uint32_t> frontier{kEntry};
    seen[kEntry] = 1;
    size_t reached_active = 0;
    while (!frontier.empty()) {
        uint32_t u = frontier.back();
        frontier.pop_back();
        if (u != kEntry && state_[u] == ACTIVE) ++reached_active;
        auto push = [&](uint32_t v) {
            if (v < next_slot_ && !seen[v] && (v == kEntry || state_[v] != FREE)) {
                seen[v] = 1;
                frontier.push_back(v);
            }
        };
        for (uint32_t v : nbrs_[u]) push(v);
        if (include_extra)
            for (uint32_t v : extra_[u]) push(v);
    }
    return {reached_active, label_to_slot_.size()};
}

void Index::degree_stats(double &avg, uint32_t &max_deg, double &avg_extra, size_t &n_extra_edges) const {
    size_t cnt = 0, tot = 0;
    max_deg = 0;
    n_extra_edges = 0;
    for (uint32_t sl = 1; sl < next_slot_; ++sl) {
        if (state_[sl] != ACTIVE) continue;
        ++cnt;
        tot += nbrs_[sl].size();
        max_deg = std::max<uint32_t>(max_deg, static_cast<uint32_t>(nbrs_[sl].size()));
        n_extra_edges += extra_[sl].size();
    }
    avg = cnt ? static_cast<double>(tot) / cnt : 0.0;
    avg_extra = cnt ? static_cast<double>(n_extra_edges) / cnt : 0.0;
}

static const uint32_t kMagic = 0x47434E49;
static const uint32_t kVersion = 1;

void Index::save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("incgraph: cannot open " + path + " for writing");
    write_pod(out, kMagic);
    write_pod(out, kVersion);
    write_pod(out, static_cast<uint64_t>(dim_));
    write_pod(out, static_cast<int32_t>(metric_));
    write_pod(out, params_.R);
    write_pod(out, params_.L);
    write_pod(out, params_.alpha);
    write_pod(out, params_.max_candidates);
    write_pod(out, params_.slack);
    write_pod(out, params_.R_extra);
    write_pod(out, static_cast<uint64_t>(capacity_));
    write_pod(out, next_slot_);
    write_pod(out, static_cast<uint64_t>(num_deleted_));
    write_pod(out, static_cast<uint8_t>(entry_set_));
    out.write(reinterpret_cast<const char *>(state_.data()), next_slot_);
    out.write(reinterpret_cast<const char *>(slot_label_.data()), next_slot_ * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(data_.data()), (size_t)next_slot_ * dim_ * sizeof(float));
    for (const auto *g : {&nbrs_, &extra_}) {
        for (uint32_t sl = 0; sl < next_slot_; ++sl) {
            uint32_t k = static_cast<uint32_t>((*g)[sl].size());
            write_pod(out, k);
            out.write(reinterpret_cast<const char *>((*g)[sl].data()), k * sizeof(uint32_t));
        }
    }
    write_pod(out, static_cast<uint64_t>(free_slots_.size()));
    out.write(reinterpret_cast<const char *>(free_slots_.data()), free_slots_.size() * sizeof(uint32_t));
    if (!out) throw std::runtime_error("incgraph: write failed for " + path);
}

std::unique_ptr<Index> Index::load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("incgraph: cannot open " + path);
    uint32_t magic, version;
    read_pod(in, magic);
    read_pod(in, version);
    if (magic != kMagic || version != kVersion) throw std::runtime_error("incgraph: not an incgraph file: " + path);
    uint64_t dim, capacity, num_deleted;
    int32_t metric;
    Params p;
    uint32_t next_slot;
    uint8_t entry_set;
    read_pod(in, dim);
    read_pod(in, metric);
    read_pod(in, p.R);
    read_pod(in, p.L);
    read_pod(in, p.alpha);
    read_pod(in, p.max_candidates);
    read_pod(in, p.slack);
    read_pod(in, p.R_extra);
    read_pod(in, capacity);
    read_pod(in, next_slot);
    read_pod(in, num_deleted);
    read_pod(in, entry_set);
    auto idx = std::make_unique<Index>(dim, static_cast<Metric>(metric), std::max<uint64_t>(capacity, next_slot), p);
    idx->next_slot_ = next_slot;
    idx->num_deleted_ = num_deleted;
    idx->entry_set_ = entry_set != 0;
    in.read(reinterpret_cast<char *>(idx->state_.data()), next_slot);
    in.read(reinterpret_cast<char *>(idx->slot_label_.data()), next_slot * sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(idx->data_.data()), (size_t)next_slot * dim * sizeof(float));
    for (auto *g : {&idx->nbrs_, &idx->extra_}) {
        for (uint32_t sl = 0; sl < next_slot; ++sl) {
            uint32_t k;
            read_pod(in, k);
            (*g)[sl].resize(k);
            in.read(reinterpret_cast<char *>((*g)[sl].data()), k * sizeof(uint32_t));
        }
    }
    uint64_t nfree;
    read_pod(in, nfree);
    idx->free_slots_.resize(nfree);
    in.read(reinterpret_cast<char *>(idx->free_slots_.data()), nfree * sizeof(uint32_t));
    if (!in) throw std::runtime_error("incgraph: truncated file " + path);
    for (uint32_t sl = 1; sl < next_slot; ++sl)
        if (idx->state_[sl] == ACTIVE) idx->label_to_slot_[idx->slot_label_[sl]] = sl;
    return idx;
}

static const uint32_t kGraphMagic = 0x47474E49;

void Index::save_graph(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("incgraph: cannot open " + path + " for writing");
    write_pod(out, kGraphMagic);
    write_pod(out, kVersion);
    write_pod(out, static_cast<uint64_t>(dim_));
    write_pod(out, static_cast<int32_t>(metric_));
    write_pod(out, params_.R);
    write_pod(out, params_.L);
    write_pod(out, params_.alpha);
    write_pod(out, params_.max_candidates);
    write_pod(out, params_.slack);
    write_pod(out, params_.R_extra);
    write_pod(out, static_cast<uint64_t>(capacity_));
    write_pod(out, next_slot_);
    write_pod(out, static_cast<uint64_t>(num_deleted_));
    write_pod(out, static_cast<uint8_t>(entry_set_));
    out.write(reinterpret_cast<const char *>(state_.data()), next_slot_);
    out.write(reinterpret_cast<const char *>(slot_label_.data()), next_slot_ * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(vec_of(kEntry)), dim_ * sizeof(float));
    for (const auto *g : {&nbrs_, &extra_}) {
        for (uint32_t sl = 0; sl < next_slot_; ++sl) {
            uint32_t k = static_cast<uint32_t>((*g)[sl].size());
            write_pod(out, k);
            out.write(reinterpret_cast<const char *>((*g)[sl].data()), k * sizeof(uint32_t));
        }
    }
    write_pod(out, static_cast<uint64_t>(free_slots_.size()));
    out.write(reinterpret_cast<const char *>(free_slots_.data()), free_slots_.size() * sizeof(uint32_t));
    if (!out) throw std::runtime_error("incgraph: write failed for " + path);
}

std::unique_ptr<Index> Index::load_graph(const std::string &path, const float *base, size_t n_base) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("incgraph: cannot open " + path);
    uint32_t magic, version;
    read_pod(in, magic);
    read_pod(in, version);
    if (magic != kGraphMagic || version != kVersion)
        throw std::runtime_error("incgraph: not an incgraph graph snapshot: " + path);
    uint64_t dim, capacity, num_deleted;
    int32_t metric;
    Params p;
    uint32_t next_slot;
    uint8_t entry_set;
    read_pod(in, dim);
    read_pod(in, metric);
    read_pod(in, p.R);
    read_pod(in, p.L);
    read_pod(in, p.alpha);
    read_pod(in, p.max_candidates);
    read_pod(in, p.slack);
    read_pod(in, p.R_extra);
    read_pod(in, capacity);
    read_pod(in, next_slot);
    read_pod(in, num_deleted);
    read_pod(in, entry_set);
    auto idx = std::make_unique<Index>(dim, static_cast<Metric>(metric), std::max<uint64_t>(capacity, next_slot), p);
    idx->next_slot_ = next_slot;
    idx->num_deleted_ = num_deleted;
    idx->entry_set_ = entry_set != 0;
    in.read(reinterpret_cast<char *>(idx->state_.data()), next_slot);
    in.read(reinterpret_cast<char *>(idx->slot_label_.data()), next_slot * sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(idx->vec_of(kEntry)), dim * sizeof(float));
    for (auto *g : {&idx->nbrs_, &idx->extra_}) {
        for (uint32_t sl = 0; sl < next_slot; ++sl) {
            uint32_t k;
            read_pod(in, k);
            (*g)[sl].resize(k);
            in.read(reinterpret_cast<char *>((*g)[sl].data()), k * sizeof(uint32_t));
        }
    }
    uint64_t nfree;
    read_pod(in, nfree);
    idx->free_slots_.resize(nfree);
    in.read(reinterpret_cast<char *>(idx->free_slots_.data()), nfree * sizeof(uint32_t));
    if (!in) throw std::runtime_error("incgraph: truncated file " + path);
    idx->state_[kEntry] = ACTIVE;
    for (uint32_t sl = 1; sl < next_slot; ++sl) {
        if (idx->state_[sl] == FREE) continue;
        const uint32_t lab = idx->slot_label_[sl];
        if (lab >= n_base) throw std::runtime_error("incgraph: label beyond the base matrix in " + path);
        std::memcpy(idx->vec_of(sl), base + static_cast<size_t>(lab) * dim, dim * sizeof(float));
        if (idx->state_[sl] == ACTIVE) idx->label_to_slot_[lab] = sl;
    }
    return idx;
}

uint64_t Index::graph_hash() const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void *p, size_t n) {
        const unsigned char *c = static_cast<const unsigned char *>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= c[i];
            h *= 1099511628211ull;
        }
    };
    mix(&next_slot_, sizeof next_slot_);
    mix(vec_of(kEntry), dim_ * sizeof(float));
    for (uint32_t sl = 0; sl < next_slot_; ++sl) {
        const uint8_t st = state_[sl];
        mix(&st, 1);
        if (st == FREE) continue;
        const uint32_t lab = label_of(sl);
        mix(&lab, sizeof lab);
        for (const auto *g : {&nbrs_, &extra_}) {
            const uint32_t k = static_cast<uint32_t>((*g)[sl].size());
            mix(&k, sizeof k);
            for (uint32_t x : (*g)[sl]) {
                const uint32_t xl = label_of(x);
                mix(&xl, sizeof xl);
            }
        }
    }
    return h;
}

std::vector<std::pair<uint32_t, uint32_t>> Index::extra_edge_list() const {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (uint32_t sl = 1; sl < next_slot_; ++sl) {
        if (state_[sl] != ACTIVE) continue;
        for (uint32_t x : extra_[sl]) out.emplace_back(slot_label_[sl], label_of(x));
    }
    return out;
}

}

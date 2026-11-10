#include "incgraph/policy.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "incgraph/util.h"

namespace incgraph {

Policy::Policy(const KV &kv, int threads) : threads_(threads > 0 ? threads : 1), kv_(kv) {}

double Policy::getf(const std::string &k, double def) {
    used_.insert(k);
    auto it = kv_.find(k);
    double v = it == kv_.end() ? def : std::stod(it->second);
    params_.emplace_back(k, fmt_num(v));
    return v;
}
long Policy::geti(const std::string &k, long def) {
    used_.insert(k);
    auto it = kv_.find(k);
    long v = it == kv_.end() ? def : std::stol(it->second);
    params_.emplace_back(k, std::to_string(v));
    return v;
}
std::string Policy::gets(const std::string &k, const std::string &def) {
    used_.insert(k);
    auto it = kv_.find(k);
    std::string v = it == kv_.end() ? def : it->second;
    params_.emplace_back(k, json_escape(v));
    return v;
}
void Policy::check_unused() const {
    for (auto &kv : kv_)
        if (!used_.count(kv.first)) throw std::invalid_argument("policy " + name() + ": unknown parameter " + kv.first);
}
std::string Policy::describe() const {
    JObj o;
    o.add("policy", name());
    for (auto &p : params_) o.raw(p.first, p.second);
    return o.fields();
}
void Policy::insert(Index &idx, const float *vecs, const uint32_t *labels, size_t n) {
    idx.insert(vecs, labels, n, threads_, -1.f);
}
void Policy::remove(Index &idx, const uint32_t *labels, size_t n) { idx.lazy_delete(labels, n); }
void Policy::consolidate(Index &idx) { idx.consolidate(threads_); }

std::unique_ptr<Policy> make_policy(const std::string &name, const KV &kv, int threads) {
    std::unique_ptr<Policy> p;
    if (name == "noop")
        p = std::make_unique<NoopPolicy>(kv, threads);
    else if (name == "oodbridge")
        p = std::make_unique<OODBridgePolicy>(kv, threads);
    else if (name == "comember")
        p = std::make_unique<CoMemberPolicy>(kv, threads);
    else
        throw std::invalid_argument("unknown policy " + name + " (noop, oodbridge, comember)");
    p->check_unused();
    return p;
}

OODBridgePolicy::OODBridgePolicy(const KV &kv, int threads) : Policy(kv, threads) {
    m_ = static_cast<uint32_t>(geti("m", 1));
    q_ = getf("q", 1.0);
    L_bridge_ = static_cast<uint32_t>(geti("L_bridge", 50));
    seed_ = static_cast<uint64_t>(geti("seed", 0));
}

void OODBridgePolicy::setup(Index &idx) {
    if (idx.params().R_extra < m_)
        throw std::invalid_argument("oodbridge needs overlay capacity R_extra >= m (--R-extra)");
}

void OODBridgePolicy::calibrate(Index &idx) {
    std::vector<uint32_t> labels = idx.labels();
    Rng rng(seed_);
    std::vector<uint32_t> pick = rng.sample(labels.size(), std::min<size_t>(2000, labels.size()));
    const size_t n = pick.size(), d = idx.dim();
    std::vector<float> vecs(n * d);
    for (size_t i = 0; i < n; ++i) std::copy_n(idx.vector(labels[pick[i]]), d, vecs.data() + i * d);
    std::vector<uint32_t> res(n * 2);
    std::vector<float> dist(n * 2);
    idx.batch_search(vecs.data(), n, 2, L_bridge_, res.data(), dist.data(), nullptr, nullptr, threads_);
    std::vector<double> nn(n);
    for (size_t i = 0; i < n; ++i) nn[i] = res[i * 2] == labels[pick[i]] ? dist[i * 2 + 1] : dist[i * 2];
    std::sort(nn.begin(), nn.end());
    const double pos = (100.0 - q_) / 100.0 * static_cast<double>(n - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, n - 1);
    tau_ = nn[lo] + (pos - static_cast<double>(lo)) * (nn[hi] - nn[lo]);
}

void OODBridgePolicy::bridge(Index &idx, const std::vector<uint32_t> &pts, const float *vecs,
                             std::vector<uint32_t> &res, std::vector<float> &dist) const {
    res.assign(pts.size() * m_, ENTRY_ID);
    dist.assign(pts.size() * m_, 0.f);
    idx.batch_search(vecs, pts.size(), m_, L_bridge_, res.data(), dist.data(), nullptr, nullptr, threads_, 0);
}

void OODBridgePolicy::insert(Index &idx, const float *vecs, const uint32_t *labels, size_t n) {
    if (tau_ < 0) calibrate(idx);
    idx.insert(vecs, labels, n, threads_, -1.f);
    const double t0 = now_s();
    std::vector<uint32_t> pts(labels, labels + n);
    idx.set_group(pts, 1);
    std::vector<uint32_t> res;
    std::vector<float> dist;
    bridge(idx, pts, vecs, res, dist);
    std::vector<uint32_t> in_dist;
    for (size_t i = 0; i < n; ++i) {
        if (!(dist[i * m_] > tau_)) {
            in_dist.push_back(pts[i]);
            continue;
        }
        for (uint32_t j = 0; j < m_; ++j)
            if (res[i * m_ + j] != ENTRY_ID) idx.add_extra_edge(pts[i], res[i * m_ + j]);
        ood_.insert(pts[i]);
        ++n_ood_;
    }
    idx.set_group(in_dist, 2);
    n_id_ += in_dist.size();
    bridge_s_ += now_s() - t0;
}

void OODBridgePolicy::remove(Index &idx, const uint32_t *labels, size_t n) {
    idx.lazy_delete(labels, n);
    for (size_t i = 0; i < n; ++i) ood_.erase(labels[i]);
}

void OODBridgePolicy::consolidate(Index &idx) {
    idx.consolidate(threads_);
    const double t0 = now_s();
    std::vector<uint32_t> broken;
    for (uint32_t p : ood_)
        if (idx.extra_neighbors(p).size() < m_) broken.push_back(p);
    if (!broken.empty()) {
        const size_t d = idx.dim();
        std::vector<float> vecs(broken.size() * d);
        for (size_t i = 0; i < broken.size(); ++i) std::copy_n(idx.vector(broken[i]), d, vecs.data() + i * d);
        std::vector<uint32_t> res;
        std::vector<float> dist;
        bridge(idx, broken, vecs.data(), res, dist);
        for (size_t i = 0; i < broken.size(); ++i) {
            std::vector<uint32_t> have = idx.extra_neighbors(broken[i]);
            for (uint32_t j = 0; j < m_ && have.size() < m_; ++j) {
                uint32_t b = res[i * m_ + j];
                if (b == ENTRY_ID || std::find(have.begin(), have.end(), b) != have.end()) continue;
                if (idx.add_extra_edge(broken[i], b)) have.push_back(b);
            }
        }
    }
    n_rebridged_ += broken.size();
    bridge_s_ += now_s() - t0;
}

std::string OODBridgePolicy::stats() const {
    return JObj()
        .add("tau", tau_)
        .add("n_id_inserted", static_cast<int64_t>(n_id_))
        .add("n_ood_inserted", static_cast<int64_t>(n_ood_))
        .add("n_ood_live", static_cast<int64_t>(ood_.size()))
        .add("n_rebridged", static_cast<int64_t>(n_rebridged_))
        .add("bridge_s", bridge_s_)
        .str();
}

CoMemberPolicy::CoMemberPolicy(const KV &kv, int threads) : Policy(kv, threads) {
    K_ = static_cast<uint32_t>(geti("K", 30));
    L_member_ = static_cast<uint32_t>(geti("L_member", 200));
    prune_alpha_ = static_cast<float>(getf("prune_alpha", 1.2));
    removal_ = gets("removal", "score");
    if (removal_ != "score" && removal_ != "none")
        throw std::invalid_argument("comember: removal must be score or none");
    decay_ = removal_ == "score" ? getf("decay", 0.5) : 0.5;
    min_score_ = removal_ == "score" ? getf("min_score", 0.25) : 0.25;
    budget_ = static_cast<size_t>(geti("budget", 0));
}

void CoMemberPolicy::setup(Index &idx) {
    if (idx.params().R_extra == 0) throw std::invalid_argument("comember needs overlay capacity (--R-extra N)");
}

void CoMemberPolicy::drop(Index &idx, uint64_t e) {
    const uint32_t u = static_cast<uint32_t>(e >> 32), v = static_cast<uint32_t>(e);
    if (idx.contains(u)) idx.remove_extra_edge(u, v);
    score_.erase(e);
}

void CoMemberPolicy::maintain(Index &idx, const float *Q, size_t nq, int epoch) {
    const double t0 = now_s();
    const uint32_t rx = idx.params().R_extra;
    std::vector<uint32_t> T(nq * K_);
    std::vector<float> D(nq * K_);
    idx.batch_search(Q, nq, K_, std::max(L_member_, K_), T.data(), D.data(), nullptr, nullptr, threads_);
    const double t_search = now_s();
    std::unordered_map<uint32_t, uint32_t> mid;
    std::vector<uint32_t> mlab;
    std::vector<std::vector<uint32_t>> qs, cand;
    std::vector<uint32_t> row;
    for (size_t qi = 0; qi < nq; ++qi) {
        row.clear();
        for (uint32_t j = 0; j < K_; ++j) {
            const uint32_t x = T[qi * K_ + j];
            if (x != ENTRY_ID && idx.contains(x)) row.push_back(x);
        }
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (uint32_t x : row) {
            auto it = mid.find(x);
            uint32_t id;
            if (it == mid.end()) {
                id = static_cast<uint32_t>(mlab.size());
                mid.emplace(x, id);
                mlab.push_back(x);
                qs.emplace_back();
                cand.emplace_back();
            } else {
                id = it->second;
            }
            qs[id].push_back(static_cast<uint32_t>(qi));
            cand[id].insert(cand[id].end(), row.begin(), row.end());
        }
    }
    for (size_t i = 0; i < cand.size(); ++i) {
        auto &c = cand[i];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        c.erase(std::remove(c.begin(), c.end(), mlab[i]), c.end());
    }
    auto co = [&](uint32_t u, uint32_t v) -> float {
        auto a = mid.find(u), b = mid.find(v);
        if (a == mid.end() || b == mid.end()) return 0.f;
        const auto &x = qs[a->second], &y = qs[b->second];
        size_t i = 0, j = 0, c = 0;
        while (i < x.size() && j < y.size()) {
            if (x[i] < y[j])
                ++i;
            else if (y[j] < x[i])
                ++j;
            else {
                ++c;
                ++i;
                ++j;
            }
        }
        return static_cast<float>(c);
    };

    size_t confirmed = 0, expired = 0, added = 0, evicted = 0, trimmed = 0;
    const double t_members = now_s();
    std::vector<uint64_t> keys;
    keys.reserve(score_.size());
    for (auto &kv : score_) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (uint64_t e : keys) {
        const float c = co(static_cast<uint32_t>(e >> 32), static_cast<uint32_t>(e));
        if (c > 0) ++confirmed;
        if (removal_ != "score") continue;
        const float sc = static_cast<float>(decay_) * score_[e] + c;
        if (sc < min_score_) {
            drop(idx, e);
            ++expired;
        } else {
            score_[e] = sc;
        }
    }
    const double t_update = now_s();
    std::vector<uint32_t> order(mlab.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return mlab[a] < mlab[b]; });
    const bool capped = removal_ == "none" && budget_ > 0;
    std::vector<uint32_t> nw;
    for (uint32_t oi : order) {
        if (capped && score_.size() >= budget_) break;
        const uint32_t u = mlab[oi];
        const std::vector<uint32_t> base_n = idx.neighbors(u);
        const std::vector<uint32_t> ex = idx.extra_neighbors(u);
        nw.clear();
        for (uint32_t v : cand[oi])
            if (std::find(base_n.begin(), base_n.end(), v) == base_n.end() &&
                std::find(ex.begin(), ex.end(), v) == ex.end())
                nw.push_back(v);
        if (nw.empty()) continue;
        const std::vector<uint32_t> sel = nw.size() > rx ? idx.robust_prune(u, nw, prune_alpha_, rx) : nw;
        for (uint32_t v : sel) {
            if (capped && score_.size() >= budget_) break;
            const float c = co(u, v);
            if (!idx.add_extra_edge(u, v)) {
                if (removal_ == "none") break;
                const std::vector<uint32_t> cur = idx.extra_neighbors(u);
                if (cur.empty()) continue;
                uint32_t worst = cur[0];
                float ws = 1e30f;
                for (uint32_t x : cur) {
                    auto it = score_.find(key(u, x));
                    const float s = it == score_.end() ? 0.f : it->second;
                    if (s < ws) {
                        ws = s;
                        worst = x;
                    }
                }
                if (ws >= c) continue;
                drop(idx, key(u, worst));
                ++evicted;
                if (!idx.add_extra_edge(u, v)) continue;
            }
            score_[key(u, v)] = c;
            ++added;
        }
    }
    const double t_add = now_s();
    if (budget_ && removal_ == "score" && score_.size() > budget_) {
        trimmed = score_.size() - budget_;
        std::vector<std::pair<float, uint64_t>> all;
        all.reserve(score_.size());
        for (auto &kv : score_) all.emplace_back(kv.second, kv.first);
        std::nth_element(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(trimmed), all.end());
        for (size_t i = 0; i < trimmed; ++i) drop(idx, all[i].second);
    }
    double smean = 0;
    for (auto &kv : score_) smean += kv.second;
    if (!score_.empty()) smean /= static_cast<double>(score_.size());
    const double dt = now_s() - t0;
    added_ += added;
    removed_ += expired + trimmed;
    evicted_ += evicted;
    ++rounds_;
    maintain_s_ += dt;
    last_round = JObj()
                     .add("log_epoch", epoch)
                     .add("members", static_cast<int64_t>(mlab.size()))
                     .add("confirmed", static_cast<int64_t>(confirmed))
                     .add("expired", static_cast<int64_t>(expired))
                     .add("added", static_cast<int64_t>(added))
                     .add("evicted", static_cast<int64_t>(evicted))
                     .add("trimmed", static_cast<int64_t>(trimmed))
                     .add("n_extra_edges", static_cast<int64_t>(score_.size()))
                     .add("score_mean", removal_ == "score" ? smean : 0.0)
                     .add("maintain_s", dt)
                     .add("search_s", t_search - t0)
                     .add("members_s", t_members - t_search)
                     .add("update_s", t_update - t_members)
                     .add("add_s", t_add - t_update)
                     .add("budget_s", now_s() - t_add)
                     .str();
}

std::string CoMemberPolicy::stats() const {
    return JObj()
        .add("rounds", static_cast<int64_t>(rounds_))
        .add("added", static_cast<int64_t>(added_))
        .add("removed", static_cast<int64_t>(removed_))
        .add("evicted", static_cast<int64_t>(evicted_))
        .add("maintain_s", maintain_s_)
        .str();
}

}

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "../apps/common.h"
#include "incgraph/policy.h"

using namespace incgraph;

namespace {
int fails = 0;
void check(bool ok, const std::string &what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) ++fails;
}

FMatrix clusters(size_t n, size_t d, int nc, double sd, uint64_t seed, bool normalize) {
    Rng r(seed);
    auto gauss = [&]() {
        double u = std::max(r.uniform(), 1e-12), v = r.uniform();
        return std::sqrt(-2 * std::log(u)) * std::cos(2 * M_PI * v);
    };
    std::vector<float> c(nc * d);
    for (auto &x : c) x = static_cast<float>(gauss());
    FMatrix m;
    m.n = n;
    m.d = d;
    m.v.resize(n * d);
    for (size_t i = 0; i < n; ++i) {
        const size_t cl = i % nc;
        double nrm = 0;
        for (size_t j = 0; j < d; ++j) {
            m.v[i * d + j] = c[cl * d + j] + static_cast<float>(sd * gauss());
            nrm += m.v[i * d + j] * m.v[i * d + j];
        }
        if (normalize)
            for (size_t j = 0; j < d; ++j) m.v[i * d + j] /= static_cast<float>(std::sqrt(nrm));
    }
    return m;
}

FMatrix near_points(const FMatrix &base, size_t nq, double sd, uint64_t seed, bool normalize) {
    Rng r(seed);
    FMatrix q;
    q.n = nq;
    q.d = base.d;
    q.v.resize(nq * base.d);
    for (size_t i = 0; i < nq; ++i) {
        const float *b = base.row(r.below(base.n));
        double nrm = 0;
        for (size_t j = 0; j < base.d; ++j) {
            double u = std::max(r.uniform(), 1e-12), v = r.uniform();
            q.v[i * base.d + j] = b[j] + static_cast<float>(sd * std::sqrt(-2 * std::log(u)) * std::cos(2 * M_PI * v));
            nrm += q.v[i * base.d + j] * q.v[i * base.d + j];
        }
        if (normalize)
            for (size_t j = 0; j < base.d; ++j) q.v[i * base.d + j] /= static_cast<float>(std::sqrt(nrm));
    }
    return q;
}

double recall_of(const Index &idx, const FMatrix &base, const FMatrix &q, const std::vector<uint32_t> *ids, Metric m,
                 uint32_t L) {
    Truth gt = knn(base, q, 10, m, ids, 0);
    std::vector<uint32_t> res(q.n * 10);
    std::vector<float> dist(q.n * 10);
    idx.batch_search(q.v.data(), q.n, 10, L, res.data(), dist.data(), nullptr, nullptr, 0);
    return recall_at(res.data(), q.n, 10, gt, 10);
}
}

int main() try {
    const size_t n = 20000, d = 32;
    for (Metric m : {Metric::L2, Metric::IP}) {
        const std::string mn = m == Metric::L2 ? "l2" : "ip";
        FMatrix base = clusters(n, d, 50, 0.15, 1, m == Metric::IP);
        FMatrix q = near_points(base, 500, 0.05, 2, m == Metric::IP);
        std::vector<uint32_t> labels(n);
        for (size_t i = 0; i < n; ++i) labels[i] = static_cast<uint32_t>(i);
        Params p;
        p.R = 32;
        p.L = 64;
        Index a(d, m, n, p), b(d, m, n, p);
        a.build(base.v.data(), labels.data(), n, 1);
        b.build(base.v.data(), labels.data(), n, 1);
        check(a.graph_hash() == b.graph_hash(), mn + ": single-thread build is deterministic");
        const double r = recall_of(a, base, q, nullptr, m, 64);
        check(r > 0.95, mn + ": static recall@10 (L=64) = " + fmt_num(r));
        auto [reach, act] = a.reachability(false);
        check(act - reach <= act / 1000, mn + ": unreachable points " + std::to_string(act - reach) + " / " + std::to_string(act));

        a.save_graph("/tmp/ig_test.graph");
        auto c = Index::load_graph("/tmp/ig_test.graph", base.v.data(), base.n);
        check(c->graph_hash() == a.graph_hash(), mn + ": graph snapshot round trip");
        check(std::fabs(recall_of(*c, base, q, nullptr, m, 64) - r) < 1e-12, mn + ": same recall after reload");

        Rng dr(9);
        std::vector<uint32_t> del = dr.sample(n, n / 2);
        std::vector<uint8_t> gone(n, 0);
        for (uint32_t x : del) gone[x] = 1;
        a.lazy_delete(del.data(), del.size());
        const size_t freed = a.consolidate(1);
        check(freed == del.size() && a.size() == n - del.size(), mn + ": consolidate frees deleted slots");
        std::vector<uint32_t> live;
        for (size_t i = 0; i < n; ++i)
            if (!gone[i]) live.push_back(static_cast<uint32_t>(i));
        const double rd = recall_of(a, base, q, &live, m, 64);
        check(rd > 0.93, mn + ": recall after deleting half = " + fmt_num(rd));
        std::vector<float> v = gather_rows(base, del.data(), del.size());
        a.insert(v.data(), del.data(), del.size(), 1, -1.f);
        const double ri = recall_of(a, base, q, nullptr, m, 64);
        check(ri > 0.93 && a.size() == n, mn + ": recall after re-inserting = " + fmt_num(ri));
    }

    {
        const size_t n0 = 10000, n1 = 2000;
        FMatrix a = clusters(n0, d, 10, 0.3, 3, false), b = clusters(n1, d, 2, 0.3, 4, false);
        for (auto &x : b.v) x += 6.f;
        FMatrix all;
        all.n = n0 + n1;
        all.d = d;
        all.v = a.v;
        all.v.insert(all.v.end(), b.v.begin(), b.v.end());
        Params p;
        p.R = 32;
        p.L = 64;
        p.R_extra = 1;
        Index idx(d, Metric::L2, all.n, p);
        std::vector<uint32_t> l0(n0), l1(n1);
        for (size_t i = 0; i < n0; ++i) l0[i] = static_cast<uint32_t>(i);
        for (size_t i = 0; i < n1; ++i) l1[i] = static_cast<uint32_t>(n0 + i);
        idx.build(a.v.data(), l0.data(), n0, 1);
        auto pol = make_policy("oodbridge", {}, 1);
        pol->setup(idx);
        pol->insert(idx, b.v.data(), l1.data(), n1);
        std::printf("        oodbridge stats %s\n", pol->stats().c_str());
        double avg, avg_x;
        uint32_t mx;
        size_t nx;
        idx.degree_stats(avg, mx, avg_x, nx);
        size_t linked = 0;
        for (uint32_t x : l1) {
            bool ok = false;
            for (uint32_t y : idx.extra_neighbors(x)) ok |= y < n0;
            for (uint32_t y : idx.neighbors(x)) ok |= y < n0;
            linked += ok;
        }
        check(nx <= n1 && linked == n1, "oodbridge: every OOD insert links back (" + std::to_string(linked) + "/" +
                                            std::to_string(n1) + ", " + std::to_string(nx) + " return edges)");
        bool to_anchor = true;
        for (uint32_t x : l1)
            for (uint32_t y : idx.extra_neighbors(x)) to_anchor &= y < n0;
        check(to_anchor, "oodbridge: return edges point to the initial points");
        FMatrix c = clusters(1000, d, 10, 0.3, 3, false);
        std::vector<uint32_t> l2(1000);
        Index idx2(d, Metric::L2, n0 + 1000, p);
        idx2.build(a.v.data(), l0.data(), n0, 1);
        auto pol2 = make_policy("oodbridge", {}, 1);
        for (size_t i = 0; i < 1000; ++i) l2[i] = static_cast<uint32_t>(n0 + i);
        pol2->insert(idx2, c.v.data(), l2.data(), 1000);
        idx2.degree_stats(avg, mx, avg_x, nx);
        check(nx < 50, "oodbridge: in-distribution inserts get (almost) no edges (" + std::to_string(nx) + ")");
    }

    {
        FMatrix base = clusters(n, d, 20, 0.3, 5, true);
        Rng r(7);
        FMatrix qo, qn;
        qo.d = qn.d = d;
        for (size_t i = 0; i < n && (qo.n < 1000 || qn.n < 1000); ++i) {
            FMatrix &t = (i % 20) < 10 ? qo : qn;
            if (t.n >= 1000) continue;
            double nrm = 0;
            std::vector<float> v(base.row(i), base.row(i) + d);
            for (auto &x : v) {
                x += static_cast<float>(0.05 * (r.uniform() - 0.5));
                nrm += x * x;
            }
            for (auto &x : v) t.v.push_back(static_cast<float>(x / std::sqrt(nrm)));
            ++t.n;
        }
        std::vector<uint32_t> labels(n);
        for (size_t i = 0; i < n; ++i) labels[i] = static_cast<uint32_t>(i);
        Params p;
        p.R = 16;
        p.L = 64;
        p.R_extra = 8;
        Index x(d, Metric::IP, n, p), y(d, Metric::IP, n, p);
        x.build(base.v.data(), labels.data(), n, 1);
        y.build(base.v.data(), labels.data(), n, 1);
        auto px = make_policy("comember", {{"K", "20"}}, 1), py = make_policy("comember", {{"K", "20"}}, 1);
        px->setup(x);
        py->setup(y);
        px->maintain(x, qo.v.data(), qo.n, 0);
        py->maintain(y, qo.v.data(), qo.n, 0);
        check(x.graph_hash() == y.graph_hash(), "comember: deterministic");
        double avg, avg_x;
        uint32_t mx;
        size_t n1, n2, n3;
        x.degree_stats(avg, mx, avg_x, n1);
        check(n1 > 0, "comember: adds overlay edges (" + std::to_string(n1) + ")");
        px->maintain(x, qo.v.data(), qo.n, 1);
        x.degree_stats(avg, mx, avg_x, n2);
        check(n2 >= n1, "comember: a repeated log keeps its edges (" + std::to_string(n2) + ")");
        for (int e = 2; e < 12; ++e) px->maintain(x, qn.v.data(), qn.n, e);
        size_t old_left = 0;
        for (auto &ed : x.extra_edge_list()) old_left += (ed.first % 20) < 10 && (ed.second % 20) < 10;
        x.degree_stats(avg, mx, avg_x, n3);
        check(old_left == 0 && n3 > 0, "comember: edges of vanished queries decay away (" + std::to_string(old_left) +
                                           " old left, " + std::to_string(n3) + " new)");
        auto pb = make_policy("comember", {{"K", "20"}, {"budget", "500"}}, 1);
        Index z(d, Metric::IP, n, p);
        z.build(base.v.data(), labels.data(), n, 1);
        pb->maintain(z, qo.v.data(), qo.n, 0);
        z.degree_stats(avg, mx, avg_x, n3);
        check(n3 <= 500, "comember: budget caps the overlay (" + std::to_string(n3) + ")");
        bool threw = false;
        try {
            make_policy("comember", {{"typo", "1"}}, 1);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "unknown policy parameters are rejected");
    }
    std::printf(fails ? "%d FAILED\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

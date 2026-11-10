#include <algorithm>
#include <cstdio>

#include "common.h"

using namespace incgraph;

namespace {
struct Pass {
    double best = 1e30, median = 0;
};

Pass time_passes(const Index &idx, const float *Q, size_t nq, size_t k, uint32_t L, int reps, std::vector<uint32_t> &res,
                 std::vector<uint64_t> &cmps) {
    const size_t d = idx.dim();
    std::vector<float> dist(k);
    std::vector<double> t;
    for (int r = 0; r <= reps; ++r) {
        const double t0 = now_s();
        for (size_t i = 0; i < nq; ++i) {
            SearchStats st;
            idx.search(Q + i * d, static_cast<uint32_t>(k), L, res.data() + i * k, dist.data(), &st);
            cmps[i] = st.dist_cmps;
        }
        const double dt = now_s() - t0;
        if (r > 0) t.push_back(dt);
    }
    std::sort(t.begin(), t.end());
    Pass p;
    p.best = t.front();
    p.median = t[t.size() / 2];
    return p;
}

std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}
}

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    const size_t k = static_cast<size_t>(a.i("--k", 10));
    const int reps = static_cast<int>(a.i("--reps", 5));
    const std::string out = a.req("--out");
    FMatrix base = read_fbin(ds.base_file);
    make_dirs(dirname_of(out));
    FILE *fo = std::fopen(out.c_str(), "a");
    if (!fo) throw std::runtime_error("cannot open " + out);
    auto emit = [&](const std::string &s) {
        std::fputs((s + "\n").c_str(), fo);
        std::fflush(fo);
        std::puts(s.c_str());
        std::fflush(stdout);
    };

    if (a.has("--runbook")) {
        Runbook rb = load_runbook(ds, a.req("--runbook"));
        const size_t step = static_cast<size_t>(a.i("--step", -1));
        const size_t nq = static_cast<size_t>(a.i("--nq", 10000));
        const std::vector<long> Ls = a.ints("--search-L", "10,15,20,30,50,75,100,150,200,300");
        std::vector<Range> groups;
        FMatrix Q = test_queries(ds, nq, &groups);
        Truth gt = runbook_gt(rb, step, base, Q, k, ds.metric, 0);
        auto graphs = split(a.req("--graphs"));
        auto labels = split(a.str("--labels", a.req("--graphs")));
        if (labels.size() != graphs.size()) throw std::invalid_argument("--labels must match --graphs");
        std::vector<uint32_t> res(Q.n * k);
        for (size_t gi = 0; gi < graphs.size(); ++gi) {
            auto idx = Index::load_graph(graphs[gi], base.v.data(), base.n);
            for (long L : Ls) {
                JObj r;
                r.add("type", "curve").add("dataset", ds.name).add("runbook", rb.name).add("step", static_cast<int64_t>(step));
                r.add("label", labels[gi]).add("graph", graphs[gi]).add("L", static_cast<int64_t>(L));
                std::vector<Range> parts = groups;
                if (parts.empty()) parts.push_back({"all", 0, Q.n});
                double tb = 0, tm = 0, call = 0;
                for (auto &g : parts) {
                    const size_t ng = g.hi - g.lo;
                    std::vector<uint32_t> rg(ng * k);
                    std::vector<uint64_t> cg(ng);
                    Pass p = time_passes(*idx, Q.row(g.lo), ng, k, static_cast<uint32_t>(L), reps, rg, cg);
                    std::copy(rg.begin(), rg.end(), res.begin() + g.lo * k);
                    tb += p.best;
                    tm += p.median;
                    double s = 0;
                    for (auto x : cg) s += static_cast<double>(x);
                    call += s;
                    if (groups.empty()) continue;
                    r.add("recall_" + g.name, recall_at(rg.data(), ng, k, gt, k, g.lo));
                    r.add("qps_" + g.name, ng / p.best).add("cmps_" + g.name, s / ng);
                }
                r.add("recall", recall_at(res.data(), Q.n, k, gt, k)).add("qps", Q.n / tb).add("qps_median", Q.n / tm);
                r.add("cmps", call / Q.n);
                r.add("reps", reps).add("threads", 1);
                emit(r.str());
            }
        }
    } else {
        const std::string wdir = ds.dir + "/workloads/" + a.req("--workload");
        JValue meta = read_json(wdir + "/meta.json");
        const int epochs = static_cast<int>(meta["epochs"].i64());
        const size_t n_eval = static_cast<size_t>(meta["n_eval"].i64());
        std::vector<uint32_t> eval = read_u32(wdir + "/eval.u32");
        Truth gt = read_truth(wdir + "/gt.bin");
        FMatrix learn = read_fbin(ds.learn_file, 0, static_cast<size_t>(meta["learn_rows"].i64()));
        const std::vector<long> Ls = a.ints("--search-L", "10,15,20,30,40,50,75,100,150");
        const std::string ovdir = a.str("--overlays", "");
        auto idx = Index::load_graph(a.req("--index"), base.v.data(), base.n);
        idx->set_R_extra(static_cast<uint32_t>(a.i("--R-extra", 64)));
        std::vector<int> eps;
        for (int e = 0; e < epochs; ++e)
            if (ovdir.empty() || file_exists(ovdir + "/epoch" + std::to_string(e) + ".edges")) eps.push_back(e);
        if (a.has("--epochs")) {
            eps.clear();
            for (long e : a.ints("--epochs", "")) eps.push_back(static_cast<int>(e));
        }
        const size_t nL = Ls.size();
        std::vector<double> t_best(nL, 0), t_med(nL, 0), hits(nL, 0), cm(nL, 0);
        std::vector<std::string> per_epoch(nL);
        std::vector<uint32_t> res(n_eval * k);
        std::vector<uint64_t> cmps(n_eval);
        for (int e : eps) {
            idx->clear_extra_edges();
            if (!ovdir.empty())
                for (auto &ed : read_edges(ovdir + "/epoch" + std::to_string(e) + ".edges"))
                    idx->add_extra_edge(ed.first, ed.second);
            std::vector<float> ev = gather_rows(learn, eval.data() + static_cast<size_t>(e) * n_eval, n_eval);
            for (size_t li = 0; li < nL; ++li) {
                Pass p = time_passes(*idx, ev.data(), n_eval, k, static_cast<uint32_t>(Ls[li]), reps, res, cmps);
                t_best[li] += p.best;
                t_med[li] += p.median;
                const double rc = recall_at(res.data(), n_eval, k, gt, k, static_cast<size_t>(e) * n_eval);
                hits[li] += rc * n_eval;
                for (auto x : cmps) cm[li] += static_cast<double>(x);
                per_epoch[li] += (per_epoch[li].empty() ? "" : ",") + fmt_num(rc);
            }
            std::fprintf(stderr, "epoch %d done\n", e);
        }
        const double nqt = static_cast<double>(n_eval * eps.size());
        std::string elist;
        for (int e : eps) elist += (elist.empty() ? "" : ",") + std::to_string(e);
        for (size_t li = 0; li < nL; ++li) {
            JObj r;
            r.add("type", "curve").add("dataset", ds.name).add("workload", a.req("--workload"));
            r.add("label", a.str("--label", ovdir.empty() ? "static" : ovdir)).add("L", static_cast<int64_t>(Ls[li]));
            r.add("recall", hits[li] / nqt).add("qps", nqt / t_best[li]).add("qps_median", nqt / t_med[li]);
            r.add("cmps", cm[li] / nqt).raw("epochs", "[" + elist + "]").raw("recall_epochs", "[" + per_epoch[li] + "]");
            r.add("reps", reps).add("threads", 1);
            emit(r.str());
        }
    }
    std::fclose(fo);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

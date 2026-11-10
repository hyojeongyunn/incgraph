#include <cstdio>
#include <memory>

#include "common.h"
#include "incgraph/policy.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    const std::string rbname = a.req("--runbook");
    Runbook rb = load_runbook(ds, rbname);
    const std::string pname = a.str("--policy", "noop");
    Params prm;
    prm.R = static_cast<uint32_t>(a.i("--R", 32));
    prm.L = static_cast<uint32_t>(a.i("--L", 100));
    prm.alpha = static_cast<float>(a.f("--alpha", 1.2));
    prm.R_extra = static_cast<uint32_t>(a.i("--R-extra", 0));
    const int threads = static_cast<int>(a.i("--threads", 1));
    const int sthreads = static_cast<int>(a.i("--search-threads", 8));
    const size_t k = static_cast<size_t>(a.i("--k", 10));
    const size_t nq = static_cast<size_t>(a.i("--nq", 10000));
    const std::vector<long> Ls = a.ints("--search-L", "10,20,30,50,75,100,150,200,300");
    const std::string tag = a.str("--tag", "");
    const std::string label = pname + (tag.empty() ? "" : "-" + tag);
    const std::set<size_t> chosen = pick_steps(rb.search_steps, a.str("--search-at", "all"));
    const std::set<size_t> snap = pick_steps(rb.search_steps, a.str("--snap", "none"));
    const std::string out = a.str("--out", "results/" + ds.name + "/" + rbname + "/" +
                                               (a.has("--prepare") ? std::string("prepare") : label) + ".jsonl");
    const std::string snapdir = "results/snapshots/" + ds.name + "/" + rbname + "/" + label;
    const bool prepare = a.has("--prepare");

    double t0 = now_s();
    FMatrix base = read_fbin(ds.base_file);
    std::vector<Range> groups;
    FMatrix queries = test_queries(ds, nq, &groups);
    std::fprintf(stderr, "loaded base %zu x %zu, %zu queries (%.1fs)\n", base.n, base.d, queries.n, now_s() - t0);

    std::unique_ptr<Policy> policy;
    if (!prepare) policy = make_policy(pname, a.kv, threads);
    LineWriter w(out);
    {
        JObj c;
        c.add("type", "config").add("dataset", ds.name).add("runbook", rbname).add("R", prm.R).add("L", prm.L);
        c.add("alpha", prm.alpha).add("R_extra", prm.R_extra).add("k", static_cast<int64_t>(k));
        c.add("nq", static_cast<int64_t>(queries.n)).add("threads", threads);
        if (policy) c.merge(policy->describe());
        c.add("tag", tag).add("cmdline", a.cmdline);
        w.line(c.str());
    }

    auto idx = std::make_unique<Index>(ds.dim, ds.metric, base.n, prm);
    if (policy) policy->setup(*idx);
    double op_s[4] = {0, 0, 0, 0};
    size_t n_ins = 0, n_del = 0;
    const double t_start = now_s();
    for (size_t i = 0; i < rb.steps.size(); ++i) {
        const Step &st = rb.steps[i];
        const double ts = now_s();
        if (st.op == "build") {
            const std::string cache = "results/indexes/" + ds.name + "-" + rbname + "-init-R" + std::to_string(prm.R) +
                                      "-L" + std::to_string(prm.L) + "-a" + fmt_num(prm.alpha) + ".graph";
            if (file_exists(cache)) {
                idx = Index::load_graph(cache, base.v.data(), base.n);
                idx->set_R_extra(prm.R_extra);
                std::fprintf(stderr, "initial index from %s\n", cache.c_str());
            } else {
                std::vector<float> v = gather_rows(base, rb.ids_of(st), st.n);
                idx->build(v.data(), rb.ids_of(st), st.n, threads);
                make_dirs("results/indexes");
                const std::string tmp = cache + ".tmp" + std::to_string(::getpid());
                idx->set_R_extra(0);
                idx->save_graph(tmp);
                std::rename(tmp.c_str(), cache.c_str());
                idx->set_R_extra(prm.R_extra);
                op_s[0] += now_s() - ts;
            }
            if (policy) policy->setup(*idx);
        } else if (prepare) {
            if (st.op == "search" && chosen.count(i)) runbook_gt(rb, i, base, queries, k, ds.metric, static_cast<int>(a.i("--gt-threads", 16)));
            continue;
        } else if (st.op == "insert") {
            std::vector<float> v = gather_rows(base, rb.ids_of(st), st.n);
            policy->insert(*idx, v.data(), rb.ids_of(st), st.n);
            n_ins += st.n;
            op_s[1] += now_s() - ts;
        } else if (st.op == "delete") {
            policy->remove(*idx, rb.ids_of(st), st.n);
            n_del += st.n;
            op_s[2] += now_s() - ts;
        } else if (st.op == "consolidate") {
            policy->consolidate(*idx);
            op_s[3] += now_s() - ts;
        } else if (st.op == "search") {
            if (!chosen.count(i)) continue;
            Truth gt = runbook_gt(rb, i, base, queries, k, ds.metric, static_cast<int>(a.i("--gt-threads", 16)));
            JObj r;
            r.add("type", "search").add("step", static_cast<int64_t>(i));
            r.add("ordinal", static_cast<int64_t>(std::find(rb.search_steps.begin(), rb.search_steps.end(), i) -
                                                  rb.search_steps.begin()));
            r.add("n_live", static_cast<int64_t>(idx->size())).add("n_tombstones", static_cast<int64_t>(idx->num_deleted()));
            r.add("inserted", static_cast<int64_t>(n_ins)).add("deleted", static_cast<int64_t>(n_del));
            r.add("build_s", op_s[0]).add("insert_s", op_s[1]).add("delete_s", op_s[2]).add("consolidate_s", op_s[3]);
            std::vector<uint32_t> res(queries.n * k);
            std::vector<float> dist(queries.n * k);
            std::vector<uint64_t> cmps(queries.n);
            for (long L : Ls) {
                idx->batch_search(queries.v.data(), queries.n, static_cast<uint32_t>(k), static_cast<uint32_t>(L),
                                  res.data(), dist.data(), cmps.data(), nullptr, sthreads);
                const std::string sfx = "_L" + std::to_string(L);
                r.add("recall" + sfx, recall_at(res.data(), queries.n, k, gt, k));
                double c = 0;
                for (auto x : cmps) c += static_cast<double>(x);
                r.add("cmps" + sfx, c / static_cast<double>(queries.n));
                for (auto &g : groups) {
                    r.add("recall" + sfx + "_" + g.name,
                          recall_at(res.data() + g.lo * k, g.hi - g.lo, k, gt, k, g.lo));
                    double cg = 0;
                    for (size_t q = g.lo; q < g.hi; ++q) cg += static_cast<double>(cmps[q]);
                    r.add("cmps" + sfx + "_" + g.name, cg / static_cast<double>(g.hi - g.lo));
                }
            }
            auto [reach, act] = idx->reachability(true);
            r.add("unreachable", static_cast<int64_t>(act - reach));
            double avg, avg_x;
            uint32_t mx;
            size_t nx;
            idx->degree_stats(avg, mx, avg_x, nx);
            r.add("avg_degree", avg).add("n_extra_edges", static_cast<int64_t>(nx));
            r.add("graph_hash", hex64(idx->graph_hash()));
            if (snap.count(i)) {
                make_dirs(snapdir);
                const std::string sp = snapdir + "/step" + std::to_string(i) + ".graph";
                idx->save_graph(sp);
                r.add("snapshot", sp);
            }
            r.add("elapsed_s", now_s() - t_start);
            w.line(r.str());
            continue;
        } else {
            throw std::runtime_error("unknown op " + st.op);
        }
    }
    JObj s;
    s.add("type", "summary").add("total_s", now_s() - t_start).add("build_s", op_s[0]).add("insert_s", op_s[1]);
    s.add("delete_s", op_s[2]).add("consolidate_s", op_s[3]).add("inserted", static_cast<int64_t>(n_ins));
    s.add("deleted", static_cast<int64_t>(n_del));
    s.raw("policy_stats", policy ? policy->stats() : "{}");
    w.line(s.str());
    std::fprintf(stderr, "-> %s\n", out.c_str());
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

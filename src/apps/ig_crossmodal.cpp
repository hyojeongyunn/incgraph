#include <cstdio>

#include "common.h"
#include "incgraph/policy.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    const bool use_rb = a.has("--runbook");
    if (use_rb == a.has("--seed")) throw std::runtime_error("give exactly one of --runbook and --seed");
    const std::string rbname = use_rb ? a.str("--runbook", "") : "s" + a.str("--seed", "0");
    Runbook rb;
    if (use_rb) rb = load_runbook(ds, rbname);
    const std::string pname = a.str("--policy", "noop");
    const uint32_t R = static_cast<uint32_t>(a.i("--R", 32)), L = static_cast<uint32_t>(a.i("--L", 100));
    const double alpha = a.f("--alpha", 1.2);
    const uint32_t R_extra = static_cast<uint32_t>(a.i("--R-extra", 0));
    const int threads = static_cast<int>(a.i("--threads", 1));
    const int sthreads = static_cast<int>(a.i("--search-threads", 8));
    const size_t k = static_cast<size_t>(a.i("--k", 10));
    const size_t nq = static_cast<size_t>(a.i("--nq", 10000));
    const long rounds = a.i("--rounds", 0);
    const size_t per_round = static_cast<size_t>(a.i("--per-round", 5000));
    const std::vector<long> eval_at = a.ints("--eval-at", "0,1,2,5,10,20");
    const std::vector<long> Ls = a.ints("--search-L", "10,15,20,30,50,75,100,150,200,300,500");
    const std::string image_group = a.str("--single", "image"), text_group = a.str("--multi", "text");
    const std::string tag = a.str("--tag", "");
    const std::string label = pname + (tag.empty() ? "" : "-" + tag);
    const std::string gparams = "R" + std::to_string(R) + "-L" + std::to_string(L) + "-a" + fmt_num(alpha);
    const std::string index_path =
        a.str("--index", use_rb ? "results/indexes/" + ds.name + "-" + rbname + "-init-" + gparams + ".graph"
                                : "results/indexes/" + ds.name + "-" + gparams + "-" + rbname + ".graph");
    const std::string out = a.str("--out", "results/crossmodal/" + ds.name + "/" + rbname + "/" + label + ".jsonl");
    if (use_rb && (rb.search_steps.empty() || rb.steps[0].op != "build"))
        throw std::runtime_error("runbook must start with build");

    FMatrix base = read_fbin(ds.base_file);
    std::vector<Range> groups;
    FMatrix queries = test_queries(ds, nq, &groups);
    const Range *gi = nullptr, *gt_ = nullptr;
    for (auto &g : groups) {
        if (g.name == image_group) gi = &g;
        if (g.name == text_group) gt_ = &g;
    }
    if (!gi || !gt_) throw std::runtime_error("dataset needs query groups " + image_group + " and " + text_group);
    const int gt_threads = static_cast<int>(a.i("--gt-threads", 16));
    Truth gt;
    if (use_rb) {
        gt = runbook_gt(rb, rb.search_steps[0], base, queries, k, ds.metric, gt_threads);
    } else {
        const std::string gp = ds.dir + "/gt/nq" + std::to_string(queries.n) + "-k" + std::to_string(k) + ".bin";
        if (file_exists(gp)) {
            gt = read_truth(gp);
        } else {
            make_dirs(ds.dir + "/gt");
            gt = knn(base, queries, k, ds.metric, nullptr, gt_threads);
            const std::string tmp = gp + ".tmp" + std::to_string(::getpid());
            write_truth(tmp, gt);
            std::rename(tmp.c_str(), gp.c_str());
        }
    }
    const size_t need = static_cast<size_t>(rounds) * per_round;
    FMatrix learn = read_fbin(ds.learn_file, 0, need);
    if (learn.n < need) throw std::runtime_error("learn set too small for rounds x per-round");

    if (!file_exists(index_path))
        throw std::runtime_error("missing index " + index_path + (use_rb ? " (run ig_runbook --prepare)" : " (run ig_build)"));
    auto idx = Index::load_graph(index_path, base.v.data(), base.n);
    idx->set_R_extra(R_extra);
    auto policy = make_policy(pname, a.kv, threads);
    policy->setup(*idx);

    LineWriter w(out);
    {
        JObj c;
        c.add("type", "config").add("dataset", ds.name).add("runbook", rbname).add("R", R).add("L", L).add("alpha", alpha);
        c.add("R_extra", R_extra).add("threads", threads).add("k", static_cast<int64_t>(k));
        c.add("n_single", static_cast<int64_t>(gi->hi - gi->lo)).add("n_multi", static_cast<int64_t>(gt_->hi - gt_->lo));
        c.add("per_round", static_cast<int64_t>(per_round)).add("index", index_path);
        c.add("index_hash", hex64(idx->graph_hash())).merge(policy->describe()).add("tag", tag).add("cmdline", a.cmdline);
        w.line(c.str());
    }
    std::vector<uint32_t> res(queries.n * k);
    std::vector<float> dist(queries.n * k);
    std::vector<uint64_t> cmps(queries.n);
    auto mean_cmps = [&](const Range &g) {
        double c = 0;
        for (size_t i = g.lo; i < g.hi; ++i) c += static_cast<double>(cmps[i]);
        return c / static_cast<double>(g.hi - g.lo);
    };
    double maintain_total = 0;
    for (long r = 0; r <= rounds; ++r) {
        double ms = 0;
        std::string round = "{}";
        if (r > 0) {
            const double t0 = now_s();
            policy->maintain(*idx, learn.row(static_cast<size_t>(r - 1) * per_round), per_round, static_cast<int>(r - 1));
            ms = now_s() - t0;
            maintain_total += ms;
            round = policy->last_round;
        }
        if (std::find(eval_at.begin(), eval_at.end(), r) == eval_at.end() && r != rounds) continue;
        JObj o;
        o.add("type", "round").add("round", r).add("logged_queries", static_cast<int64_t>(r) * static_cast<int64_t>(per_round));
        o.add("maintain_s", ms).raw("maintain", round);
        for (long Lq : Ls) {
            idx->batch_search(queries.v.data(), queries.n, static_cast<uint32_t>(k), static_cast<uint32_t>(Lq), res.data(),
                              dist.data(), cmps.data(), nullptr, sthreads);
            const double rs = recall_at(res.data() + gi->lo * k, gi->hi - gi->lo, k, gt, k, gi->lo);
            const double rm = recall_at(res.data() + gt_->lo * k, gt_->hi - gt_->lo, k, gt, k, gt_->lo);
            const std::string s = std::to_string(Lq);
            o.add("recall_L" + s + "_single", rs).add("recall_L" + s + "_multi", rm).add("drop_L" + s, rs - rm);
            o.add("cmps_L" + s + "_single", mean_cmps(*gi)).add("cmps_L" + s + "_multi", mean_cmps(*gt_));
        }
        double avg, avg_x;
        uint32_t mx;
        size_t nx;
        idx->degree_stats(avg, mx, avg_x, nx);
        o.add("n_extra_edges", static_cast<int64_t>(nx)).add("graph_hash", hex64(idx->graph_hash()));
        w.line(o.str());
    }
    JObj s;
    s.add("type", "summary").add("maintain_s", maintain_total).raw("policy_stats", policy->stats());
    w.line(s.str());
    std::fprintf(stderr, "-> %s\n", out.c_str());
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

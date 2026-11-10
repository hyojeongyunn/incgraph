#include <cstdio>

#include "common.h"
#include "incgraph/policy.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    const std::string wl = a.req("--workload");
    const std::string wdir = ds.dir + "/workloads/" + wl;
    const std::string pname = a.str("--policy", "noop");
    const uint32_t R = static_cast<uint32_t>(a.i("--R", 32)), L = static_cast<uint32_t>(a.i("--L", 100));
    const double alpha = a.f("--alpha", 1.2);
    const uint32_t R_extra = static_cast<uint32_t>(a.i("--R-extra", 0));
    const uint64_t seed = static_cast<uint64_t>(a.i("--seed", 0));
    const int lag = static_cast<int>(a.i("--lag", 1));
    const int threads = static_cast<int>(a.i("--threads", 1));
    const int sthreads = static_cast<int>(a.i("--search-threads", 8));
    const size_t k = static_cast<size_t>(a.i("--k", 10));
    const std::vector<long> Ls = a.ints("--search-L", "10,15,20,30,40,50,75,100,150");
    const long snap_every = a.i("--snap-every", 1);
    const std::string tag = a.str("--tag", "");
    const std::string label = pname + (tag.empty() ? "" : "-" + tag);
    const std::string index_path =
        a.str("--index", "results/indexes/" + ds.name + "-R" + std::to_string(R) + "-L" + std::to_string(L) + "-a" +
                             fmt_num(alpha) + "-s" + std::to_string(seed) + ".graph");
    const std::string out = a.str("--out", "results/" + ds.name + "/" + wl + "/" + label + ".jsonl");
    const std::string snapdir = "results/snapshots/" + ds.name + "/" + wl + "/" + label;

    JValue meta = read_json(wdir + "/meta.json");
    const int epochs = static_cast<int>(meta["epochs"].i64());
    const size_t n_obs = static_cast<size_t>(meta["n_observe"].i64()), n_eval = static_cast<size_t>(meta["n_eval"].i64());
    std::vector<uint32_t> observe = read_u32(wdir + "/observe.u32"), eval = read_u32(wdir + "/eval.u32");
    Truth gt = read_truth(wdir + "/gt.bin");
    FMatrix learn = read_fbin(ds.learn_file, 0, static_cast<size_t>(meta["learn_rows"].i64()));
    FMatrix base = read_fbin(ds.base_file);
    if (!file_exists(index_path)) throw std::runtime_error("missing index " + index_path + " (run ig_build)");
    auto idx = Index::load_graph(index_path, base.v.data(), base.n);
    idx->set_R_extra(R_extra);
    auto policy = make_policy(pname, a.kv, threads);
    policy->setup(*idx);

    LineWriter w(out);
    {
        JObj c;
        c.add("type", "config").add("dataset", ds.name).add("workload", wl).add("R", R).add("L", L).add("alpha", alpha);
        c.add("R_extra", R_extra).add("seed", static_cast<int64_t>(seed)).add("lag", lag).add("threads", threads);
        c.add("index", index_path).add("index_hash", hex64(idx->graph_hash())).merge(policy->describe());
        c.add("tag", tag).add("cmdline", a.cmdline);
        w.line(c.str());
    }
    std::vector<uint32_t> res(n_eval * k);
    std::vector<float> dist(n_eval * k);
    std::vector<uint64_t> cmps(n_eval);
    double maintain_total = 0;
    for (int e = 0; e < epochs; ++e) {
        const int src = e - lag;
        std::string round = "{}";
        double ms = 0;
        if (src >= 0) {
            std::vector<float> Q = gather_rows(learn, observe.data() + static_cast<size_t>(src) * n_obs, n_obs);
            const double t0 = now_s();
            policy->maintain(*idx, Q.data(), n_obs, src);
            ms = now_s() - t0;
            maintain_total += ms;
            round = policy->last_round;
        }
        std::vector<float> ev = gather_rows(learn, eval.data() + static_cast<size_t>(e) * n_eval, n_eval);
        JObj r;
        r.add("type", "epoch").add("epoch", e).add("maintain_s", ms).raw("round", round);
        for (long Lq : Ls) {
            idx->batch_search(ev.data(), n_eval, static_cast<uint32_t>(k), static_cast<uint32_t>(Lq), res.data(),
                              dist.data(), cmps.data(), nullptr, sthreads);
            r.add("recall_L" + std::to_string(Lq), recall_at(res.data(), n_eval, k, gt, k, static_cast<size_t>(e) * n_eval));
            double c = 0;
            for (auto x : cmps) c += static_cast<double>(x);
            r.add("cmps_L" + std::to_string(Lq), c / static_cast<double>(n_eval));
        }
        double avg, avg_x;
        uint32_t mx;
        size_t nx;
        idx->degree_stats(avg, mx, avg_x, nx);
        r.add("n_extra_edges", static_cast<int64_t>(nx)).add("graph_hash", hex64(idx->graph_hash()));
        if (snap_every > 0 && (e % snap_every == 0 || e == epochs - 1)) {
            make_dirs(snapdir);
            const std::string sp = snapdir + "/epoch" + std::to_string(e) + ".edges";
            write_edges(sp, idx->extra_edge_list());
            r.add("snapshot", sp);
        }
        w.line(r.str());
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

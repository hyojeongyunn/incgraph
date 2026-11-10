#include <cmath>
#include <cstdio>

#include "common.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    const int nt = static_cast<int>(a.i("--ntopics", 32));
    const int epochs = static_cast<int>(a.i("--epochs", 10));
    const size_t n_obs = static_cast<size_t>(a.i("--n-observe", 5000));
    const size_t n_eval = static_cast<size_t>(a.i("--n-eval", 1000));
    const double zipf = a.f("--zipf", 1.2);
    const std::string drift = a.str("--drift", "gradual");
    const int shift = static_cast<int>(a.i("--shift", 2));
    const int period = static_cast<int>(a.i("--period", 3));
    const size_t learn_limit = static_cast<size_t>(a.i("--learn-limit", 0));
    const size_t k = static_cast<size_t>(a.i("--k", 100));
    const uint64_t seed = static_cast<uint64_t>(a.i("--seed", 0));
    const uint64_t kseed = static_cast<uint64_t>(a.i("--kmeans-seed", 0));
    const std::string name = a.req("--name") + "-s" + std::to_string(seed);
    if (drift != "gradual" && drift != "abrupt") throw std::invalid_argument("--drift gradual|abrupt");

    FMatrix learn = read_fbin(ds.learn_file, 0, learn_limit ? learn_limit : SIZE_MAX);
    double t0 = now_s();
    std::vector<int> topic = kmeans_assign(learn.v.data(), learn.n, learn.d, nt, kseed);
    std::fprintf(stderr, "k-means(%d) on %zu learn queries: %.1fs\n", nt, learn.n, now_s() - t0);
    Rng rng(seed);
    std::vector<std::vector<uint32_t>> pools(nt);
    for (size_t i = 0; i < learn.n; ++i) pools[topic[i]].push_back(static_cast<uint32_t>(i));
    for (auto &p : pools) rng.shuffle(p);
    std::vector<double> base_w(nt);
    double tot = 0;
    for (int r = 0; r < nt; ++r) tot += (base_w[r] = 1.0 / std::pow(r + 1.0, zipf));
    for (auto &x : base_w) x /= tot;
    std::vector<uint32_t> ranking = rng.permutation(nt);
    std::vector<uint32_t> observe, eval;
    std::string emeta;
    for (int e = 0; e < epochs; ++e) {
        if (e > 0) {
            if (drift == "gradual")
                std::rotate(ranking.begin(), ranking.end() - (shift % nt), ranking.end());
            else if (e % period == 0)
                ranking = rng.permutation(nt);
        }
        std::vector<double> w(nt), cum(nt);
        for (int r = 0; r < nt; ++r) w[ranking[r]] = base_w[r];
        double c = 0;
        for (int t = 0; t < nt; ++t) cum[t] = (c += w[t]);
        for (size_t j = 0; j < n_obs + n_eval; ++j) {
            const double u = rng.uniform() * c;
            int t = static_cast<int>(std::upper_bound(cum.begin(), cum.end(), u) - cum.begin());
            t = std::min(t, nt - 1);
            if (pools[t].empty())
                for (uint32_t tt : ranking)
                    if (!pools[tt].empty()) {
                        t = static_cast<int>(tt);
                        break;
                    }
            (j < n_obs ? observe : eval).push_back(pools[t].back());
            pools[t].pop_back();
        }
        double top5 = 0;
        for (int r = 0; r < 5 && r < nt; ++r) top5 += base_w[r];
        JObj em;
        em.add("epoch", e).add("top_topic", static_cast<int64_t>(ranking[0])).add("top5_mass", top5);
        emeta += (e ? "," : "") + em.str();
    }
    FMatrix base = read_fbin(ds.base_file);
    FMatrix eq;
    eq.d = learn.d;
    eq.n = eval.size();
    eq.v = gather_rows(learn, eval.data(), eval.size());
    t0 = now_s();
    Truth gt = knn(base, eq, k, ds.metric, nullptr, 0);
    std::fprintf(stderr, "ground truth for %zu eval queries: %.1fs\n", eval.size(), now_s() - t0);

    const std::string dir = ds.dir + "/workloads/" + name;
    make_dirs(dir);
    write_u32(dir + "/observe.u32", observe);
    write_u32(dir + "/eval.u32", eval);
    write_truth(dir + "/gt.bin", gt);
    JObj m;
    m.add("dataset", ds.name).add("name", name).add("ntopics", nt).add("epochs", epochs);
    m.add("n_observe", static_cast<int64_t>(n_obs)).add("n_eval", static_cast<int64_t>(n_eval)).add("zipf", zipf);
    m.add("drift", drift).add("shift", shift).add("period", period).add("seed", static_cast<int64_t>(seed));
    m.add("kmeans_seed", static_cast<int64_t>(kseed)).add("learn_rows", static_cast<int64_t>(learn.n));
    m.add("k", static_cast<int64_t>(k)).add("cmdline", a.cmdline).raw("epoch_info", "[" + emeta + "]");
    FILE *f = std::fopen((dir + "/meta.json").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write meta.json");
    std::fputs(m.str().c_str(), f);
    std::fclose(f);
    std::printf("wrote %s (%d epochs x %zu observe + %zu eval)\n", dir.c_str(), epochs, n_obs, n_eval);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

#include <cstdio>

#include "common.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    Params prm;
    prm.R = static_cast<uint32_t>(a.i("--R", 32));
    prm.L = static_cast<uint32_t>(a.i("--L", 100));
    prm.alpha = static_cast<float>(a.f("--alpha", 1.2));
    const int threads = static_cast<int>(a.i("--threads", 1));
    const uint64_t seed = static_cast<uint64_t>(a.i("--seed", 0));
    const std::string out =
        a.str("--out", "results/indexes/" + ds.name + "-R" + std::to_string(prm.R) + "-L" + std::to_string(prm.L) +
                           "-a" + fmt_num(prm.alpha) + "-s" + std::to_string(seed) + ".graph");
    FMatrix base = read_fbin(ds.base_file);
    Rng rng(seed);
    std::vector<uint32_t> order = rng.permutation(base.n);
    std::vector<float> v = gather_rows(base, order.data(), order.size());
    Index idx(ds.dim, ds.metric, base.n, prm);
    const double t0 = now_s();
    idx.build(v.data(), order.data(), order.size(), threads);
    const double bs = now_s() - t0;
    make_dirs(dirname_of(out));
    const std::string tmp = out + ".tmp";
    idx.save_graph(tmp);
    std::rename(tmp.c_str(), out.c_str());
    double avg, avg_x;
    uint32_t mx;
    size_t nx;
    idx.degree_stats(avg, mx, avg_x, nx);
    JObj o;
    o.add("dataset", ds.name).add("R", prm.R).add("L", prm.L).add("alpha", prm.alpha).add("seed", static_cast<int64_t>(seed));
    o.add("threads", threads).add("build_s", bs).add("avg_degree", avg).add("graph_hash", hex64(idx.graph_hash()));
    o.add("out", out);
    std::puts(o.str().c_str());
    if (FILE *f = std::fopen((out + ".json").c_str(), "w")) {
        std::fputs(o.str().c_str(), f);
        std::fclose(f);
    }
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

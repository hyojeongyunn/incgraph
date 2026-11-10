#include <cstdio>

#include "common.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    const std::string srcname = a.req("--src");
    Dataset src = load_dataset(srcname);
    const size_t n_image = static_cast<size_t>(a.i("--n-image", 800000)), n_text = static_cast<size_t>(a.i("--n-text", 400000));
    const size_t nq_text = static_cast<size_t>(a.i("--nq-text", 5000)), nq_image = static_cast<size_t>(a.i("--nq-image", 5000));
    const size_t iq0 = static_cast<size_t>(a.i("--image-query-start", 900000));
    const size_t n_learn = static_cast<size_t>(a.i("--n-learn", 100000));
    const std::string name = a.str("--name", srcname.substr(0, srcname.find('-')) + "-mm");
    if (n_image > iq0) throw std::invalid_argument("image queries must not overlap the DB images");
    const std::string out = "data/" + name;
    make_dirs(out);

    FMatrix img = read_fbin(src.base_file, 0, n_image), txt = read_fbin(src.learn_file, 0, n_text);
    FMatrix base;
    base.d = img.d;
    base.n = n_image + n_text;
    base.v = img.v;
    base.v.insert(base.v.end(), txt.v.begin(), txt.v.end());
    FMatrix tq = read_fbin(src.query_file, 0, nq_text), iq = read_fbin(src.base_file, iq0, nq_image);
    FMatrix query;
    query.d = img.d;
    query.n = nq_text + nq_image;
    query.v = tq.v;
    query.v.insert(query.v.end(), iq.v.begin(), iq.v.end());
    FMatrix learn = read_fbin(src.learn_file, n_text, n_learn);
    write_fbin(out + "/base.fbin", base.v.data(), base.n, base.d);
    write_fbin(out + "/query.fbin", query.v.data(), query.n, query.d);
    write_fbin(out + "/learn.fbin", learn.v.data(), learn.n, learn.d);
    write_truth(out + "/gt.k100.bin", knn(base, query, 100, src.metric, nullptr, 0));

    const std::string metric = src.metric == Metric::IP ? "ip" : "l2";
    char buf[4096];
    std::snprintf(buf, sizeof buf,
                  "{\n \"name\": \"%s\", \"metric\": \"%s\", \"dim\": %zu,\n \"nb\": %zu, \"nq\": %zu, \"nl\": %zu, "
                  "\"gt_k\": 100,\n \"files\": {\"base\": \"base.fbin\", \"query\": \"query.fbin\", \"learn\": "
                  "\"learn.fbin\", \"gt\": \"gt.k100.bin\"},\n \"base_groups\": {\"image\": [0, %zu], \"text\": [%zu, "
                  "%zu]},\n \"query_groups\": {\"text\": [0, %zu], \"image\": [%zu, %zu]},\n \"source\": \"derived "
                  "from %s by ig_make_mm\"\n}\n",
                  name.c_str(), metric.c_str(), base.d, base.n, query.n, learn.n, n_image, n_image, n_image + n_text,
                  nq_text, nq_text, nq_text + nq_image, srcname.c_str());
    FILE *f = std::fopen((out + "/dataset.json").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write dataset.json");
    std::fputs(buf, f);
    std::fclose(f);
    std::printf("wrote %s: base %zu, query %zu, learn %zu\n", out.c_str(), base.n, query.n, learn.n);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

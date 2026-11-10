#include <cstdio>
#include <cstring>

#include "common.h"

using namespace incgraph;

static void must_equal_prefix(const FMatrix &q, const FMatrix &data, const char *what) {
    if (q.d != data.d || q.n > data.n || std::memcmp(q.v.data(), data.v.data(), q.v.size() * sizeof(float)) != 0)
        throw std::runtime_error(std::string(what) + ": query file is not the prefix of the data file");
}

int main(int argc, char **argv) try {
    Args a(argc, argv);
    const std::string src = a.str("--src", "data/coco_openclip_l");
    const size_t nq = static_cast<size_t>(a.i("--nq", 1000));
    const bool holdout = a.has("--holdout");
    const size_t n_mm = static_cast<size_t>(a.i("--mm", 0));
    if (holdout && n_mm) throw std::runtime_error("--holdout and --mm are exclusive");
    size_t nl = static_cast<size_t>(a.i("--nl", n_mm ? 10000 : 0));
    const std::string name = n_mm ? "coco-mm" : holdout ? "coco-clip-ho" : "coco-clip";
    const std::string out = a.str("--out", "data/" + name);
    const std::string img_file = src + "/data/image.fbin", txt_file = src + "/data/text.fbin";

    FMatrix qi = read_fbin(src + "/query/image.fbin"), qt = read_fbin(src + "/query/text.fbin");
    must_equal_prefix(qi, read_fbin(img_file, 0, qi.n), "image");
    must_equal_prefix(qt, read_fbin(txt_file, 0, qt.n), "text");
    if (nq > qi.n || nq > qt.n) throw std::runtime_error("--nq larger than the query files");
    const size_t n_img = fbin_shape(img_file).first, n_txt = fbin_shape(txt_file).first;
    const size_t c0 = qt.n;
    if (c0 + n_mm > n_txt) throw std::runtime_error("--mm larger than the available captions");
    if (nl == 0) nl = n_txt - c0 - n_mm;
    if (c0 + n_mm + nl > n_txt) throw std::runtime_error("--nl larger than the available captions");
    make_dirs(out);

    std::vector<float> q(qt.row(0), qt.row(0) + nq * qt.d);
    q.insert(q.end(), qi.row(0), qi.row(0) + nq * qi.d);
    write_fbin(out + "/query.fbin", q.data(), 2 * nq, qt.d);
    {
        FMatrix learn = read_fbin(txt_file, c0 + n_mm, nl);
        write_fbin(out + "/learn.fbin", learn.v.data(), learn.n, learn.d);
    }
    size_t nb;
    std::string base_rel = "base.fbin", groups, base_note;
    if (n_mm) {
        FMatrix b = read_fbin(img_file);
        FMatrix t = read_fbin(txt_file, c0, n_mm);
        b.v.insert(b.v.end(), t.v.begin(), t.v.end());
        b.n += t.n;
        write_fbin(out + "/base.fbin", b.v.data(), b.n, b.d);
        nb = b.n;
        groups = "{\"image\": [0, " + std::to_string(n_img) + "], \"text\": [" + std::to_string(n_img) + ", " +
                 std::to_string(nb) + "]}";
        base_note = "data/image ++ data/text[" + std::to_string(c0) + ":" + std::to_string(c0 + n_mm) + "]";
    } else if (holdout) {
        FMatrix b = read_fbin(img_file, qi.n);
        write_fbin(out + "/base.fbin", b.v.data(), b.n, b.d);
        nb = b.n;
        groups = "{\"image\": [0, " + std::to_string(nb) + "]}";
        base_note = "data/image[" + std::to_string(qi.n) + ":] (query images removed)";
    } else {
        nb = n_img;
        base_rel = "../" + src.substr(src.find_last_of('/') + 1) + "/data/image.fbin";
        groups = "{\"image\": [0, " + std::to_string(nb) + "]}";
        base_note = "data/image (contains the query images)";
    }

    JObj j;
    j.add("name", name).add("metric", "ip").add("dim", static_cast<int64_t>(qt.d)).add("nb", static_cast<int64_t>(nb));
    j.add("nq", static_cast<int64_t>(2 * nq)).add("nl", static_cast<int64_t>(nl));
    j.raw("files", "{\"base\": \"" + base_rel + "\", \"query\": \"query.fbin\", \"learn\": \"learn.fbin\"}");
    j.raw("base_groups", groups);
    j.raw("query_groups", "{\"text\": [0, " + std::to_string(nq) + "], \"image\": [" + std::to_string(nq) + ", " +
                              std::to_string(2 * nq) + "]}");
    j.add("source", "OpenCLIP ViT-L/14 embeddings of MS-COCO (" + src + "), made by ig_make_coco");
    j.add("note", "query = query/text[:" + std::to_string(nq) + "] ++ query/image[:" + std::to_string(nq) +
                      "]; learn = data/text[" + std::to_string(c0 + n_mm) + ":" + std::to_string(c0 + n_mm + nl) +
                      "]; base = " + base_note);
    FILE *f = std::fopen((out + "/dataset.json").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + out + "/dataset.json");
    std::fputs((j.str() + "\n").c_str(), f);
    std::fclose(f);
    std::printf("%s\n", j.str().c_str());
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

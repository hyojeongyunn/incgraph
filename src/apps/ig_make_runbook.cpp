#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "incgraph/util.h"

using namespace incgraph;

int main(int argc, char **argv) try {
    Args a(argc, argv);
    Dataset ds = load_dataset(a.req("--dataset"));
    const std::string kind = a.str("--kind", "clustered");
    const int ncl = static_cast<int>(a.i("--nclusters", 16));
    const int init_ncl = static_cast<int>(a.i("--init-nclusters", 16));
    const double init_frac = a.f("--init-frac", 0.5);
    const size_t batch = static_cast<size_t>(a.i("--batch", 10000));
    const double delete_frac = a.f("--delete-frac", 0.0);
    const int consolidate_every = static_cast<int>(a.i("--consolidate-every", 4));
    const int search_every = static_cast<int>(a.i("--search-every", 1));
    const uint64_t seed = static_cast<uint64_t>(a.i("--seed", 0));
    const uint64_t kseed = static_cast<uint64_t>(a.i("--kmeans-seed", 0));
    const std::string init_group = a.str("--init-group", ""), stream_group = a.str("--stream-group", "");
    const std::string name = a.req("--name") + "-s" + std::to_string(seed);
    if (kind != "clustered" && kind != "random") throw std::invalid_argument("--kind clustered|random");

    FMatrix base = read_fbin(ds.base_file);
    const size_t n = base.n;
    std::vector<int> assign(n, -1);
    Rng rng(seed);
    std::vector<std::vector<uint32_t>> members;
    std::vector<uint32_t> init_ids, init_cl, stream_cl;
    std::vector<std::vector<uint32_t>> chunks;
    double t0 = now_s();
    if (!init_group.empty()) {
        const Range *gi = nullptr, *gs = nullptr;
        for (auto &g : ds.base_groups) {
            if (g.name == init_group) gi = &g;
            if (g.name == stream_group) gs = &g;
        }
        if (!gi || !gs) throw std::invalid_argument("unknown --init-group / --stream-group");
        auto ai = kmeans_assign(base.row(gi->lo), gi->hi - gi->lo, base.d, init_ncl, kseed);
        auto as = kmeans_assign(base.row(gs->lo), gs->hi - gs->lo, base.d, ncl, kseed);
        for (size_t i = 0; i < ai.size(); ++i) assign[gi->lo + i] = ai[i];
        for (size_t i = 0; i < as.size(); ++i) assign[gs->lo + i] = init_ncl + as[i];
        members.assign(init_ncl + ncl, {});
        for (size_t i = 0; i < n; ++i)
            if (assign[i] >= 0) members[assign[i]].push_back(static_cast<uint32_t>(i));
        for (auto &m : members) rng.shuffle(m);
        init_cl = rng.permutation(init_ncl);
        for (uint32_t c : rng.permutation(ncl)) stream_cl.push_back(init_ncl + c);
        for (size_t i = gi->lo; i < gi->hi; ++i) init_ids.push_back(static_cast<uint32_t>(i));
        rng.shuffle(init_ids);
        for (uint32_t c : stream_cl) chunks.push_back(members[c]);
        if (kind == "random") {
            std::vector<uint32_t> flat;
            for (size_t i = gs->lo; i < gs->hi; ++i) flat.push_back(static_cast<uint32_t>(i));
            rng.shuffle(flat);
            size_t pos = 0;
            for (auto &c : chunks) {
                std::copy(flat.begin() + pos, flat.begin() + pos + c.size(), c.begin());
                pos += c.size();
            }
        }
    } else {
        auto as = kmeans_assign(base.v.data(), n, base.d, ncl, kseed);
        members.assign(ncl, {});
        for (size_t i = 0; i < n; ++i) members[as[i]].push_back(static_cast<uint32_t>(i));
        for (auto &m : members) rng.shuffle(m);
        std::vector<uint32_t> order = rng.permutation(ncl);
        const size_t n_init = std::max<size_t>(1, static_cast<size_t>(std::llround(init_frac * ncl)));
        init_cl.assign(order.begin(), order.begin() + n_init);
        stream_cl.assign(order.begin() + n_init, order.end());
        for (uint32_t c : init_cl) init_ids.insert(init_ids.end(), members[c].begin(), members[c].end());
        rng.shuffle(init_ids);
        for (uint32_t c : stream_cl) chunks.push_back(members[c]);
        if (kind == "random") {
            std::vector<uint32_t> flat;
            for (auto &c : chunks) flat.insert(flat.end(), c.begin(), c.end());
            rng.shuffle(flat);
            size_t pos = 0;
            for (auto &c : chunks) {
                std::copy(flat.begin() + pos, flat.begin() + pos + c.size(), c.begin());
                pos += c.size();
            }
        }
    }
    std::fprintf(stderr, "k-means + layout: %.1fs\n", now_s() - t0);

    std::vector<uint32_t> ids;
    std::string steps;
    size_t n_steps = 0, n_ins = 0, n_del = 0, n_search = 0;
    auto add = [&](const std::string &op, const std::vector<uint32_t> *v) {
        JObj o;
        o.add("op", op);
        if (v) {
            o.add("off", static_cast<int64_t>(ids.size())).add("n", static_cast<int64_t>(v->size()));
            ids.insert(ids.end(), v->begin(), v->end());
        }
        steps += (n_steps++ ? ",\n  " : "  ") + o.str();
    };
    add("build", &init_ids);
    add("search", nullptr);
    ++n_search;
    std::vector<std::vector<uint32_t>> live_groups;
    if (kind == "clustered") {
        for (uint32_t c : init_cl) live_groups.push_back(members[c]);
    } else {
        const size_t g = init_cl.size();
        for (size_t i = 0; i < g; ++i)
            live_groups.emplace_back(init_ids.begin() + init_ids.size() * i / g,
                                     init_ids.begin() + init_ids.size() * (i + 1) / g);
    }
    int n_del_steps = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const auto &chunk = chunks[ci];
        for (size_t s = 0; s < chunk.size(); s += batch) {
            std::vector<uint32_t> b(chunk.begin() + s, chunk.begin() + std::min(chunk.size(), s + batch));
            add("insert", &b);
            n_ins += b.size();
        }
        live_groups.push_back(chunk);
        if (delete_frac > 0) {
            std::vector<uint32_t> g = live_groups.front();
            live_groups.erase(live_groups.begin());
            const size_t k = static_cast<size_t>(std::llround(delete_frac * static_cast<double>(g.size())));
            std::vector<uint32_t> del(g.begin(), g.begin() + k);
            add("delete", &del);
            n_del += del.size();
            if (g.size() > k) live_groups.emplace_back(g.begin() + k, g.end());
            ++n_del_steps;
            if (consolidate_every && n_del_steps % consolidate_every == 0) add("consolidate", nullptr);
        }
        if ((ci + 1) % search_every == 0 || ci + 1 == chunks.size()) {
            add("search", nullptr);
            ++n_search;
        }
    }

    const std::string dir = ds.dir + "/runbooks/" + name;
    make_dirs(dir);
    write_u32(dir + "/ids.u32", ids);
    JObj meta;
    meta.add("dataset", ds.name).add("name", name).add("kind", kind).add("n_base", static_cast<int64_t>(n));
    meta.add("seed", static_cast<int64_t>(seed)).add("kmeans_seed", static_cast<int64_t>(kseed));
    meta.add("init_group", init_group).add("stream_group", stream_group);
    meta.add("nclusters", ncl).add("init_nclusters", init_ncl).add("batch", static_cast<int64_t>(batch));
    meta.add("delete_frac", delete_frac).add("consolidate_every", consolidate_every);
    meta.add("cmdline", a.cmdline);
    meta.raw("steps", "[\n" + steps + "\n]");
    FILE *f = std::fopen((dir + "/steps.json").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + dir + "/steps.json");
    std::fputs(meta.str().c_str(), f);
    std::fclose(f);
    std::printf("wrote %s: %zu steps, build %zu, insert %zu, delete %zu, searches %zu\n", dir.c_str(), n_steps,
                init_ids.size(), n_ins, n_del, n_search);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}

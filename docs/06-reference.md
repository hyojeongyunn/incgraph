# 6. Reference

## 6.1 Programs

All programs are built into `bin/` and run from the repository root.

- **Options.** Options are written `--key value`.
- **Policy parameters.** They are written `-p key=value`. The flag can be repeated, and an unknown key is an error.
- **Datasets.** `--dataset NAME` reads `data/NAME/dataset.json`.
- **Checkpoint lists.** Options such as `--snap` take comma-separated checkpoint ordinals, where −1 is the last checkpoint.

### Data preparation

| Program | Purpose | Options (default) |
|---|---|---|
| `ig_make_mm` | Builds a data-change dataset `data/<name>`. Base = images (or videos) followed by texts; queries = texts + held-out images; learn = further texts. | `--src` (`t2i-1M` or `webvid-1M`), `--n-image` (800000), `--n-text` (400000), `--nq-text` (5000), `--nq-image` (5000), `--image-query-start` (900000), `--n-learn` (100000), `--name` (`<src prefix>-mm`) |
| `ig_make_coco` | Builds `coco-clip` from COCO embeddings; `coco-clip-ho` with `--holdout`; `coco-mm` with `--mm N` (images + `N` captions, for data changes) | `--src` (`data/coco_openclip_l`), `--nq` (1000 per modality), `--holdout`, `--mm` (0), `--nl` (all remaining captions), `--out` |
| `ig_make_runbook` | Builds a data-change workload `data/<ds>/runbooks/<name>-s<seed>/` | `--dataset`, `--name`, `--init-group`, `--stream-group`, `--nclusters` (16), `--init-nclusters` (16), `--batch` (10000), `--delete-frac` (0), `--consolidate-every` (4), `--search-every` (1), `--kind` (`clustered` or `random`), `--init-frac` (0.5, for datasets without groups), `--seed` (0), `--kmeans-seed` (0) |
| `ig_make_drift` | Builds a query-change workload `data/<ds>/workloads/<name>-s<seed>/` | `--dataset`, `--name`, `--drift` (`gradual` or `abrupt`), `--epochs` (10), `--period` (3, abrupt), `--shift` (2, gradual), `--ntopics` (32 clusters), `--zipf` (1.2), `--n-observe` (5000), `--n-eval` (1000), `--k` (100), `--learn-limit` (0 = all), `--seed` (0), `--kmeans-seed` (0) |

### Index construction and modification

| Program | Purpose | Options (default) |
|---|---|---|
| `ig_build` | Builds a static index over all base rows, inserted in a seeded random order, and saves it with `save_graph`. Writes the build time and the graph hash to `<out>.json`. | `--dataset`, `--R` (32), `--L` (100), `--alpha` (1.2), `--seed` (0), `--threads` (1), `--out` (`results/indexes/<ds>-R<R>-L<L>-a<alpha>-s<seed>.graph`) |
| `ig_runbook` | Runs a data-change workload with a policy. Logs recall and distance computations at every checkpoint. | `--dataset`, `--runbook`, `--policy` (`noop`), `-p`, `--R` (32), `--L` (100), `--alpha` (1.2), `--R-extra` (0), `--threads` (1), `--search-threads` (8), `--k` (10), `--nq` (10000), `--search-L` (10,20,30,50,75,100,150,200,300), `--search-at` (`all`), `--snap` (`none`), `--prepare` (only the shared initial index and the ground truth), `--gt-threads` (16), `--tag`, `--out` |
| `ig_drift` | Runs a query-change workload with a policy. Before evaluating period `e`, it runs one maintenance round on the log of period `e − lag`. | `--dataset`, `--workload`, `--policy` (`noop`), `-p`, `--seed` (0, selects the static index), `--index`, `--R-extra` (0), `--lag` (1), `--threads` (1), `--search-threads` (8), `--k` (10), `--search-L` (10,15,20,30,40,50,75,100,150), `--snap-every` (1; 0 = no overlay snapshots), `--tag`, `--out` |
| `ig_crossmodal` | Measures the cross-modal accuracy drop on an image-only index: static (round 0), then after maintenance rounds on logged text queries | `--dataset`; one of `--seed s` (static index of `ig_build`) or `--runbook R` (initial index of a data-change runbook); `--policy` (`noop`), `-p`, `--R-extra` (0), `--rounds` (0), `--per-round` (5000), `--eval-at` (0,1,2,5,10,20), `--single` (`image`), `--multi` (`text`), `--search-L` (10,15,20,30,50,75,100,150,200,300,500), `--gt-threads` (16), `--tag`, `--out` |

### Measurement and tests

| Program | Purpose | Options (default) |
|---|---|---|
| `ig_curve` | Recall–QPS curve from snapshots, on one thread. Per search width: one warm-up pass and `--reps` timed passes; `qps` comes from the fastest pass and `qps_median` from the median pass. | Data change: `--dataset --runbook --step --graphs a.graph,b.graph --labels A,B`. Query change: `--dataset --workload --index base.graph [--overlays dir] --epochs list --label X`. Both: `--nq` (10000), `--reps` (5), `--k` (10), `--search-L`, `--R-extra` (64, overlay capacity on load), `--out` |
| `ig_test` | Tests on synthetic data: deterministic build, recall, snapshot round trip, delete and consolidation, the return edges of `oodbridge`, and the addition, decay and budget of `comember` | none |

## 6.2 Scripts

Each script reads its settings from environment variables; defaults are in parentheses.

| Script | Purpose | Environment variables (default) |
|---|---|---|
| `scripts/build.sh` | CMake + Ninja build into `bin/` | `BUILD_TYPE` (`Release`) |
| `data/download.sh [t2i\|webvid\|all]` | Downloads the first vectors of the public T2I and CLIP WebVid files | `NB` (1000000), `NQ` (10000), `NL` (1000000) |
| `scripts/exp_incremental.sh` | Data-change datasets, runbooks, initial indexes and runs | `DATASETS` (`t2i-mm webvid-mm`), `RUNBOOKS` (`mm-clustered mm-clustered-del`), `METHODS` (`noop oodbridge`), `SEEDS` (`0 1 2 3 4`), `JOBS` (8), `SNAP` (`8,-1`) |
| `scripts/exp_adaptive.sh` | Query-change workloads, static indexes and runs | `DATASETS` (`t2i-1M webvid-1M coco-clip`), `WORKLOADS` (`abrupt32 gradual32-long`), `SEEDS`, `JOBS` (8), `BUDGET` (300000) |
| `scripts/exp_crossmodal.sh` | Cross-modal accuracy drop | `SETS` (`coco-clip coco-clip-ho`; also `t2i-mm`, `webvid-mm`), `SEEDS`, `ROUNDS` (20) |
| `scripts/exp_curves.sh` | QPS curves of every snapshot; finished outputs are skipped | `TASKS` (`1 2`; 1 = data change, 2 = query change), `STEPS` (`all`, `last` or `rest`), `REPS` (3), `NQ` (10000), `L1`, `L2` (search widths), `T1_DATASETS`, `T2_DATASETS`, `LANES` (0) |
| `scripts/exp_costs.sh` | Update times, maintenance times and the static build time | `T1_DATASETS`, `T2_DATASETS`, `LANES` (0) |
| `scripts/exp_measure_all.sh` | `exp_curves.sh` for the last checkpoints and the query changes, then `exp_costs.sh`, then the remaining checkpoints | `LANES` |
| `scripts/lanes.sh` | `run_lanes`: runs a job list on the cores in `LANES`, one job per core at a time | none |
| `scripts/plot_curves.py` | Writes the figures and `summary.json` (QPS at recall 0.8, 0.9 and 0.95 for every curve) to `results/figures/`, and prints the cross-modal tables | none |

## 6.3 File formats

All integers and floats are little-endian.

| File | Layout |
|---|---|
| `*.fbin` | `[int32 n][int32 d][float32 n·d]`. Row `i` has label `i`. |
| `*.u32` | `[uint64 n][uint32 n]` |
| Ground truth (`gt.bin`, `gt/*.bin`) | `[int32 n][int32 k][uint32 ids n·k][float32 dists n·k]` |
| `dataset.json` | `name`, `metric` (`ip` or `l2`), `dim`, and `files` {`base`, `query`, `learn`}, with paths relative to the dataset directory. Optional: `base_groups` and `query_groups` {`name`: [`lo`, `hi`)}, as row ranges. |
| Runbook directory | `steps.json`: workload parameters and `steps` = [{`op`: `build`, `insert`, `delete`, `consolidate` or `search`, `off`, `n`}]. `ids.u32`: the labels that a step addresses as `ids[off : off + n]`. `gt/`: cached ground truth per checkpoint. |
| Workload directory | `meta.json`: parameters, plus the most popular cluster and the share of the top-5 clusters per period. `observe.u32` and `eval.u32`: learn-set row ids, period after period (`n_observe` or `n_eval` per period). `gt.bin`: the top-100 of every evaluation query over the whole base. |
| `*.graph` | Graph-only snapshot (`save_graph`): labels, base and overlay adjacency lists, entry vector |
| `*.edges` | Overlay edges, as a `*.u32` of `from, to` pairs |
| COCO input (`data/coco_openclip_l/`) | `data/image.fbin`: all images. `data/text.fbin`: all captions. `query/image.fbin`: the query images, a prefix of `data/image.fbin`. `query/text.fbin`: the captions of the query images, a prefix of `data/text.fbin`. `ig_make_coco` checks the prefixes. |

## 6.4 Result files

| Output | Path |
|---|---|
| Data change | `results/<ds>/<runbook>/<policy>[-tag].jsonl`. Snapshots: `results/snapshots/<ds>/<runbook>/<policy>[-tag]/step<i>.graph`. |
| Query change | `results/<ds>/<workload>/<policy>[-tag].jsonl`. Overlays: `results/snapshots/<ds>/<workload>/<policy>[-tag]/epoch<e>.edges`. |
| Cross-modal | `results/crossmodal/<ds>/<s<seed> or runbook>/<policy>[-tag].jsonl` |
| Curves | `results/curves/<ds>/<runbook>/<label>/step<i>.jsonl` and `results/curves/<ds>/<workload>/<label>.jsonl` |

Result files are JSON lines.

- **First line** (`type: config`): every parameter and the command line.
- **Last line** (`type: summary`): times and policy statistics.
- **Lines in between:**

| `type` | Written by | Main fields |
|---|---|---|
| `search` | `ig_runbook` | `recall_L<L>` and `cmps_L<L>` (plus `_<group>` per query group), `n_live`, `n_extra_edges`, `graph_hash`, update times |
| `epoch` | `ig_drift` | `recall_L<L>`, `cmps_L<L>`, `maintain_s`, statistics of the round, `graph_hash` |
| `round` | `ig_crossmodal` | `recall_L<L>_single`, `recall_L<L>_multi`, `drop_L<L>`, `cmps_L<L>_*`, `logged_queries` |
| `curve` | `ig_curve` | `L`, `recall`, `qps`, `qps_median`, `cmps` (plus `_<group>`) |

## 6.5 C++ API

Include `incgraph/index.h` and `incgraph/policy.h`, and link the CMake target `incgraph_core`. Everything is in the namespace `incgraph`.

### `Index`

Constructor: `Index(dim, Metric::L2 or Metric::IP, capacity, Params{R, L, alpha, max_candidates, slack, R_extra})`.

| Group | Methods |
|---|---|
| Data | `build(data, labels, n, threads)`, `insert(data, labels, n, threads, alpha_override)`, `lazy_delete(labels, n)`, `consolidate(threads)` |
| Search | `search(q, k, L, out_labels, out_dists, stats, entry_labels, group)`, `batch_search(queries, nq, k, L, out_labels, out_dists, out_cmps, out_hops, threads, group)`, `search_trace(q, L, labels, dists, stats)` |
| Groups | `set_group(labels, g)`, `group_of(label)` |
| Edges | `neighbors`, `extra_neighbors`, `set_neighbors`, `add_edge`, `remove_edge`, `add_extra_edge`, `remove_extra_edge`, `clear_extra_edges`, `set_R_extra`, `robust_prune(label, candidates, alpha, R)` (computes only; edits nothing), `in_neighbors` |
| Inspection | `size`, `num_deleted`, `contains`, `labels`, `vector`, `distance`, `distance_to`, `reachability(include_extra)`, `degree_stats`, `extra_edge_list` |
| Persistence | `save`, `load`, `save_graph`, `load_graph(path, base, n_base)`, `graph_hash` |

### `Policy`

`make_policy(name, {key: value}, threads)` returns `noop`, `oodbridge` or `comember`. A policy is used through these calls:

| Call | When |
|---|---|
| `setup(idx)` | Once, before the first round |
| `insert(idx, vecs, labels, n)`, `remove(idx, labels, n)`, `consolidate(idx)` | Data-change rounds |
| `maintain(idx, queries, nq, epoch)` | Query-change rounds |
| `describe()`, `stats()` | Parameters and statistics as JSON, for logging |

### Examples

Incremental modification:

```cpp
#include "incgraph/policy.h"
using namespace incgraph;

Params prm;
prm.R = 32; prm.L = 100; prm.alpha = 1.2f; prm.R_extra = 1;
Index idx(dim, Metric::IP, capacity, prm);
idx.build(init_vecs, init_labels, n_init, 1);

auto policy = make_policy("oodbridge", {}, 1);
policy->setup(idx);
policy->insert(idx, new_vecs, new_labels, n_new);
policy->remove(idx, deleted_labels, n_deleted);
policy->consolidate(idx);
```

Workload-adaptive modification:

```cpp
auto idx = Index::load_graph("results/indexes/webvid-1M-R32-L100-a1.2-s0.graph", base, n_base);
idx->set_R_extra(16);
auto policy = make_policy("comember", {{"removal", "score"}}, 1);
policy->setup(*idx);
for (int e = 1; e < n_periods; ++e)
    policy->maintain(*idx, query_log[e - 1], log_size, e - 1);
```

### Adding a policy

1. Derive from `Policy` in `policy.h`. Override `insert`, `remove` and `consolidate` for data changes, or `maintain` for query changes.
2. Read the parameters in the constructor with `geti`, `getf` and `gets`. Unknown keys are then rejected automatically.
3. Register the class in `make_policy`.
4. Keep the policy deterministic:
   - edit the graph in a fixed order, never by iterating an unordered container;
   - draw random numbers only from `incgraph::Rng`;
   - run graph edits with `threads_` threads.

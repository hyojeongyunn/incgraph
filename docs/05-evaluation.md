# 5. Evaluation

Each technique is compared with the **unmodified index**: the same index with policy `noop`. For data changes this means plain Vamana insert, lazy delete and consolidation. For query changes it means the static index.

## 5.1 Protocol

| Item | Setting |
|---|---|
| Graph | `R` = 32, insert width `L` = 100, `α` = 1.2 for every index |
| Updates | One thread ([§2.6](02-base-index.md#26-persistence-and-determinism)) |
| Repetitions | Seeds 0–4. All k-means clusterings use k-means seed 0, so the clusters are the same for every seed. |
| Accuracy | Recall@10 against the exact top-10 over the points that are live at that moment |
| Speed | QPS measured on saved snapshots on an otherwise idle machine. One thread, one query at a time. Per search width: one warm-up pass, then 3 timed passes; the fastest pass counts. At most one measurement per CPU socket, pinned to a core: a second measurement on the same socket lowered QPS by about 8%. |
| Curves | One point per search width `L`: mean recall and geometric-mean QPS over the seeds |
| QPS at recall r | Interpolated linearly in log QPS between the two neighboring search widths |
| Machine | 2 × Intel Xeon Gold 6130 (16 cores each), 503 GB RAM, Ubuntu 22.04, GCC 11.4, `-O3 -march=native` |

## 5.2 Datasets

| Name | Indexed vectors | Evaluated queries | Dim, metric | Source |
|---|---|---|---|---|
| `t2i-1M` | First 1M image embeddings of Yandex Text-to-Image | Text | 200, inner product | big-ann-benchmarks |
| `webvid-1M` | First 1M video-frame embeddings of CLIP WebVid | Captions | 512, inner product | RoarGraph release (Zenodo record 11090378) |
| `t2i-mm`, `webvid-mm` | 800K images (videos) at the start; 400K texts inserted | 5,000 texts + 5,000 held-out images (videos) | As above | Derived from the 1M sets |
| `coco-clip` | All 123,287 MS-COCO images | First 1,000 captions + first 1,000 images of the query files | 768, inner product | OpenCLIP ViT-L/14 embeddings, not distributed |
| `coco-clip-ho` | `coco-clip` without its 5,000 query images (118,287) | Same as `coco-clip` | 768, inner product | As `coco-clip` |

- **Query logs.** The query logs and the query pools of the workloads come from each dataset's learn set:
  - T2I and WebVid: the first 1M training queries;
  - COCO: the 591,753 captions of the images that are not query images, so they never overlap the evaluated queries.
- **COCO.** The COCO embeddings are not distributed with this repository. The COCO results can be reproduced only with embeddings of the same layout ([§6.3](06-reference.md#63-file-formats)).

## 5.3 Incremental modification

**Workloads.**
- The 400K texts of `t2i-mm` and `webvid-mm` are clustered into 16 k-means clusters and inserted cluster by cluster, in batches of 10K.
- There is a checkpoint before the first insert and after every cluster (17 in total). Snapshots for the QPS curves are taken at checkpoint 8 and at the last one.
- The seed changes the cluster order, the order within batches and the deleted points.
- Technique 1 runs with `R_extra` = 1.

| Runbook | Steps |
|---|---|
| `mm-clustered` | Inserts only. The index grows from 800K to 1.2M points. |
| `mm-clustered-del` | After each cluster, half of the oldest live image group (of 16) is deleted. A consolidation follows every 4 delete batches. About 800K points stay live. |

**T2I, image queries** (existing distribution):

| | Unmodified | Incremental modification |
|---|---|---|
| Recall@10 at `L` = 20, before the first insert | 0.957 | 0.957 |
| Same, range over the 16 checkpoints after it, `mm-clustered` | 0.79–0.84 | 0.948–0.959 |
| Same, `mm-clustered-del` | 0.77–0.83 | 0.943–0.963 |
| QPS at recall 0.95, last checkpoint, `mm-clustered` | 2,516 | **9,555 (3.8×)** |
| QPS at recall 0.95, last checkpoint, `mm-clustered-del` | 1,429 | **10,330 (7.2×)** |

**T2I, text queries** (inserted distribution):
- Both indexes reach the same maximum recall, 0.989.
- QPS at recall 0.95 is 4,501 → 4,288 (−5%) on `mm-clustered` and 4,384 → 4,094 (−7%) on `mm-clustered-del`.

**WebVid.**
- Inserting captions does not degrade video queries in either index: QPS at recall 0.95 is 3,849 (unmodified) and 3,841 (incremental modification).
- The inserted captions lie so far from the videos that searches for video queries do not enter their region.
- Caption queries are 5–6% slower with incremental modification.

**Update time** (400K inserts, one thread):

| Dataset | Unmodified | Incremental modification | Of which: OOD test and return edges |
|---|---|---|---|
| T2I | 141 s | 263 s | 95 s |
| WebVid | 320 s | 543 s | 174 s |

## 5.4 Workload-adaptive modification

**Workloads.**
- The learn queries are clustered into 32 k-means clusters.
- Cluster popularity follows a Zipf law (exponent 1.2) over a ranking of the clusters: the top cluster receives 32% of the queries and the top five receive 66%.
- Each period samples 5,000 logged queries and 1,000 evaluation queries.
- Before period `e` is evaluated, one maintenance round runs on the log of period `e − 1`.
- The seed changes the insertion order of the static index and the sampled queries.
- Workload-adaptive modification runs with `R_extra` = 16.
- Results are averaged over all periods and seeds.

| Workload | Periods | Ranking |
|---|---|---|
| `abrupt32` | 12 | A new random ranking every 3 periods |
| `gradual32-long` | 30 | The ranking rotates by 2 every period |

**Accuracy and speed** (`removal=score`, no budget):

| Dataset | Workload | Recall@10, unmodified → adaptive | QPS at recall r, unmodified → adaptive |
|---|---|---|---|
| WebVid | `abrupt32` | 0.749 → 0.862 (`L` = 100) | r = 0.80: 959 → **1,807 (1.9×)** |
| WebVid | `gradual32-long` | 0.759 → 0.861 (`L` = 100) | r = 0.80: 1,018 → **1,771 (1.7×)** |
| COCO | `abrupt32` | 0.901 → 0.954 (`L` = 30) | r = 0.90: 3,939 → **4,920 (1.25×)** |
| COCO | `gradual32-long` | 0.902 → 0.950 (`L` = 30) | r = 0.90: 3,971 → **4,779 (1.20×)** |
| T2I | `abrupt32` | 0.926 → 0.941 (`L` = 100) | r = 0.90: 3,268 → 3,442 (1.05×) |
| T2I | `gradual32-long` | 0.930 → 0.941 (`L` = 100) | r = 0.90: 3,400 → 3,517 (1.03×) |

- **COCO width.** For COCO the table uses `L` = 30, the smallest width at which the unmodified index exceeds recall 0.9. At `L` = 100 both indexes are above 0.97.
- **Period-to-period variation.** On WebVid `abrupt32`, the unmodified recall at `L` = 100 varies between 0.70 and 0.81 across periods and seeds.

**Edge removal** (`abrupt32`; `removal=score` vs. `removal=none`):

| Dataset | Overlay edges at the end, no budget: score / none | Recall@10, no budget: score / none | Recall@10, budget of 300K edges: score / none |
|---|---|---|---|
| WebVid (`L` = 100) | 2.39M / 4.75M | 0.862 / 0.854 | **0.819 / 0.786** |
| COCO (`L` = 30) | 1.31M / 1.73M | 0.954 / 0.954 | **0.936 / 0.926** |
| T2I (`L` = 100) | 3.29M / 8.03M | 0.941 / 0.943 | 0.930 / 0.929 |

- **No budget.** Removal keeps 24–59% fewer edges at about the same recall.
- **Budget.** With a budget, removal gives higher recall.
- **`gradual32-long`.** Without removal the overlay grows to 7.50M edges (WebVid) and 12.4M edges (T2I) over 30 periods; with removal it stays at 2.43M and 3.40M.

**Maintenance time** (5,000 logged queries per round, one thread): 9.8 s on T2I and 15.7 s on WebVid. For comparison, a static build of the same 1M vectors takes 514 s and 1,072 s.

## 5.5 Cross-modal accuracy drop

**Measure.**
- Build an index over images only.
- At the same search width, compare recall@10 of image queries (single modal) with that of text queries (multi modal). The **drop** is image recall minus text recall.
- Then run workload-adaptive modification for 20 rounds of 5,000 logged captions (100K in total) from the learn set.

**Results** (`coco-clip`, seeds 0–4; the seed is the insertion order of the static index):

| `L` | Image-query recall@10 | Drop, static | Drop, adaptive | Text-query distance computations, static → adaptive |
|---|---|---|---|---|
| 10 | 0.959 | 0.217 | 0.101 | 411 → 509 |
| 20 | 0.985 | 0.135 | 0.056 | 564 → 694 |
| 50 | 0.996 | 0.054 | 0.023 | 937 → 1,139 |
| 100 | 0.998 | 0.023 | **0.010** | 1,439 → 1,728 |
| 150 | 0.999 | 0.014 | **0.006** | 1,871 → 2,231 |
| 200 | 0.999 | 0.009 | 0.004 | 2,264 → 2,685 |

- **Reduction.** The drop falls by 53–58% at every width in the table.
- **Image queries.** Image-query recall does not fall; it rises slightly (0.959 → 0.969 at `L` = 10).
- **Query images in the index.** In `coco-clip` the query images are also indexed, which inflates image-query recall and therefore the drop. Without them (`coco-clip-ho`) the result is nearly the same: 0.023 → 0.009 at `L` = 100 and 0.014 → 0.005 at `L` = 150.

## 5.6 Reproduction

Run from the repository root after building ([README](../README.md#build)):

```bash
data/download.sh all
scripts/exp_incremental.sh
scripts/exp_adaptive.sh
scripts/exp_crossmodal.sh
LANES="2 18" scripts/exp_measure_all.sh
python3 scripts/plot_curves.py
```

| Step | Produces |
|---|---|
| `data/download.sh all` | `t2i-1M` and `webvid-1M` (about 6 GB) |
| `scripts/exp_incremental.sh` | Recall of §5.3 |
| `scripts/exp_adaptive.sh` | Recall of §5.4 |
| `scripts/exp_crossmodal.sh` | §5.5 |
| `scripts/exp_measure_all.sh` | QPS curves and update times of §5.3 and §5.4 |
| `scripts/plot_curves.py` | Figures and `summary.json` in `results/figures/` |

- **Missing datasets.** Datasets that are not present are skipped. This includes COCO when `data/coco_openclip_l` is absent.
- **Run order.** The recall scripts can run in parallel and can share the machine. `exp_measure_all.sh` must run alone.
- **`LANES`.** Lists one core per CPU socket for the timed runs.
- **Configuration.** The datasets, seeds and parallelism of every script are set by environment variables, listed in [§6.2](06-reference.md#62-scripts).

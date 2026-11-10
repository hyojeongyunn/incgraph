# 3. Incremental modification

Technique for data changes. Policy `oodbridge`; class `OODBridgePolicy` in `src/include/incgraph/policy.h` and `src/core/policies.cpp`.

## 3.1 Problem

Vectors of a different distribution are inserted in bulk, for example text embeddings into an index of image embeddings.

1. Text and image embeddings share one space but lie far apart.
2. Inserting a text adds reverse edges from images to it, so searches for image queries step into the text region.
3. Inside that region, RobustPrune keeps the short edges among the texts and discards the long edges back to the images.
4. A search that has entered the text region therefore cannot return within its search width, and recall for image queries drops.

The technique gives the inserted points that lie outside the existing distribution a way back to it, without changing how points are inserted. [§5.3](05-evaluation.md#53-incremental-modification) reports the measured effect.

## 3.2 Method

**Terms.**

| Term | Meaning |
|---|---|
| Anchors | The points in the index before the first insert. Together they form the **existing distribution**, and they make up node group 0. |
| Out-of-distribution (OOD) point | An inserted point whose distance to its nearest anchor exceeds the threshold `τ` |
| Return edge | An overlay edge from an OOD point to one of its nearest anchors |

**Calibration** (once, before the first insert):

1. Sample 2,000 anchors with a seeded RNG.
2. Search for the nearest other anchor of each sampled anchor, with width `L_bridge`.
3. Set `τ` to the (100 − `q`)-th percentile of these nearest-neighbor distances.

**Insert batch:**

1. Insert the batch with plain Vamana insertion ([§2.3](02-base-index.md#23-insert)).
2. For every new point `p`, search the anchors only (group 0, width `L_bridge`) for its `m` nearest anchors.
3. If `p` is an OOD point, add return edges `p → a` to these `m` anchors. A point inside the existing distribution gets no edge.
4. Put new points in other groups (OOD = 1, others = 2). New points never become anchors, so the existing distribution does not drift with the inserted stream.

**Delete batch:** lazy delete ([§2.4](02-base-index.md#24-delete)). Deleted OOD points are forgotten.

**Consolidation:**

1. Consolidate ([§2.4](02-base-index.md#24-delete)). This also drops return edges whose anchor was deleted.
2. Every live OOD point with fewer than `m` return edges searches the anchors again and gets new ones.

**Effect.** A search for an existing-distribution query that wanders into the inserted region expands OOD points. Each of them has a return edge to its nearest anchor, so the search gets back to the existing distribution after a few hops instead of spending its whole width in the inserted region.

## 3.3 Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `m` | 1 | Return edges per OOD point |
| `q` | 1 | `τ` is the (100 − `q`)-th percentile of the anchors' nearest-neighbor distances |
| `L_bridge` | 50 | Search width of the calibration and anchor searches |
| `seed` | 0 | RNG seed of the calibration sample |

The index must have an overlay capacity `R_extra` of at least `m`.

## 3.4 Complexity

- **Memory:** at most `m` overlay edges per inserted point, and only for OOD points.
- **Time:**
  - one search, restricted to the anchors, per inserted point;
  - 2,000 searches once, for the calibration;
  - one anchor search per OOD point that lost a return edge in a consolidation.

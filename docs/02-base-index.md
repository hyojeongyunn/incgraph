# 2. Base index

Both techniques modify one index: a dynamic Vamana graph with FreshDiskANN-style updates. It is implemented in `src/include/incgraph/index.h` and `src/core/index.cpp`. The Vamana and FreshDiskANN algorithms are reimplemented here from their papers ([§2.7](#27-references)); no third-party source code is included.

## 2.1 Graph

- **Labels.** Every point has a user label (`uint32`).
- **Edges.** Every point has two adjacency lists:
  - **base edges**: at most `R`;
  - **overlay edges**: at most `R_extra` ([§2.5](#25-overlay-edges)).
- **Entry point.** Search starts from a frozen entry point.
  - Its vector is a copy of the point of the first inserted batch that is closest to the batch centroid.
  - It is never deleted or returned.
  - Its label in neighbor lists is `ENTRY_ID` (`0xFFFFFFFF`).
- **Distances.** Two distances are supported, and for both a smaller value is closer:
  - L2, as the squared Euclidean distance;
  - inner product, as `−⟨q, x⟩`.

## 2.2 Search

Search is greedy best-first search with a sorted candidate list of size `L`.

1. Put the entry point in the list.
2. Repeatedly expand the closest unexpanded candidate. Expanding a point computes the distance to each of its base and overlay neighbors not seen before and inserts them into the list, which keeps the best `L`.
3. Stop when every candidate in the list is expanded. Return the best `k` live points.

The cost of a search is counted as the number of distance computations. A search can also be restricted to one **node group** (`set_group`). Points outside the group are then neither visited nor returned.

## 2.3 Insert

Insert follows Vamana, with the reverse-edge slack of DiskANN.

1. Search for the new point `p` with the insert width `L`, and collect the expanded points as candidates (at most 750).
2. **RobustPrune** the candidates to at most `R` neighbors with parameter `α`.
   - Rounds run with α_t = 1, 1.2, 1.44, … ≤ α.
   - In each round, a candidate is dropped if an already chosen neighbor occludes it.
   - Inner product uses the DiskANN rule.
3. Add the reverse edge `n → p` to every chosen neighbor `n`. A list that grows beyond `1.3 R` is RobustPruned back to `R`.

`build` inserts a whole set: the first 1,000 points on one thread, then the rest. Every list longer than `R` is then pruned to `R`.

## 2.4 Delete

Deletes follow FreshDiskANN.

- **Lazy delete** marks a point as a tombstone. Search still traverses it, but never returns it.
- **Consolidation** removes the tombstones in one batch:
  1. Every live point that has a base edge to a tombstone takes its live neighbors plus the tombstone's out-neighbors as candidates, and RobustPrunes them to `R`.
  2. Overlay edges that point to tombstones are dropped.
  3. The tombstones' slots are freed for reuse.

## 2.5 Overlay edges

Overlay edges form a second, separate adjacency list per point. Both techniques add their edges here.

- **Use.** Search and insertion search follow overlay edges exactly like base edges.
- **Isolation.** RobustPrune, reverse-edge handling and consolidation never rewrite overlay edges. The only exception is consolidation, which drops overlay edges to deleted points.
- **Rationale.**
  - The edges that the techniques need are the ones RobustPrune would discard: long edges, or edges justified by queries rather than by geometry.
  - In a separate list they survive later inserts.
  - They can be removed without touching the base graph.
  - The base graph stays an unmodified Vamana graph.
- **Duplicates.** An edge that already exists as a base edge is not added again.

## 2.6 Persistence and determinism

- **Snapshots.**
  - `save` / `load` write and read the whole index, vectors included.
  - `save_graph` / `load_graph` write and read only the graph; the vectors are taken from a base matrix, with label = row id.
  - Node groups are not stored.
- **Graph hash.** `graph_hash` is an FNV-1a hash of the labels, both adjacency lists in slot order, and the entry vector. Equal hashes mean identical indexes.
- **Determinism.**
  - Every update (`build`, `insert`, `consolidate`, policies) takes a thread count. With one thread, the same inputs and seeds give the same graph on every run.
  - Searches are read-only, so their results do not depend on the thread count.

## 2.7 References

- **Vamana and RobustPrune:** S. J. Subramanya, F. Devvrit, R. Kadekodi, R. Krishnaswamy, H. V. Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
- **Insert slack, lazy delete and consolidation:** A. Singh, S. J. Subramanya, R. Krishnaswamy, H. V. Simhadri. *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search.* arXiv:2105.09613, 2021.

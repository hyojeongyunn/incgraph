# 1. Overview

## 1.1 Problem

A graph ANN index such as Vamana links each vector to a few neighbors. The links are chosen when the vector is inserted, from the vectors that are in the index at that time. Two kinds of change degrade the graph afterwards.

- **Data change.** Inserted vectors may not follow the distribution of the indexed data, and they may arrive in bulk and in a non-uniform order.
  - An example is text embeddings inserted cluster by cluster into an index of image embeddings.
  - Queries for the existing data then get lost in the region of the new vectors, so accuracy at a fixed search cost drops.
- **Query change.** The set of frequent queries shifts over time.
  - A query is expensive when the vectors around it are poorly connected. This is common when queries come from a different modality than the data, such as text queries over image vectors.
  - Which regions matter changes with the workload, so a graph shaped once at build time does not stay suited to it.

Rebuilding the index after every change would repeat the full build cost each time. incgraph instead modifies the existing graph.

## 1.2 Operating model

The index serves queries. Changes are applied only in periodic **maintenance rounds**, never while a query runs. Each round receives the input of one period:

- the inserts and deletes of the period, for data changes;
- the **query log** of the period, i.e. the queries the index served, for query changes.

## 1.3 Techniques

| Change | Technique | Policy name | Page |
|---|---|---|---|
| Data | Incremental modification | `oodbridge` | [3](03-incremental-modification.md) |
| Queries | Workload-adaptive modification | `comember` | [4](04-workload-adaptive-modification.md) |

Both techniques only add and remove edges, and they keep every added edge apart from the Vamana edges ([§2.5](02-base-index.md#25-overlay-edges)). Neither changes the search algorithm.

## 1.4 Terms

Terms that are used on several pages are defined here. Terms of a single technique are defined on its page.

| Term | Meaning |
|---|---|
| Search width `L` | Size of the candidate list of greedy search ([§2.2](02-base-index.md#22-search)). A larger `L` gives higher recall at a higher cost. |
| Period | One interval of the workload, followed by one maintenance round |
| Maintenance round | One batch modification of the index, between two periods |
| Query log | The queries served in one period |
| Recall@10 | Fraction of the exact 10 nearest neighbors that a search returns |
| QPS | Queries per second |

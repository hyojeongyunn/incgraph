# 4. Workload-adaptive modification

Technique for query changes. Policy `comember`; class `CoMemberPolicy` in `src/include/incgraph/policy.h` and `src/core/policies.cpp`.

## 4.1 Problem

Greedy search finds the top-`K` of a query by expanding the points inside and around the ball that holds those results.

- When these points are well connected to each other, a small search width is enough.
- When they are not, the search needs a large width. This is common for queries from another modality: their result balls cut across regions that the data-only graph never linked.

Which balls matter depends on the current workload, and the workload shifts. The technique uses the query log to connect the points of the balls that currently receive queries, and removes the edges of balls that no longer do. [§5.4](05-evaluation.md#54-workload-adaptive-modification) reports the measured effect.

## 4.2 Method

**Terms.**

| Term | Meaning |
|---|---|
| Query hypersphere | The ball around a query that holds its top-`K` results, as found by an index search |
| Co-members | Points that appear together in the top-`K` of one logged query |
| `c(u, v)` | The number of logged queries of the round whose top-`K` contains both `u` and `v` |
| Rising queries | Queries of hyperspheres that receive queries in the current log |
| Falling queries | Queries of hyperspheres that stopped receiving queries |

Every overlay edge `u → v` added by this technique carries a score `s(u, v)`.

**Maintenance round** (input: the query log `Q` of the last period):

1. **Members.** Search the index for the top-`K` of every logged query, with width `L_member`.
2. **Score update (falling queries).** For every existing overlay edge, set
   `s(u, v) ← decay · s(u, v) + c(u, v)`.
   Remove the edge if `s < min_score`.
3. **Addition (rising queries).** Visit the members in label order. For each member `u`:
   1. Take as candidates its co-members that are not yet neighbors of `u` (base or overlay).
   2. If there are more than `R_extra` candidates, RobustPrune them to `R_extra` with `prune_alpha`, so that the kept edges point in diverse directions.
   3. Add each kept candidate `v` as the overlay edge `u → v`, with score `c(u, v)`.
   4. If the overlay list of `u` is full, evict its lowest-score edge, but only if that score is below `c(u, v)`. Otherwise skip `v`.
4. **Budget.** If `budget > 0` and the total number of overlay edges exceeds it, remove the lowest-score edges until it fits.

**Removal timing.** With the default `decay` and `min_score`, an edge that was confirmed once is removed in the third round in which no logged query confirms it again.

**Purpose of removal.**

- **Memory.** Without removal, the overlay only grows from round to round.
- **Placement.** Under a fixed budget, removal frees room for edges where the current queries are.

**Ablation.** With `removal=none`, steps 2, 3.4 and 4 are turned off. Edges are never removed, and addition stops when a point's list or the budget is full.

## 4.3 Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `K` | 30 | Size of the top-`K` that defines a query hypersphere |
| `L_member` | 200 | Search width used to find the top-`K` of logged queries |
| `prune_alpha` | 1.2 | RobustPrune `α` applied to a member's candidates |
| `removal` | `score` | `score`: steps 2–4. `none`: no removal (ablation). |
| `decay` | 0.5 | Score decay per round |
| `min_score` | 0.25 | Edges below this score are removed |
| `budget` | 0 | Maximum total number of overlay edges (0 = no limit) |

The index must have an overlay capacity `R_extra` of at least 1.

## 4.4 Complexity

- **Memory:** at most `R_extra` overlay edges per point, and at most `budget` in total.
- **Time per round:**
  - one index search per logged query;
  - set operations over the members;
  - one RobustPrune per member with more than `R_extra` candidates.

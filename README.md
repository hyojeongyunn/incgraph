# incgraph

incgraph is a graph-based approximate nearest neighbor (ANN) index that is modified in place, instead of rebuilt, when its data or its queries change. It extends a Vamana graph with two batch modification techniques:

- **Incremental modification** handles inserted vectors that do not follow the distribution of the indexed data, as well as deletes.
- **Workload-adaptive modification** follows shifts in the query distribution, using the log of served queries.

## Build

```bash
docker compose up -d --build
docker compose exec dev bash
scripts/build.sh
bin/ig_test
```

`scripts/build.sh` builds the library, the programs and the tests into `bin/`. `bin/ig_test` prints `ALL OK` when every test passes.

Without Docker, the build needs CMake ≥ 3.18, Ninja and a C++17 compiler with OpenMP. The plotting script also needs Python 3 with NumPy and Matplotlib.

## Documentation

| Page | Contents |
|---|---|
| [1. Overview](docs/01-overview.md) | Problem, operating model, the two techniques, common terms |
| [2. Base index](docs/02-base-index.md) | The Vamana index that both techniques modify |
| [3. Incremental modification](docs/03-incremental-modification.md) | Technique for data changes |
| [4. Workload-adaptive modification](docs/04-workload-adaptive-modification.md) | Technique for query changes |
| [5. Evaluation](docs/05-evaluation.md) | Datasets, protocol, results, reproduction |
| [6. Reference](docs/06-reference.md) | Programs, scripts, file formats, C++ API |

## Repository layout

```
src/include/incgraph/   index.h (index), policy.h (techniques), util.h (I/O, exact k-NN, k-means, RNG)
src/core/               index.cpp, policies.cpp, util.cpp
src/apps/               command-line programs
src/tests/              tests
scripts/                build, experiment and plotting scripts
data/download.sh        download of the public datasets
docs/                   documentation
```

## Acknowledgement

This work was supported by an Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00454666, Developing a Vector DB for Long-Term Memory Storage of Hyperscale AI Models).

# datafusion-vector-search-ext

A DataFusion extension that integrates [USearch](https://github.com/unum-cloud/usearch) HNSW approximate nearest-neighbour (ANN) vector search as a first-class SQL operator.

Queries matching the `ORDER BY distance_fn(col, query) LIMIT k` pattern are **transparently rewritten** by an optimizer rule into a native USearch index call — no query rewrite needed from the caller. `WHERE` clause filters are handled adaptively: high-selectivity filters use USearch's in-graph predicate API; low-selectivity filters bypass HNSW entirely and scan the data directly.

**DataFusion:** 52.2 &nbsp; **USearch:** 2.24

---

## Usage

### Add the dependency

```toml
[dependencies]
datafusion-vector-search-ext = { git = "https://github.com/hotdata-dev/datafusion-vector-search-ext" }
usearch = "2.24.0"
```

### Build or load a USearch index

`USearchIndexConfig` is the single source of truth for index parameters. Persist it alongside your `.index` file — USearch does not embed metadata inside the saved file.

```rust
use datafusion_vector_search_ext::USearchIndexConfig;
use usearch::MetricKind;

let cfg = USearchIndexConfig::new(768, MetricKind::L2sq);

// Build a new index:
let index = cfg.build_index()?;
index.reserve(n_rows)?;
for (key, vec) in rows {
    index.add(key, &vec)?;  // key must be a unique u64
}
index.save("my_table.index")?;

// Reload from disk:
let index = cfg.load_index("my_table.index")?;
```

**Key mapping:** USearch stores `u64` keys only. Each key must map 1:1 to a row in your providers. A simple monotonic row-ID (0, 1, 2, ...) works well.

### Register providers and set up the SessionContext

Registration requires two providers:

- **`scan_provider`** (`Arc<dyn TableProvider>`) — used for WHERE evaluation and the low-selectivity Parquet-native path. Should contain all columns including the vector column.
- **`lookup_provider`** (`Arc<dyn PointLookupProvider>`) — used for O(k) key-based row fetch after HNSW search. Does not need the vector column.

`PointLookupProvider` extends DataFusion's `TableProvider` with a single method:

```rust
#[async_trait]
impl PointLookupProvider for MyTable {
    async fn fetch_by_keys(
        &self,
        keys: &[u64],
        key_col: &str,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>> {
        // Look up each key via your primary-key index.
        // Missing keys are silently omitted.
        todo!()
    }
}
```

For development and tests, use the bundled `HashKeyProvider`:

```rust
use datafusion_vector_search_ext::HashKeyProvider;
let provider = Arc::new(HashKeyProvider::try_new(schema, batches, "id")?);
```

Wire everything together:

```rust
use std::sync::Arc;
use datafusion::execution::context::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_vector_search_ext::{
    USearchRegistry, USearchQueryPlanner, register_all,
};
use usearch::{MetricKind, ScalarKind};

let registry = USearchRegistry::new();
registry.add(
    "my_table::vector",     // "<table>::<vector_col>"
    Arc::new(index),
    scan_provider.clone(),  // TableProvider (e.g. Parquet)
    lookup_provider.clone(),// PointLookupProvider (e.g. SQLite)
    "id",                   // key column name
    MetricKind::L2sq,
    ScalarKind::F32,
)?;
let registry = registry.into_arc();

let state = SessionStateBuilder::new()
    .with_default_features()
    .with_query_planner(Arc::new(USearchQueryPlanner::new(registry.clone())))
    .build();
let ctx = SessionContext::new_with_state(state);
register_all(&ctx, registry)?;

ctx.register_table("my_table", scan_provider)?;
```

### Write queries

No special syntax — the extension transparently rewrites standard SQL:

```sql
SELECT id, title, l2_distance(vector, ARRAY[0.1, 0.2, ...]) AS dist
FROM my_table
ORDER BY dist ASC
LIMIT 10
```

**Recognised patterns:**

| Distance function       | Index metric       | Order |
|-------------------------|--------------------|-------|
| `l2_distance`           | `MetricKind::L2sq` | `ASC` |
| `cosine_distance`       | `MetricKind::Cos`  | `ASC` |
| `negative_dot_product`  | `MetricKind::IP`   | `ASC` |

All three are lower-is-closer. `ORDER BY ... DESC` falls back to exact computation. Metric mismatch (e.g. cosine UDF on an L2 index) also falls back silently.

### WHERE clause filtering

Scalar `WHERE` conditions are absorbed and handled adaptively:

```sql
SELECT id, l2_distance(vector, ARRAY[...]) AS dist
FROM my_table
WHERE category = 'nlp'
ORDER BY dist ASC
LIMIT 10
```

See [Adaptive filtering](#adaptive-filtering) for details on how the execution path is chosen.

### UDTF path

For runtime query vectors, complex joins, or explicit over-fetch control:

```sql
SELECT vs.key, vs._distance, d.title
FROM vector_usearch('my_table', ARRAY[0.1, 0.2, ...], 20) vs
JOIN my_table d ON d.id = vs.key
ORDER BY vs._distance ASC
LIMIT 10
```

The UDTF always calls `index.search()` directly — no filter absorption. Apply `WHERE` on the outer query to post-filter.

### Tuning

Pass `USearchTableConfig` to `add_with_config` for per-table tuning:

```rust
use datafusion_vector_search_ext::USearchTableConfig;

registry.add_with_config(
    "my_table::vector",
    Arc::new(index),
    scan_provider.clone(),
    lookup_provider.clone(),
    "id",
    MetricKind::L2sq,
    ScalarKind::F32,
    USearchTableConfig {
        expansion_search: 128,                    // ef_search (default: 64)
        brute_force_selectivity_threshold: 0.03,  // low-sel cutoff (default: 0.05)
    },
)?;
```

| Parameter | Default | Notes |
|---|---|---|
| `expansion_search` | 64 | HNSW beam width at query time. Higher = better recall, slower. Set once at registration. |
| `brute_force_selectivity_threshold` | 0.05 | Below this fraction, bypass HNSW and scan directly. |

**`USearchIndexConfig` build parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `connectivity` (M) | 16 | Graph degree. Higher = better recall, more memory. |
| `expansion_add` (ef_construction) | 128 | Must be >= 2xM. Higher = better graph, slower build. |
| `quantization` | `ScalarKind::F32` | `F16` halves memory with ~1% recall loss at high dims. |

---

## Architecture

### Module structure

```
src/
  lib.rs       — public API, register_all()
  registry.rs  — USearchRegistry, RegisteredTable, USearchIndexConfig, USearchTableConfig
  node.rs      — USearchNode: custom logical plan leaf
  rule.rs      — USearchRule: optimizer rewrite rule
  planner.rs   — USearchExecPlanner, USearchExec: physical execution
  udf.rs       — l2_distance, cosine_distance, negative_dot_product scalar UDFs
  udtf.rs      — vector_usearch table function
  lookup.rs    — PointLookupProvider trait + HashKeyProvider
  keys.rs      — DatasetLayout, pack_key/unpack_key key encoding

tests/
  optimizer_rule.rs  — rewrite rule matching/rejection tests
  execution.rs       — end-to-end execution tests (HNSW + Parquet-native paths)
```

### Optimizer rewrite

The rule (`rule.rs`) matches two logical plan shapes:

```
Sort(fetch=k, ORDER BY dist ASC)
  Projection([..., distance_fn(col, lit) AS dist, ...])
    TableScan(name)

Sort(fetch=k, ORDER BY dist ASC)
  Projection([..., distance_fn(col, lit) AS dist, ...])
    Filter(predicate)
      TableScan(name)
```

Preconditions: sort is `ASC`, distance UDF matches index metric, table is registered, query vector is a literal. When the rule fires, it replaces the inner nodes with a `USearchNode` leaf carrying: table name, vector column, query vector, k, distance type, and absorbed filter predicates. The `Sort` node is preserved above for final ordering.

Physical planning (`planner.rs`) translates `USearchNode` into `USearchExec`, a physical plan node that executes the actual search.

### Adaptive filtering

When `WHERE` filters are present, the execution follows three possible paths:

```
Query arrives
  |
  +-- No WHERE clause
  |     -> USearch HNSW search -> lookup_provider fetch(k) -> result
  |
  +-- Has WHERE clause
        |
        +-- Pre-scan: scan_provider (scalar + _key cols only, filter pushdown)
        |     -> collect valid_keys, compute selectivity
        |
        +-- Low selectivity (<= threshold, default 5%)
        |     -> Full scan from scan_provider (all cols including vector)
        |     -> evaluate WHERE, compute distances, top-k heap
        |     -> return directly -- NO USearch, NO lookup_provider
        |
        +-- High selectivity (> threshold)
              -> HNSW filtered_search(valid_keys predicate)
              -> lookup_provider fetch(k) -> result
```

**Pre-scan phase:** Projects only scalar columns and the key column (excludes the vector column for efficiency). Filter expressions are pushed down to the scan provider. Collects the set of valid keys and computes `selectivity = valid_keys.len() / index.size()`.

**Low-selectivity path (Parquet-native):** When few rows pass the filter, HNSW graph traversal becomes expensive (it must explore ~`k/selectivity` nodes to find k passing candidates). Instead, the full scan streams all columns including the vector, evaluates filters per batch, computes exact distances for passing rows, and maintains a top-k heap (`ScoredRow`). Returns results directly without touching USearch or the lookup provider.

**High-selectivity path (HNSW filtered):** Passes valid keys as a predicate to `index.filtered_search()` — HNSW skips non-passing nodes during traversal. Result keys are fetched from the lookup provider.

**Why 5%:** The crossover is approximately `sqrt(k * M / n)`. For k=10, M=16, n=240k this is ~2.6%. The 5% default provides a safety margin.

### Distance metrics

All three distance functions are **lower-is-closer**:

| SQL function | Index metric | Kernel |
|---|---|---|
| `l2_distance(a, b)` | `L2sq` | `sqrt(sum((a_i - b_i)^2))` (UDF) / `sum((a_i - b_i)^2)` (index) |
| `cosine_distance(a, b)` | `Cos` | `1 - dot(a,b) / (norm(a) * norm(b))` |
| `negative_dot_product(a, b)` | `IP` | `-(a . b)` |

Note: `l2_distance` UDF returns actual L2 (with sqrt) for human-readable distances; USearch uses L2sq internally (no sqrt). The sort order is identical.

### Running tests

```bash
cargo test
```

Tests cover optimizer rule matching/rejection, end-to-end execution through both HNSW and Parquet-native paths, registration validation, and provider error handling.

---

## Limitations

| Limitation | Notes |
|---|---|
| Stacked `Filter` nodes | Only one `Filter -> TableScan` layer is absorbed. `Filter -> Filter -> TableScan` falls back to exact execution. DataFusion typically combines multiple WHERE conditions into a single Filter, so this rarely occurs. |
| Runtime query vectors | The query vector must be a compile-time literal (`ARRAY[0.1, ...]`). Column references or subquery results are not rewritten. Use the UDTF path for runtime vectors. |
| `ef_search` per-query | `expansion_search` is global to the index instance. Per-query adjustment is not supported. |
| No DELETE / compaction | USearch soft-deletes entries but requires a full rebuild to reclaim space. |

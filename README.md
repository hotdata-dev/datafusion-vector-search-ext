# datafusion-vector-search-ext

A DataFusion extension that integrates [USearch](https://github.com/unum-cloud/usearch) HNSW approximate nearest-neighbour (ANN) vector search as a first-class SQL operator.

Queries matching the `ORDER BY distance_fn(col, query) LIMIT k` pattern are **transparently rewritten** by an optimizer rule into a native USearch index call — no query rewrite needed from the caller. `WHERE` clause filters are handled adaptively: high-selectivity filters use USearch's in-graph predicate API; low-selectivity filters bypass HNSW and brute-force the small valid subset.

**DataFusion version:** 51.0.0
**USearch version:** 2.24.0

---

## Table of contents

1. [Usage guide](#usage-guide)
   - [Add the dependency](#1-add-the-dependency)
   - [Implement PointLookupProvider](#2-implement-pointlookupprovider)
   - [Build or load a USearch index](#3-build-or-load-a-usearch-index)
   - [Register and set up the SessionContext](#4-register-and-set-up-the-sessioncontext)
   - [Write queries](#5-write-queries)
   - [WHERE clause filtering](#where-clause-filtering)
   - [UDTF path](#udtf-path)
   - [Tuning](#tuning)
2. [Developer guide](#developer-guide)
   - [Module structure](#module-structure)
   - [Optimizer path](#optimizer-path)
   - [Adaptive filtering](#adaptive-filtering)
   - [Distance metrics](#distance-metrics)
   - [Running tests](#running-tests)
   - [Key DataFusion 51 API notes](#key-datafusion-51-api-notes)
3. [Known limitations](#known-limitations)
4. [Performance reference](#performance-reference)

---

## Usage guide

### 1. Add the dependency

```toml
[dependencies]
datafusion-vector-search-ext = { path = "../datafusion-vector-search-ext" }
usearch = "2.24.0"
```

`datafusion` **must** be `51.0.0` in your workspace — the extension embeds deeply into DataFusion's optimizer and physical planner internals and is not version-agnostic.

---

### 2. Implement PointLookupProvider

`PointLookupProvider` is the only integration work required. It is a thin extension of DataFusion's `TableProvider` that adds an O(k) row-lookup method — the extension calls this after a USearch ANN search to retrieve full rows for the k result keys, without scanning the whole table.

```rust
use std::sync::Arc;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::common::Result;
use datafusion_vector_search_ext::PointLookupProvider;

#[async_trait]
impl PointLookupProvider for MyEngineTable {
    async fn fetch_by_keys(
        &self,
        keys: &[u64],                  // USearch u64 keys to retrieve
        key_col: &str,                 // name of the key column in the schema
        projection: Option<&[usize]>,  // column indices to return (None = all)
    ) -> Result<Vec<RecordBatch>> {
        // Look up each key via your primary-key index (B-tree, LSM, etc.).
        // Return only the requested rows.  Missing keys are silently omitted.
        // Apply `projection` if Some — only return those column indices.
        todo!()
    }
}
```

`MyEngineTable` must also implement DataFusion's `TableProvider` (so it can be registered with `ctx.register_table`) and `Send + Sync`.

**For development and tests**, use the bundled `HashKeyProvider`:

```rust
use datafusion_vector_search_ext::HashKeyProvider;

let provider = Arc::new(
    HashKeyProvider::try_new(schema, batches, "id")?
);
```

`HashKeyProvider` builds a `HashMap<u64 → (batch, row)>` at construction time and answers `fetch_by_keys` in O(k). It is not intended for large production tables.

---

### 3. Build or load a USearch index

`USearchIndexConfig` is the single source of truth for index parameters. Persist it alongside your `.index` file so load always uses the same options — USearch does not embed metadata inside the saved file.

```rust
use datafusion_vector_search_ext::USearchIndexConfig;
use usearch::MetricKind;

let cfg = USearchIndexConfig::new(768, MetricKind::L2sq);

// Build a new index:
let index = cfg.build_index()?;
index.reserve(n_rows)?;                    // pre-allocate before adding
for (key, vec) in rows {
    index.add(key, &vec)?;                 // key must be a unique u64
}
index.save("my_table.index")?;

// Reload from disk (e.g. on server restart):
let index = cfg.load_index("my_table.index")?;
```

**Key mapping:** USearch stores only `u64` keys. The key must map 1:1 to a row in your `PointLookupProvider`. A simple monotonic row-ID (0, 1, 2, …) works well. The key column in the provider schema is identified by name at registration time.

---

### 4. Register and set up the SessionContext

```rust
use std::sync::Arc;
use datafusion::execution::context::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_vector_search_ext::{
    USearchRegistry, USearchQueryPlanner, register_all,
};
use usearch::MetricKind;

// Build registry.
let mut registry = USearchRegistry::new();
registry.add(
    "my_table",          // table name — must match ctx.register_table name
    Arc::new(index),     // loaded USearch index
    provider.clone(),    // your PointLookupProvider
    "id",                // key column name
    MetricKind::L2sq,    // must match the index's build metric
)?;
let registry = registry.into_arc();

// Install the custom query planner at SessionState build time.
// This MUST happen before SessionContext is created.
let state = SessionStateBuilder::new()
    .with_query_planner(Arc::new(USearchQueryPlanner::new(registry.clone())))
    .build();
let ctx = SessionContext::new_with_state(state);

// Register UDFs, UDTF, and optimizer rule.
register_all(&ctx, registry)?;

// Register the table for SQL column-name resolution.
ctx.register_table("my_table", provider)?;
```

---

### 5. Write queries

No special syntax. Write standard SQL — the extension transparently rewrites it:

```sql
-- Optimizer detects ORDER BY l2_distance ASC LIMIT k → rewrites to USearch call.
SELECT id, title, l2_distance(vector, ARRAY[0.1, 0.2, ...]) AS dist
FROM my_table
ORDER BY dist ASC
LIMIT 10
```

**Recognised patterns:**

| Distance function       | Index metric     | Order   |
|-------------------------|-----------------|---------|
| `l2_distance`           | `MetricKind::L2sq` | `ASC` |
| `cosine_distance`       | `MetricKind::Cos`  | `ASC` |
| `negative_dot_product`  | `MetricKind::IP`   | `ASC` |

All three distance functions are lower-is-closer. `ORDER BY ... DESC` is never rewritten — it falls back to exact row-by-row computation (correct, just slow). Metric mismatch (e.g. cosine UDF on an L2 index) also falls back silently.

---

### WHERE clause filtering

Scalar `WHERE` conditions are absorbed by the optimizer rule and handled adaptively at execution time:

```sql
SELECT id, l2_distance(vector, ARRAY[...]) AS dist
FROM my_table
WHERE category = 'nlp'        -- absorbed into the USearch execution
ORDER BY dist ASC
LIMIT 10
```

Two strategies are chosen automatically based on filter selectivity:

| Selectivity | Strategy | How |
|---|---|---|
| > 5% | **In-graph filtering** | Pre-scan builds `valid_keys`. `index.filtered_search(query, k, \|key\| valid_keys.contains(key))` — HNSW skips non-passing nodes during traversal, always returns exactly k results. |
| ≤ 5% | **Brute-force subset** | HNSW bypassed. Exact distances computed over the valid subset only; heap-select top-k. Cheaper when the valid set is small. |

The 5% threshold is configurable per table. See [Tuning](#tuning).

---

### UDTF path

For cases where the optimizer path does not apply (runtime query vectors, complex joins, explicit control over over-fetch), use the `vector_usearch` table function directly:

```sql
-- Returns (key: UInt64, _distance: Float32)
SELECT vs.key, vs._distance, d.title
FROM vector_usearch('my_table', ARRAY[0.1, 0.2, ...], 20) vs
JOIN my_table d ON d.id = vs.key
ORDER BY vs._distance ASC
LIMIT 10
```

The UDTF path is always a direct `index.search()` call — no filter absorption. Apply `WHERE` on the outer query to post-filter:

```sql
-- Over-fetch 50, post-filter to 'nlp', return top 10
SELECT vs.key, vs._distance, d.title
FROM vector_usearch('my_table', ARRAY[...], 50) vs
JOIN my_table d ON d.id = vs.key
WHERE d.category = 'nlp'
ORDER BY vs._distance ASC
LIMIT 10
```

**Two-phase ANN + exact re-rank** (fetch approximate candidates, rerank with exact distances):

```sql
SELECT d.id, cosine_distance(d.vector, ARRAY[...]) AS exact_dist
FROM vector_usearch('my_table', ARRAY[...], 50) vs
JOIN my_table d ON d.id = vs.key
ORDER BY exact_dist ASC
LIMIT 10
```

---

### Tuning

Pass `USearchTableConfig` to `add_with_config` for per-table tuning:

```rust
use datafusion_vector_search_ext::USearchTableConfig;

registry.add_with_config(
    "my_table",
    Arc::new(index),
    provider.clone(),
    "id",
    MetricKind::L2sq,
    USearchTableConfig {
        expansion_search: 128,              // ef_search — higher = better recall, slower
        brute_force_selectivity_threshold: 0.03,  // use brute-force below 3% selectivity
    },
)?;
```

**`expansion_search` (ef_search):**
- Controls beam width during HNSW graph traversal at query time.
- Higher → more nodes explored → higher recall, slower queries.
- Default: 64. Typical range: 32–200.
- `LIMIT k > expansion_search` is handled automatically: USearch uses `max(expansion_search, k)` per call.
- Set once at registration — never call `index.change_expansion_search()` externally (data race on `Arc<Index>`).

**`brute_force_selectivity_threshold`:**
- Fraction of table rows that must pass the `WHERE` filter before the in-graph path is preferred.
- Default: 0.05 (5%). The theoretical crossover is `sqrt(k × M / n)`.
- Increase for tables where typical filters are highly selective; decrease for tables with mostly loose filters.

**`USearchIndexConfig` build parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `connectivity` (M) | 16 | Graph degree. Higher → better recall, more memory. |
| `expansion_add` (ef_construction) | 128 | Must be ≥ 2×M. Higher → better graph, slower build. |
| `quantization` | `ScalarKind::F32` | `F16` halves memory with ~1% recall loss at high dims. |

---

## Developer guide

### Module structure

```
src/
  lib.rs       — public API, register_all()
  registry.rs  — USearchRegistry, USearchTableConfig, USearchIndexConfig
  node.rs      — USearchNode: custom logical plan leaf
  rule.rs      — USearchRule: optimizer rewrite rule
  planner.rs   — USearchExecPlanner, USearchExec: physical execution
  udf.rs       — l2_distance, cosine_distance, negative_dot_product scalar UDFs
  udtf.rs      — vector_usearch table function
  lookup.rs    — PointLookupProvider trait + HashKeyProvider

tests/
  optimizer_rule.rs — 10 integration tests for USearchRule
```

---

### Optimizer path

The rule (`rule.rs`) runs top-down over the logical plan and matches two shapes:

```
Sort(fetch=k, ORDER BY dist ASC)
  Projection([..., l2_distance(col, lit) AS dist, ...])
    TableScan(name)

Sort(fetch=k, ORDER BY dist ASC)
  Projection([..., l2_distance(col, lit) AS dist, ...])
    Filter(predicate)          ← WHERE clause, optional
      TableScan(name)
```

Preconditions checked before rewriting:
1. Sort direction is `ASC` — `DESC` is never rewritten.
2. The distance UDF name matches the metric the index was built with (`dist_type_matches_metric`). Mismatch → fallback to exact execution, no wrong results.
3. The table name is registered in `USearchRegistry`.
4. The query vector is a compile-time literal (not a runtime column reference).

When the rule fires, it replaces the inner `Filter → TableScan` (or just `TableScan`) with a `USearchNode` leaf that carries: table name, vector column, query vector, k, distance type, and absorbed filter predicates.

The `Sort` node is preserved above the `USearchNode` — DataFusion handles the final ordering of the k returned rows.

Physical planning (`planner.rs`) translates `USearchNode → USearchExec`, a leaf `ExecutionPlan` that returns pre-computed `RecordBatch`es from a `MemoryStream`.

---

### Adaptive filtering

When `USearchNode.filters` is non-empty, `USearchExecPlanner` runs:

```
1. Compile each filter Expr → PhysicalExpr  (via create_physical_expr)
2. Full scan of PointLookupProvider → evaluate PhysicalExprs per batch
3. Collect valid_keys: HashSet<u64>  and  key_distances: Vec<(u64, f32)>
4. selectivity = valid_keys.len() / index.size()

if selectivity > threshold:
    index.filtered_search(query, k, |key| valid_keys.contains(&key))
    → fetch_by_keys on result keys → attach _distance

else:
    key_distances already has exact distances for all valid rows
    → heap-select top-k
    → fetch_by_keys on top-k keys → attach _distance
```

**Why the crossover is ~3–5%:**
In-graph filtering explores approximately `k / selectivity` nodes. Each node costs `O(M × d)` work. Brute-force over the valid subset costs `O(n × selectivity × d)`. They are equal when `selectivity ≈ sqrt(k × M / n)`. For k=10, M=16, n=240k this is ~2.6%; the default threshold of 5% provides a safety margin.

**Distance kernels in the brute-force path** match USearch's internal metrics:
- L2: `sum((a[i] - b[i])²)` — squared, no sqrt, matching `MetricKind::L2sq`
- Cosine: `1 - dot(a,b) / (|a| × |b|)`
- Negative dot: `-(a · b)`, matching `MetricKind::IP`

---

### Distance metrics

All three distance functions are **lower-is-closer** and use `ORDER BY ASC`:

| SQL function | Index metric | Kernel |
|---|---|---|
| `l2_distance(a, b)` | `L2sq` | `√Σ(aᵢ-bᵢ)²` (UDF) / `Σ(aᵢ-bᵢ)²` (index) |
| `cosine_distance(a, b)` | `Cos` | `1 - dot(a,b)/(‖a‖‖b‖)` |
| `negative_dot_product(a, b)` | `IP` | `-(a·b)` |

Note: `l2_distance` UDF returns actual L2 (with sqrt) for human-readable distances; the USearch index uses L2sq internally (no sqrt). The sort order is identical — the rewrite is correct.

---

### Running tests

```bash
cargo test
```

The 10 tests in `tests/optimizer_rule.rs` cover:
- All 3 matching metric combinations (rule fires)
- All 6 mismatching metric combinations (rule does not fire)
- `ORDER BY DESC` rejection (rule does not fire)

Tests use an empty `HashKeyProvider` — no real data or physical plan execution. They only inspect the optimized logical plan via `DataFrame::into_optimized_plan()`.

---

### Key DataFusion 51 API notes

These are non-obvious behaviours discovered during development:

- `df.logical_plan()` returns the **unoptimized** plan. Use `df.into_optimized_plan()` to trigger the optimizer and see the rewritten plan.
- `Expr::Literal` is a 2-tuple `(ScalarValue, Option<FieldMetadata>)`. Match the second field with `_`.
- `ScalarValue::FixedSizeList(arr)` — `arr` is `Arc<FixedSizeListArray>` directly, no downcast needed.
- `PlanProperties::new` takes 4 arguments: `(EquivalenceProperties, Partitioning, EmissionType, Boundedness)`.
- `EmissionType` and `Boundedness` are in `datafusion::physical_plan::execution_plan`.
- No `SessionStateBuilder::with_physical_planner` — use `with_query_planner(Arc<dyn QueryPlanner>)` instead.
- `TableFunctionImpl` is in `datafusion::catalog`, not `datafusion::logical_expr`.
- `UserDefinedLogicalNodeCore` requires `PartialEq + Eq + PartialOrd + Hash`. `Expr` does not implement `Hash` — use `format!("{:?}", exprs).hash(state)`.
- `DataFusionError::ArrowError` variant is `(Box<ArrowError>, Option<String>)`.
- `MemTable` scan does not exist at `datafusion::physical_plan::memory` — use `MemTable::try_new().scan()`.
- UDTF `TableProvider::scan()` **must** honour `projection: Option<&Vec<usize>>` — ignoring it causes JOIN column index mismatches.
- `DFSchema` field qualifiers must match exactly when building schemas for `UserDefinedLogicalNode`. Use `DFSchema::new_with_metadata` with explicit `(Option<TableReference>, Arc<Field>)` pairs.

---

## Known limitations

| Limitation | Notes |
|---|---|
| Stacked `Filter` nodes | Only one `Filter → TableScan` layer is absorbed. `Filter → Filter → TableScan` falls back to exact execution. DataFusion typically combines multiple WHERE conditions into a single Filter node, so this rarely occurs in practice. |
| Runtime query vectors | The query vector must be a compile-time literal (`ARRAY[0.1, ...]`). Column references or subquery results as the query vector are not rewritten. Use the UDTF path for runtime vectors. |
| `ef_search` per-query | `expansion_search` is global to the index instance. Per-query ef_search adjustment is not supported (would require per-query index clones or the Rust USearch bindings to expose a per-call API). |
| No DELETE / compaction | USearch soft-deletes entries but requires a full rebuild to reclaim space. Not relevant for read-heavy workloads. |
| No recall measurement | The extension does not measure ANN recall. For recall validation, compare `index.search()` results against `index.exact_search()` on a representative query sample. |

---

## Performance reference

Measured on `ftopal/huggingface-models-embeddings` (240,530 vectors × 768 dims, 6 Parquet shards, M=16, ef_construction=128):

| Configuration | Median query latency | QPS |
|---|---|---|
| USearch HNSW, ef_search=64, k=10, no filter | 0.16 ms | 4,064 |
| LanceDB IvfFlat (512 partitions, nprobes=20), k=10 | 0.65 ms | 1,418 |

USearch: **4× lower median latency**, **2.9× higher QPS** vs LanceDB IvfFlat on the same dataset (sequential queries; parallel querying would improve both).

Index build: 19.8 s total (0.6 s Parquet read + 19.2 s HNSW build, 8-thread Rayon). Memory footprint: 1,080 MB. Saved index: 739 MB on disk, 223 ms cold load.

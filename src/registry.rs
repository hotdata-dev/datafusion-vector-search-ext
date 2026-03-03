// registry.rs — USearchRegistry, USearchTableConfig, USearchIndexConfig.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::lookup::PointLookupProvider;

// ── USearchIndexConfig ────────────────────────────────────────────────────────

/// Parameters used to build and reload a USearch HNSW index.
///
/// This struct is the single source of truth for index construction options.
/// Persist it alongside your index file so that `load_index` can reconstruct
/// the index with exactly the same parameters — USearch does not embed metadata
/// inside the saved file.
///
/// # Example
///
/// ```rust,ignore
/// use datafusion_vector_search_ext::USearchIndexConfig;
/// use usearch::MetricKind;
///
/// let cfg = USearchIndexConfig::new(768, MetricKind::L2sq);
///
/// // Build and populate the index.
/// let index = cfg.build_index()?;
/// index.reserve(240_000)?;
/// for (key, vec) in rows { index.add(key, vec)?; }
/// index.save("my_table.index")?;
///
/// // Reload from disk — same config guarantees consistent options.
/// let index = cfg.load_index("my_table.index")?;
/// ```
#[derive(Debug, Clone)]
pub struct USearchIndexConfig {
    /// Number of dimensions in each vector.  Must match the data exactly.
    pub dimensions: usize,

    /// Distance metric.  Must match the SQL UDF used at query time:
    /// - `MetricKind::L2sq`  → `l2_distance(col, ARRAY[...])`
    /// - `MetricKind::Cos`   → `cosine_distance(col, ARRAY[...])`
    /// - `MetricKind::IP`    → `negative_dot_product(col, ARRAY[...])`
    pub metric: MetricKind,

    /// Graph degree M: number of bidirectional edges per node per layer.
    /// Higher M → better recall and faster search, but more memory and slower
    /// index build.  Default: 16 (canonical HNSW default).
    pub connectivity: usize,

    /// ef_construction: beam width used when inserting nodes into the graph.
    /// Higher values → better graph quality → higher recall at query time,
    /// at the cost of slower index build.  Must be ≥ 2 × connectivity.
    /// Default: 128.
    pub expansion_add: usize,

    /// Scalar quantization format.  `F32` stores full-precision vectors
    /// (safest, most accurate).  `F16` halves memory with typically < 1%
    /// recall loss at high dimensions.  Default: `ScalarKind::F32`.
    pub quantization: ScalarKind,
}

impl USearchIndexConfig {
    /// Create a config for the given dimensions and metric, with all other
    /// parameters set to their defaults (M=16, ef_construction=128, F32).
    pub fn new(dimensions: usize, metric: MetricKind) -> Self {
        Self { dimensions, metric, ..Self::default() }
    }

    /// Build a new, empty USearch index with these parameters.
    ///
    /// Call `index.reserve(n)` before adding vectors to pre-allocate graph
    /// nodes.  Adding vectors without reserving causes incremental
    /// reallocation and is significantly slower for large datasets.
    pub fn build_index(&self) -> Result<Index> {
        Index::new(&self.to_index_options())
            .map_err(|e| DataFusionError::Execution(format!("USearch Index::new failed: {e}")))
    }

    /// Load a previously saved index from `path`.
    ///
    /// Uses the same `IndexOptions` as `build_index()`.  The options must
    /// match those used when the index was originally built — passing wrong
    /// dimensions or metric produces silently incorrect results.
    pub fn load_index(&self, path: &str) -> Result<Index> {
        let index = Index::new(&self.to_index_options())
            .map_err(|e| DataFusionError::Execution(format!("USearch Index::new failed: {e}")))?;
        index.load(path)
            .map_err(|e| DataFusionError::Execution(format!("USearch index load failed: {e}")))?;
        Ok(index)
    }

    fn to_index_options(&self) -> IndexOptions {
        IndexOptions {
            dimensions: self.dimensions,
            metric: self.metric,
            quantization: self.quantization,
            connectivity: self.connectivity,
            expansion_add: self.expansion_add,
            // expansion_search (ef_search) is set at registration time via
            // USearchTableConfig, not at index build time.
            expansion_search: 0,
            ..Default::default()
        }
    }
}

impl Default for USearchIndexConfig {
    fn default() -> Self {
        Self {
            dimensions: 0,
            metric: MetricKind::L2sq,
            connectivity: 16,
            expansion_add: 128,
            quantization: ScalarKind::F32,
        }
    }
}

// ── USearchTableConfig ────────────────────────────────────────────────────────

/// Per-table configuration for query execution behaviour.
///
/// Pass to [`USearchRegistry::add_with_config`] to override the defaults.
/// Use [`USearchRegistry::add`] for the common case — it applies
/// [`USearchTableConfig::default()`] automatically.
///
/// `expansion_search` (ef_search) is applied to the index exactly once,
/// inside [`USearchRegistry::add_with_config`], before any query touches it.
/// **Never call `index.change_expansion_search()` after registration** —
/// the index is shared via `Arc` and mutation after that point is a data race.
#[derive(Debug, Clone)]
pub struct USearchTableConfig {
    /// HNSW ef_search: beam width during graph traversal at query time.
    /// Higher → better recall, slower queries.
    ///
    /// Queries with `LIMIT k > expansion_search` are handled automatically:
    /// USearch uses `max(expansion_search, k)` per call, so they are correct
    /// but cost proportionally more.
    ///
    /// Default: 64.  Recommended range: 32–200.
    pub expansion_search: usize,

    /// Selectivity fraction below which the planner bypasses the HNSW index
    /// and runs an exact brute-force search over only the rows that pass the
    /// WHERE filter.
    ///
    /// At low selectivity, `filtered_search` must explore ~`k/selectivity`
    /// graph nodes before finding k passing candidates — eventually slower
    /// than a linear scan over the small valid subset.
    ///
    /// The theoretical crossover is `sqrt(k × M / n)` ≈ 2.6% for a 240k-row
    /// dataset with k=10 and M=16.  The default 0.05 (5%) gives a safety
    /// margin that works across a wide range of n, k, and M.
    pub brute_force_selectivity_threshold: f64,
}

impl Default for USearchTableConfig {
    fn default() -> Self {
        Self {
            expansion_search: 64,
            brute_force_selectivity_threshold: 0.05,
        }
    }
}

// ── RegisteredTable ───────────────────────────────────────────────────────────

pub struct RegisteredTable {
    pub index: Arc<Index>,
    pub provider: Arc<dyn PointLookupProvider>,
    pub key_col: String,
    pub metric: MetricKind,
    pub schema: SchemaRef,
    pub config: USearchTableConfig,
}

// ── USearchRegistry ───────────────────────────────────────────────────────────

pub struct USearchRegistry {
    tables: HashMap<String, RegisteredTable>,
}

impl USearchRegistry {
    pub fn new() -> Self {
        Self { tables: HashMap::new() }
    }

    /// Register a USearch index with default query configuration.
    ///
    /// Convenience wrapper around [`add_with_config`] that uses
    /// [`USearchTableConfig::default()`] (ef_search=64, threshold=5%).
    ///
    /// - `index` — must already be loaded / populated.
    /// - `provider` — must implement [`PointLookupProvider`].
    ///   [`HashKeyProvider`] is the bundled in-memory implementation.
    ///   For production, implement the trait on your storage engine's table type.
    /// - `key_col` — column in `provider.schema()` that stores the USearch key
    ///   (`u64`).  Supported Arrow types: `UInt64`, `Int64`, `UInt32`, `Int32`.
    /// - `metric` — must match how the index was built.  The optimizer rule
    ///   validates this and refuses to rewrite on mismatch.
    ///
    /// [`add_with_config`]: USearchRegistry::add_with_config
    /// [`HashKeyProvider`]: crate::lookup::HashKeyProvider
    pub fn add(
        &mut self,
        name: &str,
        index: Arc<Index>,
        provider: Arc<dyn PointLookupProvider>,
        key_col: &str,
        metric: MetricKind,
    ) -> Result<()> {
        self.add_with_config(name, index, provider, key_col, metric, USearchTableConfig::default())
    }

    /// Register a USearch index with explicit query configuration.
    ///
    /// Sets `ef_search` on the index exactly once before storing it.
    /// Do not call `index.change_expansion_search()` after this point.
    pub fn add_with_config(
        &mut self,
        name: &str,
        index: Arc<Index>,
        provider: Arc<dyn PointLookupProvider>,
        key_col: &str,
        metric: MetricKind,
        config: USearchTableConfig,
    ) -> Result<()> {
        // Set ef_search once, here, before any query touches the index.
        index.change_expansion_search(config.expansion_search);

        let data_schema = provider.schema();

        let _ = data_schema.index_of(key_col).map_err(|_| {
            DataFusionError::Execution(format!(
                "USearchRegistry: key column '{key_col}' not found in table '{name}' schema"
            ))
        })?;

        let mut fields: Vec<Field> =
            data_schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new("_distance", DataType::Float32, true));
        let schema = Arc::new(Schema::new(fields));

        self.tables.insert(
            name.to_string(),
            RegisteredTable { index, provider, key_col: key_col.to_string(), metric, schema, config },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&RegisteredTable> {
        self.tables.get(name)
    }

    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl Default for USearchRegistry {
    fn default() -> Self { Self::new() }
}

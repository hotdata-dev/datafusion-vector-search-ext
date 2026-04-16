//! # datafusion-vector-search-ext
//!
//! A DataFusion extension that integrates USearch HNSW approximate nearest
//! neighbour (ANN) search as a first-class SQL operator.
//!
//! Queries that match the `ORDER BY distance_fn(...) LIMIT k` pattern are
//! transparently rewritten by an optimizer rule into a native USearch index
//! call — no query rewrite required from the caller.  WHERE clause filters
//! are handled adaptively: high-selectivity filters use in-graph predicate
//! filtering; low-selectivity filters bypass HNSW entirely and run an exact
//! brute-force search over the valid subset.
//!
//! # Quick setup
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use datafusion::execution::context::SessionStateBuilder;
//! use datafusion::prelude::SessionContext;
//! use datafusion_vector_search_ext::{
//!     USearchIndexConfig, USearchRegistry, USearchQueryPlanner, register_all,
//! };
//! use usearch::MetricKind;
//!
//! // 1. Build or load the index.
//! let cfg = USearchIndexConfig::new(768, MetricKind::L2sq);
//! let index = cfg.load_index("my_table.index")?;
//!
//! // 2. Wrap your table in a PointLookupProvider.
//! let provider = Arc::new(MyTableProvider::new(...));
//!
//! // 3. Register with USearchRegistry.
//! let mut registry = USearchRegistry::new();
//! registry.add("my_table", Arc::new(index), provider.clone(), "id", MetricKind::L2sq)?;
//! let registry = registry.into_arc();
//!
//! // 4. Build SessionContext with the custom query planner.
//! let state = SessionStateBuilder::new()
//!     .with_query_planner(Arc::new(USearchQueryPlanner::new(registry.clone())))
//!     .build();
//! let ctx = SessionContext::new_with_state(state);
//!
//! // 5. Register UDFs, UDTF, and optimizer rule.
//! register_all(&ctx, registry)?;
//!
//! // Also register your table so DataFusion can resolve column names.
//! ctx.register_table("my_table", provider)?;
//! ```
//!
//! Queries now use the HNSW index automatically:
//!
//! ```sql
//! SELECT id, l2_distance(vector, ARRAY[...]) AS dist
//! FROM my_table
//! WHERE category = 'nlp'
//! ORDER BY dist ASC
//! LIMIT 10
//! ```

pub mod keys;
pub mod lookup;
pub mod node;
pub mod planner;
pub mod registry;
pub mod rule;
pub mod udf;
pub mod udtf;

#[cfg(feature = "parquet-provider")]
pub mod parquet_provider;
#[cfg(feature = "sqlite-provider")]
pub mod sqlite_provider;

pub use keys::{DatasetLayout, pack_key, unpack_key};
pub use lookup::{HashKeyProvider, PointLookupProvider};
pub use node::{DistanceType, USearchNode};
pub use planner::{USearchExec, USearchExecPlanner, USearchQueryPlanner};
pub use registry::{
    RegisteredTable, USearchIndexConfig, USearchRegistry, USearchTableConfig, VectorIndexMeta,
    VectorIndexResolver,
};
pub use rule::USearchRule;
pub use udf::{cosine_distance_udf, l2_distance_udf, negative_dot_product_udf};
pub use udtf::VectorSearchVectorUDTF;

#[cfg(feature = "parquet-provider")]
pub use parquet_provider::ParquetLookupProvider;
#[cfg(feature = "sqlite-provider")]
pub use sqlite_provider::SqliteLookupProvider;

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;

/// Register all extension components with a DataFusion [`SessionContext`].
///
/// Registers:
/// - `l2_distance(col, query)`          — squared Euclidean distance (L2sq)
/// - `cosine_distance(col, query)`      — cosine distance
/// - `negative_dot_product(col, query)` — negated inner product
/// - `vector_search_vector('conn.schema.table', 'column', ARRAY[...], k)`
///   — explicit ANN table function returning full rows + `_distance`
///   (cache-only for async-backed resolvers; does not trigger async loads)
/// - [`USearchRule`]                    — optimizer rewrite rule
///
/// The [`USearchQueryPlanner`] must be installed at `SessionState` build time
/// (before this call) via `SessionStateBuilder::with_query_planner`.
pub fn register_all(ctx: &SessionContext, registry: Arc<dyn VectorIndexResolver>) -> Result<()> {
    ctx.register_udf(ScalarUDF::new_from_impl(l2_distance_udf()));
    ctx.register_udf(ScalarUDF::new_from_impl(cosine_distance_udf()));
    ctx.register_udf(ScalarUDF::new_from_impl(negative_dot_product_udf()));
    ctx.register_udtf(
        "vector_search_vector",
        Arc::new(VectorSearchVectorUDTF::new(registry.clone())),
    );
    ctx.add_optimizer_rule(Arc::new(USearchRule::new(registry)));
    Ok(())
}

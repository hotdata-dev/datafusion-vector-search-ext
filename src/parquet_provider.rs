// parquet_provider.rs — ParquetLookupProvider backed by any ObjectStore.
//
// No rg_map needed. Each usearch key directly encodes (file_idx, rg_idx, local_offset)
// via pack_key / unpack_key. Decoding is O(1) bitwise — no binary search.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::catalog::Session;
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use futures::future::{BoxFuture, try_join_all};
use object_store::ObjectStore;
use object_store::path::Path;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection, RowSelector};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::file::metadata::ParquetMetaData;

use crate::keys::unpack_key;
use crate::lookup::PointLookupProvider;

pub struct ParquetLookupProvider {
    /// Object-store path per file, indexed by file_idx encoded in the usearch key.
    file_keys: Vec<String>,
    store: Arc<dyn ObjectStore>,
    pub schema: SchemaRef,
    /// Parquet column indices for the provider schema positions 1, 2, …
    parquet_col_indices: Vec<usize>,
    /// When true, use `with_row_selection` to skip pages for non-target rows.
    use_row_selection: bool,
    /// Decoded parquet footer per file, cached at init time so fetch_by_keys
    /// never reads the footer over the network.
    metadata_cache: Vec<Arc<ParquetMetaData>>,
}

impl fmt::Debug for ParquetLookupProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParquetLookupProvider(files={}, schema_cols={})",
            self.file_keys.len(),
            self.schema.fields().len()
        )
    }
}

impl ParquetLookupProvider {
    /// Build the provider and eagerly cache parquet footers (one round-trip per file).
    pub async fn new(
        file_keys: Vec<String>,
        store: Arc<dyn ObjectStore>,
        schema: SchemaRef,
        parquet_col_indices: Vec<usize>,
    ) -> Result<Self> {
        let metadata_cache = load_metadata_cache(&file_keys, &store).await?;
        Ok(Self {
            file_keys,
            store,
            schema,
            parquet_col_indices,
            use_row_selection: false,
            metadata_cache,
        })
    }

    /// Same as [`new`], but enables row-selection to skip non-target pages during fetch.
    ///
    /// [`new`]: ParquetLookupProvider::new
    pub async fn new_with_row_selection(
        file_keys: Vec<String>,
        store: Arc<dyn ObjectStore>,
        schema: SchemaRef,
        parquet_col_indices: Vec<usize>,
    ) -> Result<Self> {
        let metadata_cache = load_metadata_cache(&file_keys, &store).await?;
        Ok(Self {
            file_keys,
            store,
            schema,
            parquet_col_indices,
            use_row_selection: true,
            metadata_cache,
        })
    }
}

/// Wraps `ParquetObjectReader` and short-circuits `get_metadata` with a
/// pre-loaded footer, eliminating 2 HTTP round trips per query.
struct CachedMetaReader {
    inner: ParquetObjectReader,
    meta: Arc<ParquetMetaData>,
}

impl AsyncFileReader for CachedMetaReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let meta = self.meta.clone();
        Box::pin(async move { Ok(meta) })
    }
}

/// Read the parquet footer for every file in parallel and cache it.
/// Called once at provider init — eliminates all per-query footer fetches.
async fn load_metadata_cache(
    file_keys: &[String],
    store: &Arc<dyn ObjectStore>,
) -> Result<Vec<Arc<ParquetMetaData>>> {
    let futs = file_keys.iter().map(|key| {
        let store = store.clone();
        let key = key.clone();
        async move {
            let path = Path::parse(&key).map_err(|e| DataFusionError::Execution(format!("{e}")))?;
            let reader = ParquetObjectReader::new(store, path);
            let builder = ParquetRecordBatchStreamBuilder::new(reader)
                .await
                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
            Ok::<Arc<ParquetMetaData>, DataFusionError>(builder.metadata().clone())
        }
    });
    try_join_all(futs).await
}

#[async_trait]
impl PointLookupProvider for ParquetLookupProvider {
    async fn fetch_by_keys(
        &self,
        keys: &[u64],
        _key_col: &str,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        // Decode every key in O(1) and group by (file_idx, rg_idx).
        let mut groups: BTreeMap<(usize, usize), Vec<(u64, usize)>> = BTreeMap::new();
        for &key in keys {
            let (file_idx, rg_idx, local_offset) = unpack_key(key);
            groups
                .entry((file_idx, rg_idx))
                .or_default()
                .push((key, local_offset));
        }

        // Build parquet column projection (same for every row group read).
        let selected_parquet_cols: Arc<Vec<usize>> = Arc::new(match projection {
            None => self.parquet_col_indices.clone(),
            Some(idxs) => idxs
                .iter()
                .filter(|&&i| i > 0 && (i - 1) < self.parquet_col_indices.len())
                .map(|&i| self.parquet_col_indices[i - 1])
                .collect(),
        });
        let out_schema: SchemaRef = match projection {
            None => self.schema.clone(),
            Some(idxs) => Arc::new(arrow_schema::Schema::new(
                idxs.iter()
                    .map(|&i| self.schema.field(i).clone())
                    .collect::<Vec<_>>(),
            )),
        };
        let projection_owned: Option<Arc<Vec<usize>>> = projection.map(|p| Arc::new(p.to_vec()));
        let use_row_selection = self.use_row_selection;

        // Fan out all row-group reads concurrently.
        let futures: Vec<_> = groups
            .into_iter()
            .map(|((file_idx, rg_idx), mut kv_pairs)| {
                let store = self.store.clone();
                let file_key = self.file_keys[file_idx].clone();
                let cached_meta = self.metadata_cache[file_idx].clone();
                let selected_parquet_cols = selected_parquet_cols.clone();
                let out_schema = out_schema.clone();
                let projection_owned = projection_owned.clone();

                async move {
                    let path = Path::parse(&file_key)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                    let reader = CachedMetaReader {
                        inner: ParquetObjectReader::new(store, path),
                        meta: cached_meta,
                    };
                    let builder = ParquetRecordBatchStreamBuilder::new(reader)
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                    let mask = ProjectionMask::roots(
                        builder.parquet_schema(),
                        selected_parquet_cols.iter().copied(),
                    );

                    // Sort kv_pairs by local_offset so row_idx values are in order.
                    kv_pairs.sort_by_key(|&(_, l)| l);

                    let (stream_builder, global_keys) = if use_row_selection {
                        let mut sel: Vec<RowSelector> = Vec::new();
                        let mut prev = 0usize;
                        for &(_, off) in &kv_pairs {
                            if off > prev {
                                sel.push(RowSelector::skip(off - prev));
                            }
                            sel.push(RowSelector::select(1));
                            prev = off + 1;
                        }
                        let row_selection = RowSelection::from(sel);
                        let gkeys: Vec<u64> = kv_pairs.iter().map(|&(k, _)| k).collect();
                        (
                            builder
                                .with_projection(mask)
                                .with_row_groups(vec![rg_idx])
                                .with_row_selection(row_selection),
                            gkeys,
                        )
                    } else {
                        (
                            builder.with_projection(mask).with_row_groups(vec![rg_idx]),
                            Vec::new(),
                        )
                    };

                    let mut stream = stream_builder
                        .build()
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                    let mut rg_batches: Vec<RecordBatch> = Vec::new();
                    while let Some(r) = stream.next().await {
                        rg_batches.push(r.map_err(|e| DataFusionError::External(Box::new(e)))?);
                    }
                    if rg_batches.is_empty() {
                        return Ok::<Option<RecordBatch>, DataFusionError>(None);
                    }

                    let combined = if rg_batches.len() == 1 {
                        rg_batches.remove(0)
                    } else {
                        let schema = rg_batches[0].schema();
                        datafusion::arrow::compute::concat_batches(&schema, &rg_batches)
                            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
                    };

                    let (filtered, global_keys) = if use_row_selection {
                        let cols: Vec<Arc<dyn Array>> = combined.columns().to_vec();
                        (cols, global_keys)
                    } else {
                        let local_set: std::collections::HashSet<usize> =
                            kv_pairs.iter().map(|&(_, l)| l).collect();
                        let n_rows = combined.num_rows();
                        let mask_bools: Vec<bool> =
                            (0..n_rows).map(|i| local_set.contains(&i)).collect();
                        if !mask_bools.iter().any(|&b| b) {
                            return Ok(None);
                        }
                        let bool_arr = BooleanArray::from(mask_bools);
                        let cols: Vec<Arc<dyn Array>> = combined
                            .columns()
                            .iter()
                            .map(|col| {
                                datafusion::arrow::compute::filter(col.as_ref(), &bool_arr)
                                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                            })
                            .collect::<Result<_>>()?;
                        let gkeys: Vec<u64> = (0..n_rows)
                            .filter(|i| local_set.contains(i))
                            .map(|local| {
                                kv_pairs
                                    .iter()
                                    .find(|&&(_, l)| l == local)
                                    .map(|&(k, _)| k)
                                    .unwrap_or(u64::MAX)
                            })
                            .collect();
                        (cols, gkeys)
                    };

                    let row_idx_arr: Arc<dyn Array> = Arc::new(UInt64Array::from(global_keys));

                    let out_cols: Vec<Arc<dyn Array>> = match projection_owned.as_deref() {
                        None => {
                            let mut cols = vec![row_idx_arr];
                            cols.extend(filtered);
                            cols
                        }
                        Some(idxs) => {
                            let mut content = filtered.into_iter();
                            idxs.iter()
                                .map(|&i| {
                                    if i == 0 {
                                        row_idx_arr.clone()
                                    } else {
                                        content.next().expect("column mismatch")
                                    }
                                })
                                .collect()
                        }
                    };

                    let batch = RecordBatch::try_new(out_schema, out_cols)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                    Ok(Some(batch))
                }
            })
            .collect();

        let results: Vec<Option<RecordBatch>> = try_join_all(futures).await?;
        Ok(results.into_iter().flatten().collect())
    }
}

#[async_trait]
impl datafusion::catalog::TableProvider for ParquetLookupProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        MemTable::try_new(self.schema.clone(), vec![])?
            .scan(state, projection, &[], None)
            .await
    }
}

// sqlite_provider.rs — SQLite-backed PointLookupProvider.
//
// Stores all non-embedding columns in a local SQLite database (bundled libsqlite3).
// Scalar columns map to INTEGER/TEXT/REAL; list columns are serialised as JSON TEXT.
// Lookups use `WHERE row_idx IN (?, ...)` against the INTEGER PRIMARY KEY B-tree.
//
// Schema: row_idx INTEGER PRIMARY KEY, <col> TEXT/INTEGER/REAL, ...
//
// Persistence: the database is written once to the given path and reused on
// subsequent runs. The first build reads all parquet files and inserts rows
// inside a single transaction.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow_array::builder::{Int32Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use rusqlite::{Connection, types::Value as SqlValue};
use tokio::sync::Semaphore;

use crate::keys::{DatasetLayout, pack_key};
use crate::lookup::PointLookupProvider;

// ── Provider ──────────────────────────────────────────────────────────────────

pub struct SqliteLookupProvider {
    schema: SchemaRef,
    table_name: String,
    pool: Arc<Mutex<Vec<Connection>>>,
    sem: Arc<Semaphore>,
}

impl fmt::Debug for SqliteLookupProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SqliteLookupProvider(table={}, schema_cols={})",
            self.table_name,
            self.schema.fields().len()
        )
    }
}

/// RAII guard that returns the connection to the pool on drop, even on panic.
struct ConnGuard {
    pool: Arc<Mutex<Vec<Connection>>>,
    conn: Option<Connection>,
}

impl ConnGuard {
    fn new(pool: Arc<Mutex<Vec<Connection>>>, conn: Connection) -> Self {
        Self {
            pool,
            conn: Some(conn),
        }
    }
}

impl Drop for ConnGuard {
    fn drop(&mut self) {
        if let Some(c) = self.conn.take() {
            // best-effort: ignore poison so a panicking query doesn't
            // permanently shrink the pool.
            let _ = self.pool.lock().map(|mut p| p.push(c));
        }
    }
}

/// Double-quote a SQLite identifier, escaping embedded double-quotes by
/// doubling them.  This prevents SQL injection when a caller-supplied name
/// is interpolated into a statement as an identifier.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn open_conn(db_path: &str) -> DFResult<Connection> {
    let conn = Connection::open(db_path).map_err(|e| DataFusionError::Execution(e.to_string()))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous  = NORMAL;
         PRAGMA cache_size   = -65536;",
    )
    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    Ok(conn)
}

impl SqliteLookupProvider {
    /// Open the existing SQLite database at `db_path`, or build it from
    /// parquet files on first run.  Opens a pool of `pool_size` read
    /// connections (WAL allows N concurrent readers).
    ///
    /// `local_parquet_files`, `layout`, `schema`, and `parquet_col_indices`
    /// are only used if the table does not yet exist.
    #[allow(clippy::too_many_arguments)]
    pub fn open_or_build(
        db_path: &str,
        table_name: &str,
        pool_size: usize,
        local_parquet_files: &[String],
        layout: &DatasetLayout,
        schema: SchemaRef,
        parquet_col_indices: &[usize],
    ) -> DFResult<Self> {
        if pool_size == 0 {
            return Err(DataFusionError::Execution(
                "pool_size must be at least 1".into(),
            ));
        }
        let conn = open_conn(db_path)?;

        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table_name],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| DataFusionError::Execution(e.to_string()))?
            > 0;

        if table_exists {
            let n: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {}", quote_ident(table_name)),
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            tracing::info!(
                "SQLite table '{}' already exists ({} rows), skipping build.",
                table_name,
                n
            );
        } else {
            tracing::info!(
                "First run: building SQLite table '{}' (one-time).",
                table_name
            );
            build_table(
                &conn,
                table_name,
                local_parquet_files,
                layout,
                &schema,
                parquet_col_indices,
            )?;
        }

        let mut conns = vec![conn];
        for _ in 1..pool_size {
            conns.push(open_conn(db_path)?);
        }
        Ok(Self {
            schema,
            table_name: table_name.to_string(),
            pool: Arc::new(Mutex::new(conns)),
            sem: Arc::new(Semaphore::new(pool_size)),
        })
    }
}

// ── PointLookupProvider ───────────────────────────────────────────────────────

#[async_trait]
impl PointLookupProvider for SqliteLookupProvider {
    async fn fetch_by_keys(
        &self,
        keys: &[u64],
        _key_col: &str,
        projection: Option<&[usize]>,
    ) -> DFResult<Vec<RecordBatch>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        let out_schema = match projection {
            None => self.schema.clone(),
            Some(idxs) => Arc::new(arrow_schema::Schema::new(
                idxs.iter()
                    .map(|&i| self.schema.field(i).clone())
                    .collect::<Vec<_>>(),
            )),
        };
        let keys_vec = keys.to_vec();
        let pool = self.pool.clone();
        let table_name = self.table_name.clone();

        // Acquire a semaphore permit to bound concurrency to the pool size,
        // then run the synchronous SQLite query on a blocking thread.
        let _permit = self
            .sem
            .acquire()
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        let result = tokio::task::spawn_blocking(move || {
            let conn = pool
                .lock()
                .map_err(|e| {
                    DataFusionError::Execution(format!("connection pool mutex poisoned: {e}"))
                })?
                .pop()
                .ok_or_else(|| {
                    DataFusionError::Execution("connection pool unexpectedly empty".into())
                })?;
            let guard = ConnGuard::new(pool, conn);
            let res = execute_query_sync(
                guard.conn.as_ref().unwrap(),
                &keys_vec,
                &out_schema,
                &table_name,
            );
            drop(guard); // explicit but not required — Drop handles it
            res
        })
        .await
        .map_err(|e| DataFusionError::Execution(e.to_string()))??;

        Ok(result)
    }
}

fn execute_query_sync(
    conn: &Connection,
    keys: &[u64],
    out_schema: &SchemaRef,
    table_name: &str,
) -> DFResult<Vec<RecordBatch>> {
    let placeholders = keys.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    // Select only the columns in out_schema (already projection-applied by the
    // caller) so we don't fetch unused columns from SQLite.
    let col_list = out_schema
        .fields()
        .iter()
        .map(|f| quote_ident(f.name()))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT {col_list} FROM {tn} WHERE row_idx IN ({placeholders}) ORDER BY row_idx",
        tn = quote_ident(table_name)
    );

    let n_out = out_schema.fields().len();
    let mut col_bufs: Vec<Vec<SqlValue>> = vec![Vec::with_capacity(keys.len()); n_out];

    let key_params: Vec<SqlValue> = keys.iter().map(|&k| SqlValue::Integer(k as i64)).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    let mut rows = stmt
        .query(rusqlite::params_from_iter(key_params.iter()))
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    while let Some(row) = rows
        .next()
        .map_err(|e| DataFusionError::Execution(e.to_string()))?
    {
        for (out_idx, buf) in col_bufs.iter_mut().enumerate() {
            let v: SqlValue = row
                .get(out_idx)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            buf.push(v);
        }
    }

    if col_bufs.first().is_none_or(|v| v.is_empty()) {
        return Ok(vec![]);
    }

    let arrays: Vec<ArrayRef> = out_schema
        .fields()
        .iter()
        .zip(col_bufs)
        .map(|(field, values)| sql_values_to_arrow(field.data_type(), values))
        .collect::<DFResult<_>>()?;

    let batch = RecordBatch::try_new(out_schema.clone(), arrays)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
    Ok(vec![batch])
}

// ── TableProvider ─────────────────────────────────────────────────────────────

#[async_trait]
impl TableProvider for SqliteLookupProvider {
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
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(SqliteFullScanExec::new(
            self.pool.clone(),
            self.sem.clone(),
            self.table_name.clone(),
            self.schema.clone(),
        )))
    }
}

// ── Full-scan execution plan ──────────────────────────────────────────────────

/// Batch size used when streaming rows from SQLite during a full table scan.
/// Larger values reduce round-trip overhead; smaller values reduce peak memory.
const SCAN_BATCH_SIZE: usize = 1024;

/// Physical execution plan that streams all rows from a SQLite table in
/// [`SCAN_BATCH_SIZE`]-row batches.  Used by the adaptive filtered path in
/// `USearchExec` to evaluate WHERE-clause predicates without loading the
/// entire table into memory at once.
#[derive(Debug)]
struct SqliteFullScanExec {
    pool: Arc<Mutex<Vec<Connection>>>,
    sem: Arc<Semaphore>,
    table_name: String,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl SqliteFullScanExec {
    fn new(
        pool: Arc<Mutex<Vec<Connection>>>,
        sem: Arc<Semaphore>,
        table_name: String,
        schema: SchemaRef,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            pool,
            sem,
            table_name,
            schema,
            properties,
        }
    }
}

impl DisplayAs for SqliteFullScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SqliteFullScanExec: table={}", self.table_name)
    }
}

impl ExecutionPlan for SqliteFullScanExec {
    fn name(&self) -> &str {
        "SqliteFullScanExec"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "SqliteFullScanExec is a leaf node and takes no children".into(),
            ))
        }
    }

    fn execute(
        &self,
        _partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let pool = self.pool.clone();
        let sem = Arc::clone(&self.sem);
        let table_name = self.table_name.clone();
        let schema = self.schema.clone();

        // Bounded channel: backpressure limits how many batches are buffered
        // ahead of the consumer, keeping peak memory to O(batch_size × 2).
        let (tx, rx) = tokio::sync::mpsc::channel::<DFResult<RecordBatch>>(2);

        let schema_task = schema.clone();
        tokio::spawn(async move {
            // Acquire a semaphore permit so the scan counts against the
            // same concurrency limit as fetch_by_keys.
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx
                        .send(Err(DataFusionError::Execution(e.to_string())))
                        .await;
                    return;
                }
            };

            let conn = match pool.lock() {
                Ok(mut g) => g.pop().ok_or_else(|| {
                    DataFusionError::Execution("SqliteFullScanExec: connection pool empty".into())
                }),
                Err(e) => Err(DataFusionError::Execution(format!(
                    "connection pool mutex poisoned: {e}"
                ))),
            };
            let conn = match conn {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            let pool_c = pool.clone();
            let tx_c = tx.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || {
                let guard = ConnGuard::new(pool_c, conn);
                let conn = guard.conn.as_ref().unwrap();

                let col_list = schema_task
                    .fields()
                    .iter()
                    .map(|f| quote_ident(f.name()))
                    .collect::<Vec<_>>()
                    .join(", ");
                // No ORDER BY — the adaptive filter doesn't require ordering.
                let sql = format!("SELECT {col_list} FROM {}", quote_ident(&table_name));

                let mut stmt = match conn.prepare(&sql) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx_c.blocking_send(Err(DataFusionError::Execution(e.to_string())));
                        return;
                    }
                };

                let mut rows = match stmt.query([]) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx_c.blocking_send(Err(DataFusionError::Execution(e.to_string())));
                        return;
                    }
                };

                let n_cols = schema_task.fields().len();
                let mut col_bufs: Vec<Vec<SqlValue>> = (0..n_cols)
                    .map(|_| Vec::with_capacity(SCAN_BATCH_SIZE))
                    .collect();
                let mut rows_in_batch = 0usize;

                loop {
                    match rows.next() {
                        Ok(Some(row)) => {
                            let mut row_ok = true;
                            for (ci, buf) in col_bufs.iter_mut().enumerate() {
                                match row.get::<_, SqlValue>(ci) {
                                    Ok(v) => buf.push(v),
                                    Err(e) => {
                                        let _ = tx_c.blocking_send(Err(
                                            DataFusionError::Execution(e.to_string()),
                                        ));
                                        row_ok = false;
                                        break;
                                    }
                                }
                            }
                            if !row_ok {
                                // Discard partial row data so the final flush
                                // doesn't see mismatched column buffer lengths.
                                for buf in col_bufs.iter_mut() {
                                    buf.truncate(rows_in_batch);
                                }
                                break;
                            }
                            rows_in_batch += 1;
                            if rows_in_batch >= SCAN_BATCH_SIZE {
                                let drained: Vec<Vec<SqlValue>> = col_bufs
                                    .iter_mut()
                                    .map(|b| {
                                        std::mem::replace(b, Vec::with_capacity(SCAN_BATCH_SIZE))
                                    })
                                    .collect();
                                rows_in_batch = 0;
                                match build_scan_batch(&schema_task, drained) {
                                    Ok(batch) => {
                                        if tx_c.blocking_send(Ok(batch)).is_err() {
                                            return; // consumer dropped
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx_c.blocking_send(Err(e));
                                        return;
                                    }
                                }
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            let _ =
                                tx_c.blocking_send(Err(DataFusionError::Execution(e.to_string())));
                            return;
                        }
                    }
                }

                // Flush the last partial batch.
                if rows_in_batch > 0 {
                    match build_scan_batch(&schema_task, col_bufs) {
                        Ok(batch) => {
                            let _ = tx_c.blocking_send(Ok(batch));
                        }
                        Err(e) => {
                            let _ = tx_c.blocking_send(Err(e));
                        }
                    }
                }
            })
            .await
            {
                let _ = tx
                    .send(Err(DataFusionError::Execution(format!(
                        "scan task panicked: {e}"
                    ))))
                    .await;
            }
        });

        // Convert the channel receiver into a RecordBatch stream.
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

/// Build a [`RecordBatch`] from column buffers of [`SqlValue`]s.
fn build_scan_batch(schema: &SchemaRef, col_bufs: Vec<Vec<SqlValue>>) -> DFResult<RecordBatch> {
    let arrays: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .zip(col_bufs)
        .map(|(field, values)| sql_values_to_arrow(field.data_type(), values))
        .collect::<DFResult<_>>()?;
    RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

// ── Build helpers ─────────────────────────────────────────────────────────────

fn build_table(
    conn: &Connection,
    table_name: &str,
    parquet_files: &[String],
    layout: &DatasetLayout,
    schema: &SchemaRef,
    parquet_col_indices: &[usize],
) -> DFResult<()> {
    let col_defs = schema
        .fields()
        .iter()
        .map(|f| {
            let sql_type = arrow_type_to_sql(f.data_type());
            if f.name() == "row_idx" {
                "row_idx INTEGER PRIMARY KEY".to_string()
            } else {
                format!("{} {}", quote_ident(f.name()), sql_type)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    let placeholders = schema
        .fields()
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let insert_sql = format!(
        "INSERT INTO {} VALUES ({placeholders})",
        quote_ident(table_name)
    );

    // CREATE TABLE and all INSERTs share one transaction so a mid-build crash
    // leaves no half-built table. If the table exists with zero rows on the
    // next startup, open_or_build would wrongly skip the build; atomicity
    // ensures the table either doesn't exist or is fully populated.
    let tx = conn
        .unchecked_transaction()
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    {
        tx.execute_batch(&format!(
            "CREATE TABLE {} ({col_defs});",
            quote_ident(table_name)
        ))
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        let mut stmt = tx
            .prepare(&insert_sql)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        for (file_idx, file_path) in parquet_files.iter().enumerate() {
            let f = std::fs::File::open(file_path)
                .map_err(|e| DataFusionError::Execution(format!("open {file_path}: {e}")))?;
            let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            let reader = builder
                .with_batch_size(2048)
                .build()
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            let mut file_row: u64 = 0;

            for batch_result in reader {
                let batch = batch_result.map_err(|e| DataFusionError::Execution(e.to_string()))?;
                let n = batch.num_rows();

                for row_i in 0..n {
                    let r = file_row + row_i as u64;
                    let rg = layout.rg_cum_rows[file_idx].partition_point(|&s| s <= r) - 1;
                    let lo = (r - layout.rg_cum_rows[file_idx][rg]) as usize;
                    let packed_key = pack_key(file_idx, rg, lo);

                    let mut params: Vec<SqlValue> = Vec::with_capacity(schema.fields().len());
                    params.push(SqlValue::Integer(packed_key as i64));

                    for &ci in parquet_col_indices {
                        params.push(arrow_cell_to_sql(batch.column(ci), row_i));
                    }

                    stmt.execute(rusqlite::params_from_iter(params.iter()))
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                }
                file_row += n as u64;
            }
        }
    }
    tx.commit()
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    tracing::info!("SQLite table '{}' built and committed.", table_name);
    Ok(())
}

// ── Type conversion helpers ───────────────────────────────────────────────────

fn arrow_type_to_sql(dt: &DataType) -> &'static str {
    match dt {
        DataType::UInt64 | DataType::UInt32 | DataType::Int32 | DataType::Int64 => "INTEGER",
        DataType::Float32 | DataType::Float64 => "REAL",
        _ => "TEXT", // Utf8, LargeUtf8, List variants → TEXT (JSON for lists)
    }
}

fn arrow_cell_to_sql(col: &ArrayRef, row: usize) -> SqlValue {
    if col.is_null(row) {
        return SqlValue::Null;
    }
    match col.data_type() {
        DataType::Utf8 => {
            let v = col
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row);
            SqlValue::Text(v.to_string())
        }
        DataType::LargeUtf8 => {
            let v = col
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .unwrap()
                .value(row);
            SqlValue::Text(v.to_string())
        }
        DataType::Int32 => SqlValue::Integer(
            col.as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row) as i64,
        ),
        DataType::Int64 => SqlValue::Integer(
            col.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
        ),
        DataType::UInt32 => SqlValue::Integer(
            col.as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .value(row) as i64,
        ),
        // UInt64 values > i64::MAX (2^63-1) will wrap to negative when cast to
        // SQLite INTEGER. This is acceptable for packed usearch keys (which use
        // only 63 bits) but callers storing arbitrary u64 data should be aware.
        DataType::UInt64 => SqlValue::Integer(
            col.as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row) as i64,
        ),
        DataType::Float32 => SqlValue::Real(
            col.as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Float64 => SqlValue::Real(
            col.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row),
        ),
        DataType::List(_) | DataType::LargeList(_) => SqlValue::Text(serialize_list(col, row)),
        _ => SqlValue::Null,
    }
}

fn serialize_list(col: &ArrayRef, row: usize) -> String {
    use serde_json::Value as JV;

    let list_val: ArrayRef =
        if let Some(arr) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
            arr.value(row)
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            arr.value(row)
        } else {
            return "[]".to_string();
        };

    let items: Vec<JV> = (0..list_val.len())
        .map(|i| {
            if list_val.is_null(i) {
                return JV::Null;
            }
            match list_val.data_type() {
                DataType::Utf8 => {
                    let s = list_val
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(i);
                    JV::String(s.to_string())
                }
                DataType::LargeUtf8 => {
                    let s = list_val
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .unwrap()
                        .value(i);
                    JV::String(s.to_string())
                }
                DataType::Int64 => {
                    let v = list_val
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(i);
                    JV::Number(v.into())
                }
                DataType::Int32 => {
                    let v = list_val
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .value(i);
                    JV::Number(v.into())
                }
                _ => JV::Null,
            }
        })
        .collect();

    serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_string())
}

fn sql_values_to_arrow(dt: &DataType, values: Vec<SqlValue>) -> DFResult<ArrayRef> {
    Ok(match dt {
        DataType::UInt64 => {
            let arr: UInt64Array = values
                .iter()
                .map(|v| match v {
                    SqlValue::Integer(i) => Some(*i as u64),
                    _ => None,
                })
                .collect();
            Arc::new(arr)
        }
        DataType::UInt32 => {
            let arr: UInt32Array = values
                .iter()
                .map(|v| match v {
                    SqlValue::Integer(i) => Some(*i as u32),
                    _ => None,
                })
                .collect();
            Arc::new(arr)
        }
        DataType::Int32 => {
            let mut b = Int32Builder::with_capacity(values.len());
            for v in &values {
                match v {
                    SqlValue::Integer(i) => b.append_value(*i as i32),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(values.len());
            for v in &values {
                match v {
                    SqlValue::Integer(i) => b.append_value(*i),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(values.len(), values.len() * 32);
            for v in &values {
                match v {
                    SqlValue::Text(s) => b.append_value(s),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::List(item_field) => match item_field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {
                let mut b =
                    ListBuilder::new(StringBuilder::new()).with_field(item_field.as_ref().clone());
                for v in &values {
                    match v {
                        SqlValue::Text(s) => {
                            let items: Vec<Option<String>> =
                                serde_json::from_str(s).unwrap_or_default();
                            for item in items {
                                b.values().append_option(item);
                            }
                            b.append(true);
                        }
                        _ => b.append(false),
                    }
                }
                Arc::new(b.finish())
            }
            DataType::Int64 => {
                let mut b =
                    ListBuilder::new(Int64Builder::new()).with_field(item_field.as_ref().clone());
                for v in &values {
                    match v {
                        SqlValue::Text(s) => {
                            let items: Vec<Option<i64>> =
                                serde_json::from_str(s).unwrap_or_default();
                            for item in items {
                                b.values().append_option(item);
                            }
                            b.append(true);
                        }
                        _ => b.append(false),
                    }
                }
                Arc::new(b.finish())
            }
            DataType::Int32 => {
                let mut b =
                    ListBuilder::new(Int32Builder::new()).with_field(item_field.as_ref().clone());
                for v in &values {
                    match v {
                        SqlValue::Text(s) => {
                            let items: Vec<Option<i32>> =
                                serde_json::from_str(s).unwrap_or_default();
                            for item in items {
                                b.values().append_option(item);
                            }
                            b.append(true);
                        }
                        _ => b.append(false),
                    }
                }
                Arc::new(b.finish())
            }
            inner => Err(DataFusionError::Execution(format!(
                "SqliteLookupProvider: unsupported list item type {inner:?}"
            )))?,
        },
        DataType::Float64 => {
            let arr: Float64Array = values
                .iter()
                .map(|v| match v {
                    SqlValue::Real(f) => Some(*f),
                    _ => None,
                })
                .collect();
            Arc::new(arr)
        }
        DataType::Float32 => {
            let arr: Float32Array = values
                .iter()
                .map(|v| match v {
                    SqlValue::Real(f) => Some(*f as f32),
                    _ => None,
                })
                .collect();
            Arc::new(arr)
        }
        other => Err(DataFusionError::Execution(format!(
            "SqliteLookupProvider: unsupported Arrow type {other:?}"
        )))?,
    })
}

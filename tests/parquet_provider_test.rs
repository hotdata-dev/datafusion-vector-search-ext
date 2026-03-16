#![cfg(feature = "parquet-provider")]

use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use datafusion_vector_search_ext::{ParquetLookupProvider, PointLookupProvider, pack_key};
use object_store::local::LocalFileSystem;
use parquet::arrow::ArrowWriter;
use tempfile::tempdir;

/// Write a small 3-row, 1-column parquet file to `dir/test.parquet` and
/// return a `ParquetLookupProvider` wrapping it.
async fn make_provider(dir: &tempfile::TempDir) -> ParquetLookupProvider {
    // Parquet schema: just one column (without synthesised row_idx).
    let parquet_schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));

    // Provider schema = row_idx (synthesised) + name.
    let provider_schema = Arc::new(Schema::new(vec![
        Field::new("row_idx", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        parquet_schema.clone(),
        vec![Arc::new(StringArray::from(vec![
            Some("alice"),
            Some("bob"),
            Some("carol"),
        ]))],
    )
    .unwrap();

    let file_path = dir.path().join("test.parquet");
    let file = std::fs::File::create(&file_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, parquet_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    // parquet_col_indices: provider col 1 (name) maps to parquet col 0
    ParquetLookupProvider::new(
        vec!["test.parquet".to_string()],
        store,
        provider_schema,
        vec![0],
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn test_fetch_existing_keys() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir).await;

    // Fetch rows 0 and 2 (alice and carol).
    let key0 = pack_key(0, 0, 0);
    let key2 = pack_key(0, 0, 2);
    let batches = provider
        .fetch_by_keys(&[key0, key2], "row_idx", None)
        .await
        .unwrap();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);

    // Both batches combined should contain the correct names.
    let names: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            b.column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
        })
        .collect();
    assert!(names.contains(&"alice".to_string()));
    assert!(names.contains(&"carol".to_string()));
    assert!(!names.contains(&"bob".to_string()));
}

#[tokio::test]
async fn test_projection() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir).await;

    let key1 = pack_key(0, 0, 1);
    // Project only row_idx (index 0).
    let batches = provider
        .fetch_by_keys(&[key1], "row_idx", Some(&[0]))
        .await
        .unwrap();

    assert!(!batches.is_empty());
    // Returned schema should have only row_idx.
    assert_eq!(batches[0].schema().fields().len(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "row_idx");

    let row_idx_col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(row_idx_col.value(0), key1);
}

#[tokio::test]
async fn test_missing_keys_return_empty() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir).await;

    // Keys that reference file_idx=1 which doesn't exist — provider would
    // panic on index, so use a local_offset beyond the 3-row file instead.
    // Actually the easiest way is to request a row that passes filter but
    // whose local_offset is beyond the 3 rows → bool mask is all false → None.
    let missing = pack_key(0, 0, 99); // local_offset 99, file has only 3 rows
    let batches = provider
        .fetch_by_keys(&[missing], "row_idx", None)
        .await
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0);
}

#[tokio::test]
async fn test_empty_key_slice() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir).await;

    let batches = provider.fetch_by_keys(&[], "row_idx", None).await.unwrap();
    assert!(batches.is_empty());
}

#[tokio::test]
async fn test_row_selection_variant() {
    let dir = tempdir().unwrap();
    let parquet_schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));
    let provider_schema = Arc::new(Schema::new(vec![
        Field::new("row_idx", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        parquet_schema.clone(),
        vec![Arc::new(StringArray::from(vec![
            Some("alice"),
            Some("bob"),
            Some("carol"),
        ]))],
    )
    .unwrap();

    let file_path = dir.path().join("test_rs.parquet");
    let file = std::fs::File::create(&file_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, parquet_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let provider = ParquetLookupProvider::new_with_row_selection(
        vec!["test_rs.parquet".to_string()],
        store,
        provider_schema,
        vec![0],
    )
    .await
    .unwrap();

    let key0 = pack_key(0, 0, 0);
    let key2 = pack_key(0, 0, 2);
    let batches = provider
        .fetch_by_keys(&[key0, key2], "row_idx", None)
        .await
        .unwrap();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
}

/// Regression test for the projection column ordering bug:
/// when `selected_parquet_cols` is not monotonically increasing,
/// `ProjectionMask::roots` still returns columns in parquet schema order.
/// The provider must reorder them back to match the requested `idxs` order.
#[tokio::test]
async fn test_projection_non_monotonic_column_order() {
    let dir = tempdir().unwrap();

    // Parquet schema: col_a (UInt32, parquet idx 0), col_b (Utf8, parquet idx 1).
    let parquet_schema = Arc::new(Schema::new(vec![
        Field::new("col_a", DataType::UInt32, false),
        Field::new("col_b", DataType::Utf8, true),
    ]));
    // Provider schema: row_idx (idx 0), col_a (idx 1, parquet 0), col_b (idx 2, parquet 1).
    let provider_schema = Arc::new(Schema::new(vec![
        Field::new("row_idx", DataType::UInt64, false),
        Field::new("col_a", DataType::UInt32, false),
        Field::new("col_b", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        parquet_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![10u32, 20, 30])),
            Arc::new(StringArray::from(vec![
                Some("alice"),
                Some("bob"),
                Some("carol"),
            ])),
        ],
    )
    .unwrap();

    let file_path = dir.path().join("two_col.parquet");
    let file = std::fs::File::create(&file_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, parquet_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    // parquet_col_indices: provider col 1 → parquet col 0 (col_a),
    //                      provider col 2 → parquet col 1 (col_b).
    let provider = ParquetLookupProvider::new(
        vec!["two_col.parquet".to_string()],
        store,
        provider_schema,
        vec![0, 1],
    )
    .await
    .unwrap();

    // Project [col_b (2), col_a (1)] — reverse order, non-monotonic parquet indices.
    // selected_parquet_cols becomes [1, 0]; without the reorder fix the values
    // would be swapped (col_a value 10 returned as col_b, etc.).
    let key0 = pack_key(0, 0, 0);
    let batches = provider
        .fetch_by_keys(&[key0], "row_idx", Some(&[2, 1]))
        .await
        .unwrap();

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.schema().field(0).name(), "col_b");
    assert_eq!(batch.schema().field(1).name(), "col_a");

    let col_b = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let col_a = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    assert_eq!(col_b.value(0), "alice");
    assert_eq!(col_a.value(0), 10);
}

/// Regression test for the stale-key bounds check:
/// a packed key referencing a file_idx beyond the provider's file list must
/// return Err, not panic via an out-of-bounds index.
#[tokio::test]
async fn test_stale_file_idx_returns_error() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir).await; // single-file provider (file_idx 0 only)

    let stale_key = pack_key(1, 0, 0); // file_idx=1 doesn't exist
    let result = provider.fetch_by_keys(&[stale_key], "row_idx", None).await;
    assert!(result.is_err(), "expected Err for out-of-bounds file_idx");
    assert!(result.unwrap_err().to_string().contains("file_idx=1"));
}

/// Regression test for the silent-empty-scan bug:
/// scan() used to return an empty MemTable, producing zero rows with no error.
/// It must now return NotImplemented so callers get a clear failure.
#[tokio::test]
async fn test_scan_returns_not_implemented() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir).await;

    let ctx = SessionContext::new();
    let state = ctx.state();
    let result = provider.scan(&state, None, &[], None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not support full table scans"),
        "expected NotImplemented error, got: {err}"
    );
}

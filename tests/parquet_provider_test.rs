#![cfg(feature = "parquet-provider")]

use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
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

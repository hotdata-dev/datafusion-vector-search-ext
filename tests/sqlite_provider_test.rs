#![cfg(feature = "sqlite-provider")]

use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use datafusion_vector_search_ext::{
    DatasetLayout, PointLookupProvider, SqliteLookupProvider, pack_key,
};
use parquet::arrow::ArrowWriter;
use tempfile::tempdir;

/// Write a small 3-row parquet file and build a `SqliteLookupProvider` from it.
fn make_provider(dir: &tempfile::TempDir) -> SqliteLookupProvider {
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

    let parquet_path = dir.path().join("test.parquet");
    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, parquet_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Build a minimal DatasetLayout for 1 file with 1 row group of 3 rows.
    let layout = DatasetLayout {
        file_keys: vec!["parquet/test.parquet".to_string()],
        file_cum_rows: vec![0, 3],
        rg_cum_rows: vec![vec![0, 3]],
    };

    let db_path = dir.path().join("test.db");
    let parquet_files = vec![parquet_path.to_str().unwrap().to_string()];

    SqliteLookupProvider::open_or_build(
        db_path.to_str().unwrap(),
        "models",
        4,
        &parquet_files,
        &layout,
        provider_schema,
        &[0], // parquet col 0 (name) → provider col 1
    )
    .unwrap()
}

#[tokio::test]
async fn test_fetch_existing_keys() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir);

    let key0 = pack_key(0, 0, 0);
    let key2 = pack_key(0, 0, 2);
    let batches = provider
        .fetch_by_keys(&[key0, key2], "row_idx", None)
        .await
        .unwrap();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);

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
    let provider = make_provider(&dir);

    let key1 = pack_key(0, 0, 1);
    // Project only row_idx (index 0).
    let batches = provider
        .fetch_by_keys(&[key1], "row_idx", Some(&[0]))
        .await
        .unwrap();

    assert!(!batches.is_empty());
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
    let provider = make_provider(&dir);

    let missing = pack_key(0, 0, 99); // offset 99 doesn't exist
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
    let provider = make_provider(&dir);

    let batches = provider.fetch_by_keys(&[], "row_idx", None).await.unwrap();
    assert!(batches.is_empty());
}

/// Regression test for the silent-empty-scan bug:
/// scan() used to return an empty MemTable, producing zero rows with no error.
/// It must now return NotImplemented so callers get a clear failure.
#[tokio::test]
async fn test_scan_returns_not_implemented() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir);

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

/// Regression test for the SQL injection fix via quote_ident:
/// a table name containing spaces (and thus requiring quoting) must work
/// correctly rather than producing a SQL syntax error.
#[tokio::test]
async fn test_table_name_with_spaces() {
    let dir = tempdir().unwrap();
    let parquet_schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));
    let provider_schema = Arc::new(Schema::new(vec![
        Field::new("row_idx", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        parquet_schema.clone(),
        vec![Arc::new(StringArray::from(vec![Some("alice")]))],
    )
    .unwrap();

    let parquet_path = dir.path().join("test.parquet");
    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, parquet_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let layout = DatasetLayout {
        file_keys: vec!["parquet/test.parquet".to_string()],
        file_cum_rows: vec![0, 1],
        rg_cum_rows: vec![vec![0, 1]],
    };

    let db_path = dir.path().join("test.db");
    // Table name with spaces — previously this would have produced a SQL syntax error.
    let provider = SqliteLookupProvider::open_or_build(
        db_path.to_str().unwrap(),
        "my models",
        2,
        &[parquet_path.to_str().unwrap().to_string()],
        &layout,
        provider_schema,
        &[0],
    )
    .unwrap();

    let key0 = pack_key(0, 0, 0);
    let batches = provider
        .fetch_by_keys(&[key0], "row_idx", None)
        .await
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

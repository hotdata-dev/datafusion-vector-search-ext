#![cfg(feature = "sqlite-provider")]

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use datafusion_vector_search_ext::{PointLookupProvider, SqliteLookupProvider};
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

    let db_path = dir.path().join("test.db");
    let parquet_files = vec![parquet_path.to_str().unwrap().to_string()];

    SqliteLookupProvider::open_or_build(
        db_path.to_str().unwrap(),
        "models",
        4,
        &parquet_files,
        provider_schema,
        &[0], // parquet col 0 (name) → provider col 1
    )
    .unwrap()
}

#[tokio::test]
async fn test_fetch_existing_keys() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir);

    let batches = provider
        .fetch_by_keys(&[0, 2], "row_idx", None)
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

    // Project only row_idx (index 0).
    let batches = provider
        .fetch_by_keys(&[1], "row_idx", Some(&[0]))
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
    assert_eq!(row_idx_col.value(0), 1);
}

#[tokio::test]
async fn test_missing_keys_return_empty() {
    let dir = tempdir().unwrap();
    let provider = make_provider(&dir);

    let batches = provider
        .fetch_by_keys(&[99], "row_idx", None)
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

/// scan() returns a streaming ExecutionPlan that yields all rows in batches.
#[tokio::test]
async fn test_scan_streams_all_rows() {
    use datafusion::execution::TaskContext;
    use futures::StreamExt;

    let dir = tempdir().unwrap();
    let provider = make_provider(&dir);

    let ctx = SessionContext::new();
    let state = ctx.state();
    let plan = provider.scan(&state, None, &[], None).await.unwrap();

    let task_ctx = Arc::new(TaskContext::default());
    let mut stream = plan.execute(0, task_ctx).unwrap();

    let mut total_rows = 0usize;
    let mut all_names: Vec<String> = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        total_rows += batch.num_rows();

        let names_col = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..names_col.len() {
            all_names.push(names_col.value(i).to_string());
        }
    }

    assert_eq!(total_rows, 3);
    assert!(all_names.contains(&"alice".to_string()));
    assert!(all_names.contains(&"bob".to_string()));
    assert!(all_names.contains(&"carol".to_string()));
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

    let db_path = dir.path().join("test.db");
    // Table name with spaces — previously this would have produced a SQL syntax error.
    let provider = SqliteLookupProvider::open_or_build(
        db_path.to_str().unwrap(),
        "my models",
        2,
        &[parquet_path.to_str().unwrap().to_string()],
        provider_schema,
        &[0],
    )
    .unwrap();

    let batches = provider.fetch_by_keys(&[0], "row_idx", None).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

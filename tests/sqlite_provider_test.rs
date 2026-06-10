#![cfg(feature = "sqlite-provider")]

use std::sync::Arc;

use arrow_array::{
    Array, Int64Array, LargeStringArray, RecordBatch, StringArray, StringViewArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use datafusion_vector_search_ext::{
    PointLookupProvider, SqliteLookupProvider, SqliteSidecarBuilder,
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
async fn test_stream_builder_with_explicit_rowid_keys() {
    // The streaming builder reads each row's key from a column (e.g. a storage
    // engine's native rowid) instead of synthesising 0..N. Keys here are sparse
    // and non-monotonic across two batches to prove that works end to end.
    let dir = tempdir().unwrap();

    let batch_schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    // Output schema: key column first, then the stored value column.
    let provider_schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let b1 = RecordBatch::try_new(
        batch_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![100_i64, 250])),
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob")])),
        ],
    )
    .unwrap();
    let b2 = RecordBatch::try_new(
        batch_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![999_i64])),
            Arc::new(StringArray::from(vec![Some("carol")])),
        ],
    )
    .unwrap();

    let db_path = dir.path().join("stream.db");
    let mut builder = SqliteSidecarBuilder::begin(
        db_path.to_str().unwrap(),
        "models",
        4,
        provider_schema,
        0,       // key (rowid) is column 0 of the input batches
        vec![1], // input column 1 (name) → provider field 1
    )
    .unwrap();
    builder.push_batch(&b1).unwrap();
    builder.push_batch(&b2).unwrap();
    let provider = builder.finish().unwrap();

    // Point-lookup by sparse rowids.
    let batches = provider
        .fetch_by_keys(&[100, 999], "rowid", None)
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
    assert!(!names.contains(&"bob".to_string())); // rowid 250 not requested

    // A rowid that was never inserted returns nothing.
    let empty = provider.fetch_by_keys(&[42], "rowid", None).await.unwrap();
    assert_eq!(empty.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
}

#[test]
fn test_stream_builder_abandon_rolls_back() {
    // Dropping the builder before finish() must roll the transaction back, so
    // the table is never persisted. We prove it by re-running begin() on the
    // same path: its CREATE TABLE only succeeds if the abandoned build left no
    // table behind.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("abandon.db");
    let db = db_path.to_str().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();

    {
        let mut builder =
            SqliteSidecarBuilder::begin(db, "models", 1, schema.clone(), 0, vec![1]).unwrap();
        builder.push_batch(&batch).unwrap();
        // builder dropped here without finish() → rollback
    }

    // A fresh build on the same path must succeed (table does not already exist).
    assert!(
        SqliteSidecarBuilder::begin(db, "models", 1, schema, 0, vec![1]).is_ok(),
        "abandoned build must not persist its table"
    );
}

#[tokio::test]
async fn test_stream_builder_uint64_keys() {
    // The key column may be UInt64 (as well as Int64); it is stored as SQLite
    // INTEGER and looked up via the u64 fetch API.
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![7_u64, 11])),
            Arc::new(StringArray::from(vec![Some("x"), Some("y")])),
        ],
    )
    .unwrap();
    let db_path = dir.path().join("u64.db");
    let mut builder =
        SqliteSidecarBuilder::begin(db_path.to_str().unwrap(), "t", 2, schema, 0, vec![1]).unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    let batches = provider.fetch_by_keys(&[11], "rowid", None).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

#[test]
fn test_stream_builder_validation_errors() {
    let dir = tempdir().unwrap();
    let db = |n: &str| dir.path().join(n).to_str().unwrap().to_string();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    // pool_size must be >= 1.
    assert!(SqliteSidecarBuilder::begin(&db("a.db"), "t", 0, schema.clone(), 0, vec![1]).is_err());

    // schema has 2 fields → exactly 1 value column index expected, not 2.
    assert!(
        SqliteSidecarBuilder::begin(&db("b.db"), "t", 1, schema.clone(), 0, vec![1, 2]).is_err()
    );

    // key_col_index out of range for the pushed batch (2 columns, index 9).
    let mut b_oob =
        SqliteSidecarBuilder::begin(&db("c.db"), "t", 1, schema.clone(), 9, vec![1]).unwrap();
    let two_col = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(StringArray::from(vec![Some("a")])),
        ],
    )
    .unwrap();
    assert!(b_oob.push_batch(&two_col).is_err());

    // value_col_index out of range for the pushed batch (clean error, no panic).
    let mut b_voob =
        SqliteSidecarBuilder::begin(&db("c2.db"), "t", 1, schema.clone(), 0, vec![9]).unwrap();
    assert!(b_voob.push_batch(&two_col).is_err());

    // A null key value is rejected.
    let nullable = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mut b_null =
        SqliteSidecarBuilder::begin(&db("d.db"), "t", 1, nullable.clone(), 0, vec![1]).unwrap();
    let null_key = RecordBatch::try_new(
        nullable,
        vec![
            Arc::new(Int64Array::from(vec![None, Some(2)])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    assert!(b_null.push_batch(&null_key).is_err());

    // A non-integer key column type is rejected.
    let text_key = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mut b_text =
        SqliteSidecarBuilder::begin(&db("e.db"), "t", 1, text_key.clone(), 0, vec![1]).unwrap();
    let text_batch = RecordBatch::try_new(
        text_key,
        vec![
            Arc::new(StringArray::from(vec![Some("k")])),
            Arc::new(StringArray::from(vec![Some("a")])),
        ],
    )
    .unwrap();
    assert!(b_text.push_batch(&text_batch).is_err());
}

/// Regression test for hotdata-dev/runtimedb#631: payload columns typed as
/// `LargeUtf8` (what DuckDB/parquet readers emit for strings) or `Utf8View`
/// (what a `::text` cast produces) must round-trip through the sidecar. The
/// write side already mapped both to TEXT, but the read-back path was missing
/// the reconstruction arms and failed with "unsupported Arrow type LargeUtf8".
#[tokio::test]
async fn test_string_view_and_large_utf8_roundtrip() {
    let dir = tempdir().unwrap();

    // key + a LargeUtf8 column + a Utf8View column.
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("large", DataType::LargeUtf8, true),
        Field::new("view", DataType::Utf8View, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![10_i64, 20, 30])),
            Arc::new(LargeStringArray::from(vec![
                Some("alpha"),
                None,
                Some("gamma"),
            ])),
            Arc::new(StringViewArray::from(vec![Some("one"), Some("two"), None])),
        ],
    )
    .unwrap();

    let db_path = dir.path().join("strings.db");
    let mut builder = SqliteSidecarBuilder::begin(
        db_path.to_str().unwrap(),
        "models",
        2,
        schema,
        0,          // key (rowid) is column 0
        vec![1, 2], // columns 1 (large) and 2 (view) are stored values
    )
    .unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    let batches = provider
        .fetch_by_keys(&[10, 30], "rowid", None)
        .await
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

    // The reconstructed columns must preserve their original Arrow types and values.
    let mut large: Vec<Option<String>> = Vec::new();
    let mut view: Vec<Option<String>> = Vec::new();
    for b in &batches {
        let l = b
            .column_by_name("large")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large column should reconstruct as LargeStringArray");
        let v = b
            .column_by_name("view")
            .unwrap()
            .as_any()
            .downcast_ref::<StringViewArray>()
            .expect("view column should reconstruct as StringViewArray");
        for i in 0..b.num_rows() {
            large.push(l.is_valid(i).then(|| l.value(i).to_string()));
            view.push(v.is_valid(i).then(|| v.value(i).to_string()));
        }
    }

    assert!(large.contains(&Some("alpha".to_string())));
    assert!(large.contains(&Some("gamma".to_string())));
    assert!(view.contains(&Some("one".to_string())));
    // rowid 30 had a null view value, which must survive the round-trip.
    assert!(view.contains(&None));
}

/// Companion to the scalar regression above: a `List<Utf8View>` payload must
/// also round-trip. The write side serializes list elements to JSON TEXT, so a
/// missing `Utf8View` reconstruction arm would write real values then fail on
/// read-back with "unsupported list item type Utf8View".
#[tokio::test]
async fn test_list_utf8view_roundtrip() {
    use arrow_array::ListArray;
    use arrow_array::builder::{ListBuilder, StringViewBuilder};

    let dir = tempdir().unwrap();

    let item_field = Arc::new(Field::new("item", DataType::Utf8View, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("tags", DataType::List(item_field.clone()), true),
    ]));

    // Four rows exercising: a populated list with a null element, another
    // populated list, a NULL list cell, and an empty list.
    let mut lb = ListBuilder::new(StringViewBuilder::new()).with_field(item_field);
    lb.values().append_value("red");
    lb.values().append_null();
    lb.append(true); // rowid 1: ["red", null]
    lb.values().append_value("blue");
    lb.values().append_value("green");
    lb.append(true); // rowid 2: ["blue", "green"]
    lb.append(false); // rowid 3: NULL list
    lb.append(true); // rowid 4: [] (empty list)
    let tags = lb.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4])),
            Arc::new(tags),
        ],
    )
    .unwrap();

    let db_path = dir.path().join("lists.db");
    let mut builder =
        SqliteSidecarBuilder::begin(db_path.to_str().unwrap(), "models", 2, schema, 0, vec![1])
            .unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    let batches = provider.fetch_by_keys(&[1], "rowid", None).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    let list = batches[0]
        .column_by_name("tags")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("tags should reconstruct as a List");
    let inner = list.value(0);
    let strs = inner
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("list items should reconstruct as StringViewArray");
    assert_eq!(strs.len(), 2);
    assert_eq!(strs.value(0), "red");
    assert!(strs.is_null(1));

    // A NULL list cell survives as a null list entry; an empty list survives
    // as a present-but-empty list (length 0), not as null.
    let nullable = provider
        .fetch_by_keys(&[3, 4], "rowid", None)
        .await
        .unwrap();
    let mut null_lists = 0usize;
    let mut empty_lists = 0usize;
    for b in &nullable {
        let rowid = b
            .column_by_name("rowid")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let list = b
            .column_by_name("tags")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            match rowid.value(i) {
                3 => {
                    assert!(list.is_null(i), "rowid 3 should be a NULL list");
                    null_lists += 1;
                }
                4 => {
                    assert!(list.is_valid(i), "rowid 4 should be a present empty list");
                    assert_eq!(list.value(i).len(), 0, "rowid 4 should be empty");
                    empty_lists += 1;
                }
                other => panic!("unexpected rowid {other}"),
            }
        }
    }
    assert_eq!(null_lists, 1);
    assert_eq!(empty_lists, 1);
}

/// A `List<LargeUtf8>` payload must reconstruct with a `LargeStringArray` child.
/// Arrow's `ListBuilder::finish()` panics if the declared item field type does
/// not match the child builder, so `LargeUtf8` lists must not be routed through
/// a `Utf8` `StringBuilder`.
#[tokio::test]
async fn test_list_largeutf8_roundtrip() {
    use arrow_array::ListArray;
    use arrow_array::builder::{LargeStringBuilder, ListBuilder};

    let dir = tempdir().unwrap();

    let item_field = Arc::new(Field::new("item", DataType::LargeUtf8, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("tags", DataType::List(item_field.clone()), true),
    ]));

    let mut lb = ListBuilder::new(LargeStringBuilder::new()).with_field(item_field);
    lb.values().append_value("red");
    lb.values().append_null();
    lb.append(true);
    let tags = lb.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1_i64])), Arc::new(tags)],
    )
    .unwrap();

    let db_path = dir.path().join("large_lists.db");
    let mut builder =
        SqliteSidecarBuilder::begin(db_path.to_str().unwrap(), "models", 2, schema, 0, vec![1])
            .unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    let batches = provider.fetch_by_keys(&[1], "rowid", None).await.unwrap();
    let list = batches[0]
        .column_by_name("tags")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("tags should reconstruct as a List");
    let inner = list.value(0);
    let strs = inner
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("list items should reconstruct as LargeStringArray");
    assert_eq!(strs.len(), 2);
    assert_eq!(strs.value(0), "red");
    assert!(strs.is_null(1));
}

/// A `LargeList<Utf8>` payload must round-trip. The write side serializes both
/// `List` and `LargeList` to JSON TEXT, but the read-back path only reconstructed
/// `List`, so a `LargeList` column failed with "unsupported Arrow type LargeList".
#[tokio::test]
async fn test_largelist_roundtrip() {
    use arrow_array::LargeListArray;
    use arrow_array::builder::{LargeListBuilder, StringBuilder};

    let dir = tempdir().unwrap();

    let item_field = Arc::new(Field::new("item", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("tags", DataType::LargeList(item_field.clone()), true),
    ]));

    let mut lb = LargeListBuilder::new(StringBuilder::new()).with_field(item_field);
    lb.values().append_value("red");
    lb.values().append_null();
    lb.append(true);
    let tags = lb.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1_i64])), Arc::new(tags)],
    )
    .unwrap();

    let db_path = dir.path().join("largelist.db");
    let mut builder =
        SqliteSidecarBuilder::begin(db_path.to_str().unwrap(), "models", 2, schema, 0, vec![1])
            .unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    let batches = provider.fetch_by_keys(&[1], "rowid", None).await.unwrap();
    let list = batches[0]
        .column_by_name("tags")
        .unwrap()
        .as_any()
        .downcast_ref::<LargeListArray>()
        .expect("tags should reconstruct as a LargeList");
    let inner = list.value(0);
    let strs = inner
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("list items should reconstruct as StringArray");
    assert_eq!(strs.len(), 2);
    assert_eq!(strs.value(0), "red");
    assert!(strs.is_null(1));
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

/// Verify that a non-default key column name (e.g. "_key") works correctly.
/// This is the scenario used by runtimedb where Parquet files have a `_key` column.
#[tokio::test]
async fn test_custom_key_column_name() {
    let dir = tempdir().unwrap();

    let parquet_schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));

    // Provider schema uses "_key" instead of the default "row_idx".
    let provider_schema = Arc::new(Schema::new(vec![
        Field::new("_key", DataType::UInt64, false),
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

    let db_path = dir.path().join("test_key.db");
    let provider = SqliteLookupProvider::open_or_build(
        db_path.to_str().unwrap(),
        "vectors",
        2,
        &[parquet_path.to_str().unwrap().to_string()],
        provider_schema,
        &[0],
    )
    .unwrap();

    // fetch_by_keys should work with the custom key column
    let batches = provider.fetch_by_keys(&[0, 2], "_key", None).await.unwrap();
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
                .map(|v| v.unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(names, vec!["alice", "carol"]);

    // projection to only the key column should also work
    let batches = provider
        .fetch_by_keys(&[1], "_key", Some(&[0]))
        .await
        .unwrap();
    assert_eq!(batches[0].schema().field(0).name(), "_key");
    let key_col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(key_col.value(0), 1);
}

/// The defect class behind hotdata-dev/runtimedb#631 was that the write side
/// accepted any Arrow type (silent TEXT/NULL fallback) while the read side only
/// reconstructed a subset — so an unsupported payload column built a "successful"
/// sidecar that then exploded on the first query. This asserts the failure now
/// surfaces at *build* time (`begin`) with the offending column named, which is
/// where runtimedb's index-create step will catch it.
#[tokio::test]
async fn test_unsupported_payload_type_rejected_at_build() {
    let dir = tempdir().unwrap();

    // Decimal128 is a common parquet type the read-back path cannot reconstruct.
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("price", DataType::Decimal128(10, 2), true),
    ]));

    let db_path = dir.path().join("bad.db");
    let err =
        SqliteSidecarBuilder::begin(db_path.to_str().unwrap(), "models", 2, schema, 0, vec![1])
            .map(|_| ())
            .expect_err("a Decimal128 payload column must be rejected at build time");

    let msg = err.to_string();
    assert!(
        msg.contains("price") && msg.contains("unsupported type"),
        "error should name the offending column and type, got: {msg}"
    );
}

/// Round-trips the basic scalar types added alongside the validation gate
/// (Boolean, Date32, and a timezone-carrying Timestamp). These are exactly the
/// kinds of columns DuckDB/parquet emit that previously fell through to the
/// "unsupported Arrow type" error on read-back.
#[tokio::test]
async fn test_basic_scalar_types_roundtrip() {
    use arrow_array::{BooleanArray, Date32Array, TimestampMicrosecondArray};
    use arrow_schema::TimeUnit;

    let dir = tempdir().unwrap();

    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("flag", DataType::Boolean, true),
        Field::new("day", DataType::Date32, true),
        Field::new("ts", ts_type.clone(), true),
    ]));

    let ts_array =
        TimestampMicrosecondArray::from(vec![Some(1_000_000_i64), None, Some(3_000_000)])
            .with_timezone_opt(Some("UTC".to_string()));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            Arc::new(Date32Array::from(vec![
                Some(19_000_i32),
                None,
                Some(19_002),
            ])),
            Arc::new(ts_array),
        ],
    )
    .unwrap();

    let db_path = dir.path().join("basics.db");
    let mut builder = SqliteSidecarBuilder::begin(
        db_path.to_str().unwrap(),
        "models",
        2,
        schema,
        0,
        vec![1, 2, 3],
    )
    .unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    let batches = provider
        .fetch_by_keys(&[1, 3], "rowid", None)
        .await
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

    // The timestamp column must reconstruct with its original unit *and* time
    // zone, otherwise the column's DataType won't match the declared schema.
    for b in &batches {
        assert_eq!(b.column_by_name("ts").unwrap().data_type(), &ts_type);
        b.column_by_name("flag")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("flag should reconstruct as BooleanArray");
        b.column_by_name("day")
            .unwrap()
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("day should reconstruct as Date32Array");
    }

    // Spot-check rowid 1's values survived the round-trip.
    let rowid_one = batches
        .iter()
        .flat_map(|b| {
            let rid = b
                .column_by_name("rowid")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let flag = b
                .column_by_name("flag")
                .unwrap()
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            let day = b
                .column_by_name("day")
                .unwrap()
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap();
            (0..b.num_rows())
                .filter(|&i| rid.value(i) == 1)
                .map(|i| (flag.value(i), day.value(i)))
                .collect::<Vec<_>>()
        })
        .next()
        .expect("rowid 1 should be present");
    assert_eq!(rowid_one, (true, 19_000));
}

/// Round-trips the remaining "basic" scalar types added in the same pass:
/// the small integers (Int8/16, UInt8/16), time-of-day (Time32/Time64), and
/// binary (Binary/LargeBinary). Each column must reconstruct with its original
/// Arrow type and values, including nulls.
#[tokio::test]
async fn test_all_basic_types_roundtrip() {
    use arrow_array::{
        BinaryArray, Int8Array, Int16Array, LargeBinaryArray, Time32MillisecondArray,
        Time64NanosecondArray, UInt8Array, UInt16Array,
    };
    use arrow_schema::TimeUnit;

    let dir = tempdir().unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, true),
        Field::new("u8", DataType::UInt8, true),
        Field::new("u16", DataType::UInt16, true),
        Field::new("t32", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("t64", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("bin", DataType::Binary, true),
        Field::new("lbin", DataType::LargeBinary, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(Int8Array::from(vec![Some(-8_i8), None, Some(127)])),
            Arc::new(Int16Array::from(vec![Some(-16_i16), Some(300), None])),
            Arc::new(UInt8Array::from(vec![Some(8_u8), None, Some(255)])),
            Arc::new(UInt16Array::from(vec![Some(16_u16), Some(65535), None])),
            Arc::new(Time32MillisecondArray::from(vec![
                Some(1_000_i32),
                None,
                Some(86_399_000),
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                Some(1_i64),
                Some(2),
                None,
            ])),
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(b"\x00\x01".as_ref()),
                None,
                Some(b"\xff".as_ref()),
            ])),
            Arc::new(LargeBinaryArray::from_opt_vec(vec![
                Some(b"abc".as_ref()),
                Some(b"".as_ref()),
                None,
            ])),
        ],
    )
    .unwrap();

    let db_path = dir.path().join("allbasic.db");
    let mut builder = SqliteSidecarBuilder::begin(
        db_path.to_str().unwrap(),
        "models",
        2,
        schema,
        0,
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    )
    .unwrap();
    builder.push_batch(&batch).unwrap();
    let provider = builder.finish().unwrap();

    // Fetch rows 1 and 3 and collect them keyed by rowid so assertions don't
    // depend on row/batch ordering.
    let batches = provider
        .fetch_by_keys(&[1, 3], "rowid", None)
        .await
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

    // Every column must reconstruct with its declared Arrow type.
    for b in &batches {
        for f in b.schema().fields() {
            assert_eq!(
                b.column_by_name(f.name()).unwrap().data_type(),
                f.data_type(),
                "column {} reconstructed with wrong type",
                f.name()
            );
        }
    }

    // Pull row 1's values out and check them concretely.
    let (i8v, i16v, u8v, u16v, t32v, t64v, binv, lbinv) = batches
        .iter()
        .flat_map(|b| {
            let rid = b
                .column_by_name("rowid")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let i8a = b
                .column_by_name("i8")
                .unwrap()
                .as_any()
                .downcast_ref::<Int8Array>()
                .expect("i8 should reconstruct as Int8Array");
            let i16a = b
                .column_by_name("i16")
                .unwrap()
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap();
            let u8a = b
                .column_by_name("u8")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            let u16a = b
                .column_by_name("u16")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap();
            let t32a = b
                .column_by_name("t32")
                .unwrap()
                .as_any()
                .downcast_ref::<Time32MillisecondArray>()
                .expect("t32 should reconstruct as Time32MillisecondArray");
            let t64a = b
                .column_by_name("t64")
                .unwrap()
                .as_any()
                .downcast_ref::<Time64NanosecondArray>()
                .unwrap();
            let bina = b
                .column_by_name("bin")
                .unwrap()
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("bin should reconstruct as BinaryArray");
            let lbina = b
                .column_by_name("lbin")
                .unwrap()
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap();
            (0..b.num_rows())
                .filter(|&i| rid.value(i) == 1)
                .map(|i| {
                    (
                        i8a.value(i),
                        i16a.value(i),
                        u8a.value(i),
                        u16a.value(i),
                        t32a.value(i),
                        t64a.value(i),
                        bina.value(i).to_vec(),
                        lbina.value(i).to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .next()
        .expect("rowid 1 should be present");

    assert_eq!(i8v, -8);
    assert_eq!(i16v, -16);
    assert_eq!(u8v, 8);
    assert_eq!(u16v, 16);
    assert_eq!(t32v, 1_000);
    assert_eq!(t64v, 1);
    assert_eq!(binv, vec![0x00, 0x01]);
    assert_eq!(lbinv, b"abc".to_vec());

    // Null cells in row 3 must survive as nulls (row 3: i8=127, i16=null,
    // u16=null, t64=null, lbin=null).
    let row3 = provider.fetch_by_keys(&[3], "rowid", None).await.unwrap();
    let b = &row3[0];
    let i16a = b
        .column_by_name("i16")
        .unwrap()
        .as_any()
        .downcast_ref::<Int16Array>()
        .unwrap();
    let lbina = b
        .column_by_name("lbin")
        .unwrap()
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    assert!(i16a.is_null(0), "row 3 i16 should be null");
    assert!(lbina.is_null(0), "row 3 lbin should be null");
}

/// The output schema's key is always field 0, but `begin`'s `key_col_index`
/// indexes the *input batch* and may be non-zero (e.g. a storage engine whose
/// key is not the first batch column). Validation must still key off output
/// field 0 — otherwise a non-zero `key_col_index` would skip a real payload
/// column (leaving its unsupported type to fail at query time) and validate the
/// key column instead. This locks that in.
#[tokio::test]
async fn test_unsupported_payload_rejected_with_nonzero_key_index() {
    let dir = tempdir().unwrap();

    // Output schema: key at field 0, an unsupported Decimal payload at field 1.
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("price", DataType::Decimal128(10, 2), true),
    ]));

    // key_col_index = 1 (key is the second column of the input batch). A bug that
    // skipped output field `key_col_index` would skip "price" and wrongly pass.
    let db_path = dir.path().join("nonzero_key.db");
    let err =
        SqliteSidecarBuilder::begin(db_path.to_str().unwrap(), "models", 2, schema, 1, vec![0])
            .map(|_| ())
            .expect_err("unsupported payload must be rejected regardless of key_col_index");

    let msg = err.to_string();
    assert!(
        msg.contains("price") && msg.contains("unsupported type"),
        "error should name the payload column, got: {msg}"
    );
}

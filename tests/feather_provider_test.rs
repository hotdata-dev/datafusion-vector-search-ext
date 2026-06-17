#![cfg(feature = "feather-provider")]
//! Tests for `FeatherLookupProvider` / `FeatherSidecarBuilder` (ticket #702).
//!
//! Coverage:
//!  - **Parity** (`#[cfg(feature = "sqlite-provider")]`): a Rust port of the
//!    benchmark's `verify_same_data.py` — Feather returns the same rows as the
//!    SQLite provider for sparse/holey rowids, including edges and absent keys.
//!  - **Type fidelity**: Feather round-trips Decimal / FixedSizeList / Struct /
//!    Dictionary / nested-List losslessly — the types `SqliteSidecarBuilder`
//!    rejects at build time.
//!  - **Projection**, **build-order / sort-agnosticism**, and an informational
//!    **footprint + build-time** comparison.

use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, ListBuilder};
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, Decimal128Array, DictionaryArray, Int64Array, RecordBatch, StringArray,
    StructArray, TimestampMicrosecondArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_vector_search_ext::{
    FeatherLookupProvider, FeatherSidecarBuilder, PointLookupProvider,
};
use tempfile::tempdir;

// ── Synthetic sparse-rowid payload (DuckLake-shaped) ────────────────────────────

/// Deterministic PRNG (SplitMix64) so tests are reproducible without a `rand` dep.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Columns mirror the benchmark payload: rowid (key) + id/url/title/sha/raw/filename.
/// `title` is intermittently null to exercise null handling.
fn payload_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
        Field::new("url", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("sha", DataType::Utf8, true),
        Field::new("raw", DataType::Utf8, true),
        Field::new("filename", DataType::Utf8, true),
    ]))
}

/// Generate `n` rows with strictly-increasing **sparse** rowids (~60% holes:
/// gaps drawn from 1..=4), chunked into `chunk`-row batches. Returns the schema,
/// the batches (in ascending-key order), and the sorted rowid list.
fn gen_payload(n: usize, chunk: usize, seed: u64) -> (SchemaRef, Vec<RecordBatch>, Vec<i64>) {
    let schema = payload_schema();
    let mut rng = Rng(seed);

    let mut rowids: Vec<i64> = Vec::with_capacity(n);
    let mut cur: i64 = 5;
    for _ in 0..n {
        cur += 1 + rng.below(4) as i64; // gap 1..=4 → sparse, strictly increasing
        rowids.push(cur);
    }

    let mut batches = Vec::new();
    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        let rid: Vec<i64> = rowids[start..end].to_vec();
        let id: Vec<i64> = rid.iter().map(|r| r.wrapping_mul(7)).collect();
        let url: Vec<Option<String>> = rid
            .iter()
            .map(|r| Some(format!("https://example.com/doc/{r}")))
            .collect();
        let title: Vec<Option<String>> = rid
            .iter()
            .map(|r| (r % 7 != 0).then(|| format!("Title for row {r}")))
            .collect();
        let sha: Vec<Option<String>> = rid.iter().map(|r| Some(format!("{r:040x}"))).collect();
        let raw: Vec<Option<String>> = rid
            .iter()
            .map(|r| Some(format!("raw payload bytes for row {r}: {}", "x".repeat(40))))
            .collect();
        let filename: Vec<Option<String>> =
            rid.iter().map(|r| Some(format!("file_{r}.txt"))).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(rid)),
                Arc::new(Int64Array::from(id)),
                Arc::new(StringArray::from(url)),
                Arc::new(StringArray::from(title)),
                Arc::new(StringArray::from(sha)),
                Arc::new(StringArray::from(raw)),
                Arc::new(StringArray::from(filename)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }
    (schema, batches, rowids)
}

fn build_feather(
    dir: &tempfile::TempDir,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> FeatherLookupProvider {
    let path = dir.path().join("payload.feather");
    let value_cols: Vec<usize> = (1..schema.fields().len()).collect();
    let mut builder =
        FeatherSidecarBuilder::begin(path.to_str().unwrap(), schema, 0, value_cols).unwrap();
    for b in batches {
        builder.push_batch(b).unwrap();
    }
    builder.finish().unwrap()
}

fn fmt(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches).unwrap().to_string()
}

// ── A. Provider correctness & parity ───────────────────────────────────────────

#[cfg(feature = "sqlite-provider")]
fn build_sqlite(
    dir: &tempfile::TempDir,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> datafusion_vector_search_ext::SqliteLookupProvider {
    use datafusion_vector_search_ext::SqliteSidecarBuilder;
    let path = dir.path().join("payload.db");
    let value_cols: Vec<usize> = (1..schema.fields().len()).collect();
    let mut builder =
        SqliteSidecarBuilder::begin(path.to_str().unwrap(), "payload", 4, schema, 0, value_cols)
            .unwrap();
    for b in batches {
        builder.push_batch(b).unwrap();
    }
    builder.finish().unwrap()
}

#[cfg(feature = "sqlite-provider")]
#[tokio::test]
async fn parity_with_sqlite_sparse_rowids() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(3000, 256, 0xC0FFEE);

    let feather = build_feather(&dir, schema.clone(), &batches);
    let sqlite = build_sqlite(&dir, schema.clone(), &batches);

    let n = rowids.len();
    let mut rng = Rng(42);
    let pick = |rng: &mut Rng, count: usize| -> Vec<u64> {
        (0..count)
            .map(|_| rowids[rng.below(n as u64) as usize] as u64)
            .collect()
    };

    // Build a battery of key sets covering scattered, edges, duplicates, absent.
    let min = rowids[0] as u64;
    let max = *rowids.last().unwrap() as u64;
    let mid = rowids[n / 2] as u64;

    let mut key_sets: Vec<(&str, Vec<u64>)> = vec![
        ("empty", vec![]),
        ("smallest", vec![min]),
        ("largest", vec![max]),
        ("mid", vec![mid]),
        ("edges_together", vec![min, mid, max]),
        ("duplicates", vec![mid, mid, min, min, max]),
        // Absent: below min, above max, and a value in a gap between two rowids.
        ("absent_below", vec![0]),
        ("absent_above", vec![max + 1000]),
        ("absent_in_gap", vec![rowids[10] as u64 + 1]),
        ("mixed_present_absent", vec![min, 0, mid, max + 1000, max]),
    ];
    for k in [1usize, 10, 100, 1000] {
        key_sets.push(("scattered", pick(&mut rng, k)));
    }

    for (label, keys) in &key_sets {
        let f = feather.fetch_by_keys(keys, "rowid", None).await.unwrap();
        let s = sqlite.fetch_by_keys(keys, "rowid", None).await.unwrap();
        assert_eq!(
            fmt(&f),
            fmt(&s),
            "feather vs sqlite mismatch for key set '{label}' ({} keys)",
            keys.len()
        );
    }
}

#[cfg(feature = "sqlite-provider")]
#[tokio::test]
async fn parity_with_sqlite_projection() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(1500, 512, 7);
    let feather = build_feather(&dir, schema.clone(), &batches);
    let sqlite = build_sqlite(&dir, schema.clone(), &batches);

    let keys: Vec<u64> = rowids.iter().step_by(13).map(|&r| r as u64).collect();

    // narrow (rowid, title, filename) and full + a reordered projection.
    for proj in [
        vec![0usize, 3, 6],
        (0..schema.fields().len()).collect::<Vec<_>>(),
        vec![3, 0, 6, 1],
    ] {
        let f = feather
            .fetch_by_keys(&keys, "rowid", Some(&proj))
            .await
            .unwrap();
        let s = sqlite
            .fetch_by_keys(&keys, "rowid", Some(&proj))
            .await
            .unwrap();
        assert_eq!(fmt(&f), fmt(&s), "projection {proj:?} mismatch");
        // Only the projected columns are present, in projection order.
        assert_eq!(f[0].num_columns(), proj.len());
        for (out_i, &src_i) in proj.iter().enumerate() {
            assert_eq!(
                f[0].schema().field(out_i).name(),
                schema.field(src_i).name()
            );
        }
    }
}

/// The exact-match guard: an absent key (one that falls between two stored
/// rowids, or below/above the range) must never alias onto a neighbour. After
/// hydration the returned rowid column equals exactly the present, sorted,
/// deduped requested keys.
#[tokio::test]
async fn binary_search_lands_exactly() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(2000, 256, 99);
    let feather = build_feather(&dir, schema, &batches);

    let present: Vec<u64> = rowids.iter().step_by(7).map(|&r| r as u64).collect();
    let absent: Vec<u64> = rowids
        .iter()
        .step_by(7)
        .map(|&r| r as u64 + 1) // gaps are ≥1, so +1 may or may not exist; filter below
        .filter(|k| rowids.binary_search(&(*k as i64)).is_err())
        .collect();

    let mut request = present.clone();
    request.extend(&absent);
    request.push(0);
    request.push(*rowids.last().unwrap() as u64 + 9999);

    let out = feather
        .fetch_by_keys(&request, "rowid", None)
        .await
        .unwrap();
    let got: Vec<i64> = out[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();

    let mut expected: Vec<i64> = present.iter().map(|&k| k as i64).collect();
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(got, expected, "binary search aliased an absent key");
}

#[tokio::test]
async fn empty_and_all_absent_return_no_batches() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(500, 128, 3);
    let feather = build_feather(&dir, schema, &batches);

    assert!(
        feather
            .fetch_by_keys(&[], "rowid", None)
            .await
            .unwrap()
            .is_empty()
    );
    let absent = vec![0u64, 1, *rowids.last().unwrap() as u64 + 1_000_000];
    assert!(
        feather
            .fetch_by_keys(&absent, "rowid", None)
            .await
            .unwrap()
            .is_empty()
    );
}

// ── B. Type fidelity ────────────────────────────────────────────────────────────

/// Build a payload with types `SqliteSidecarBuilder` rejects, and confirm Feather
/// round-trips them losslessly — same Arrow types out, same values.
#[tokio::test]
async fn type_fidelity_roundtrip() {
    let dir = tempdir().unwrap();

    let struct_fields = Fields::from(vec![
        Field::new("lat", DataType::Float64, false),
        Field::new("lon", DataType::Float64, false),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("price", DataType::Decimal128(20, 4), false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into())),
            false,
        ),
        Field::new(
            "embedding3",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 3),
            false,
        ),
        Field::new("loc", DataType::Struct(struct_fields.clone()), false),
        Field::new(
            "tag",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new(
            "scores",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
    ]));

    let rowid = Int64Array::from(vec![10_i64, 20, 30]);
    let price = Decimal128Array::from(vec![12345_i128, -98765, 0])
        .with_precision_and_scale(20, 4)
        .unwrap();
    let ts = TimestampMicrosecondArray::from(vec![1_000_000_i64, 2_000_000, 3_000_000])
        .with_timezone("America/New_York");

    let mut fsl = FixedSizeListBuilder::new(Float64Builder::new(), 3);
    for triple in [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] {
        for v in triple {
            fsl.values().append_value(v);
        }
        fsl.append(true);
    }
    let embedding3 = fsl.finish();

    let loc = StructArray::from(vec![
        (
            Arc::new(Field::new("lat", DataType::Float64, false)),
            Arc::new(arrow_array::Float64Array::from(vec![40.0, 41.0, 42.0])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("lon", DataType::Float64, false)),
            Arc::new(arrow_array::Float64Array::from(vec![-74.0, -75.0, -76.0])) as ArrayRef,
        ),
    ]);

    let tag: DictionaryArray<Int32Type> = vec![Some("nlp"), Some("vision"), Some("nlp")]
        .into_iter()
        .collect();

    let mut scores = ListBuilder::new(Float64Builder::new());
    scores.values().append_value(0.1);
    scores.values().append_value(0.2);
    scores.append(true);
    scores.values().append_value(0.3);
    scores.append(true);
    scores.append(false); // null list
    let scores = scores.finish();

    let input = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(rowid),
            Arc::new(price),
            Arc::new(ts),
            Arc::new(embedding3),
            Arc::new(loc),
            Arc::new(tag),
            Arc::new(scores),
        ],
    )
    .unwrap();

    let feather = build_feather(&dir, schema.clone(), std::slice::from_ref(&input));
    let out = feather
        .fetch_by_keys(&[10, 20, 30], "rowid", None)
        .await
        .unwrap();

    // Types preserved exactly (not coerced to i64/JSON/TEXT).
    for i in 0..schema.fields().len() {
        assert_eq!(
            out[0].schema().field(i).data_type(),
            schema.field(i).data_type(),
            "type of column '{}' was not preserved",
            schema.field(i).name()
        );
    }
    // Values preserved exactly (input is already key-sorted, so rows align).
    assert_eq!(fmt(&out), fmt(std::slice::from_ref(&input)));
}

/// The contrast: `SqliteSidecarBuilder::begin` rejects each of these payload
/// types at build time (validated against its supported set), whereas Feather
/// accepts them (proven above).
#[cfg(feature = "sqlite-provider")]
#[test]
fn sqlite_rejects_types_feather_accepts() {
    use datafusion_vector_search_ext::SqliteSidecarBuilder;
    let dir = tempdir().unwrap();

    let exotic: Vec<(&str, DataType)> = vec![
        ("decimal", DataType::Decimal128(20, 4)),
        (
            "fixed_size_list",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 3),
        ),
        (
            "struct",
            DataType::Struct(Fields::from(vec![Field::new(
                "a",
                DataType::Float64,
                false,
            )])),
        ),
        (
            "dictionary",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        ),
        (
            "list_of_float",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        ),
    ];

    for (name, dt) in exotic {
        let schema = Arc::new(Schema::new(vec![
            Field::new("rowid", DataType::Int64, false),
            Field::new(name, dt, true),
        ]));
        let db = dir.path().join(format!("{name}.db"));
        let res =
            SqliteSidecarBuilder::begin(db.to_str().unwrap(), "t", 1, schema.clone(), 0, vec![1]);
        assert!(
            res.is_err(),
            "expected SQLite to reject payload type for column '{name}'"
        );

        // Feather accepts the same schema at begin.
        let feather_path = dir.path().join(format!("{name}.feather"));
        assert!(
            FeatherSidecarBuilder::begin(feather_path.to_str().unwrap(), schema, 0, vec![1])
                .is_ok(),
            "expected Feather to accept payload type for column '{name}'"
        );
    }
}

// ── D/E. Build path: sort-agnosticism & ordering ────────────────────────────────

/// The builder must produce a sorted-by-rowid file regardless of input order,
/// and report whether the input was already sorted.
#[tokio::test]
async fn builder_is_sort_agnostic() {
    let dir = tempdir().unwrap();
    let (schema, sorted_batches, rowids) = gen_payload(1000, 100, 5);

    // Sorted input → reported sorted; output sorted.
    let path_sorted = dir.path().join("sorted.feather");
    let value_cols: Vec<usize> = (1..schema.fields().len()).collect();
    let mut b = FeatherSidecarBuilder::begin(
        path_sorted.to_str().unwrap(),
        schema.clone(),
        0,
        value_cols.clone(),
    )
    .unwrap();
    for batch in &sorted_batches {
        b.push_batch(batch).unwrap();
    }
    assert!(b.input_was_sorted(), "ascending input should report sorted");
    let p_sorted = b.finish().unwrap();

    // Shuffled input (reversed batch order + reversed rows) → reported unsorted,
    // but output is still correctly sorted and identical to the sorted build.
    let path_shuf = dir.path().join("shuffled.feather");
    let mut b2 =
        FeatherSidecarBuilder::begin(path_shuf.to_str().unwrap(), schema.clone(), 0, value_cols)
            .unwrap();
    for batch in sorted_batches.iter().rev() {
        b2.push_batch(&reverse_rows(batch)).unwrap();
    }
    assert!(
        !b2.input_was_sorted(),
        "shuffled input should report unsorted"
    );
    let p_shuf = b2.finish().unwrap();

    let all: Vec<u64> = rowids.iter().map(|&r| r as u64).collect();
    let from_sorted = p_sorted.fetch_by_keys(&all, "rowid", None).await.unwrap();
    let from_shuf = p_shuf.fetch_by_keys(&all, "rowid", None).await.unwrap();
    assert_eq!(
        fmt(&from_sorted),
        fmt(&from_shuf),
        "sort-agnostic build produced different output for shuffled input"
    );

    // And the keys really are ascending.
    let keys: Vec<i64> = from_shuf[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert!(
        keys.windows(2).all(|w| w[0] < w[1]),
        "output not sorted ascending"
    );
}

/// Reverse the row order of a batch (helper for the shuffle test).
fn reverse_rows(batch: &RecordBatch) -> RecordBatch {
    let n = batch.num_rows();
    let idx = arrow_array::UInt32Array::from((0..n).rev().map(|i| i as u32).collect::<Vec<_>>());
    let cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| datafusion::arrow::compute::take(c.as_ref(), &idx, None).unwrap())
        .collect();
    RecordBatch::try_new(batch.schema(), cols).unwrap()
}

#[tokio::test]
async fn reopen_from_file_roundtrips() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(800, 200, 11);
    let path = dir.path().join("reopen.feather");
    let value_cols: Vec<usize> = (1..schema.fields().len()).collect();
    let mut b =
        FeatherSidecarBuilder::begin(path.to_str().unwrap(), schema, 0, value_cols).unwrap();
    for batch in &batches {
        b.push_batch(batch).unwrap();
    }
    let from_builder = b.finish().unwrap();

    // Open a fresh provider straight off the published file (the replica path).
    let reopened = FeatherLookupProvider::open(path.to_str().unwrap()).unwrap();
    assert_eq!(reopened.len(), rowids.len());

    let keys: Vec<u64> = rowids.iter().step_by(5).map(|&r| r as u64).collect();
    let a = from_builder
        .fetch_by_keys(&keys, "rowid", None)
        .await
        .unwrap();
    let c = reopened.fetch_by_keys(&keys, "rowid", None).await.unwrap();
    assert_eq!(fmt(&a), fmt(&c));
}

// ── E/F. Footprint & build-time (informational) ─────────────────────────────────

#[cfg(feature = "sqlite-provider")]
#[tokio::test]
async fn footprint_and_build_time_vs_sqlite() {
    use std::time::Instant;
    let dir = tempdir().unwrap();
    let (schema, batches, _rowids) = gen_payload(50_000, 2048, 1234);

    let t0 = Instant::now();
    let _f = build_feather(&dir, schema.clone(), &batches);
    let feather_build = t0.elapsed();

    let t1 = Instant::now();
    let _s = build_sqlite(&dir, schema.clone(), &batches);
    let sqlite_build = t1.elapsed();

    let feather_bytes = std::fs::metadata(dir.path().join("payload.feather"))
        .unwrap()
        .len();
    let sqlite_bytes = std::fs::metadata(dir.path().join("payload.db"))
        .unwrap()
        .len();

    println!(
        "[#702 footprint] rows=50000  feather={feather_bytes} bytes ({:.1} MiB)  \
         sqlite={sqlite_bytes} bytes ({:.1} MiB)  ratio(feather/sqlite)={:.2}",
        feather_bytes as f64 / 1048576.0,
        sqlite_bytes as f64 / 1048576.0,
        feather_bytes as f64 / sqlite_bytes as f64,
    );
    println!(
        "[#702 build-time] feather={feather_build:?}  sqlite={sqlite_build:?}  \
         speedup(sqlite/feather)={:.1}x",
        sqlite_build.as_secs_f64() / feather_build.as_secs_f64().max(1e-9),
    );

    // Sanity only (not a perf gate): both files exist and are non-trivial.
    assert!(feather_bytes > 0 && sqlite_bytes > 0);
}

// ── open() error paths & robustness ─────────────────────────────────────────────

/// Write a batch to a `.feather` file directly (bypassing the builder's sort), so
/// we can construct deliberately-malformed sidecars for the rejection tests.
fn write_raw_feather(path: &str, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut w = arrow_ipc::writer::FileWriter::try_new(file, &batch.schema()).unwrap();
    w.write(batch).unwrap();
    w.finish().unwrap();
}

/// `open` must fail loudly (not silently return wrong rows) when the stored key
/// column is not strictly ascending — both the unsorted and duplicate-key cases.
/// This guards the "fail at open, not at query time" contract the binary search
/// depends on.
#[test]
fn open_rejects_unsorted_or_duplicate_keys() {
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let unsorted = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![10_i64, 3, 1])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
        ],
    )
    .unwrap();
    let p1 = dir.path().join("unsorted.feather");
    write_raw_feather(p1.to_str().unwrap(), &unsorted);
    assert!(
        FeatherLookupProvider::open(p1.to_str().unwrap()).is_err(),
        "descending key column must be rejected"
    );

    // Duplicate keys must also be rejected (mirrors SQLite's INTEGER PRIMARY KEY).
    let dup = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 1, 2])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
        ],
    )
    .unwrap();
    let p2 = dir.path().join("dup.feather");
    write_raw_feather(p2.to_str().unwrap(), &dup);
    assert!(
        FeatherLookupProvider::open(p2.to_str().unwrap()).is_err(),
        "duplicate keys must be rejected"
    );
}

/// `open` returns an error (never panics) on a non-Arrow file and on a missing
/// file, rather than surfacing a corrupt provider.
#[test]
fn open_rejects_malformed_or_missing_file() {
    let dir = tempdir().unwrap();
    let garbage = dir.path().join("garbage.feather");
    std::fs::write(&garbage, b"this is not an arrow ipc file").unwrap();
    assert!(FeatherLookupProvider::open(garbage.to_str().unwrap()).is_err());

    let missing = dir.path().join("does_not_exist.feather");
    assert!(FeatherLookupProvider::open(missing.to_str().unwrap()).is_err());
}

/// Regression for the build-sort domain fix: a genuinely-ascending Int64 key
/// column containing negative values must BUILD (not be rejected as "corrupt")
/// and round-trip via the u64 lookup domain the engine uses for rowids. Before
/// the fix, `finish()` sorted by signed Arrow order while the verify/search used
/// u64 order, so this input was spuriously rejected.
#[tokio::test]
async fn signed_negative_keys_build_and_query() {
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![-5_i64, -1, 3, 10])),
            Arc::new(StringArray::from(vec![
                Some("neg5"),
                Some("neg1"),
                Some("three"),
                Some("ten"),
            ])),
        ],
    )
    .unwrap();
    let path = dir.path().join("neg.feather");
    let mut b = FeatherSidecarBuilder::begin(path.to_str().unwrap(), schema, 0, vec![1]).unwrap();
    b.push_batch(&batch).unwrap();
    let provider = b
        .finish()
        .expect("negative-but-ascending Int64 keys must build");

    // The engine passes rowids as u64; a negative i64 arrives as its u64 reinterpretation.
    let out = provider
        .fetch_by_keys(&[(-5_i64) as u64, 3], "rowid", None)
        .await
        .unwrap();
    let names: Vec<String> = out
        .iter()
        .flat_map(|bch| {
            bch.column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
        })
        .collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"neg5".to_string()) && names.contains(&"three".to_string()));
}

/// UInt64 key column (the other common rowid type) round-trips. Only Int64 was
/// covered before; this exercises the UInt64 arm of `extract_keys_as_u64`.
#[tokio::test]
async fn uint64_keys_roundtrip() {
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![5_u64, 50, 500, 5000])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
                Some("d"),
            ])),
        ],
    )
    .unwrap();
    let path = dir.path().join("u64.feather");
    let mut b = FeatherSidecarBuilder::begin(path.to_str().unwrap(), schema, 0, vec![1]).unwrap();
    b.push_batch(&batch).unwrap();
    let provider = b.finish().unwrap();

    let out = provider
        .fetch_by_keys(&[50, 5000], "rowid", None)
        .await
        .unwrap();
    let names: Vec<String> = out[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(names, vec!["b".to_string(), "d".to_string()]);
}

/// The `TableProvider::scan` path (used for SQL column-name resolution) actually
/// returns rows — a working MemTable scan, unlike the parquet sibling which is
/// NotImplemented. Guards against the silent zero-row scan that bit parquet.
#[tokio::test]
async fn table_provider_scan_roundtrips() {
    use datafusion::catalog::TableProvider;
    use datafusion::prelude::SessionContext;

    let dir = tempdir().unwrap();
    let (schema, batches, _rowids) = gen_payload(300, 128, 21);
    let provider = build_feather(&dir, schema, &batches);
    let n = provider.len();

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider) as Arc<dyn TableProvider>)
        .unwrap();

    let all = ctx
        .sql("SELECT rowid, title FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = all.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, n, "full scan should see every row");
    assert_eq!(
        all[0].num_columns(),
        2,
        "projected scan returns only 2 columns"
    );
}

// ── Multi-batch (spill-sort + coarse-index) ─────────────────────────────────────

/// Build > BUILD_CHUNK_ROWS rows from fully-unsorted input so the spill-sort runs
/// and the final file is multi-batch, then check fetch parity vs SQLite — with
/// keys straddling the internal batch boundary to exercise the coarse index +
/// cross-batch gather.
#[cfg(feature = "sqlite-provider")]
#[tokio::test]
async fn parity_multibatch_shuffled_build() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(20_000, 1024, 0xABCD);

    // Feather: push batches reversed + each batch's rows reversed → spill is fully
    // unsorted, forcing the external sort to do real work.
    let fpath = dir.path().join("mb.feather");
    let value_cols: Vec<usize> = (1..schema.fields().len()).collect();
    let mut fb =
        FeatherSidecarBuilder::begin(fpath.to_str().unwrap(), schema.clone(), 0, value_cols)
            .unwrap();
    for b in batches.iter().rev() {
        fb.push_batch(&reverse_rows(b)).unwrap();
    }
    assert!(
        !fb.input_was_sorted(),
        "reversed input should report unsorted"
    );
    let feather = fb.finish().unwrap();
    assert_eq!(feather.len(), rowids.len());

    // SQLite oracle (order-independent B-tree).
    let sqlite = build_sqlite(&dir, schema.clone(), &batches);

    let n = rowids.len();
    let mut rng = Rng(7);
    let mut keys: Vec<u64> = (0..1500)
        .map(|_| rowids[rng.below(n as u64) as usize] as u64)
        .collect();
    // Straddle the first internal batch boundary (sorted chunk size 8192).
    for &i in &[0usize, 8191, 8192, 8193, n - 1] {
        keys.push(rowids[i] as u64);
    }

    let f = feather.fetch_by_keys(&keys, "rowid", None).await.unwrap();
    let s = sqlite.fetch_by_keys(&keys, "rowid", None).await.unwrap();
    assert_eq!(
        fmt(&f),
        fmt(&s),
        "multi-batch shuffled-build parity mismatch"
    );

    // Projected fetch across the boundary too.
    let proj = vec![0usize, 3, 6];
    let f = feather
        .fetch_by_keys(&keys, "rowid", Some(&proj))
        .await
        .unwrap();
    let s = sqlite
        .fetch_by_keys(&keys, "rowid", Some(&proj))
        .await
        .unwrap();
    assert_eq!(fmt(&f), fmt(&s), "multi-batch projected parity mismatch");
}

// ── Regression + edge cases for the mmap + spill-sort rewrite ────────────────────

/// `open` must return an error (not panic) on a file whose IPC trailer claims a
/// footer larger than the file — the corrupt-footer guard.
#[test]
fn open_rejects_corrupt_footer_length() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("badfooter.feather");
    // 32 bytes: padding, then a valid 10-byte trailer [footer_len i32 LE][b"ARROW1"]
    // with an oversized footer_len.
    let mut bytes = vec![0u8; 22];
    bytes.extend_from_slice(&1_000_000_i32.to_le_bytes());
    bytes.extend_from_slice(b"ARROW1");
    std::fs::write(&p, &bytes).unwrap();
    assert!(
        FeatherLookupProvider::open(p.to_str().unwrap()).is_err(),
        "oversized footer_len must error, not panic"
    );
}

/// A build with zero pushed rows yields a valid, empty, re-openable sidecar.
#[tokio::test]
async fn empty_build_is_valid() {
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let p = dir.path().join("empty.feather");
    let builder = FeatherSidecarBuilder::begin(p.to_str().unwrap(), schema, 0, vec![1]).unwrap();
    let provider = builder.finish().unwrap();
    assert!(provider.is_empty());
    assert!(
        provider
            .fetch_by_keys(&[1, 2, 3], "rowid", None)
            .await
            .unwrap()
            .is_empty()
    );
    let reopened = FeatherLookupProvider::open(p.to_str().unwrap()).unwrap();
    assert_eq!(reopened.len(), 0);
}

/// Single-row build (smallest non-empty case).
#[tokio::test]
async fn single_row_build() {
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![42_i64])),
            Arc::new(StringArray::from(vec![Some("answer")])),
        ],
    )
    .unwrap();
    let p = dir.path().join("one.feather");
    let mut b = FeatherSidecarBuilder::begin(p.to_str().unwrap(), schema, 0, vec![1]).unwrap();
    b.push_batch(&batch).unwrap();
    let provider = b.finish().unwrap();
    assert_eq!(provider.len(), 1);
    let out = provider.fetch_by_keys(&[42], "rowid", None).await.unwrap();
    assert_eq!(out[0].num_rows(), 1);
    assert!(
        provider
            .fetch_by_keys(&[7], "rowid", None)
            .await
            .unwrap()
            .is_empty()
    );
}

/// Build exactly 2× BUILD_CHUNK_ROWS (16384) rows so the gather hits an exact
/// chunk boundary, and check parity vs SQLite for keys straddling it.
#[cfg(feature = "sqlite-provider")]
#[tokio::test]
async fn build_at_exact_chunk_boundary() {
    let dir = tempdir().unwrap();
    let (schema, batches, rowids) = gen_payload(16_384, 4096, 0xBEEF);
    let feather = build_feather(&dir, schema.clone(), &batches);
    let sqlite = build_sqlite(&dir, schema.clone(), &batches);
    assert_eq!(feather.len(), 16_384);

    let mut keys: Vec<u64> = [0usize, 8191, 8192, 8193, 16_383]
        .iter()
        .map(|&i| rowids[i] as u64)
        .collect();
    let mut rng = Rng(1);
    for _ in 0..500 {
        keys.push(rowids[rng.below(16_384) as usize] as u64);
    }
    let f = feather.fetch_by_keys(&keys, "rowid", None).await.unwrap();
    let s = sqlite.fetch_by_keys(&keys, "rowid", None).await.unwrap();
    assert_eq!(fmt(&f), fmt(&s), "chunk-boundary parity mismatch");
}

/// Duplicate keys split across separate push_batch calls must be rejected at
/// finish (the dup check runs over the global sorted order, not per-batch).
#[test]
fn duplicate_keys_across_pushes_rejected() {
    let dir = tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mk = |rid: Vec<i64>, names: Vec<&'static str>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(rid)),
                Arc::new(StringArray::from(
                    names.into_iter().map(Some).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };
    let p = dir.path().join("dup.feather");
    let mut b =
        FeatherSidecarBuilder::begin(p.to_str().unwrap(), schema.clone(), 0, vec![1]).unwrap();
    b.push_batch(&mk(vec![1, 2], vec!["a", "b"])).unwrap();
    b.push_batch(&mk(vec![2, 3], vec!["c", "d"])).unwrap(); // key 2 duplicated across pushes
    assert!(
        b.finish().is_err(),
        "duplicate key across pushes must be rejected"
    );
}

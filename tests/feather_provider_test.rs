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
    StructArray, TimestampMicrosecondArray,
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

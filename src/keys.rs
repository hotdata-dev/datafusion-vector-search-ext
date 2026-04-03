// keys.rs — Key encoding utilities for packed row addresses.
//
// Bit layout:
//   bits 63-48  (16 bits)  file_idx       — which parquet file (up to 65,536)
//   bits 47-32  (16 bits)  rg_idx         — which row group  (up to 65,536)
//   bits  31-0  (32 bits)  local_offset   — row offset within row group (up to 4 B)

/// Pack a physical row address into a single `u64` key for use with USearch
/// and the lookup providers.
///
/// # Panics
/// Panics if any component exceeds its allocated bit range:
/// `file_idx` < 65 536, `rg_idx` < 65 536, `local_offset` ≤ 4 294 967 295.
#[inline]
pub fn pack_key(file_idx: usize, rg_idx: usize, local_offset: usize) -> u64 {
    assert!(
        file_idx < (1 << 16),
        "file_idx {file_idx} overflows 16 bits"
    );
    assert!(rg_idx < (1 << 16), "rg_idx {rg_idx} overflows 16 bits");
    assert!(
        local_offset <= u32::MAX as usize,
        "local_offset {local_offset} overflows 32 bits"
    );
    ((file_idx as u64) << 48) | ((rg_idx as u64) << 32) | (local_offset as u64)
}

/// Unpack a `u64` key back to `(file_idx, rg_idx, local_offset)`.
#[inline]
pub fn unpack_key(key: u64) -> (usize, usize, usize) {
    let file_idx = (key >> 48) as usize;
    let rg_idx = ((key >> 32) & 0xFFFF) as usize;
    let local_offset = (key & 0xFFFF_FFFF) as usize;
    (file_idx, rg_idx, local_offset)
}

/// Physical layout of a sharded parquet dataset.
///
/// Computed once at startup from parquet file footers (reads only the last few
/// KB of each file). Not persisted to disk.
#[derive(Debug)]
pub struct DatasetLayout {
    /// Object-store path (or S3 key) for each file, indexed by `file_idx`.
    pub file_keys: Vec<String>,
    /// Cumulative row counts at the start of each file: `file_cum_rows[i]` is
    /// the total number of rows in files 0..i. `file_cum_rows[n_files]` is the
    /// total row count of the dataset.
    pub file_cum_rows: Vec<u64>,
    /// For each file, cumulative row count at the start of each row group:
    /// `rg_cum_rows[file][rg]` = rows in row groups 0..rg within that file.
    pub rg_cum_rows: Vec<Vec<u64>>,
}

impl DatasetLayout {
    /// Convert a packed usearch key back to a global (dataset-wide) row index.
    #[inline]
    pub fn packed_key_to_global(&self, key: u64) -> u64 {
        let (file_idx, rg_idx, local_offset) = unpack_key(key);
        self.file_cum_rows[file_idx] + self.rg_cum_rows[file_idx][rg_idx] + local_offset as u64
    }

    /// Scan parquet footers to build the layout. No vector data is read.
    ///
    /// `key_prefix` is prepended to each bare filename when storing the
    /// object-store path in `file_keys`. It must match the prefix used when
    /// constructing the `ObjectStore` passed to `ParquetLookupProvider`, so
    /// that `store.get("{key_prefix}{filename}")` resolves to the correct
    /// object at query time.
    ///
    /// # Examples
    /// ```text
    /// // Files at /data/shard_00.parquet, store rooted at /data
    /// DatasetLayout::from_files(&["/data/shard_00.parquet"], "")
    ///
    /// // Files under a parquet/ subdirectory, store rooted at /data
    /// DatasetLayout::from_files(&["/data/parquet/shard_00.parquet"], "parquet/")
    ///
    /// // Local footer reads, but keys point at S3 prefix
    /// DatasetLayout::from_files(&["/local/cache/shard_00.parquet"], "year=2024/")
    /// ```
    ///
    /// Only compiled when the `parquet-provider` or `sqlite-provider` feature is enabled.
    #[cfg(any(feature = "parquet-provider", feature = "sqlite-provider"))]
    pub fn from_files(local_paths: &[&str], key_prefix: &str) -> datafusion::common::Result<Self> {
        use datafusion::error::DataFusionError;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs;
        use std::path::Path;

        let mut file_keys = Vec::with_capacity(local_paths.len());
        let mut file_cum_rows = vec![0u64];
        let mut rg_cum_rows: Vec<Vec<u64>> = Vec::with_capacity(local_paths.len());

        let mut running_total = 0u64;
        for &path in local_paths {
            let raw = Path::new(path);

            // Reject paths containing '..' components to prevent directory
            // traversal when paths originate from untrusted input.
            for component in raw.components() {
                if matches!(component, std::path::Component::ParentDir) {
                    return Err(DataFusionError::Execution(format!(
                        "path traversal not allowed: {path}"
                    )));
                }
            }

            let file_name = raw
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| DataFusionError::Execution(format!("invalid path: {path}")))?;
            file_keys.push(format!("{key_prefix}{file_name}"));

            let f = fs::File::open(path)
                .map_err(|e| DataFusionError::Execution(format!("open {path}: {e}")))?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(f)
                .map_err(|e| DataFusionError::Execution(format!("read footer {path}: {e}")))?;
            let meta = builder.metadata();

            let mut rg_cum = vec![0u64];
            let mut file_rows = 0u64;
            for rg in 0..meta.num_row_groups() {
                let n = meta.row_group(rg).num_rows() as u64;
                file_rows += n;
                rg_cum.push(file_rows);
            }
            rg_cum_rows.push(rg_cum);
            running_total += file_rows;
            file_cum_rows.push(running_total);
        }

        Ok(Self {
            file_keys,
            file_cum_rows,
            rg_cum_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let cases = [
            (0, 0, 0),
            (1, 2, 3),
            (65535, 65535, u32::MAX as usize),
            (0, 0, u32::MAX as usize),
        ];
        for (fi, rg, lo) in cases {
            let key = pack_key(fi, rg, lo);
            assert_eq!(unpack_key(key), (fi, rg, lo));
        }
    }

    #[test]
    fn test_pack_key_boundary_values() {
        // u32::MAX as local_offset is within range and must round-trip cleanly.
        let key = pack_key(0, 0, u32::MAX as usize);
        let (_, _, lo) = unpack_key(key);
        assert_eq!(lo, u32::MAX as usize);
    }

    #[test]
    #[should_panic(expected = "overflows 32 bits")]
    fn test_pack_key_local_offset_overflow_panics() {
        pack_key(0, 0, u32::MAX as usize + 1);
    }

    #[test]
    #[should_panic(expected = "overflows 16 bits")]
    fn test_pack_key_file_idx_overflow_panics() {
        pack_key(1 << 16, 0, 0);
    }

    #[cfg(any(feature = "parquet-provider", feature = "sqlite-provider"))]
    mod from_files_tests {
        use super::super::DatasetLayout;
        use arrow_array::{RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        /// Write a minimal 1-row parquet file at the given path.
        fn write_parquet(path: &std::path::Path) {
            let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(vec![Some("a")]))],
            )
            .unwrap();
            let file = std::fs::File::create(path).unwrap();
            let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }

        #[test]
        fn rejects_parent_dir_traversal() {
            let dir = tempfile::tempdir().unwrap();
            let sub = dir.path().join("sub");
            std::fs::create_dir(&sub).unwrap();
            let file = sub.join("test.parquet");
            write_parquet(&file);

            // A path using '..' to escape the expected directory must be rejected.
            let traversal = format!("{}/sub/../sub/test.parquet", dir.path().display());
            let result = DatasetLayout::from_files(&[&traversal], "");
            assert!(result.is_err());
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("path traversal"),
                "expected 'path traversal' error, got: {msg}"
            );
        }

        #[test]
        fn rejects_dotdot_prefix_traversal() {
            // Even relative paths with '..' must be rejected.
            let result = DatasetLayout::from_files(&["../../../etc/passwd"], "");
            assert!(result.is_err());
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("path traversal"),
                "expected 'path traversal' error, got: {msg}"
            );
        }

        #[test]
        fn accepts_valid_absolute_path() {
            let dir = tempfile::tempdir().unwrap();
            let file = dir.path().join("valid.parquet");
            write_parquet(&file);

            let path_str = file.to_str().unwrap();
            let layout = DatasetLayout::from_files(&[path_str], "").unwrap();
            assert_eq!(layout.file_keys, vec!["valid.parquet"]);
            assert_eq!(layout.file_cum_rows, vec![0, 1]);
        }
    }
}

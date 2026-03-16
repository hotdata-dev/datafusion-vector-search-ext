// keys.rs — Key encoding utilities for packed row addresses.
//
// Bit layout:
//   bits 63-48  (16 bits)  file_idx       — which parquet file (up to 65,536)
//   bits 47-32  (16 bits)  rg_idx         — which row group  (up to 65,536)
//   bits  31-0  (32 bits)  local_offset   — row offset within row group (up to 4 B)

/// Pack a physical row address into a single `u64` key for use with USearch
/// and the lookup providers.
///
/// # Panics (debug builds only)
/// Asserts that each component fits within its allocated bit range:
/// `file_idx` < 65 536, `rg_idx` < 65 536, `local_offset` < 4 294 967 296.
#[inline]
pub fn pack_key(file_idx: usize, rg_idx: usize, local_offset: usize) -> u64 {
    debug_assert!(
        file_idx < (1 << 16),
        "file_idx {file_idx} overflows 16 bits"
    );
    debug_assert!(rg_idx < (1 << 16), "rg_idx {rg_idx} overflows 16 bits");
    debug_assert!(
        local_offset < (1 << 32),
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
    /// Only compiled when the `parquet-provider` or `sqlite-provider` feature is enabled.
    #[cfg(any(feature = "parquet-provider", feature = "sqlite-provider"))]
    pub fn from_files(local_paths: &[&str]) -> datafusion::common::Result<Self> {
        use datafusion::error::DataFusionError;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs;
        use std::path::Path;

        let mut file_keys = Vec::with_capacity(local_paths.len());
        let mut file_cum_rows = vec![0u64];
        let mut rg_cum_rows: Vec<Vec<u64>> = Vec::with_capacity(local_paths.len());

        let mut running_total = 0u64;
        for &path in local_paths {
            let file_name = Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| DataFusionError::Execution(format!("invalid path: {path}")))?;
            file_keys.push(format!("parquet/{file_name}"));

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

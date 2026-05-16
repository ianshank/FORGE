//! HuggingFace Datasets (Parquet) export for FORGE trajectories.
//!
//! Emits sharded Apache Parquet files alongside a `dataset_info.json`
//! (and optional `README.md`) so that
//! `datasets.load_dataset(path)` works directly. The schema is derived
//! once via [`schema`] and is the single source of truth — schema changes
//! must bump [`SCHEMA_VERSION`].
//!
//! # Granularity
//!
//! Each row in the output represents one `(step, agent)` pair, mirroring
//! the CSV export. The columns are stable; new columns appended in future
//! schema versions will be filled with nulls in old shards.
//!
//! # Backward compatibility
//!
//! Disabled by default. Available behind `--features hf` only. Default
//! `forge-replay` builds carry zero arrow/parquet code or deps.
//!
//! All thresholds — shard size, compression, file names — flow through
//! [`HfExportConfig`]. No hard-coded values land in the writers.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float32Builder, StringBuilder, UInt16Builder, UInt32Builder,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::GzipLevel;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

use crate::trajectory::Trajectory;

// ────────────────────────────── public constants ─────────────────────────

/// Current on-disk schema version. Bump this when columns are added or
/// renamed; old shards stay loadable thanks to Parquet's nullable
/// projection but downstream tooling can branch on this string.
pub const SCHEMA_VERSION: &str = "1";

/// Default rows per Parquet shard. 100k rows ≈ a few MB on typical workloads
/// and balances read parallelism vs metadata overhead.
pub const DEFAULT_SHARD_SIZE_ROWS: usize = 100_000;

/// Filename for the dataset-info JSON sidecar.
pub const DATASET_INFO_FILENAME: &str = "dataset_info.json";

/// Filename for the optional dataset README.
pub const DATASET_CARD_FILENAME: &str = "README.md";

/// Sub-directory under [`HfExportConfig::output_dir`] for Parquet shards.
pub const DATA_SUBDIR: &str = "data";

/// Default shard filename template — `{prefix}-{idx:05}-of-{count:05}.parquet`.
pub const DEFAULT_SHARD_PREFIX: &str = "data";

// ─────────────────────────────── compression ─────────────────────────────

/// Parquet column compression algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParquetCompression {
    /// No compression.
    None,
    /// Snappy — fastest, modest ratio. Default.
    #[default]
    Snappy,
    /// Gzip — best portability across HF Datasets, slower.
    Gzip,
    /// Zstd — best ratio, supported by modern readers.
    Zstd,
}

impl ParquetCompression {
    fn to_parquet(self) -> Compression {
        match self {
            Self::None => Compression::UNCOMPRESSED,
            Self::Snappy => Compression::SNAPPY,
            // Default levels are tuned by the parquet crate for sensible
            // compromise — we don't override here so the choice is data-driven.
            Self::Gzip => Compression::GZIP(GzipLevel::default()),
            Self::Zstd => Compression::ZSTD(ZstdLevel::default()),
        }
    }
}

// ───────────────────────────── dataset card ──────────────────────────────

/// Optional dataset README emitted as `README.md`.
///
/// Fields map onto HuggingFace dataset-card front-matter conventions but
/// are not interpreted by the writer beyond serialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetCard {
    /// Title displayed at the top of the README.
    pub title: String,
    /// Short description of the dataset.
    pub description: String,
    /// Optional SPDX license identifier (e.g. `"apache-2.0"`).
    #[serde(default)]
    pub license: Option<String>,
    /// Free-form tags surfaced in HF Hub search.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl DatasetCard {
    fn to_markdown(&self) -> String {
        let mut md = String::new();
        if !self.tags.is_empty() || self.license.is_some() {
            md.push_str("---\n");
            if let Some(license) = &self.license {
                md.push_str(&format!("license: {license}\n"));
            }
            if !self.tags.is_empty() {
                md.push_str("tags:\n");
                for tag in &self.tags {
                    md.push_str(&format!("  - {tag}\n"));
                }
            }
            md.push_str("---\n\n");
        }
        if !self.title.is_empty() {
            md.push_str(&format!("# {}\n\n", self.title));
        }
        if !self.description.is_empty() {
            md.push_str(&self.description);
            md.push('\n');
        }
        md
    }
}

// ─────────────────────────────── config ──────────────────────────────────

/// Configuration for [`write_parquet_shards`].
///
/// Defaults give a HuggingFace-compatible layout under
/// `<output_dir>/data/data-00000-of-00001.parquet` plus a
/// `dataset_info.json`. All knobs are settable; no value is hard-coded
/// into the writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HfExportConfig {
    /// Root directory for the exported dataset.
    pub output_dir: PathBuf,
    /// Maximum rows per Parquet shard.
    pub shard_size_rows: usize,
    /// Compression algorithm for column chunks.
    pub compression: ParquetCompression,
    /// Schema version recorded in `dataset_info.json`.
    pub schema_version: String,
    /// Optional dataset README written as `README.md`.
    pub dataset_card: Option<DatasetCard>,
    /// Filename prefix for shards (`{prefix}-NNNNN-of-MMMMM.parquet`).
    pub shard_prefix: String,
    /// Whether to create `output_dir` (and its `data/` sub-dir) if absent.
    pub create_dir: bool,
}

impl Default for HfExportConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::new(),
            shard_size_rows: DEFAULT_SHARD_SIZE_ROWS,
            compression: ParquetCompression::default(),
            schema_version: SCHEMA_VERSION.to_string(),
            dataset_card: None,
            shard_prefix: DEFAULT_SHARD_PREFIX.to_string(),
            create_dir: true,
        }
    }
}

impl HfExportConfig {
    /// Validates the configuration, returning a list of errors.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.output_dir.as_os_str().is_empty() {
            errors.push("output_dir must not be empty".to_string());
        }
        if self.shard_size_rows == 0 {
            errors.push("shard_size_rows must be > 0".to_string());
        }
        if self.schema_version.is_empty() {
            errors.push("schema_version must not be empty".to_string());
        }
        if self.shard_prefix.is_empty() {
            errors.push("shard_prefix must not be empty".to_string());
        }
        errors
    }

    /// Returns true if `validate` returns no errors.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Returns the data sub-directory (`<output_dir>/data`).
    pub fn data_dir(&self) -> PathBuf {
        self.output_dir.join(DATA_SUBDIR)
    }

    /// Builds the path of the `idx`-th shard out of `count` total shards.
    pub fn shard_path(&self, idx: usize, count: usize) -> PathBuf {
        let name = format!("{}-{:05}-of-{:05}.parquet", self.shard_prefix, idx, count);
        self.data_dir().join(name)
    }

    /// Path of the `dataset_info.json` sidecar.
    pub fn dataset_info_path(&self) -> PathBuf {
        self.output_dir.join(DATASET_INFO_FILENAME)
    }

    /// Path of the optional `README.md` (only meaningful when `dataset_card`
    /// is Some).
    pub fn dataset_card_path(&self) -> PathBuf {
        self.output_dir.join(DATASET_CARD_FILENAME)
    }
}

// ─────────────────────────────── manifest ────────────────────────────────

/// Result of a successful export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfExportManifest {
    /// Schema version recorded in the sidecar.
    pub schema_version: String,
    /// Total rows written across every shard.
    pub row_count: u64,
    /// Total bytes written across every shard.
    pub byte_count: u64,
    /// Paths of the written shards in order.
    pub shard_paths: Vec<PathBuf>,
    /// Path of the `dataset_info.json` sidecar.
    pub info_path: PathBuf,
    /// Path of the README, if a [`DatasetCard`] was provided.
    pub card_path: Option<PathBuf>,
}

// ──────────────────────────────── errors ─────────────────────────────────

/// Errors returned by [`write_parquet_shards`].
#[derive(Debug, Error)]
pub enum HfError {
    /// Config-level validation failure.
    #[error("invalid HfExportConfig: {0:?}")]
    InvalidConfig(Vec<String>),
    /// Filesystem I/O failure.
    #[error("I/O error at {path}: {message}")]
    Io {
        /// Offending path.
        path: PathBuf,
        /// Underlying error message.
        message: String,
    },
    /// Arrow array construction failure.
    #[error("arrow error: {0}")]
    Arrow(String),
    /// Parquet writer failure.
    #[error("parquet error: {0}")]
    Parquet(String),
    /// Manifest / dataset-info serialization failure.
    #[error("serialization error: {0}")]
    Serde(String),
}

// ─────────────────────────────── schema ──────────────────────────────────

/// Returns the canonical Arrow schema for FORGE trajectory exports.
///
/// Field order is part of the schema-version contract — do not reorder
/// without bumping [`SCHEMA_VERSION`].
pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("seed", DataType::UInt64, false),
        Field::new("scenario_id", DataType::Utf8, true),
        Field::new("tick", DataType::UInt64, false),
        Field::new("agent_idx", DataType::UInt32, false),
        Field::new("action", DataType::UInt32, false),
        Field::new("reward", DataType::Float32, false),
        Field::new("terminated", DataType::Boolean, false),
        Field::new("truncated", DataType::Boolean, false),
        Field::new("reasoning", DataType::Utf8, true),
        Field::new("confidence", DataType::Float32, false),
        Field::new("decision_time_ms", DataType::UInt64, false),
        Field::new("agent_health", DataType::Float32, false),
        Field::new("agent_stamina", DataType::Float32, false),
        Field::new("agent_position_x", DataType::UInt16, false),
        Field::new("agent_position_y", DataType::UInt16, false),
        Field::new("agent_battery", DataType::Float32, false),
    ]))
}

// ───────────────────────────── shard writer ──────────────────────────────

/// Column builders for one in-flight shard.
struct ShardBuilders {
    seed: UInt64Builder,
    scenario_id: StringBuilder,
    tick: UInt64Builder,
    agent_idx: UInt32Builder,
    action: UInt32Builder,
    reward: Float32Builder,
    terminated: BooleanBuilder,
    truncated: BooleanBuilder,
    reasoning: StringBuilder,
    confidence: Float32Builder,
    decision_time_ms: UInt64Builder,
    agent_health: Float32Builder,
    agent_stamina: Float32Builder,
    agent_position_x: UInt16Builder,
    agent_position_y: UInt16Builder,
    agent_battery: Float32Builder,
    row_count: usize,
}

impl ShardBuilders {
    fn new(capacity: usize) -> Self {
        Self {
            seed: UInt64Builder::with_capacity(capacity),
            scenario_id: StringBuilder::with_capacity(capacity, capacity * 16),
            tick: UInt64Builder::with_capacity(capacity),
            agent_idx: UInt32Builder::with_capacity(capacity),
            action: UInt32Builder::with_capacity(capacity),
            reward: Float32Builder::with_capacity(capacity),
            terminated: BooleanBuilder::with_capacity(capacity),
            truncated: BooleanBuilder::with_capacity(capacity),
            reasoning: StringBuilder::with_capacity(capacity, capacity * 32),
            confidence: Float32Builder::with_capacity(capacity),
            decision_time_ms: UInt64Builder::with_capacity(capacity),
            agent_health: Float32Builder::with_capacity(capacity),
            agent_stamina: Float32Builder::with_capacity(capacity),
            agent_position_x: UInt16Builder::with_capacity(capacity),
            agent_position_y: UInt16Builder::with_capacity(capacity),
            agent_battery: Float32Builder::with_capacity(capacity),
            row_count: 0,
        }
    }

    fn finish(mut self) -> Result<RecordBatch, HfError> {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.seed.finish()),
            Arc::new(self.scenario_id.finish()),
            Arc::new(self.tick.finish()),
            Arc::new(self.agent_idx.finish()),
            Arc::new(self.action.finish()),
            Arc::new(self.reward.finish()),
            Arc::new(self.terminated.finish()),
            Arc::new(self.truncated.finish()),
            Arc::new(self.reasoning.finish()),
            Arc::new(self.confidence.finish()),
            Arc::new(self.decision_time_ms.finish()),
            Arc::new(self.agent_health.finish()),
            Arc::new(self.agent_stamina.finish()),
            Arc::new(self.agent_position_x.finish()),
            Arc::new(self.agent_position_y.finish()),
            Arc::new(self.agent_battery.finish()),
        ];
        RecordBatch::try_new(schema(), arrays).map_err(|e| HfError::Arrow(e.to_string()))
    }
}

/// Appends one `(step, agent)` row to the in-flight shard.
fn append_row(
    builders: &mut ShardBuilders,
    seed: u64,
    scenario_id: Option<&str>,
    step: &crate::trajectory::TrajectoryStep,
    agent_idx: usize,
) {
    let obs = step.observations.get(agent_idx);
    builders.seed.append_value(seed);
    builders.scenario_id.append_option(scenario_id);
    builders.tick.append_value(step.tick);
    builders.agent_idx.append_value(agent_idx as u32);
    builders
        .action
        .append_value(step.actions.get(agent_idx).copied().unwrap_or(0));
    builders
        .reward
        .append_value(step.rewards.get(agent_idx).copied().unwrap_or(0.0));
    builders.terminated.append_value(step.terminated);
    builders.truncated.append_value(step.truncated);
    builders
        .reasoning
        .append_option(step.reasoning.get(agent_idx).and_then(|r| r.as_deref()));
    builders
        .confidence
        .append_value(step.confidences.get(agent_idx).copied().unwrap_or(0.0));
    builders
        .decision_time_ms
        .append_value(step.decision_times_ms.get(agent_idx).copied().unwrap_or(0));
    builders
        .agent_health
        .append_value(obs.map(|o| o.health).unwrap_or(0.0));
    builders
        .agent_stamina
        .append_value(obs.map(|o| o.stamina).unwrap_or(0.0));
    let pos = obs.map(|o| o.position).unwrap_or((0, 0));
    builders.agent_position_x.append_value(pos.0);
    builders.agent_position_y.append_value(pos.1);
    builders
        .agent_battery
        .append_value(obs.map(|o| o.battery).unwrap_or(0.0));
    builders.row_count += 1;
}

// ───────────────────────────── public writer ─────────────────────────────

/// Writes a sequence of [`Trajectory`] records as sharded Parquet under
/// `cfg.output_dir`, and emits a HuggingFace-compatible `dataset_info.json`
/// sidecar.
#[instrument(skip_all, fields(dir = %cfg.output_dir.display()))]
pub fn write_parquet_shards<I>(
    trajectories: I,
    cfg: &HfExportConfig,
) -> Result<HfExportManifest, HfError>
where
    I: IntoIterator<Item = Trajectory>,
{
    let errors = cfg.validate();
    if !errors.is_empty() {
        return Err(HfError::InvalidConfig(errors));
    }

    if cfg.create_dir {
        let data_dir = cfg.data_dir();
        fs::create_dir_all(&data_dir).map_err(|e| HfError::Io {
            path: data_dir.clone(),
            message: e.to_string(),
        })?;
    }

    // Pass 1: gather rows into in-memory shards. We can stream-write each
    // shard as it fills, but we don't know the total shard count until done,
    // so we use a two-pass strategy: write each shard with a temporary name,
    // then rename at the end.
    let mut temp_paths: Vec<PathBuf> = Vec::new();
    let mut builders = ShardBuilders::new(cfg.shard_size_rows);
    let mut total_rows: u64 = 0;

    let writer_props = WriterProperties::builder()
        .set_compression(cfg.compression.to_parquet())
        .build();
    let writer_props = Arc::new(writer_props);

    let flush = |builders: ShardBuilders,
                 idx: usize,
                 props: Arc<WriterProperties>,
                 cfg: &HfExportConfig,
                 temp_paths: &mut Vec<PathBuf>|
     -> Result<u64, HfError> {
        if builders.row_count == 0 {
            return Ok(0);
        }
        let rows = builders.row_count as u64;
        let batch = builders.finish()?;
        let temp = cfg
            .data_dir()
            .join(format!("{}.partial-{:05}.parquet", cfg.shard_prefix, idx));
        let file = File::create(&temp).map_err(|e| HfError::Io {
            path: temp.clone(),
            message: e.to_string(),
        })?;
        let mut aw = ArrowWriter::try_new(file, schema(), Some((*props).clone()))
            .map_err(|e| HfError::Parquet(e.to_string()))?;
        aw.write(&batch)
            .map_err(|e| HfError::Parquet(e.to_string()))?;
        aw.close().map_err(|e| HfError::Parquet(e.to_string()))?;
        temp_paths.push(temp);
        Ok(rows)
    };

    for traj in trajectories.into_iter() {
        let seed = traj.metadata.seed;
        let scenario_id = traj.metadata.scenario_id.clone();
        for step in &traj.steps {
            let agents = step
                .actions
                .len()
                .max(step.observations.len())
                .max(step.rewards.len());
            for agent_idx in 0..agents {
                append_row(&mut builders, seed, scenario_id.as_deref(), step, agent_idx);
                if builders.row_count >= cfg.shard_size_rows {
                    let next_idx = temp_paths.len();
                    let flushed = flush(
                        std::mem::replace(&mut builders, ShardBuilders::new(cfg.shard_size_rows)),
                        next_idx,
                        writer_props.clone(),
                        cfg,
                        &mut temp_paths,
                    )?;
                    total_rows += flushed;
                }
            }
        }
    }
    // Final shard.
    let next_idx = temp_paths.len();
    total_rows += flush(builders, next_idx, writer_props, cfg, &mut temp_paths)?;

    let shard_count = temp_paths.len();
    let mut shard_paths: Vec<PathBuf> = Vec::with_capacity(shard_count);
    let mut byte_count: u64 = 0;
    for (idx, temp) in temp_paths.into_iter().enumerate() {
        let final_path = cfg.shard_path(idx, shard_count.max(1));
        fs::rename(&temp, &final_path).map_err(|e| HfError::Io {
            path: final_path.clone(),
            message: e.to_string(),
        })?;
        let meta = fs::metadata(&final_path).map_err(|e| HfError::Io {
            path: final_path.clone(),
            message: e.to_string(),
        })?;
        byte_count += meta.len();
        shard_paths.push(final_path);
    }

    let info_path = cfg.dataset_info_path();
    let info = serde_json::json!({
        "schema_version": cfg.schema_version,
        "row_count": total_rows,
        "shard_paths": shard_paths
            .iter()
            .map(|p| p.strip_prefix(&cfg.output_dir).unwrap_or(p).to_string_lossy())
            .collect::<Vec<_>>(),
        "compression": format!("{:?}", cfg.compression).to_lowercase(),
        "columns": schema().fields().iter().map(|f| f.name().clone()).collect::<Vec<_>>(),
    });
    let info_str =
        serde_json::to_string_pretty(&info).map_err(|e| HfError::Serde(e.to_string()))?;
    fs::write(&info_path, info_str).map_err(|e| HfError::Io {
        path: info_path.clone(),
        message: e.to_string(),
    })?;

    let card_path = if let Some(card) = &cfg.dataset_card {
        let path = cfg.dataset_card_path();
        fs::write(&path, card.to_markdown()).map_err(|e| HfError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;
        Some(path)
    } else {
        None
    };

    info!(
        rows = total_rows,
        shards = shard_count,
        bytes = byte_count,
        "Wrote HF Parquet dataset"
    );
    if shard_count == 0 {
        warn!("HF export produced zero shards — input was empty");
    }
    debug!(path = %info_path.display(), "Wrote dataset_info.json");

    Ok(HfExportManifest {
        schema_version: cfg.schema_version.clone(),
        row_count: total_rows,
        byte_count,
        shard_paths,
        info_path,
        card_path,
    })
}

/// Helper for callers that have a single `Trajectory` to export.
#[inline]
pub fn write_single_trajectory(
    traj: &Trajectory,
    cfg: &HfExportConfig,
) -> Result<HfExportManifest, HfError> {
    write_parquet_shards(std::iter::once(traj.clone()), cfg)
}

/// Verifies that `path` exists and is readable. Public for downstream tools.
pub fn ensure_dataset_loadable(path: &Path) -> Result<(), HfError> {
    if !path.exists() {
        return Err(HfError::Io {
            path: path.to_path_buf(),
            message: "dataset path does not exist".to_string(),
        });
    }
    let info = path.join(DATASET_INFO_FILENAME);
    if !info.exists() {
        return Err(HfError::Io {
            path: info,
            message: "missing dataset_info.json".to_string(),
        });
    }
    Ok(())
}

// ─────────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Trajectory, TrajectoryBuilder};
    use forge_types::agent_interface::AgentResponse;
    use forge_types::constants;
    use forge_types::observation::{InventoryObservation, Observation, TileObservation};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    fn make_obs(health: f32, stamina: f32, x: u16, y: u16) -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health,
            stamina,
            position: (x, y),
            messages: vec![],
            day_phase: 0,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    fn make_trajectory(seed: u64, scenario: &str, steps: usize, agents: usize) -> Trajectory {
        let mut b = TrajectoryBuilder::new();
        for tick in 0..steps {
            let obs: Vec<Observation> = (0..agents)
                .map(|i| make_obs(1.0 - i as f32 * 0.01, 0.9, tick as u16, i as u16))
                .collect();
            let responses: Vec<AgentResponse> = (0..agents)
                .map(|i| {
                    let mut r = AgentResponse::from_action((tick + i) as u32);
                    r.confidence = 0.5;
                    r.decision_time_ms = (tick * 10 + i) as u64;
                    if i == 0 && tick == 0 {
                        r.reasoning = Some("first move".to_string());
                    }
                    r
                })
                .collect();
            let rewards: Vec<f32> = (0..agents).map(|i| 0.1 * (tick + i) as f32).collect();
            b.record_step(
                tick as u64,
                obs,
                &responses,
                rewards,
                tick + 1 == steps,
                false,
            );
        }
        b.seed(seed)
            .scenario_id(scenario.to_string())
            .build(vec![1.0; agents])
    }

    fn test_cfg(dir: &std::path::Path) -> HfExportConfig {
        HfExportConfig {
            output_dir: dir.to_path_buf(),
            shard_size_rows: 4,
            compression: ParquetCompression::None,
            ..Default::default()
        }
    }

    #[test]
    fn test_default_config_is_invalid_until_dir_set() {
        let cfg = HfExportConfig::default();
        assert!(!cfg.is_valid(), "empty output_dir must fail validation");
    }

    #[test]
    fn test_config_validate_zero_shard_size() {
        let cfg = HfExportConfig {
            output_dir: PathBuf::from("/tmp/x"),
            shard_size_rows: 0,
            ..Default::default()
        };
        assert!(!cfg.is_valid());
        assert!(cfg.validate().iter().any(|e| e.contains("shard_size_rows")));
    }

    #[test]
    fn test_config_validate_empty_schema_version() {
        let cfg = HfExportConfig {
            output_dir: PathBuf::from("/tmp/x"),
            schema_version: String::new(),
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_validate_empty_prefix() {
        let cfg = HfExportConfig {
            output_dir: PathBuf::from("/tmp/x"),
            shard_prefix: String::new(),
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let mut cfg = HfExportConfig {
            output_dir: PathBuf::from("./out"),
            shard_size_rows: 10,
            compression: ParquetCompression::Zstd,
            ..Default::default()
        };
        cfg.dataset_card = Some(DatasetCard {
            title: "t".into(),
            description: "d".into(),
            license: Some("apache-2.0".into()),
            tags: vec!["forge".into()],
        });
        let toml_str = toml::to_string(&cfg).unwrap();
        let back: HfExportConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(back.shard_size_rows, cfg.shard_size_rows);
        assert_eq!(back.compression, ParquetCompression::Zstd);
        assert_eq!(back.dataset_card.as_ref().unwrap().tags, vec!["forge"]);
    }

    #[test]
    fn test_schema_field_set_is_stable() {
        let s = schema();
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "seed",
                "scenario_id",
                "tick",
                "agent_idx",
                "action",
                "reward",
                "terminated",
                "truncated",
                "reasoning",
                "confidence",
                "decision_time_ms",
                "agent_health",
                "agent_stamina",
                "agent_position_x",
                "agent_position_y",
                "agent_battery",
            ],
            "schema field order is part of SCHEMA_VERSION={SCHEMA_VERSION} — bump the version before changing"
        );
    }

    #[test]
    fn test_compression_to_parquet_all_variants() {
        // Ensure every variant maps without panic.
        let _ = ParquetCompression::None.to_parquet();
        let _ = ParquetCompression::Snappy.to_parquet();
        let _ = ParquetCompression::Gzip.to_parquet();
        let _ = ParquetCompression::Zstd.to_parquet();
    }

    #[test]
    fn test_dataset_card_markdown_includes_frontmatter_when_meta_present() {
        let card = DatasetCard {
            title: "FORGE Trajectories".into(),
            description: "Test dataset".into(),
            license: Some("apache-2.0".into()),
            tags: vec!["robotics".into(), "rl".into()],
        };
        let md = card.to_markdown();
        assert!(md.starts_with("---\n"));
        assert!(md.contains("license: apache-2.0"));
        assert!(md.contains("- robotics"));
        assert!(md.contains("# FORGE Trajectories"));
        assert!(md.contains("Test dataset"));
    }

    #[test]
    fn test_dataset_card_markdown_skips_frontmatter_when_empty() {
        let card = DatasetCard {
            title: "T".into(),
            description: "D".into(),
            license: None,
            tags: vec![],
        };
        let md = card.to_markdown();
        assert!(!md.starts_with("---"));
        assert!(md.contains("# T"));
    }

    #[test]
    fn test_write_parquet_shards_single_trajectory_round_trip() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let traj = make_trajectory(99, "scn", 3, 2);
        let manifest = write_parquet_shards(std::iter::once(traj), &cfg).unwrap();

        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(manifest.row_count, 6); // 3 steps * 2 agents
        assert!(!manifest.shard_paths.is_empty());
        assert!(manifest.info_path.exists());
        assert!(manifest.card_path.is_none());

        // Round-trip: read each shard and check the row count matches.
        let mut rows_read = 0u64;
        for path in &manifest.shard_paths {
            let file = File::open(path).unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            for batch in reader.build().unwrap() {
                rows_read += batch.unwrap().num_rows() as u64;
            }
        }
        assert_eq!(rows_read, manifest.row_count);
    }

    #[test]
    fn test_write_parquet_shards_respects_shard_size() {
        let dir = tempdir().unwrap();
        let cfg = HfExportConfig {
            output_dir: dir.path().to_path_buf(),
            shard_size_rows: 5,
            compression: ParquetCompression::None,
            ..Default::default()
        };
        // 8 steps × 2 agents = 16 rows → expect 4 shards of 5,5,5,1.
        let traj = make_trajectory(1, "shard", 8, 2);
        let manifest = write_parquet_shards(std::iter::once(traj), &cfg).unwrap();
        assert_eq!(manifest.row_count, 16);
        assert_eq!(
            manifest.shard_paths.len(),
            4,
            "16 rows / 5 per shard = 4 shards"
        );
    }

    /// Verify each compression algorithm actually round-trips a real row.
    /// Catches the regression where `Gzip` would compile but fail at write
    /// time without the `flate2` feature on the `parquet` dep.
    /// (review thread r3252813119)
    #[test]
    fn test_every_compression_writes_and_reads_back() {
        for compression in [
            ParquetCompression::None,
            ParquetCompression::Snappy,
            ParquetCompression::Gzip,
            ParquetCompression::Zstd,
        ] {
            let dir = tempdir().unwrap();
            let cfg = HfExportConfig {
                output_dir: dir.path().to_path_buf(),
                shard_size_rows: 8,
                compression,
                ..Default::default()
            };
            let traj = make_trajectory(1, "compr", 4, 1);
            let manifest = write_parquet_shards(std::iter::once(traj), &cfg)
                .unwrap_or_else(|e| panic!("write failed for {compression:?}: {e:?}"));
            assert_eq!(manifest.row_count, 4, "compression={compression:?}");
            assert!(!manifest.shard_paths.is_empty());

            // Round-trip read to confirm bytes are decompressible.
            let mut total = 0u64;
            for path in &manifest.shard_paths {
                let file = File::open(path).unwrap();
                let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
                for batch in reader.build().unwrap() {
                    total += batch.unwrap().num_rows() as u64;
                }
            }
            assert_eq!(
                total, manifest.row_count,
                "read-back row count mismatch for {compression:?}"
            );
        }
    }

    #[test]
    fn test_write_parquet_shards_emits_dataset_info_json() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let traj = make_trajectory(7, "info", 2, 1);
        let manifest = write_parquet_shards(std::iter::once(traj), &cfg).unwrap();
        let info: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest.info_path).unwrap()).unwrap();
        assert_eq!(info["schema_version"], SCHEMA_VERSION);
        assert_eq!(info["row_count"], 2);
        assert!(info["columns"].is_array());
        assert_eq!(info["columns"].as_array().unwrap().len(), 16);
    }

    #[test]
    fn test_write_parquet_shards_emits_card_when_provided() {
        let dir = tempdir().unwrap();
        let mut cfg = test_cfg(dir.path());
        cfg.dataset_card = Some(DatasetCard {
            title: "Card".into(),
            description: "Body".into(),
            license: Some("apache-2.0".into()),
            tags: vec!["t1".into()],
        });
        let traj = make_trajectory(2, "card", 1, 1);
        let manifest = write_parquet_shards(std::iter::once(traj), &cfg).unwrap();
        let card_path = manifest
            .card_path
            .expect("card_path must be present when dataset_card is set");
        let md = std::fs::read_to_string(&card_path).unwrap();
        assert!(md.contains("# Card"));
        assert!(md.contains("Body"));
    }

    #[test]
    fn test_write_parquet_shards_invalid_config_errors() {
        let bad = HfExportConfig::default(); // empty output_dir
        let err = write_parquet_shards(std::iter::empty::<Trajectory>(), &bad).unwrap_err();
        assert!(matches!(err, HfError::InvalidConfig(_)));
    }

    #[test]
    fn test_write_parquet_shards_empty_input_writes_only_info() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let manifest = write_parquet_shards(std::iter::empty::<Trajectory>(), &cfg).unwrap();
        assert_eq!(manifest.row_count, 0);
        assert!(manifest.shard_paths.is_empty());
        assert!(manifest.info_path.exists());
    }

    #[test]
    fn test_write_parquet_shards_multi_trajectory_preserves_rows() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let trajectories = vec![
            make_trajectory(10, "a", 2, 2),
            make_trajectory(20, "b", 3, 1),
            make_trajectory(30, "c", 1, 4),
        ];
        let manifest = write_parquet_shards(trajectories, &cfg).unwrap();
        // 2*2 + 3*1 + 1*4 = 11
        assert_eq!(manifest.row_count, 11);
    }

    #[test]
    fn test_write_parquet_shards_payload_columns_match_source() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let traj = make_trajectory(42, "verify", 2, 1);
        let manifest = write_parquet_shards(std::iter::once(traj), &cfg).unwrap();

        let file = File::open(&manifest.shard_paths[0]).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<RecordBatch> = reader.filter_map(Result::ok).collect();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let seed_col = batch
            .column_by_name("seed")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(seed_col.value(0), 42);
        assert_eq!(seed_col.value(1), 42);

        let scenario_col = batch
            .column_by_name("scenario_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        assert_eq!(scenario_col.value(0), "verify");
    }

    #[test]
    fn test_ensure_dataset_loadable_happy_path() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let _ = write_parquet_shards(std::iter::empty::<Trajectory>(), &cfg).unwrap();
        ensure_dataset_loadable(dir.path()).unwrap();
    }

    #[test]
    fn test_ensure_dataset_loadable_missing_dir() {
        let err = ensure_dataset_loadable(Path::new("/this/does/not/exist")).unwrap_err();
        assert!(matches!(err, HfError::Io { .. }));
    }

    #[test]
    fn test_ensure_dataset_loadable_missing_info() {
        let dir = tempdir().unwrap();
        let err = ensure_dataset_loadable(dir.path()).unwrap_err();
        assert!(matches!(err, HfError::Io { .. }));
    }

    #[test]
    fn test_shard_path_and_info_path_formation() {
        let cfg = HfExportConfig {
            output_dir: PathBuf::from("/tmp/x"),
            shard_prefix: "trace".into(),
            ..Default::default()
        };
        let p = cfg.shard_path(3, 7);
        assert!(p.ends_with("data/trace-00003-of-00007.parquet"));
        assert_eq!(
            cfg.dataset_info_path(),
            PathBuf::from("/tmp/x/dataset_info.json")
        );
    }

    #[test]
    fn test_write_single_trajectory_helper() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        let traj = make_trajectory(5, "single", 2, 1);
        let manifest = write_single_trajectory(&traj, &cfg).unwrap();
        assert_eq!(manifest.row_count, 2);
    }
}

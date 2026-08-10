//! `forge-gen-dataset` — HuggingFace-ready Parquet trajectory dataset generator.
//!
//! Drives [`ExpertDemoGenerator`] over a cross-product of configuration
//! cells (world size × agent count × task tier × policy) and streams the
//! resulting episodes into [`forge_replay::hf::write_parquet_shards`],
//! producing `data/data-NNNNN-of-MMMMM.parquet` shards plus a
//! `dataset_info.json` sidecar that `datasets.load_dataset` opens natively.
//!
//! Episodes are generated in bounded parallel batches (rayon inside each
//! batch, lazily pulled by the Parquet writer) so memory stays flat no
//! matter how many episodes are requested. Every cell gets a disjoint,
//! contiguous seed block, making `(scenario_id, seed)` a globally unique,
//! fully reproducible episode key.
//!
//! Only the Square grid is generated for now: hex movement actions are not
//! representable in the base discrete action encoding the generator uses
//! (`Action::to_discrete` panics on `MoveHex`), so hex cells are deferred
//! until the generator adopts the `_configured` encoding surface.
//!
//! ```bash
//! cargo run --release -p forge-data --features hf --bin forge-gen-dataset -- \
//!     --out /tmp/forge-trajectories --episodes-per-cell 185
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context};
use clap::Parser;
use forge_data::generator::{DemoPolicy, ExpertDemoConfig, ExpertDemoGenerator};
use forge_observability::{init_tracing, TracingOptions};
use forge_replay::hf::{write_parquet_shards, HfExportConfig, ParquetCompression};
use forge_replay::trajectory::Trajectory;
use rayon::prelude::*;
use tracing::{info, warn};

/// Default episodes per configuration cell (54 cells × 185 ≈ 10K episodes).
const DEFAULT_EPISODES_PER_CELL: u64 = 185;

/// Filename of the machine-readable export summary written into `--out`.
const EXPORT_MANIFEST_FILENAME: &str = "export_manifest.json";

/// Default number of episodes generated per parallel batch. Bounds peak
/// memory at roughly `batch × max_steps × ~1 KB` while keeping every core
/// busy between Parquet flushes.
const DEFAULT_GENERATION_BATCH: usize = 64;

#[derive(Debug, Parser)]
#[command(
    name = "forge-gen-dataset",
    version,
    about = "Generate a HuggingFace-ready Parquet dataset of FORGE gridworld trajectories"
)]
struct Cli {
    /// Output directory for the dataset (created if missing).
    #[arg(long)]
    out: PathBuf,

    /// Episodes generated per configuration cell.
    #[arg(long, default_value_t = DEFAULT_EPISODES_PER_CELL)]
    episodes_per_cell: u64,

    /// Base seed; each cell receives the disjoint block
    /// `base_seed + cell_idx * episodes_per_cell ..` of this length.
    #[arg(long, default_value_t = 0)]
    base_seed: u64,

    /// Maximum steps per episode.
    #[arg(long, default_value_t = 500)]
    max_steps: u64,

    /// Rows per Parquet shard.
    #[arg(long, default_value_t = forge_replay::hf::DEFAULT_SHARD_SIZE_ROWS)]
    shard_size: usize,

    /// Parquet compression: none | snappy | gzip | zstd.
    #[arg(long, default_value = "snappy")]
    compression: String,

    /// MCTS simulations per decision for the `mcts` policy.
    #[arg(long, default_value_t = 50)]
    mcts_sims: u32,

    /// Square world sizes to include in the cell grid.
    #[arg(long, value_delimiter = ',', default_values_t = vec![32u16, 64, 128])]
    world_sizes: Vec<u16>,

    /// Agent counts to include in the cell grid.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1u32, 2, 4])]
    agent_counts: Vec<u32>,

    /// Task curriculum tiers to include in the cell grid.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1u8, 2, 3])]
    tiers: Vec<u8>,

    /// Policies to include: mcts | random (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec!["mcts".to_string(), "random".to_string()])]
    policies: Vec<String>,

    /// Optional exact cell-label filter (comma-separated). Labels look like
    /// `square-64-a2-t2-mcts`; unknown labels are an error.
    #[arg(long, value_delimiter = ',')]
    cells: Option<Vec<String>>,

    /// Episodes per parallel generation batch (memory/parallelism knob).
    #[arg(long, default_value_t = DEFAULT_GENERATION_BATCH)]
    generation_batch: usize,
}

/// One configuration cell of the dataset cross-product.
#[derive(Debug, Clone)]
struct CellSpec {
    world_size: u16,
    num_agents: u32,
    max_tier: u8,
    policy: DemoPolicy,
    /// Position in the FULL (unfiltered) cell grid. Seed blocks derive from
    /// this — not from the position in a `--cells`-filtered list — so
    /// regenerating a subset reproduces exactly the published episodes.
    grid_index: usize,
}

impl CellSpec {
    /// Stable label recorded as `scenario_id` for every episode in the cell.
    fn label(&self) -> String {
        format!(
            "square-{}-a{}-t{}-{}",
            self.world_size, self.num_agents, self.max_tier, self.policy
        )
    }

    /// First seed of this cell's disjoint block.
    fn seed_start(&self, cli: &Cli) -> u64 {
        cli.base_seed + self.grid_index as u64 * cli.episodes_per_cell
    }

    /// Builds the generator configuration for this cell.
    fn to_config(&self, cli: &Cli) -> ExpertDemoConfig {
        let mut cfg = ExpertDemoConfig::default();
        cfg.forge_config.world.width = self.world_size;
        cfg.forge_config.world.height = self.world_size;
        cfg.forge_config.agents.num_agents = self.num_agents;
        cfg.forge_config.agents.comm_vocab_size = 0;
        cfg.forge_config.task.max_tier = self.max_tier;
        cfg.forge_config.task.max_episode_length = cli.max_steps;
        cfg.mcts_config.num_simulations = cli.mcts_sims;
        cfg.max_steps = cli.max_steps;
        // Parallelism is handled by the batched driver below, not per-corpus.
        cfg.parallel = false;
        cfg.policy = self.policy;
        cfg.scenario_label = Some(self.label());
        cfg
    }
}

fn parse_compression(s: &str) -> anyhow::Result<ParquetCompression> {
    match s {
        "none" => Ok(ParquetCompression::None),
        "snappy" => Ok(ParquetCompression::Snappy),
        "gzip" => Ok(ParquetCompression::Gzip),
        "zstd" => Ok(ParquetCompression::Zstd),
        other => bail!("unknown compression {other:?} (expected none|snappy|gzip|zstd)"),
    }
}

/// Expands the CLI axes into the ordered list of cells, applying `--cells`.
///
/// Duplicate labels (repeated axis values) are an error: they would silently
/// double a configuration under two different seed blocks.
fn build_cells(cli: &Cli) -> anyhow::Result<Vec<CellSpec>> {
    let policies = cli
        .policies
        .iter()
        .map(|p| p.parse::<DemoPolicy>().map_err(anyhow::Error::msg))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut cells = Vec::new();
    for &world_size in &cli.world_sizes {
        for &num_agents in &cli.agent_counts {
            for &max_tier in &cli.tiers {
                for &policy in &policies {
                    cells.push(CellSpec {
                        world_size,
                        num_agents,
                        max_tier,
                        policy,
                        grid_index: cells.len(),
                    });
                }
            }
        }
    }

    let mut sorted_labels: Vec<String> = cells.iter().map(CellSpec::label).collect();
    sorted_labels.sort();
    if let Some(dup) = sorted_labels.windows(2).find(|w| w[0] == w[1]) {
        bail!(
            "duplicate cell label {:?} — repeated axis value in \
             --world-sizes/--agent-counts/--tiers/--policies?",
            dup[0]
        );
    }

    if let Some(filter) = &cli.cells {
        let all_labels: Vec<String> = cells.iter().map(CellSpec::label).collect();
        for wanted in filter {
            if !all_labels.iter().any(|l| l == wanted) {
                bail!("--cells label {wanted:?} does not match any cell (labels: {all_labels:?})");
            }
        }
        // Filtering keeps each cell's grid_index, preserving its seed block.
        cells.retain(|c| filter.iter().any(|w| *w == c.label()));
    }

    if cells.is_empty() {
        bail!("cell grid is empty — check --world-sizes/--agent-counts/--tiers/--policies");
    }
    Ok(cells)
}

/// Lazily yields every episode across all cells, generating in bounded
/// parallel batches so the Parquet writer's pull drives rayon fan-out
/// without materializing the corpus.
///
/// Every failed `generate_episode` (invalid `WorldState` config) increments
/// `dropped` so the caller can fail loudly instead of shipping a silently
/// incomplete dataset.
fn episode_stream<'a>(
    cells: &'a [CellSpec],
    cli: &'a Cli,
    dropped: &'a AtomicU64,
) -> impl Iterator<Item = Trajectory> + 'a {
    cells.iter().flat_map(move |cell| {
        let generator = ExpertDemoGenerator::new(cell.to_config(cli));
        let cell_label = cell.label();
        let start = cell.seed_start(cli);
        info!(
            cell = %cell_label,
            seed_start = start,
            episodes = cli.episodes_per_cell,
            "Generating cell"
        );
        let batch_size = cli.generation_batch.max(1) as u64;
        // Seed ranges, not materialized seed vectors: memory stays bounded
        // by one in-flight batch regardless of --episodes-per-cell.
        let batch_starts = (start..start + cli.episodes_per_cell).step_by(batch_size as usize);
        let end = start + cli.episodes_per_cell;
        batch_starts.flat_map(move |batch_start| {
            let batch_end = (batch_start + batch_size).min(end);
            let expected = batch_end - batch_start;
            let episodes: Vec<Trajectory> = (batch_start..batch_end)
                .into_par_iter()
                .filter_map(|seed| generator.generate_episode(seed))
                .collect();
            if (episodes.len() as u64) < expected {
                let missing = expected - episodes.len() as u64;
                dropped.fetch_add(missing, Ordering::Relaxed);
                warn!(
                    cell = %cell_label,
                    missing,
                    "generate_episode returned None — invalid ForgeConfig for this cell?"
                );
            }
            episodes.into_iter()
        })
    })
}

/// Validates cross-field CLI invariants before any generation work.
fn validate_cli(cli: &Cli) -> anyhow::Result<()> {
    if cli.episodes_per_cell == 0 {
        bail!("--episodes-per-cell must be >= 1 (0 would publish an empty dataset)");
    }
    if cli.max_steps == 0 {
        bail!("--max-steps must be >= 1");
    }
    Ok(())
}

fn run(cli: &Cli) -> anyhow::Result<()> {
    validate_cli(cli)?;
    let cells = build_cells(cli)?;
    let total_episodes = cells.len() as u64 * cli.episodes_per_cell;
    info!(
        cells = cells.len(),
        episodes_per_cell = cli.episodes_per_cell,
        total_episodes,
        out = %cli.out.display(),
        "Starting dataset generation"
    );

    // No in-code dataset card: the published README is rendered from
    // docs/hf/dataset-card.md by the hf-dataset.yml workflow, keeping a
    // single source of truth for the card.
    let export_cfg = HfExportConfig {
        output_dir: cli.out.clone(),
        shard_size_rows: cli.shard_size,
        compression: parse_compression(&cli.compression)?,
        ..HfExportConfig::default()
    };

    let dropped = AtomicU64::new(0);
    let manifest = write_parquet_shards(episode_stream(&cells, cli, &dropped), &export_cfg)
        .context("writing Parquet shards")?;

    let dropped = dropped.load(Ordering::Relaxed);
    if dropped > 0 {
        bail!(
            "{dropped} of {total_episodes} episodes failed to generate — refusing to \
             emit a silently incomplete dataset (see warnings above for cells)"
        );
    }
    if manifest.row_count == 0 {
        bail!("generation produced zero rows — refusing to emit an empty dataset");
    }

    info!(
        rows = manifest.row_count,
        shards = manifest.shard_paths.len(),
        bytes = manifest.byte_count,
        "Dataset generation complete"
    );
    // Machine-readable summary next to the dataset for workflow consumption
    // (stdout carries tracing output, so a file is the reliable channel).
    let manifest_path = cli.out.join(EXPORT_MANIFEST_FILENAME);
    let manifest_json = serde_json::to_string_pretty(&manifest).context("serializing manifest")?;
    std::fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("writing {}", manifest_path.display()))?;
    Ok(())
}

fn main() -> ExitCode {
    init_tracing(TracingOptions::new(
        "forge_gen_dataset=info,forge_data=info,forge_replay=info",
    ));
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("forge-gen-dataset: {err:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cli() -> Cli {
        Cli::parse_from(["forge-gen-dataset", "--out", "/tmp/x"])
    }

    #[test]
    fn default_grid_is_54_cells() {
        let cli = base_cli();
        let cells = build_cells(&cli).unwrap();
        assert_eq!(cells.len(), 3 * 3 * 3 * 2);
    }

    #[test]
    fn cell_labels_are_stable_and_unique() {
        let cli = base_cli();
        let cells = build_cells(&cli).unwrap();
        let labels: Vec<String> = cells.iter().map(CellSpec::label).collect();
        let mut dedup = labels.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), labels.len(), "labels must be unique");
        assert!(labels.contains(&"square-64-a2-t2-mcts".to_string()));
    }

    #[test]
    fn cells_filter_selects_exact_labels() {
        let mut cli = base_cli();
        cli.cells = Some(vec!["square-32-a1-t1-random".to_string()]);
        let cells = build_cells(&cli).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].label(), "square-32-a1-t1-random");
    }

    #[test]
    fn filtered_cell_keeps_full_grid_seed_block() {
        // Regeneration promise: `--cells <label>` must reproduce exactly the
        // seed block the cell had in the full unfiltered grid.
        let cli = base_cli();
        let full = build_cells(&cli).unwrap();
        let target = "square-64-a2-t2-mcts";
        let in_full = full.iter().find(|c| c.label() == target).unwrap();
        assert!(in_full.grid_index > 0, "test needs a non-first cell");

        let mut filtered_cli = base_cli();
        filtered_cli.cells = Some(vec![target.to_string()]);
        let filtered = build_cells(&filtered_cli).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].grid_index, in_full.grid_index);
        assert_eq!(filtered[0].seed_start(&cli), in_full.seed_start(&cli));
    }

    #[test]
    fn duplicate_axis_values_are_rejected() {
        let mut cli = base_cli();
        cli.world_sizes = vec![32, 32, 64];
        let err = build_cells(&cli).unwrap_err();
        assert!(err.to_string().contains("duplicate cell label"), "{err}");
    }

    #[test]
    fn validate_cli_rejects_zero_episodes_and_steps() {
        let mut cli = base_cli();
        cli.episodes_per_cell = 0;
        assert!(validate_cli(&cli).is_err());
        let mut cli = base_cli();
        cli.max_steps = 0;
        assert!(validate_cli(&cli).is_err());
        assert!(validate_cli(&base_cli()).is_ok());
    }

    #[test]
    fn cells_filter_rejects_unknown_label() {
        let mut cli = base_cli();
        cli.cells = Some(vec!["hex-32-a1-t1-mcts".to_string()]);
        assert!(build_cells(&cli).is_err());
    }

    #[test]
    fn seed_blocks_are_disjoint() {
        let cli = base_cli();
        let cells = build_cells(&cli).unwrap();
        let mut all_ranges: Vec<(u64, u64)> = cells
            .iter()
            .map(|c| {
                let start = c.seed_start(&cli);
                (start, start + cli.episodes_per_cell)
            })
            .collect();
        all_ranges.sort();
        for pair in all_ranges.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "seed blocks overlap: {pair:?}");
        }
    }

    #[test]
    fn episode_stream_counts_dropped_episodes() {
        // world_size 0 fails WorldState construction, so every episode of
        // the cell is dropped — the counter must record all of them and the
        // stream must yield nothing (this is what makes run() bail instead
        // of publishing an incomplete dataset).
        let mut cli = base_cli();
        cli.episodes_per_cell = 3;
        cli.max_steps = 5;
        let cells = vec![CellSpec {
            world_size: 0,
            num_agents: 1,
            max_tier: 1,
            policy: DemoPolicy::Random,
            grid_index: 0,
        }];
        let dropped = AtomicU64::new(0);
        let produced: Vec<Trajectory> = episode_stream(&cells, &cli, &dropped).collect();
        assert!(produced.is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn episode_stream_valid_cell_drops_nothing() {
        let mut cli = base_cli();
        cli.episodes_per_cell = 2;
        cli.max_steps = 5;
        cli.mcts_sims = 2;
        let cells = vec![CellSpec {
            world_size: 16,
            num_agents: 1,
            max_tier: 1,
            policy: DemoPolicy::Random,
            grid_index: 0,
        }];
        let dropped = AtomicU64::new(0);
        let produced: Vec<Trajectory> = episode_stream(&cells, &cli, &dropped).collect();
        assert_eq!(produced.len(), 2);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        // Disjoint block starts at base_seed; labels carry the cell id.
        assert_eq!(produced[0].metadata.seed, cli.base_seed);
        assert_eq!(
            produced[0].metadata.scenario_id.as_deref(),
            Some("square-16-a1-t1-random")
        );
    }

    #[test]
    fn parse_compression_all_variants() {
        assert!(parse_compression("none").is_ok());
        assert!(parse_compression("snappy").is_ok());
        assert!(parse_compression("gzip").is_ok());
        assert!(parse_compression("zstd").is_ok());
        assert!(parse_compression("lz4").is_err());
    }

    #[test]
    fn parse_policy_variants() {
        assert_eq!("mcts".parse::<DemoPolicy>().unwrap(), DemoPolicy::Mcts);
        assert_eq!("random".parse::<DemoPolicy>().unwrap(), DemoPolicy::Random);
        assert!("llm".parse::<DemoPolicy>().is_err());
    }

    #[test]
    fn cell_config_carries_label_and_policy() {
        let cli = base_cli();
        let cell = CellSpec {
            world_size: 64,
            num_agents: 2,
            max_tier: 2,
            policy: DemoPolicy::Random,
            grid_index: 0,
        };
        let cfg = cell.to_config(&cli);
        assert_eq!(
            cfg.scenario_label.as_deref(),
            Some("square-64-a2-t2-random")
        );
        assert_eq!(cfg.policy, DemoPolicy::Random);
        assert_eq!(cfg.forge_config.world.width, 64);
        assert_eq!(cfg.forge_config.agents.num_agents, 2);
        assert_eq!(cfg.forge_config.task.max_tier, 2);
        assert!(!cfg.parallel, "driver owns parallelism, not the corpus API");
    }
}

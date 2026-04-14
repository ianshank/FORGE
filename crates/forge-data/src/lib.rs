#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-data
//!
//! Training data loaders, dataset adapters, and expert demo generation for FORGE.
//!
//! This crate provides:
//!
//! - **[`DatasetLoader`]**: Trait for all dataset readers. Produces [`Trajectory`] slices
//!   in the same format as `forge-replay`, ready for offline RL pipelines.
//!
//! - **[`loader`]**: Core trait definitions and the `OfflineDataset` container.
//!
//! - **[`minari`]**: Reader for [Minari](https://minari.farama.org/)-format JSONL exports
//!   (D4RL Maze2D, etc.). Converts observation / action / reward triples into FORGE
//!   [`Trajectory`] objects.
//!
//! - **[`minerl`]**: Action adapter for the [MineRL](https://minerl.io/) dataset.
//!   Maps Minecraft's resource-gathering and crafting actions onto FORGE's discrete
//!   `Action` enum so MineRL episodes can seed imitation-learning pipelines.
//!
//! - **[`maze`]**: Loader for the
//!   [Strategic Game Maze](https://huggingface.co/datasets/laion/strategic_game_maze)
//!   dataset (350 K 30×30 ASCII mazes with BFS solutions). Converts mazes into
//!   [`ForgeConfig`] world-generation seeds and pre-planned action sequences.
//!
//! - **[`generator`]**: [`ExpertDemoGenerator`] — runs FORGE's own MCTS agent to
//!   generate high-quality, natively-labeled trajectory corpora with zero external
//!   dependencies.
//!
//! - **[`edge_replay`]**: [`EdgeReplayLoader`] — reads compact replay files
//!   uploaded from edge devices, deterministically reconstructs full trajectories
//!   for the cloud-edge offline RL pipeline.
//!
//! # Quick-start
//!
//! ```rust,no_run
//! use forge_data::generator::{ExpertDemoConfig, ExpertDemoGenerator};
//! use forge_data::loader::OfflineDataset;
//!
//! let cfg = ExpertDemoConfig::default();
//! let gen = ExpertDemoGenerator::new(cfg);
//! let dataset: OfflineDataset = gen.generate_corpus(0..10);
//! println!("Generated {} trajectories", dataset.len());
//! ```

pub mod edge_replay;
pub mod generator;
pub mod loader;
pub mod maze;
pub mod minari;
pub mod minerl;

// ---------------------------------------------------------------------------
// Internal helpers shared by loaders
// ---------------------------------------------------------------------------

use forge_types::constants::OBS_EMPTY_SLOT_ITEM;
use forge_types::observation::{InventoryObservation, Observation, TileObservation};

/// Default observation grid view side length (tiles).
///
/// Matches `ForgeGymnasiumEnv`'s `_DEFAULT_VIEW_SIDE` and is used by all
/// loaders that produce placeholder observations for data sources that do not
/// provide a full grid view (MineRL, Minari, Strategic Game Maze).
pub(crate) const DEFAULT_VIEW_SIZE: u16 = 11;

/// Constructs an [`Observation`] filled with safe defaults for every field
/// that the external dataset does not provide.
///
/// All agent/drone fields (`altitude`, `battery`, `heading`, etc.) are zeroed
/// or set to neutral values; the caller supplies only the fields that the
/// source dataset actually contains.
pub(crate) fn build_default_obs(
    position: (u16, u16),
    health: f32,
    stamina: f32,
    view_size: u16,
    day_phase: u8,
    task_progress: Vec<f32>,
) -> Observation {
    let num_tiles = (view_size as usize) * (view_size as usize);
    Observation {
        grid_view: vec![TileObservation::default(); num_tiles],
        view_width: view_size,
        view_height: view_size,
        inventory: InventoryObservation {
            slots: vec![(OBS_EMPTY_SLOT_ITEM, 0); 10],
        },
        health,
        stamina,
        position,
        messages: vec![],
        day_phase,
        task_progress,
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

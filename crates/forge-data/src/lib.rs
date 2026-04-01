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

pub mod generator;
pub mod loader;
pub mod maze;
pub mod minerl;
pub mod minari;

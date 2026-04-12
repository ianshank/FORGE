//! ONNX Runtime backend for MuZero latent forward model.
//!
//! This module provides an [`OnnxMuZeroModel`] that loads three ONNX models
//! (representation, dynamics, prediction) and implements the
//! [`LatentForwardModel`] trait for production deployment.
//!
//! # Feature flag
//!
//! This module is only available when the `onnx` feature is enabled.
//! Add `ort` to your dependencies and compile with `--features onnx`.

use std::path::Path;
use std::sync::Mutex;
use anyhow::{Context, Result};

use ort::session::Session;

use super::model::{LatentForwardModel, LatentInferenceOutput};
use super::state::LatentState;

/// Configuration for the ONNX MuZero model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxModelConfig {
    /// Path to the representation network ONNX file.
    pub representation_path: String,
    /// Path to the dynamics network ONNX file.
    pub dynamics_path: String,
    /// Path to the prediction network ONNX file.
    pub prediction_path: String,
    /// Number of discrete actions.
    pub action_space_size: u32,
    /// Dimensionality of the latent state.
    pub latent_dim: usize,
    /// Number of inter-op threads for ONNX Runtime.
    pub num_threads: usize,
}

impl Default for OnnxModelConfig {
    fn default() -> Self {
        Self {
            representation_path: "representation.onnx".to_string(),
            dynamics_path: "dynamics.onnx".to_string(),
            prediction_path: "prediction.onnx".to_string(),
            action_space_size: 75,
            latent_dim: 256,
            num_threads: 1,
        }
    }
}

/// ONNX Runtime-backed MuZero model for production inference.
///
/// Loads three ONNX models and provides the [`LatentForwardModel`] interface
/// for latent MCTS search. Each model runs in its own ONNX session.
///
/// Sessions are wrapped in `Mutex` because `ort` v2's `Session::run` requires
/// `&mut self`, while [`LatentForwardModel`] takes `&self`.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "onnx")]
/// # {
/// use forge_agent::latent_mcts::onnx_model::{OnnxMuZeroModel, OnnxModelConfig};
///
/// let config = OnnxModelConfig {
///     representation_path: "models/representation.onnx".into(),
///     dynamics_path: "models/dynamics.onnx".into(),
///     prediction_path: "models/prediction.onnx".into(),
///     action_space_size: 75,
///     latent_dim: 256,
///     num_threads: 1,
/// };
/// let model = OnnxMuZeroModel::load(config).expect("Failed to load ONNX models");
/// # }
/// ```
pub struct OnnxMuZeroModel {
    // Model configuration (paths, dimensions, threading).
    config: OnnxModelConfig,
    // ONNX session for the representation network (observation → latent state).
    representation: Mutex<Session>,
    // ONNX session for the dynamics network (latent + action → next latent + reward).
    dynamics: Mutex<Session>,
    // ONNX session for the prediction network (latent → policy + value).
    prediction: Mutex<Session>,
}

impl OnnxMuZeroModel {
    /// Load ONNX models from the configured paths.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the ONNX files cannot be loaded.
    pub fn load(config: OnnxModelConfig) -> Result<Self, ort::Error> {
        let rep = Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.representation_path)?;

        let dyn_ = Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.dynamics_path)?;

        let pred = Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.prediction_path)?;

        Ok(Self {
            config,
            representation: Mutex::new(rep),
            dynamics: Mutex::new(dyn_),
            prediction: Mutex::new(pred),
        })
    }

    /// Check that all ONNX model files exist on disk.
    ///
    /// Returns `true` if the representation, dynamics, and prediction
    /// ONNX files all exist at the paths specified in `config`.
    /// Use this before calling [`Self::load`] to provide a friendlier
    /// error message when files are missing.
    pub fn validate_paths(config: &OnnxModelConfig) -> bool {
        Path::new(&config.representation_path).exists()
            && Path::new(&config.dynamics_path).exists()
            && Path::new(&config.prediction_path).exists()
    }

    /// Helper: create a 2D ONNX input DynValue from a flat Vec<f32>.
    ///
    /// Uses `(shape, Vec<T>)` tuple constructor to avoid ndarray version
    /// conflicts between the workspace `ndarray 0.16` and ort's `ndarray 0.15`.
    fn make_input(data: Vec<f32>, cols: usize) -> Result<ort::value::DynValue> {
        Ok(ort::value::Value::from_array(([1usize, cols], data))
            .context("failed to create ONNX tensor")?
            .into())
    }

    /// Helper: extract a flat Vec<f32> from an ONNX output DynValue.
    fn extract_f32(value: &ort::value::DynValue) -> Result<Vec<f32>> {
        let (_shape, slice) = value
            .try_extract_tensor::<f32>()
            .context("failed to extract f32 tensor")?;
        Ok(slice.iter().copied().collect())
    }

    /// Run representation network and return latent data.
    fn run_representation(&self, observation: &[f32]) -> Result<Vec<f32>> {
        let obs_value = Self::make_input(observation.to_vec(), observation.len())?;
        let mut session = self
            .representation
            .lock()
            .map_err(|_| anyhow::anyhow!("representation lock poisoned"))?;
        let outputs = session
            .run(ort::inputs![obs_value])
            .context("Representation inference failed")?;
        Self::extract_f32(&outputs[0])
    }

    /// Run prediction network and return (policy_logits, value).
    fn run_prediction(&self, latent_data: Vec<f32>) -> Result<(Vec<f32>, f32)> {
        let latent_value = Self::make_input(latent_data, self.config.latent_dim)?;
        let mut session = self
            .prediction
            .lock()
            .map_err(|_| anyhow::anyhow!("prediction lock poisoned"))?;
        let outputs = session
            .run(ort::inputs![latent_value])
            .context("Prediction inference failed")?;
        let policy_logits = Self::extract_f32(&outputs[0])?;
        let value = Self::extract_f32(&outputs[1])?;
        Ok((policy_logits, value.first().copied().unwrap_or(0.0)))
    }

    /// Run dynamics network and return (next_latent_data, reward).
    fn run_dynamics(&self, dyn_input: Vec<f32>, input_dim: usize) -> Result<(Vec<f32>, f32)> {
        let dyn_value = Self::make_input(dyn_input, input_dim)?;
        let mut session = self
            .dynamics
            .lock()
            .map_err(|_| anyhow::anyhow!("dynamics lock poisoned"))?;
        let outputs = session
            .run(ort::inputs![dyn_value])
            .context("Dynamics inference failed")?;
        let next_latent = Self::extract_f32(&outputs[0])?;
        let reward = Self::extract_f32(&outputs[1])?;
        Ok((next_latent, reward.first().copied().unwrap_or(0.0)))
    }
}

impl LatentForwardModel for OnnxMuZeroModel {
    fn initial_inference(&self, observation: &[f32]) -> Result<LatentInferenceOutput> {
        let latent_data = self.run_representation(observation)?;
        let latent_state = LatentState::new(latent_data.clone());

        let (policy_logits, value) = self.run_prediction(latent_data)?;

        Ok(LatentInferenceOutput {
            latent_state,
            reward: 0.0,
            policy_logits,
            value,
        })
    }

    fn recurrent_inference(&self, state: &LatentState, action: u32) -> Result<LatentInferenceOutput> {
        // Build one-hot action
        let mut action_oh = vec![0.0f32; self.config.action_space_size as usize];
        if (action as usize) < action_oh.len() {
            action_oh[action as usize] = 1.0;
        }

        // Concatenate latent + action for dynamics input
        let mut dyn_input = state.data.clone();
        dyn_input.extend_from_slice(&action_oh);
        let input_dim = self.config.latent_dim + self.config.action_space_size as usize;

        let (next_latent_data, reward) = self.run_dynamics(dyn_input, input_dim)?;
        let next_state = LatentState::new(next_latent_data.clone());

        let (policy_logits, value) = self.run_prediction(next_latent_data)?;

        Ok(LatentInferenceOutput {
            latent_state: next_state,
            reward,
            policy_logits,
            value,
        })
    }

    fn action_space_size(&self) -> u32 {
        self.config.action_space_size
    }
}

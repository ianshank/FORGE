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
    config: OnnxModelConfig,
    representation: ort::Session,
    dynamics: ort::Session,
    prediction: ort::Session,
}

impl OnnxMuZeroModel {
    /// Load ONNX models from the configured paths.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the ONNX files cannot be loaded.
    pub fn load(config: OnnxModelConfig) -> Result<Self, ort::Error> {
        let rep = ort::Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.representation_path)?;

        let dyn_ = ort::Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.dynamics_path)?;

        let pred = ort::Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.prediction_path)?;

        Ok(Self {
            config,
            representation: rep,
            dynamics: dyn_,
            prediction: pred,
        })
    }

    /// Check that all ONNX model files exist.
    pub fn validate_paths(config: &OnnxModelConfig) -> bool {
        Path::new(&config.representation_path).exists()
            && Path::new(&config.dynamics_path).exists()
            && Path::new(&config.prediction_path).exists()
    }
}

impl LatentForwardModel for OnnxMuZeroModel {
    fn initial_inference(&self, observation: &[f32]) -> LatentInferenceOutput {
        let obs_dim = observation.len();

        // Run representation network
        let obs_array =
            ndarray::Array2::from_shape_vec((1, obs_dim), observation.to_vec()).unwrap();
        let rep_outputs = self
            .representation
            .run(ort::inputs![obs_array].unwrap())
            .expect("Representation inference failed");

        let latent_data: Vec<f32> = rep_outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract latent tensor")
            .iter()
            .copied()
            .collect();
        let latent_state = LatentState::new(latent_data.clone());

        // Run prediction network
        let latent_array =
            ndarray::Array2::from_shape_vec((1, self.config.latent_dim), latent_data).unwrap();
        let pred_outputs = self
            .prediction
            .run(ort::inputs![latent_array].unwrap())
            .expect("Prediction inference failed");

        let policy_logits: Vec<f32> = pred_outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract policy tensor")
            .iter()
            .copied()
            .collect();

        let value_raw: Vec<f32> = pred_outputs[1]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract value tensor")
            .iter()
            .copied()
            .collect();

        LatentInferenceOutput {
            latent_state,
            reward: 0.0,
            policy_logits,
            value: value_raw.first().copied().unwrap_or(0.0),
        }
    }

    fn recurrent_inference(&self, state: &LatentState, action: u32) -> LatentInferenceOutput {
        // Build one-hot action
        let mut action_oh = vec![0.0f32; self.config.action_space_size as usize];
        if (action as usize) < action_oh.len() {
            action_oh[action as usize] = 1.0;
        }

        // Concatenate latent + action for dynamics input
        let mut dyn_input = state.data.clone();
        dyn_input.extend_from_slice(&action_oh);
        let input_dim = self.config.latent_dim + self.config.action_space_size as usize;
        let dyn_array = ndarray::Array2::from_shape_vec((1, input_dim), dyn_input).unwrap();

        let dyn_outputs = self
            .dynamics
            .run(ort::inputs![dyn_array].unwrap())
            .expect("Dynamics inference failed");

        let next_latent_data: Vec<f32> = dyn_outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract next latent")
            .iter()
            .copied()
            .collect();

        let reward_raw: Vec<f32> = dyn_outputs[1]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract reward")
            .iter()
            .copied()
            .collect();

        let next_state = LatentState::new(next_latent_data.clone());

        // Run prediction on next latent
        let pred_array =
            ndarray::Array2::from_shape_vec((1, self.config.latent_dim), next_latent_data).unwrap();
        let pred_outputs = self
            .prediction
            .run(ort::inputs![pred_array].unwrap())
            .expect("Prediction inference failed");

        let policy_logits: Vec<f32> = pred_outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract policy")
            .iter()
            .copied()
            .collect();

        let value_raw: Vec<f32> = pred_outputs[1]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract value")
            .iter()
            .copied()
            .collect();

        LatentInferenceOutput {
            latent_state: next_state,
            reward: reward_raw.first().copied().unwrap_or(0.0),
            policy_logits,
            value: value_raw.first().copied().unwrap_or(0.0),
        }
    }

    fn action_space_size(&self) -> u32 {
        self.config.action_space_size
    }
}

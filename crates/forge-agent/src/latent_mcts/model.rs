//! Latent forward model trait for MuZero planning.
//!
//! The [`LatentForwardModel`] trait abstracts neural network inference so
//! the MCTS search can operate with different backends (ONNX, PyTorch, stubs).

use anyhow::Result;

use super::state::LatentState;

/// Output of a MuZero inference step (initial or recurrent).
#[derive(Clone, Debug)]
pub struct LatentInferenceOutput {
    /// The latent state vector.
    pub latent_state: LatentState,
    /// Predicted reward (0.0 for initial inference).
    pub reward: f32,
    /// Policy logits (unnormalized) over the action space.
    pub policy_logits: Vec<f32>,
    /// Scalar value estimate.
    pub value: f32,
}

/// A forward model operating on latent states for MuZero MCTS.
///
/// This trait abstracts the three MuZero network functions:
///
/// - **Initial inference** (representation + prediction):
///   raw observation → latent state + policy + value
///
/// - **Recurrent inference** (dynamics + prediction):
///   (latent state, action) → next latent state + reward + policy + value
///
/// Implementations may use ONNX Runtime, direct PyTorch FFI, or
/// mock/stub models for testing.
pub trait LatentForwardModel: Send + Sync {
    /// Run initial inference: encode an observation into latent space
    /// and predict policy + value.
    ///
    /// # Arguments
    ///
    /// * `observation` - Flat observation vector (e.g., 920 floats for FORGE drone).
    fn initial_inference(&self, observation: &[f32]) -> Result<LatentInferenceOutput>;

    /// Run recurrent inference: predict the next latent state, reward,
    /// policy, and value given a current latent state and action.
    ///
    /// # Arguments
    ///
    /// * `state` - Current latent state from a previous inference call.
    /// * `action` - Discrete action index.
    fn recurrent_inference(&self, state: &LatentState, action: u32) -> Result<LatentInferenceOutput>;

    /// Returns the total number of discrete actions.
    fn action_space_size(&self) -> u32;
}

/// A stub latent forward model for testing.
///
/// Returns fixed-size outputs with uniform policy priors and zero values.
/// Useful for verifying MCTS search logic without a trained model.
#[derive(Debug, Clone)]
pub struct StubLatentModel {
    /// Number of discrete actions.
    action_space: u32,
    /// Dimensionality of the latent state.
    latent_dim: usize,
}

impl StubLatentModel {
    /// Creates a new stub model.
    pub fn new(action_space: u32, latent_dim: usize) -> Self {
        Self {
            action_space,
            latent_dim,
        }
    }
}

impl LatentForwardModel for StubLatentModel {
    fn initial_inference(&self, _observation: &[f32]) -> Result<LatentInferenceOutput> {
        let uniform_prior = 1.0 / self.action_space as f32;
        Ok(LatentInferenceOutput {
            latent_state: LatentState::zeros(self.latent_dim),
            reward: 0.0,
            policy_logits: vec![uniform_prior; self.action_space as usize],
            value: 0.0,
        })
    }

    fn recurrent_inference(&self, _state: &LatentState, _action: u32) -> Result<LatentInferenceOutput> {
        let uniform_prior = 1.0 / self.action_space as f32;
        Ok(LatentInferenceOutput {
            latent_state: LatentState::zeros(self.latent_dim),
            reward: 0.0,
            policy_logits: vec![uniform_prior; self.action_space as usize],
            value: 0.0,
        })
    }

    fn action_space_size(&self) -> u32 {
        self.action_space
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_model_initial_inference() {
        let model = StubLatentModel::new(10, 64);
        let obs = vec![0.0; 100];
        let output = model.initial_inference(&obs).unwrap();

        assert_eq!(output.latent_state.dim(), 64);
        assert_eq!(output.policy_logits.len(), 10);
        assert!((output.reward - 0.0).abs() < f32::EPSILON);
        assert!((output.value - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stub_model_recurrent_inference() {
        let model = StubLatentModel::new(5, 32);
        let state = LatentState::zeros(32);
        let output = model.recurrent_inference(&state, 2).unwrap();

        assert_eq!(output.latent_state.dim(), 32);
        assert_eq!(output.policy_logits.len(), 5);
    }

    #[test]
    fn test_stub_model_action_space() {
        let model = StubLatentModel::new(75, 256);
        assert_eq!(model.action_space_size(), 75);
    }

    #[test]
    fn test_inference_output_clone() {
        let output = LatentInferenceOutput {
            latent_state: LatentState::new(vec![1.0, 2.0]),
            reward: 0.5,
            policy_logits: vec![0.1, 0.9],
            value: 0.7,
        };
        let cloned = output.clone();
        assert_eq!(cloned.reward, 0.5);
        assert_eq!(cloned.value, 0.7);
        assert_eq!(cloned.policy_logits, vec![0.1, 0.9]);
    }
}

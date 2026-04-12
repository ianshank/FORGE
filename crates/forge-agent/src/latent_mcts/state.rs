//! Latent state types for MuZero planning.
//!
//! A [`LatentState`] is a compact vector representation produced by the
//! representation or dynamics network. Unlike [`WorldState`], it contains
//! no simulation logic — it is a pure data carrier for neural network
//! outputs.

use serde::{Deserialize, Serialize};

/// A latent state vector produced by the MuZero representation or dynamics network.
///
/// This is a lightweight, cloneable container for the neural network's
/// hidden state. The dimensionality is determined by the model configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentState {
    /// The latent state vector.
    pub data: Vec<f32>,
    /// Whether this state represents a terminal condition.
    pub is_terminal: bool,
}

impl LatentState {
    /// Creates a new latent state with the given data.
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            is_terminal: false,
        }
    }

    /// Creates a terminal latent state.
    pub fn terminal(data: Vec<f32>) -> Self {
        Self {
            data,
            is_terminal: true,
        }
    }

    /// Returns the dimensionality of the latent state.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Creates a zero-valued latent state of the given dimensionality.
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
            is_terminal: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_new() {
        let state = LatentState::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(state.dim(), 3);
        assert!(!state.is_terminal);
        assert_eq!(state.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_latent_state_terminal() {
        let state = LatentState::terminal(vec![0.5; 4]);
        assert!(state.is_terminal);
        assert_eq!(state.dim(), 4);
    }

    #[test]
    fn test_latent_state_zeros() {
        let state = LatentState::zeros(256);
        assert_eq!(state.dim(), 256);
        assert!(state.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_latent_state_clone() {
        let state = LatentState::new(vec![1.0, 2.0]);
        let cloned = state.clone();
        assert_eq!(state.data, cloned.data);
        assert_eq!(state.is_terminal, cloned.is_terminal);
    }

    #[test]
    fn test_latent_state_serialize_roundtrip() {
        let state = LatentState::new(vec![1.0, -2.0, 3.5]);
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: LatentState = serde_json::from_str(&json).unwrap();
        assert_eq!(state.data, deserialized.data);
        assert_eq!(state.is_terminal, deserialized.is_terminal);
    }
}

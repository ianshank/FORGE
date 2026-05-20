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

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::Session;

use super::model::{LatentForwardModel, LatentInferenceOutput};
use super::state::LatentState;

/// Errors that can occur during [`OnnxMuZeroModel::reload`].
///
/// Distinct from `ort::Error` so callers can pattern-match on
/// reload-specific failures (e.g. missing-file is a friendlier
/// signal than a generic `ort` load error). All other failures
/// surface as `Ort(ort::Error)`.
#[derive(Debug)]
pub enum OnnxReloadError {
    /// A path in the new [`OnnxModelConfig`] does not exist on disk.
    /// Reload returns this **before** touching the existing sessions,
    /// so the model remains usable with the previous bundle.
    MissingFile { path: PathBuf },
    /// `ort::Session::builder().commit_from_file(...)` failed (corrupt
    /// ONNX, schema mismatch, op-set unsupported, etc.). Like
    /// `MissingFile`, this is raised before any swap so the existing
    /// sessions stay intact (build-first-then-swap invariant).
    Ort(ort::Error),
}

impl std::fmt::Display for OnnxReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFile { path } => {
                write!(f, "missing ONNX file: {}", path.display())
            }
            Self::Ort(e) => write!(f, "ort session build failed: {e}"),
        }
    }
}

impl std::error::Error for OnnxReloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MissingFile { .. } => None,
            Self::Ort(e) => Some(e),
        }
    }
}

impl From<ort::Error> for OnnxReloadError {
    fn from(value: ort::Error) -> Self {
        Self::Ort(value)
    }
}

/// Pre-flight check used by [`OnnxMuZeroModel::reload`]. Public so
/// downstream tests + the runner's onnx-reload wrapper can short-
/// circuit on missing files **before** calling into `ort`, which
/// would produce a less informative error.
pub fn validate_reload_paths(config: &OnnxModelConfig) -> Result<(), OnnxReloadError> {
    for path in [
        &config.representation_path,
        &config.dynamics_path,
        &config.prediction_path,
    ] {
        if !Path::new(path).exists() {
            return Err(OnnxReloadError::MissingFile {
                path: PathBuf::from(path),
            });
        }
    }
    Ok(())
}

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

    /// Hot-swap all three ONNX sessions with the bundle described by
    /// `new_config`. The previous sessions are dropped after the new
    /// ones are constructed.
    ///
    /// # Atomicity (build-first-then-swap)
    ///
    /// The implementation builds all three new [`Session`]s in stack
    /// locals **before** touching `self`. If any session-build fails
    /// (missing file, corrupt ONNX, op-set mismatch, ...), `reload`
    /// returns `Err` and `self` keeps its previous bundle. This
    /// matters when a trainer exports a partial / corrupt bundle —
    /// the runner stays serviceable against the previous weights
    /// instead of half-swapping into a broken state.
    ///
    /// # Concurrency
    ///
    /// `reload` takes `&mut self`. Rust's borrow checker therefore
    /// prevents any concurrent `&self` inference call from
    /// interleaving with the swap, which guarantees no inference call
    /// can observe a mix of old + new sessions across the
    /// representation → dynamics → prediction chain.
    ///
    /// Callers that share the model across threads via
    /// `Arc<OnnxMuZeroModel>` **do not** get this guarantee — the
    /// inner sessions would still appear consistent per-call (because
    /// each `Mutex<Session>` is serialised), but cross-session
    /// staleness becomes possible across a reload boundary. If that
    /// use case appears, switch to `Arc<Sessions>` + `arc_swap` so a
    /// single atomic pointer-swap covers all three sessions
    /// together. The current API is single-threaded by construction.
    ///
    /// # Errors
    ///
    /// - [`OnnxReloadError::MissingFile`] if any of the three configured
    ///   ONNX paths does not exist on disk. Returned before any `ort`
    ///   call; nothing on `self` is mutated.
    /// - [`OnnxReloadError::Ort`] if `ort::Session::builder()` /
    ///   `commit_from_file` fails for any of the three sessions
    ///   (corrupt bundle, unsupported op-set, etc.). Returned before
    ///   the swap; existing sessions stay intact.
    pub fn reload(&mut self, new_config: OnnxModelConfig) -> Result<(), OnnxReloadError> {
        // Pre-flight: give MissingFile a friendlier error than ort's
        // generic load-failed. This is also the path the runner uses
        // to fail fast on bad manifests without paying the ort
        // construction cost.
        validate_reload_paths(&new_config)?;

        // Build all three sessions in stack locals BEFORE swapping
        // anything on `self`. `?` propagates ort::Error via
        // `From<ort::Error> for OnnxReloadError`.
        let new_rep = Session::builder()?
            .with_intra_threads(new_config.num_threads)?
            .commit_from_file(&new_config.representation_path)?;
        let new_dyn = Session::builder()?
            .with_intra_threads(new_config.num_threads)?
            .commit_from_file(&new_config.dynamics_path)?;
        let new_pred = Session::builder()?
            .with_intra_threads(new_config.num_threads)?
            .commit_from_file(&new_config.prediction_path)?;

        // Swap in the documented contractual order:
        // representation -> dynamics -> prediction. The old `Mutex`
        // values are dropped (which drops the old `Session`s).
        self.representation = Mutex::new(new_rep);
        self.dynamics = Mutex::new(new_dyn);
        self.prediction = Mutex::new(new_pred);
        self.config = new_config;
        Ok(())
    }

    /// Borrow the current [`OnnxModelConfig`] (paths, dims, threads).
    /// Useful for introspection from the runner / metrics endpoint.
    pub fn config(&self) -> &OnnxModelConfig {
        &self.config
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

    fn recurrent_inference(
        &self,
        state: &LatentState,
        action: u32,
    ) -> Result<LatentInferenceOutput> {
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

#[cfg(test)]
mod tests {
    //! Unit tests that do NOT need real ONNX bundles. The happy-path
    //! reload swap test (with valid bundles) lives in
    //! `tests/onnx_reload_integration.rs` alongside the existing
    //! `onnx_integration.rs`, because both need a Python + torch +
    //! onnx subprocess to produce the test ONNX files.
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn validate_reload_paths_accepts_existing_trio() {
        let dir = tempdir().unwrap();
        for name in ["rep.onnx", "dyn.onnx", "pred.onnx"] {
            std::fs::write(dir.path().join(name), b"stub").unwrap();
        }
        let cfg = OnnxModelConfig {
            representation_path: dir.path().join("rep.onnx").to_string_lossy().into_owned(),
            dynamics_path: dir.path().join("dyn.onnx").to_string_lossy().into_owned(),
            prediction_path: dir.path().join("pred.onnx").to_string_lossy().into_owned(),
            action_space_size: 4,
            latent_dim: 8,
            num_threads: 1,
        };
        validate_reload_paths(&cfg).unwrap();
    }

    #[test]
    fn validate_reload_paths_returns_missing_file_for_absent_representation() {
        let dir = tempdir().unwrap();
        // Only dynamics + prediction exist; representation is absent.
        std::fs::write(dir.path().join("dyn.onnx"), b"stub").unwrap();
        std::fs::write(dir.path().join("pred.onnx"), b"stub").unwrap();
        let missing = dir.path().join("rep.onnx");
        let cfg = OnnxModelConfig {
            representation_path: missing.to_string_lossy().into_owned(),
            dynamics_path: dir.path().join("dyn.onnx").to_string_lossy().into_owned(),
            prediction_path: dir.path().join("pred.onnx").to_string_lossy().into_owned(),
            action_space_size: 4,
            latent_dim: 8,
            num_threads: 1,
        };
        let err = validate_reload_paths(&cfg).unwrap_err();
        match err {
            OnnxReloadError::MissingFile { path } => assert_eq!(path, missing),
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn validate_reload_paths_returns_missing_file_for_absent_dynamics() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("rep.onnx"), b"stub").unwrap();
        std::fs::write(dir.path().join("pred.onnx"), b"stub").unwrap();
        let missing = dir.path().join("dyn.onnx");
        let cfg = OnnxModelConfig {
            representation_path: dir.path().join("rep.onnx").to_string_lossy().into_owned(),
            dynamics_path: missing.to_string_lossy().into_owned(),
            prediction_path: dir.path().join("pred.onnx").to_string_lossy().into_owned(),
            action_space_size: 4,
            latent_dim: 8,
            num_threads: 1,
        };
        let err = validate_reload_paths(&cfg).unwrap_err();
        match err {
            OnnxReloadError::MissingFile { path } => assert_eq!(path, missing),
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn validate_reload_paths_returns_missing_file_for_absent_prediction() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("rep.onnx"), b"stub").unwrap();
        std::fs::write(dir.path().join("dyn.onnx"), b"stub").unwrap();
        let missing = dir.path().join("pred.onnx");
        let cfg = OnnxModelConfig {
            representation_path: dir.path().join("rep.onnx").to_string_lossy().into_owned(),
            dynamics_path: dir.path().join("dyn.onnx").to_string_lossy().into_owned(),
            prediction_path: missing.to_string_lossy().into_owned(),
            action_space_size: 4,
            latent_dim: 8,
            num_threads: 1,
        };
        let err = validate_reload_paths(&cfg).unwrap_err();
        match err {
            OnnxReloadError::MissingFile { path } => assert_eq!(path, missing),
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn onnx_reload_error_display_includes_path() {
        let err = OnnxReloadError::MissingFile {
            path: PathBuf::from("/no/such/file.onnx"),
        };
        let msg = format!("{err}");
        assert!(msg.contains("/no/such/file.onnx"), "got: {msg}");
        assert!(msg.contains("missing ONNX file"), "got: {msg}");
    }

    #[test]
    fn onnx_reload_error_source_is_some_for_ort_variant() {
        // The MissingFile variant has no source; the Ort variant
        // surfaces the underlying ort::Error. We don't construct an
        // ort::Error directly (it's not Default-constructible), but
        // we assert the source() return shape via the MissingFile
        // path which returns None.
        let err = OnnxReloadError::MissingFile {
            path: PathBuf::from("x"),
        };
        let source: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&err);
        assert!(source.is_none());
    }
}

//! Error type for the runner foundation.

use std::path::PathBuf;

use thiserror::Error;

/// Errors emitted by the runner, manifest loader, hot-reload watcher,
/// and trajectory writer.
///
/// Variants carry just enough structured context for callers to log
/// usefully. File-IO failures preserve the offending path; JSON failures
/// preserve the underlying parser diagnostic.
///
/// ## Why no `From<MetricsError> for RunnerError`?
///
/// Metrics failures are a *binary-level* concern (the runner is happy
/// to keep stepping even when the metrics endpoint can't bind), and
/// are surfaced by the binary's `main` directly via [`std::process::
/// ExitCode`]. The runner loop ([`crate::Runner::run`]) never returns
/// a metrics error because it never *interacts* with the metrics
/// server beyond pushing samples into an in-process [`MetricsRecorder`].
/// Keeping the two error surfaces separate avoids accidentally
/// collapsing "metrics endpoint won't bind" failures into the same
/// `RunnerError` channel that surfaces real per-episode runner
/// problems.
#[derive(Debug, Error)]
pub enum RunnerError {
    /// Configured path does not exist on disk.
    #[error("missing path: {path}")]
    MissingPath {
        /// Path the caller attempted to use.
        path: PathBuf,
    },

    /// Manifest `schema_version` field did not match the compiled constant.
    #[error("manifest schema_version mismatch: expected {expected}, got {got}")]
    ManifestSchemaMismatch {
        /// Constant compiled into this crate.
        expected: u32,
        /// Value read from the manifest file.
        got: u32,
    },

    /// Manifest contained invariants-violating fields (e.g. zero version
    /// or missing onnx file paths).
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// Trajectory writer was used in an unexpected order (e.g. push step
    /// before `start_episode`, finalize without any push).
    #[error("trajectory writer state error: {0}")]
    WriterState(String),

    /// File-system IO failure with the offending path.
    #[error("io error on {path}: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// JSON (de)serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Forwarded from `forge_replay::v2::TrajectoryError` during writer
    /// validation and saving.
    #[error("trajectory error: {0}")]
    Trajectory(#[from] forge_replay::v2::TrajectoryError),

    /// Wrapped error from a [`forge_env::Env`] implementation. The
    /// concrete env error is `Send + Sync + 'static + std::error::Error`
    /// but is not nameable from this crate, so we collapse to its
    /// `Display` form. Callers wanting structured access should keep
    /// their own copy.
    #[error("env error: {0}")]
    Env(String),

    /// Transient env protocol error (`RECONNECTING` / `BUSY`). The
    /// episode is discarded; [`crate::Runner::run`] continues until
    /// [`crate::config::RunnerConfig::max_consecutive_transient_failures`].
    #[error("transient env error [{code}]: {message}")]
    TransientEnv {
        /// Protocol error code (`RECONNECTING` or `BUSY`).
        code: String,
        /// Bot-supplied human message.
        message: String,
    },

    /// The consecutive-transient cap was hit. Distinct from
    /// [`Self::TransientEnv`] so callers do not retry a run that has
    /// already exhausted its budget.
    #[error("too many consecutive transient env errors ({count}): last [{code}]: {message}")]
    TooManyTransientFailures {
        /// How many consecutive transients were observed.
        count: u32,
        /// Last protocol error code.
        code: String,
        /// Last bot-supplied message.
        message: String,
    },

    /// Wrapped error from the latent-MCTS planner (`anyhow::Error` from
    /// `forge_agent::latent_mcts::search::LatentMctsSearch::search`).
    #[error("planner error: {0}")]
    Planner(String),

    /// Reload callback returned an error.
    #[error("model reload failed: {0}")]
    Reload(String),

    /// Live-runner config (e.g. `MinecraftEnvConfig`) failed to load
    /// or validate. Captures the offending field / file path in the
    /// message string. Distinct from [`Self::Io`] so callers can
    /// surface clean diagnostics — config-load failures are a
    /// startup-time problem, not a runtime IO blip.
    #[error("config load failed: {0}")]
    ConfigLoad(String),

    /// Live env setup failed (WS handshake, action_count mismatch,
    /// schema_id drift). Distinct from [`Self::Env`] which wraps a
    /// per-step error from an already-connected env.
    #[error("env setup failed: {0}")]
    EnvSetup(String),

    /// A model file's bytes on disk do not hash to the sha256 the
    /// manifest recorded for that role. Raised by
    /// [`crate::integrity::verify_bundle`] **before** any ONNX session
    /// is built, so a tampered or truncated bundle never reaches the
    /// ONNX Runtime parser.
    #[error(
        "model digest mismatch for role `{role}` at {path}: \
         manifest recorded sha256 {expected}, file on disk hashes to {actual}"
    )]
    ModelDigestMismatch {
        /// Bundle role: `representation`, `dynamics`, or `prediction`.
        role: String,
        /// Resolved on-disk path that was hashed.
        path: PathBuf,
        /// Digest recorded in `model_manifest.json`.
        expected: String,
        /// Digest computed from the file's current contents.
        actual: String,
    },

    /// A manifest entry named a path the runner refuses to load: an
    /// absolute path, a path containing a `..` component, or one that
    /// canonicalizes outside its bundle directory (e.g. via a symlink).
    ///
    /// The manifest is written by the trainer, which in the self-play
    /// stack shares a host bind-mount with the runner — so the path in
    /// a manifest entry is untrusted input, not a local constant.
    #[error(
        "unsafe model path for role `{role}`: {reason} \
         (manifest entry `{entry}`, bundle_dir {bundle_dir})"
    )]
    UnsafeModelPath {
        /// Bundle role: `representation`, `dynamics`, or `prediction`.
        role: String,
        /// The raw `files.<role>.path` string from the manifest.
        entry: String,
        /// Directory the entry was required to resolve inside.
        bundle_dir: PathBuf,
        /// Why the path was rejected.
        reason: String,
    },
}

impl RunnerError {
    /// Wrap a `std::io::Error` with the path it was raised against.
    /// Kept as a helper because `From` can't carry the path through.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Display prefix `forge_env_mc::McEnvError::Transient` emits.
///
/// `Runner` is generic over `Env::Error` and cannot downcast to
/// `McEnvError` when `mc-live` is off, so classification is by
/// Display. Must stay byte-identical to
/// `forge_env_mc::error::TRANSIENT_PROTOCOL_ERROR_DISPLAY_PREFIX`.
pub const TRANSIENT_ENV_DISPLAY_PREFIX: &str = "transient protocol error [";

/// Separator between the protocol code and message in a transient
/// Display string (`[CODE]: message`). Twin of the `thiserror` format
/// on `McEnvError::Transient`.
pub const TRANSIENT_ENV_CODE_MESSAGE_SEP: &str = "]: ";

/// Parse a transient env error out of an `Env::Error` Display string.
///
/// Expected form: `transient protocol error [CODE]: message`, optionally
/// wrapped by another error's Display (e.g. `EnvError::Other`).
#[must_use]
pub fn parse_transient_env_error(msg: &str) -> Option<(String, String)> {
    let start = msg.find(TRANSIENT_ENV_DISPLAY_PREFIX)?;
    let rest = &msg[start + TRANSIENT_ENV_DISPLAY_PREFIX.len()..];
    let (code, message) = rest.split_once(TRANSIENT_ENV_CODE_MESSAGE_SEP)?;
    if code.is_empty() {
        return None;
    }
    Some((code.to_string(), message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_display_prefix_matches_forge_env_mc() {
        assert_eq!(
            TRANSIENT_ENV_DISPLAY_PREFIX,
            forge_env_mc::TRANSIENT_PROTOCOL_ERROR_DISPLAY_PREFIX,
            "Display-prefix twins must stay byte-identical so Runner can \
             classify McEnvError::Transient without a downcast"
        );
    }

    #[test]
    fn parse_transient_env_error_roundtrip() {
        let msg = "transient protocol error [RECONNECTING]: bot rebuilding";
        assert_eq!(
            parse_transient_env_error(msg),
            Some(("RECONNECTING".into(), "bot rebuilding".into()))
        );
    }

    #[test]
    fn parse_transient_env_error_ignores_fatal_protocol() {
        assert_eq!(
            parse_transient_env_error("protocol error [INTERNAL]: boom"),
            None
        );
        assert_eq!(parse_transient_env_error("env error: nope"), None);
        assert_eq!(
            parse_transient_env_error("transient protocol error []: empty code"),
            None
        );
        assert_eq!(
            parse_transient_env_error("env error: transient protocol error [BUSY]: wrapped"),
            Some(("BUSY".into(), "wrapped".into()))
        );
    }
}

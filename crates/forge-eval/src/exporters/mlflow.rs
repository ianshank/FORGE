//! MLflow filesystem-layout exporter — see crate docs §3.x for the
//! produced directory shape. Full implementation lands in Task 4.

use std::path::{Path, PathBuf};

use super::{ExportError, Exporter};
use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

/// MLflow tracking exporter. Writes the
/// `<tracking_uri>/<experiment_id>/<run_id>/{meta.yaml, params, metrics,
/// tags, artifacts, inputs}` tree that `mlflow ui --backend-store-uri
/// <tracking_uri>` reads natively.
#[derive(Debug, Clone)]
pub struct MlflowExporter {
    tracking_uri: PathBuf,
}

impl MlflowExporter {
    /// Construct an exporter rooted at `tracking_uri`. The path is the
    /// argument you'd pass to `mlflow ui --backend-store-uri`.
    pub fn new(tracking_uri: PathBuf) -> Self {
        Self { tracking_uri }
    }
}

impl Exporter for MlflowExporter {
    fn name(&self) -> &'static str {
        "mlflow"
    }

    fn export(
        &self,
        _scorecard: &Scorecard,
        _manifest: &RunManifest,
        _artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        // TODO(Task 4): write meta.yaml, params/, metrics/, tags/,
        // artifacts/, inputs/inputs.yaml — both parent and child runs.
        Err(ExportError::InvalidTarget(format!(
            "MlflowExporter::export not yet implemented (tracking_uri={})",
            self.tracking_uri.display()
        )))
    }
}

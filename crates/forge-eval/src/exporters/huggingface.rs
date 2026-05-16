//! HuggingFace `datasets`-compatible exporter — see crate docs §3.x for
//! the produced `DatasetDict` directory shape. Full implementation lands
//! in Task 5.

use std::path::{Path, PathBuf};

use super::{ExportError, Exporter};
use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

/// HuggingFace `datasets` exporter. Writes a `DatasetDict`-compatible
/// directory at `<huggingface_export_root>/<run_id>/` containing per-tier
/// JSONL splits, an explicit `features` schema, `dataset_dict.json`,
/// `state.json` per split, and an HF-Hub-ready `README.md` dataset card.
#[derive(Debug, Clone)]
pub struct HuggingFaceExporter {
    export_root: PathBuf,
}

impl HuggingFaceExporter {
    /// Construct an exporter rooted at `export_root`. The exporter will
    /// create the `<run_id>` subdirectory underneath it.
    pub fn new(export_root: PathBuf) -> Self {
        Self { export_root }
    }
}

impl Exporter for HuggingFaceExporter {
    fn name(&self) -> &'static str {
        "huggingface"
    }

    fn export(
        &self,
        _scorecard: &Scorecard,
        _manifest: &RunManifest,
        _artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        // TODO(Task 5): write dataset_dict.json, manifest.json,
        // README.md (YAML frontmatter), and per-split
        // (dataset_info.json, state.json, data-NNNNN-of-NNNNN.jsonl).
        Err(ExportError::InvalidTarget(format!(
            "HuggingFaceExporter::export not yet implemented (export_root={})",
            self.export_root.display()
        )))
    }
}

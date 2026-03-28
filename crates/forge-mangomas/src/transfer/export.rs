//! Weight and configuration export for MangoMAS consumption.
//!
//! Provides serialization of sweep results, BDI training data,
//! RSSM sequences, and curriculum state to formats that MangoMAS
//! can consume (JSON configs, serialized datasets).

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::{MangoMasError, MangoMasResult};
use crate::sweep::results::SweepReport;
use crate::transfer::bdi_collector::BdiTrainingData;
use crate::transfer::rssm_adapter::SequenceDataset;

/// Configuration for weight export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightExportConfig {
    /// Output directory for exported files.
    pub output_dir: String,
    /// Whether to export MCTS sweep results.
    pub export_sweep: bool,
    /// Whether to export BDI training data.
    pub export_bdi: bool,
    /// Whether to export RSSM sequence data.
    pub export_rssm: bool,
}

impl Default for WeightExportConfig {
    fn default() -> Self {
        Self {
            output_dir: "exports/mangomas".to_string(),
            export_sweep: true,
            export_bdi: true,
            export_rssm: true,
        }
    }
}

/// Serialized export bundle containing all transferable data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportBundle {
    /// MCTS optimal configuration (from sweep).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sweep_report: Option<SweepReport>,
    /// BDI training dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bdi_data: Option<BdiTrainingData>,
    /// RSSM sequence dataset metadata (actual data exported separately).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rssm_metadata: Option<RssmExportMetadata>,
}

/// Metadata about the exported RSSM dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssmExportMetadata {
    /// State vector dimensionality.
    pub state_dim: usize,
    /// Action space size.
    pub action_dim: u32,
    /// Total number of transitions.
    pub total_transitions: u64,
    /// Number of sequences.
    pub num_sequences: usize,
}

impl From<&SequenceDataset> for RssmExportMetadata {
    fn from(dataset: &SequenceDataset) -> Self {
        Self {
            state_dim: dataset.state_dim,
            action_dim: dataset.action_dim,
            total_transitions: dataset.total_transitions,
            num_sequences: dataset.sequences.len(),
        }
    }
}

/// Serializes an export bundle to JSON.
#[instrument(skip_all)]
pub fn serialize_bundle(bundle: &ExportBundle) -> MangoMasResult<String> {
    serde_json::to_string_pretty(bundle)
        .map_err(|e| MangoMasError::Export(format!("JSON serialization failed: {}", e)))
}

/// Serializes a sweep report to JSON for MangoMAS config import.
#[instrument(skip_all)]
pub fn serialize_sweep_report(report: &SweepReport) -> MangoMasResult<String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| MangoMasError::Export(format!("sweep report serialization failed: {}", e)))
}

/// Serializes BDI training data to JSON.
#[instrument(skip_all)]
pub fn serialize_bdi_data(data: &BdiTrainingData) -> MangoMasResult<String> {
    serde_json::to_string(data)
        .map_err(|e| MangoMasError::Export(format!("BDI data serialization failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_bundle_serde() {
        let bundle = ExportBundle {
            sweep_report: None,
            bdi_data: None,
            rssm_metadata: None,
        };
        let json = serialize_bundle(&bundle).unwrap();
        let deser: ExportBundle = serde_json::from_str(&json).unwrap();
        assert!(deser.sweep_report.is_none());
    }

    #[test]
    fn test_export_config_defaults() {
        let config = WeightExportConfig::default();
        assert!(config.export_sweep);
        assert!(config.export_bdi);
        assert!(config.export_rssm);
    }

    #[test]
    fn test_rssm_metadata_from_dataset() {
        let dataset = SequenceDataset {
            sequences: vec![],
            state_dim: 22,
            action_dim: 56,
            total_transitions: 1000,
        };
        let meta = RssmExportMetadata::from(&dataset);
        assert_eq!(meta.state_dim, 22);
        assert_eq!(meta.total_transitions, 1000);
    }
}

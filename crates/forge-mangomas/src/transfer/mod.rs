//! Weight transfer and pre-training data extraction.
//!
//! Provides BDI intention mapping, constitutional constraint mapping,
//! RSSM sequence extraction, and weight export for MangoMAS.

pub mod bdi_collector;
pub mod constitutional;
pub mod export;
pub mod rssm_adapter;

pub use bdi_collector::{BdiEpisodeCollector, BdiIntentionMapper, BdiTrainingData};
pub use constitutional::{ConstitutionalConstraintMapper, ConstraintViolation};
pub use export::WeightExportConfig;
pub use rssm_adapter::{RssmSequenceBuilder, SequenceDataset};

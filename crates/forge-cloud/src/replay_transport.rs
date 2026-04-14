//! Replay compression, decompression, and batching for transport.
//!
//! Provides utilities for preparing compact replays for network transport
//! between workers and the training coordinator.

use forge_replay::compact::CompactReplay;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::config::ReplayTransportConfig;
use crate::error::{CloudResult, TransportError};

/// Compresses serialized replay bytes using a simple run-length encoding
/// when compression is enabled.
///
/// If compression is disabled in the config, returns the input unchanged.
#[instrument(skip(data, config))]
pub fn compress_replay(data: &[u8], config: &ReplayTransportConfig) -> CloudResult<Vec<u8>> {
    if !config.compression_enabled {
        debug!(size = data.len(), "compression disabled, passing through");
        return Ok(data.to_vec());
    }

    if data.len() > config.max_payload_bytes {
        debug!(
            size = data.len(),
            max = config.max_payload_bytes,
            "payload exceeds max size before compression, attempting compression"
        );
    }

    // Simple RLE compression: [byte, count] pairs for runs of 3+
    // Header byte 0x00 = literal run, 0x01 = RLE run
    let mut compressed = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run_len: usize = 1;
        while i + run_len < data.len() && data[i + run_len] == byte && run_len < 255 {
            run_len += 1;
        }
        if run_len >= 3 {
            compressed.push(0x01); // RLE marker
            compressed.push(byte);
            compressed.push(run_len as u8);
        } else {
            for _ in 0..run_len {
                compressed.push(0x00); // literal marker
                compressed.push(byte);
            }
        }
        i += run_len;
    }

    // Check compressed size against limit (compression can expand data for non-repeating input)
    if compressed.len() > config.max_payload_bytes {
        return Err(TransportError::PayloadTooLarge {
            size: compressed.len(),
            max: config.max_payload_bytes,
        }
        .into());
    }

    debug!(
        original = data.len(),
        compressed = compressed.len(),
        "compressed replay"
    );
    Ok(compressed)
}

/// Decompresses replay bytes that were compressed with [`compress_replay`].
///
/// Returns the original uncompressed bytes.
#[instrument(skip(data))]
pub fn decompress_replay(data: &[u8]) -> CloudResult<Vec<u8>> {
    let mut decompressed = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 1 >= data.len() {
            return Err(TransportError::DeserializationFailed(
                "truncated compressed data".to_string(),
            )
            .into());
        }
        match data[i] {
            0x00 => {
                // Literal byte
                decompressed.push(data[i + 1]);
                i += 2;
            }
            0x01 => {
                // RLE run
                if i + 2 >= data.len() {
                    return Err(TransportError::DeserializationFailed(
                        "truncated RLE data".to_string(),
                    )
                    .into());
                }
                let byte = data[i + 1];
                let count = data[i + 2] as usize;
                for _ in 0..count {
                    decompressed.push(byte);
                }
                i += 3;
            }
            marker => {
                return Err(TransportError::DeserializationFailed(format!(
                    "unknown compression marker: 0x{marker:02x}"
                ))
                .into());
            }
        }
    }

    debug!(
        compressed = data.len(),
        decompressed = decompressed.len(),
        "decompressed replay"
    );
    Ok(decompressed)
}

/// A batch of replays prepared for transport.
///
/// Batching amortizes transport overhead by grouping multiple replays
/// into a single message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBatch {
    /// Replays in this batch.
    pub replays: Vec<CompactReplay>,
    /// Worker ID that produced this batch.
    pub worker_id: String,
    /// Model version used to generate these replays, if any.
    pub model_version: Option<u32>,
    /// ISO 8601 timestamp when the batch was created.
    pub timestamp: String,
}

impl ReplayBatch {
    /// Creates a new empty replay batch.
    #[instrument(skip_all)]
    pub fn new(worker_id: String, model_version: Option<u32>) -> Self {
        let ts = chrono::Utc::now().to_rfc3339();
        debug!(worker_id = %worker_id, "created replay batch");
        Self {
            replays: Vec::new(),
            worker_id,
            model_version,
            timestamp: ts,
        }
    }

    /// Returns the number of replays in this batch.
    pub fn len(&self) -> usize {
        self.replays.len()
    }

    /// Returns whether this batch is empty.
    pub fn is_empty(&self) -> bool {
        self.replays.is_empty()
    }

    /// Adds a replay to this batch.
    #[instrument(skip(self, replay))]
    pub fn push(&mut self, replay: CompactReplay) {
        debug!(batch_size = self.replays.len() + 1, "added replay to batch");
        self.replays.push(replay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReplayTransportConfig;

    fn default_config() -> ReplayTransportConfig {
        ReplayTransportConfig::default()
    }

    fn disabled_compression_config() -> ReplayTransportConfig {
        ReplayTransportConfig {
            compression_enabled: false,
            ..ReplayTransportConfig::default()
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let config = default_config();
        let original = b"hello world, this is a test of compression!";
        let compressed = compress_replay(original, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compress_decompress_empty() {
        let config = default_config();
        let original: &[u8] = b"";
        let compressed = compress_replay(original, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compress_decompress_repeated_bytes() {
        let config = default_config();
        let original = vec![0xAA; 100];
        let compressed = compress_replay(&original, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();
        assert_eq!(decompressed, original);
        // RLE should compress repeated bytes significantly
        assert!(compressed.len() < original.len());
    }

    #[test]
    fn test_compress_disabled() {
        let config = disabled_compression_config();
        let original = b"some data";
        let result = compress_replay(original, &config).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_compress_payload_too_large() {
        let config = ReplayTransportConfig {
            max_payload_bytes: 10,
            ..ReplayTransportConfig::default()
        };
        // Use non-repeating data so RLE cannot shrink it below the limit.
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let result = compress_replay(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_invalid_marker() {
        let data = vec![0xFF, 0x00];
        let result = decompress_replay(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_truncated_literal() {
        let data = vec![0x00]; // Missing byte after literal marker
        let result = decompress_replay(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_truncated_rle() {
        let data = vec![0x01, 0xAA]; // Missing count byte
        let result = decompress_replay(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_batch_new() {
        let batch = ReplayBatch::new("w-001".to_string(), Some(1));
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.worker_id, "w-001");
        assert_eq!(batch.model_version, Some(1));
        assert!(!batch.timestamp.is_empty());
    }

    #[test]
    fn test_replay_batch_push() {
        let mut batch = ReplayBatch::new("w-001".to_string(), None);
        assert!(batch.is_empty());

        let replay = CompactReplay {
            format_version: 1,
            config_hash: 0,
            config: forge_types::config::ForgeConfig::default(),
            seed: 42,
            actions: vec![],
            metadata: forge_replay::compact::ReplayMetadata::default(),
        };
        batch.push(replay);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_replay_batch_serde_roundtrip() {
        let batch = ReplayBatch::new("w-001".to_string(), Some(3));
        let json = serde_json::to_string(&batch).expect("serialize");
        let rt: ReplayBatch = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.worker_id, "w-001");
        assert_eq!(rt.model_version, Some(3));
    }

    #[test]
    fn test_compress_decompress_various_patterns() {
        let config = default_config();
        // Test with alternating bytes (no runs)
        let alternating: Vec<u8> = (0..100)
            .map(|i| if i % 2 == 0 { 0xAA } else { 0xBB })
            .collect();
        let compressed = compress_replay(&alternating, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();
        assert_eq!(decompressed, alternating);
    }

    #[test]
    fn test_compress_decompress_single_byte() {
        let config = default_config();
        let original = vec![0x42];
        let compressed = compress_replay(&original, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compress_decompress_two_bytes_same() {
        let config = default_config();
        let original = vec![0x42, 0x42];
        let compressed = compress_replay(&original, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_compress_decompress_arbitrary(data in proptest::collection::vec(any::<u8>(), 0..1000)) {
                let config = ReplayTransportConfig {
                    compression_enabled: true,
                    max_payload_bytes: 10 * 1024 * 1024,
                    ..ReplayTransportConfig::default()
                };
                let compressed = compress_replay(&data, &config).unwrap();
                let decompressed = decompress_replay(&compressed).unwrap();
                prop_assert_eq!(decompressed, data);
            }
        }
    }
}

//! Replay compression, decompression, and batching for transport.
//!
//! Provides utilities for preparing compact replays for network transport
//! between workers and the training coordinator.

use forge_replay::compact::CompactReplay;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::config::ReplayTransportConfig;
use crate::error::{CloudResult, TransportError};

/// Maximum decompressed replay size (100 MB) to prevent memory exhaustion from crafted input.
const MAX_DECOMPRESSED_SIZE: usize = 100 * 1024 * 1024;

/// Maximum bytes encoded in a single literal or repeated run.
const MAX_RUN_LEN: usize = 128;

/// Frame header indicating RLE-compressed payload follows.
const FRAME_COMPRESSED: u8 = 0xFE;
/// Frame header indicating uncompressed (passthrough) payload follows.
const FRAME_UNCOMPRESSED: u8 = 0xFF;

/// Compresses serialized replay bytes using a simple run-length encoding
/// when compression is enabled.
///
/// If compression is disabled in the config, returns a framed uncompressed payload.
#[instrument(skip(data, config))]
pub fn compress_replay(data: &[u8], config: &ReplayTransportConfig) -> CloudResult<Vec<u8>> {
    if !config.compression_enabled {
        if data.len() + 1 > config.max_payload_bytes {
            return Err(TransportError::PayloadTooLarge {
                size: data.len() + 1,
                max: config.max_payload_bytes,
            }
            .into());
        }
        debug!(
            size = data.len(),
            "compression disabled, sending framed uncompressed"
        );
        let mut out = Vec::with_capacity(1 + data.len());
        out.push(FRAME_UNCOMPRESSED);
        out.extend_from_slice(data);
        return Ok(out);
    }

    if data.len() > config.max_payload_bytes {
        debug!(
            size = data.len(),
            max = config.max_payload_bytes,
            "payload exceeds max size before compression, attempting compression"
        );
    }

    // Simple RLE compression with literal-run framing.
    // High-bit clear control bytes encode a literal run of 1..=128 bytes,
    // followed by that many literal bytes. High-bit set control bytes encode
    // a repeated run of 1..=128 copies of the next byte.
    let mut compressed = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let mut repeat_len = 1usize;
        while i + repeat_len < data.len()
            && data[i + repeat_len] == data[i]
            && repeat_len < MAX_RUN_LEN
        {
            repeat_len += 1;
        }

        if repeat_len >= 3 {
            compressed.push(0x80 | ((repeat_len - 1) as u8));
            compressed.push(data[i]);
            i += repeat_len;
        } else {
            let literal_start = i;
            let mut literal_len = 0usize;
            while i < data.len() && literal_len < MAX_RUN_LEN {
                let mut next_repeat_len = 1usize;
                while i + next_repeat_len < data.len()
                    && data[i + next_repeat_len] == data[i]
                    && next_repeat_len < MAX_RUN_LEN
                {
                    next_repeat_len += 1;
                }

                if next_repeat_len >= 3 {
                    break;
                }

                let chunk_len = next_repeat_len.min(MAX_RUN_LEN - literal_len);
                literal_len += chunk_len;
                i += chunk_len;
            }

            compressed.push((literal_len - 1) as u8);
            compressed.extend_from_slice(&data[literal_start..literal_start + literal_len]);
        }
    }

    // If compression expanded the data, fall back to uncompressed
    if compressed.len() > data.len() {
        debug!(
            original = data.len(),
            compressed = compressed.len(),
            "compression not beneficial, using uncompressed"
        );
        // +1 for the frame header byte
        if data.len() + 1 > config.max_payload_bytes {
            return Err(TransportError::PayloadTooLarge {
                size: data.len() + 1,
                max: config.max_payload_bytes,
            }
            .into());
        }
        let mut out = Vec::with_capacity(1 + data.len());
        out.push(FRAME_UNCOMPRESSED);
        out.extend_from_slice(data);
        return Ok(out);
    }

    // +1 for the frame header byte
    if compressed.len() + 1 > config.max_payload_bytes {
        return Err(TransportError::PayloadTooLarge {
            size: compressed.len() + 1,
            max: config.max_payload_bytes,
        }
        .into());
    }

    debug!(
        original = data.len(),
        compressed = compressed.len(),
        "compressed replay"
    );
    let mut out = Vec::with_capacity(1 + compressed.len());
    out.push(FRAME_COMPRESSED);
    out.extend(compressed);
    Ok(out)
}

/// Decompresses replay bytes that were compressed with [`compress_replay`].
///
/// Returns the original uncompressed bytes. The first byte is a frame
/// header: [`FRAME_UNCOMPRESSED`] means the rest is raw data;
/// [`FRAME_COMPRESSED`] means the rest is RLE-encoded.
#[instrument(skip(data))]
pub fn decompress_replay(data: &[u8]) -> CloudResult<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    match data[0] {
        FRAME_UNCOMPRESSED => {
            let payload = &data[1..];
            if payload.len() > MAX_DECOMPRESSED_SIZE {
                return Err(TransportError::PayloadTooLarge {
                    size: payload.len(),
                    max: MAX_DECOMPRESSED_SIZE,
                }
                .into());
            }
            debug!(
                compressed = data.len(),
                decompressed = payload.len(),
                "decompressed replay (uncompressed frame)"
            );
            return Ok(payload.to_vec());
        }
        FRAME_COMPRESSED => {}
        other => {
            return Err(TransportError::DeserializationFailed(format!(
                "unknown frame header: 0x{other:02x}"
            ))
            .into());
        }
    }

    // RLE decode the payload after the frame header.
    let rle_data = &data[1..];
    let mut decompressed = Vec::with_capacity(rle_data.len());
    let mut i = 0;
    while i < rle_data.len() {
        let control = rle_data[i];
        i += 1;

        if control & 0x80 == 0 {
            let literal_len = control as usize + 1;
            if i + literal_len > rle_data.len() {
                return Err(TransportError::DeserializationFailed(
                    "truncated literal run".to_string(),
                )
                .into());
            }
            decompressed.extend_from_slice(&rle_data[i..i + literal_len]);
            i += literal_len;
        } else {
            let repeat_len = (control & 0x7F) as usize + 1;
            if i >= rle_data.len() {
                return Err(TransportError::DeserializationFailed(
                    "truncated repeated run".to_string(),
                )
                .into());
            }
            let byte = rle_data[i];
            i += 1;
            decompressed.extend(std::iter::repeat_n(byte, repeat_len));
        }

        if decompressed.len() > MAX_DECOMPRESSED_SIZE {
            return Err(TransportError::PayloadTooLarge {
                size: decompressed.len(),
                max: MAX_DECOMPRESSED_SIZE,
            }
            .into());
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
        assert_eq!(result[0], FRAME_UNCOMPRESSED);
        assert_eq!(&result[1..], original);
        assert_eq!(decompress_replay(&result).unwrap(), original);
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
    fn test_decompress_invalid_frame_header() {
        let data = vec![0x42, 0x00]; // Unknown frame header
        let result = decompress_replay(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_truncated_literal() {
        // FRAME_COMPRESSED header followed by a literal run marker with no payload bytes.
        let data = vec![FRAME_COMPRESSED, 0x02];
        let result = decompress_replay(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_truncated_rle() {
        // FRAME_COMPRESSED header followed by a repeated-run marker, but missing the repeated byte.
        let data = vec![FRAME_COMPRESSED, 0x80];
        let result = decompress_replay(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_literal_run_encoding_falls_back_to_uncompressed_frame() {
        let config = default_config();
        let original: Vec<u8> = (0..32).map(|value| value as u8).collect();
        let compressed = compress_replay(&original, &config).unwrap();

        assert_eq!(compressed[0], FRAME_UNCOMPRESSED);
        assert_eq!(compressed.len(), original.len() + 1);
    }

    #[test]
    fn test_compress_mixed_literal_and_repeated_runs() {
        let config = default_config();
        let original = b"abcdefffffghij";
        let compressed = compress_replay(original, &config).unwrap();
        let decompressed = decompress_replay(&compressed).unwrap();

        assert_eq!(decompressed, original);
        assert_eq!(compressed[0], FRAME_COMPRESSED);
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

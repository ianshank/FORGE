//! Store-and-forward telemetry buffer for edge devices.
//!
//! Accumulates serialized [`CompactReplay`] payloads in a bounded buffer.
//! When [`flush()`](TelemetryCollector::flush) is called, sends the stored
//! raw bytes via a [`ReplayTransport`] backend.

use forge_replay::compact::CompactReplay;
use forge_types::config::EdgeConfig;
use forge_types::error::{EdgeError, ForgeError, ForgeResult};
use forge_types::transport::ReplayTransport;
use tracing::{debug, info, instrument, warn};

use crate::metrics::TelemetrySnapshot;

struct BufferedReplay {
    seed: u64,
    payload: Vec<u8>,
}

/// Store-and-forward telemetry buffer for edge devices.
///
/// Accumulates serialized [`CompactReplay`] payloads in a bounded buffer.
/// When [`flush()`](TelemetryCollector::flush) is called, sends the stored
/// bytes via a [`ReplayTransport`] backend. If the buffer is full,
/// [`record()`](TelemetryCollector::record) returns an error rather than
/// flushing automatically.
pub struct TelemetryCollector {
    /// Buffered replay payloads waiting to be flushed.
    buffer: Vec<BufferedReplay>,
    /// Current estimated buffer size in bytes.
    buffer_bytes: u64,
    /// Maximum buffer capacity in bytes.
    max_buffer_bytes: u64,
    /// Whether telemetry compression is enabled in configuration.
    /// This flag is retained for observability/logging only; the collector
    /// does not apply compression directly (transport backends handle it).
    compress: bool,
    /// Total replays recorded since creation.
    total_recorded: u64,
    /// Total replays successfully flushed.
    total_flushed: u64,
    /// Total flush operations that failed.
    /// Per-replay send failures (incremented once per failed replay, not per flush call).
    total_replay_send_failures: u64,
}

impl TelemetryCollector {
    /// Creates a new telemetry collector from an [`EdgeConfig`].
    pub fn new(config: &EdgeConfig) -> Self {
        if config.compress_telemetry {
            debug!("compress_telemetry enabled; compression is delegated to the transport layer");
        }

        Self {
            buffer: Vec::new(),
            buffer_bytes: 0,
            max_buffer_bytes: config.telemetry_buffer_bytes,
            compress: config.compress_telemetry,
            total_recorded: 0,
            total_flushed: 0,
            total_replay_send_failures: 0,
        }
    }

    /// Records a replay into the buffer.
    ///
    /// Returns an error if the buffer would exceed its byte capacity.
    #[instrument(skip_all)]
    pub fn record(&mut self, replay: CompactReplay) -> ForgeResult<()> {
        let seed = replay.seed;
        let payload = replay
            .to_bytes()
            .map_err(|error| ForgeError::Edge(EdgeError::Telemetry(error)))?;
        let replay_bytes = payload.len() as u64;

        if self.buffer_bytes + replay_bytes > self.max_buffer_bytes {
            warn!(
                current_bytes = self.buffer_bytes,
                max_bytes = self.max_buffer_bytes,
                "Telemetry buffer full"
            );
            return Err(ForgeError::Edge(EdgeError::TelemetryBufferFull {
                current_bytes: self.buffer_bytes,
                max_bytes: self.max_buffer_bytes,
            }));
        }

        self.buffer_bytes += replay_bytes;
        self.buffer.push(BufferedReplay { seed, payload });
        self.total_recorded += 1;

        debug!(
            pending = self.buffer.len(),
            buffer_bytes = self.buffer_bytes,
            "Replay recorded"
        );

        Ok(())
    }

    /// Flushes all buffered replays via the transport.
    ///
    /// Returns the number of replays successfully sent. On transport
    /// failure, remaining replays stay in the buffer and the failure
    /// count is incremented.
    #[instrument(skip_all)]
    pub fn flush(&mut self, transport: &dyn ReplayTransport) -> ForgeResult<u32> {
        if self.buffer.is_empty() {
            return Ok(0);
        }

        let count = self.buffer.len();
        info!(
            replays = count,
            compress = self.compress,
            "Flushing telemetry buffer"
        );

        let mut sent = 0u32;
        let replays = std::mem::take(&mut self.buffer);
        self.buffer_bytes = 0;
        let mut failed_replays = Vec::new();
        let mut failed_bytes = 0u64;

        let flush_nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        for (i, replay) in replays.into_iter().enumerate() {
            let key = format!("edge_replay_{}_{}_{}", replay.seed, flush_nonce, i);

            match transport.send(&key, &replay.payload) {
                Ok(()) => {
                    sent += 1;
                    self.total_flushed += 1;
                }
                Err(e) => {
                    warn!(error = %e, "Transport send failed, re-queuing replay");
                    self.total_replay_send_failures += 1;
                    failed_bytes += replay.payload.len() as u64;
                    failed_replays.push(replay);
                }
            }
        }

        // Restore failed replays to buffer
        if !failed_replays.is_empty() {
            debug!(count = failed_replays.len(), "Re-queuing failed replays");
            self.buffer = failed_replays;
            self.buffer_bytes = failed_bytes;
        }

        if sent < count as u32 {
            warn!(sent, total = count, "Not all replays flushed successfully");
        }

        debug!(sent, "Telemetry flush complete");

        Ok(sent)
    }

    /// Returns current telemetry stats.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            pending_replays: self.buffer.len(),
            buffer_bytes: self.buffer_bytes,
            buffer_capacity_bytes: self.max_buffer_bytes,
            total_replays_recorded: self.total_recorded,
            total_replays_flushed: self.total_flushed,
            total_replay_send_failures: self.total_replay_send_failures,
        }
    }

    /// Returns the number of pending replays.
    pub fn pending_count(&self) -> usize {
        self.buffer.len()
    }

    /// Returns buffer utilization as a fraction (0.0 to 1.0).
    pub fn buffer_utilization(&self) -> f32 {
        if self.max_buffer_bytes == 0 {
            return 0.0;
        }
        self.buffer_bytes as f32 / self.max_buffer_bytes as f32
    }

    /// Clears the buffer without sending. Counters are preserved.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.buffer_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::error::ForgeResult;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// Mock transport that stores sent payloads in memory.
    struct MockTransport {
        queue: Mutex<VecDeque<(String, Vec<u8>)>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                queue: Mutex::new(VecDeque::new()),
            }
        }

        fn sent_count(&self) -> usize {
            self.queue.lock().unwrap().len()
        }
    }

    impl ReplayTransport for MockTransport {
        fn send(&self, key: &str, payload: &[u8]) -> ForgeResult<()> {
            self.queue
                .lock()
                .unwrap()
                .push_back((key.to_string(), payload.to_vec()));
            Ok(())
        }

        fn receive(&self) -> ForgeResult<Option<(String, Vec<u8>)>> {
            Ok(self.queue.lock().unwrap().pop_front())
        }

        fn backend_name(&self) -> &str {
            "mock"
        }
    }

    /// Mock transport that always fails on send.
    struct FailingTransport;

    impl ReplayTransport for FailingTransport {
        fn send(&self, _key: &str, _payload: &[u8]) -> ForgeResult<()> {
            Err(ForgeError::Edge(EdgeError::Telemetry(
                "mock send failure".to_string(),
            )))
        }

        fn receive(&self) -> ForgeResult<Option<(String, Vec<u8>)>> {
            Ok(None)
        }

        fn backend_name(&self) -> &str {
            "failing-mock"
        }
    }

    fn make_edge_config() -> EdgeConfig {
        EdgeConfig {
            telemetry_buffer_bytes: 1_048_576, // 1 MB
            compress_telemetry: false,
            ..EdgeConfig::default()
        }
    }

    fn make_test_replay() -> CompactReplay {
        let config = ForgeConfig::default();
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]);
        builder.build()
    }

    #[test]
    fn test_record_adds_to_buffer() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);
        let replay = make_test_replay();

        collector.record(replay).unwrap();
        assert_eq!(collector.pending_count(), 1);
        assert_eq!(collector.snapshot().total_replays_recorded, 1);
    }

    #[test]
    fn test_record_rejects_when_buffer_full() {
        let mut cfg = make_edge_config();
        cfg.telemetry_buffer_bytes = 1; // 1 byte limit
        let mut collector = TelemetryCollector::new(&cfg);
        let replay = make_test_replay();

        let result = collector.record(replay);
        assert!(result.is_err());
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_record_uses_serialized_size_for_capacity() {
        let replay = make_test_replay();
        let replay_bytes = replay.to_bytes().unwrap().len() as u64;

        let mut cfg = make_edge_config();
        cfg.telemetry_buffer_bytes = replay_bytes.saturating_sub(1);
        let mut collector = TelemetryCollector::new(&cfg);

        let result = collector.record(replay);
        assert!(result.is_err());
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_flush_sends_all_replays() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);
        let transport = MockTransport::new();

        collector.record(make_test_replay()).unwrap();
        collector.record(make_test_replay()).unwrap();

        let sent = collector.flush(&transport).unwrap();
        assert_eq!(sent, 2);
        assert_eq!(transport.sent_count(), 2);
        assert_eq!(collector.pending_count(), 0);
        assert_eq!(collector.snapshot().total_replays_flushed, 2);
    }

    #[test]
    fn test_flush_with_failing_transport() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);
        let transport = FailingTransport;

        collector.record(make_test_replay()).unwrap();
        let sent = collector.flush(&transport).unwrap();
        assert_eq!(sent, 0);
        assert_eq!(collector.snapshot().total_replay_send_failures, 1);
    }

    #[test]
    fn test_snapshot_accuracy() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);

        collector.record(make_test_replay()).unwrap();
        collector.record(make_test_replay()).unwrap();

        let snap = collector.snapshot();
        assert_eq!(snap.pending_replays, 2);
        assert!(snap.buffer_bytes > 0);
        assert_eq!(snap.buffer_capacity_bytes, 1_048_576);
        assert_eq!(snap.total_replays_recorded, 2);
        assert_eq!(snap.total_replays_flushed, 0);
        assert_eq!(snap.total_replay_send_failures, 0);
    }

    #[test]
    fn test_buffer_utilization() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);
        assert_eq!(collector.buffer_utilization(), 0.0);

        collector.record(make_test_replay()).unwrap();
        assert!(collector.buffer_utilization() > 0.0);
        assert!(collector.buffer_utilization() <= 1.0);
    }

    #[test]
    fn test_clear_resets_buffer_not_counters() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);

        collector.record(make_test_replay()).unwrap();
        collector.record(make_test_replay()).unwrap();
        assert_eq!(collector.snapshot().total_replays_recorded, 2);

        collector.clear();
        assert_eq!(collector.pending_count(), 0);
        assert_eq!(collector.buffer_utilization(), 0.0);
        // Counters preserved
        assert_eq!(collector.snapshot().total_replays_recorded, 2);
    }

    #[test]
    fn test_flush_empty_buffer() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);
        let transport = MockTransport::new();

        let sent = collector.flush(&transport).unwrap();
        assert_eq!(sent, 0);
        assert_eq!(transport.sent_count(), 0);
    }

    #[test]
    fn test_buffer_utilization_zero_capacity() {
        let mut cfg = make_edge_config();
        cfg.telemetry_buffer_bytes = 0;
        let collector = TelemetryCollector::new(&cfg);
        assert_eq!(collector.buffer_utilization(), 0.0);
    }

    #[test]
    fn test_flush_requeues_on_transport_failure() {
        let cfg = make_edge_config();
        let mut collector = TelemetryCollector::new(&cfg);
        let transport = FailingTransport;

        collector.record(make_test_replay()).unwrap();
        collector.record(make_test_replay()).unwrap();
        assert_eq!(collector.pending_count(), 2);

        let sent = collector.flush(&transport).unwrap();
        assert_eq!(sent, 0);

        // Failed replays should be re-queued back into the buffer.
        assert_eq!(collector.pending_count(), 2);
        assert!(collector.buffer_bytes > 0);
        assert_eq!(collector.snapshot().total_replay_send_failures, 2);
    }
}

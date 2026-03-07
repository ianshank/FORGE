//! Replay recording for simulation state snapshots.
//!
//! The [`ReplayRecorder`] captures periodic [`ReplayFrame`]s that contain
//! agent positions, health values, and serialized event summaries. Frames
//! can be exported as JSON for offline analysis or playback in the dashboard.

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Default recording interval in ticks.
const DEFAULT_RECORDING_INTERVAL: u32 = 1;

/// Default maximum number of frames retained by a [`ReplayRecorder`].
const DEFAULT_MAX_FRAMES: usize = 100_000;

/// Default schema version for replay data.
const DEFAULT_SCHEMA_VERSION: u32 = 1;

/// Configuration for replay recording behaviour.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Whether replay recording is enabled.
    pub enabled: bool,
    /// Record a frame every N ticks.
    pub recording_interval: u32,
    /// Maximum number of frames to retain before evicting the oldest.
    pub max_frames: usize,
    /// Schema version tag written into exported data.
    pub schema_version: u32,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            recording_interval: DEFAULT_RECORDING_INTERVAL,
            max_frames: DEFAULT_MAX_FRAMES,
            schema_version: DEFAULT_SCHEMA_VERSION,
        }
    }
}

/// A single snapshot of simulation state at a given tick.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReplayFrame {
    /// The simulation tick this frame was captured at.
    pub tick: u64,
    /// Agent positions as `(id, x, y)` tuples.
    pub agent_positions: Vec<(u32, u16, u16)>,
    /// Agent health values as `(id, health)` tuples.
    pub agent_health: Vec<(u32, i32)>,
    /// Serialized event summaries that occurred during this tick.
    pub events: Vec<String>,
}

/// Records [`ReplayFrame`]s according to a [`ReplayConfig`].
///
/// The recorder maintains a bounded buffer of frames. When the buffer
/// exceeds [`ReplayConfig::max_frames`], the oldest frame is evicted.
#[derive(Clone, Debug, Default)]
pub struct ReplayRecorder {
    /// Active configuration.
    config: ReplayConfig,
    /// Recorded frames in insertion order.
    frames: Vec<ReplayFrame>,
}

impl ReplayRecorder {
    /// Creates a new [`ReplayRecorder`] with the given configuration.
    #[instrument(level = "debug", skip(config))]
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
        }
    }

    /// Returns `true` if a frame should be recorded at the given tick.
    ///
    /// Recording occurs when the recorder is enabled and the tick aligns
    /// with the configured recording interval.
    #[instrument(level = "trace", skip(self))]
    pub fn should_record(&self, tick: u64) -> bool {
        self.config.enabled && tick % u64::from(self.config.recording_interval) == 0
    }

    /// Records a frame, evicting the oldest frame if the buffer is full.
    #[instrument(level = "trace", skip(self, frame))]
    pub fn record_frame(&mut self, frame: ReplayFrame) {
        if self.frames.len() >= self.config.max_frames {
            self.frames.remove(0);
        }
        self.frames.push(frame);
    }

    /// Returns a slice of all recorded frames.
    #[instrument(level = "trace", skip(self))]
    pub fn frames(&self) -> &[ReplayFrame] {
        &self.frames
    }

    /// Returns the number of recorded frames.
    #[instrument(level = "trace", skip(self))]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Serializes all recorded frames to a JSON string.
    #[instrument(level = "debug", skip(self))]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.frames)
    }

    /// Removes all recorded frames.
    #[instrument(level = "debug", skip(self))]
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame(tick: u64) -> ReplayFrame {
        ReplayFrame {
            tick,
            agent_positions: vec![(1, 10, 20), (2, 30, 40)],
            agent_health: vec![(1, 100), (2, 80)],
            events: vec!["combat".to_string()],
        }
    }

    #[test]
    fn default_config_values() {
        let config = ReplayConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.recording_interval, DEFAULT_RECORDING_INTERVAL);
        assert_eq!(config.max_frames, DEFAULT_MAX_FRAMES);
        assert_eq!(config.schema_version, DEFAULT_SCHEMA_VERSION);
    }

    #[test]
    fn default_recorder_is_disabled() {
        let recorder = ReplayRecorder::default();
        assert!(!recorder.should_record(0));
        assert_eq!(recorder.frame_count(), 0);
    }

    #[test]
    fn should_record_respects_enabled_flag() {
        let disabled = ReplayRecorder::new(ReplayConfig {
            enabled: false,
            recording_interval: 1,
            ..Default::default()
        });
        assert!(!disabled.should_record(0));

        let enabled = ReplayRecorder::new(ReplayConfig {
            enabled: true,
            recording_interval: 1,
            ..Default::default()
        });
        assert!(enabled.should_record(0));
        assert!(enabled.should_record(42));
    }

    #[test]
    fn should_record_respects_interval() {
        let recorder = ReplayRecorder::new(ReplayConfig {
            enabled: true,
            recording_interval: 5,
            ..Default::default()
        });

        assert!(recorder.should_record(0));
        assert!(!recorder.should_record(1));
        assert!(!recorder.should_record(4));
        assert!(recorder.should_record(5));
        assert!(recorder.should_record(10));
    }

    #[test]
    fn record_and_retrieve_frames() {
        let mut recorder = ReplayRecorder::new(ReplayConfig {
            enabled: true,
            ..Default::default()
        });

        recorder.record_frame(sample_frame(0));
        recorder.record_frame(sample_frame(1));

        assert_eq!(recorder.frame_count(), 2);
        assert_eq!(recorder.frames().len(), 2);
        assert_eq!(recorder.frames()[0].tick, 0);
        assert_eq!(recorder.frames()[1].tick, 1);
    }

    #[test]
    fn max_frames_eviction() {
        let mut recorder = ReplayRecorder::new(ReplayConfig {
            enabled: true,
            max_frames: 3,
            ..Default::default()
        });

        for tick in 0..5 {
            recorder.record_frame(sample_frame(tick));
        }

        assert_eq!(recorder.frame_count(), 3);
        let ticks: Vec<u64> = recorder.frames().iter().map(|f| f.tick).collect();
        assert_eq!(ticks, vec![2, 3, 4]);
    }

    #[test]
    fn to_json_serialization() {
        let mut recorder = ReplayRecorder::new(ReplayConfig {
            enabled: true,
            ..Default::default()
        });
        recorder.record_frame(sample_frame(0));

        let json = recorder.to_json().expect("serialization should succeed");
        let restored: Vec<ReplayFrame> =
            serde_json::from_str(&json).expect("deserialization should succeed");

        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0], sample_frame(0));
    }

    #[test]
    fn clear_removes_all_frames() {
        let mut recorder = ReplayRecorder::new(ReplayConfig {
            enabled: true,
            ..Default::default()
        });
        recorder.record_frame(sample_frame(0));
        recorder.record_frame(sample_frame(1));
        assert_eq!(recorder.frame_count(), 2);

        recorder.clear();
        assert_eq!(recorder.frame_count(), 0);
        assert!(recorder.frames().is_empty());
    }

    #[test]
    fn replay_config_serialization_roundtrip() {
        let config = ReplayConfig {
            enabled: true,
            recording_interval: 10,
            max_frames: 500,
            schema_version: 2,
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let restored: ReplayConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.enabled, config.enabled);
        assert_eq!(restored.recording_interval, config.recording_interval);
        assert_eq!(restored.max_frames, config.max_frames);
        assert_eq!(restored.schema_version, config.schema_version);
    }
}

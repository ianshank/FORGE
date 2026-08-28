//! Persistent history for training metrics and decision traces.
//!
//! The REST surface accepts training metrics and decision traces via POST and
//! broadcasts them live over WebSocket. This module adds durable storage so the
//! dashboard can also *query* past data (`/api/training-metrics/history`,
//! `/api/decision-traces/history`, `/api/runs`).
//!
//! Storage is intentionally a simple append-only JSONL file format
//! ([`JsonlHistoryStore`]) rather than an embedded database: the data volume is
//! low, bounded retention keeps full-file reads cheap, it adds no native/C
//! build dependency, and it survives process restarts. The [`HistoryStore`]
//! trait keeps the choice swappable (e.g. for SQLite later) with no change at
//! the call sites, and [`InMemoryHistoryStore`] gives tests a filesystem-free
//! implementation.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use forge_types::time::now_ms;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::metrics::{DecisionTraceEntry, TrainingMetrics};

/// Errors raised by a [`HistoryStore`].
#[derive(Debug, Error)]
pub enum HistoryError {
    /// An I/O error while reading or writing the backing store.
    #[error("history I/O error: {0}")]
    Io(String),

    /// A (de)serialization error for a history record.
    #[error("history serialization error: {0}")]
    Serde(String),
}

/// A stored training-metrics sample, tagged with its run id and capture time.
///
/// The inner [`TrainingMetrics`] is flattened so the JSON wire shape matches the
/// existing camelCase fields the dashboard already consumes (plus `runId` and
/// `recordedAtMs`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRecord {
    /// Run this sample belongs to.
    pub run_id: String,
    /// Capture time (ms since Unix epoch).
    pub recorded_at_ms: u64,
    /// The training metrics payload.
    #[serde(flatten)]
    pub metrics: TrainingMetrics,
}

/// A stored decision-trace entry, tagged with its run id and capture time.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceRecord {
    /// Run this trace belongs to.
    pub run_id: String,
    /// Capture time (ms since Unix epoch).
    pub recorded_at_ms: u64,
    /// The decision-trace payload.
    #[serde(flatten)]
    pub trace: DecisionTraceEntry,
}

/// Aggregate summary of a single run, derived from its stored records.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    /// Run identifier.
    pub run_id: String,
    /// Earliest record time observed for this run (ms since Unix epoch).
    pub started_at_ms: u64,
    /// Most recent record time observed for this run (ms since Unix epoch).
    pub last_seen_ms: u64,
    /// Number of stored training samples for this run.
    pub episodes: u64,
    /// Mean reward from the most recent training sample (0.0 if none).
    pub latest_mean_reward: f64,
}

/// Persistent (or in-memory) store for training metrics and decision traces.
///
/// Implementations must be cheap to share across async tasks
/// (`Send + Sync`); the server holds one behind an `Arc`.
pub trait HistoryStore: Send + Sync {
    /// Append a training-metrics sample for `run_id`.
    fn append_training(&self, run_id: &str, metrics: &TrainingMetrics) -> Result<(), HistoryError>;

    /// Append decision-trace entries for `run_id`.
    fn append_traces(
        &self,
        run_id: &str,
        traces: &[DecisionTraceEntry],
    ) -> Result<(), HistoryError>;

    /// Return up to `limit` most-recent training records, optionally filtered to
    /// a single `run_id`. Newest records come last (chronological order).
    fn training_history(
        &self,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TrainingRecord>, HistoryError>;

    /// Return up to `limit` most-recent trace records, optionally filtered to a
    /// single `run_id`. Newest records come last.
    fn traces_history(
        &self,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TraceRecord>, HistoryError>;

    /// Return one [`RunSummary`] per distinct run id seen across both training
    /// and trace records, ordered by `started_at_ms` ascending.
    fn runs(&self) -> Result<Vec<RunSummary>, HistoryError>;
}

/// Build the ordered list of run summaries from raw records. Shared by the
/// file-backed and in-memory stores so aggregation logic lives in one place.
fn summarize_runs(training: &[TrainingRecord], traces: &[TraceRecord]) -> Vec<RunSummary> {
    use std::collections::BTreeMap;

    // BTreeMap keeps a stable (run-id) order; we re-sort by started_at below.
    let mut map: BTreeMap<String, RunSummary> = BTreeMap::new();

    let touch = |map: &mut BTreeMap<String, RunSummary>, run_id: &str, ts: u64| {
        let entry = map.entry(run_id.to_string()).or_insert_with(|| RunSummary {
            run_id: run_id.to_string(),
            started_at_ms: ts,
            last_seen_ms: ts,
            episodes: 0,
            latest_mean_reward: 0.0,
        });
        entry.started_at_ms = entry.started_at_ms.min(ts);
        entry.last_seen_ms = entry.last_seen_ms.max(ts);
    };

    for rec in training {
        touch(&mut map, &rec.run_id, rec.recorded_at_ms);
        let entry = map.get_mut(&rec.run_id).expect("entry just inserted");
        entry.episodes += 1;
        // Records are appended chronologically, so the last one wins.
        entry.latest_mean_reward = rec.metrics.mean_reward;
    }
    for rec in traces {
        touch(&mut map, &rec.run_id, rec.recorded_at_ms);
    }

    let mut out: Vec<RunSummary> = map.into_values().collect();
    out.sort_by_key(|s| (s.started_at_ms, s.run_id.clone()));
    out
}

/// Filter by optional `run_id` and keep the last `limit` records (newest).
fn filter_tail<T, F>(records: Vec<T>, run_id: Option<&str>, limit: usize, run_of: F) -> Vec<T>
where
    F: Fn(&T) -> &str,
{
    let filtered: Vec<T> = match run_id {
        Some(id) => records.into_iter().filter(|r| run_of(r) == id).collect(),
        None => records,
    };
    if filtered.len() > limit {
        let skip = filtered.len() - limit;
        filtered.into_iter().skip(skip).collect()
    } else {
        filtered
    }
}

/// Append-only, JSONL file-backed [`HistoryStore`].
///
/// Two files live under a base directory: `training.jsonl` and `traces.jsonl`.
/// Each append writes one JSON line and then trims the file to the newest
/// `retention` lines. Writes are serialized by an internal mutex; reads parse
/// the whole (bounded) file.
pub struct JsonlHistoryStore {
    training_path: PathBuf,
    traces_path: PathBuf,
    retention: usize,
    write_lock: Mutex<()>,
}

impl JsonlHistoryStore {
    /// File name for stored training records.
    const TRAINING_FILE: &'static str = "training.jsonl";
    /// File name for stored trace records.
    const TRACES_FILE: &'static str = "traces.jsonl";

    /// Open (creating the directory if needed) a store rooted at `dir`, keeping
    /// at most `retention` records per file (a `retention` of 0 is treated as 1
    /// so at least the latest record is always retained).
    pub fn open(dir: impl AsRef<Path>, retention: usize) -> Result<Self, HistoryError> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir).map_err(|e| HistoryError::Io(e.to_string()))?;
        Ok(Self {
            training_path: dir.join(Self::TRAINING_FILE),
            traces_path: dir.join(Self::TRACES_FILE),
            retention: retention.max(1),
            write_lock: Mutex::new(()),
        })
    }

    /// Append one serialized record line to `path`, then enforce retention.
    fn append_line<T: Serialize>(&self, path: &Path, record: &T) -> Result<(), HistoryError> {
        let _guard = self.write_lock.lock().unwrap_or_else(|e| e.into_inner());
        let line = serde_json::to_string(record).map_err(|e| HistoryError::Serde(e.to_string()))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| HistoryError::Io(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| HistoryError::Io(e.to_string()))?;
        drop(file);
        self.enforce_retention(path)
    }

    /// Trim `path` to its newest `retention` lines (rewrite via temp + rename).
    fn enforce_retention(&self, path: &Path) -> Result<(), HistoryError> {
        let lines = read_lines(path)?;
        if lines.len() <= self.retention {
            return Ok(());
        }
        let keep = &lines[lines.len() - self.retention..];
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f = File::create(&tmp).map_err(|e| HistoryError::Io(e.to_string()))?;
            for line in keep {
                writeln!(f, "{line}").map_err(|e| HistoryError::Io(e.to_string()))?;
            }
        }
        fs::rename(&tmp, path).map_err(|e| HistoryError::Io(e.to_string()))
    }
}

/// Read all non-empty lines from `path`, returning an empty vec if it is absent.
fn read_lines(path: &Path) -> Result<Vec<String>, HistoryError> {
    match File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let mut out = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| HistoryError::Io(e.to_string()))?;
                if !line.trim().is_empty() {
                    out.push(line);
                }
            }
            Ok(out)
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(HistoryError::Io(e.to_string())),
    }
}

/// Parse every line of `path` into `T`, skipping (and logging) malformed lines
/// so a single corrupt record never breaks a query.
fn read_records<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, HistoryError> {
    let mut out = Vec::new();
    for line in read_lines(path)? {
        match serde_json::from_str::<T>(&line) {
            Ok(rec) => out.push(rec),
            Err(e) => tracing::warn!(error = %e, "skipping malformed history record"),
        }
    }
    Ok(out)
}

impl HistoryStore for JsonlHistoryStore {
    fn append_training(&self, run_id: &str, metrics: &TrainingMetrics) -> Result<(), HistoryError> {
        let record = TrainingRecord {
            run_id: run_id.to_string(),
            recorded_at_ms: now_ms(),
            metrics: metrics.clone(),
        };
        self.append_line(&self.training_path, &record)
    }

    fn append_traces(
        &self,
        run_id: &str,
        traces: &[DecisionTraceEntry],
    ) -> Result<(), HistoryError> {
        for trace in traces {
            let record = TraceRecord {
                run_id: run_id.to_string(),
                recorded_at_ms: now_ms(),
                trace: trace.clone(),
            };
            self.append_line(&self.traces_path, &record)?;
        }
        Ok(())
    }

    fn training_history(
        &self,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TrainingRecord>, HistoryError> {
        let records = read_records::<TrainingRecord>(&self.training_path)?;
        Ok(filter_tail(records, run_id, limit, |r| r.run_id.as_str()))
    }

    fn traces_history(
        &self,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TraceRecord>, HistoryError> {
        let records = read_records::<TraceRecord>(&self.traces_path)?;
        Ok(filter_tail(records, run_id, limit, |r| r.run_id.as_str()))
    }

    fn runs(&self) -> Result<Vec<RunSummary>, HistoryError> {
        let training = read_records::<TrainingRecord>(&self.training_path)?;
        let traces = read_records::<TraceRecord>(&self.traces_path)?;
        Ok(summarize_runs(&training, &traces))
    }
}

/// In-memory [`HistoryStore`] for tests and ephemeral use. Honors `retention`
/// by dropping the oldest records once a buffer exceeds it.
#[derive(Default)]
pub struct InMemoryHistoryStore {
    training: Mutex<Vec<TrainingRecord>>,
    traces: Mutex<Vec<TraceRecord>>,
    retention: usize,
}

impl InMemoryHistoryStore {
    /// Create an in-memory store keeping at most `retention` records per kind
    /// (0 is treated as 1).
    pub fn new(retention: usize) -> Self {
        Self {
            training: Mutex::new(Vec::new()),
            traces: Mutex::new(Vec::new()),
            retention: retention.max(1),
        }
    }
}

/// Drop oldest entries so `buf` holds at most `retention` records.
fn trim_front<T>(buf: &mut Vec<T>, retention: usize) {
    if buf.len() > retention {
        let drop_n = buf.len() - retention;
        buf.drain(0..drop_n);
    }
}

impl HistoryStore for InMemoryHistoryStore {
    fn append_training(&self, run_id: &str, metrics: &TrainingMetrics) -> Result<(), HistoryError> {
        let mut buf = self.training.lock().unwrap_or_else(|e| e.into_inner());
        buf.push(TrainingRecord {
            run_id: run_id.to_string(),
            recorded_at_ms: now_ms(),
            metrics: metrics.clone(),
        });
        trim_front(&mut buf, self.retention);
        Ok(())
    }

    fn append_traces(
        &self,
        run_id: &str,
        traces: &[DecisionTraceEntry],
    ) -> Result<(), HistoryError> {
        let mut buf = self.traces.lock().unwrap_or_else(|e| e.into_inner());
        for trace in traces {
            buf.push(TraceRecord {
                run_id: run_id.to_string(),
                recorded_at_ms: now_ms(),
                trace: trace.clone(),
            });
        }
        trim_front(&mut buf, self.retention);
        Ok(())
    }

    fn training_history(
        &self,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TrainingRecord>, HistoryError> {
        let buf = self.training.lock().unwrap_or_else(|e| e.into_inner());
        Ok(filter_tail(buf.clone(), run_id, limit, |r| {
            r.run_id.as_str()
        }))
    }

    fn traces_history(
        &self,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TraceRecord>, HistoryError> {
        let buf = self.traces.lock().unwrap_or_else(|e| e.into_inner());
        Ok(filter_tail(buf.clone(), run_id, limit, |r| {
            r.run_id.as_str()
        }))
    }

    fn runs(&self) -> Result<Vec<RunSummary>, HistoryError> {
        let training = self.training.lock().unwrap_or_else(|e| e.into_inner());
        let traces = self.traces.lock().unwrap_or_else(|e| e.into_inner());
        Ok(summarize_runs(&training, &traces))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(episode: u64, mean_reward: f64) -> TrainingMetrics {
        TrainingMetrics {
            episode,
            mean_reward,
            ..Default::default()
        }
    }

    fn trace(tick: u64) -> DecisionTraceEntry {
        DecisionTraceEntry {
            tick,
            intent_label: "explore".to_string(),
            ..Default::default()
        }
    }

    /// Run the shared behavior suite against any store implementation.
    fn exercise_store(store: &dyn HistoryStore) {
        // Empty store returns empty vecs.
        assert!(store.training_history(None, 100).unwrap().is_empty());
        assert!(store.traces_history(None, 100).unwrap().is_empty());
        assert!(store.runs().unwrap().is_empty());

        store.append_training("run-a", &metrics(1, 0.5)).unwrap();
        store.append_training("run-a", &metrics(2, 1.5)).unwrap();
        store.append_training("run-b", &metrics(1, -0.5)).unwrap();
        store
            .append_traces("run-a", &[trace(10), trace(11)])
            .unwrap();

        // Round-trip + chronological order (newest last).
        let all = store.training_history(None, 100).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all.last().unwrap().run_id, "run-b");

        // Filter by run id.
        let a = store.training_history(Some("run-a"), 100).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a[1].metrics.mean_reward, 1.5);

        // Limit keeps the newest N.
        let last_one = store.training_history(Some("run-a"), 1).unwrap();
        assert_eq!(last_one.len(), 1);
        assert_eq!(last_one[0].metrics.episode, 2);

        // Traces round-trip + filter.
        assert_eq!(store.traces_history(None, 100).unwrap().len(), 2);
        assert_eq!(store.traces_history(Some("run-b"), 100).unwrap().len(), 0);

        // Run summaries: two runs, run-a seen first, episodes counted.
        let runs = store.runs().unwrap();
        assert_eq!(runs.len(), 2);
        let run_a = runs.iter().find(|r| r.run_id == "run-a").unwrap();
        assert_eq!(run_a.episodes, 2);
        assert_eq!(run_a.latest_mean_reward, 1.5);
    }

    #[test]
    fn in_memory_store_behaves() {
        exercise_store(&InMemoryHistoryStore::new(10_000));
    }

    #[test]
    fn jsonl_store_behaves() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlHistoryStore::open(dir.path(), 10_000).unwrap();
        exercise_store(&store);
    }

    #[test]
    fn jsonl_store_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = JsonlHistoryStore::open(dir.path(), 10_000).unwrap();
            store.append_training("run-x", &metrics(7, 2.0)).unwrap();
        }
        // Re-open the same directory: the record survives.
        let store = JsonlHistoryStore::open(dir.path(), 10_000).unwrap();
        let recs = store.training_history(None, 100).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].metrics.episode, 7);
    }

    #[test]
    fn retention_prunes_oldest_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlHistoryStore::open(dir.path(), 3).unwrap();
        for i in 0..10 {
            store
                .append_training("run-r", &metrics(i, i as f64))
                .unwrap();
        }
        let recs = store.training_history(None, 100).unwrap();
        assert_eq!(recs.len(), 3, "retention should cap stored records");
        // Only the newest three episodes (7,8,9) remain.
        assert_eq!(recs[0].metrics.episode, 7);
        assert_eq!(recs[2].metrics.episode, 9);
    }

    #[test]
    fn retention_prunes_oldest_in_memory() {
        let store = InMemoryHistoryStore::new(2);
        for i in 0..5 {
            store.append_training("r", &metrics(i, 0.0)).unwrap();
        }
        let recs = store.training_history(None, 100).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].metrics.episode, 3);
    }

    #[test]
    fn retention_zero_keeps_at_least_one() {
        let store = InMemoryHistoryStore::new(0);
        store.append_training("r", &metrics(1, 0.0)).unwrap();
        store.append_training("r", &metrics(2, 0.0)).unwrap();
        assert_eq!(store.training_history(None, 100).unwrap().len(), 1);
    }

    #[test]
    fn training_record_serializes_camel_case_and_flattens() {
        let rec = TrainingRecord {
            run_id: "run-a".to_string(),
            recorded_at_ms: 123,
            metrics: metrics(4, 0.25),
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("\"runId\":\"run-a\""));
        assert!(json.contains("\"recordedAtMs\":123"));
        // Flattened TrainingMetrics field keeps its camelCase name.
        assert!(json.contains("\"meanReward\":0.25"));
        // Round-trips back.
        let back: TrainingRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run_id, "run-a");
        assert_eq!(back.metrics.episode, 4);
    }

    #[test]
    fn run_summary_serializes_camel_case() {
        let summary = RunSummary {
            run_id: "r".to_string(),
            started_at_ms: 1,
            last_seen_ms: 2,
            episodes: 3,
            latest_mean_reward: 0.5,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"runId\":\"r\""));
        assert!(json.contains("\"startedAtMs\":1"));
        assert!(json.contains("\"lastSeenMs\":2"));
        assert!(json.contains("\"latestMeanReward\":0.5"));
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlHistoryStore::open(dir.path(), 10_000).unwrap();
        store.append_training("run-a", &metrics(1, 0.0)).unwrap();
        // Append a junk line directly.
        let path = dir.path().join("training.jsonl");
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{{ not valid json").unwrap();
        drop(f);
        // The valid record is still returned; the junk line is ignored.
        assert_eq!(store.training_history(None, 100).unwrap().len(), 1);
    }
}

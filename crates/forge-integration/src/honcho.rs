use std::path::Path;
use tracing::{debug, error, instrument};

use forge_replay::journal::{AppendOnlyJournalReader, JournalEntry};
use forge_types::config::HonchoConfig;

/// Service that reads from the `AppendOnlyJournal` and exports memory to Honcho.
///
/// Ensures that latent states, Q-values, and checkpoint-specific data are filtered out,
/// providing only semantic, safe observations to the external cognitive mirror.
pub struct HonchoExporter {
    config: HonchoConfig,
    reader: AppendOnlyJournalReader,
}

impl HonchoExporter {
    /// Creates a new `HonchoExporter`.
    pub fn new<P: AsRef<Path>>(config: HonchoConfig, journal_path: P) -> std::io::Result<Self> {
        let reader = AppendOnlyJournalReader::new(journal_path)?;
        Ok(Self { config, reader })
    }

    /// Syncs the journal to Honcho up to the current EOF.
    ///
    /// Reads entries sequentially, applies filtration rules to strip out
    /// latent states or internal metrics, and issues RPCs to the Honcho endpoint.
    #[instrument(skip(self))]
    pub fn sync(&mut self) -> std::io::Result<usize> {
        if !self.config.enabled {
            return Ok(0);
        }

        let mut synced_count = 0;
        let client = ureq::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build();

        loop {
            match self.reader.next_entry() {
                Ok(Some(entry)) => {
                    if let Some(filtered) = self.filter_entry(entry) {
                        // In a real system, you might batch these
                        self.export_to_honcho(&client, &filtered);
                        synced_count += 1;
                    }
                }
                Ok(None) => break, // EOF reached
                Err(e) => {
                    error!(error = %e, "Failed to read journal entry during Honcho sync");
                    return Err(e);
                }
            }
        }

        if synced_count > 0 {
            debug!(synced_count, "Synced entries to Honcho");
        }

        Ok(synced_count)
    }

    /// Filters a `JournalEntry` to ensure no latent state leaks.
    fn filter_entry(&self, entry: JournalEntry) -> Option<serde_json::Value> {
        // Strip out sensitive/latent state if configured
        if !self.config.strip_latent_state {
            return Some(serde_json::to_value(&entry).unwrap_or_default());
        }

        match entry {
            JournalEntry::Event(ev) => {
                // Pure events are generally safe (no latent arrays)
                Some(serde_json::json!({
                    "type": "event",
                    "data": ev
                }))
            }
            JournalEntry::ActionProposal {
                agent_id,
                action_type,
                tick,
                ..
            } => {
                // Drop payload (which might contain Q-values or raw internal vectors)
                Some(serde_json::json!({
                    "type": "action_proposal",
                    "agent_id": agent_id,
                    "action_type": action_type,
                    "tick": tick
                }))
            }
            JournalEntry::TickBoundary(tick) => Some(serde_json::json!({
                "type": "tick_boundary",
                "tick": tick
            })),
        }
    }

    /// Sends the filtered payload to Honcho.
    fn export_to_honcho(&self, client: &ureq::Agent, payload: &serde_json::Value) {
        let endpoint = format!("{}/v1/memory/ingest", self.config.endpoint);

        match client.post(&endpoint).send_json(payload) {
            Ok(_) => {
                // Successfully synced
            }
            Err(e) => {
                debug!(error = %e, endpoint, "Failed to export memory to Honcho (expected if Honcho isn't running)");
            }
        }
    }
}

/// A wrapper for contexts retrieved from Honcho.
///
/// Ensures the rest of the simulation knows this memory originates from an external,
/// potentially hallucinating source, preventing blind trust in external data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UntrustedContext {
    /// Source identifier (e.g., "honcho").
    pub source: String,
    /// The unverified payload data.
    pub payload: serde_json::Value,
    /// Estimated confidence score or relevance.
    pub confidence_score: f32,
}

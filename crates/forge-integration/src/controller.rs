use std::sync::mpsc::Sender;
use std::time::Duration;
use tracing::{debug, instrument, warn, error};

use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use forge_replay::journal::JournalEntry;

/// An external controller representing an experimental control plane (e.g., Google ADK).
///
/// This controller implements the `AgentInterface` trait to integrate seamlessly
/// into the simulation's tick loop, but its actions are treated strictly as proposals.
/// Real execution paths and mutations are gated by `forge-core::systems`.
pub struct ExternalController {
    name: String,
    endpoint: String,
    journal_tx: Option<Sender<JournalEntry>>,
}

impl ExternalController {
    /// Creates a new external controller connected to the specified endpoint.
    pub fn new(name: String, endpoint: String, journal_tx: Option<Sender<JournalEntry>>) -> Self {
        Self {
            name,
            endpoint,
            journal_tx,
        }
    }
}

impl AgentInterface for ExternalController {
    #[instrument(skip(self, obs))]
    fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse {
        let mut action_id = 0u32; // Noop by default
        
        let client = ureq::builder()
            .timeout(Duration::from_millis(500)) // Strict timeout for the simulation loop
            .build();
            
        // Attempt to send RPC
        match client.post(&self.endpoint)
            .send_json(serde_json::json!({
                "agent_id": agent_idx,
                "observation": obs,
            })) {
            Ok(response) => {
                // Parse the response (assume it returns `{"action_id": u32}`)
                if let Ok(json) = response.into_json::<serde_json::Value>() {
                    if let Some(act) = json.get("action_id").and_then(|v| v.as_u64()) {
                        action_id = act as u32;
                    }
                }
            }
            Err(e) => {
                error!(error = %e, endpoint = %self.endpoint, "Failed to contact external controller");
            }
        }
        
        if let Some(tx) = &self.journal_tx {
            let proposal = JournalEntry::ActionProposal {
                agent_id: agent_idx as u32,
                action_type: "external_rpc".to_string(),
                payload: serde_json::to_vec(&action_id).unwrap_or_default(),
                tick: 0, // Tick should be supplied by the environment or observer layer
            };
            
            if let Err(e) = tx.send(proposal) {
                warn!(error = %e, "Failed to send action proposal to journal");
            }
        }
        
        debug!(agent_idx, endpoint = %self.endpoint, action_id, "External controller proposed action");
        
        AgentResponse::from_action(action_id)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> AgentMetadata {
        let mut meta = AgentMetadata::default();
        meta.agent_type = "external_controller".to_string();
        meta.model_name = self.name.clone();
        meta.version = "1.0".to_string();
        meta.parameters.insert("endpoint".to_string(), self.endpoint.clone());
        meta
    }
}

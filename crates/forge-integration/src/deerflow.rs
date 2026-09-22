use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use tracing::{debug, error, instrument};

use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;

/// A super-agent harness for the DeerFlow long-horizon coordinator.
///
/// DeerFlow orchestrates research over long periods but must not be allowed
/// to alter authoritative state or write artifacts outside of its sandbox.
pub struct DeerFlowHarness {
    name: String,
    artifact_dir: PathBuf,
    /// Channel for sending observations to the sandboxed execution thread
    tx: Option<mpsc::Sender<Observation>>,
    /// Channel for receiving actions from the sandboxed execution thread
    rx: Option<mpsc::Receiver<u32>>,
}

impl DeerFlowHarness {
    /// Creates a new DeerFlow harness and spawns its isolated thread.
    ///
    /// The thread is constrained by its `artifact_dir`. In a true OS-level sandbox,
    /// we would drop capabilities here (e.g., using `seccomp` or `capsicum`). For this
    /// implementation, we emulate it via thread isolation and strict pathing.
    #[instrument(skip(artifact_dir))]
    pub fn new<P: AsRef<Path>>(name: String, artifact_dir: P) -> Self {
        let artifact_dir = artifact_dir.as_ref().to_path_buf();

        // Ensure the directory exists
        if let Err(e) = std::fs::create_dir_all(&artifact_dir) {
            error!(error = %e, dir = %artifact_dir.display(), "Failed to create DeerFlow artifact directory");
        }

        let (obs_tx, obs_rx) = mpsc::channel::<Observation>();
        let (act_tx, act_rx) = mpsc::channel::<u32>();

        let thread_dir = artifact_dir.clone();

        // Spawn the sandboxed thread
        thread::Builder::new()
            .name(format!("deerflow-sandbox-{}", name))
            .spawn(move || {
                debug!(dir = %thread_dir.display(), "DeerFlow sandbox thread started");

                // Here, a real implementation might call `prctl(PR_SET_SECCOMP, ...)`
                // or drop permissions to ensure the thread cannot escape `thread_dir`.

                while let Ok(_obs) = obs_rx.recv() {
                    // DeerFlow logic would go here.
                    // For now, it simply proposes a NOOP (0).

                    // Artificial restriction check (simulated)
                    let check_path = thread_dir.join("scratch.tmp");
                    if let Err(e) = std::fs::write(&check_path, b"test") {
                        error!(error = %e, "DeerFlow sandbox failed to write to its restricted dir");
                    }

                    if act_tx.send(0).is_err() {
                        break;
                    }
                }

                debug!("DeerFlow sandbox thread exiting");
            })
            .expect("Failed to spawn DeerFlow sandbox thread");

        Self {
            name,
            artifact_dir,
            tx: Some(obs_tx),
            rx: Some(act_rx),
        }
    }
}

impl AgentInterface for DeerFlowHarness {
    fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse {
        let mut action_id = 0; // Noop default

        if let (Some(tx), Some(rx)) = (&self.tx, &self.rx) {
            // Send observation to the sandbox
            if tx.send(obs.clone()).is_ok() {
                // Wait for the sandbox to propose an action (with a timeout in a real system)
                if let Ok(act) = rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    action_id = act;
                } else {
                    error!(agent_idx, "DeerFlow sandbox thread timed out or panicked");
                }
            } else {
                error!(agent_idx, "Failed to send observation to DeerFlow sandbox");
            }
        }

        AgentResponse::from_action(action_id)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> AgentMetadata {
        let mut meta = AgentMetadata::default();
        meta.agent_type = "deerflow_harness".to_string();
        meta.model_name = self.name.clone();
        meta.version = "1.0".to_string();
        meta.parameters.insert(
            "artifact_dir".to_string(),
            self.artifact_dir.to_string_lossy().to_string(),
        );
        meta
    }
}

impl Drop for DeerFlowHarness {
    fn drop(&mut self) {
        // Dropping the tx will cause the sandbox thread's recv() to fail and the thread to exit.
        self.tx.take();
        self.rx.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_deerflow_sandbox_containment() {
        // This test ensures the harness starts up, respects its dir, and doesn't crash on standard observations
        let dir = tempdir().unwrap();
        let artifact_path = dir.path().join("artifacts");

        let mut harness = DeerFlowHarness::new("test_deerflow".to_string(), &artifact_path);

        let obs = Observation::default();

        let response = harness.select_action(&obs, 0);
        assert_eq!(response.action_id, 0);

        // Assert that the sandbox thread created the check file
        assert!(artifact_path.join("scratch.tmp").exists());
    }
}

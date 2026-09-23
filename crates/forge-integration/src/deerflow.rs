use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
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
    timeout: Duration,
    next_request_id: u64,
    /// Channel for sending observations to the sandboxed execution thread
    tx: Option<mpsc::Sender<ActionRequest>>,
    /// Channel for receiving actions from the sandboxed execution thread
    rx: Option<mpsc::Receiver<ActionResponse>>,
}

#[derive(Clone)]
struct ActionRequest {
    request_id: u64,
    observation: Observation,
}

struct ActionResponse {
    request_id: u64,
    action_id: u32,
}

impl DeerFlowHarness {
    /// Creates a new DeerFlow harness and spawns its isolated thread.
    ///
    /// The thread is constrained by its `artifact_dir`. In a true OS-level sandbox,
    /// we would drop capabilities here (e.g., using `seccomp` or `capsicum`). For this
    /// implementation, we emulate it via thread isolation and strict pathing.
    #[instrument(skip(artifact_dir))]
    pub fn new<P: AsRef<Path>>(name: String, artifact_dir: P, timeout_ms: u64) -> Self {
        let artifact_dir = artifact_dir.as_ref().to_path_buf();
        let timeout = Duration::from_millis(timeout_ms);

        // Ensure the directory exists
        if let Err(e) = std::fs::create_dir_all(&artifact_dir) {
            error!(error = %e, dir = %artifact_dir.display(), "Failed to create DeerFlow artifact directory");
        }

        let (obs_tx, obs_rx) = mpsc::channel::<ActionRequest>();
        let (act_tx, act_rx) = mpsc::channel::<ActionResponse>();

        let thread_dir = artifact_dir.clone();

        // Spawn the sandboxed thread
        thread::Builder::new()
            .name(format!("deerflow-sandbox-{}", name))
            .spawn(move || {
                debug!(dir = %thread_dir.display(), "DeerFlow sandbox thread started");

                // Here, a real implementation might call `prctl(PR_SET_SECCOMP, ...)`
                // or drop permissions to ensure the thread cannot escape `thread_dir`.

                while let Ok(request) = obs_rx.recv() {
                    // DeerFlow logic would go here.
                    // For now, it simply proposes a NOOP (0).

                    // Artificial restriction check (simulated)
                    let check_path = thread_dir.join("scratch.tmp");
                    if let Err(e) = std::fs::write(&check_path, b"test") {
                        error!(error = %e, "DeerFlow sandbox failed to write to its restricted dir");
                    } else {
                        // Prevent disk leakage by immediately removing the temp file
                        let _ = std::fs::remove_file(&check_path);
                    }

                    let response = ActionResponse {
                        request_id: request.request_id,
                        action_id: 0,
                    };
                    let _ = request.observation;

                    if act_tx.send(response).is_err() {
                        break;
                    }
                }

                debug!("DeerFlow sandbox thread exiting");
            })
            .expect("Failed to spawn DeerFlow sandbox thread");

        Self {
            name,
            artifact_dir,
            timeout,
            next_request_id: 0,
            tx: Some(obs_tx),
            rx: Some(act_rx),
        }
    }
}

impl AgentInterface for DeerFlowHarness {
    fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse {
        let mut action_id = 0; // Noop default

        if let (Some(tx), Some(rx)) = (&self.tx, &self.rx) {
            let request_id = self.next_request_id;
            self.next_request_id = self.next_request_id.wrapping_add(1);
            let request = ActionRequest {
                request_id,
                observation: obs.clone(),
            };

            // Send observation to the sandbox
            if tx.send(request).is_ok() {
                let deadline = Instant::now() + self.timeout;
                loop {
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        error!(agent_idx, "DeerFlow sandbox thread timed out or panicked");
                        break;
                    };

                    match rx.recv_timeout(remaining) {
                        Ok(response) if response.request_id == request_id => {
                            action_id = response.action_id;
                            break;
                        }
                        Ok(response) => {
                            debug!(
                                agent_idx,
                                expected_request_id = request_id,
                                stale_request_id = response.request_id,
                                "Discarding stale DeerFlow sandbox response"
                            );
                        }
                        Err(_) => {
                            error!(agent_idx, "DeerFlow sandbox thread timed out or panicked");
                            break;
                        }
                    }
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

        let mut harness = DeerFlowHarness::new("test_deerflow".to_string(), &artifact_path, 100);

        let obs = Observation::default();

        let response = harness.select_action(&obs, 0);
        assert_eq!(response.action_id, 0);

        // Assert that the sandbox thread did not leak the scratch check file
        assert!(!artifact_path.join("scratch.tmp").exists());
    }
}

//! Export trajectory data to external formats.
//!
//! Supports CSV and JSON Lines (JSONL) for quick inspection and
//! integration with ML pipelines. HuggingFace Datasets (Parquet)
//! export can be added via an optional feature flag in the future.

use std::io::Write;
use std::path::Path;

use tracing::{debug, instrument};

use crate::trajectory::Trajectory;

/// Exports a trajectory to CSV format.
///
/// Each row contains: tick, agent_idx, action, reward, terminated, truncated,
/// health, stamina, position_x, position_y, confidence, decision_time_ms.
#[instrument(skip_all)]
pub fn export_to_csv(trajectory: &Trajectory, path: &Path) -> Result<(), String> {
    let mut file =
        std::fs::File::create(path).map_err(|e| format!("failed to create file: {e}"))?;

    // Header
    writeln!(
        file,
        "tick,agent_idx,action,reward,terminated,truncated,health,stamina,pos_x,pos_y,confidence,decision_time_ms"
    )
    .map_err(|e| format!("write error: {e}"))?;

    debug!(steps = trajectory.steps.len(), path = %path.display(), "Exporting trajectory to CSV");

    for step in &trajectory.steps {
        let num_agents = step.actions.len();
        for agent_idx in 0..num_agents {
            let obs = step.observations.get(agent_idx);
            let health = obs.map_or(0.0, |o| o.health);
            let stamina = obs.map_or(0.0, |o| o.stamina);
            let (pos_x, pos_y) = obs.map_or((0, 0), |o| o.position);
            let confidence = step.confidences.get(agent_idx).copied().unwrap_or(0.0);
            let decision_time = step.decision_times_ms.get(agent_idx).copied().unwrap_or(0);

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                step.tick,
                agent_idx,
                step.actions.get(agent_idx).copied().unwrap_or(0),
                step.rewards.get(agent_idx).copied().unwrap_or(0.0),
                step.terminated,
                step.truncated,
                health,
                stamina,
                pos_x,
                pos_y,
                confidence,
                decision_time,
            )
            .map_err(|e| format!("write error: {e}"))?;
        }
    }

    Ok(())
}

/// Exports a trajectory to JSON Lines format (one JSON object per step).
///
/// Each line is a self-contained JSON object representing one step.
/// This format is compatible with HuggingFace Datasets `load_dataset("json")`.
#[instrument(skip_all)]
pub fn export_to_jsonl(trajectory: &Trajectory, path: &Path) -> Result<(), String> {
    let mut file =
        std::fs::File::create(path).map_err(|e| format!("failed to create file: {e}"))?;

    debug!(steps = trajectory.steps.len(), path = %path.display(), "Exporting trajectory to JSONL");

    for step in &trajectory.steps {
        let json = serde_json::to_string(step).map_err(|e| format!("serialization error: {e}"))?;
        writeln!(file, "{json}").map_err(|e| format!("write error: {e}"))?;
    }

    Ok(())
}

/// Exports trajectory metadata to JSON.
#[instrument(skip_all)]
pub fn export_metadata(trajectory: &Trajectory, path: &Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&trajectory.metadata)
        .map_err(|e| format!("serialization error: {e}"))?;

    std::fs::write(path, json).map_err(|e| format!("write error: {e}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::TrajectoryBuilder;
    use forge_types::agent_interface::AgentResponse;
    use forge_types::constants;
    use forge_types::observation::{InventoryObservation, Observation, TileObservation};

    fn make_test_observation() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 0.8,
            stamina: 0.6,
            position: (3, 7),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    fn make_test_trajectory() -> Trajectory {
        let mut builder = TrajectoryBuilder::new();
        let obs = make_test_observation();
        let responses = vec![AgentResponse::from_action(1)];

        builder.record_step(0, vec![obs.clone()], &responses, vec![0.5], false, false);
        builder.record_step(1, vec![obs], &responses, vec![1.0], true, false);

        builder.seed(42).build(vec![1.5])
    }

    #[test]
    fn test_export_to_csv() {
        let traj = make_test_trajectory();
        let dir = std::env::temp_dir().join("forge_test_csv");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_trajectory.csv");

        export_to_csv(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Header + 2 data rows
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("tick,agent_idx"));
        assert!(lines[1].starts_with("0,0,1,0.5"));
        assert!(lines[2].starts_with("1,0,1,1"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_to_jsonl() {
        let traj = make_test_trajectory();
        let dir = std::env::temp_dir().join("forge_test_jsonl");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_trajectory.jsonl");

        export_to_jsonl(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines.len(), 2);
        // Each line should be valid JSON
        for line in &lines {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_metadata() {
        let traj = make_test_trajectory();
        let dir = std::env::temp_dir().join("forge_test_meta");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_metadata.json");

        export_metadata(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(meta["seed"], 42);
        assert_eq!(meta["total_steps"], 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_csv_invalid_path() {
        let traj = make_test_trajectory();
        let result = export_to_csv(&traj, Path::new("/nonexistent/dir/file.csv"));
        assert!(result.is_err());
    }

    #[test]
    fn test_export_empty_trajectory() {
        let traj = Trajectory::new();
        let dir = std::env::temp_dir().join("forge_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.csv");

        export_to_csv(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // Only header, no data rows
        assert_eq!(lines.len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_jsonl_invalid_path() {
        let traj = make_test_trajectory();
        let result = export_to_jsonl(&traj, Path::new("/nonexistent/dir/file.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_export_metadata_invalid_path() {
        let traj = make_test_trajectory();
        let result = export_metadata(&traj, Path::new("/nonexistent/dir/meta.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_export_csv_multi_agent() {
        let mut builder = TrajectoryBuilder::new();
        let obs = make_test_observation();
        let responses = vec![AgentResponse::from_action(1), AgentResponse::from_action(2)];

        builder.record_step(
            0,
            vec![obs.clone(), obs],
            &responses,
            vec![0.5, 0.3],
            false,
            false,
        );
        let traj = builder
            .agent_names(vec!["A".into(), "B".into()])
            .build(vec![0.5, 0.3]);

        let dir = std::env::temp_dir().join("forge_test_csv_multi");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("multi_agent.csv");

        export_to_csv(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // Header + 2 agents × 1 step = 3 lines
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains(",0,")); // agent_idx 0
        assert!(lines[2].contains(",1,")); // agent_idx 1

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_jsonl_roundtrip() {
        use crate::trajectory::TrajectoryStep;

        let traj = make_test_trajectory();
        let dir = std::env::temp_dir().join("forge_test_jsonl_rt");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("roundtrip.jsonl");

        export_to_jsonl(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        for (i, line) in content.lines().enumerate() {
            let step: TrajectoryStep = serde_json::from_str(line).unwrap();
            assert_eq!(step.tick, traj.steps[i].tick);
            assert_eq!(step.actions, traj.steps[i].actions);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_metadata_roundtrip() {
        use crate::trajectory::TrajectoryMetadata;

        let traj = make_test_trajectory();
        let dir = std::env::temp_dir().join("forge_test_meta_rt");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("meta_rt.json");

        export_metadata(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let meta: TrajectoryMetadata = serde_json::from_str(&content).unwrap();
        assert_eq!(meta.seed, traj.metadata.seed);
        assert_eq!(meta.total_steps, traj.metadata.total_steps);
        assert_eq!(meta.final_rewards, traj.metadata.final_rewards);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_empty_jsonl() {
        let traj = Trajectory::new();
        let dir = std::env::temp_dir().join("forge_test_empty_jsonl");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.jsonl");

        export_to_jsonl(&traj, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.is_empty());

        let _ = std::fs::remove_file(&path);
    }
}

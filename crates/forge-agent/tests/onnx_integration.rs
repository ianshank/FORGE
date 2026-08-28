#[cfg(feature = "onnx")]
#[test]
// requires Python with torch, onnx, and onnxscript installed (exports real
// MuZero networks via a subprocess) plus a real ONNX Runtime >=1.23.2 loaded
// via ORT_DYLIB_PATH; run with --ignored. Matches the
// forge-eval/tests/exporters_e2e.rs precedent for heavy-external-dep tests.
// (ONNX Runtime <1.23.2 hits a known upstream ort rc.13 teardown segfault on
// process exit after this test's real inference completes successfully --
// pykeio/ort#614, fixed by pykeio/ort#610 in the runtime, not in this crate.)
#[ignore]
fn test_onnx_pipeline_integration() {
    use forge_agent::latent_mcts::onnx_model::{OnnxModelConfig, OnnxMuZeroModel};
    use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};
    use std::fs;
    use std::process::Command;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let out_dir = dir.path().join("onnx_out");
    fs::create_dir_all(&out_dir).unwrap();

    let python_script = r#"
import sys
from pathlib import Path
from forge.models.muzero_config import MuZeroConfig
from forge.models.muzero_world_model import MuZeroWorldModel
from forge.models.muzero_export import MuZeroExporter

out_dir = Path(sys.argv[1])
config = MuZeroConfig(
    obs_dim=920,
    action_dim=75,
    latent_dim=16,
    hidden_dim=16,
    num_blocks=1,
    reward_support_size=11,
    value_support_size=11,
    cnn_channels=(8,),
    cnn_kernel_sizes=(3,),
    cnn_strides=(1,),
)
model = MuZeroWorldModel(config)
exporter = MuZeroExporter(model)
exporter.export_onnx(out_dir)
"#;

    let script_path = dir.path().join("export.py");
    fs::write(&script_path, python_script).unwrap();

    // Set PYTHONPATH to point to the python directory from the workspace root where the test runs
    let workspace_root = std::env::current_dir()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let python_dir = workspace_root.join("python");

    let status = Command::new("python3")
        .arg(&script_path)
        .arg(&out_dir)
        .env("PYTHONPATH", python_dir)
        .status()
        .expect("Failed to execute python3 command");

    assert!(
        status.success(),
        "Python ONNX export script failed. Note: Requires torch and onnx runtime installed."
    );

    let config = OnnxModelConfig {
        representation_path: out_dir
            .join("representation.onnx")
            .to_string_lossy()
            .to_string(),
        dynamics_path: out_dir.join("dynamics.onnx").to_string_lossy().to_string(),
        prediction_path: out_dir
            .join("prediction.onnx")
            .to_string_lossy()
            .to_string(),
        action_space_size: 75,
        latent_dim: 16,
        num_threads: 1,
    };

    assert!(
        OnnxMuZeroModel::validate_paths(&config),
        "ONNX paths missing"
    );

    let model = OnnxMuZeroModel::load(config).expect("Failed to load ONNX models in Rust");

    let mcts_config = LatentMctsConfig::default();
    let search = LatentMctsSearch::new(model, mcts_config);

    let obs = vec![0.0f32; 920];
    let result = search.search(&obs).unwrap();

    assert!(result.action < 75);
    assert!(result.visit_counts.len() == 75);

    let total_visits: u32 = result.visit_counts.iter().sum();
    // Default is 50 simulations (from MctsConfig::default()) + initial node expansion
    assert!(total_visits > 0);
}

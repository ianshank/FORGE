//! End-to-end coverage of the model-bundle integrity gate.
//!
//! The gate itself lives in `forge_mc_runner::integrity` and is invoked
//! from `onnx_reload::config_from_manifest`, which only compiles with
//! the `onnx-reload` feature (it needs `ort`). These tests exercise the
//! same sequence the runner performs — write a bundle, save a manifest,
//! poll the [`HotReloadWatcher`], verify the emitted manifest — without
//! requiring an ONNX Runtime on the host, so the security-relevant
//! behaviour is covered by a default `cargo test -p forge-mc-runner`.

use std::path::Path;

use forge_mc_runner::integrity::{file_sha256_hex, verify_bundle};
use forge_mc_runner::{
    HotReloadWatcher, ModelFileEntry, ModelManifest, ModelManifestFiles, RunnerError,
};

/// Write `contents` to `dir/name` and return its sha256.
fn write_model(dir: &Path, name: &str, contents: &[u8]) -> String {
    let path = dir.join(name);
    std::fs::write(&path, contents).unwrap();
    file_sha256_hex(&path).unwrap()
}

/// Lay down a three-file bundle plus a matching manifest, exactly as a
/// trainer export would, and return the manifest path.
fn export_bundle(dir: &Path, version: u64, payload_suffix: &str) -> std::path::PathBuf {
    let manifest = ModelManifest::new(
        version,
        "schema-id-under-test",
        "2026-09-05T00:00:00Z",
        ModelManifestFiles {
            representation: ModelFileEntry {
                path: "representation.onnx".into(),
                sha256: write_model(
                    dir,
                    "representation.onnx",
                    format!("representation{payload_suffix}").as_bytes(),
                ),
            },
            dynamics: ModelFileEntry {
                path: "dynamics.onnx".into(),
                sha256: write_model(
                    dir,
                    "dynamics.onnx",
                    format!("dynamics{payload_suffix}").as_bytes(),
                ),
            },
            prediction: ModelFileEntry {
                path: "prediction.onnx".into(),
                sha256: write_model(
                    dir,
                    "prediction.onnx",
                    format!("prediction{payload_suffix}").as_bytes(),
                ),
            },
        },
    );
    let path = dir.join("model_manifest.json");
    manifest.save_json(&path).unwrap();
    path
}

/// The happy path a runner startup takes: load the manifest the trainer
/// wrote, verify the bundle it points at.
#[test]
fn freshly_exported_bundle_verifies() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = export_bundle(tmp.path(), 1, "-v1");
    let manifest = ModelManifest::load_json(&manifest_path).unwrap();
    let verified = verify_bundle(&manifest, tmp.path()).unwrap();
    assert!(verified.representation.is_absolute());
    assert!(verified.dynamics.exists());
    assert!(verified.prediction.exists());
}

/// The hot-reload sequence: the watcher sees a version bump, and the
/// manifest it emits is verified before anything would be loaded.
#[test]
fn hot_reload_event_manifest_verifies_against_the_new_bundle() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = export_bundle(tmp.path(), 1, "-v1");
    let mut watcher = HotReloadWatcher::new(&manifest_path);
    let first = watcher.poll().unwrap().expect("first poll emits an event");
    verify_bundle(&first.manifest, tmp.path()).unwrap();

    // Trainer exports a new bundle version over the same directory.
    export_bundle(tmp.path(), 2, "-v2");
    let second = watcher.poll().unwrap().expect("version bump emits");
    assert_eq!(second.new_version, 2);
    verify_bundle(&second.manifest, tmp.path()).unwrap();
}

/// The realistic corruption case: the ONNX files are rewritten but the
/// manifest's digests are not (a crashed export, a partial rsync, or a
/// deliberate swap on the shared bind-mount). The reload must fail with
/// a digest mismatch naming the role, not load whatever is on disk.
#[test]
fn hot_reload_rejects_bundle_whose_files_no_longer_match_the_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = export_bundle(tmp.path(), 1, "-v1");
    let mut watcher = HotReloadWatcher::new(&manifest_path);
    let event = watcher.poll().unwrap().expect("first poll emits an event");

    // Swap the bytes under the recorded digest.
    std::fs::write(tmp.path().join("representation.onnx"), b"swapped").unwrap();

    match verify_bundle(&event.manifest, tmp.path()).unwrap_err() {
        RunnerError::ModelDigestMismatch {
            role,
            expected,
            actual,
            ..
        } => {
            assert_eq!(role, "representation");
            assert_eq!(expected, event.manifest.files.representation.sha256);
            assert_ne!(actual, expected);
        }
        other => panic!("expected ModelDigestMismatch, got {other:?}"),
    }
}

/// A manifest that points outside its bundle directory must be refused
/// even though every file it names exists and hashes correctly — the
/// containment rule is independent of the digest check.
#[test]
fn manifest_pointing_outside_the_bundle_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("models");
    std::fs::create_dir_all(&bundle).unwrap();
    let manifest_path = export_bundle(&bundle, 1, "-v1");

    // A file the operator never intended the runner to parse.
    let outside_sha = write_model(tmp.path(), "attacker.onnx", b"attacker-controlled");

    let mut manifest = ModelManifest::load_json(&manifest_path).unwrap();
    manifest.files.dynamics = ModelFileEntry {
        path: "../attacker.onnx".into(),
        sha256: outside_sha,
    };

    match verify_bundle(&manifest, &bundle).unwrap_err() {
        RunnerError::UnsafeModelPath { role, reason, .. } => {
            assert_eq!(role, "dynamics");
            assert!(reason.contains(".."), "got reason: {reason}");
        }
        other => panic!("expected UnsafeModelPath, got {other:?}"),
    }
}

/// An absolute path in the manifest — the shape that used to pass
/// through `resolve_relative` verbatim — must also be refused.
#[test]
fn manifest_with_an_absolute_path_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("models");
    std::fs::create_dir_all(&bundle).unwrap();
    let manifest_path = export_bundle(&bundle, 1, "-v1");
    let outside = tmp.path().join("attacker.onnx");
    let outside_sha = write_model(tmp.path(), "attacker.onnx", b"attacker-controlled");

    let mut manifest = ModelManifest::load_json(&manifest_path).unwrap();
    manifest.files.prediction = ModelFileEntry {
        path: outside.to_string_lossy().into_owned(),
        sha256: outside_sha,
    };

    match verify_bundle(&manifest, &bundle).unwrap_err() {
        RunnerError::UnsafeModelPath { role, reason, .. } => {
            assert_eq!(role, "prediction");
            assert!(reason.contains("absolute"), "got reason: {reason}");
        }
        other => panic!("expected UnsafeModelPath, got {other:?}"),
    }
}

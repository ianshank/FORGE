//! Model-bundle integrity: path containment + sha256 verification.
//!
//! ## Why this exists
//!
//! [`crate::ModelManifest`] records a sha256 per ONNX file, but nothing
//! used to re-compute it: [`crate::ModelManifest::validate`] only
//! checked the field was non-empty, and the load path went straight to
//! `ort`'s `Session::commit_from_file`. The path resolution was equally
//! trusting — absolute paths passed through verbatim and `..`
//! components were never rejected — so whoever wrote the manifest chose
//! which bytes the runner's ONNX Runtime parsed, including a file
//! outside the bundle directory entirely.
//!
//! That is a real trust boundary: in the self-play stack the trainer
//! and the runner share a host bind-mount, so the manifest is untrusted
//! input to the runner rather than a local constant.
//!
//! ## Contract
//!
//! [`verify_bundle`] is the single choke point. It
//!
//! 1. resolves each role's manifest entry with [`resolve_bundle_path`],
//!    which rejects absolute paths, any `..` component, and anything
//!    that canonicalizes outside `bundle_dir` (which also catches a
//!    symlink inside the bundle pointing elsewhere), then
//! 2. streams the resolved file through sha256 in
//!    [`DIGEST_CHUNK_BYTES`] chunks and compares the digest against the
//!    manifest's recorded value.
//!
//! Both the initial bundle load and the between-episode hot-reload go
//! through `crate::onnx_reload::config_from_manifest`, which calls
//! `verify_bundle` before handing any path to the ONNX Runtime — so
//! neither path can skip verification.
//!
//! ## What this does not defend against
//!
//! Verification is time-of-check; `ort` re-opens the files when it
//! builds its sessions, so a writer with access to the bundle directory
//! can still swap bytes in that window. Closing that would require
//! holding the file handles across the session build, which the `ort`
//! API does not expose. The check still stops the realistic failure
//! modes — a corrupt/truncated export, a stale file the trainer never
//! rewrote, and a manifest pointing at a file the operator never meant
//! to load.

use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::{debug, instrument};

use crate::error::RunnerError;
use crate::manifest::{ModelFileEntry, ModelManifest};

/// Bytes read per `read` call when streaming a model file through the
/// sha256 hasher.
///
/// Model files are large (tens to hundreds of MB), so they are hashed
/// incrementally rather than read into memory whole. 64 KiB is the
/// usual sweet spot: large enough that syscall overhead disappears
/// against the hashing cost, small enough that the buffer stays in L2.
pub const DIGEST_CHUNK_BYTES: usize = 64 * 1024;

/// Manifest role name for the observation → latent network.
pub const ROLE_REPRESENTATION: &str = "representation";
/// Manifest role name for the (latent, action) → latent + reward network.
pub const ROLE_DYNAMICS: &str = "dynamics";
/// Manifest role name for the latent → policy + value network.
pub const ROLE_PREDICTION: &str = "prediction";

/// A model bundle whose three files have been resolved inside their
/// bundle directory *and* whose contents hash to the digests the
/// manifest recorded.
///
/// The paths are canonical (symlinks resolved, absolute), so callers
/// hand the ONNX Runtime exactly the files that were hashed rather than
/// re-resolving the manifest strings a second time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBundle {
    /// Canonical path to the representation network.
    pub representation: PathBuf,
    /// Canonical path to the dynamics network.
    pub dynamics: PathBuf,
    /// Canonical path to the prediction network.
    pub prediction: PathBuf,
}

/// Resolve one manifest entry path against `bundle_dir`, rejecting
/// anything that could escape it.
///
/// `entry_path` is untrusted (see the module docs). Accepted shapes are
/// relative paths built only from normal components and `.` — e.g.
/// `representation.onnx` or `v00000007/representation.onnx`.
///
/// # Errors
///
/// - [`RunnerError::UnsafeModelPath`] if the entry is empty, absolute,
///   contains a `..` component, or canonicalizes outside `bundle_dir`.
/// - [`RunnerError::MissingPath`] if the resolved file does not exist.
/// - [`RunnerError::Io`] if `bundle_dir` or the resolved file cannot be
///   canonicalized.
#[instrument(skip_all, fields(role = %role, entry = %entry_path, bundle_dir = %bundle_dir.display()))]
pub fn resolve_bundle_path(
    role: &str,
    entry_path: &str,
    bundle_dir: &Path,
) -> Result<PathBuf, RunnerError> {
    let unsafe_path = |reason: &str| RunnerError::UnsafeModelPath {
        role: role.to_string(),
        entry: entry_path.to_string(),
        bundle_dir: bundle_dir.to_path_buf(),
        reason: reason.to_string(),
    };

    if entry_path.trim().is_empty() {
        return Err(unsafe_path("manifest entry path is empty"));
    }

    let candidate = Path::new(entry_path);
    if candidate.is_absolute() {
        return Err(unsafe_path(
            "absolute paths are rejected; manifest entries must be relative to the bundle directory",
        ));
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(unsafe_path(
                    "`..` components are rejected; a manifest entry may not walk out of the bundle directory",
                ));
            }
            // Unreachable on Unix (`is_absolute` already caught them),
            // but a Windows prefix such as `C:file.onnx` is *relative*
            // and would otherwise slip past the check above.
            Component::RootDir | Component::Prefix(_) => {
                return Err(unsafe_path(
                    "rooted or drive-prefixed paths are rejected; manifest entries must be relative",
                ));
            }
        }
    }

    // Canonicalize the base first so the containment comparison below
    // is between two fully-resolved paths (no `..`, no symlinks, no
    // relative segments on either side).
    let base = bundle_dir
        .canonicalize()
        .map_err(|e| RunnerError::io(bundle_dir, e))?;
    let joined = base.join(candidate);
    if !joined.exists() {
        return Err(RunnerError::MissingPath { path: joined });
    }
    let resolved = joined
        .canonicalize()
        .map_err(|e| RunnerError::io(&joined, e))?;
    if !resolved.to_string_lossy().starts_with(&*base.to_string_lossy()) {
        return Err(unsafe_path(
            "path canonicalizes outside the bundle directory (symlink escape)",
        ));
    }
    Ok(resolved)
}

/// Stream `path` through sha256 and return the lowercase hex digest.
///
/// Reads [`DIGEST_CHUNK_BYTES`] at a time so a multi-hundred-MB model
/// never lands in memory whole.
///
/// # Errors
///
/// [`RunnerError::Io`] (carrying `path`) if the file cannot be opened
/// or read.
#[instrument(skip_all, fields(path = %path.display()))]
pub fn file_sha256_hex(path: &Path) -> Result<String, RunnerError> {
    let mut file = File::open(path).map_err(|e| RunnerError::io(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; DIGEST_CHUNK_BYTES];
    let mut total: u64 = 0;
    loop {
        let read = file.read(&mut buf).map_err(|e| RunnerError::io(path, e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        total += read as u64;
    }
    let digest = hex_encode(&hasher.finalize());
    debug!(bytes = total, digest = %digest, "hashed model file");
    Ok(digest)
}

/// Verify a single manifest entry: resolve its path safely, then check
/// the file's sha256 against the digest the manifest recorded.
///
/// Returns the canonical path on success.
///
/// # Errors
///
/// Everything [`resolve_bundle_path`] returns, plus
/// [`RunnerError::ModelDigestMismatch`] when the recorded and computed
/// digests differ.
#[instrument(skip_all, fields(role = %role, entry = %entry.path))]
pub fn verify_entry(
    role: &str,
    entry: &ModelFileEntry,
    bundle_dir: &Path,
) -> Result<PathBuf, RunnerError> {
    let path = resolve_bundle_path(role, &entry.path, bundle_dir)?;
    let actual = file_sha256_hex(&path)?;
    let expected = entry.sha256.trim();
    // Hex digests are case-insensitive; trainers have historically
    // written both cases, and rejecting on case alone would be a
    // false positive rather than a real integrity failure.
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(RunnerError::ModelDigestMismatch {
            role: role.to_string(),
            path,
            expected: expected.to_string(),
            actual,
        });
    }
    debug!(path = %path.display(), "model file digest verified");
    Ok(path)
}

/// Verify every file in `manifest` against `bundle_dir`.
///
/// This is the choke point both the initial bundle load and the
/// between-episode hot-reload go through (via
/// `crate::onnx_reload::config_from_manifest`), so no ONNX session is
/// ever built from an unverified file.
///
/// # Errors
///
/// The first failing role short-circuits with
/// [`RunnerError::UnsafeModelPath`], [`RunnerError::MissingPath`],
/// [`RunnerError::Io`], or [`RunnerError::ModelDigestMismatch`]. Roles
/// are checked in manifest order: representation, dynamics, prediction.
#[instrument(skip_all, fields(version = manifest.version, bundle_dir = %bundle_dir.display()))]
pub fn verify_bundle(
    manifest: &ModelManifest,
    bundle_dir: &Path,
) -> Result<VerifiedBundle, RunnerError> {
    let files = &manifest.files;
    let bundle = VerifiedBundle {
        representation: verify_entry(ROLE_REPRESENTATION, &files.representation, bundle_dir)?,
        dynamics: verify_entry(ROLE_DYNAMICS, &files.dynamics, bundle_dir)?,
        prediction: verify_entry(ROLE_PREDICTION, &files.prediction, bundle_dir)?,
    };
    debug!(
        version = manifest.version,
        "model bundle integrity verified"
    );
    Ok(bundle)
}

/// Lowercase hex encoding of a digest. Kept private so the crate has
/// exactly one spelling of it.
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(&mut out, "{b:02x}").expect("writing to a String never fails");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifestFiles, MANIFEST_SCHEMA_VERSION};

    /// sha256 of the empty input — the canonical known-answer test for
    /// the streaming hasher.
    const SHA256_OF_EMPTY: &str =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    /// Write `contents` to `dir/name` and return its real sha256.
    fn write_file(dir: &Path, name: &str, contents: &[u8]) -> String {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        hex_encode(&Sha256::digest(contents))
    }

    /// Build a bundle dir with three real ONNX-shaped files plus a
    /// manifest whose digests match them.
    fn bundle_with_correct_digests(dir: &Path) -> ModelManifest {
        let rep = write_file(dir, "representation.onnx", b"representation-bytes");
        let dyn_ = write_file(dir, "dynamics.onnx", b"dynamics-bytes");
        let pred = write_file(dir, "prediction.onnx", b"prediction-bytes");
        ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            version: 3,
            schema_id: "sid".into(),
            created_at: "2026-09-05T00:00:00Z".into(),
            files: ModelManifestFiles {
                representation: ModelFileEntry {
                    path: "representation.onnx".into(),
                    sha256: rep,
                },
                dynamics: ModelFileEntry {
                    path: "dynamics.onnx".into(),
                    sha256: dyn_,
                },
                prediction: ModelFileEntry {
                    path: "prediction.onnx".into(),
                    sha256: pred,
                },
            },
        }
    }

    #[test]
    fn file_sha256_hex_matches_known_answer_for_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.bin");
        std::fs::write(&path, b"").unwrap();
        assert_eq!(file_sha256_hex(&path).unwrap(), SHA256_OF_EMPTY);
    }

    /// The chunked read loop must produce the same digest as a
    /// single-shot hash for inputs spanning several chunks (this is the
    /// bug a naive `read` -> `update` loop would hide).
    #[test]
    fn file_sha256_hex_matches_single_shot_across_chunk_boundaries() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("big.bin");
        // 2.5 chunks, with a byte pattern so a dropped/duplicated chunk
        // changes the digest.
        let payload: Vec<u8> = (0..(DIGEST_CHUNK_BYTES * 5 / 2))
            .map(|i| (i % 251) as u8)
            .collect();
        std::fs::write(&path, &payload).unwrap();
        assert_eq!(
            file_sha256_hex(&path).unwrap(),
            hex_encode(&Sha256::digest(&payload))
        );
    }

    #[test]
    fn file_sha256_hex_missing_file_is_io_error_with_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nope.onnx");
        match file_sha256_hex(&path).unwrap_err() {
            RunnerError::Io { path: p, .. } => assert_eq!(p, path),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn verify_bundle_accepts_correct_digests() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = bundle_with_correct_digests(tmp.path());
        let verified = verify_bundle(&manifest, tmp.path()).unwrap();
        assert_eq!(
            verified.representation,
            tmp.path()
                .canonicalize()
                .unwrap()
                .join("representation.onnx")
        );
        assert!(verified.dynamics.ends_with("dynamics.onnx"));
        assert!(verified.prediction.ends_with("prediction.onnx"));
    }

    #[test]
    fn verify_bundle_accepts_uppercase_hex_digest() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = bundle_with_correct_digests(tmp.path());
        manifest.files.dynamics.sha256 = manifest.files.dynamics.sha256.to_uppercase();
        verify_bundle(&manifest, tmp.path()).unwrap();
    }

    #[test]
    fn verify_bundle_accepts_nested_subdirectory_entry() {
        // The v0.4 continuous trainer exports into `v{NNNNNNNN}/`
        // subdirs, so a nested relative path must stay legal.
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = bundle_with_correct_digests(tmp.path());
        let nested = write_file(tmp.path(), "v00000007/representation.onnx", b"nested-bytes");
        manifest.files.representation = ModelFileEntry {
            path: "v00000007/representation.onnx".into(),
            sha256: nested,
        };
        let verified = verify_bundle(&manifest, tmp.path()).unwrap();
        assert!(verified
            .representation
            .ends_with("v00000007/representation.onnx"));
    }

    #[test]
    fn verify_bundle_rejects_tampered_file() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = bundle_with_correct_digests(tmp.path());
        let recorded = manifest.files.dynamics.sha256.clone();
        // Rewrite the file after the manifest recorded its digest.
        std::fs::write(tmp.path().join("dynamics.onnx"), b"tampered-bytes").unwrap();
        let err = verify_bundle(&manifest, tmp.path()).unwrap_err();
        match err {
            RunnerError::ModelDigestMismatch {
                role,
                path,
                expected,
                actual,
            } => {
                assert_eq!(role, ROLE_DYNAMICS);
                assert!(path.ends_with("dynamics.onnx"), "got {}", path.display());
                assert_eq!(expected, recorded);
                assert_eq!(actual, hex_encode(&Sha256::digest(b"tampered-bytes")));
                assert_ne!(expected, actual);
            }
            other => panic!("expected ModelDigestMismatch, got {other:?}"),
        }
    }

    /// A truncated export (partial write that still landed) is the most
    /// likely real-world corruption; it must be caught too.
    #[test]
    fn verify_bundle_rejects_truncated_file() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = bundle_with_correct_digests(tmp.path());
        std::fs::write(tmp.path().join("prediction.onnx"), b"prediction-byt").unwrap();
        match verify_bundle(&manifest, tmp.path()).unwrap_err() {
            RunnerError::ModelDigestMismatch { role, .. } => assert_eq!(role, ROLE_PREDICTION),
            other => panic!("expected ModelDigestMismatch, got {other:?}"),
        }
    }

    #[test]
    fn verify_bundle_rejects_absolute_path() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = bundle_with_correct_digests(tmp.path());
        let absolute = tmp.path().join("representation.onnx");
        manifest.files.representation.path = absolute.to_string_lossy().into_owned();
        match verify_bundle(&manifest, tmp.path()).unwrap_err() {
            RunnerError::UnsafeModelPath {
                role,
                entry,
                bundle_dir,
                reason,
            } => {
                assert_eq!(role, ROLE_REPRESENTATION);
                assert_eq!(entry, absolute.to_string_lossy());
                assert_eq!(bundle_dir, tmp.path());
                assert!(reason.contains("absolute"), "got reason: {reason}");
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    #[test]
    fn verify_bundle_rejects_parent_dir_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("bundle");
        std::fs::create_dir_all(&bundle).unwrap();
        let mut manifest = bundle_with_correct_digests(&bundle);
        // A sibling file the runner must never be talked into loading.
        write_file(tmp.path(), "evil.onnx", b"evil-bytes");
        manifest.files.dynamics.path = "../evil.onnx".into();
        match verify_bundle(&manifest, &bundle).unwrap_err() {
            RunnerError::UnsafeModelPath { role, reason, .. } => {
                assert_eq!(role, ROLE_DYNAMICS);
                assert!(reason.contains(".."), "got reason: {reason}");
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    /// `..` buried mid-path (`sub/../../evil.onnx`) must be rejected on
    /// the component scan, not merely by the containment check.
    #[test]
    fn verify_bundle_rejects_embedded_parent_dir_component() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("bundle");
        std::fs::create_dir_all(bundle.join("sub")).unwrap();
        let mut manifest = bundle_with_correct_digests(&bundle);
        manifest.files.prediction.path = "sub/../../evil.onnx".into();
        match verify_bundle(&manifest, &bundle).unwrap_err() {
            RunnerError::UnsafeModelPath { role, reason, .. } => {
                assert_eq!(role, ROLE_PREDICTION);
                assert!(reason.contains(".."), "got reason: {reason}");
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    /// A relative entry with no `..` can still escape via a symlink;
    /// the post-canonicalization containment check is what catches it.
    #[cfg(unix)]
    #[test]
    fn verify_bundle_rejects_symlink_that_canonicalizes_outside_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("bundle");
        std::fs::create_dir_all(&bundle).unwrap();
        let mut manifest = bundle_with_correct_digests(&bundle);
        let outside = tmp.path().join("outside.onnx");
        std::fs::write(&outside, b"outside-bytes").unwrap();
        let link = bundle.join("linked.onnx");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        manifest.files.representation = ModelFileEntry {
            path: "linked.onnx".into(),
            // Correct digest for the linked-to file: the point is that
            // containment is enforced independently of the digest.
            sha256: hex_encode(&Sha256::digest(b"outside-bytes")),
        };
        match verify_bundle(&manifest, &bundle).unwrap_err() {
            RunnerError::UnsafeModelPath { role, reason, .. } => {
                assert_eq!(role, ROLE_REPRESENTATION);
                assert!(
                    reason.contains("outside the bundle"),
                    "got reason: {reason}"
                );
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    /// A symlink that stays inside the bundle is legitimate (trainers
    /// publish `latest/` this way) and must be accepted.
    #[cfg(unix)]
    #[test]
    fn verify_bundle_accepts_symlink_that_stays_inside_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = bundle_with_correct_digests(tmp.path());
        let link = tmp.path().join("alias.onnx");
        std::os::unix::fs::symlink(tmp.path().join("dynamics.onnx"), &link).unwrap();
        manifest.files.dynamics.path = "alias.onnx".into();
        let verified = verify_bundle(&manifest, tmp.path()).unwrap();
        // Canonicalization resolves the alias back to the real file.
        assert!(verified.dynamics.ends_with("dynamics.onnx"));
    }

    #[test]
    fn verify_bundle_rejects_empty_entry_path() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = bundle_with_correct_digests(tmp.path());
        manifest.files.representation.path = String::new();
        match verify_bundle(&manifest, tmp.path()).unwrap_err() {
            RunnerError::UnsafeModelPath { reason, .. } => {
                assert!(reason.contains("empty"), "got reason: {reason}");
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    #[test]
    fn verify_bundle_missing_file_is_missing_path_error() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = bundle_with_correct_digests(tmp.path());
        manifest.files.prediction.path = "not-exported-yet.onnx".into();
        match verify_bundle(&manifest, tmp.path()).unwrap_err() {
            RunnerError::MissingPath { path } => {
                assert!(path.ends_with("not-exported-yet.onnx"));
            }
            other => panic!("expected MissingPath, got {other:?}"),
        }
    }

    #[test]
    fn verify_bundle_missing_bundle_dir_is_io_error() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = bundle_with_correct_digests(tmp.path());
        let absent = tmp.path().join("no-such-dir");
        match verify_bundle(&manifest, &absent).unwrap_err() {
            RunnerError::Io { path, .. } => assert_eq!(path, absent),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn resolve_bundle_path_accepts_current_dir_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        write_file(tmp.path(), "representation.onnx", b"x");
        let resolved =
            resolve_bundle_path(ROLE_REPRESENTATION, "./representation.onnx", tmp.path()).unwrap();
        assert!(resolved.ends_with("representation.onnx"));
        assert!(resolved.is_absolute());
    }

    #[test]
    fn digest_mismatch_error_message_names_role_and_both_digests() {
        let err = RunnerError::ModelDigestMismatch {
            role: ROLE_DYNAMICS.to_string(),
            path: PathBuf::from("/bundle/dynamics.onnx"),
            expected: "a".repeat(64),
            actual: "b".repeat(64),
        };
        let msg = err.to_string();
        assert!(msg.contains("dynamics"), "got: {msg}");
        assert!(msg.contains(&"a".repeat(64)), "got: {msg}");
        assert!(msg.contains(&"b".repeat(64)), "got: {msg}");
    }

    /// TEMPORARY REVIEW PROBE - not part of the PR.
    #[cfg(unix)]
    #[test]
    fn review_probe_sibling_prefix_dir_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("models");
        let sibling = tmp.path().join("models-attacker");
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let mut manifest = bundle_with_correct_digests(&bundle);
        let evil = sibling.join("evil.onnx");
        std::fs::write(&evil, b"evil-bytes").unwrap();
        std::os::unix::fs::symlink(&evil, bundle.join("linked.onnx")).unwrap();
        manifest.files.representation = ModelFileEntry {
            path: "linked.onnx".into(),
            sha256: hex_encode(&Sha256::digest(b"evil-bytes")),
        };
        let res = verify_bundle(&manifest, &bundle);
        assert!(res.is_err(), "ESCAPED THE BUNDLE: {:?}", res.unwrap());
    }
}

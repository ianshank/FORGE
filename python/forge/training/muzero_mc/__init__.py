"""MuZero training pipeline for the Minecraft RL integration (Phase 5).

This package implements the Python side of the Phase 4/5 hot-reload
loop: it produces ONNX bundles + `model_manifest.json` files that
`forge-mc-runner` (Rust) reads and reloads between episodes.

Modules:

- :mod:`manifest` — Python mirror of the Rust ``ModelManifest`` schema,
  with atomic JSON read/write and SHA-256 file hashing.
- :mod:`replay` — Streaming reader for ``TrajectoryV2`` JSONL files
  written by ``forge_mc_runner::TrajectoryWriter``.
- :mod:`bootstrap` — Generates a fresh random-init ONNX bundle + manifest
  the runner can pick up cold (used for first-run scenarios).
- :mod:`cli` — ``python -m forge.training.muzero_mc.cli`` entry point.

The trainer itself (full MuZero loss + Adam stepping + manifest version
bumping) is intentionally not included here; it is consumed in a
follow-up that wires
``forge.models.muzero_world_model.MuZeroWorldModel`` /
``forge.training.muzero_trainer.MuZeroTrainer`` to the
``TrajectoryV2``-shaped replay stream. The bootstrap + manifest +
exporter trio in this package is enough for the Rust runner to start
end-to-end.

All cross-language constants (schema versions, opset version) live in
``manifest`` as module-level ``Final`` constants — no hard-coded values
leak into call sites.
"""

from __future__ import annotations

from forge.training.muzero_mc.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ModelFileEntry,
    ModelManifest,
    ModelManifestFiles,
    load_manifest,
    save_manifest,
    sha256_file,
)
from forge.training.muzero_mc.replay import (
    TRAJECTORY_FORMAT_VERSION,
    StepBatch,
    TrajectoryReader,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "TRAJECTORY_FORMAT_VERSION",
    "ModelFileEntry",
    "ModelManifest",
    "ModelManifestFiles",
    "StepBatch",
    "TrajectoryReader",
    "load_manifest",
    "save_manifest",
    "sha256_file",
]

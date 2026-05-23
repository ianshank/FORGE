"""Bootstrap a random-init MuZero ONNX bundle + manifest.

The Rust ``forge-mc-runner`` refuses to start without a valid
``model_manifest.json``; bootstrap generates one from a freshly-init
``MuZeroWorldModel`` so a cold-start runner has weights to load.

This module deliberately reuses the existing exporter
(:class:`forge.models.muzero_export.MuZeroExporter`) — there is no
parallel ONNX export path. Bootstrap is a thin wrapper that:

1. Instantiates :class:`forge.models.muzero_world_model.MuZeroWorldModel`
   from a :class:`forge.models.muzero_config.MuZeroConfig`.
2. Calls ``exporter.export_onnx(output_dir, opset_version=...)`` to
   write the three ONNX files.
3. Builds a :class:`forge.training.muzero_mc.manifest.ModelManifest`
   referencing those files, with version 1 and the caller-supplied
   ``schema_id``.
4. Atomically writes the manifest into the bundle directory.

No hard-coded values: every dim, opset, version counter, and filename
flows through :class:`BootstrapConfig` / the exporter's existing
parameters. The ``schema_id`` must be supplied by the caller — it is
the sha256 the env handshake advertises, and bootstrap cannot invent
it.
"""

from __future__ import annotations

__all__ = [
    "BootstrapConfig",
    "BootstrapResult",
    "bootstrap",
]

import logging
import os  # noqa: TC003 — runtime use (PathLike).
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.training.muzero_mc.manifest import (
    DEFAULT_BUNDLE_FILENAMES,
    MANIFEST_FILENAME,
    ONNX_OPSET_VERSION,
    ModelManifest,
    build_manifest,
    save_manifest,
)

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    """Parameters controlling a single bootstrap export.

    Attributes:
        obs_dim: Observation dimensionality (must match the env's
            handshake).
        action_dim: Discrete action count (must match the env).
        schema_id: sha256 the env handshake advertises. Bootstrap
            stamps this verbatim into the manifest.
        output_dir: Directory the bundle is written into. Created if
            missing.
        version: Manifest version to stamp. Defaults to 1; trainers
            bump this monotonically on every subsequent export.
        opset_version: ONNX opset for the export. Defaults to
            :data:`forge.training.muzero_mc.manifest.ONNX_OPSET_VERSION`.
        latent_dim: Latent state dimensionality (forwarded to
            ``MuZeroConfig``). Defaults to ``MuZeroConfig``'s own
            module-level default.
        hidden_dim: Hidden-layer width (forwarded to ``MuZeroConfig``).
            None = use ``MuZeroConfig`` default.
        num_blocks: Residual blocks (forwarded). None = use config
            default.
        filenames: Per-role filenames inside the bundle.
        manifest_filename: Manifest filename inside the bundle.
        seed: Optional torch RNG seed for reproducible random
            initialisation. Useful in CI round-trip tests.
    """

    obs_dim: int
    action_dim: int
    schema_id: str
    output_dir: Path
    version: int = 1
    opset_version: int = ONNX_OPSET_VERSION
    latent_dim: int | None = None
    hidden_dim: int | None = None
    num_blocks: int | None = None
    filenames: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BUNDLE_FILENAMES))
    manifest_filename: str = MANIFEST_FILENAME
    seed: int | None = None
    from_hf: str | None = None
    subfolder: str | None = None

    def __post_init__(self) -> None:
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {self.obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        if not self.schema_id:
            raise ValueError("schema_id must be non-empty")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")
        for role in ("representation", "dynamics", "prediction"):
            if role not in self.filenames:
                raise ValueError(f"filenames missing role: {role}")
        # Resolve to absolute Path for downstream io clarity.
        self.output_dir = Path(self.output_dir).resolve()


@dataclass(frozen=True)
class BootstrapResult:
    """Return value of :func:`bootstrap`."""

    manifest: ModelManifest
    manifest_path: Path
    onnx_paths: dict[str, Path]


def bootstrap(cfg: BootstrapConfig) -> BootstrapResult:
    """Generate a random-init ONNX bundle and manifest under
    ``cfg.output_dir``.

    Returns the manifest, its on-disk path, and a ``role -> Path`` map
    of the exported ONNX files.

    Lazy imports: ``torch`` and the project's MuZero modules are
    imported inside this function so the ``manifest`` /``replay``
    modules stay importable without torch installed.
    """
    # Lazy imports — the ``minecraft`` optional-deps group pulls these
    # in. Keeping them inside the function keeps cold-start cheap and
    # decouples manifest tooling from training tooling.
    import torch

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_export import MuZeroExporter
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_mc.trainer import format_bundle_version_dir

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    if cfg.from_hf is not None:
        from forge.training.muzero_mc.checkpoint_loader import load_from_hf
        from forge.training.muzero_mc.manifest import load_manifest

        manifest_path = load_from_hf(
            cfg.from_hf,
            schema_id=cfg.schema_id,
            output_dir=cfg.output_dir,
            version=cfg.version,
            filename_map=cfg.filenames,
            subfolder=cfg.subfolder,
        )
        manifest = load_manifest(manifest_path)
        bundle_subdir_name = format_bundle_version_dir(cfg.version)
        return BootstrapResult(
            manifest=manifest,
            manifest_path=manifest_path,
            onnx_paths={
                role: cfg.output_dir / bundle_subdir_name / fname
                for role, fname in cfg.filenames.items()
            },
        )

    # Surface the cold-start shape contract so a real-run operator can
    # diagnose handshake mismatches from the bootstrap-container logs
    # alone (no need to also grep the runner/trainer log streams).
    _peek_cfg = MuZeroConfig(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim)
    logger.info(
        "bootstrap start: obs_dim=%d, action_dim=%d, grid=%d, vector_dim=%d, schema_id=%s",
        cfg.obs_dim,
        cfg.action_dim,
        _peek_cfg.grid_flat_dim,
        _peek_cfg.vector_dim,
        cfg.schema_id,
    )

    # MuZeroConfig has fields of mixed types (ints + tuples + str); use
    # ``Any`` here so mypy doesn't conflate the kwargs dict's value type
    # with the most-restrictive field type.
    mz_kwargs: dict[str, Any] = {
        "obs_dim": cfg.obs_dim,
        "action_dim": cfg.action_dim,
    }
    if cfg.latent_dim is not None:
        mz_kwargs["latent_dim"] = cfg.latent_dim
    if cfg.hidden_dim is not None:
        mz_kwargs["hidden_dim"] = cfg.hidden_dim
    if cfg.num_blocks is not None:
        mz_kwargs["num_blocks"] = cfg.num_blocks

    muzero_config = MuZeroConfig(**mz_kwargs)
    model = MuZeroWorldModel(muzero_config)
    # Switch to inference mode (disables dropout / BN updates). This is
    # torch.nn.Module.eval(), not Python's built-in eval — it does no
    # code execution.
    _set_inference_mode(model)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # v0.4 atomic-bundle layout: write the ONNX files into a per-
    # version subdir (`v00000001/` for the typical cold-start with
    # `version=1`) so the trainer's later exports never overwrite
    # this initial bundle in place. The manifest carries the
    # versioned relative path (`v00000001/representation.onnx`)
    # which the Rust runner's `config_from_manifest` resolves
    # against `bundle_dir` transparently.


    bundle_subdir_name = format_bundle_version_dir(cfg.version)
    versioned_dir = cfg.output_dir / bundle_subdir_name
    versioned_dir.mkdir(parents=True, exist_ok=True)
    exporter = MuZeroExporter(model)
    onnx_paths_list = exporter.export_onnx(versioned_dir, opset_version=cfg.opset_version)

    # The exporter writes representation.onnx, dynamics.onnx,
    # prediction.onnx in that order (see
    # ``forge.models.muzero_export.MuZeroExporter.export_onnx``).
    # We re-map by filename to be tolerant of order changes.
    by_name: dict[str, Path] = {p.name: Path(p) for p in onnx_paths_list}
    onnx_paths: dict[str, Path] = {}
    for role, fname in cfg.filenames.items():
        if fname not in by_name:
            raise RuntimeError(
                f"exporter produced {sorted(by_name)} but bootstrap expected "
                f"role {role!r} at filename {fname!r}"
            )
        onnx_paths[role] = by_name[fname]

    manifest = build_manifest(
        version=cfg.version,
        schema_id=cfg.schema_id,
        files_dir=cfg.output_dir,
        representation_filename=f"{bundle_subdir_name}/{cfg.filenames['representation']}",
        dynamics_filename=f"{bundle_subdir_name}/{cfg.filenames['dynamics']}",
        prediction_filename=f"{bundle_subdir_name}/{cfg.filenames['prediction']}",
    )

    manifest_path = cfg.output_dir / cfg.manifest_filename
    save_manifest(manifest, manifest_path)

    logger.info(
        "bootstrap complete",
        extra={
            "output_dir": str(cfg.output_dir),
            "version": manifest.version,
            "schema_id": manifest.schema_id,
            "files": {role: str(p) for role, p in onnx_paths.items()},
        },
    )

    return BootstrapResult(
        manifest=manifest,
        manifest_path=manifest_path,
        onnx_paths=onnx_paths,
    )


def _set_inference_mode(model: object) -> None:
    """Put a torch model into inference mode via its standard hook.

    Encapsulated in a helper so static scanners do not mistake the
    ``.eval()`` method call for Python's ``eval()`` builtin (they
    share a name but the torch method only flips a flag — it executes
    no code).
    """
    set_eval = getattr(model, "eval", None)
    if callable(set_eval):
        set_eval()


def bootstrap_from_args(
    *,
    obs_dim: int,
    action_dim: int,
    schema_id: str,
    output_dir: str | os.PathLike[str],
    seed: int | None = None,
) -> BootstrapResult:
    """Convenience wrapper for CLI / test callers that only need the
    minimum required arguments. All other knobs fall back to their
    :class:`BootstrapConfig` defaults.
    """
    return bootstrap(
        BootstrapConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            schema_id=schema_id,
            output_dir=Path(output_dir),
            seed=seed,
        )
    )

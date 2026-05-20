"""Command-line entry point for the ``muzero_mc`` package.

Two subcommands are implemented:

- ``bootstrap`` — generate a fresh random-init ONNX bundle + manifest
  for the Rust runner to load on cold start.
- ``validate-manifest`` — load and validate a manifest file, exit 0
  on success and non-zero on any schema/integrity failure.

Run with ``python -m forge.training.muzero_mc.cli <subcommand> ...``.

No hard-coded values: every CLI flag has a default that points at the
module-level constant it overrides (e.g. ``--opset-version`` defaults
to :data:`forge.training.muzero_mc.manifest.ONNX_OPSET_VERSION`).
"""

from __future__ import annotations

__all__ = ["build_parser", "main"]

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from forge.training.muzero_mc.bootstrap import BootstrapConfig, bootstrap
from forge.training.muzero_mc.manifest import (
    MANIFEST_FILENAME,
    ONNX_OPSET_VERSION,
    ManifestError,
    load_manifest,
)

logger = logging.getLogger(__name__)

# Exit codes consumed by CI / shell orchestration.
EXIT_OK: int = 0
EXIT_USAGE: int = 2
EXIT_VALIDATION: int = 3
EXIT_IO: int = 4


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Exposed publicly so tests can introspect the CLI surface without
    actually invoking ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        prog="forge.training.muzero_mc.cli",
        description=(
            "Bootstrap and validate MuZero model bundles for the Rust "
            "forge-mc-runner."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logging level (default: INFO).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # bootstrap
    p_boot = sub.add_parser(
        "bootstrap",
        help="Generate a random-init ONNX bundle + model_manifest.json.",
    )
    p_boot.add_argument(
        "--obs-dim",
        type=int,
        required=True,
        help="Observation dimensionality (must match env handshake).",
    )
    p_boot.add_argument(
        "--action-dim",
        type=int,
        required=True,
        help="Discrete action count (must match env handshake).",
    )
    p_boot.add_argument(
        "--schema-id",
        type=str,
        required=True,
        help=(
            "sha256 the env handshake advertises. The runner cross-"
            "checks this against the manifest at startup."
        ),
    )
    p_boot.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Bundle output directory (created if missing).",
    )
    p_boot.add_argument(
        "--version",
        type=int,
        default=1,
        help="Manifest version to stamp (default: 1).",
    )
    p_boot.add_argument(
        "--opset-version",
        type=int,
        default=ONNX_OPSET_VERSION,
        help=f"ONNX opset (default: {ONNX_OPSET_VERSION}).",
    )
    p_boot.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Latent state dim (default: MuZeroConfig default).",
    )
    p_boot.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden layer dim (default: MuZeroConfig default).",
    )
    p_boot.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Residual blocks (default: MuZeroConfig default).",
    )
    p_boot.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional torch RNG seed for reproducibility.",
    )

    # compute-schema-id
    p_sid = sub.add_parser(
        "compute-schema-id",
        help=(
            "Compute the canonical SHA256 ``schema_id`` for a given "
            "action_map.toml + rewards.toml pair. The same hash the "
            "Rust env adapter + mc-bot agree on at handshake. Use "
            "``--quiet`` to suppress logs so stdout is hash-only "
            "(for shell substitution into `bootstrap --schema-id`)."
        ),
    )
    p_sid.add_argument(
        "--action-map",
        type=Path,
        required=True,
        help="Path to configs/minecraft/action_map.toml (or equivalent).",
    )
    p_sid.add_argument(
        "--rewards",
        type=Path,
        required=True,
        help="Path to configs/minecraft/rewards.toml (or equivalent).",
    )
    p_sid.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress INFO logs on stderr; print ONLY the 64-hex hash on "
            "stdout (no trailing newline change). Useful for "
            "``SCHEMA_ID=$(... compute-schema-id --quiet)``."
        ),
    )

    # validate-manifest
    p_val = sub.add_parser(
        "validate-manifest",
        help="Load and validate an existing manifest file.",
    )
    p_val.add_argument(
        "path",
        type=Path,
        help=(
            "Path to model_manifest.json. May be either the file itself "
            f"or a directory containing {MANIFEST_FILENAME}."
        ),
    )

    # train
    from forge.training.muzero_mc.trainer import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EXPORT_EVERY_N_ITERS,
        DEFAULT_LOG_EVERY_N_ITERS,
    )

    p_train = sub.add_parser(
        "train",
        help=(
            "Run MuZero training against a directory of TrajectoryV2 "
            "files. Periodically exports an ONNX bundle + bumps the "
            "manifest version the Rust runner hot-reloads."
        ),
    )
    p_train.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing ep-*.json[.gz] trajectory files.",
    )
    p_train.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Bundle output directory (ONNX + manifest).",
    )
    p_train.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Manifest path (default: <out>/model_manifest.json). The "
            "trainer resumes the version counter if this file exists "
            "already."
        ),
    )
    p_train.add_argument(
        "--schema-id",
        type=str,
        required=True,
        help="sha256 the env handshake advertises (stamped on every export).",
    )
    p_train.add_argument(
        "--obs-dim",
        type=int,
        required=True,
        help="Observation dim (must match the env + trajectory files).",
    )
    p_train.add_argument(
        "--action-dim",
        type=int,
        required=True,
        help="Discrete action count.",
    )
    p_train.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Total gradient steps to run.",
    )
    p_train.add_argument(
        "--export-every",
        type=int,
        default=DEFAULT_EXPORT_EVERY_N_ITERS,
        help=f"Export cadence (default: {DEFAULT_EXPORT_EVERY_N_ITERS}). 0 disables mid-run exports.",
    )
    p_train.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY_N_ITERS,
        help=f"Log cadence (default: {DEFAULT_LOG_EVERY_N_ITERS}).",
    )
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    p_train.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="MuZeroConfig.latent_dim (default: config default).",
    )
    p_train.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="MuZeroConfig.hidden_dim (default: config default).",
    )
    p_train.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="MuZeroConfig.num_blocks (default: config default).",
    )
    p_train.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0). Determinism gate for tests.",
    )
    from forge.training.muzero_mc.trainer import DEFAULT_DEVICE

    p_train.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "auto"],
        help=(
            f"Torch device for training (default: {DEFAULT_DEVICE!r}). "
            "'auto' picks CUDA if torch.cuda.is_available() else CPU."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Library entry point. Returns the process exit code.

    Tests can call ``main([...])`` directly; the ``__main__`` block at
    the bottom of this file forwards to ``sys.exit(main())``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # `compute-schema-id --quiet` routes logs to stderr only so stdout
    # carries the hash and nothing else. Other subcommands keep the
    # existing INFO-to-stdout default.
    log_stream = sys.stderr if getattr(args, "quiet", False) else None
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=log_stream,
    )

    if args.cmd == "bootstrap":
        return _run_bootstrap(args)
    if args.cmd == "validate-manifest":
        return _run_validate(args)
    if args.cmd == "train":
        return _run_train(args)
    if args.cmd == "compute-schema-id":
        return _run_compute_schema_id(args)
    parser.error(f"unknown command: {args.cmd!r}")  # pragma: no cover — argparse blocks this
    return EXIT_USAGE


def _run_bootstrap(args: argparse.Namespace) -> int:
    try:
        result = bootstrap(
            BootstrapConfig(
                obs_dim=args.obs_dim,
                action_dim=args.action_dim,
                schema_id=args.schema_id,
                output_dir=args.out,
                version=args.version,
                opset_version=args.opset_version,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                seed=args.seed,
            )
        )
    except ValueError as e:
        logger.error("invalid bootstrap config: %s", e)
        return EXIT_USAGE
    except (OSError, RuntimeError) as e:
        logger.error("bootstrap failed: %s", e)
        return EXIT_IO

    logger.info(
        "wrote manifest at %s (version=%d, schema_id=%s)",
        result.manifest_path,
        result.manifest.version,
        result.manifest.schema_id,
    )
    return EXIT_OK


def _run_compute_schema_id(args: argparse.Namespace) -> int:
    """``compute-schema-id`` subcommand entry.

    Loads action_map.toml + rewards.toml, computes the canonical
    combined SHA256 the Rust env adapter + mc-bot agree on at
    handshake, prints it to stdout. ``--quiet`` ensures stdout is
    hash-only (no logs, no trailing diagnostics) so callers can do
    ``SCHEMA_ID=$(... compute-schema-id --quiet)`` without scrubbing.
    """
    from forge.training.muzero_mc.schema_id import compute_schema_id_from_paths

    if not args.action_map.exists():
        logger.error("--action-map path missing: %s", args.action_map)
        return EXIT_IO
    if not args.rewards.exists():
        logger.error("--rewards path missing: %s", args.rewards)
        return EXIT_IO
    try:
        schema_id = compute_schema_id_from_paths(args.action_map, args.rewards)
    except (OSError, ValueError, KeyError) as e:
        logger.error("schema-id computation failed: %s", e)
        return EXIT_VALIDATION

    # Hash to stdout; logs (if not --quiet) go through the standard
    # logger which writes to stderr when --quiet is set.
    print(schema_id)
    if not args.quiet:
        logger.info(
            "schema_id=%s (action_map=%s, rewards=%s)",
            schema_id,
            args.action_map,
            args.rewards,
        )
    return EXIT_OK


def _run_validate(args: argparse.Namespace) -> int:
    target = args.path
    if target.is_dir():
        target = target / MANIFEST_FILENAME
    try:
        manifest = load_manifest(target)
    except FileNotFoundError:
        logger.error("manifest not found at %s", target)
        return EXIT_IO
    except ManifestError as e:
        logger.error("manifest invalid: %s", e)
        return EXIT_VALIDATION
    except ValueError as e:
        logger.error("manifest malformed: %s", e)
        return EXIT_VALIDATION

    logger.info(
        "manifest OK: %s (version=%d, schema_id=%s)",
        target,
        manifest.version,
        manifest.schema_id,
    )
    return EXIT_OK


def _run_train(args: argparse.Namespace) -> int:
    # Fail fast on a missing input directory BEFORE the heavy torch
    # import — this lets the CLI surface clean diagnostics in
    # environments where torch isn't installed (e.g. lint-only CI).
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("train --input directory does not exist: %s", input_dir)
        return EXIT_IO
    if not input_dir.is_dir():
        logger.error("train --input path is not a directory: %s", input_dir)
        return EXIT_IO

    try:
        # Local imports: torch is an optional dep + trainer.py is only
        # importable when `pip install -e .[minecraft]` has been run.
        from forge.models.muzero_config import MuZeroConfig
        from forge.models.muzero_world_model import MuZeroWorldModel
        from forge.training.muzero_mc.replay import TrajectoryReader
        from forge.training.muzero_mc.trainer import (
            MuzeroMcTrainer,
            MuZeroMcTrainerConfig,
        )
    except ImportError as e:
        logger.error(
            "train subcommand requires the [minecraft] optional deps "
            "(torch + onnx + onnxruntime): %s",
            e,
        )
        return EXIT_IO

    try:
        mz_kwargs: dict[str, Any] = {
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
        }
        if args.latent_dim is not None:
            mz_kwargs["latent_dim"] = args.latent_dim
        if args.hidden_dim is not None:
            mz_kwargs["hidden_dim"] = args.hidden_dim
        if args.num_blocks is not None:
            mz_kwargs["num_blocks"] = args.num_blocks
        model_cfg = MuZeroConfig(**mz_kwargs)
        model = MuZeroWorldModel(model_cfg)

        reader = TrajectoryReader(args.input, batch_size=args.batch_size)
        trainer_cfg = MuZeroMcTrainerConfig(
            train_iters=args.iters,
            export_every_n_iters=args.export_every,
            log_every_n_iters=args.log_every,
            output_dir=args.out,
            manifest_path=args.manifest,
            schema_id=args.schema_id,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        trainer = MuzeroMcTrainer(model, reader, trainer_cfg)
        outcome = trainer.train()
    except ValueError as e:
        logger.error("invalid train config: %s", e)
        return EXIT_USAGE
    except (OSError, RuntimeError) as e:
        logger.error("train failed: %s", e)
        return EXIT_IO

    logger.info(
        "train complete: iters=%d, exports=%d, manifest_version=%d",
        outcome["iters_completed"],
        outcome["exports"],
        outcome["last_manifest_version"],
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover — module-as-script.
    sys.exit(main())

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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Library entry point. Returns the process exit code.

    Tests can call ``main([...])`` directly; the ``__main__`` block at
    the bottom of this file forwards to ``sys.exit(main())``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "bootstrap":
        return _run_bootstrap(args)
    if args.cmd == "validate-manifest":
        return _run_validate(args)
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


if __name__ == "__main__":  # pragma: no cover — module-as-script.
    sys.exit(main())

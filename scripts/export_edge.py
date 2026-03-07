#!/usr/bin/env python3
"""ONNX model export stub for edge deployment.

Usage::

    python scripts/export_edge.py --checkpoint checkpoints/best --output model.onnx
"""

from __future__ import annotations

import argparse
import logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export FORGE model to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint to export"
    )
    parser.add_argument("--output", type=str, default="model.onnx", help="Output path")
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    return parser.parse_args()


def main() -> None:
    """Export model to ONNX format."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("forge.export")

    logger.info(
        "Exporting checkpoint=%s to %s (opset %d)",
        args.checkpoint,
        args.output,
        args.opset,
    )

    # ONNX export requires torch — stub implementation
    logger.warning(
        "ONNX export is a stub. Install torch and implement model export."
    )
    logger.info("Export complete (stub)")


if __name__ == "__main__":
    main()

"""Cross-file version-pin consistency check.

Two version pins are duplicated by necessity (no single file a Dockerfile
`ARG` *and* a GitHub Actions `with:` input can both read from without much
more invasive templating): the Rust toolchain version and the ONNX Runtime
version used by ``docker/mc-runner.Dockerfile`` + the `onnx-features` CI
job. Nothing enforced that the copies stayed in sync -- e.g. `rust-
toolchain.toml` could be bumped without touching the 15 `dtolnay/rust-
toolchain@stable` `toolchain:` inputs across 6 workflow files, silently
leaving CI on the old compiler while local dev moved to the new one.

This script re-derives each duplicate's value from its file and compares
it against a canonical source, so drift becomes a CI failure instead of a
silent divergence. It intentionally does NOT touch
``docker/trainer.Dockerfile``'s own ``ONNXRUNTIME_VERSION`` -- that pins
the *Python* `onnxruntime` wheel for the trainer image, a different
artifact on a different release cadence than the C++ redistributable the
Rust `ort` crate dlopens; coupling them would be wrong, not a fix.

Invocation
----------

    python scripts/check_version_consistency.py

Exit codes
----------

* ``0`` -- every duplicate matches its canonical source.
* ``1`` -- one or more duplicates have drifted; see stdout for the diff.
* ``2`` -- a canonical source file is missing or its pin couldn't be parsed.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_INPUT_ERROR = 2

REPO_ROOT = Path(__file__).resolve().parent.parent

# Workflow files that pin a Rust toolchain via `dtolnay/rust-toolchain@stable`.
# (Not a bare glob over .github/workflows/*.yml: some workflows in that
# directory don't touch Rust at all, and silently accepting "zero matches"
# from a file that never had a toolchain pin would defeat the point of
# this check -- an explicit list makes a newly-added Rust job opt in.)
RUST_TOOLCHAIN_WORKFLOWS = (
    ".github/workflows/ci.yml",
    ".github/workflows/e2e-long.yml",
    ".github/workflows/gh-pages.yml",
    ".github/workflows/hf-dataset.yml",
    ".github/workflows/hf-space.yml",
)

RUST_TOOLCHAIN_DOCKERFILES = (
    "docker/Dockerfile",
    "docker/mc-runner.Dockerfile",
)


@dataclass(frozen=True)
class Occurrence:
    file: str
    line: int
    value: str


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)


def _find_all(path: Path, pattern: re.Pattern[str]) -> list[Occurrence]:
    text = _read(path)
    rel = str(path.relative_to(REPO_ROOT))
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            hits.append(Occurrence(rel, lineno, m.group(1)))
    return hits


def _canonical_rust_toolchain() -> str:
    path = REPO_ROOT / "rust-toolchain.toml"
    m = re.search(r'channel\s*=\s*"([^"]+)"', _read(path))
    if not m:
        print(f"ERROR: no `channel = \"...\"` found in {path}", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)
    return m.group(1)


def _canonical_onnxruntime_version() -> str:
    path = REPO_ROOT / "docker" / "mc-runner.Dockerfile"
    m = re.search(r"ARG\s+ONNXRUNTIME_VERSION=(\S+)", _read(path))
    if not m:
        print(f"ERROR: no `ARG ONNXRUNTIME_VERSION=...` default found in {path}", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)
    return m.group(1)


def check_rust_toolchain() -> list[str]:
    canonical = _canonical_rust_toolchain()
    mismatches = []

    toolchain_re = re.compile(r'toolchain:\s*"([^"]+)"')
    for rel in RUST_TOOLCHAIN_WORKFLOWS:
        occurrences = _find_all(REPO_ROOT / rel, toolchain_re)
        if not occurrences:
            mismatches.append(f"{rel}: expected a toolchain: \"{canonical}\" pin, found none")
        mismatches.extend(
            f"{occ.file}:{occ.line}: toolchain \"{occ.value}\" != rust-toolchain.toml's \"{canonical}\""
            for occ in occurrences
            if occ.value != canonical
        )

    image_tag_re = re.compile(r"ARG\s+RUST_IMAGE_TAG=([0-9][^\s-]*)-bookworm")
    for rel in RUST_TOOLCHAIN_DOCKERFILES:
        occurrences = _find_all(REPO_ROOT / rel, image_tag_re)
        if not occurrences:
            mismatches.append(f"{rel}: expected an ARG RUST_IMAGE_TAG={canonical}-bookworm default, found none")
        mismatches.extend(
            f"{occ.file}:{occ.line}: RUST_IMAGE_TAG \"{occ.value}\" != rust-toolchain.toml's \"{canonical}\""
            for occ in occurrences
            if occ.value != canonical
        )
    return mismatches


def check_onnxruntime_version() -> list[str]:
    canonical = _canonical_onnxruntime_version()
    mismatches = []

    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    occurrences = _find_all(ci_path, re.compile(r"ORT_VERSION=(\S+)"))
    if not occurrences:
        mismatches.append(
            f"{ci_path.relative_to(REPO_ROOT)}: expected an ORT_VERSION={canonical} pin "
            "in the onnx-features job, found none"
        )
    mismatches.extend(
        f"{occ.file}:{occ.line}: ORT_VERSION \"{occ.value}\" != "
        f"docker/mc-runner.Dockerfile's ONNXRUNTIME_VERSION \"{canonical}\""
        for occ in occurrences
        if occ.value != canonical
    )
    return mismatches


def main() -> int:
    mismatches = [*check_rust_toolchain(), *check_onnxruntime_version()]
    if mismatches:
        print("Version pin drift detected:")
        for m in mismatches:
            print(f"  - {m}")
        print(
            "\nEvery occurrence above must match its canonical source "
            "(rust-toolchain.toml's channel, or docker/mc-runner.Dockerfile's "
            "ONNXRUNTIME_VERSION) -- update the drifted line(s), or update the "
            "canonical source and every dependent line together."
        )
        return EXIT_MISMATCH

    print("OK: all Rust toolchain and ONNX Runtime version pins are consistent.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

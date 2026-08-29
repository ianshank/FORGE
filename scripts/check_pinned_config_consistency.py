"""Cross-file pinned-config consistency check.

Three values are duplicated by necessity -- no single file a Dockerfile
`ARG`, a GitHub Actions `env:`/`with:` entry, and a Python/TOML source can
all read from without much more invasive templating -- so each has one
canonical source and one or more dependent copies that must be kept in
lock-step by hand:

* The Rust toolchain version (canonical: ``rust-toolchain.toml``), copied
  into 15 `dtolnay/rust-toolchain@stable` `toolchain:` inputs across 5
  workflow files and 2 Dockerfiles' `RUST_IMAGE_TAG`.
* The ONNX Runtime version for the Rust `ort` crate (canonical:
  ``docker/mc-runner.Dockerfile``'s `ONNXRUNTIME_VERSION`), copied into
  `ci.yml`'s `onnx-features` job as `ORT_VERSION`. Intentionally does NOT
  touch ``docker/trainer.Dockerfile``'s own `ONNXRUNTIME_VERSION` -- that
  pins the *Python* `onnxruntime` wheel for the trainer image, a different
  artifact on a different release cadence than the C++ redistributable the
  Rust `ort` crate dlopens; coupling them would be wrong, not a fix.
* The LM Studio port/base-URL (canonical:
  ``python/forge/cognitive/providers.py``'s `DEFAULT_LMSTUDIO_BASE_URL`),
  copied into `ci.yml`'s workflow-level `LMSTUDIO_PORT`/`LMSTUDIO_BASE_URL`
  env (already comment-annotated as "keep in lock-step" -- this script
  makes that comment enforced, not just requested) and
  `e2e-long.yml`'s own `LMSTUDIO_PORT`, which had no such comment at all.
* The pinned `wasm-pack` release version (canonical: `gh-pages.yml`'s
  `WASM_PACK_VERSION` step env -- picked as canonical because it's the
  primary Pages deploy target; `hf-space.yml`'s own header comment
  already calls itself "Companion to gh-pages.yml -- same build, second
  publish target"), copied into `hf-space.yml`'s identical `Install
  wasm-pack` step. Both steps used to take `version:` as a declarative
  action input; replacing `jetli/wasm-pack-action` with a plain shell
  install (see its step comment for why) turned that single input into
  a hand-maintained literal duplicated across two files, with nothing
  catching a future "bumped one, forgot the other" edit.

Nothing previously enforced that any of these copies stayed in sync --
e.g. `rust-toolchain.toml` could be bumped without touching a single one
of its 15 dependent `toolchain:` lines, silently leaving CI on the old
compiler while local dev moved to the new one. This script re-derives
each duplicate's value from its file and compares it against its
canonical source, so drift becomes a CI failure instead of a silent
divergence.

Invocation
----------

    python scripts/check_pinned_config_consistency.py

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

# Workflow files that pin the LM Studio port/base-URL as an env default for
# their opt-in live-smoke jobs. Same rationale as RUST_TOOLCHAIN_WORKFLOWS
# above: an explicit list so a newly-added workflow has to opt in rather
# than silently going unchecked.
LMSTUDIO_WORKFLOWS = (
    ".github/workflows/ci.yml",
    ".github/workflows/e2e-long.yml",
)

# The wasm-pack pin is a YAML `env:` key, so anchor to the start of the
# (whitespace-indented) line: an unanchored pattern would also match the
# substring inside a `run:` shell command -- e.g. the very install script
# this pin feeds, `curl ... v${WASM_PACK_VERSION}/...`, in a future
# refactor that inlines the value -- and count it as a second pin. One
# shared compiled pattern for the canonical and dependent lookups so the
# two sides can't drift apart in what they consider a pin.
WASM_PACK_PIN_RE = re.compile(r'^\s*WASM_PACK_VERSION:\s*"([^"]+)"')


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


def _strip_comment(line: str) -> str:
    """Truncate `line` at its first `#` (TOML/YAML/Dockerfile/Python all
    use `#` for a line comment), so a commented-out stale pin like
    `# toolchain: "1.60.0"` isn't parsed as a live occurrence -- confirmed
    a real false-positive risk, not a hypothetical one: every pattern this
    script looks for is a simple `KEY: "VALUE"` / `ARG KEY=VALUE` shape
    that never legitimately contains a literal `#` inside the value
    itself, so truncating at the first `#` regardless of quoting is safe
    here without needing a full comment-aware tokenizer.
    """
    return line.split("#", 1)[0]


def _find_all(path: Path, pattern: re.Pattern[str]) -> list[Occurrence]:
    text = _read(path)
    rel = str(path.relative_to(REPO_ROOT))
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(_strip_comment(line))
        if m:
            hits.append(Occurrence(rel, lineno, m.group(1)))
    return hits


def _canonical(path: Path, pattern: re.Pattern[str], description: str) -> str:
    """First non-comment match of `pattern` in `path`, or exit with an error."""
    hits = _find_all(path, pattern)
    if not hits:
        print(f"ERROR: no {description} found in {path}", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)
    return hits[0].value


def _canonical_rust_toolchain() -> str:
    path = REPO_ROOT / "rust-toolchain.toml"
    return _canonical(path, re.compile(r'channel\s*=\s*"([^"]+)"'), 'a `channel = "..."` line')


def _canonical_onnxruntime_version() -> str:
    path = REPO_ROOT / "docker" / "mc-runner.Dockerfile"
    return _canonical(
        path, re.compile(r"ARG\s+ONNXRUNTIME_VERSION=(\S+)"), "an `ARG ONNXRUNTIME_VERSION=...` default"
    )


def _canonical_wasm_pack_version() -> str:
    path = REPO_ROOT / ".github" / "workflows" / "gh-pages.yml"
    return _canonical(
        path, WASM_PACK_PIN_RE, 'a `WASM_PACK_VERSION: "..."` step env'
    )


def _canonical_lmstudio_base_url() -> tuple[str, str]:
    """Return (base_url, port) derived from the Python default."""
    path = REPO_ROOT / "python" / "forge" / "cognitive" / "providers.py"
    base_url = _canonical(
        path,
        re.compile(r'DEFAULT_LMSTUDIO_BASE_URL\s*:\s*str\s*=\s*"([^"]+)"'),
        'a `DEFAULT_LMSTUDIO_BASE_URL: str = "..."` line',
    )
    port_match = re.search(r":(\d+)/", base_url)
    if not port_match:
        print(f"ERROR: could not derive a port from DEFAULT_LMSTUDIO_BASE_URL={base_url!r}", file=sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)
    return base_url, port_match.group(1)


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


def check_wasm_pack_version() -> list[str]:
    # Only hf-space.yml (the dependent) is scanned below; the canonical is
    # gh-pages.yml's FIRST pin (see _canonical), and any extra occurrence
    # inside gh-pages.yml itself is not cross-checked -- unlike
    # check_rust_toolchain, which re-scans every listed file including the
    # canonical's own. Fine while each file carries exactly one step-level
    # pin; revisit if gh-pages.yml ever grows a second WASM_PACK_VERSION.
    canonical = _canonical_wasm_pack_version()
    mismatches = []

    dependent = REPO_ROOT / ".github" / "workflows" / "hf-space.yml"
    occurrences = _find_all(dependent, WASM_PACK_PIN_RE)
    if not occurrences:
        mismatches.append(
            f"{dependent.relative_to(REPO_ROOT)}: expected a "
            f'WASM_PACK_VERSION: "{canonical}" pin, found none'
        )
    mismatches.extend(
        f'{occ.file}:{occ.line}: WASM_PACK_VERSION "{occ.value}" != '
        f'gh-pages.yml\'s WASM_PACK_VERSION "{canonical}"'
        for occ in occurrences
        if occ.value != canonical
    )
    return mismatches


def check_lmstudio_endpoint() -> list[str]:
    canonical_url, canonical_port = _canonical_lmstudio_base_url()
    mismatches = []

    port_re = re.compile(r'LMSTUDIO_PORT:\s*"(\d+)"')
    url_re = re.compile(r'LMSTUDIO_BASE_URL:\s*"([^"]+)"')
    for rel in LMSTUDIO_WORKFLOWS:
        port_occurrences = _find_all(REPO_ROOT / rel, port_re)
        if not port_occurrences:
            mismatches.append(f"{rel}: expected an LMSTUDIO_PORT: \"{canonical_port}\" pin, found none")
        mismatches.extend(
            f"{occ.file}:{occ.line}: LMSTUDIO_PORT \"{occ.value}\" != "
            f"providers.py's DEFAULT_LMSTUDIO_BASE_URL port \"{canonical_port}\""
            for occ in port_occurrences
            if occ.value != canonical_port
        )

        url_occurrences = _find_all(REPO_ROOT / rel, url_re)
        mismatches.extend(
            f"{occ.file}:{occ.line}: LMSTUDIO_BASE_URL \"{occ.value}\" != "
            f"providers.py's DEFAULT_LMSTUDIO_BASE_URL \"{canonical_url}\""
            for occ in url_occurrences
            if occ.value != canonical_url
        )
    return mismatches


def main() -> int:
    mismatches = [
        *check_rust_toolchain(),
        *check_onnxruntime_version(),
        *check_wasm_pack_version(),
        *check_lmstudio_endpoint(),
    ]
    if mismatches:
        print("Pinned-config drift detected:")
        for m in mismatches:
            print(f"  - {m}")
        print(
            "\nEvery occurrence above must match its canonical source "
            "(rust-toolchain.toml's channel, docker/mc-runner.Dockerfile's "
            "ONNXRUNTIME_VERSION, gh-pages.yml's WASM_PACK_VERSION, or "
            "providers.py's DEFAULT_LMSTUDIO_BASE_URL) -- update the drifted "
            "line(s), or update the canonical source and every dependent "
            "line together."
        )
        return EXIT_MISMATCH

    print(
        "OK: all Rust toolchain, ONNX Runtime, wasm-pack version, and "
        "LM Studio endpoint pins are consistent."
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

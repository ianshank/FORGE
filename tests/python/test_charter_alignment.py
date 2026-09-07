"""Guards ``docs/CHARTER.md`` against drifting away from the code it cites.

The charter is unusual in that nearly every claim it makes is falsifiable: each
of the Seven Core Invariants names the exact file, Cargo feature, or CI job that
enforces it, and it delegates the authoritative crate list to
``docs/architecture.md``. That precision is the document's main value — and the
reason it rots silently. A stale enforcement citation is worse than no citation,
because it tells a reader a guarantee is defended when it is not.

These tests mechanise the checks that a human audit would otherwise have to
repeat. Each one corresponds to a requirement in the ``charter-alignment``
capability spec:

* :func:`test_charter_path_citations_resolve` — every cited path exists.
* :func:`test_charter_cited_features_are_declared` — every cited Cargo feature
  is real.
* :func:`test_enforcement_cited_features_have_cfg_sites` — a feature the charter
  names as *enforcement* must actually be gated on in code. Declaration alone is
  not enough: ``live-test-stub`` was declared for two releases, was cited by
  Invariant 3, and had no ``#[cfg(feature = …)]`` site anywhere.
* :func:`test_declared_features_are_reachable` — no orphaned feature flags.
* :func:`test_architecture_doc_covers_every_workspace_crate` — the charter's
  designated source of truth for the crate list is actually complete.
* :func:`test_docs_name_no_deleted_crates` — no document names a crate that has
  been removed. Detection handles names split across lines by ASCII box art,
  which is how ``forge-procgen`` survived the ``e5eca3d`` cleanup sweep in two
  diagrams despite a repo-wide ``forge-procgen`` grep.
* :func:`test_charter_ci_job_citations_exist` — jobs named by Invariant 6 exist.

Stdlib and ``pytest`` only, by design: the ``python-test`` CI job installs
``maturin pytest numpy gymnasium httpx jsonschema`` and nothing else, so
``ci.yml`` is parsed with a regex rather than by adding a YAML dependency.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

#: Repository root, resolved from this file so the tests ignore the caller's cwd.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

CHARTER: Path = REPO_ROOT / "docs" / "CHARTER.md"
ARCHITECTURE: Path = REPO_ROOT / "docs" / "architecture.md"
WORKSPACE_MANIFEST: Path = REPO_ROOT / "Cargo.toml"
WORKFLOWS_DIR: Path = REPO_ROOT / ".github" / "workflows"

#: Directory prefixes that mark a backticked token as a repository path rather
#: than a type name, a shell fragment, or prose. Anchoring on these avoids
#: false positives on things like ``Arc<OnnxMuZeroModel>`` or ``serde(default)``.
PATH_PREFIXES: tuple[str, ...] = (
    ".github/",
    "benchmarks/",
    "configs/",
    "crates/",
    "dashboard/",
    "docker/",
    "docs/",
    "examples/",
    "mc-bot/",
    "python/",
    "scripts/",
    "tests/",
)

#: Root-level files the charter cites by bare name.
ROOT_FILES: frozenset[str] = frozenset(
    {
        ".coveragerc",
        ".gitignore",
        "CHANGELOG.md",
        "CLAUDE.md",
        "Cargo.toml",
        "LICENSE",
        "README.md",
        "deny.toml",
        "pyproject.toml",
    }
)

#: Single-word CI job names cited by Invariant 6. These are pinned explicitly
#: because a bare word like ``test`` cannot be told apart from prose by shape
#: alone; hyphenated job names are discovered automatically instead.
SINGLE_WORD_CI_JOBS: tuple[str, ...] = ("fmt", "clippy", "test", "coverage")

#: ``forge-*`` identifiers that are deliberately not crate names. CI job names
#: are excluded automatically by reading the workflows, so only genuinely
#: unrelated identifiers belong here.
NON_CRATE_FORGE_IDENTIFIERS: frozenset[str] = frozenset(
    {
        # Docker Compose bridge network in the deployment diagram.
        "forge-net",
        # `forge-integration` ships as package `forge-integration-layer`.
        "forge-integration-layer",
        # Root package for workspace-wide integration tests.
        "forge-integration-tests",
    }
)

#: Cargo features that legitimately have no ``#[cfg]`` site because their whole
#: purpose is to switch on an optional dependency or aggregate other features.
#: Keep this list short and justified — it is an escape hatch, not a dumping
#: ground.
CFG_LESS_FEATURES: frozenset[str] = frozenset(
    {
        "default",
        # Aggregate: turns on mc-live + onnx-reload + forge-agent/onnx-bundled.
        "mc-live-bundled",
        # Enables `ort/load-dynamic` on forge-agent; no code branches on it.
        "onnx-bundled",
        # Gates the optional `dhat` dependency for the allocation-audit binary.
        "dhat-heap",
    }
)

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_CRATE_NAME = re.compile(r"forge-[a-z][a-z0-9-]*")
_DANGLING_FORGE = re.compile(r"forge-(?![a-z0-9])")
_LEADING_IDENT = re.compile(r"[a-z][a-z0-9_-]*")
_CI_JOB_KEY = re.compile(r"^  ([a-z][a-z0-9-]*):\s*$")
_FEATURE_ENTRY = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*=\s*\[")
#: A feature name selected by a `--features` CLI flag inside a backticked
#: command span, e.g. `` `--features onnx-reload` `` or
#: `` `cargo build -p forge-mc-runner --features mc-live-bundled` ``. Catches
#: citations a plain "is this whole span a bare feature name" check misses.
_FEATURES_FLAG_ARG = re.compile(r"--features\s+([A-Za-z][A-Za-z0-9_-]*)")

#: Box-drawing and padding characters to strip when reading a column out of an
#: ASCII diagram. Built from code points so the source stays free of characters
#: that lint flags as visually ambiguous with ASCII.
_BOX_CHARS = "|/\\ \t" + "".join(
    chr(c)
    for c in (
        0x2502,
        0x2503,
        0x2571,
        0x2572,
        0x25BC,
        0x25B2,
        0x2514,
        0x2518,
        0x251C,
        0x2524,
        0x250C,
        0x2510,
        0x2500,
    )
)

#: Lookahead width (chars) into the line below a dangling `forge-` when
#: reconstructing a name ASCII art split across two lines. Comfortably covers
#: the longest crate name (`forge-mc-runner`, 16 chars) plus box-art padding.
_COLUMN_LOOKAHEAD: int = 40


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_hint(text: str, needle: str) -> str:
    """Best-effort ``:<line>`` suffix for the first occurrence of ``needle``.

    Imprecise by design: it's a "look here first" pointer, not a guarantee —
    it can land on a plain-prose mention instead of a citation, or resolve to
    nothing (empty string) if the token only exists reconstructed from a
    diagram split across two lines. Never asserted on.
    """
    idx = text.find(needle)
    return f":{text.count(chr(10), 0, idx) + 1}" if idx != -1 else ""


def _crate_manifest(crate: str) -> Path:
    """Path to a workspace crate's manifest."""
    return REPO_ROOT / "crates" / crate / "Cargo.toml"


def workspace_members() -> list[str]:
    """Return crate names from ``[workspace] members`` in the root manifest."""
    text = _read(WORKSPACE_MANIFEST)
    block = re.search(r"\[workspace\][^\[]*?members\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert block is not None, "could not locate [workspace] members in Cargo.toml"
    return [Path(m).name for m in re.findall(r'"([^"]+)"', block.group(1))]


def declared_features() -> dict[str, str]:
    """Map every feature declared in a workspace crate to its owning crate."""
    features: dict[str, str] = {}
    for crate in workspace_members():
        manifest = _crate_manifest(crate)
        if not manifest.is_file():
            continue
        in_features = False
        for line in _read(manifest).splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_features = stripped == "[features]"
                continue
            if in_features:
                match = _FEATURE_ENTRY.match(stripped)
                if match:
                    features[match.group(1)] = crate
    return features


def cfg_gated_features() -> set[str]:
    """Return features that appear in a ``#[cfg(feature = "…")]`` site."""
    found: set[str] = set()
    for rust_file in (REPO_ROOT / "crates").rglob("*.rs"):
        if "target" in rust_file.parts:
            continue
        found.update(re.findall(r'feature\s*=\s*"([^"]+)"', _read(rust_file)))
    return found


def features_enabled_by_others() -> set[str]:
    """Return features named inside another feature's dependency list."""
    enabled: set[str] = set()
    for crate in workspace_members():
        manifest = _crate_manifest(crate)
        if not manifest.is_file():
            continue
        text = _read(manifest)
        block = re.search(r"\[features\](.*?)(?:\n\[|\Z)", text, re.DOTALL)
        if block is None:
            continue
        for entry in re.findall(r'"([^"]+)"', block.group(1)):
            # `forge-agent/onnx` → record the bare feature name too.
            enabled.add(entry.split("/")[-1].removeprefix("dep:"))
    return enabled


def build_command_text() -> str:
    """Concatenate the docs and CI files that carry ``--features`` invocations."""
    parts: list[str] = []
    for pattern in ("*.md", "docs/*.md", ".github/workflows/*.yml", "docker/*"):
        parts.extend(_read(p) for p in REPO_ROOT.glob(pattern) if p.is_file())
    return "\n".join(parts)


def charter_paths() -> set[str]:
    """Return every backticked token in the charter that names a repo path."""
    cited: set[str] = set()
    for token in _BACKTICKED.findall(_read(CHARTER)):
        # A backticked span may be a whole command (`script.py --flag 0`); the
        # path is its first word. Trailing `::symbol` is a Rust item, not a path.
        candidate = token.strip().split()[0].split("::")[0] if token.strip() else ""
        if candidate in ROOT_FILES or candidate.startswith(PATH_PREFIXES):
            cited.add(candidate)
    return cited


def enforcement_paragraphs() -> list[str]:
    """Return the charter's ``*Enforced by:*`` paragraphs."""
    text = _read(CHARTER)
    return re.findall(r"\*Enforced by:\*(.*?)(?:\n\n)", text, re.DOTALL)


def _jobs_in(workflow: Path) -> set[str]:
    """Return top-level job keys from a workflow (regex, so PyYAML is not needed)."""
    text = _read(workflow)
    jobs_start = re.search(r"^jobs:\s*$", text, re.MULTILINE)
    if jobs_start is None:
        return set()
    body = text[jobs_start.end() :]
    return {m.group(1) for line in body.splitlines() if (m := _CI_JOB_KEY.match(line))}


def ci_job_names() -> set[str]:
    """Return job keys across every workflow.

    The charter's Invariant 6 names gates in ``ci.yml`` and, separately, the
    advisory supply-chain jobs in ``security.yml``, so both must be in scope.
    """
    jobs: set[str] = set()
    for workflow in sorted(WORKFLOWS_DIR.glob("*.yml")):
        jobs |= _jobs_in(workflow)
    return jobs


def reconstruct_split_crate_names(text: str) -> set[str]:
    """Recover crate names broken across lines by ASCII box art.

    A diagram may render ``forge-procgen`` as ``forge-`` on one line and
    ``procgen`` on the next, at the same column. A plain grep for the full name
    then finds nothing, which is exactly how two live references survived the
    crate deletion in ``e5eca3d``.
    """
    names: set[str] = set()
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        for match in _DANGLING_FORGE.finditer(line):
            column = match.start()
            below = lines[index + 1]
            if column >= len(below):
                continue
            fragment = below[column : column + _COLUMN_LOOKAHEAD].lstrip(_BOX_CHARS)
            ident = _LEADING_IDENT.match(fragment)
            if ident is not None:
                names.add(f"forge-{ident.group(0)}")
    return names


# --------------------------------------------------------------------------
# Requirement: Charter Path Citations Resolve
# --------------------------------------------------------------------------
def test_charter_path_citations_resolve() -> None:
    """Every repository path cited in the charter must exist."""
    text = _read(CHARTER)
    cited = charter_paths()
    assert cited, "no paths extracted from the charter — the parser is broken"
    # Invariant 7 deliberately cites git-ignored secret files (e.g.
    # `docker/compose.minecraft.env`). Those are absent by design, and the thing
    # that must exist is the committed template beside them.
    missing = sorted(
        p
        for p in cited
        if not (REPO_ROOT / p).exists() and not (REPO_ROOT / f"{p}.example").exists()
    )
    assert not missing, (
        f"docs/CHARTER.md cites {len(missing)} path(s) that no longer exist: "
        f"{[p + _line_hint(text, p) for p in missing]}. Update the charter or "
        "restore the path — see the charter's Development Guidance."
    )


# --------------------------------------------------------------------------
# Requirement: Charter Feature Citations Are Implemented
# --------------------------------------------------------------------------
def test_charter_cited_features_are_declared() -> None:
    """Cargo features named in the charter must be declared by a workspace crate.

    Two citation shapes are checked: a bare feature name in its own backticks
    (`` `onnx-reload` ``) and a feature name selected inside a backticked
    command span (`` `--features onnx-reload` ``). Checking only whether the
    *whole* backticked span equals a declared name would miss the second
    shape entirely — command spans never equal a bare feature name — which is
    exactly how this test previously went blind on `docs/CHARTER.md`'s one
    `--features onnx-reload` citation.
    """
    declared = declared_features()
    charter_text = _read(CHARTER)
    cited: set[str] = set()
    for span in _BACKTICKED.findall(charter_text):
        stripped = span.strip()
        if stripped in declared:
            cited.add(stripped)
        cited.update(_FEATURES_FLAG_ARG.findall(span))
    undeclared = sorted(t for t in cited if t not in declared)
    assert not undeclared, (
        f"docs/CHARTER.md cites Cargo feature(s) that no crate declares: "
        f"{[t + _line_hint(charter_text, t) for t in undeclared]}"
    )


def test_enforcement_cited_features_have_cfg_sites() -> None:
    """A feature cited as *enforcement* must be gated on somewhere in the code.

    Declaration alone is not enforcement. This is the check that catches the
    ``live-test-stub`` class of defect: a feature declared in the manifest,
    cited by an invariant, and never read by a single line of Rust.
    """
    declared = declared_features()
    gated = cfg_gated_features()
    charter_text = _read(CHARTER)
    offenders: list[str] = []
    for paragraph in enforcement_paragraphs():
        offenders.extend(
            token
            for token in _BACKTICKED.findall(paragraph)
            if token in declared and token not in gated and token not in CFG_LESS_FEATURES
        )
    offenders = sorted(set(offenders))
    assert not offenders, (
        f"docs/CHARTER.md names feature(s) as enforcement that have no "
        f"`#[cfg(feature = ...)]` site: "
        f"{[t + _line_hint(charter_text, t) for t in offenders]}. Either implement "
        "the gate, or stop citing the feature as an enforcement mechanism."
    )


def test_declared_features_are_reachable() -> None:
    """No workspace crate declares a feature that nothing can ever select."""
    declared = declared_features()
    gated = cfg_gated_features()
    enabled = features_enabled_by_others()
    commands = build_command_text()
    orphans = sorted(
        f"{crate}:{name}"
        for name, crate in declared.items()
        if name not in gated
        and name not in enabled
        and name not in CFG_LESS_FEATURES
        and f"--features {name}" not in commands
    )
    assert not orphans, (
        f"Cargo feature(s) are declared but unreachable — no cfg site, not enabled "
        f"by another feature, and no build command selects them: {orphans}. Remove "
        "them, or wire them up."
    )


# --------------------------------------------------------------------------
# Requirement: Architecture Doc Covers Every Workspace Crate
# --------------------------------------------------------------------------
def test_architecture_doc_covers_every_workspace_crate() -> None:
    """The charter delegates the crate list here, so the delegation must be sound."""
    text = _read(ARCHITECTURE)
    missing = sorted(c for c in workspace_members() if c not in text)
    assert not missing, (
        f"docs/architecture.md omits {len(missing)} workspace crate(s): {missing}. "
        "docs/CHARTER.md names this file as the single source of truth for the "
        "crate list, so every workspace member must appear here."
    )


# --------------------------------------------------------------------------
# Requirement: Docs Name No Deleted Crates
# --------------------------------------------------------------------------
@pytest.mark.parametrize("doc", ["docs/CHARTER.md", "docs/architecture.md"])
def test_docs_name_no_deleted_crates(doc: str) -> None:
    """No governance doc may name a crate that is not a workspace member."""
    members = set(workspace_members())
    text = _read(REPO_ROOT / doc)
    named = set(_CRATE_NAME.findall(text)) | reconstruct_split_crate_names(text)
    # Not every `forge-*` token is a crate: CI jobs (`forge-mc-runner-bin`) are
    # discovered from the workflows, and a short allowlist covers the rest.
    known = members | ci_job_names() | NON_CRATE_FORGE_IDENTIFIERS
    stale = sorted(n for n in named if n not in known)
    assert not stale, (
        f"{doc} names crate(s) that are not workspace members: "
        f"{[n + _line_hint(text, n) for n in stale]}. If the crate was deleted, "
        "remove the reference — including any ASCII diagram that splits the "
        "name across two lines (no line hint is available for those)."
    )


# --------------------------------------------------------------------------
# Requirement: Charter CI Job Citations Exist
# --------------------------------------------------------------------------
def test_charter_ci_job_citations_exist() -> None:
    """Every CI job the Determinism invariant names must still exist in a workflow."""
    jobs = ci_job_names()
    assert jobs, "no jobs parsed from the workflows — the parser is broken"

    text = _read(CHARTER)
    # Matches on the "Determinism" heading text with any `### <digit>. ` prefix
    # rather than hardcoding "6", so a future charter renumbering doesn't break
    # this test for a reason unrelated to what it actually checks.
    determinism = re.search(r"### \d+\. Determinism.*?(?=\n### |\Z)", text, re.DOTALL)
    assert determinism is not None, "the Determinism invariant was not found in docs/CHARTER.md"

    cited = {
        token
        for token in _BACKTICKED.findall(determinism.group(0))
        if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", token)
    }
    cited.update(SINGLE_WORD_CI_JOBS)

    missing = sorted(job for job in cited if job not in jobs)
    assert not missing, (
        f"docs/CHARTER.md's Determinism invariant names CI job(s) absent from any "
        f"workflow: {[j + _line_hint(text, j) for j in missing]}. Update the "
        "charter if a job was renamed, or restore the gate."
    )


# ==========================================================================
# Unit tests for this module's own parsers, in isolation from the live repo.
#
# The tests above exercise these helpers only incidentally, through whatever
# docs/CHARTER.md, docs/architecture.md, and the workflows currently contain.
# That leaves defensive branches (a workspace member with no Cargo.toml, a
# workflow with no `jobs:` key) permanently unexercised, since every crate and
# workflow in this repo happens to be well-formed today, and leaves
# `reconstruct_split_crate_names` — the function that specifically exists to
# catch the `forge-procgen` class of drift — with no test pinning its
# contract independent of what architecture.md happens to contain right now.
# These tests close that gap with synthetic, hand-traced inputs.
# ==========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("forge-\nprocgen\n", {"forge-procgen"}),
        ("    forge-\n    │  procgen\n", {"forge-procgen"}),  # box-art stripped
        ("x forge-\n\n", set()),  # column >= len(below) — bound skipped, not crashed
        ("forge-mc-runner\nsomething\n", set()),  # not dangling — negative lookahead excludes it
        ("procgen\nforge-", set()),  # dangling forge- on the *last* line is never checked
    ],
)
def test_reconstruct_split_crate_names(text: str, expected: set[str]) -> None:
    assert reconstruct_split_crate_names(text) == expected


def test_jobs_in_returns_empty_set_when_no_jobs_key(tmp_path: Path) -> None:
    workflow = tmp_path / "no_jobs.yml"
    workflow.write_text("name: CI\non: push\n", encoding="utf-8")
    assert _jobs_in(workflow) == set()


def test_workspace_members_asserts_on_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_manifest = tmp_path / "Cargo.toml"
    bad_manifest.write_text('[package]\nname = "x"\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "WORKSPACE_MANIFEST", bad_manifest)
    with pytest.raises(AssertionError, match="could not locate"):
        workspace_members()


@pytest.fixture
def synthetic_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point REPO_ROOT/WORKSPACE_MANIFEST at an isolated, minimal repo tree."""
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(
        '[workspace]\nmembers = ["crates/fake-crate", "crates/missing-crate"]\n',
        encoding="utf-8",
    )
    crate_dir = tmp_path / "crates" / "fake-crate"
    crate_dir.mkdir(parents=True)
    (crate_dir / "Cargo.toml").write_text(
        '[package]\nname = "fake-crate"\n\n'
        "[features]\n"
        "default = []\n"
        'alpha = ["dep:something"]\n'
        'beta = ["fake-crate/alpha", "other-crate/gamma"]\n',
        encoding="utf-8",
    )
    # crates/missing-crate is a workspace member with no Cargo.toml on disk —
    # exercises the `if not manifest.is_file(): continue` guard.
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "WORKSPACE_MANIFEST", manifest)
    return tmp_path


def test_declared_features_skips_crates_without_manifest(synthetic_repo: Path) -> None:
    assert declared_features() == {
        "default": "fake-crate",
        "alpha": "fake-crate",
        "beta": "fake-crate",
    }


def test_features_enabled_by_others_strips_dep_prefix_and_crate_qualifier(
    synthetic_repo: Path,
) -> None:
    assert features_enabled_by_others() == {"something", "alpha", "gamma"}


def test_build_command_text_includes_readme_features(synthetic_repo: Path) -> None:
    (synthetic_repo / "README.md").write_text(
        "Run `cargo build --features onnx-reload`.\n", encoding="utf-8"
    )
    assert "--features onnx-reload" in build_command_text()

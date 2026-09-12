"""Version resolution for the :mod:`forge_env` package.

FORGE's configuration-driven convention (``CLAUDE.md`` — "No hard-coded
values": *all constants flow through config structs*) applies to the package
version as much as to simulation tunables. The workspace ``Cargo.toml``
``[workspace.package].version`` is the single source of truth; every other
surface derives from it rather than restating it.

A restated literal is not a hypothetical risk here. ``forge_env.__version__``
was pinned to a string in ``__init__.py`` and the only test asserting anything
about it checked ``hasattr(forge_env, "__version__")``, so it could disagree
with the workspace version indefinitely without a single gate going red.

Resolution order, most authoritative first:

1. **The workspace Cargo manifest, resolved relative to this file.** It answers
   only when ``forge_env`` is being imported *from a workspace checkout*, and in
   that case the checkout is what is running, so its manifest is the truth. This
   deliberately outranks metadata: after a version bump, a previously installed
   wheel still reports the old version, and preferring it would make
   ``forge_env.__version__`` disagree with the very tree under test.
2. **Installed distribution metadata.** The path for a real wheel, where no
   co-located manifest exists. ``maturin`` stamps that metadata from this same
   Cargo manifest at build time, so it *is* the Cargo version, already resolved.
3. **A sentinel.** Importing ``forge_env`` must never fail because a version
   could not be resolved.

Every step takes its collaborator as a parameter so the resolution ladder is
exercisable without mutating global interpreter state (Charter Invariant 3,
"dependency injection enabling testability without hardware").
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "CARGO_MANIFEST_PATH",
    "DISTRIBUTION_NAME",
    "SENTINEL_VERSION",
    "resolve_version",
    "version_from_cargo_manifest",
    "version_from_metadata",
]

#: Distribution name as declared by ``pyproject.toml``'s ``[project].name``.
#: Kept as a module constant rather than inlined so a rename is a one-line
#: change with a test that can import the same symbol.
DISTRIBUTION_NAME: str = "forge-env"

#: Returned when neither metadata nor the manifest yields a version. PEP 440
#: local-version syntax, so packaging tools still parse it rather than raising.
SENTINEL_VERSION: str = "0.0.0+unknown"

#: Workspace manifest, relative to this file: ``<repo>/python/forge_env/_version.py``
#: → ``<repo>/Cargo.toml``. Absent in an installed wheel, which is expected —
#: step 1 answers there.
CARGO_MANIFEST_PATH: Path = Path(__file__).resolve().parents[2] / "Cargo.toml"

#: Body of the ``[workspace.package]`` table, up to the next table header or EOF.
#: Scoped deliberately: a bare ``version = "..."`` search would match the first
#: ``version`` key in the file, which belongs to a different table.
_WORKSPACE_PACKAGE_TABLE = re.compile(
    r"^\[workspace\.package\][^\n]*\n(?P<body>.*?)(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)

#: ``version = "1.2.3"`` within an already-scoped table body.
_VERSION_ASSIGNMENT = re.compile(
    r"^[ \t]*version[ \t]*=[ \t]*[\"'](?P<version>[^\"']+)[\"']",
    re.MULTILINE,
)


def version_from_metadata(
    distribution: str = DISTRIBUTION_NAME,
    *,
    reader: Callable[[str], str] | None = None,
) -> str | None:
    """Return the installed distribution's version, or ``None`` if absent.

    Args:
        distribution: Distribution name to look up.
        reader: Metadata lookup, injected for tests. Defaults to
            :func:`importlib.metadata.version`.

    Returns:
        The version string, or ``None`` when the distribution is not installed.
    """
    lookup = reader if reader is not None else _distribution_version
    try:
        return lookup(distribution)
    except PackageNotFoundError:
        logger.debug("Distribution %r is not installed.", distribution)
        return None


def version_from_cargo_manifest(manifest: Path | None = None) -> str | None:
    """Return ``[workspace.package].version`` from the workspace manifest.

    Parsed with a scoped regular expression rather than ``tomllib`` because the
    supported floor is Python 3.9 (``pyproject.toml``: ``requires-python =
    ">=3.9"``) and ``tomllib`` only lands in 3.11. Adding ``tomli`` as a runtime
    dependency to read one string would be a heavier cost than this pattern.

    Args:
        manifest: Manifest path. Defaults to :data:`CARGO_MANIFEST_PATH`.

    Returns:
        The version string, or ``None`` when the manifest is missing,
        unreadable, or has no ``[workspace.package].version``.
    """
    path = manifest if manifest is not None else CARGO_MANIFEST_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("Cargo manifest at %s is unreadable: %s", path, exc)
        return None

    table = _WORKSPACE_PACKAGE_TABLE.search(text)
    if table is None:
        logger.debug("Cargo manifest at %s declares no [workspace.package] table.", path)
        return None

    assignment = _VERSION_ASSIGNMENT.search(table.group("body"))
    if assignment is None:
        logger.debug("[workspace.package] in %s declares no version key.", path)
        return None

    return assignment.group("version")


def resolve_version(
    *,
    distribution: str = DISTRIBUTION_NAME,
    manifest: Path | None = None,
    metadata_reader: Callable[[str], str] | None = None,
) -> str:
    """Resolve the package version, never raising.

    Args:
        distribution: Distribution name for the metadata lookup.
        manifest: Cargo manifest path for the fallback lookup.
        metadata_reader: Metadata lookup, injected for tests.

    Returns:
        The resolved version, or :data:`SENTINEL_VERSION` if every source fails.
    """
    from_manifest = version_from_cargo_manifest(manifest)
    if from_manifest:
        return from_manifest

    from_metadata = version_from_metadata(distribution, reader=metadata_reader)
    if from_metadata:
        return from_metadata

    logger.warning(
        "Could not resolve the forge_env version from distribution metadata (%r) "
        "or the Cargo manifest; reporting %s. Install the package "
        "(`maturin develop`) or run from a workspace checkout to get a real version.",
        distribution,
        SENTINEL_VERSION,
    )
    return SENTINEL_VERSION

"""Guard that every surface reports the same version.

FORGE publishes its version from four places — the Cargo workspace, the Python
package, the dashboard's ``package.json``, and its lockfile — and before this
gate existed nothing compared them. ``forge_env.__version__`` was a literal
``"0.5.0"`` in ``__init__.py`` whose only test asserted ``hasattr``, and the
root ``forge-integration-tests`` package carried its own literal instead of
inheriting the workspace version. Either could disagree with Cargo indefinitely.

The Cargo workspace is the single source of truth. These tests assert the other
surfaces agree with it, and separately cover the resolution ladder in
:mod:`forge_env._version` — including the fallbacks, which are the paths a
plain ``hasattr`` check can never reach.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from conftest import REPO_ROOT, read_toml
from forge_env._version import (
    SENTINEL_VERSION,
    resolve_version,
    version_from_cargo_manifest,
    version_from_metadata,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Semantic-version shape the release process relies on (``MAJOR.MINOR.PATCH``).
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

CARGO_MANIFEST: Path = REPO_ROOT / "Cargo.toml"
DASHBOARD_PACKAGE_JSON: Path = REPO_ROOT / "dashboard" / "package.json"
DASHBOARD_PACKAGE_LOCK: Path = REPO_ROOT / "dashboard" / "package-lock.json"


def workspace_version() -> str:
    """Return ``[workspace.package].version`` from the Cargo manifest.

    Parsed independently of :mod:`forge_env._version` so this test cannot pass
    by agreeing with the code it is checking.
    """
    manifest: dict[str, Any] = read_toml(CARGO_MANIFEST)
    version = manifest["workspace"]["package"]["version"]
    assert isinstance(version, str)
    return version


class TestWorkspaceVersionIsTheSourceOfTruth:
    """Every other surface must agree with the Cargo workspace version."""

    def test_workspace_version_is_semver(self) -> None:
        version = workspace_version()
        assert SEMVER_PATTERN.match(version), (
            f"[workspace.package].version is {version!r}, which is not MAJOR.MINOR.PATCH. "
            "Release tooling and the dashboard package version both assume that shape."
        )

    def test_root_package_inherits_the_workspace_version(self) -> None:
        """The root package must inherit rather than restate the version."""
        manifest: dict[str, Any] = read_toml(CARGO_MANIFEST)
        package = manifest["package"]
        assert package.get("version") == {"workspace": True}, (
            "The root `forge-integration-tests` package must declare "
            "`version.workspace = true`. A literal here is left behind by a "
            f"release bump; found {package.get('version')!r}."
        )

    def test_python_package_version_matches_cargo(self) -> None:
        """``forge_env.__version__`` must resolve to the workspace version.

        In a source checkout this comes from the Cargo manifest; in an installed
        wheel it comes from distribution metadata that maturin stamped from the
        same manifest. Either way it must agree.
        """
        import forge_env

        assert forge_env.__version__ == workspace_version(), (
            f"forge_env.__version__ is {forge_env.__version__!r} but the Cargo "
            f"workspace is {workspace_version()!r}. Remedy: the version is "
            "derived, not stored -- if these disagree, either a stale wheel is "
            "installed (`maturin develop` to refresh) or the manifest lookup in "
            "forge_env/_version.py is resolving the wrong file."
        )

    @pytest.mark.parametrize(
        "path", [DASHBOARD_PACKAGE_JSON, DASHBOARD_PACKAGE_LOCK], ids=["package", "lock"]
    )
    def test_dashboard_version_matches_cargo(self, path: Path) -> None:
        """The dashboard package and its lockfile track the workspace version."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == workspace_version(), (
            f"{path.relative_to(REPO_ROOT)} declares version {payload['version']!r} "
            f"but the Cargo workspace is {workspace_version()!r}. Remedy: bump it "
            "and run `npm install --package-lock-only` in dashboard/ so the "
            "lockfile's root entry follows."
        )

    def test_dashboard_lockfile_root_entry_matches(self) -> None:
        """The lockfile embeds the version twice; both must track Cargo."""
        payload = json.loads(DASHBOARD_PACKAGE_LOCK.read_text(encoding="utf-8"))
        root_entry = payload["packages"][""]
        assert root_entry["version"] == workspace_version(), (
            "dashboard/package-lock.json's root package entry disagrees with the "
            f"Cargo workspace version {workspace_version()!r}. Remedy: run "
            "`npm install --package-lock-only` in dashboard/."
        )


class TestVersionResolutionLadder:
    """Cover every branch of the resolution ladder, including the fallbacks."""

    def test_reads_the_workspace_table_not_the_first_version_key(self) -> None:
        """Resolution must be scoped to ``[workspace.package]``.

        An unscoped ``version =`` search would match whichever table came first,
        which is exactly the bug a naive implementation ships with.
        """
        assert version_from_cargo_manifest(CARGO_MANIFEST) == workspace_version()

    def test_missing_manifest_returns_none(self, tmp_path: Path) -> None:
        assert version_from_cargo_manifest(tmp_path / "absent.toml") is None

    def test_manifest_without_workspace_package_returns_none(self, tmp_path: Path) -> None:
        manifest = tmp_path / "Cargo.toml"
        manifest.write_text('[package]\nname = "x"\nversion = "9.9.9"\n', encoding="utf-8")
        assert version_from_cargo_manifest(manifest) is None, (
            "a [package] version must not be mistaken for the workspace version"
        )

    def test_workspace_package_without_version_returns_none(self, tmp_path: Path) -> None:
        manifest = tmp_path / "Cargo.toml"
        manifest.write_text('[workspace.package]\nedition = "2021"\n', encoding="utf-8")
        assert version_from_cargo_manifest(manifest) is None

    def test_a_co_located_manifest_outranks_stale_metadata(self, tmp_path: Path) -> None:
        """The source checkout wins over a previously installed wheel.

        The manifest only resolves when ``forge_env`` is imported from a
        workspace checkout, and then that checkout is what is running. Ordering
        it after metadata made ``forge_env.__version__`` report the version of
        whatever wheel happened to be installed, so a version bump disagreed
        with its own tree until someone remembered to rebuild.
        """
        manifest = tmp_path / "Cargo.toml"
        manifest.write_text('[workspace.package]\nversion = "1.2.3"\n', encoding="utf-8")

        resolved = resolve_version(manifest=manifest, metadata_reader=lambda _name: "7.8.9")
        assert resolved == "1.2.3"

    def test_falls_back_to_metadata_outside_a_checkout(self, tmp_path: Path) -> None:
        """An installed wheel has no co-located manifest; metadata answers."""
        resolved = resolve_version(
            manifest=tmp_path / "absent.toml", metadata_reader=lambda _name: "7.8.9"
        )
        assert resolved == "7.8.9"

    def test_falls_back_to_the_sentinel_when_every_source_fails(self, tmp_path: Path) -> None:
        from importlib.metadata import PackageNotFoundError

        def _absent(name: str) -> str:
            raise PackageNotFoundError(name)

        resolved = resolve_version(
            manifest=tmp_path / "absent.toml", metadata_reader=_absent
        )
        assert resolved == SENTINEL_VERSION

    def test_metadata_reader_returns_none_for_absent_distribution(self) -> None:
        from importlib.metadata import PackageNotFoundError

        def _absent(name: str) -> str:
            raise PackageNotFoundError(name)

        assert version_from_metadata("definitely-not-installed", reader=_absent) is None

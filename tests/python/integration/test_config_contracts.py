"""Regression tests for the configuration defects this branch fixed.

Every case here pins a bug that actually shipped, and each belongs to one
class: **a declared knob whose behaviour did not match its declaration.**

- `FORGE_MC_REQUIRE_E2E` was added to turn silent skips into failures,
  and nothing anywhere set it — the fix was inert for as long as it
  existed.
- `FORGE_MC_RUNNER_EPISODES` was given a compose default of `10` on the
  same day it gained its first reader, which silently overrode
  `runner.toml`'s `episodes` for every operator.
- A `pull_policy` knob defaulting to `build` was added on the belief that
  Compose "never pulls" for a service with a `build:` section. The spec
  says the opposite, and `build` means *always rebuild* — so the default
  forced a cold rebuild on every bring-up.
- `FORGE_MC_BOT_URL` and `FORGE_MC_RUNNER_CONFIG` sat in compose for
  years with zero readers in `crates/`.

None of these could fail a test, because nothing tested the *contract*
between the config files and the code that consumes them. These do.

Pure parsing — no Docker, no stack. They run on every PR, deliberately
not behind the `minecraft_e2e` marker, so a drift surfaces in the PR that
introduces it rather than at `docker compose up` time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from ._helpers import COMPOSE_UP_BUILD_TIMEOUT_SECS, REQUIRE_E2E_ENV_VAR

#: Workflow job that exists to run the opt-in Minecraft E2E suite.
E2E_JOB_ID = "python-test-minecraft-e2e"

#: Compose service running `forge-mc-runner`, whose config precedence is
#: CLI > env > TOML.
RUNNER_SERVICE = "runner"

#: Prefix identifying env vars this project owns (as opposed to
#: third-party ones like `RUST_LOG` or the `MC_*` server knobs).
FORGE_ENV_PREFIX = "FORGE_"

#: Trees searched for a reader of a compose-declared env var. `docker/`
#: is excluded on purpose: a var mentioned only there is exactly the
#: dead-knob case being detected.
READER_ROOTS = ("crates", "python", "scripts", "mc-bot/src", "tests/python")

#: Source suffixes worth grepping for a reader.
READER_SUFFIXES = (".rs", ".py", ".ts", ".js", ".sh", ".toml")

#: `pull_policy` values that force a build rather than preferring a
#: published image. Compose's documented default (the key absent) pulls
#: first and falls back to building, which is what this stack wants.
REBUILD_FORCING_PULL_POLICIES = frozenset({"build"})


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """The FORGE repo root, anchored to this test file."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def compose_data(repo_root: Path) -> dict[str, Any]:
    """Parsed `docker/compose.minecraft.yml`."""
    yaml = pytest.importorskip("yaml")
    text = (repo_root / "docker" / "compose.minecraft.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def ci_workflow(repo_root: Path) -> dict[str, Any]:
    """Parsed `.github/workflows/ci.yml`."""
    yaml = pytest.importorskip("yaml")
    text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data


def _service_environment(service: dict[str, Any]) -> dict[str, str]:
    """Normalise a compose `environment:` block to a mapping.

    Compose accepts both the mapping and the `KEY=value` list form; a
    service switching between them must not silently drop these checks.
    """
    raw = service.get("environment") or {}
    if isinstance(raw, dict):
        return {str(k): "" if v is None else str(v) for k, v in raw.items()}
    pairs: dict[str, str] = {}
    for entry in raw:
        key, _, value = str(entry).partition("=")
        pairs[key] = value
    return pairs


def _interpolation_default(value: str) -> str | None:
    """The `X` in `${VAR:-X}`, or None if the value is not that shape."""
    match = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-(?P<default>.*)\}", value)
    return match.group("default") if match else None


# --- The E2E job must actually demand execution ----------------------


def test_e2e_job_requires_the_suite_to_execute(ci_workflow: dict[str, Any]) -> None:
    """The job that exists to RUN the suite must set the require flag.

    `require_e2e_execution()` converts environment-related skips into
    hard failures. It shipped with no setter anywhere in the repo, which
    made the whole green-hole fix inert: the job kept reporting success
    while a bad image tag or an unaccepted EULA skipped every test.

    Asserted against the workflow rather than a doc comment because the
    workflow is the only thing that can make it true.
    """
    job = ci_workflow["jobs"][E2E_JOB_ID]
    env = job.get("env") or {}
    assert REQUIRE_E2E_ENV_VAR in env, (
        f"{E2E_JOB_ID} must set {REQUIRE_E2E_ENV_VAR}; without it the job "
        "reports success when it skips every test"
    )
    assert str(env[REQUIRE_E2E_ENV_VAR]).strip() not in {"", "0", "false"}, (
        f"{REQUIRE_E2E_ENV_VAR}={env[REQUIRE_E2E_ENV_VAR]!r} is falsey, "
        "which disables the guard it exists to enable"
    )


def test_e2e_job_timeout_covers_the_bring_up_budget(
    ci_workflow: dict[str, Any],
) -> None:
    """The job cap must exceed the fixture's own bring-up budget.

    If the job dies first, the suite never gets to report *why* it
    failed — the operator sees a runner kill, not the diagnosable
    timeout message the fixture raises. Keeping the two in step is
    stated in `COMPOSE_UP_BUILD_TIMEOUT_SECS`'s docstring; this makes it
    checkable.
    """
    job = ci_workflow["jobs"][E2E_JOB_ID]
    timeout_minutes = job.get("timeout-minutes")
    assert timeout_minutes is not None, (
        f"{E2E_JOB_ID} must set timeout-minutes; the GitHub default is 6 "
        "hours, long enough for a hung bring-up to burn a runner"
    )
    budget_minutes = COMPOSE_UP_BUILD_TIMEOUT_SECS / 60
    assert int(timeout_minutes) > budget_minutes, (
        f"timeout-minutes={timeout_minutes} does not exceed the "
        f"{budget_minutes:.0f}-minute bring-up budget, so the job would be "
        "killed before the suite can report a diagnosable failure"
    )


# --- Compose must not silently outrank the TOML ----------------------


def test_runner_env_defaults_are_empty(compose_data: dict[str, Any]) -> None:
    """No `FORGE_*` var may carry a non-empty compose default.

    The runner resolves config as **CLI > env > TOML**, so a non-empty
    `${VAR:-value}` default in compose outranks `runner.toml` for every
    operator who never sets `VAR` — silently, and only for people running
    the container. `FORGE_MC_RUNNER_EPISODES` shipped as
    `${RUNNER_EPISODES:-10}` on the same day it gained its first reader,
    which would have capped the v0.5 baseline run at 10 episodes instead
    of the 100 `runner.toml` asks for.

    Empty is the correct default: it reads as unset, so the TOML wins and
    an operator opts in explicitly.
    """
    env = _service_environment(compose_data["services"][RUNNER_SERVICE])
    offenders = {
        key: value
        for key, value in env.items()
        if key.startswith(FORGE_ENV_PREFIX) and (_interpolation_default(value) or "") != ""
    }
    assert offenders == {}, (
        "compose defaults outrank runner.toml for every operator who does "
        f"not set the variable: {offenders}. Use `${{VAR:-}}` so the TOML "
        "value stays in charge."
    )


def test_no_build_service_forces_a_rebuild(compose_data: dict[str, Any]) -> None:
    """A `build:` service must not pin a rebuild-forcing `pull_policy`.

    Per the Compose spec, omitting `pull_policy` makes Compose "attempt
    to pull the image first and fall back to building from source if the
    image is not found" — the behaviour this stack wants. `build` means
    Compose *always* builds, so pinning it turns every bring-up into a
    cold `cargo build --release` inside Docker.

    That is not hypothetical: a `FORGE_MC_PULL_POLICY` knob defaulting to
    `build` was added here to "preserve local-dev behaviour" and did the
    exact opposite.
    """
    offenders = {
        name: service.get("pull_policy")
        for name, service in compose_data["services"].items()
        if "build" in service
        and str(service.get("pull_policy", "")).strip() in REBUILD_FORCING_PULL_POLICIES
    }
    assert offenders == {}, (
        f"these services force a rebuild on every bring-up: {offenders}. "
        "Omit pull_policy to get pull-then-build, which is the default."
    )


# --- Every declared knob must have a reader --------------------------


def test_every_forge_env_var_in_compose_has_a_reader(
    compose_data: dict[str, Any], repo_root: Path
) -> None:
    """A compose env var nobody reads is a lie in the config.

    `FORGE_MC_BOT_URL` and `FORGE_MC_RUNNER_CONFIG` sat in
    `compose.minecraft.yml` for years with zero readers, and
    `FORGE_MC_RUNNER_EPISODES` sat beside them documented as working
    while the runner ignored it — an operator setting it got the full
    100-episode baseline run and no warning.

    Searches the source trees for the variable's literal name. A reader
    that builds the name dynamically would be missed, but nothing here
    does that: the Rust side declares each as a `&'static str` constant
    whose value is the literal.
    """
    declared: set[str] = set()
    for service in compose_data["services"].values():
        declared.update(
            key for key in _service_environment(service) if key.startswith(FORGE_ENV_PREFIX)
        )
    assert declared, "no FORGE_* env vars found — this guard has drifted"

    sources = [
        path
        for root in READER_ROOTS
        for path in (repo_root / root).rglob("*")
        if path.is_file() and path.suffix in READER_SUFFIXES
    ]
    assert sources, "no source files found — this guard has drifted"
    haystack = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in sources)

    unread = sorted(name for name in declared if name not in haystack)
    assert unread == [], (
        f"declared in docker/compose.minecraft.yml but read nowhere in "
        f"{list(READER_ROOTS)}: {unread}. Either wire it up or delete it — "
        "a knob that does nothing is worse than no knob."
    )

#!/usr/bin/env python3
"""Assert every CI job has a documented local counterpart, or a stated reason.

The failure this prevents is quiet and expensive: a contributor runs
``make verify``, it passes, they open a PR, and CI goes red on a job that has
no local equivalent at all. Before this check, ten of the twenty-five jobs in
``ci.yml`` were in that state — ``markdownlint``, ``machete``, ``alloc-audit``,
``hf-export``, ``mlflow-http``, ``onnx-features`` and ``forge-mc-runner-bin``
among them, every one of which runs perfectly well on a laptop.

The mapping below is the artefact. Each CI job is either:

* **mapped** to the ``make`` target that runs the same thing locally, or
* **excepted** with a one-line reason it cannot or should not run locally.

Both directions are enforced, which is what keeps the mapping honest:

1. a CI job that is neither mapped nor excepted fails this check — so adding
   a job forces the author to either wire a local path or say why not;
2. a mapping naming a ``make`` target that no longer exists fails too — so
   renaming or deleting a target cannot silently orphan the claim;
3. an exception for a job that no longer exists fails — so the list cannot
   accumulate entries for jobs deleted years ago.

Stdlib-only and run from the ``python-lint`` job, following the pattern
``check_pinned_config_consistency.py`` and ``check_text_encoding.py``
established here. Exit 0 on success, 1 with an actionable report otherwise.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("check_local_ci_parity")

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"

#: Exit code when the mapping and the real files disagree.
EXIT_DRIFT = 1
#: Exit code when an input file is missing or unparseable.
EXIT_INPUT_ERROR = 2

#: CI job id -> the ``make`` target that runs the equivalent check locally.
#:
#: A job may map to a target that does more than the job does (``make lint``
#: covers ``clippy``); the contract is only that running the target locally
#: would have caught what the job catches.
JOB_TO_MAKE_TARGET: dict[str, str] = {
    "fmt": "fmt-check",
    "clippy": "lint",
    "test": "test",
    "coverage": "coverage",
    "python-lint": "py-lint",
    "python-test": "py-test",
    "python-test-fast": "py-test",
    "api-compliance": "api-compliance",
    "pip-install-clean": "pip-install-smoke",
    "mc-bot-test": "mc-bot-test",
    "dashboard": "dashboard-test",
    "dashboard-e2e": "dashboard-e2e",
    "demo-ui": "demo-ui-test",
    "wasm": "wasm-check",
    "wasm-e2e": "web-e2e",
    "onnx-features": "onnx-check",
    "hf-export": "hf-check",
    "mlflow-http": "mlflow-check",
    "alloc-audit": "alloc-audit",
    "forge-mc-runner-bin": "mc-runner-smoke",
    "machete": "machete",
    "markdownlint": "md-lint",
    "mutants": "mutants",
    "openspec-validate": "openspec-validate",
}

#: CI job id -> why no ``make`` target exists for it.
#:
#: An entry here is a decision, not a backlog item. "Nobody got round to it"
#: is not a reason; if a job could reasonably run locally, wire a target
#: instead of excepting it.
JOB_EXCEPTIONS: dict[str, str] = {
    "docker": (
        "publishes images to GHCR under repository credentials; a local run "
        "would either fail on auth or push from a developer machine"
    ),
    "python-test-lmstudio": (
        "opt-in, needs a running LM Studio server on the host; gated behind a "
        "pytest marker that a default run excludes"
    ),
    "python-test-minecraft-e2e": (
        "opt-in, needs docker and a Minecraft server; gated behind the "
        "minecraft_e2e pytest marker"
    ),
    "python-test-minecraft-real-run": (
        "opt-in, drives the full live stack for many minutes; gated behind a "
        "pytest marker"
    ),
    "bench": (
        "criterion comparison against a CI-cached baseline; a local run has no "
        "baseline to compare against and its numbers are not portable between "
        "machines. `cargo bench -p forge-bench` runs the benchmarks themselves"
    ),
    "mc-runner-bundled-image": (
        "builds docker/mc-runner.Dockerfile with FEATURES=mc-live-bundled; "
        "needs the daemon and a full rust+ORT compile inside it, not a "
        "`make verify` laptop gate"
    ),
}

#: Jobs that must be reachable from `make verify` specifically, not merely
#: from *some* target. These are the fast, always-relevant gates; a
#: contributor who runs `make verify` and nothing else should not be able to
#: go red on one of them.
VERIFY_REQUIRED_JOBS: frozenset[str] = frozenset(
    {
        "fmt",
        "clippy",
        "test",
        "python-lint",
        "python-test-fast",
        "mc-bot-test",
        "dashboard",
        "wasm",
        # Seconds-fast, and both were jobs a contributor could go red on after
        # a fully green `make verify`.
        "markdownlint",
        "openspec-validate",
        "forge-mc-runner-bin",
    }
)

#: Matches a Makefile target definition at column 0: `name:` or `name: deps`.
#: Deliberately excludes pattern rules (`%.o:`) and variable assignments.
_MAKE_TARGET_RE = re.compile(r"^(?P<name>[a-zA-Z][a-zA-Z0-9_-]*)\s*:(?!=)", re.MULTILINE)

#: Matches a job id in a GitHub Actions workflow: exactly two spaces of
#: indent under the top-level `jobs:` mapping. Parsed with a regex rather
#: than PyYAML so this check has no third-party dependency, matching the
#: other scripts/check_*.py in this repo.
_CI_JOB_RE = re.compile(r"^  (?P<name>[a-zA-Z][a-zA-Z0-9_-]*):\s*$", re.MULTILINE)


def parse_make_targets(makefile_text: str) -> set[str]:
    """Return every target name defined at column 0 of ``makefile_text``."""
    return {m.group("name") for m in _MAKE_TARGET_RE.finditer(makefile_text)}


def parse_verify_prerequisites(makefile_text: str) -> set[str]:
    """Return the targets `verify` depends on, transitively through `verify-full`.

    Only one level of indirection is followed, which is all this Makefile
    uses; a deeper chain would need a real graph walk and this returns what it
    can see rather than pretending otherwise.
    """
    direct: set[str] = set()
    for target in ("verify", "verify-full"):
        match = re.search(rf"^{re.escape(target)}\s*:(?P<deps>[^\n#]*)", makefile_text, re.MULTILINE)
        if match:
            direct.update(match.group("deps").split())
    return direct


def parse_ci_jobs(workflow_text: str) -> set[str]:
    """Return every job id under the workflow's top-level ``jobs:`` mapping."""
    jobs_start = workflow_text.find("\njobs:")
    if jobs_start == -1:
        raise ValueError("no top-level `jobs:` mapping found in the workflow")
    return {m.group("name") for m in _CI_JOB_RE.finditer(workflow_text[jobs_start:])}


def evaluate(ci_jobs: set[str], make_targets: set[str], verify_deps: set[str]) -> list[str]:
    """Compare the declared mapping against reality; return failure messages."""
    problems: list[str] = []

    unaccounted = sorted(ci_jobs - JOB_TO_MAKE_TARGET.keys() - JOB_EXCEPTIONS.keys())
    if unaccounted:
        problems.append(
            "CI job(s) with no local counterpart and no stated reason. Add a "
            "`make` target and map it in JOB_TO_MAKE_TARGET, or add a reason to "
            "JOB_EXCEPTIONS:\n  " + "\n  ".join(unaccounted)
        )

    dangling = sorted(
        f"{job} -> {target}"
        for job, target in JOB_TO_MAKE_TARGET.items()
        if target not in make_targets
    )
    if dangling:
        problems.append(
            "JOB_TO_MAKE_TARGET names `make` target(s) that no longer exist:\n  "
            + "\n  ".join(dangling)
        )

    stale = sorted((JOB_TO_MAKE_TARGET.keys() | JOB_EXCEPTIONS.keys()) - ci_jobs)
    if stale:
        problems.append(
            "Mapping/exception entries for CI job(s) that no longer exist; "
            "delete them:\n  " + "\n  ".join(stale)
        )

    missing_from_verify = sorted(
        f"{job} (target `{JOB_TO_MAKE_TARGET[job]}`)"
        for job in VERIFY_REQUIRED_JOBS & ci_jobs
        if job in JOB_TO_MAKE_TARGET and JOB_TO_MAKE_TARGET[job] not in verify_deps
    )
    if missing_from_verify:
        problems.append(
            "Job(s) in VERIFY_REQUIRED_JOBS whose target is not a prerequisite "
            "of `make verify` or `make verify-full`, so `make verify` can pass "
            "while CI goes red:\n  " + "\n  ".join(missing_from_verify)
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the parity check. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workflow", type=Path, default=CI_WORKFLOW, help="path to ci.yml"
    )
    parser.add_argument("--makefile", type=Path, default=MAKEFILE, help="path to the Makefile")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="log the parsed job/target sets"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        workflow_text = args.workflow.read_text(encoding="utf-8")
        makefile_text = args.makefile.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.error("cannot read an input file: %s", exc)
        return EXIT_INPUT_ERROR

    try:
        ci_jobs = parse_ci_jobs(workflow_text)
    except ValueError as exc:
        LOGGER.error("cannot parse %s: %s", args.workflow, exc)
        return EXIT_INPUT_ERROR

    make_targets = parse_make_targets(makefile_text)
    verify_deps = parse_verify_prerequisites(makefile_text)

    LOGGER.debug("parsed %d CI jobs: %s", len(ci_jobs), sorted(ci_jobs))
    LOGGER.debug("parsed %d make targets: %s", len(make_targets), sorted(make_targets))
    LOGGER.debug("`verify`/`verify-full` prerequisites: %s", sorted(verify_deps))

    if not ci_jobs:
        LOGGER.error("parsed zero CI jobs from %s; refusing to pass vacuously", args.workflow)
        return EXIT_INPUT_ERROR
    if not make_targets:
        LOGGER.error("parsed zero targets from %s; refusing to pass vacuously", args.makefile)
        return EXIT_INPUT_ERROR

    problems = evaluate(ci_jobs, make_targets, verify_deps)
    if problems:
        print("\nLocal/CI parity drift:\n", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}\n", file=sys.stderr)
        return EXIT_DRIFT

    print(
        f"check_local_ci_parity: OK ({len(ci_jobs)} CI job(s); "
        f"{len(JOB_TO_MAKE_TARGET)} mapped to `make` targets, "
        f"{len(JOB_EXCEPTIONS)} documented as CI-only)"
    )

    # Disposition A: packaging floor must not outrun CI smoke pins.
    matrix = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_python_support_matrix.py")],
        check=False,
    )
    return EXIT_DRIFT if matrix.returncode != 0 else 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())

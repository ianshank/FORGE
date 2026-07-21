# Security Policy

FORGE ships an advisory-first supply-chain scanning workflow
(`.github/workflows/security.yml`: cargo-deny, pip-audit, npm-audit, Trivy, and
optional CodeQL). This document is the disclosure policy that workflow implies.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's private vulnerability reporting:

> **[Report a vulnerability](https://github.com/ianshank/FORGE/security/advisories/new)**
> (repository → *Security* → *Advisories* → *Report a vulnerability*)

Include, where possible: the affected component (crate / mc-bot / Python /
dashboard), a minimal reproduction (FORGE's core is seed-deterministic, so a seed
+ action sequence usually pins core issues), the impact, and any known mitigation.

## What to expect

- Acknowledgement of the report as soon as a maintainer sees it.
- An initial assessment (affected versions, severity) and, for confirmed issues,
  a fix or mitigation plan communicated through the advisory thread.
- Coordinated disclosure: we prefer to publish the advisory once a fix is
  available. Please give maintainers reasonable time before any public
  disclosure.

## Supported versions

FORGE is pre-1.0 and under active development; security fixes target the default
branch. There is no long-term-support branch yet.

## Scope notes

- Secrets are never committed (CHARTER Invariant 7): `.env*` files are
  git-ignored and CI reads secrets only from GitHub Actions secrets/vars.
- The `security.yml` scanners are currently advisory (report-only). Reports of
  genuinely exploitable dependency advisories are still welcome and help
  prioritise flipping a scanner to blocking.

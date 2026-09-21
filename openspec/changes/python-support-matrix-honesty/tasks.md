# Tasks: python-support-matrix-honesty

## Spec Writer

- [x] 1. Author proposal.md, design.md, tasks.md, specs/python-distribution/spec.md under openspec/changes/python-support-matrix-honesty/
- [x] 2. Keep package green under existing CI job `openspec-validate` (`npx @fission-ai/openspec validate --all --strict`); do not invent a separate local-only validate path

## Implementer

- [x] 3. Re-verify live `requires-python` and CI `python-version` pins on main (do not trust stale notes)
- [x] 4. Record Disposition **A** in PR (narrow `>=3.11`); do not switch to B without Intake/Critic override
- [x] 5. Update `pyproject.toml` `requires-python` to `>=3.11` (and related metadata as needed)
- [x] 6. Keep or extend `pip-install-clean` / `python-test` so every declared minor has extension build + `forge_env` import smoke
- [x] 7. Add parity test (declaration vs CI pins); wire into existing Python CI path as a blocker
- [x] 8. Keep `api-compliance` blocking on at least 3.11
- [x] 9. Align README / CHARTER / CONTRIBUTING / CHANGELOG (Unreleased or 0.6.1 section) to the closed matrix; migration note for dropped 3.9/3.10 declaration
- [x] 10. Confirm no cibuildwheel/PyPI/Cosign/trained-MuZero/fault claims introduced

## Conductor (release)

- [ ] 11. After gates green: workspace bump to 0.6.1 and annotated tag `v0.6.1` (Gate D)
- [ ] 12. Continue archive/promotion hygiene for closed `v0.6.0-release-baseline-hardening` (not this change's implement scope)

## Explicit defer checklist

- [ ] cibuildwheel / PyPI matrix
- [ ] Cosign / SBOM provenance
- [ ] Trained public MuZero
- [ ] Motor/GPS fault product
- [ ] Omnibus `run.json`
- [ ] Distilled_Agents / Neuroharness

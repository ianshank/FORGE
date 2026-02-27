# PRD — E17: Mypy Strict Mode Cleanup

**Epic Slug**: `mypy-strict-cleanup`
**Priority**: P1
**Sprint**: 6 (parallel — no feature dependencies)
**Size**: M

---

## User Story

> **As a** contributor to FORGE,
> **I want** the entire Python codebase to pass `mypy --strict` with zero overrides,
> **So that** type errors are caught at PR time rather than at runtime, and the CI signal is trustworthy.

---

## Problem Statement

The current mypy configuration in `pyproject.toml` is already strict for the `python/` package but has two dangerous overrides:

1. `[tool.mypy.overrides] module = ["tests.*", "demo_ui.*"] warn_unused_ignores = false` — silently masks stale `# type: ignore` comments that should be resolved.
2. Several `Any`-typed assignments throughout `demo_ui/` and `examples/` bypass type checking entirely.
3. `callbacks.py` has `_fh: None` typed field but is assigned `TextIOWrapper` — Mypy reports this at severity=error.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | `pyproject.toml` `[tool.mypy]` config | `mypy python/ --strict` | Zero errors; zero `# type: ignore` with stale suppression |
| AC2 | `pyproject.toml` `[tool.mypy]` config | `mypy demo_ui/ --strict` | Zero errors |
| AC3 | `pyproject.toml` `[tool.mypy]` config | `mypy tests/ --strict` | Zero errors |
| AC4 | CI `mypy` step | Any PR | Fails on new `# type: ignore` comments that don't resolve an actual error |
| AC5 | `callbacks.py` `CsvCallback._fh` | mypy runs | Type is `IO[str] \| None` — no incompatible assignment error |
| AC6 | `api.py` `_janitor_task` local | mypy runs | `asyncio.Task[None]` typed correctly; no `RUF006` |
| AC7 | `examples/train_ppo.py` | mypy runs | All `# type: ignore` comments justified by actual errors |

---

## Out of Scope

- Converting JS/TS frontend to TypeScript (separate epic)
- Adding mypy to Rust codebase

---

## Success Metrics

- `mypy python/ demo_ui/ tests/ examples/` exits 0 on `main`
- CI `mypy` step added to `ci.yml` without `--ignore-missing-imports` global flag
- Zero `warn_unused_ignores = false` overrides remaining

---

## Open Questions

1. Should mypy run in CI with `--strict` or only the flags currently set (`disallow_untyped_defs`, etc.)? Recommend **current flags + remove the overrides** —  full `--strict` includes `disallow_any_generics` which would require large-scale changes.
2. Should `wandb` / `mlflow` stubs be committed to `typestubs/` or added as `types-wandb` / `types-mlflow` deps?

---

## Implementation Notes

### `python/forge_env/callbacks.py`

- `_fh: IO[str] | None = None` (import `IO` from `typing`)
- Remove the `# type: ignore[assignment]` from wandb/mlflow `None` assignments — replace with `TYPE_CHECKING` guard

### `python/forge_env/api.py`

- `_janitor_task: asyncio.Task[None]` (not bare `asyncio.Task`)
- Replace `env: Any` in `Session` dataclass with `Protocol` or `Env = gymnasium.Env`

### `pyproject.toml` mypy overrides

```toml
[[tool.mypy.overrides]]
# Remove warn_unused_ignores = false — rely on global setting
module = ["tests.*", "demo_ui.*"]
ignore_missing_imports = true
# warn_unused_ignores = false  ← DELETE THIS LINE
```

### `examples/train_ppo.py` + `train_ppo_cleanrl.py`

- Remove `# type: ignore` on SB3/torch imports; replace with `TYPE_CHECKING` guards
- Use `if TYPE_CHECKING: from stable_baselines3 import PPO` pattern

### `demo_ui/`

- `main.py`: replace `dict[str, Any]` returns with typed `TypedDict`
- `forge_runner.py`: type `SECTIONS: dict[str, str]`, add return types to all async generators

### Tests

- `tests/python/test_mypy.py` *(new)*: subprocess call to `mypy python/ --strict --no-error-summary` asserts exit 0
- Add `mypy python/ demo_ui/ tests/python/ examples/` to `ci.yml` `lint` job

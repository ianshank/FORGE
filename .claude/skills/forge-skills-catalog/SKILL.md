---
name: forge-skills-catalog
description: Add, validate, or wire a hierarchical FORGE skill (options/HRL primitive family) so Rust HierarchicalSkillAgent, Python HierarchicalSkillPolicy, the collector `skill` policy, and configs/agents/skills_default.toml stay in lockstep. Use when adding a reusable skill, changing action families, or debugging catalog/CLI wiring.
---

FORGE skills are **options over primitive `Action`s**, not a new discrete
action-space branch (Sutton–Precup–Singh 1999; SIDM/HMASD/LARAP-style
high-level selection over a library of action primitives).

## Source of truth

1. Horizons, ids, and recipe/comm tokens live in
   `crates/forge-types/src/constants.rs` (`DEFAULT_SKILL_*`, `MIN_SKILL_HORIZON`,
   `MAX_SKILL_HORIZON`).
2. The committed catalog is `configs/agents/skills_default.toml`.
   `SkillsConfig::default()` and `SkillCatalog._default_skills()` must match
   those ids/horizons. The Rust unit test
   `committed_toml_catalog_matches_builtin_ids` pins this.
3. Primitive encoding widths live in `ACTION_BASE_COUNT` (Rust) and
   `FORGE_BASE_ACTIONS` (Python `action_decoder.py`). Do not inline `40`/`19`/`14`.
4. Collection policy names live in `python/forge/policy_names.py`. The CLI
   (`scripts/train.py --collection-policy`) and collector
   (`_create_policy_agent`) must import those constants.

## Adding a skill (checklist)

1. Add a `SkillCategory` variant **only if** a new primitive family is required.
   Prefer mapping a new `Action` onto an existing family in
   `SkillCategory::from_action` (must stay exhaustive).
2. Add named horizon/id constants in `constants.rs`. Never put numeric
   literals in `skill.rs` / `skills.py` executors.
3. Append a `[[skills]]` table to `configs/agents/skills_default.toml` with
   the same field set as existing entries (`id`, `category`, `enabled`,
   `max_horizon`, `requires_drone`, `recipe_index`, `comm_token`). Extra keys
   fail closed (`deny_unknown_fields` / Python `ValueError`).
4. Mirror the builtin in `builtin_skill_catalog()` and Python `_default_skills()`.
5. If the skill needs a new collector policy name, add it to
   `COLLECTION_POLICY_CHOICES` and a branch in `_create_policy_agent`.
6. Gate initiation sets: `requires_drone` + `config.drone.enabled` /
   `can_fly`; agriculture also requires `config.agri.enabled`.
7. Tests (all required):
   - Rust: `cargo test -p forge-types -p forge-agent --test integration_agent_skills_deterministic`
   - Python: `pytest tests/python/test_skill_catalog.py tests/python/test_agent_skills_deterministic.py tests/python/test_train_cli.py -q --no-cov`
8. Keep `[skills]` **opt-in** (`enabled = false` by default) so existing
   TOML/JSON without the section still deserializes.

## Logging

Use `tracing` (`info` on skill switch, `debug` on primitive) in Rust and
the `logging` module in Python. Do not print. Collector rollouts already
label steps with `skill_id` or the decoded skill family.

## Out of scope

- Splitting `WorldState::step_into` or other hot-path god files. Test
  extraction only (`#[path = ".../tests.rs"]`).
- Minecraft `mc-bot` action registries. Those are a different action space;
  do not alias them onto `Action` ids.

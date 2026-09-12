---
name: forge-docs-audit
description: Re-verify factual claims in FORGE's own documentation (docs/next_steps.md, CHANGELOG.md, README.md, docs/architecture.md, Agent.md, CLAUDE.md) against the actual codebase — file:line references, crate/job/test counts, "done"/"resolved" status markers — and correct anything that's drifted. Use when asked to audit docs, check for stale documentation, or before/after a release when these files haven't been touched in a while.
---

Documentation drift is this repo's single most-repeated tech-debt pattern:
`docs/next_steps.md`'s own Technical Debt table currently carries multiple
rows marked resolved specifically because they were re-verified and found
false, and recent history includes several standalone `docs: fix stale ...`
commits. Each required independently re-deriving evidence rather than
trusting the prior claim. This skill packages that recurring verification
work instead of re-inventing it by hand each time.

## Scope

Audit these files for claims that can be mechanically checked, in this
order (cheapest/highest-signal first):

1. `docs/next_steps.md` — the Technical Debt table's own rows, especially
   any NOT already marked `✅ Done`/`✅ Resolved` (those are candidates for
   having become true or false since they were written).
2. `CHANGELOG.md`'s `[Unreleased]` section — claims about what's landed.
3. `README.md`, `docs/architecture.md`, `Agent.md`, `CLAUDE.md`,
   `docs/config-catalog.md` — counts and structural claims (crate counts,
   CI job lists, badge versions, command examples, scenario counts).
4. Graded agri/drone loop (A0–F) claims — if a doc says the compiler,
   orchard grader, CompactReplay v2, VecEnv SPS@N, evidential capture,
   or OpenEnv sidecar landed, confirm the files exist and do **not**
   treat `docs/results/v0.5-trained-vs-random.md` as filled evidence
   (`evidential_episodes >= 3` is still ops-blocked without Docker).

## What counts as a "verifiable claim"

- **A count**: "26 crates", "N CI jobs", "15 `toolchain:` pins across 5
  workflow files", "2,700+ tests". Re-derive with `ls crates/`, `grep -c`,
  or the actual tool (`cargo test --workspace 2>&1 | grep -c 'test result'`).
- **A file:line reference**: "`crates/forge-mc-runner/src/config.rs:287`".
  Read that exact location; confirm the cited content is still there at
  (approximately) that line.
- **A named test**: "covered by `test_torch_path_kl_only_branch`". Confirm
  the test still exists (`grep -rn "def test_torch_path_kl_only_branch"`)
  and, where cheap, that it still passes.
- **A "done"/"resolved"/"fixed" status marker on an OPEN item**: the
  reverse case — something marked open/pending that a `git log --grep`
  or direct inspection shows was actually already fixed.
- **A command example**: does the exact command in the doc still work as
  written (right flags, right paths)? `CONTRIBUTING.md`'s copy-paste
  block and `Makefile` targets are the canonical source of truth for
  these — a doc's own copy should match, not drift independently.
- **A version/tag claim**: "rust:1.85-bookworm", "ort rc.12". Check the
  actual current value (`rust-toolchain.toml`, `Cargo.toml`, the
  Dockerfile) rather than trusting the doc.

## Steps

1. Pick a scope (one file, or the full list above) based on what the user
   asked for or how long it's been since these files were last touched
   (`git log -1 --format=%ad -- docs/next_steps.md`).
2. For each claim in scope, re-derive the actual current fact using the
   cheapest tool that answers it — `git grep`, `wc -l`, `ls`, reading the
   cited file:line, or (only when a static check can't answer it) actually
   running the relevant test/command. Don't assume; the entire point is
   that the prior claim might be exactly what's wrong.
3. When a claim is confirmed accurate, leave it alone — don't touch
   correct content just to reformat it.
4. **Before editing `CHANGELOG.md` specifically: check its line endings
   first** (`file CHANGELOG.md`, or `git check-attr text eol -- CHANGELOG.md`
   — it's `-text` in `.gitattributes`). It is the one file in this repo
   carrying CRLF; every other `.md` is LF. A Python `open(p).read()` /
   `open(p, 'w').write(s)` round-trip silently strips CRLF to LF on read
   and never restores it — this exact accident once turned a ~40-line
   `CHANGELOG.md` edit into a ~2000-line whole-file rewrite (see its own
   `[Unreleased]` entry). Use `sed`/the Edit tool, not a Python text-mode
   read/write, and re-check `file CHANGELOG.md` after editing. (A
   `PreToolUse` hook, `.claude/hooks/guard_line_ending_drift.py`, blocks a
   commit that flips most of a tracked file's line endings — a safety net
   for this specific mistake, not a reason to skip checking first.)
   When a claim has drifted, fix it in place using the house style already
   established in `docs/next_steps.md`:
   `**Stale entry, verified false during the <YYYY-MM> pass.** <what's
   actually true, with the specific evidence — file:line, test name, or
   command output>. No action remains.` (or the equivalent close-out
   phrasing for a row that's genuinely done rather than never having been
   true). Never silently delete a row — mark it resolved with the
   evidence, matching every existing precedent in that file.
5. When a claim is genuinely still open and accurate, leave its status as
   open — this skill corrects drift, it doesn't manufacture progress.
6. Run `npx --yes markdownlint-cli2` on every file touched before calling
   the audit done.
7. Report a short summary: how many claims were checked, how many were
   already accurate, how many were corrected (with a one-line reason
   each) — not a wall of diffs without context.

## Out of scope

- Don't rewrite prose style or restructure sections — this is a factual
  audit, not a copyedit pass.
- Don't invent new Technical Debt rows for things you notice while
  auditing that aren't documentation claims (a real code issue belongs in
  its own fix or its own tracked row, added deliberately — not smuggled
  into a docs-audit commit).
- If a claim requires a slow or environment-dependent check (e.g. a full
  `cargo tarpaulin` run, a GPU-dependent test) to verify precisely, say so
  explicitly rather than guessing — a claim marked "unverified, needs
  `<specific env>`" is more honest than a wrong confirmation.

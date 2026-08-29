---
name: forge-pr-review
description: Dispatch parallel adversarial-review subagents against a PR or diff (hardcoded values & modularity, dead/redundant code, branch-coverage & test-quality via mutation-style reasoning, independent GitHub CI/review-status re-verification, correctness & security), triage every finding as fixed or declined-with-reason, then push, update the PR body's Findings-fixed/Findings-declined section, reply to and resolve addressed review threads, and re-request review. Use before merging a FORGE PR, when asked to "review this PR", "adversarial review the diff", or "triage PR feedback", or as a second pass once `forge-verify` is already green.
---

`forge-verify` confirms the code compiles, lints, and passes; it says nothing
about whether the code is *good* — reusable, free of hardcoded values, free
of dead paths, actually well-tested rather than merely covered, or free of
logic/security defects a green test suite doesn't catch. PR #129 on this
branch is the motivating case: `forge-verify`-equivalent checks were clean,
and a five-lens adversarial pass still found a socket leak on a failed
handshake, a step counted as executed when it had actually failed, an
unvalidated CLI threshold, a bare exception bypassing the module's own error
taxonomy, and a hand-rolled format duplicating an existing cross-language-
pinned constant — plus Copilot's own review independently caught two of the
same defects. This skill packages that dispatch instead of hand-assembling
it via a long freeform prompt each time.

## When to use this

- Before merging any non-trivial FORGE PR, once `forge-verify` is already
  green (this skill assumes the code runs; it hunts for what still runs
  *wrong*).
- When the user asks to "review this PR", "adversarial review the diff", or
  "triage PR feedback".
- Never as a substitute for `forge-verify` — run that first.

## Steps

1. Identify the diff in scope: an open PR number, a branch, or the working
   tree's uncommitted changes. Read the actual diff before writing agent
   prompts — each agent needs concrete file paths and enough context to
   work without re-discovering the PR's own purpose from scratch.

2. Dispatch these five lenses as parallel subagents (one Agent tool call
   per lens, sent in a single message so they run concurrently). Each
   prompt must name the specific files in scope and give enough background
   that the agent can make judgment calls, not just pattern-match:
   - **Hardcoded values & modularity** — every literal that should be a
     named `Final` constant, config value, or CLI flag; unjustified
     duplication with a sibling module that already has the constant.
   - **Dead/redundant code** — unused symbols (grep the *whole* repo, not
     just the diff, before calling anything dead), duplicated test
     fixtures, unnecessary abstraction.
   - **Branch coverage & test quality** — run the actual coverage command
     and verify the number, then reason like a mutation tester: pick 2-3
     risky conditionals and ask whether flipping a boundary would still be
     caught by an existing assertion.
   - **Independent CI/review-status re-verification** — re-check CI and
     review comments from scratch via the GitHub API rather than trusting
     an earlier claim in the conversation; distinguish genuine findings
     from a bot's routine "draft not reviewed" noise.
   - **Correctness & security** — trace the actual control flow for logic
     bugs (off-by-ones, priority-ordering assumptions that only hold by
     coincidence today), resource safety (is cleanup guaranteed on every
     exit path, including one raised before a `try` block begins?), and
     whether any externally-sourced value can reach a path, a format
     string, or a subprocess call.

3. Triage every finding explicitly — there is no third option:
   - **Fix it**: small, well-scoped, verifiable. Apply the fix, then
     re-run the relevant `forge-verify` checks (whichever the diff
     touches) before moving on — never batch unvalidated fixes.
   - **Decline it, with a stated reason**: the finding is real but the fix
     is genuinely out of scope, already a deliberate reviewed tradeoff, or
     the "fix" would itself be worse (see `docs/hardcoded-values-audit.md`
     and this repo's own "don't add abstractions beyond what the task
     requires" convention). A decline without a reason is indistinguishable
     from an ignored finding — always state one.
   Re-verify each of a bot's (Copilot/CodeRabbit) findings against the
   actual code yourself before acting — never fix or decline on the bot's
   word alone.

4. Commit and push the fixes. Update the PR body with an explicit
   "Findings fixed" / "Findings declined" section (see PR #129 on this
   branch for the shape) rather than leaving the triage only in chat.

5. Reply to each addressed review-comment thread citing the fix commit,
   then resolve it. A thread about a declined finding gets a reply
   explaining why, but stays open unless the reviewer would clearly agree
   nothing further is needed.

6. Re-request review (`request_copilot_review`, or the equivalent for a
   human reviewer) once the push lands, so the fresh diff gets checked
   against the new code rather than the stale one.

## Out of scope

- Don't run this on a PR that isn't yours to drive, or push fixes for
  findings you weren't asked to address — see this repo's PR-driving rules
  for what "yours to drive" means.
- Don't invent findings to have something to report. An agent whose lens
  turns up nothing real should say so plainly.
- Not a substitute for a human reviewer's design-level judgment — this
  catches mechanical/hygiene/correctness issues, not architecture calls.

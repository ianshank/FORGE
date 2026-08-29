## Context

FORGE's governance idiom is invariant plus named enforcement, and
`align-charter-with-codebase` established the shape: a guard in Python, under
the existing `python-test` job, that fails on the pre-fix tree. That idiom is
applied thoroughly to code and not at all to evidence.

The gap became observable when a capture run against a bot that could not
execute steps produced thirty records the plotting tool would average. The
capture script's own comment explains the decision — mark the episode
truncated and move on, so one bad step does not crash the whole capture. That
is reasonable for a capture loop and wrong for the record it leaves behind.

## Goals / Non-Goals

**Goals:**

- The aggregator cannot produce a number from records that did not measure the
  system.
- The rule decides the existing corpus, not only future captures.
- A published table cannot silently disagree with the artifact it cites.
- The gate fails on the tree as it stands today and passes after the
  corrections.

**Non-Goals:**

- Provenance verification. It requires git history, and every job in `ci.yml`
  checks out at depth one; only `security.yml` requests full depth, for
  gitleaks. Deferred with the workflow change that would make it executable.
- Readiness declarations. Deferred to a follow-on change.
- Runner outcome typing and accounting. Every committed artifact names the
  Python capture script as its source; the runner produced none of the
  evidence, and the fix changes a public return type. Deferred to its own
  change.
- Scanning prose for unbacked numeric claims. One results document alone
  carries over a hundred numeric tokens — ports, image sizes, toolchain
  versions, grid geometry — with no shape distinguishing them from rewards.
  Deferred until the rule has an operational definition.
- `forge-eval`'s conflated encoding. Its `EpisodeResult` carries the same
  two-boolean pair, and its harness fabricates a zero-reward record when world
  construction fails, which then enters the tier means. That second defect has
  nothing to do with transport and is the more serious of the two; it is a
  separate change, recorded here so the omission is deliberate rather than
  unnoticed.

## Decisions

### Decision: Decide evidentiality from signals the artifacts already carry
Every committed record carries a non-zero protocol-error count and an
observation dimension of zero against a handshake declaring nine hundred and
twenty. Filtering on those yields zero evidential records in both files.
Inventing an outcome vocabulary and applying it only to future captures would
leave the existing corpus undecidable and the gate vacuous on the day it
lands. The producer also emits an explicit outcome, but the aggregator does
not depend on it.

### Decision: Absent signals fail closed
The documented CLI producer emits records carrying neither a protocol-error
count nor a seed, because it projects them from a trajectory. A record whose
evidentiality cannot be established from its own contents is non-evidential.
Failing open would exempt the documented happy path from the rule.

### Decision: Abort on contract violations, halt on transient ones
The bot emits five error codes. Two of them — an unknown action id, and a
malformed frame — mean the action space or the wire shape disagrees across
languages, which is Invariant 2 territory; recording one as another
non-evidential episode and continuing is the silent drift the cross-language
pins exist to prevent. Those abort. The transient three halt the capture after
a pinned number of consecutive occurrences.

### Decision: Two floors, separately named and separately pinned
An episode-level step floor and a snapshot-level evidential-record floor are
different quantities. Naming them both "the minimum" hid that only one had a
proposed value. They are two constants with two paired tests. The step floor
is absolute rather than a fraction of the step budget: no snapshot records a
budget, so a relative rule is uncomputable on the existing corpus, and a
purely relative rule defeats itself — a four-step budget would make two-step
episodes evidential.

A paired pin is a review control, not a machine control: editing the constant
and its test in one change passes everything. It is weaker than the
cross-language `schema_id` pins it resembles, which span three independently
owned languages so a unilateral edit breaks two other suites. A single Python
constant beside a single Python test is a one-hand pin. It is worth having
because it makes the lowering visible in a diff; it is not worth describing as
enforcement.

### Decision: State what the gate does not detect
Evidentiality is decided from fields the producer writes, and nothing binds
them to the wire events that produced them. Hand-editing a step count and a
protocol-error count in a committed snapshot reclassifies the record. The gate
detects accident, drift, and copy-paste error; it does not detect deliberate
fabrication and is not intended to. Saying so matters more than the check
itself, because the failure this change could introduce is a future reader
trusting a number because it passed a gate.

### Decision: The gate has no exclusion mechanism, and cannot be excluded
Three of the cheapest ways to turn this gate green require no dishonesty at
all: delete the artifact, add a test marker, or add an ignore file. The marker
path is a two-line diff matching four documented precedents in
`pyproject.toml`, and it removes the gate from `python-test`, `make py-test`,
and `make verify` at once. So the gate derives its file set from the
filesystem with no ignore file and no exclusion config, a separate assertion
checks that the suite's marker expression excludes nothing this gate carries,
and an index of expected artifacts makes deletion as visible in a diff as
lowering a floor.

### Decision: The step floor applies only to truncated outcomes
A natural terminal reached in twelve steps is the most informative episode
type there is. Excluding it for brevity would discard exactly the evidence the
gate exists to protect.

### Decision: Narrow the corpus gate to two decidable assertions
A row citing a committed snapshot must agree with it, and a snapshot with no
evidential records must carry a declaration. Both are decidable with no git
history, no prose scanner, and no not-reproducible-block syntax to specify.
Both fail on the tree today. The broader claim-citation rule is deferred until
it has a definition a machine can own.

### Decision: Land advisory for one cycle, then flip to blocking
The repository has a codified advisory-first doctrine, stated in the security
workflow's header with the exact promotion recipe, followed by five scanners
and the unused-dependency job, and recorded in the charter with named exit
conditions. Landing a never-run integrity control straight into a blocking job
inverts that pattern, and the blast radius is releases rather than merges: the
job hosting it gates image publication on the default branch and on version
tags. One advisory cycle confirms the gate runs at all and fails for the
intended reason, then the marker comes off. Immediate bite is still the goal;
this is how the repository reaches it.

### Decision: Do not strengthen Invariant 6 in this change
An evidence-strengthening paragraph naming this gate would overstate what it
checks, and the charter's own headline throughput claim is itself unbacked —
one of the two referenced baseline directories is empty, already logged as
technical debt. Strengthening the invariant belongs with the change that earns
it. Two independently verified accuracy defects in Invariant 6's own lists are
corrected here, because they are exactly the citation rot the preceding change
existed to eliminate.

## Risks / Trade-offs

- [The gate checks two properties, not the general claim the incident
  suggests] → both fail on today's tree and neither can be satisfied by
  editing prose alone; the broader rule is deferred rather than shipped
  undefined.
- [Declaring rather than deleting the artifacts keeps a misleading file in the
  tree] → the declaration and the index are what keep the gate non-vacuous;
  deleting them would leave every assertion iterating an empty set.
- [A paired pin can be lowered by editing two lines in one change] → stated as
  a review control rather than presented as enforcement, so no reader mistakes
  its strength.
- [An advisory cycle delays real enforcement by one iteration] → it is the
  repository's own documented sequence, and it converts a possible silent
  no-op into an observed failure before anything depends on the gate.

## Open Questions

- Values for the two floors and the consecutive-failure threshold. Proposed:
  three consecutive failures halt; an episode needs at least five realised
  steps to be evidential when truncated; a snapshot needs at least three
  evidential records to support a comparison. Decidable in flight, since each
  is pinned by a paired test either way.

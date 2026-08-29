## Purpose

Ensure a record that did not measure the system cannot enter an aggregate, and
that a published table cannot disagree with the artifact it cites.

## ADDED Requirements

### Requirement: Environment-Reported Step Failures Are Not Truncations
When the environment reports that it could not execute a requested step, the
record SHALL carry an outcome denoting that failure. The condition SHALL NOT
be recorded by setting a flag whose other uses denote a completed episode.

#### Scenario: An error reply is not recorded as truncation
- **GIVEN** a capture whose connection is healthy
- **AND** the environment replies to a step with an error frame
- **WHEN** the per-episode record is written
- **THEN** the record SHALL carry an environment-failure outcome
- **AND** SHALL NOT be marked truncated on that basis

#### Scenario: Reaching the step budget stays a capability outcome
- **GIVEN** an episode that reaches its step budget with no error reply
- **WHEN** the record is written
- **THEN** the outcome SHALL denote truncation and the record SHALL be
  evidential

### Requirement: Contract-Violating Error Codes Abort The Capture
An error code indicating that the action space or the wire shape disagrees
across languages SHALL abort the capture. Such a condition SHALL NOT be
recorded as one more non-evidential episode.

#### Scenario: An unknown action id aborts rather than continuing
- **GIVEN** a capture in which the environment rejects an action as unknown
- **WHEN** the failure is observed
- **THEN** the capture SHALL abort and report the contract violation

### Requirement: Capture Halts On Sustained Transient Failure
A capture SHALL halt once the pinned number of consecutive episodes end in a
transient environment failure, rather than continuing to attempt and record
the remainder.

#### Scenario: Sustained failure stops the run
- **GIVEN** a capture configured for thirty episodes and a halt threshold of
  three
- **WHEN** three consecutive episodes end in environment failure
- **THEN** the capture SHALL halt and report the failure
- **AND** SHALL NOT write records for the remaining episodes

#### Scenario: An isolated failure does not stop the run
- **WHEN** one episode ends in environment failure and the next succeeds
- **THEN** the capture SHALL continue

### Requirement: Evidentiality Is Decided From Recorded Signals
A record SHALL be evidential only when its recorded protocol-error count is
zero, its recorded observation dimension equals the dimension declared at
handshake, and — where the outcome denotes truncation — its realised step
count meets the pinned step floor. A record whose evidentiality cannot be
established from its own contents SHALL be treated as non-evidential.

#### Scenario: A one-step record with a protocol error is not evidential
- **GIVEN** a record reporting one step, one protocol error, and an
  observation dimension of zero
- **WHEN** evidentiality is decided
- **THEN** the record SHALL be non-evidential

#### Scenario: A natural terminal is evidential regardless of length
- **GIVEN** a record whose outcome denotes a natural terminal at twelve steps
- **AND** whose protocol-error count is zero and whose observation dimension
  matches the handshake
- **WHEN** evidentiality is decided
- **THEN** the record SHALL be evidential

#### Scenario: A record lacking the deciding signals is not evidential
- **WHEN** a record carries no protocol-error count
- **THEN** it SHALL be treated as non-evidential

### Requirement: Aggregates Exclude Non-Evidential Records
Summary statistics SHALL be computed only over evidential records. Any output
presenting a mean, median, deviation, or percentile SHALL report the
evidential and excluded counts, and SHALL refuse to present a comparison when
the evidential count falls below the pinned evidential floor.

#### Scenario: A snapshot of non-episodes yields no statistic
- **GIVEN** a snapshot of thirty records, all non-evidential
- **WHEN** a report is requested
- **THEN** the tool SHALL refuse, exit non-zero, and state that thirty records
  were excluded

#### Scenario: An empty record set does not yield a zero mean
- **GIVEN** a snapshot with no records
- **WHEN** a report is requested
- **THEN** the tool SHALL refuse rather than present a mean of zero

### Requirement: Both Floors Are Pinned Against Silent Lowering
The episode step floor and the snapshot evidential floor SHALL each be a named
constant with a paired test asserting its value, so that lowering either
appears in the same reviewed change.

#### Scenario: Lowering a floor fails its pinning test
- **WHEN** either floor's value is changed
- **THEN** the test pinning that floor SHALL fail in the same change

### Requirement: Published Tables Agree With The Snapshots They Cite
A table row under the committed results path that links a snapshot SHALL agree
with that snapshot on episode count and on any per-episode value the row
states.

#### Scenario: A row contradicting its cited snapshot is rejected
- **GIVEN** a row citing a snapshot and stating an episode count
- **AND** the snapshot records a different count
- **WHEN** the evidence gate runs
- **THEN** it SHALL fail and report both values

### Requirement: Snapshots Without Evidential Records Are Declared
A committed snapshot containing no evidential records SHALL carry a
declaration recording what it is and why it is retained. The declaration's
rationale is an audit note, not a verified claim; its enforcement value is
that retaining such a snapshot is a visible, deliberate act.

#### Scenario: An undeclared empty snapshot is rejected
- **GIVEN** a committed snapshot whose records are all non-evidential
- **AND** no declaration accompanies it
- **WHEN** the evidence gate runs
- **THEN** it SHALL fail

### Requirement: The Expected Evidence Set Is Indexed
Committed snapshots SHALL be listed in an index carrying each path and a
digest of its contents. The gate SHALL fail when an indexed path is missing or
its digest disagrees, so that removing or silently editing evidence requires
an index change in the same diff.

#### Scenario: Deleting an indexed snapshot is rejected
- **GIVEN** a snapshot listed in the index
- **WHEN** the file is removed without editing the index
- **THEN** the evidence gate SHALL fail and name the missing path

#### Scenario: Editing a snapshot in place is rejected
- **WHEN** an indexed snapshot's contents change without its digest being
  updated
- **THEN** the evidence gate SHALL fail

### Requirement: The Gate Cannot Be Excluded Or Silenced
The gate SHALL derive its file set from the filesystem and SHALL NOT provide a
path-exclusion mechanism. A separate assertion SHALL verify that the test
suite's marker filter excludes no marker this gate carries, and the gate SHALL
fail rather than skip when a precondition it depends on is unavailable.

#### Scenario: Excluding the gate by marker is caught
- **GIVEN** the suite's default marker expression is extended to exclude this
  gate
- **WHEN** the suite runs
- **THEN** the separate assertion SHALL fail

#### Scenario: A missing precondition fails rather than skips
- **WHEN** the gate cannot establish a property it depends on
- **THEN** it SHALL fail and name the cause, and SHALL NOT report success

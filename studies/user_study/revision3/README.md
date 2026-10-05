# Revision 3: array relationships and complete attempt trails

New assignments use version-3 task IDs. Existing version-1 and version-2 links
retain their original definitions, order, saved code and budgets.

Both language conditions receive identical contracts, public examples, hidden
cases, manuals, and budgets. Public examples remain three per task; this revision
does not change feedback quantity between groups. Every hint explicitly accepts
straightforward loop implementations.

## What changed in the tasks

| Task | Relationships participants must maintain |
|---|---|
| Batch calibration and energy | `[B,C]` calibration parameters align with `[B,T,C]` readings; reduce time while preserving batch/channel |
| Cross-sensor centered products | Two observation arrays share `[B,T]` but have different channel counts `C,D`; separate centering produces a rectangular `[B,C,D]` result |
| Projected weighted distances | Batch-specific `[B,D,F]` projections map both point sets to `F`, then `[B,F]` weights apply to every within-batch point pair; `D!=F` and `N!=M` are supported |
| Strided dilated filters | Output stride and filter dilation are both 2; track effective filter extent, floor-rounded output shape, image batch, input channels and output filters |
| Rectangular block covariance | Group non-overlapping 2-by-3 blocks into six observations, center per block, then form every channel pair; output has five axes |
| Batch-specific masked diffusion | `[B,H,W]` masks and `[B,C,C]` mixing matrices align with four-dimensional state; preserve boundaries/frozen cells and synchronous rounds |

All relationships, allowed special cases, and output formulas are explicit in
the bilingual [task descriptions](tasks.md). These are programming requirements,
not hidden exceptions. Signed/asymmetric data and rectangular profiles distinguish
axis mistakes; all implementations use the same independent scalar oracle.

Provisional budgets remain 10, 12, 15, 18, 18 and 20 minutes (93 total). They require
another human pilot before recruitment.

## Full attempt trail

Each accepted **Run examples** or **Custom input** request creates an immutable
public run and a linked private `attempt_grade` record in one transaction.
Repeated identical code is still a separate attempt. Compilation errors are
retained. Practice has no hidden grading. Final submission is shown alongside
all attempts, and ordinary saved draft history remains available.

Each attempt records:

- Task, attempt ordinal, draft revision, full frozen source and SHA-256.
- Wall-clock request time and cumulative active task time at the click.
- Active interval since the preceding attempt (since task start for attempt 1).
- Public execution start/finish times and feedback, including compile/runtime errors.
- Frozen hidden inputs and a private evaluation linked by `parent_run_id`.
- Hidden outcome and passed/total cases, distinct from infrastructure failures.

**Active intervals exclude pauses but include reading, editing, and public
execution waits. They are not a measure of typing time.** Rating time is excluded
from final-submission active time. Queue time can be recovered from request and
execution timestamps; hidden evaluation time is a separate job and never added
to the captured participant time.

Capture is immediate; private evaluation is deferred until the participant
finishes the entire session or withdraws, so it does not add work during that
participant's timed tasks. A first attempt is not replaced by a later successful
compilation. Every accepted attempt is evaluated, regardless of public-test
outcome. Queued/running hidden jobs do not block public run requests.

The researcher detail page shows a bilingual chronological table with code,
revision, active times, public execution duration, public feedback and hidden
score. Export schema 2 includes a joined `attempt_trail` plus all raw runs,
snapshots and events. Retry jobs preserve the source, inputs, attempt ordinal and
original timing metadata; interrupted scoring is an infrastructure interruption,
not a zero correctness score. Historical runs without the new timing/score
metadata are displayed as missing, not reconstructed speculatively.

Participants cannot obtain private jobs through the run endpoint, participant
state, or a guessed run-ID URL. Researcher endpoints still require authentication.
Hidden examples/results are never used as participant feedback.

## Compatibility and verification

Every one of 59 inventoried Zinnia implementations must pass. Coverage includes
broadcasting, reductions, einsum, batched matrix products, per-batch dot and
tensordot, algebraic identities, explicit projection buffers, projection of
pairwise differences, reshape/transpose, spatial slices, nested/chained/while
loops, comprehensions, copies and buffer updates. No algorithm is mandated.

- 177 implementation/shape combinations, eight witness fixtures per shape:
  1,416 compiled correct-output checks and 177 incorrect-output controls.
- Actual participant engine gate: 59 Zinnia implementations and six Rust
  references, 24 inputs each (1,560 checks).
- Six real Halo2 proof smoke checks, one scalar implementation and nonzero
  fixture per task, including verification.
- Store/API tests cover every-attempt capture, pause-excluded intervals,
  duplicate code, custom runs, concurrency, immutable retry, restart recovery,
  private endpoint access, and deferred evaluation.
- Browser tests cover both language conditions, full task flows, the researcher
  trail table, completed private evaluations and private run-ID denial.

These are finite compatibility tests, not a guarantee for every possible valid
program or the entire NumPy ecosystem. New sensible alternatives remain required
regression cases. No new compiler changes were needed for this task revision.

Reports: [compiler gate](validation.json), [participant engines](site-validation.json),
[real proofs](proofs.json). Each report fingerprints its compiler and inventory.
Previous study versions remain available for regression and historical analysis.

From `zinnia-src`:

```sh
.venv/bin/python -m studies.user_study.revision3.validate --jobs 4 --output studies/user_study/revision3/validation.json
.venv/bin/python -m studies.user_study.revision3.validate_site
.venv/bin/python -m studies.user_study.revision3.validate --jobs 2 --variant nested_loops --profile 0 --fixture 6 --backend halo2 --timeout 180 --output studies/user_study/revision3/proofs.json
.venv/bin/python -m unittest studies.user_study.site.tests.test_attempt_trail studies.user_study.site.tests.test_revision3 studies.user_study.site.tests.test_api
npm run test:browser --prefix studies/user_study/site
```

The engine/API/browser tests use temporary databases and require local sandbox,
process and socket permissions. They do not use the live participant links.

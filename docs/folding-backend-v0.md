# Folding Backend v0 Design

## Purpose

This document scopes an experimental folding-scheme backend for Zinnia, using
Nova/HyperNova-style incrementally verifiable computation as the motivating
proof-system family. The v0 goal is not broad backend replacement. The goal is a
clean RAM/prover-time comparison between the current Halo2 backend and a folding
backend on workloads where folding has a natural repeated-step structure.

The comparison should answer two questions:

- Does folding reduce peak prover memory for long bounded repeated tensor
  workloads?
- Where is the crossover between Halo2's one-shot static circuit cost and a
  folding backend's per-step overhead?

## Acceptance Boundary

Folding is allowed only after Zinnia has already established a finite step count:

- The step count is public, or
- The program is padded to a public bound established by Zinnia's
  semantic/refinement analysis.

Folding must not be the mechanism that makes an unbounded program bounded. The
core Zinnia claim remains:

1. semantic/refinement analysis establishes a bounded and statically defined
   computation shape;
2. a backend proves that bounded shape.

Halo2 and folding are then alternative proof-system lowerings over an already
bounded workload. If no finite public step count or padded public bound exists,
the folding backend must reject the workload.

## Current Backend Shape

The current prover stack already has useful backend seams:

- `src/prove/traits.rs` defines `ProverBackend` with `estimate_params`, `prove`,
  and `verify`.
- The same file defines `Synthesizer`, a per-operation constraint/evaluation
  interface.
- `src/prove/interpreter.rs` walks an `IRGraph` in statement order and dispatches
  each IR instruction to a `Synthesizer`.
- `src/lib.rs` exposes `prove_circuit`, which creates a backend by name,
  preprocesses external calls, and calls `backend.prove(...)`.
- `src/prove/types.rs` defines a backend-tagged `ProofArtifact`.

The Halo2 backend is specifically a whole-program static-circuit lowering:

- `src/prove/halo2/circuit.rs` constructs one `ZinniaCircuit` for the entire
  optimized `IRGraph`.
- `ZinniaCircuit::synthesize` interprets the full IR, records Halo2 operations,
  replays them into one Halo2 region, and exposes public instances.
- `src/prove/halo2/mod.rs` estimates `k`, collects public inputs, validates the
  witness with Halo2 `MockProver`, keygens/caches by circuit shape, creates a
  proof, and verifies by reconstructing the verifying key from IR carried in the
  artifact.

The current shared arithmetic layer is not proof-system-neutral yet:

- `src/prove/kernel.rs` uses `pasta_curves::Fp` throughout.
- It also re-exports `halo2_proofs::arithmetic::Field`.
- Mock and Halo2 share that kernel to keep arithmetic semantics aligned.

This means a folding backend can reuse the same semantics initially if it uses a
compatible Pasta field, but a mature backend should eventually separate
proof-system-neutral semantic evaluation from Halo2-specific field/circuit
details.

## Why Not Fold Flat IR Statements?

The current IR is a flattened sequence of piece-wise operations. Folding each IR
statement would produce an unnatural step relation:

- the step would be too small;
- public/private state would be awkward;
- per-step folding overhead would dominate;
- memory savings would not represent the value of folding on real repeated
  computations;
- the measurement would mainly evaluate a poor encoding strategy.

The v0 folding backend must therefore reject arbitrary flat IR. It should accept
only workloads that provide an explicit semantic step relation.

## v0 Model

Introduce a backend-internal shape:

```text
FoldableProgram
  step_relation: StepRelation
  state_in: StateSchema
  step_inputs: InputSchema
  state_out: StateSchema
  public_params: PublicParamSchema
  public_outputs: PublicOutputSchema
  step_count: PublicStepCount
  padding: OptionalPaddingPolicy
```

The key invariant is:

```text
state_{i+1} = step_relation(state_i, step_inputs_i, public_params)
```

for `i` in `0..step_count`.

The folding proof should prove the repeated application of this relation for the
declared public step count or padded bound.

The proof statement must bind:

- `state_0`;
- `state_T`;
- public parameters;
- public step count;
- padding policy;
- declared public outputs.

For padded executions, inactive steps must be constrained as semantic no-ops and
the public output binding must refer to the unpadded source-level result.

### StepRelation Requirements

A v0 step relation must be:

- explicit, not inferred from arbitrary IR statement order;
- deterministic;
- bounded by a public count or padded public bound;
- uniform enough for the selected folding library;
- restricted to a supported arithmetic subset;
- connected to the same source-level workload used for the Halo2 comparison.

### Public Bound Requirements

Every v0 folding benchmark row must record:

- source-level workload name;
- semantic bound source, such as literal shape, refined symbolic shape, or
  explicit public parameter;
- step count used by folding;
- whether padding is used;
- relation between the Halo2 workload and folding workload.

Rows without this information are not accepted as folding evidence.

## Accepted v0 Workloads

Start with 2-3 workloads where the step boundary is natural.

Every accepted workload also needs a source-level equivalence obligation. It is
not enough to prove a plausible recurrence. The doc, lowering, and result row
must state why the folded step sequence computes the same source-level tensor
program that the Halo2 baseline proves.

### Dot Product by Block

Step:

```text
(acc, block_index) -> (acc + dot(x_block, y_block), block_index + 1)
```

Why it fits:

- small repeated arithmetic relation;
- public number of blocks;
- easy Halo2 comparison using the same vector length and block size;
- useful for measuring per-step overhead and memory behavior.

Source-level equivalence obligations:

- block partitioning covers every source vector element exactly once;
- blocks have no overlap and no omission;
- block order is fixed or proved irrelevant to the accumulated result;
- the final `acc` equals the source-level dot-product result;
- partial final blocks are either padded with multiplicative-zero elements or
  rejected unless the length is a multiple of the block size;
- padding steps do not change `acc`.

### Iterative Stencil or Jacobi Step

Step:

```text
grid_t -> grid_{t+1}
```

Why it fits:

- natural iterative algorithm;
- public iteration count;
- likely larger per-step relation than dot-product;
- relevant to tensor/scientific workloads.

v0 should choose a small fixed shape first. Larger shapes can be sweep points.

Source-level equivalence obligations:

- the folded step implements the same update rule as the source tensor program;
- boundary-cell handling is identical to the source program;
- the grid layout used in the step relation matches the source array layout;
- the public iteration count or padded bound matches the source workload;
- inactive padded iterations are constrained as no-ops after the real iteration
  count.

### Repeated Kernel Invocation

Step:

```text
state_i, input_i -> state_{i+1}, output_i
```

Why it fits:

- exposes folding as proof aggregation over repeated uses of the same bounded
  tensor kernel;
- provides a path toward a later subproof aggregation design.

Comparison baselines:

- Primary baseline: one Halo2 proof for the same repeated workload and public
  bound. This is the fair comparison against one folding proof.
- Secondary baseline: many standalone Halo2 proofs, one per invocation. This can
  be useful operational data, but it must not be collapsed into the primary
  crossover story.

Source-level equivalence obligations:

- each folded invocation uses the same kernel semantics as the source program;
- state carried between invocations matches the source-level data dependency;
- input/output ordering is preserved;
- the final state and declared public outputs match the repeated source
  workload;
- padding invocations are constrained as no-ops.

## Explicit Reject Categories

Unsupported features must be classified as explicit rejects, not hidden
fallbacks to Halo2 and not poor folding encodings.

Initial reject categories:

- `no_public_step_bound`: no finite public step count or padded bound.
- `flat_ir_no_step_relation`: workload only exists as arbitrary flattened IR.
- `dynamic_memory_not_supported`: dynamic ndarray memory, memory traces, or
  random-access dynamic writes are required.
- `lookup_heavy_fixed_point_not_supported`: correctness depends on current
  Halo2-specific lookup/range-check machinery that v0 folding has not modeled.
- `poseidon_not_supported`: Poseidon/hash operations appear in the foldable
  region.
- `transcendentals_not_supported`: sin/cos/exp/log/sqrt/arccos/etc. appear in
  the foldable region.
- `external_callback_not_supported`: the foldable region depends on external
  Python callbacks or witness-time hints not represented in the step relation.
- `nonuniform_step_not_supported`: the selected library/backend cannot handle
  the required non-uniform step shape.

These categories should appear in the benchmark results table alongside
successful folding rows.

Rejected rows are non-evidence for folding speed and memory claims. They are
useful coverage/classification data, but crossover curves must include only
accepted workloads with natural step relations.

## Artifact and API Changes

### Rust

Add a module behind a feature flag:

```text
src/prove/folding/
  mod.rs
  types.rs
  lower.rs
  backend.rs
  metrics.rs
```

Add backend names:

- `folding-nova`
- optionally `folding-mock` for relation validation without cryptographic proof

Wire through:

- `prove::create_prover_backend`
- `ProofArtifact.backend`
- `ProofArtifact.format`

The folding artifact should carry:

- backend name and format;
- selected folding scheme/library;
- step relation identifier/hash;
- public step count;
- padding policy;
- proof bytes;
- public inputs/outputs;
- metrics summary or a stable metrics sidecar path.

### Python

Update the public config and proof wrapper:

- Add folding backend names to `ZinniaConfig`.
- Preserve `ProofArtifact.format` in `ZKProofResult`.
- Allow backend-specific params beyond Halo2's `{k, precision_bits,
  lookup_bits}`.

Current `ZKProofResult` does not preserve `format` even though Rust artifacts
have one. A folding backend should fix that before relying on Python round-trip
verification for new artifact formats.

## Lowering Strategy

Do not implement `Synthesizer` for arbitrary IR as the first folding backend.

Instead:

1. Define `FoldableProgram` and `StepRelation`.
2. Implement a validator that accepts only the v0 subset and returns explicit
   reject categories otherwise.
3. Hand-write or pattern-lower the first accepted kernels into `StepRelation`.
4. Build the folding proof from the step relation.
5. Separately run the same source-level workload through the current Halo2 path.

After v0, if the results justify it, add compiler support to preserve step
boundaries before flattening:

- loop iteration boundaries;
- tensor row/block/tile boundaries;
- repeated kernel invocation boundaries;
- source locations tying the foldable relation back to the original program.

## Evaluation Protocol

Every accepted row must use the same source-level tensor workload and same
public bound for Halo2 and folding.

Measure:

- compile/lowering time;
- setup/keygen/preprocessing time;
- prover time;
- verifier time;
- proof size;
- peak RSS;
- per-step folding overhead;
- final step count;
- padding overhead, if any.

Report:

- successful folding rows;
- explicit reject rows by category;
- Halo2 baseline metrics;
- folding metrics;
- crossover curves over step count and/or problem size;
- qualitative classification: naturally foldable vs forced/not accepted.

Halo2 and folding results should be reported separately rather than collapsed
into a single "backend succeeded" field.

Rejected rows must not dilute or improve crossover curves. Report them in a
separate classification table, not as zero-cost, failed, or fallback folding
measurements.

## Security and Soundness Notes

- Folding does not justify dynamic or hidden execution length. The public step
  count or padded bound is part of the statement.
- Padding must constrain inactive steps so they are semantically no-ops.
- The initial state, final state, public inputs, and public outputs must be
  bound into the folding proof statement.
- Source-level preconditions used to establish bounds must be enforced without
  leaking private predicate witnesses. Public preconditions may be part of the
  public statement. Private structural preconditions must be enforced inside the
  same proof statement, or by a prior proof whose result is bound into the
  folding statement. The public artifact should reveal the public bound or
  padding class, not necessarily the private predicate witness.
- Unsupported features must not silently call Halo2 inside the folding backend.

## Milestones

### M0: Design Review

Deliver this design doc and get sign-off on:

- acceptance boundary;
- step relation shape;
- initial accepted workloads;
- reject categories;
- evaluation protocol.

### M1: Relation Validation Prototype

Implement non-cryptographic `folding-mock` relation validation for one accepted
kernel.

Output:

- accepted/rejected classification;
- public step count;
- per-step witness/state trace;
- source workload mapping.

### M2: First Cryptographic Backend

Integrate a Rust folding library for one arithmetic-only kernel.

Candidate libraries to evaluate:

- Microsoft Nova: Rust implementation of Nova IVC, suitable for sequential
  computation proofs. See <https://github.com/microsoft/Nova>.
- Spartan2 NeutronNova: useful to inspect for batch folding R1CS instances, but
  its documented `neutronnova_zk` module is non-recursive batch folding, so it is
  not the first choice for IVC-style repeated execution. See
  <https://docs.rs/spartan2/latest/spartan2/neutronnova_zk/index.html>.

Output:

- proof generation;
- verification;
- metrics collection;
- explicit unsupported-feature errors.

### M3: Halo2 vs Folding Crossover Study

Run dot-product/block, stencil/Jacobi, and repeated-kernel sweeps.

Output:

- peak RSS curves;
- prover-time curves;
- verifier-time/proof-size table;
- reject-category table;
- recommendation on whether preserving step boundaries in compiler IR is worth
  implementing.

## Non-Goals for v0

- General support for arbitrary Zinnia IR.
- General dynamic ndarray memory support.
- Transcendental-function support.
- Poseidon/hash support.
- Replacing the Halo2 backend.
- Claiming folding is faster or better without measured crossover data.
- Hiding execution length unless padding to a public bound is implemented.

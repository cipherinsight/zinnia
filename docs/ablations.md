# Ablation switches (measurement only)

Three environment variables turn off one part of the compiler's use of facts, so the
evaluation can report what each part contributes. Each is read by the compiler itself
(set it in the environment of the process that compiles). Unset, or set to anything but
`1`, they change nothing: the compiled IR is identical to a build without them.

| Variable | What it turns off | Where |
|---|---|---|
| `ZINNIA_ABLATE_SMT=1` | Every Z3 query that reasons about values answers `Unknown` without calling Z3: the IR generator's resolver and `prove()` (bounds, preconditions, index checks, strategy choices), the shape-parameter bounds, and the optimizer's resolver. Each caller treats `Unknown` as "no information", so what is left is the interval and fact-scan tiers. The construction selection of the lowering layer (a Z3 optimization over costs, not reasoning about values) stays on. | `optim::z3_budget::{check, check_assuming, check_optimize}` |
| `ZINNIA_ABLATE_LAYER_FACTS=1` | The array-lowering layer gets no free facts (no bounds, no gadget postconditions) and ignores the program's checks: no check is dropped as implied by an earlier one or by a construction's postcondition, and no check justifies a construction's precondition. Every precondition fact gets a check node (the selection may still use another construction's postcondition when that is cheaper). | `lower::emit::lower`, `lower::facts::FreeFacts::none` |
| `ZINNIA_ABLATE_CHECK_ELIM=1` | The two optimizer passes that drop program checks, `AlwaysSatisfiedElimination` and `AssertSubsumption`, are not run. Checks the IR generator never emitted (discharged at IR generation by `prove()` or by bounds) are not affected; that is `ZINNIA_ABLATE_SMT`'s territory. | `lib.rs::compile_circuit` |

Related, older switches (not ablations of the fact layer):

- `ZINNIA_SMT_ENABLE=0` replaces only the IR generator's resolver (`require_static_int`
  and the other resolver chokepoints) with the static-only resolver, which also loses the
  interval tier. It does not affect `prove()`, the shape-parameter bounds, or the
  optimizer passes.
- `ZINNIA_REQ_DISABLE=1` makes `prove()` answer `Unknown`, so every op precondition is
  checked in the circuit and every strategy takes its default; the interval fallbacks of
  the index checks still apply.

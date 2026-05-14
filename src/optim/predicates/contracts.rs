//! Per-op contract types and per-IR-kind contract registry.
//!
//! Each registered op kind gets an [`OpContract`] describing its effect on
//! structural-predicate facts:
//!
//! - `requires`: preconditions that must hold for the op to be sound.
//! - `ensures`: postconditions the op guarantees about its result.
//! - `frame`: which global state changes, if any.
//!
//! Contracts are *templates*: they reference formal `Input(name)` and
//! `Output` variables that the discharge layer substitutes at chokepoint
//! time. See `formula.rs` for the AST.
//!
//! ## Adding a contract (downstream-card recipe)
//!
//! 1. Pick the IR class-name (e.g., `"AllocateMemoryIR"`) — same string
//!    [`IR::class_name`] returns.
//! 2. Write the `requires` and `ensures` as `Vec<ContractFormula>` using
//!    the [`crate::optim::predicates::formula::ContractTerm`] builders.
//!    Each formula's top-level must be `Bool`.
//! 3. Insert into the registry via `m.insert(class_name, OpContract { … });`
//!    in [`build_contract_registry`].
//!
//! Soundness obligation: every `ensures` formula must hold for the op's
//! intended semantics. Reviewers should re-derive each one before approving
//! the PR.
//!
//! ## Scope of this card
//!
//! The framework ships empty contracts for every kind. Per-predicate cards
//! (W1–W5) populate real contract content. The foundation slot is intact;
//! this card refines the *shape* of the slot from opaque strings (the
//! foundation placeholder) to a structured AST that the discharge layer
//! can audit and lower mechanically.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ir_defs::IR;
use crate::optim::predicates::formula::ContractTerm;

// ---------------------------------------------------------------------------
// FrameCondition
// ---------------------------------------------------------------------------

/// Frame condition: what state does the op *not* modify?
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FrameCondition {
    /// Pure / functional. Default for current Zinnia ops.
    #[default]
    Pure,
    /// Op writes to one or more witness-time aux memories. The list names
    /// the affected segments. Reserved; not used in the foundation
    /// scope.
    Effectful(Vec<String>),
}

// ---------------------------------------------------------------------------
// ContractFormula
// ---------------------------------------------------------------------------

/// A single contract clause: a `ContractTerm` whose top-level lowers to a
/// `Bool`. The discharge layer instantiates this via
/// [`crate::optim::predicates::formula::lower_bool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractFormula {
    pub term: ContractTerm,
}

impl ContractFormula {
    pub fn new(term: ContractTerm) -> Self {
        Self { term }
    }
}

// ---------------------------------------------------------------------------
// OpContract
// ---------------------------------------------------------------------------

/// The contract template attached to an IR op kind.
#[derive(Debug, Clone, Default)]
pub struct OpContract {
    pub requires: Vec<ContractFormula>,
    pub ensures: Vec<ContractFormula>,
    pub frame: FrameCondition,
}

impl OpContract {
    /// The default contract: empty pre/post, pure frame.
    pub fn default_contract() -> Self {
        Self::default()
    }

    /// Convenience: build with `requires` / `ensures` and Pure frame.
    pub fn pure(requires: Vec<ContractFormula>, ensures: Vec<ContractFormula>) -> Self {
        Self {
            requires,
            ensures,
            frame: FrameCondition::Pure,
        }
    }

    pub fn is_default(&self) -> bool {
        self.requires.is_empty()
            && self.ensures.is_empty()
            && matches!(self.frame, FrameCondition::Pure)
    }
}

// ---------------------------------------------------------------------------
// Contract registry — keyed by IR class_name
// ---------------------------------------------------------------------------

/// Return the contract attached to the given IR statement, looked up by
/// the IR's `class_name()` string.
///
/// Falls back to [`OpContract::default_contract`] for any kind without a
/// real registration — the empty contract is sound (it constrains
/// nothing).
pub fn op_contract_for(ir: &IR) -> OpContract {
    let class = ir.class_name();
    contract_registry()
        .get(class)
        .cloned()
        .unwrap_or_default()
}

/// Name-keyed registry lookup for ops without a single distinguished IR
/// statement (composite/orchestrating ops like the dyn-ndarray allocator,
/// which expand into many low-level IR ops at the call site).
///
/// The name is an arbitrary identifier chosen by the op's author and
/// agreed-upon with [`build_contract_registry`]. Falls back to the empty
/// contract if no registration exists.
pub fn op_contract_by_name(name: &str) -> OpContract {
    contract_registry().get(name).cloned().unwrap_or_default()
}

/// Lazy initialiser for the contract registry. Per-predicate cards add
/// entries in [`build_contract_registry`].
fn contract_registry() -> &'static HashMap<&'static str, OpContract> {
    static R: OnceLock<HashMap<&'static str, OpContract>> = OnceLock::new();
    R.get_or_init(build_contract_registry)
}

/// Slack (positive f64) added to nominal transcendental bounds so that
/// `LitFloat(PI ± slack)` is a sound widening of the true real-valued
/// bound. Necessary because true π is not exactly representable in f64:
/// `std::f64::consts::PI` rounds *down* to the nearest representable
/// double, slightly underapproximating π. Asserting `Output <=
/// LitFloat(PI)` would therefore be unsound for outputs in the tiny
/// interval `(LitFloat(PI), π]`. Adding a positive slack makes the
/// upper-bound strictly greater than true π and reclaims soundness.
///
/// Env var `ZINNIA_TRANSCENDENTAL_BOUND_SLACK_F64` overrides the default
/// of `0.001`. Non-finite or negative values fall back to the default.
/// The value is read once at registry-init time (the registry is
/// `OnceLock`-cached), so mid-process env-var mutations have no effect.
fn transcendental_slack() -> f64 {
    std::env::var("ZINNIA_TRANSCENDENTAL_BOUND_SLACK_F64")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
        .unwrap_or(0.001)
}

/// Build the contract registry. This is the extension point for
/// per-predicate cards.
///
/// W1 (`structural-predicate-nnz`) adds contracts for `nonzero`-style ops
/// once the IR lowering for `np.zeros(k, ...)` etc. is plumbed to emit
/// structured contract markers. The foundation contracts card ships the
/// *empty* table — the framework wiring is what's load-bearing here.
fn build_contract_registry() -> HashMap<&'static str, OpContract> {
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut m: HashMap<&'static str, OpContract> = HashMap::new();

    // Helper: a single ensures clause `Var(Output) >= zero`. The literal
    // is parameterised so int- and float-sorted contracts share the
    // template (the lowering layer coerces to Real when either side is a
    // float).
    fn ensures_output_nonneg(zero: ContractTerm) -> ContractFormula {
        ContractFormula::new(ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
            rhs: Box::new(zero),
        })
    }

    // Helper: a single ensures clause `lo <= Var(Output) <= hi`, emitted
    // as `BoolComb(And, [Output >= lo, Output <= hi])`. Used for ops
    // whose output is structurally clamped to a closed interval (e.g.
    // `sign`).
    fn ensures_output_in_range(lo: ContractTerm, hi: ContractTerm) -> ContractFormula {
        ContractFormula::new(ContractTerm::BoolComb {
            op: crate::optim::predicates::formula::BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(lo),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(hi),
                },
            ],
        })
    }

    // ── dyn_fill_with_active (zeros / ones / empty on dyn-ndarray) ────
    //
    // Soundness: a dyn-ndarray returned from `dyn_fill_with_active` carries
    // a `runtime_length` that's structurally non-negative — the constructor
    // accepts an `active_len: Value` derived from a chip input whose
    // ZK-circuit type is `int` (signed but bounded by the user's
    // `@requires`). The contract asserts the postcondition that the
    // result's runtime length is `>= 0`. This is a genuine semantic
    // claim — the resolver cannot derive it from SSA aliasing alone.
    // Additionally, `runtime_length == active` holds by construction: the
    // constructor copies the input `active_len` ScalarValue into the
    // result's `runtime_length`. Surfacing this as a multi-formal equality
    // clause lets the resolver relate `len(arr)` back to the input `n` even
    // when only `n` carries call-site facts.
    //
    // Template formals:
    //   `Output` — substituted to the runtime-length SSA ptr at the call
    //   site. (The dyn-ndarray's "result ptr" for length-reasoning purposes
    //   is the ptr of `runtime_length`, not the composite value as a whole.)
    //   `Formal("active")` — substituted to the input `active_len` ValueId.
    m.insert(
        "dyn_fill_with_active",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                        "active".to_string(),
                    ))),
                }),
            ],
        ),
    );

    // ── dyn_filter (mask-based selection) ──────────────────────────────
    //
    // Soundness: `dyn_filter` constructs its runtime length by initialising
    // a write pointer to 0 and incrementing it for each kept element (loop
    // body in `src/ops/dyn_ndarray/memory_ops.rs::dyn_filter`). Therefore
    // `runtime_length >= 0` always holds. The complementary upper bound
    // `runtime_length <= max_len` (where `max_len = data.max_length()`) is
    // emitted as a call-site-local fact alongside the registry-driven one,
    // because `max_len` is a per-call-site constant not expressible as a
    // pure template.
    m.insert(
        "dyn_filter",
        OpContract::pure(vec![], vec![ensures_output_nonneg(ContractTerm::LitInt(0))]),
    );

    // ── dyn_concatenate ───────────────────────────────────────────────
    //
    // Soundness: `dyn_concatenate` (`src/ops/dyn_ndarray/memory_ops.rs`)
    // builds `runtime_length` as `sum(runtime_axis_len_i) * other_product`,
    // where each `runtime_axis_len_i` is the i-th input's runtime axis
    // length (non-negative by induction on the constructors) and
    // `other_product` is the static product of the non-concatenated axes
    // (non-negative by construction). So `runtime_length >= 0` always.
    // A richer fact (`runtime_length == sum(args[i].len) * other_product`)
    // is expressible but requires multi-formal binding; deferred until a
    // consumer needs it.
    m.insert(
        "dyn_concatenate",
        OpContract::pure(vec![], vec![ensures_output_nonneg(ContractTerm::LitInt(0))]),
    );

    // ── dyn_argextremum (argmax / argmin on dyn arrays) ──────────────
    //
    // Soundness: `dyn_aggregate_all` (`src/ops/dyn_ndarray/aggregation.rs`)
    // initialises `acc_idx` to `0` and updates it via a select chain
    // `select(cmp(elem_i, acc), i, acc_idx)` for `i ∈ 1..numel`. Every
    // branch of every select is a literal in `[0, numel-1]`. Therefore
    // the final returned index is in `[0, numel)`. The lower bound
    // `Output >= 0` is a pure template fact; the upper bound
    // `Output < Formal("len_arr")` is multi-formal — the caller binds
    // `len_arr` to the runtime-length ValueId (dyn) or to a materialised
    // static-length IR constant (static).
    //
    // Why bother: the walker COULD reconstruct the bound by visiting
    // the entire select chain, but that's `O(numel)` IR statements per
    // query. The contract caches the result of a one-time analysis so
    // downstream chokepoints (e.g. indexing with `argmax(arr)`) discharge
    // their bound check via the fact-fallback instead of paying the full
    // walk.
    //
    // Template formals:
    //   `Output` — the argmax/argmin result ValueId.
    //   `Formal("len_arr")` — the array's length ValueId. Callers must
    //   bind this; firing with an empty formals map yields a fact whose
    //   `Formal("len_arr")` leaf survives unresolved and would error if
    //   it ever reached the lowering layer (per `LowerError::UnboundFormal`).
    m.insert(
        "dyn_argextremum",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Lt,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                        "len_arr".to_string(),
                    ))),
                }),
            ],
        ),
    );

    // ── abs_i (integer absolute value) ───────────────────────────────
    //
    // Soundness: `|x| >= 0` for every integer `x` (in two's-complement
    // arithmetic the only exception is `i64::MIN`, whose negation
    // overflows; Zinnia's `AbsI` operates on field-bounded ints and the
    // backend's range check rejects that input before this fact is
    // consulted, so the postcondition is sound on every value the op
    // can legitimately accept).
    m.insert(
        "abs_i",
        OpContract::pure(vec![], vec![ensures_output_nonneg(ContractTerm::LitInt(0))]),
    );

    // ── abs_f (float absolute value) ─────────────────────────────────
    //
    // Soundness: `|x| >= 0.0` for every real `x`. NaN is the only IEEE
    // input that breaks the comparison; Zinnia's `AbsF` is defined over
    // the ZK Real fragment which excludes NaN, so the postcondition
    // holds on every value the op accepts.
    m.insert(
        "abs_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_nonneg(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            ))],
        ),
    );

    // ── sqrt_f (real square root) ────────────────────────────────────
    //
    // Soundness: `sqrt(x)` is real-valued iff `x >= 0`; the principal
    // (non-negative) root is returned, so `sqrt(x) >= 0` holds on every
    // accepted input. The requires `Var(Formal("x")) >= 0.0` is
    // discharged at call time against the input value's fact bucket.
    m.insert(
        "sqrt_f",
        OpContract::pure(
            vec![ContractFormula::new(ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Formal("x".to_string()))),
                rhs: Box::new(ContractTerm::LitFloat(
                    crate::optim::predicates::formula::ContractFloat(0.0),
                )),
            })],
            vec![ensures_output_nonneg(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            ))],
        ),
    );

    // ── log_f (real natural logarithm) ────────────────────────────────
    //
    // Soundness: `log(x)` is real-valued iff `x > 0` (strict positive);
    // `log(0) = -∞` and `log(x)` for `x < 0` is not real. The output
    // bound is deferred to a Tier-2 follow-up (per the design card)
    // because it requires an interval split on `x >= 1` vs `0 < x <= 1`.
    m.insert(
        "log_f",
        OpContract::pure(
            vec![ContractFormula::new(ContractTerm::Cmp {
                op: CmpOp::Gt,
                lhs: Box::new(ContractTerm::Var(ContractVar::Formal("x".to_string()))),
                rhs: Box::new(ContractTerm::LitFloat(
                    crate::optim::predicates::formula::ContractFloat(0.0),
                )),
            })],
            vec![],
        ),
    );

    // ── exp_f (real exponential) ─────────────────────────────────────
    //
    // Soundness: `exp(x) > 0.0` for every real `x`. The looser fact
    // `exp(x) >= 0.0` is recorded here because the strict `>` form
    // adds no consumer-visible bound (no current resolver branches on
    // strict-vs-non-strict positivity for `exp`).
    m.insert(
        "exp_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_nonneg(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            ))],
        ),
    );

    // ── sign_i (integer sign) ────────────────────────────────────────
    //
    // Soundness: every IR codepath through `SignI` produces exactly
    // `-1`, `0`, or `+1`, so `-1 <= Output <= 1` holds on every value
    // the op produces.
    m.insert(
        "sign_i",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitInt(-1),
                ContractTerm::LitInt(1),
            )],
        ),
    );

    // ── sign_f (float sign) ──────────────────────────────────────────
    //
    // Soundness: every IR codepath through `SignF` produces exactly
    // `-1.0`, `0.0`, or `+1.0`, so `-1.0 <= Output <= 1.0` holds on
    // every value the op produces.
    m.insert(
        "sign_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(-1.0)),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(1.0)),
            )],
        ),
    );

    // ── sin_f (real sine) ────────────────────────────────────────────
    //
    // Soundness: `sin(x) ∈ [-1.0, 1.0]` for every real `x`. The ZK Real
    // fragment excludes NaN, so the comparison is well-defined on every
    // accepted input.
    m.insert(
        "sin_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(-1.0)),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(1.0)),
            )],
        ),
    );

    // ── cos_f (real cosine) ──────────────────────────────────────────
    //
    // Soundness: `cos(x) ∈ [-1.0, 1.0]` for every real `x`. The ZK Real
    // fragment excludes NaN, so the comparison is well-defined on every
    // accepted input.
    m.insert(
        "cos_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(-1.0)),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(1.0)),
            )],
        ),
    );

    // ── tanh_f (real hyperbolic tangent) ─────────────────────────────
    //
    // Soundness: `tanh(x) ∈ (-1.0, 1.0)` strictly for every real `x`.
    // The recorded fact is the sound looser closed `[-1.0, 1.0]`; the
    // strict open form would require a separate template and gains
    // nothing for current consumers.
    m.insert(
        "tanh_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(-1.0)),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(1.0)),
            )],
        ),
    );

    // ── cosh_f (real hyperbolic cosine) ──────────────────────────────
    //
    // Soundness: `cosh(x) >= 1.0` for every real `x` — the codomain on
    // the ZK Real fragment is `[1, ∞)`. We reuse `ensures_output_nonneg`
    // here; the helper's name is misleading (the literal is `1.0`, not
    // `0`), but reusing it keeps the registry compact. Rename only if a
    // third arbitrary-lower-bound caller arrives.
    m.insert(
        "cosh_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_nonneg(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(1.0),
            ))],
        ),
    );

    // ── var / std (population variance and standard deviation) ───────
    //
    // Soundness: `np.var` returns `mean((x - mean(x))**2)`, which is a
    // sum of squares divided by N. Each squared term is non-negative on
    // the ZK Real fragment (NaN is excluded), so the mean of those
    // squares is `>= 0`. `np.std` is `sqrt(var)`; with the variance
    // non-negative the principal square root is real and also `>= 0`.
    //
    // Both ops are undefined on empty input — variance and std are mean-
    // based and `mean([])` divides by zero. The precondition
    // `len_arr >= 1` forbids the empty case. `len_arr` is multi-formal:
    // the caller binds it to the static array length (materialised as
    // an `ir_constant_int`) for static-array inputs, or to the dyn
    // array's `runtime_length` ValueId for the dyn-array path.
    m.insert(
        "var",
        OpContract::pure(
            vec![ContractFormula::new(ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                    "len_arr".to_string(),
                ))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            })],
            vec![ensures_output_nonneg(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            ))],
        ),
    );
    m.insert(
        "std",
        OpContract::pure(
            vec![ContractFormula::new(ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                    "len_arr".to_string(),
                ))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            })],
            vec![ensures_output_nonneg(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            ))],
        ),
    );

    // ── Comparison / logical ops: Output ∈ {0, 1} ────────────────────
    //
    // Soundness: every comparison (Eq/Ne/Lt/Le/Gt/Ge × {Int, Float}) and
    // every logical op (And/Or/Not) in Zinnia's IR returns a Bool whose
    // ZK encoding is the integer pair `{0, 1}`. The recorded fact
    // `0 <= Output <= 1` is the closed-interval relaxation of that
    // two-point codomain; sound for every codepath of every op in the
    // cluster. One shared block covers all 15 entries — the rationale
    // is universal. `bool_cast` joins the cluster on the same total-`→
    // {0,1}` semantics. The four `bit_*_i` entries (and/or/xor/not) are
    // only sound under the conditional fire-site guard in the hand-rolled
    // builder methods (Boolean-typed inputs only); the registry entries
    // are dormant unless fired by that guarded path.
    fn ensures_output_bool() -> ContractFormula {
        ensures_output_in_range(ContractTerm::LitInt(0), ContractTerm::LitInt(1))
    }
    m.insert("eq_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("eq_f", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("ne_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("ne_f", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("lt_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("lt_f", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("lte_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("lte_f", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("gt_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("gt_f", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("gte_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("gte_f", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("logical_and", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("logical_or", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("logical_not", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("bool_cast", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("bit_and_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("bit_or_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("bit_xor_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("bit_not_i", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("int_cast_from_bool", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert(
        "float_cast_from_bool",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(0.0)),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(1.0)),
            )],
        ),
    );

    // ── Boolean reductions (`np.all` / `np.any`) ─────────────────────
    //
    // Soundness: `builtin_reduce` (static, `helpers/ndarray.rs`) lowers
    // `all` through `ir_logical_and` and `any` through `ir_logical_or`;
    // the dyn-side `dyn_aggregate_all` uses the same IR ops with the
    // boolean algebraic identity (true for `all`, false for `any`) for
    // inactive slots. Every IR codepath through both ops produces a Bool
    // wire whose ZK encoding is in `{0, 1}`. The recorded fact
    // `0 <= Output <= 1` is the closed-interval relaxation of that
    // two-point codomain. Reuses the comparison cluster's
    // `ensures_output_bool` helper.
    m.insert("all", OpContract::pure(vec![], vec![ensures_output_bool()]));
    m.insert("any", OpContract::pure(vec![], vec![ensures_output_bool()]));

    // ── arccos_f / arctan2_f (transcendental bounds with global slack) ──
    //
    // Soundness: true π is not exactly representable in f64;
    // `std::f64::consts::PI` is the nearest representable double, which
    // rounds *down* from true π. A naïve `Output <= LitFloat(PI)` upper
    // bound would therefore be unsound for outputs in the (real-valued)
    // interval `(LitFloat(PI), π]`. The slack — a strictly positive f64,
    // defaulting to `0.001` and overridable via
    // `ZINNIA_TRANSCENDENTAL_BOUND_SLACK_F64` — widens the recorded
    // bound so that `LitFloat(PI + slack) > true_π` strictly, restoring
    // soundness with a small precision loss. Lower bounds are widened
    // symmetrically for consistency (arccos's exact 0.0 lower bound is
    // also widened to `0.0 - slack` purely for symmetry; 0.0 is f64-
    // exact, so the strict `0 - slack` widening is correct but
    // unnecessary).
    //
    // - `arccos(x) ∈ [0, π]` for `x ∈ [-1, 1]`
    //   → recorded as `[0 - slack, PI + slack]`.
    // - `arctan2(y, x) ∈ (-π, π]`
    //   → recorded as `[-PI - slack, PI + slack]`.
    let slack = transcendental_slack();
    let pi = std::f64::consts::PI;

    m.insert(
        "arccos_f",
        OpContract::pure(
            vec![ContractFormula::new(ContractTerm::BoolComb {
                op: crate::optim::predicates::formula::BoolOp::And,
                operands: vec![
                    ContractTerm::Cmp {
                        op: CmpOp::Ge,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Formal("x".to_string()))),
                        rhs: Box::new(ContractTerm::LitFloat(
                            crate::optim::predicates::formula::ContractFloat(-1.0),
                        )),
                    },
                    ContractTerm::Cmp {
                        op: CmpOp::Le,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Formal("x".to_string()))),
                        rhs: Box::new(ContractTerm::LitFloat(
                            crate::optim::predicates::formula::ContractFloat(1.0),
                        )),
                    },
                ],
            })],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(
                    0.0 - slack,
                )),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(
                    pi + slack,
                )),
            )],
        ),
    );

    m.insert(
        "arctan2_f",
        OpContract::pure(
            vec![],
            vec![ensures_output_in_range(
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(
                    -pi - slack,
                )),
                ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(
                    pi + slack,
                )),
            )],
        ),
    );

    // ── dyn_arange (bounded `np.arange` admission) ───────────────────
    //
    // Soundness: the bounded branch in `np_arange` sets
    // `runtime_length = stop - start` (1-arg form binds `start` to a
    // literal 0 at the firing site; the 2-arg form forwards the user's
    // literal `start`; the 3-arg form is intentionally NOT covered here
    // — its runtime_length is `ceildiv(stop - start, step)` and would
    // need a separate template). The equality `Output == stop - start`
    // mirrors that IR computation exactly; `Output >= 0` is structurally
    // guaranteed because the constructor truncates negative spans to 0.
    m.insert(
        "dyn_arange",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Arith {
                        op: crate::optim::predicates::formula::ArithOp::Sub,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "stop".to_string(),
                        ))),
                        rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "start".to_string(),
                        ))),
                    }),
                }),
            ],
        ),
    );

    // ── dyn_tile (bounded `np.tile` admission) ───────────────────────
    //
    // Soundness: the bounded branch in `np_tile` sets
    // `runtime_length = len_arr * k`, where `len_arr` is the (static)
    // length of the 1-D source array materialised at the call site as a
    // literal `ir_constant_int`, and `k` is the user's bounded reps. The
    // equality `Output == len_arr * k` mirrors that IR computation;
    // `Output >= 0` holds since both factors are non-negative (k by the
    // bounded admission's lower bound, len_arr by construction).
    m.insert(
        "dyn_tile",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Arith {
                        op: crate::optim::predicates::formula::ArithOp::Mul,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "len_arr".to_string(),
                        ))),
                        rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "k".to_string(),
                        ))),
                    }),
                }),
            ],
        ),
    );

    // ── dyn_repeat (bounded `np.repeat` / `ndarray.repeat` admission) ──
    //
    // Soundness: identical shape to `dyn_tile`. The bounded branch in
    // `ndarray_repeat` sets `runtime_length = len_arr * k`. Registered
    // under a distinct name purely for diagnostic clarity (so the
    // firing-site name in stack traces matches the user's call).
    m.insert(
        "dyn_repeat",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Arith {
                        op: crate::optim::predicates::formula::ArithOp::Mul,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "len_arr".to_string(),
                        ))),
                        rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "k".to_string(),
                        ))),
                    }),
                }),
            ],
        ),
    );

    // ── dyn_linspace (bounded `np.linspace` admission) ───────────────
    //
    // Soundness: the bounded branch in `np_linspace` sets
    // `runtime_length = num` (the result is a 1-D dyn-ndarray of length
    // equal to the user's `num` scalar). The equality `Output == num`
    // mirrors that aliasing exactly; `Output >= 0` holds since the
    // bounded admission requires `num >= 1` (or `num >= 2` when
    // endpoint=true) at the call site.
    m.insert(
        "dyn_linspace",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                        "num".to_string(),
                    ))),
                }),
            ],
        ),
    );

    // ── zeros_content / ones_content (fill-constructor content fact) ──
    //
    // Soundness: every output element of `np.zeros(shape)` is the IR
    // constant 0; every output element of `np.ones(shape)` is the IR
    // constant 1. The `*_like` variants build the output from a fresh
    // fill operation (not a copy of the input), so the same fact applies.
    // `forall_eq_const` is registered as a cached uninterpreted predicate
    // (see `registry.rs`); the deposited fact is observable via
    // `prove(forall_eq_const(out, k))` returning `Proved`.
    //
    // The `_content` suffix in the registry names distinguishes from
    // length-related entries on the same op family. `np.full(shape, k)`
    // for arbitrary k uses a multi-formal template (Group 4b, separate).
    m.insert(
        "zeros_content",
        OpContract::pure(
            vec![],
            vec![ContractFormula::new(ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Output),
                    ContractTerm::LitInt(0),
                ],
            })],
        ),
    );
    m.insert(
        "ones_content",
        OpContract::pure(
            vec![],
            vec![ContractFormula::new(ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Output),
                    ContractTerm::LitInt(1),
                ],
            })],
        ),
    );

    // ── identity_content (np.identity content fact) ──────────────────
    //
    // Soundness: `np.identity(N)` produces an N×N matrix whose element
    // at `(i, j)` equals `1` iff `i == j`, else `0` — definitionally the
    // identity matrix. The `is_identity` predicate (registered as a
    // cached uninterpreted Bool in `registry.rs`) names this property;
    // the deposited fact is observable via `prove(is_identity(out))`
    // returning `Proved`. Consumed by the matmul Phase F strategy set to
    // short-circuit `I @ B → B` and `A @ I → A`.
    m.insert(
        "identity_content",
        OpContract::pure(
            vec![],
            vec![ContractFormula::new(ContractTerm::PredicateApp {
                kind: "is_identity".to_string(),
                args: vec![ContractTerm::Var(ContractVar::Output)],
            })],
        ),
    );

    // ── dyn_identity (bounded `np.identity` admission) ───────────────
    //
    // Soundness: the bounded branch in `np_identity` builds a 2-D
    // dyn-ndarray of shape `N x N` whose `runtime_length = N * N` (set
    // via `ir_mul_i(n_arg, n_arg)`). The equality `Output == N * N`
    // mirrors that IR computation; `Output >= 0` holds since N is
    // non-negative by the bounded admission's lower bound.
    m.insert(
        "dyn_identity",
        OpContract::pure(
            vec![],
            vec![
                ensures_output_nonneg(ContractTerm::LitInt(0)),
                ContractFormula::new(ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Output)),
                    rhs: Box::new(ContractTerm::Arith {
                        op: crate::optim::predicates::formula::ArithOp::Mul,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "N".to_string(),
                        ))),
                        rhs: Box::new(ContractTerm::Var(ContractVar::Formal(
                            "N".to_string(),
                        ))),
                    }),
                }),
            ],
        ),
    );

    // ── div / floor_div / mod (integer + float divisor-nonzero requires) ──
    //
    // Soundness: division and modulo are undefined when the divisor is
    // zero; for ZK the witness emitter constrains `rhs != 0` so the prover
    // refuses to produce a witness violating the precondition. Six entries
    // (3 int + 3 float) share a single template each.
    fn requires_rhs_ne_zero_int() -> ContractFormula {
        ContractFormula::new(ContractTerm::Cmp {
            op: CmpOp::Ne,
            lhs: Box::new(ContractTerm::Var(ContractVar::Formal("rhs".to_string()))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        })
    }
    fn requires_rhs_ne_zero_float() -> ContractFormula {
        ContractFormula::new(ContractTerm::Cmp {
            op: CmpOp::Ne,
            lhs: Box::new(ContractTerm::Var(ContractVar::Formal("rhs".to_string()))),
            rhs: Box::new(ContractTerm::LitFloat(
                crate::optim::predicates::formula::ContractFloat(0.0),
            )),
        })
    }
    m.insert("div_i", OpContract::pure(vec![requires_rhs_ne_zero_int()], vec![]));
    m.insert(
        "floor_div_i",
        OpContract::pure(vec![requires_rhs_ne_zero_int()], vec![]),
    );
    m.insert("mod_i", OpContract::pure(vec![requires_rhs_ne_zero_int()], vec![]));
    m.insert("div_f", OpContract::pure(vec![requires_rhs_ne_zero_float()], vec![]));
    m.insert(
        "floor_div_f",
        OpContract::pure(vec![requires_rhs_ne_zero_float()], vec![]),
    );
    m.insert("mod_f", OpContract::pure(vec![requires_rhs_ne_zero_float()], vec![]));

    // ── inv_i (modular-inverse `x != 0` requires) ────────────────────
    //
    // Soundness: `1/x` over the field is undefined at `x == 0`; the
    // unary single-formal `"x"` is auto-bound by `ir_unary!`.
    fn requires_x_ne_zero_int() -> ContractFormula {
        ContractFormula::new(ContractTerm::Cmp {
            op: CmpOp::Ne,
            lhs: Box::new(ContractTerm::Var(ContractVar::Formal("x".to_string()))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        })
    }
    m.insert("inv_i", OpContract::pure(vec![requires_x_ne_zero_int()], vec![]));

    // ── pow_i / pow_f (domain `base != 0 OR exp >= 0` requires) ──────
    //
    // Soundness: `pow(0, e)` is `0` for `e > 0`, `1` for `e == 0`, and
    // mathematically undefined for `e < 0` (limit doesn't exist; the
    // candidate value `1/0` diverges). The disjunction `base != 0 OR
    // exp >= 0` admits every well-defined case (0^0 = 1 via the second
    // branch with `e == 0`) and refuses only `base == 0 AND exp < 0`.
    // The float entry mirrors the int shape with `LitFloat(0.0)`; the
    // out-of-scope cases (negative base with non-integer exponent
    // producing a complex result) are deferred per the card's notes.
    fn requires_pow_int() -> ContractFormula {
        ContractFormula::new(ContractTerm::BoolComb {
            op: crate::optim::predicates::formula::BoolOp::Or,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ne,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Formal("lhs".to_string()))),
                    rhs: Box::new(ContractTerm::LitInt(0)),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Formal("rhs".to_string()))),
                    rhs: Box::new(ContractTerm::LitInt(0)),
                },
            ],
        })
    }
    fn requires_pow_float() -> ContractFormula {
        ContractFormula::new(ContractTerm::BoolComb {
            op: crate::optim::predicates::formula::BoolOp::Or,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ne,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Formal("lhs".to_string()))),
                    rhs: Box::new(ContractTerm::LitFloat(
                        crate::optim::predicates::formula::ContractFloat(0.0),
                    )),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Formal("rhs".to_string()))),
                    rhs: Box::new(ContractTerm::LitFloat(
                        crate::optim::predicates::formula::ContractFloat(0.0),
                    )),
                },
            ],
        })
    }
    m.insert("pow_i", OpContract::pure(vec![requires_pow_int()], vec![]));
    m.insert("pow_f", OpContract::pure(vec![requires_pow_float()], vec![]));

    // ── arange_is_sorted / linspace_is_sorted (range-constructor content) ──
    //
    // Soundness: `np.arange(stop)`, `np.arange(start, stop)` and
    // `np.arange(start, stop, step)` with `step > 0` produce a strictly
    // ascending sequence; `np.linspace(start, stop, num)` with
    // `start <= stop` produces a (non-strictly) ascending sequence. In
    // both cases `is_sorted(out)` holds. The firing sites in
    // `np_arange` / `np_linspace` (static + dyn) gate the emission on a
    // call-site direction check, so descending or unknown-direction
    // forms simply skip the fire — no false claim. Kept as separate
    // registry entries so the firing-site name in traces matches the
    // user-visible op.
    m.insert(
        "arange_is_sorted",
        OpContract::pure(
            vec![],
            vec![ContractFormula::new(ContractTerm::PredicateApp {
                kind: "is_sorted".to_string(),
                args: vec![ContractTerm::Var(ContractVar::Output)],
            })],
        ),
    );
    m.insert(
        "linspace_is_sorted",
        OpContract::pure(
            vec![],
            vec![ContractFormula::new(ContractTerm::PredicateApp {
                kind: "is_sorted".to_string(),
                args: vec![ContractTerm::Var(ContractVar::Output)],
            })],
        ),
    );

    m
}

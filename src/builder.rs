use std::collections::HashMap;

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::optim::resolver::{Resolver, StaticOnlyResolver};
use crate::types::{ScalarValue, StmtId, StringValue, Value};

/// `ZINNIA_OP_REQUIRES_STRICT=1` flips op-build-time `requires` discharge
/// from lenient (witness-emit on Unknown) to strict (compile error on
/// Unknown). Matches the precedent set by `ZINNIA_BOUNDED_AXIS_STRICT`.
/// Read on every discharge so test guards take effect immediately; the
/// per-call overhead is a single env lookup.
pub(crate) fn op_requires_strict() -> bool {
    std::env::var("ZINNIA_OP_REQUIRES_STRICT")
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

// Optional `contract = "…"` per entry auto-fires the named op contract on
// the output's `value_id` (no-op when the output has no ptr). Implemented
// as a TT-muncher because Rust can't disambiguate `, contract` from `,
// $next_name` with a single-token-lookahead optional group.
//
// Contract-bearing entries auto-bind the inputs as formals `"lhs"` and
// `"rhs"` so per-op `requires` clauses referencing `Var(Formal("lhs"))` /
// `Var(Formal("rhs"))` instantiate without per-call-site boilerplate.
// When an input has no `value_id` (pure constant), the contract fire is
// skipped — neither the formal nor the output anchor is materialisable.
macro_rules! ir_binary {
    () => {};
    ($name:ident => $variant:expr, contract = $contract:literal $(, $($rest:tt)*)?) => {
        pub fn $name(&mut self, a: &Value, b: &Value) -> Value {
            let out = self.create_ir(&$variant, &[a.clone(), b.clone()]);
            if let (Some(vid), Some(lhs_vid), Some(rhs_vid)) =
                (out.value_id(), a.value_id(), b.value_id())
            {
                let mut formals = HashMap::new();
                formals.insert("lhs".to_string(), lhs_vid);
                formals.insert("rhs".to_string(), rhs_vid);
                self.fire_contract($contract, vid, &formals);
            }
            out
        }
        $( ir_binary!($($rest)*); )?
    };
    ($name:ident => $variant:expr $(, $($rest:tt)*)?) => {
        pub fn $name(&mut self, a: &Value, b: &Value) -> Value {
            self.create_ir(&$variant, &[a.clone(), b.clone()])
        }
        $( ir_binary!($($rest)*); )?
    };
}

// Optional `contract = "…"` per entry auto-fires the named op contract on
// the output's `value_id` (no-op when the output has no ptr). TT-muncher
// for the same reason as `ir_binary!`.
//
// Contract-bearing entries auto-bind the input as formal `"x"` so
// elementwise `requires` clauses can reference `Var(Formal("x"))` without
// per-call-site boilerplate. Skips firing if the input has no `value_id`.
macro_rules! ir_unary {
    () => {};
    ($name:ident => $variant:expr, contract = $contract:literal $(, $($rest:tt)*)?) => {
        pub fn $name(&mut self, a: &Value) -> Value {
            let out = self.create_ir(&$variant, &[a.clone()]);
            if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
                let mut formals = HashMap::new();
                formals.insert("x".to_string(), x_vid);
                self.fire_contract($contract, vid, &formals);
            }
            out
        }
        $( ir_unary!($($rest)*); )?
    };
    ($name:ident => $variant:expr $(, $($rest:tt)*)?) => {
        pub fn $name(&mut self, a: &Value) -> Value {
            self.create_ir(&$variant, &[a.clone()])
        }
        $( ir_unary!($($rest)*); )?
    };
}

macro_rules! ir_ternary {
    ($($name:ident => $variant:expr),* $(,)?) => {
        $(
            pub fn $name(&mut self, a: &Value, b: &Value, c: &Value) -> Value {
                self.create_ir(&$variant, &[a.clone(), b.clone(), c.clone()])
            }
        )*
    };
}

/// IR builder that accumulates IR statements and provides typed convenience
/// methods. Mirrors Python `IRBuilderImpl` from `builder_impl.py`.
pub struct IRBuilder {
    pub stmts: Vec<IRStatement>,
    /// Next available memory segment ID for dynamic array allocations.
    next_segment_id: u32,
    /// Next available array ID for dynamic ndarray metadata.
    next_array_id: u32,
    /// Global union-find table over dim variables for dynamic ndarray
    /// envelopes. Lives once per compilation; all envelopes refer to vars
    /// in this single namespace.
    pub dim_table: crate::types::DimTable,
    /// P1 segarr-foundation: side cache mapping a `Value::StaticArray`'s
    /// `segment_id` to the original payload wires written into it. The
    /// `to_value_list` shim looks values up here instead of issuing N
    /// `ir_read_memory` ops per lookup, which keeps the boundary cheap
    /// while ops are still being migrated. Once P6 lands and the legacy
    /// path goes away, this cache (and the shim) go with it.
    pub static_array_payload: HashMap<u32, Vec<Value>>,
    /// P0 SMT-resolver seam: the [`Resolver`] every "must be a compile-time
    /// constant" call site routes through. Default is [`StaticOnlyResolver`]
    /// (delegates to `Value::int_val`/`bool_val`), so today's behaviour is
    /// unchanged. P1 swaps in `SmtResolver`, P2 layers in `RangeResolver`,
    /// without consumer changes. See `src/optim/resolver.rs`.
    resolver: Box<dyn Resolver>,
    /// Fact-propagation framework state (compiler.fact-propagation-framework).
    /// Per-SSA-value Bool facts collected as IR-gen walks the program.
    /// IRGenerator manages enter/leave to mirror the IRContext scope stack;
    /// individual op functions call `facts.insert_for(...)` to fire their
    /// contracts. Consumers (resolver-prove-api card) read `facts.per_stmt`.
    pub facts: crate::optim::predicates::FactStack,
    /// ValueId → StmtId bridge map (compiler.value-id-and-fact-leaves).
    /// Maintained by [`Self::record_value_id`], which is called whenever a
    /// `ScalarValue` is materialised with both a value_id and an SSA ptr.
    /// The witness emitter consults this to lower `ContractVar::Value(_)`
    /// leaves back to IR-layer scalar wires. Nothing else reads it.
    pub value_to_stmt: HashMap<crate::types::ValueId, crate::types::StmtId>,
}

impl Default for IRBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::cloned_ref_to_slice_refs)]
impl IRBuilder {
    pub fn new() -> Self {
        Self {
            stmts: Vec::new(),
            next_segment_id: 0,
            next_array_id: 0,
            dim_table: crate::types::DimTable::new(),
            static_array_payload: HashMap::new(),
            resolver: Box::new(StaticOnlyResolver::new()),
            facts: crate::optim::predicates::FactStack::new(),
            value_to_stmt: HashMap::new(),
        }
    }

    /// Record the `ValueId → StmtId` mapping for a Value, if both halves
    /// are available. Called by every `ir_*` constructor that mints an
    /// SSA stmt for a scalar Value. Composite Values (dyn ndarrays) have
    /// a value_id but no single stmt_id — they're not registered here.
    pub(crate) fn record_value_id(&mut self, val: &Value) {
        if let (Some(vid), Some(sid)) = (val.value_id(), val.stmt_id()) {
            self.value_to_stmt.insert(vid, sid);
        }
    }

    /// Look up the SSA stmt_id for a `ValueId`. Used by the witness
    /// emitter (compiler.value-id-and-fact-leaves) when lowering
    /// `ContractVar::Value(_)` leaves back into IR-layer references.
    pub fn value_id_to_stmt_id(&self, vid: crate::types::ValueId) -> Option<crate::types::StmtId> {
        self.value_to_stmt.get(&vid).copied()
    }

    /// Borrow the active [`Resolver`] (immutable side; rarely used).
    pub fn resolver(&self) -> &dyn Resolver {
        &*self.resolver
    }

    /// Borrow the active [`Resolver`] mutably. The `&mut` receiver is
    /// required because P1's SMT resolver memoises per-ptr query results.
    pub fn resolver_mut(&mut self) -> &mut dyn Resolver {
        &mut *self.resolver
    }

    /// Hand out `&mut dyn Resolver` and `&[IRStatement]` simultaneously
    /// from the same `&mut IRBuilder`. The `_with_stmts` family of trait
    /// methods route through this so the SMT resolver can walk the IR
    /// without the borrow-checker forbidding the joint borrow.
    pub fn split_resolver_and_stmts(
        &mut self,
    ) -> (&mut dyn Resolver, &[IRStatement]) {
        (&mut *self.resolver, &self.stmts)
    }

    /// Swap in a different [`Resolver`] implementation. Not used in P0
    /// (the default is [`StaticOnlyResolver`]); P1's `SmtResolver` and
    /// P2's `RangeResolver` plug in here.
    pub fn set_resolver(&mut self, r: Box<dyn Resolver>) {
        self.resolver = r;
    }

    /// Borrow the resolver's telemetry handle (P5). Returns `None` if the
    /// active resolver doesn't have one (e.g., `StaticOnlyResolver`).
    pub fn resolver_telemetry(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        self.resolver.telemetry_handle()
    }

    /// Try to derive a finite `(min, max)` bound for `val` by combining the
    /// resolver's `resolve_max` / `resolve_min` passes with the op-contract
    /// fact-fallback. Returns `Some((min, max))` only when both halves are
    /// known and `min <= max`.
    ///
    /// This is the ergonomic single-call seam used by chokepoints that
    /// only need a finite range (e.g. `probe_in_range`, range-checked
    /// indices, repeat-count probes). The resolver pass runs first; if
    /// either half is missing, we then scan `self.facts.per_stmt[ptr]`
    /// for ptr-anchored `Cmp(SsaPtr, LitInt)` shapes deposited by op
    /// contracts. Facts can only tighten (or supply) bounds; they never
    /// invent looser ones.
    pub fn ask_bounds(&mut self, val: &Value) -> Option<(i64, i64)> {
        // Try the resolver first. A "useful" resolver answer requires
        // both halves AND a non-trivial range (min < max). The
        // SMT-optimizer path returns (Some(0), Some(0)) for unbounded
        // objectives (Z3's default model), so trusting a degenerate
        // resolver range would clobber genuine fact-derived bounds —
        // we treat min == max as the resolver having "nothing useful"
        // and fall back to facts.
        let (resolver_min, resolver_max) = {
            let (resolver, stmts) = self.split_resolver_and_stmts();
            (
                resolver.resolve_min_with_stmts(val, stmts),
                resolver.resolve_max_with_stmts(val, stmts),
            )
        };
        if let (Some(mn), Some(mx)) = (resolver_min, resolver_max) {
            if mn < mx {
                return Some((mn, mx));
            }
        }
        val.value_id()
            .and_then(|vid| crate::optim::resolver::derive_bounds_from_facts(&self.facts, vid))
    }

    /// Query whether `term` follows from the visible facts + path
    /// conditions on this builder's `FactStack`. Thin wrapper around
    /// [`crate::optim::prove::prove`] so chokepoint consumers and other
    /// in-tree callers don't need to know about the `optim::prove`
    /// module path.
    ///
    /// Soundness: callers MUST treat `ProveOutcome::Unknown` as "no
    /// information" — never as Proved. Disproved means the visible facts
    /// contradict the term and the caller's premise is wrong.
    pub fn prove(
        &self,
        term: &crate::optim::predicates::formula::ContractTerm,
    ) -> crate::optim::prove::ProveOutcome {
        crate::optim::prove::prove(self, term)
    }

    /// Fire an op contract by name: look up the registered contract,
    /// instantiate each `ensures` template against `output_ptr` and the
    /// supplied formal-name → SSA-ptr bindings, and deposit the resulting
    /// facts onto `self.facts.per_stmt[output_ptr]`.
    ///
    /// Collapses the 5-line idiom (registry lookup + per-ensures
    /// instantiate + insert_for) shared by every op-contract producer
    /// into a single call. Producers retain control over the call-site
    /// guard (the `Option<StmtId>` of the output's runtime-length ptr
    /// is a precondition for emitting anything) and the optional
    /// emission of additional call-site-local facts (e.g. `dyn_filter`
    /// adds an upper bound from the static `max_length()`).
    ///
    /// Discharge order is **requires-then-ensures**: an op whose
    /// precondition is `Disproved` panics before its `ensures` facts
    /// are deposited, so downstream chokepoints never observe facts
    /// implying a precondition that never held.
    ///
    /// No-op if the named contract isn't registered — `op_contract_by_name`
    /// already returns an empty default in that case.
    pub fn fire_contract(
        &mut self,
        name: &str,
        output_vid: crate::types::ValueId,
        formals: &HashMap<String, crate::types::ValueId>,
    ) {
        let contract = crate::optim::predicates::op_contract_by_name(name);
        for req in &contract.requires {
            let term = crate::optim::predicates::instantiate_contract(
                &req.term,
                Some(output_vid),
                formals,
            );
            self.discharge_requires(name, &term);
        }
        for ensures in &contract.ensures {
            let fact = crate::optim::predicates::instantiate_contract(
                &ensures.term,
                Some(output_vid),
                formals,
            );
            self.facts.insert_for(output_vid, fact);
        }
    }

    /// Discharge a single, fully-instantiated `requires` term against the
    /// live fact stack. Policy:
    ///
    /// - **Proved**: the precondition follows from the visible facts; no
    ///   action besides telemetry.
    /// - **Disproved**: the visible facts contradict the precondition; the
    ///   user's program is unsound. Panic with a diagnostic naming the op,
    ///   the substituted term, and the value-ids it references.
    /// - **Unknown**: the resolver cannot decide from compile-time context.
    ///   * If `ZINNIA_OP_REQUIRES_STRICT=1` is set, panic with the same
    ///     diagnostic (callers must annotate or restructure to make the
    ///     precondition provable).
    ///   * Otherwise (lenient default), emit a witness-time enforcement
    ///     so the prover must satisfy the precondition at proof time.
    ///     The witness IR is the generic fallback: lower the term to a
    ///     Boolean SSA via [`Self::emit_term_as_bool_value`] and constrain
    ///     it to 1 via `IR::Assert`. If the term cannot be lowered (e.g.,
    ///     it contains a `PredicateApp` that the generic emitter doesn't
    ///     know how to witness), this is a compile error — we panic
    ///     rather than silently drop the precondition. The op author
    ///     must (a) plant a fact at the call site, (b) register a
    ///     per-predicate witness emitter, or (c) annotate `@requires`
    ///     at the function boundary.
    ///
    /// Soundness invariant: `Unknown` never silently becomes `Proved`.
    /// The witness fallback delays the check to prove time but never
    /// skips it; an unemittable term in lenient mode is an error, not
    /// a soft omission.
    pub(crate) fn discharge_requires(
        &mut self,
        op_name: &str,
        term: &crate::optim::predicates::formula::ContractTerm,
    ) {
        use crate::optim::prove::ProveOutcome;

        let outcome = self.prove(term);
        match outcome {
            ProveOutcome::Proved => {
                tracing::debug!(
                    target: "zinnia::op_contract",
                    "op_requires_discharge: op={} outcome=Proved",
                    op_name,
                );
            }
            ProveOutcome::Disproved => {
                let value_ids =
                    crate::optim::predicates::collect_value_ids(term);
                tracing::debug!(
                    target: "zinnia::op_contract",
                    "op_requires_discharge: op={} outcome=Disproved (compile error)",
                    op_name,
                );
                panic!(
                    "op `{op_name}` requires precondition disproved by visible facts: \
                     term={term:?}, value_ids={value_ids:?}"
                );
            }
            ProveOutcome::Unknown => {
                if op_requires_strict() {
                    let value_ids =
                        crate::optim::predicates::collect_value_ids(term);
                    tracing::debug!(
                        target: "zinnia::op_contract",
                        "op_requires_discharge: op={} outcome=Unknown mode=strict (compile error)",
                        op_name,
                    );
                    panic!(
                        "op `{op_name}` requires precondition not provable from visible facts \
                         (strict mode): term={term:?}, value_ids={value_ids:?}"
                    );
                }
                let emitted = self.emit_requires_witness_check(term);
                if !emitted {
                    let value_ids =
                        crate::optim::predicates::collect_value_ids(term);
                    tracing::debug!(
                        target: "zinnia::op_contract",
                        "op_requires_discharge: op={} outcome=Unknown mode=lenient witness_emit=no (compile error)",
                        op_name,
                    );
                    panic!(
                        "op `{op_name}` requires precondition `{term:?}` could not be \
                         lowered to a witness constraint (term likely contains a \
                         `PredicateApp` without a registered witness emitter). \
                         Either (a) plant a fact at the call site making the precondition \
                         provable, (b) register a per-predicate witness emitter, or \
                         (c) annotate `@requires` at the function boundary. \
                         value_ids={value_ids:?}"
                    );
                }
                tracing::debug!(
                    target: "zinnia::op_contract",
                    "op_requires_discharge: op={} outcome=Unknown mode=lenient witness_emit=yes",
                    op_name,
                );
            }
        }
    }

    /// Generic witness fallback for a `Unknown`-discharged requires term.
    /// Lowers the (fully-instantiated) term to a Boolean SSA wire and
    /// emits `IR::Assert` against it, so the prover must satisfy the
    /// precondition at proof time. Returns `true` if a constraint was
    /// emitted, `false` if the term could not be lowered (sound omission
    /// — the precondition then has no proof-time enforcement, mirroring
    /// the user `@requires` flow's failure handling).
    pub(crate) fn emit_requires_witness_check(
        &mut self,
        term: &crate::optim::predicates::formula::ContractTerm,
    ) -> bool {
        match self.emit_term_as_bool_value(term) {
            Some(bool_val) => {
                self.ir_assert(&bool_val);
                true
            }
            None => false,
        }
    }

    /// Lower a fully-instantiated `ContractTerm` (top-level Bool, leaves
    /// of `Var(Value(_))` / `LitInt` / `LitFloat` / `LitBool`) to a
    /// Boolean `Value`. Returns `None` if the term cannot be lowered
    /// (e.g., references a `Var(Value(_))` whose `value_id` has no
    /// recorded `stmt_id`, or contains an unsupported `PredicateApp`).
    fn emit_term_as_bool_value(
        &mut self,
        term: &crate::optim::predicates::formula::ContractTerm,
    ) -> Option<Value> {
        use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm};
        match term {
            ContractTerm::LitBool(b) => Some(self.ir_constant_bool(*b)),
            ContractTerm::Cmp { op, lhs, rhs } => {
                let l = self.emit_term_as_num_value(lhs)?;
                let r = self.emit_term_as_num_value(rhs)?;
                let any_float =
                    matches!(l, Value::Float(_)) || matches!(r, Value::Float(_));
                let ir = if any_float {
                    match op {
                        CmpOp::Eq => IR::EqF,
                        CmpOp::Ne => IR::NeF,
                        CmpOp::Lt => IR::LtF,
                        CmpOp::Le => IR::LteF,
                        CmpOp::Gt => IR::GtF,
                        CmpOp::Ge => IR::GteF,
                    }
                } else {
                    match op {
                        CmpOp::Eq => IR::EqI,
                        CmpOp::Ne => IR::NeI,
                        CmpOp::Lt => IR::LtI,
                        CmpOp::Le => IR::LteI,
                        CmpOp::Gt => IR::GtI,
                        CmpOp::Ge => IR::GteI,
                    }
                };
                Some(self.create_ir(&ir, &[l, r]))
            }
            ContractTerm::BoolComb { op, operands } => {
                if operands.is_empty() {
                    return Some(self.ir_constant_bool(matches!(op, BoolOp::And)));
                }
                let mut acc = self.emit_term_as_bool_value(&operands[0])?;
                for next in &operands[1..] {
                    let next_v = self.emit_term_as_bool_value(next)?;
                    let ir = match op {
                        BoolOp::And => IR::LogicalAnd,
                        BoolOp::Or => IR::LogicalOr,
                    };
                    acc = self.create_ir(&ir, &[acc, next_v]);
                }
                Some(acc)
            }
            ContractTerm::Not(inner) => {
                let v = self.emit_term_as_bool_value(inner)?;
                Some(self.create_ir(&IR::LogicalNot, &[v]))
            }
            // PredicateApp can't be lowered to a Bool SSA without a
            // per-predicate emitter; per-op cards may register specialised
            // emitters. The generic fallback soundly declines to emit
            // (returns None → no constraint, no false positive).
            ContractTerm::PredicateApp { .. } => None,
            ContractTerm::Var(_)
            | ContractTerm::LitInt(_)
            | ContractTerm::LitFloat(_)
            | ContractTerm::Arith { .. } => None,
        }
    }

    /// Lower a numeric `ContractTerm` (Int- or Float-typed) to a `Value`.
    /// `Var(Value(vid))` is reconstituted from the recorded `value_id →
    /// stmt_id` map; the resulting wire's scalar sort is inferred from
    /// the producing IR's class name.
    fn emit_term_as_num_value(
        &mut self,
        term: &crate::optim::predicates::formula::ContractTerm,
    ) -> Option<Value> {
        use crate::optim::predicates::formula::{ArithOp, ContractTerm, ContractVar};
        match term {
            ContractTerm::LitInt(n) => Some(self.ir_constant_int(*n)),
            ContractTerm::LitFloat(f) => Some(self.ir_constant_float(f.0)),
            ContractTerm::Var(ContractVar::Value(vid)) => {
                let sid = self.value_id_to_stmt_id(*vid)?;
                Some(self.value_from_stmt_id(sid))
            }
            // Unsubstituted formals / outputs / inputs at this layer are
            // template-shape bugs; the contract should have been
            // instantiated before reaching the discharge layer.
            ContractTerm::Var(_) => None,
            ContractTerm::Arith { op, lhs, rhs } => {
                let l = self.emit_term_as_num_value(lhs)?;
                let r = self.emit_term_as_num_value(rhs)?;
                let any_float =
                    matches!(l, Value::Float(_)) || matches!(r, Value::Float(_));
                let ir = if any_float {
                    match op {
                        ArithOp::Add => IR::AddF,
                        ArithOp::Sub => IR::SubF,
                        ArithOp::Mul => IR::MulF,
                        ArithOp::Div => IR::DivF,
                        ArithOp::FloorDiv => IR::FloorDivF,
                        ArithOp::Mod => IR::ModF,
                        ArithOp::Pow => IR::PowF,
                    }
                } else {
                    match op {
                        ArithOp::Add => IR::AddI,
                        ArithOp::Sub => IR::SubI,
                        ArithOp::Mul => IR::MulI,
                        ArithOp::Div => IR::DivI,
                        ArithOp::FloorDiv => IR::FloorDivI,
                        ArithOp::Mod => IR::ModI,
                        ArithOp::Pow => IR::PowI,
                    }
                };
                Some(self.create_ir(&ir, &[l, r]))
            }
            ContractTerm::LitBool(_)
            | ContractTerm::Cmp { .. }
            | ContractTerm::BoolComb { .. }
            | ContractTerm::Not(_)
            | ContractTerm::PredicateApp { .. } => None,
        }
    }

    /// Reconstruct a `Value` of the appropriate scalar sort (Int / Float
    /// / Bool) from an existing `StmtId`. The sort is inferred from the
    /// producing IR. Unknown / composite producers default to `Integer`
    /// (sound for the witness-fallback path: the worst case is using an
    /// `*I` op on a wire that would have benefited from an `*F` form,
    /// which still produces a deterministic boolean answer).
    fn value_from_stmt_id(&self, sid: StmtId) -> Value {
        let stmt = match self.stmts.get(sid as usize) {
            Some(s) => s,
            None => return Value::Integer(ScalarValue::new(None, Some(sid))),
        };
        match &stmt.ir {
            IR::ConstantFloat { .. }
            | IR::AddF
            | IR::SubF
            | IR::MulF
            | IR::DivF
            | IR::FloorDivF
            | IR::ModF
            | IR::PowF
            | IR::AbsF
            | IR::SignF
            | IR::SinF
            | IR::SinHF
            | IR::CosF
            | IR::CosHF
            | IR::TanF
            | IR::TanHF
            | IR::SqrtF
            | IR::ExpF
            | IR::LogF
            | IR::ArcCosF
            | IR::ArcTan2F
            | IR::FloatCast
            | IR::ReadFloat { .. }
            | IR::SelectF => Value::Float(ScalarValue::new(None, Some(sid))),
            IR::ConstantBool { .. }
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::LogicalNot
            | IR::BoolCast
            | IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LteI
            | IR::GtI
            | IR::GteI
            | IR::EqF
            | IR::NeF
            | IR::LtF
            | IR::LteF
            | IR::GtF
            | IR::GteF
            | IR::SelectB
            | IR::EqHash => Value::Boolean(ScalarValue::new(None, Some(sid))),
            _ => Value::Integer(ScalarValue::new(None, Some(sid))),
        }
    }

    /// Allocate a unique memory segment ID.
    pub fn alloc_segment_id(&mut self) -> u32 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Allocate a unique array metadata ID.
    pub fn alloc_array_id(&mut self) -> u32 {
        let id = self.next_array_id;
        self.next_array_id += 1;
        id
    }

    /// Ensure a value has an IR pointer. If it's a pure compile-time constant
    /// with no pointer, materialize it as an IR constant instruction.
    pub fn ensure_ptr(&mut self, val: &Value) -> Value {
        if val.stmt_id().is_some() {
            return val.clone();
        }
        match val {
            Value::Integer(s) => {
                if let Some(v) = s.static_val {
                    self.ir_constant_int(v)
                } else {
                    val.clone()
                }
            }
            Value::Float(s) => {
                if let Some(v) = s.static_val {
                    self.ir_constant_float(v)
                } else {
                    val.clone()
                }
            }
            Value::Boolean(s) => {
                if let Some(v) = s.static_val {
                    self.ir_constant_bool(v)
                } else {
                    val.clone()
                }
            }
            Value::None | Value::Class(_) => {
                // Materialize None/Class as constant 0 to avoid panics downstream
                self.ir_constant_int(0)
            }
            // For composites, extract first scalar element as fallback
            Value::List(data) | Value::Tuple(data) if !data.values.is_empty() => {
                self.ensure_ptr(&data.values[0])
            }
            _ => val.clone(),
        }
    }

    /// The core method: build an IR instruction, append the statement,
    /// and return the result `Value`.  Mirrors Python
    /// `IRBuilderImpl.create_ir(operator, args, dbg)`.
    pub fn create_ir(&mut self, ir: &IR, args: &[Value]) -> Value {
        // Materialize any pure constants so they have IR pointers
        let materialized: Vec<Value> = args.iter().map(|v| self.ensure_ptr(v)).collect();
        let ir_id = self.stmts.len() as StmtId;
        let val = build_value(ir, ir_id, &materialized);
        // Phase 3 dual-view arguments (compiler.value-id-and-fact-leaves):
        // `arguments` keeps the existing stmt_id space so walkers keep
        // their direct-indexed `stmts[arg as usize]` flow. `arg_values`
        // is the parallel ValueId view consumed by fact-aware code.
        let arguments: Vec<StmtId> = materialized
            .iter()
            .filter_map(|a| a.stmt_id())
            .collect();
        let arg_values: Vec<crate::types::ValueId> = materialized
            .iter()
            .filter_map(|a| a.value_id())
            .collect();
        let value_id = val
            .value_id()
            .unwrap_or_else(crate::types::ValueId::next);
        let stmt = IRStatement::new(ir_id, value_id, ir.clone(), arguments, arg_values, None);
        self.stmts.push(stmt);
        self.record_value_id(&val);
        val
    }

    pub fn export_ir_graph(self) -> IRGraph {
        IRGraph::new(self.stmts)
    }

    // ── Convenience helpers used by optimization passes ──────────────

    pub fn ir_constant_int(&mut self, value: i64) -> Value {
        self.create_ir(&IR::ConstantInt { value }, &[])
    }

    pub fn ir_constant_float(&mut self, value: f64) -> Value {
        self.create_ir(&IR::ConstantFloat { value }, &[])
    }

    pub fn ir_constant_bool(&mut self, value: bool) -> Value {
        self.create_ir(&IR::ConstantBool { value }, &[])
    }

    pub fn ir_constant_str(&mut self, value: String) -> Value {
        self.create_ir(&IR::ConstantStr { value }, &[])
    }

    // ── Macro-generated convenience methods ──────────────────────────

    // Logic
    ir_binary!(
        ir_logical_and => IR::LogicalAnd, contract = "logical_and",
        ir_logical_or  => IR::LogicalOr,  contract = "logical_or",
    );
    ir_unary!(ir_logical_not => IR::LogicalNot, contract = "logical_not");

    // Integer arithmetic
    //
    // `ir_add_i`, `ir_sub_i`, `ir_mul_i` are hand-rolled out of the macro
    // because they deposit a call-site-fact (input-fact-driven interval
    // bound on the output) rather than a fixed registered template.
    /// Interval bounds via input facts (call-site-fact, no SMT).
    pub fn ir_add_i(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::AddI, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            if let Some(fact) = crate::optim::resolver::interval_fact_for_int_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Add,
                a_vid,
                b_vid,
                out_vid,
            ) {
                self.facts.insert_for(out_vid, fact);
            }
        }
        out
    }

    /// Interval bounds via input facts (call-site-fact, no SMT).
    pub fn ir_sub_i(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::SubI, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            if let Some(fact) = crate::optim::resolver::interval_fact_for_int_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Sub,
                a_vid,
                b_vid,
                out_vid,
            ) {
                self.facts.insert_for(out_vid, fact);
            }
        }
        out
    }

    /// Interval bounds via input facts (call-site-fact, no SMT).
    pub fn ir_mul_i(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::MulI, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            if let Some(fact) = crate::optim::resolver::interval_fact_for_int_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Mul,
                a_vid,
                b_vid,
                out_vid,
            ) {
                self.facts.insert_for(out_vid, fact);
            }
        }
        out
    }

    ir_binary!(
        ir_div_i       => IR::DivI,       contract = "div_i",
        ir_floor_div_i => IR::FloorDivI,  contract = "floor_div_i",
        ir_mod_i       => IR::ModI,       contract = "mod_i",
        ir_pow_i       => IR::PowI,       contract = "pow_i",
    );
    ir_unary!(
        ir_inv_i  => IR::InvI, contract = "inv_i",
    );

    /// Integer `sign`. Fires the `sign_i` op contract so the output
    /// value carries the `-1 <= Output <= 1` fact for downstream
    /// resolvers.
    pub fn ir_sign_i(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::SignI, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("sign_i", vid, &HashMap::new());
        }
        out
    }

    /// Integer `abs`. Fires the `abs_i` op contract so the output value
    /// carries the `Output >= 0` fact for downstream resolvers.
    pub fn ir_abs_i(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::AbsI, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("abs_i", vid, &HashMap::new());
        }
        out
    }

    // Integer bitwise
    ir_binary!(
        ir_shl_i => IR::ShlI,
        ir_shr_i => IR::ShrI,
    );

    /// Integer bitwise AND. Fires `bit_and_i` contract (Output ∈ {0,1})
    /// only when both inputs are `Value::Boolean`-typed; the IR op itself
    /// operates on general integers and would not be sound to constrain
    /// unconditionally.
    pub fn ir_bit_and_i(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::BitAndI, &[a.clone(), b.clone()]);
        if matches!(a, Value::Boolean(_)) && matches!(b, Value::Boolean(_)) {
            if let Some(vid) = out.value_id() {
                self.fire_contract("bit_and_i", vid, &HashMap::new());
            }
        }
        out
    }

    /// Integer bitwise OR. Fires `bit_or_i` contract (Output ∈ {0,1}) only
    /// when both inputs are `Value::Boolean`-typed; the IR op itself
    /// operates on general integers and would not be sound to constrain
    /// unconditionally.
    pub fn ir_bit_or_i(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::BitOrI, &[a.clone(), b.clone()]);
        if matches!(a, Value::Boolean(_)) && matches!(b, Value::Boolean(_)) {
            if let Some(vid) = out.value_id() {
                self.fire_contract("bit_or_i", vid, &HashMap::new());
            }
        }
        out
    }

    /// Integer bitwise XOR. Fires `bit_xor_i` contract (Output ∈ {0,1})
    /// only when both inputs are `Value::Boolean`-typed; the IR op itself
    /// operates on general integers and would not be sound to constrain
    /// unconditionally.
    pub fn ir_bit_xor_i(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::BitXorI, &[a.clone(), b.clone()]);
        if matches!(a, Value::Boolean(_)) && matches!(b, Value::Boolean(_)) {
            if let Some(vid) = out.value_id() {
                self.fire_contract("bit_xor_i", vid, &HashMap::new());
            }
        }
        out
    }

    /// Integer bitwise NOT. Fires `bit_not_i` contract (Output ∈ {0,1})
    /// only when the input is `Value::Boolean`-typed; the IR op itself
    /// operates on general integers and would not be sound to constrain
    /// unconditionally.
    pub fn ir_bit_not_i(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::BitNotI, &[a.clone()]);
        if matches!(a, Value::Boolean(_)) {
            if let Some(vid) = out.value_id() {
                self.fire_contract("bit_not_i", vid, &HashMap::new());
            }
        }
        out
    }

    // Float arithmetic
    //
    // `ir_add_f`, `ir_sub_f`, `ir_mul_f` are hand-rolled out of the
    // macro because they deposit a call-site-fact (input-fact-driven
    // interval bound on the output) rather than a fixed registered
    // template. Mirrors the int-arith wiring at `ir_add_i / sub_i / mul_i`.
    /// Interval bounds via input facts (call-site-fact, no SMT).
    pub fn ir_add_f(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::AddF, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            if let Some(fact) = crate::optim::resolver::interval_fact_for_float_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Add,
                a_vid,
                b_vid,
                out_vid,
            ) {
                self.facts.insert_for(out_vid, fact);
            }
        }
        out
    }

    /// Interval bounds via input facts (call-site-fact, no SMT).
    pub fn ir_sub_f(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::SubF, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            if let Some(fact) = crate::optim::resolver::interval_fact_for_float_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Sub,
                a_vid,
                b_vid,
                out_vid,
            ) {
                self.facts.insert_for(out_vid, fact);
            }
        }
        out
    }

    /// Interval bounds via input facts (call-site-fact, no SMT).
    pub fn ir_mul_f(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::MulF, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            if let Some(fact) = crate::optim::resolver::interval_fact_for_float_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Mul,
                a_vid,
                b_vid,
                out_vid,
            ) {
                self.facts.insert_for(out_vid, fact);
            }
        }
        out
    }

    ir_binary!(
        ir_div_f       => IR::DivF,       contract = "div_f",
        ir_floor_div_f => IR::FloorDivF,  contract = "floor_div_f",
        ir_mod_f       => IR::ModF,       contract = "mod_f",
        ir_pow_f       => IR::PowF,       contract = "pow_f",
    );
    /// Float `sign`. Fires the `sign_f` op contract so the output value
    /// carries the `-1.0 <= Output <= 1.0` fact for downstream
    /// resolvers.
    pub fn ir_sign_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::SignF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("sign_f", vid, &HashMap::new());
        }
        out
    }

    /// Float `abs`. Fires the `abs_f` op contract so the output value
    /// carries the `Output >= 0.0` fact for downstream resolvers.
    pub fn ir_abs_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::AbsF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("abs_f", vid, &HashMap::new());
        }
        out
    }

    // Comparisons
    ir_binary!(
        ir_equal_i                  => IR::EqI,    contract = "eq_i",
        ir_equal_f                  => IR::EqF,    contract = "eq_f",
        ir_not_equal_i              => IR::NeI,    contract = "ne_i",
        ir_not_equal_f              => IR::NeF,    contract = "ne_f",
        ir_less_than_i              => IR::LtI,    contract = "lt_i",
        ir_less_than_f              => IR::LtF,    contract = "lt_f",
        ir_less_than_or_equal_i     => IR::LteI,   contract = "lte_i",
        ir_less_than_or_equal_f     => IR::LteF,   contract = "lte_f",
        ir_greater_than_i           => IR::GtI,    contract = "gt_i",
        ir_greater_than_f           => IR::GtF,    contract = "gt_f",
        ir_greater_than_or_equal_i  => IR::GteI,   contract = "gte_i",
        ir_greater_than_or_equal_f  => IR::GteF,   contract = "gte_f",
        ir_equal_hash               => IR::EqHash,
    );

    // Selection (ternary: cond, true_val, false_val)
    ir_ternary!(
        ir_select_i => IR::SelectI,
        ir_select_f => IR::SelectF,
        ir_select_b => IR::SelectB,
    );

    // Casting
    ir_unary!(
        ir_bool_cast  => IR::BoolCast, contract = "bool_cast",
    );

    /// Cast to integer. Fires `int_cast_from_bool` contract (Output ∈ {0,1})
    /// only when the input is `Value::Boolean`-typed; the IR op accepts any
    /// scalar and would not be sound to constrain unconditionally.
    pub fn ir_int_cast(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::IntCast, &[a.clone()]);
        if matches!(a, Value::Boolean(_)) {
            if let Some(vid) = out.value_id() {
                self.fire_contract("int_cast_from_bool", vid, &HashMap::new());
            }
        }
        out
    }

    /// Cast to float. Fires `float_cast_from_bool` contract (Output ∈ {0.0, 1.0})
    /// only when the input is `Value::Boolean`-typed; same conditional rationale
    /// as `ir_int_cast`.
    pub fn ir_float_cast(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::FloatCast, &[a.clone()]);
        if matches!(a, Value::Boolean(_)) {
            if let Some(vid) = out.value_id() {
                self.fire_contract("float_cast_from_bool", vid, &HashMap::new());
            }
        }
        out
    }

    // String operations
    ir_binary!(ir_add_str => IR::AddStr, ir_print => IR::Print);
    ir_unary!(ir_str_i => IR::StrI, ir_str_f => IR::StrF);

    // Math functions
    ir_unary!(
        ir_tan_f  => IR::TanF,
    );

    /// Float `sin`. Fires the `sin_f` op contract so the output value
    /// carries the `-1.0 <= Output <= 1.0` fact for downstream
    /// resolvers.
    pub fn ir_sin_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::SinF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("sin_f", vid, &HashMap::new());
        }
        out
    }

    /// Float `cos`. Fires the `cos_f` op contract so the output value
    /// carries the `-1.0 <= Output <= 1.0` fact for downstream
    /// resolvers.
    pub fn ir_cos_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::CosF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("cos_f", vid, &HashMap::new());
        }
        out
    }

    ir_unary!(
        ir_sinh_f => IR::SinHF,
    );

    /// Float `arccos`. Fires the `arccos_f` op contract so the input
    /// value's `requires(-1.0 <= x <= 1.0)` precondition is discharged
    /// against its fact bucket (formal `"x"` ↔ input ValueId), then relays
    /// any visible input interval `[lo, hi] ⊆ [-1, 1]` into
    /// `[acos(hi), acos(lo)]` on the output (note: arccos is
    /// monotone-decreasing, so the bounds swap).
    pub fn ir_arccos_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::ArcCosF, &[a.clone()]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            let mut formals = HashMap::new();
            formals.insert("x".to_string(), x_vid);
            self.fire_contract("arccos_f", vid, &formals);
            crate::optim::resolver::relay_arccos_output_interval(self, x_vid, vid);
        }
        out
    }

    /// Float `log`. Fires the `log_f` op contract so the input value's
    /// `requires(x > 0.0)` precondition is discharged against its fact
    /// bucket (formal `"x"` ↔ input ValueId), then relays any visible
    /// input interval `[lo, hi]` (with `lo > 0`) into
    /// `[log(lo), log(hi)]` on the output.
    pub fn ir_log_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::LogF, &[a.clone()]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            let mut formals = HashMap::new();
            formals.insert("x".to_string(), x_vid);
            self.fire_contract("log_f", vid, &formals);
            crate::optim::resolver::relay_log_output_interval(self, x_vid, vid);
        }
        out
    }

    /// Float `tanh`. Fires the `tanh_f` op contract so the output value
    /// carries the `-1.0 <= Output <= 1.0` fact (sound looser bound;
    /// strict codomain is the open interval) for downstream resolvers.
    pub fn ir_tanh_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::TanHF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("tanh_f", vid, &HashMap::new());
        }
        out
    }

    /// Float `cosh`. Fires the `cosh_f` op contract so the output value
    /// carries the `Output >= 1.0` fact for downstream resolvers.
    pub fn ir_cosh_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::CosHF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("cosh_f", vid, &HashMap::new());
        }
        out
    }

    /// Float `sqrt`. Fires the `sqrt_f` op contract so the output value
    /// carries the `Output >= 0.0` fact for downstream resolvers, and so
    /// the `requires(x >= 0.0)` precondition is discharged against the
    /// input value's fact bucket (formal `"x"` ↔ input ValueId).
    pub fn ir_sqrt_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::SqrtF, &[a.clone()]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            let mut formals = HashMap::new();
            formals.insert("x".to_string(), x_vid);
            self.fire_contract("sqrt_f", vid, &formals);
            crate::optim::resolver::relay_sqrt_output_interval(self, x_vid, vid);
        }
        out
    }

    /// Float `exp`. Fires the `exp_f` op contract so the output value
    /// carries the `Output >= 0.0` fact for downstream resolvers, then
    /// relays any visible input interval `[lo, hi]` into
    /// `[exp(lo), exp(hi)]` on the output.
    pub fn ir_exp_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::ExpF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.fire_contract("exp_f", vid, &HashMap::new());
            if let Some(x_vid) = a.value_id() {
                crate::optim::resolver::relay_exp_output_interval(self, x_vid, vid);
            }
        }
        out
    }

    ir_binary!(
        ir_arctan2_f => IR::ArcTan2F, contract = "arctan2_f",
    );

    // Assert & expose
    ir_unary!(
        ir_assert          => IR::Assert,
        ir_expose_public_i => IR::ExposePublicI,
        ir_expose_public_f => IR::ExposePublicF,
    );

    /// Emit an `IR::StructuralPredicate` precondition atom. Takes no stack
    /// operands; the payload (kind / args / op / bound) lives in the IR
    /// instance itself. Returns `Value::None`.
    pub fn ir_structural_predicate(
        &mut self,
        kind: String,
        args: Vec<String>,
        op: Option<String>,
        bound: Option<String>,
    ) -> Value {
        self.create_ir(&IR::StructuralPredicate { kind, args, op, bound }, &[])
    }

    /// Emit an `IR::ScalarPrecondition` atom carrying a serialized
    /// `ContractTerm` payload. Takes no stack operands. Returns
    /// `Value::None`.
    pub fn ir_scalar_precondition(&mut self, term_json: String) -> Value {
        self.create_ir(&IR::ScalarPrecondition { term_json }, &[])
    }

    // ── I/O ───────────────────────────────────────────────────────────

    pub fn ir_read_integer(&mut self, path: crate::circuit_input::InputPath, is_public: bool) -> Value {
        self.create_ir(&IR::ReadInteger { path, is_public }, &[])
    }

    pub fn ir_read_float(&mut self, path: crate::circuit_input::InputPath, is_public: bool) -> Value {
        self.create_ir(&IR::ReadFloat { path, is_public }, &[])
    }

    pub fn ir_read_hash(&mut self, path: crate::circuit_input::InputPath, is_public: bool) -> Value {
        self.create_ir(&IR::ReadHash { path, is_public }, &[])
    }

    pub fn ir_read_external_result(&mut self, store_idx: u32, output_idx: u32, is_float: bool) -> Value {
        self.create_ir(&IR::ReadExternalResult { store_idx, output_idx, is_float }, &[])
    }

    // ── Memory ────────────────────────────────────────────────────────

    pub fn ir_allocate_memory(&mut self, segment_id: u32, size: u32, init_value: i64) -> Value {
        self.create_ir(
            &IR::AllocateMemory { segment_id, size, init_value },
            &[],
        )
    }

    pub fn ir_write_memory(&mut self, segment_id: u32, address: &Value, value: &Value) -> Value {
        self.create_ir(
            &IR::WriteMemory { segment_id },
            &[address.clone(), value.clone()],
        )
    }

    pub fn ir_read_memory(&mut self, segment_id: u32, address: &Value) -> Value {
        self.create_ir(&IR::ReadMemory { segment_id }, &[address.clone()])
    }

    // ── Dynamic NDArray ───────────────────────────────────────────────

    pub fn ir_allocate_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        dtype_name: String,
        max_length: u32,
        max_rank: u32,
    ) -> Value {
        self.create_ir(
            &IR::AllocateDynamicNDArrayMeta { array_id, dtype_name, max_length, max_rank },
            &[],
        )
    }

    pub fn ir_witness_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        args: &[Value],
    ) -> Value {
        self.create_ir(
            &IR::WitnessDynamicNDArrayMeta { array_id, max_rank },
            args,
        )
    }

    pub fn ir_dynamic_ndarray_get_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        address: &Value,
    ) -> Value {
        self.create_ir(
            &IR::DynamicNDArrayGetItem { array_id, segment_id },
            &[address.clone()],
        )
    }

    pub fn ir_dynamic_ndarray_set_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        address: &Value,
        value: &Value,
    ) -> Value {
        self.create_ir(
            &IR::DynamicNDArraySetItem { array_id, segment_id },
            &[address.clone(), value.clone()],
        )
    }

    // ── External calls ────────────────────────────────────────────────

    pub fn ir_invoke_external(
        &mut self,
        store_idx: u32,
        func_name: String,
        args: Vec<serde_json::Value>,
        kwargs: std::collections::HashMap<String, serde_json::Value>,
    ) -> Value {
        self.create_ir(
            &IR::InvokeExternal { store_idx, func_name, args, kwargs },
            &[],
        )
    }

    pub fn ir_poseidon_hash(&mut self, values: &[Value]) -> Value {
        self.create_ir(&IR::PoseidonHash, values)
    }
}

// ---------------------------------------------------------------------------
// build_ir — the Rust equivalent of AbstractIR.build_ir()
// ---------------------------------------------------------------------------

/// Build an IR statement and compute the result value.
/// This implements the `build_ir(ir_id, args, dbg)` method for all 79 IR types.
fn build_value(ir: &IR, ir_id: StmtId, args: &[Value]) -> Value {
    match ir {
        // ── Constants ─────────────────────────────────────────────
        IR::ConstantInt { value } => Value::Integer(ScalarValue::known(*value, ir_id)),
        IR::ConstantFloat { value } => Value::Float(ScalarValue::known(*value, ir_id)),
        IR::ConstantBool { value } => Value::Boolean(ScalarValue::known(*value, ir_id)),
        IR::ConstantStr { value } => Value::String(StringValue {
                val: value.clone(),
                stmt_id: ir_id,
            }),

        // ── Integer binary arithmetic ─────────────────────────────
        IR::AddI => int_binary_ir(ir_id, args, |a, b| a.checked_add(b)),
        IR::SubI => int_binary_ir(ir_id, args, |a, b| a.checked_sub(b)),
        IR::MulI => int_binary_ir(ir_id, args, |a, b| a.checked_mul(b)),
        IR::DivI => int_binary_ir(ir_id, args, |a, b| {
            if b != 0 { Some(a / b) } else { None }
        }),
        IR::FloorDivI => int_binary_ir(ir_id, args, |a, b| {
            if b != 0 {
                Some(a.div_euclid(b))
            } else {
                None
            }
        }),
        IR::ModI => int_binary_ir(ir_id, args, |a, b| {
            if b != 0 { Some(a % b) } else { None }
        }),
        IR::PowI => int_binary_ir(ir_id, args, |a, b| {
            if b >= 0 {
                Some(a.pow(b as u32))
            } else {
                None
            }
        }),

        // ── Integer bitwise ───────────────────────────────────────
        IR::BitAndI => int_binary_ir(ir_id, args, |a, b| Some(a & b)),
        IR::BitOrI  => int_binary_ir(ir_id, args, |a, b| Some(a | b)),
        IR::BitXorI => int_binary_ir(ir_id, args, |a, b| Some(a ^ b)),
        IR::ShlI    => int_binary_ir(ir_id, args, |a, b| {
            let shift = b.max(0).min(63) as u32;
            Some(a.wrapping_shl(shift))
        }),
        IR::ShrI    => int_binary_ir(ir_id, args, |a, b| {
            let shift = b.max(0).min(63) as u32;
            Some(a.wrapping_shr(shift))
        }),
        IR::BitNotI => int_unary_ir(ir_id, args, |a| Some(!a)),

        // ── Integer unary arithmetic ──────────────────────────────
        IR::AbsI => int_unary_ir(ir_id, args, |a| Some(a.abs())),
        IR::SignI => int_unary_ir(ir_id, args, |a| {
            Some(if a > 0 { 1 } else if a < 0 { -1 } else { 0 })
        }),
        IR::InvI => Value::Integer(ScalarValue::new(None, Some(ir_id))),

        // ── Float binary arithmetic ───────────────────────────────
        IR::AddF => float_binary_ir(ir_id, args, |a, b| a + b),
        IR::SubF => float_binary_ir(ir_id, args, |a, b| a - b),
        IR::MulF => float_binary_ir(ir_id, args, |a, b| a * b),
        IR::DivF => float_binary_ir(ir_id, args, |a, b| a / b),
        IR::FloorDivF => float_binary_ir(ir_id, args, |a, b| (a / b).floor()),
        IR::ModF => float_binary_ir(ir_id, args, |a, b| a % b),
        IR::PowF => float_binary_ir(ir_id, args, |a, b| a.powf(b)),

        // ── Float unary arithmetic ────────────────────────────────
        IR::AbsF => float_unary_ir(ir_id, args, |a| a.abs()),
        IR::SignF => float_unary_ir(ir_id, args, |a| {
            if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),

        // ── Integer comparisons ───────────────────────────────────
        IR::EqI => int_cmp_ir(ir_id, args, |a, b| a == b),
        IR::NeI => int_cmp_ir(ir_id, args, |a, b| a != b),
        IR::LtI => int_cmp_ir(ir_id, args, |a, b| a < b),
        IR::LteI => int_cmp_ir(ir_id, args, |a, b| a <= b),
        IR::GtI => int_cmp_ir(ir_id, args, |a, b| a > b),
        IR::GteI => int_cmp_ir(ir_id, args, |a, b| a >= b),

        // ── Float comparisons ─────────────────────────────────────
        IR::EqF => float_cmp_ir(ir_id, args, |a, b| a == b),
        IR::NeF => float_cmp_ir(ir_id, args, |a, b| a != b),
        IR::LtF => float_cmp_ir(ir_id, args, |a, b| a < b),
        IR::LteF => float_cmp_ir(ir_id, args, |a, b| a <= b),
        IR::GtF => float_cmp_ir(ir_id, args, |a, b| a > b),
        IR::GteF => float_cmp_ir(ir_id, args, |a, b| a >= b),

        // ── Math functions (float) ────────────────────────────────
        IR::SinF => float_unary_ir(ir_id, args, |a| a.sin()),
        IR::SinHF => float_unary_ir(ir_id, args, |a| a.sinh()),
        IR::CosF => float_unary_ir(ir_id, args, |a| a.cos()),
        IR::CosHF => float_unary_ir(ir_id, args, |a| a.cosh()),
        IR::TanF => float_unary_ir(ir_id, args, |a| a.tan()),
        IR::TanHF => float_unary_ir(ir_id, args, |a| a.tanh()),
        IR::SqrtF => float_unary_ir(ir_id, args, |a| a.sqrt()),
        IR::ExpF => float_unary_ir(ir_id, args, |a| a.exp()),
        IR::LogF => float_unary_ir(ir_id, args, |a| a.ln()),
        IR::ArcCosF => float_unary_ir(ir_id, args, |a| a.acos()),
        IR::ArcTan2F => float_binary_ir(ir_id, args, |a, b| a.atan2(b)),

        // ── Logical ───────────────────────────────────────────────
        IR::LogicalAnd => {
            let la = args[0].int_val();
            let lb = args[1].int_val();
            let inferred = match (la, lb) {
                (Some(a), Some(b)) => Some(a != 0 && b != 0),
                _ => None,
            };
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::LogicalOr => {
            let la = args[0].int_val();
            let lb = args[1].int_val();
            let inferred = match (la, lb) {
                (Some(a), Some(b)) => Some(a != 0 || b != 0),
                _ => None,
            };
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::LogicalNot => {
            let la = args[0].int_val();
            let inferred = la.map(|a| a == 0);
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }

        // ── Selection ─────────────────────────────────────────────
        IR::SelectI => {
            let cond = args[0].bool_val().or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].int_val(),
                Some(false) => args[2].int_val(),
            };
            Value::Integer(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::SelectF => {
            let cond = args[0].bool_val().or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].float_val(),
                Some(false) => args[2].float_val(),
            };
            Value::Float(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::SelectB => {
            let cond = args[0].bool_val().or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].bool_val(),
                Some(false) => args[2].bool_val(),
            };
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }

        // ── Casting ───────────────────────────────────────────────
        IR::IntCast => {
            let inferred = args[0].float_val().map(|v| v as i64);
            Value::Integer(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::FloatCast => {
            let inferred = args[0].int_val().map(|v| v as f64);
            Value::Float(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::BoolCast => {
            let inferred = args[0].int_val().map(|v| v != 0);
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }

        // ── String operations ─────────────────────────────────────
        IR::AddStr => {
            let sa = args[0].string_val();
            let sb = args[1].string_val();
            let val_str = match (sa, sb) {
                (Some(a), Some(b)) => format!("{}{}", a, b),
                _ => String::new(),
            };
            Value::String(StringValue {
                    val: val_str,
                    stmt_id: ir_id,
                })
        }
        IR::StrI => Value::String(StringValue {
                val: args[0]
                    .int_val()
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                stmt_id: ir_id,
            }),
        IR::StrF => Value::String(StringValue {
                val: args[0]
                    .float_val()
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                stmt_id: ir_id,
            }),

        // ── I/O (fixed, no inference) ─────────────────────────────
        IR::ReadInteger { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::ReadFloat { .. } => Value::Float(ScalarValue::new(None, Some(ir_id))),
        IR::ReadHash { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::ReadExternalResult { is_float, .. } => if *is_float {
                Value::Float(ScalarValue::new(None, Some(ir_id)))
            } else {
                Value::Integer(ScalarValue::new(None, Some(ir_id)))
            },
        IR::Print => Value::None,

        // ── Assert & expose (fixed, return None) ──────────────────
        IR::Assert => Value::None,
        IR::StructuralPredicate { .. } => Value::None,
        IR::ScalarPrecondition { .. } => Value::None,
        IR::ExposePublicI | IR::ExposePublicF => Value::None,

        // ── Memory operations (fixed) ─────────────────────────────
        IR::AllocateMemory { .. } => Value::None,
        IR::WriteMemory { .. } => Value::None,
        IR::ReadMemory { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::MemoryTraceEmit { .. } => Value::None,
        IR::MemoryTraceSeal => Value::None,

        // ── Dynamic NDArray (fixed) ───────────────────────────────
        IR::AllocateDynamicNDArrayMeta { .. } => Value::None,
        IR::WitnessDynamicNDArrayMeta { .. } | IR::AssertDynamicNDArrayMeta { .. } => {
            Value::None
        }
        IR::DynamicNDArrayGetItem { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::DynamicNDArraySetItem { .. } => Value::None,

        // ── External calls (fixed) ────────────────────────────────
        IR::InvokeExternal { .. } => Value::None,
        IR::ExportExternalI { .. } | IR::ExportExternalF { .. } => Value::None,

        // ── Hash ──────────────────────────────────────────────────
        IR::PoseidonHash => {
            Value::Integer(ScalarValue::new(None, Some(ir_id)))
        }
        IR::EqHash => Value::Boolean(ScalarValue::new(None, Some(ir_id))),
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn ptr_of(val: &Value) -> StmtId {
    val.stmt_id().unwrap_or_else(|| panic!("Value must have a pointer: {:?}", val))
}

fn int_binary_ir(ir_id: StmtId, args: &[Value], op: impl Fn(i64, i64) -> Option<i64>) -> Value {
    let la = args[0].int_val();
    let lb = args[1].int_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => op(a, b),
        _ => None,
    };
    Value::Integer(ScalarValue::new(inferred, Some(ir_id)))
}

fn int_unary_ir(ir_id: StmtId, args: &[Value], op: impl Fn(i64) -> Option<i64>) -> Value {
    let la = args[0].int_val();
    let inferred = la.and_then(op);
    Value::Integer(ScalarValue::new(inferred, Some(ir_id)))
}

fn float_binary_ir(ir_id: StmtId, args: &[Value], op: impl Fn(f64, f64) -> f64) -> Value {
    let la = args[0].float_val();
    let lb = args[1].float_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => Some(op(a, b)),
        _ => None,
    };
    Value::Float(ScalarValue::new(inferred, Some(ir_id)))
}

fn float_unary_ir(ir_id: StmtId, args: &[Value], op: impl Fn(f64) -> f64) -> Value {
    let la = args[0].float_val();
    let inferred = la.map(op);
    Value::Float(ScalarValue::new(inferred, Some(ir_id)))
}

fn int_cmp_ir(ir_id: StmtId, args: &[Value], op: impl Fn(i64, i64) -> bool) -> Value {
    let la = args[0].int_val();
    let lb = args[1].int_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => Some(op(a, b)),
        _ => None,
    };
    Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
}

fn float_cmp_ir(ir_id: StmtId, args: &[Value], op: impl Fn(f64, f64) -> bool) -> Value {
    let la = args[0].float_val();
    let lb = args[1].float_val();
    let inferred = match (la, lb) {
        (Some(a), Some(b)) => Some(op(a, b)),
        _ => None,
    };
    Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_constant_int() {
        let mut b = IRBuilder::new();
        let v = b.ir_constant_int(42);
        assert_eq!(v.int_val(), Some(42));
        assert_eq!(v.stmt_id(), Some(0));
        assert_eq!(b.stmts.len(), 1);
    }

    #[test]
    fn test_builder_add_i() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(10);
        let c = b.ir_constant_int(20);
        let sum = b.create_ir(&IR::AddI, &[a, c]);
        assert_eq!(sum.int_val(), Some(30));
        assert_eq!(sum.stmt_id(), Some(2));
        assert_eq!(b.stmts.len(), 3);
    }

    #[test]
    fn test_builder_export_graph() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(5);
        let c = b.ir_constant_int(3);
        let _ = b.create_ir(&IR::AddI, &[a, c]);
        let graph = b.export_ir_graph();
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_build_ir_select() {
        let mut b = IRBuilder::new();
        let cond = b.ir_constant_bool(true);
        let tv = b.ir_constant_int(10);
        let fv = b.ir_constant_int(20);
        let result = b.create_ir(&IR::SelectI, &[cond, tv, fv]);
        assert_eq!(result.int_val(), Some(10));
    }
}

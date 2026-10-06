use std::collections::{HashMap, HashSet};

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::lower::Constant;
use crate::optim::resolver::{Resolver, StaticOnlyResolver};
use crate::types::{NumberType, ScalarValue, StmtId, StringValue, Value};

mod fixed_width;
mod bool_dtype;
mod same_value;
pub(crate) mod guard;

#[derive(Clone)]
pub struct StaticArrayProductProvenance {
    pub lhs_cells: Vec<Value>,
    pub rhs_cells: Vec<Value>,
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub out_shape: Vec<usize>,
    pub dtype: NumberType,
}

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
            let formals = Self::binary_formals(a, b);
            self.discharge_contract_requires($contract, &formals);
            let out = self.create_ir(&$variant, &[a.clone(), b.clone()]);
            if let (Some(vid), false) = (out.value_id(), formals.is_empty()) {
                self.plant_contract_ensures($contract, vid, &formals);
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
            let formals = Self::unary_formals(a);
            self.discharge_contract_requires($contract, &formals);
            let out = self.create_ir(&$variant, &[a.clone()]);
            if let (Some(vid), false) = (out.value_id(), formals.is_empty()) {
                self.plant_contract_ensures($contract, vid, &formals);
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

pub use crate::provenance::SliceView;

/// Largest compile-time exponent a float power is expanded into products for
/// (`IRBuilder::ir_pow_f_int`); 64 needs at most 12 products.
const MAX_UNROLLED_POWER: i64 = 64;

/// One `IR::ArrayOp` placeholder at a constant position: the lane and the
/// operator's shared operands (the same `Arc` for every lane of one
/// operator instance).
#[derive(Clone)]
pub struct ArrayOpPlaceholder {
    pub op: crate::lower::ArrayOp,
    pub shared: std::sync::Arc<[Value]>,
}

/// IR builder that accumulates IR statements and provides typed convenience
/// methods. Mirrors Python `IRBuilderImpl` from `builder_impl.py`.
pub struct IRBuilder {
    pub stmts: Vec<IRStatement>,
    /// Constant return cells elided from the public instance (the
    /// constant-elision arms of `expose_circuit_return_value`). Carried
    /// onto the exported `IRGraph` so output auditors can verify the
    /// elided values against ground truth.
    pub elided_publics: Vec<crate::ir::ElidedPublic>,
    /// For each `ExposePublic` emitted so far, the number of elided
    /// constants recorded before it; locates the elided constants among
    /// the output slots at export (see `IRGraph::elided_publics`).
    elided_before_exposure: Vec<usize>,
    /// Conditional-write predicate. While `Some(p)`, every
    /// `ir_write_memory` / `ir_dynamic_ndarray_set_item` stores
    /// `select(p, new, old)` instead of `new`, so segment mutations
    /// emitted inside a dynamic branch only take effect when that
    /// branch's path condition holds. IRGen scopes this around `if`-arm
    /// visitation (`visit_cond`); a statically-true predicate costs
    /// nothing (the select is skipped). Without this, branch bodies
    /// write through to shared segments unconditionally — the
    /// last-write-wins / wrongly-constant-folded `if` bug class found
    /// by the 2026-06-11 binding audit (cases 0374/0490).
    write_predicate: Option<Value>,
    /// Memo tables of the checks-under-the-execution-predicate machinery
    /// (`builder/guard.rs`).
    guard_caches: guard::GuardCaches,
    /// Number of cells of every allocated memory segment, and of every
    /// dynamic ndarray's buffer (by array id): the range a dynamic address
    /// must stay in (see `guard_address`).
    segment_capacity: HashMap<u32, u32>,
    dyn_array_capacity: HashMap<u32, u32>,
    /// Segments allocated while a given predicate (by stmt id) was
    /// active — branch-local storage whose writes under that same
    /// predicate skip predication (see `ir_allocate_memory`).
    predicate_exempt_segments: HashMap<u32, StmtId>,
    /// Next available memory segment ID for dynamic array allocations.
    next_segment_id: u32,
    /// The `IR::ArrayOp` placeholders at constant positions this builder
    /// emitted, by statement (see [`IRBuilder::array_op_placeholder_of`]).
    array_op_placeholders: HashMap<StmtId, ArrayOpPlaceholder>,
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
    /// Index tuples produced by `np.nonzero` / one-argument `np.where`,
    /// keyed by the tuple's `ValueId`, mapped to a snapshot of the boolean
    /// mask they enumerate (taken when the tuple was built, so later writes
    /// to the source array do not change it). Assigning through such a
    /// tuple, `a[np.where(m)] = v`, is NumPy's `a[m] = v`.
    pub nonzero_index_masks: HashMap<crate::types::ValueId, Value>,
    /// Integer vectors (their elements' statements, in order) a
    /// precondition asserts to be a permutation of `0..len`
    /// (`record_permutation`).
    known_permutations: Vec<Vec<StmtId>>,
    /// Small expression-provenance side table for StaticArray results whose
    /// concrete payload can be reconstructed exactly, but may be consumed by
    /// a fused lowering before materializing every intermediate cell.
    pub static_array_product_provenance: HashMap<u32, StaticArrayProductProvenance>,
    /// Segment-backed StaticArrays whose logical payload is available in
    /// `static_array_payload` but whose per-cell memory writes have not yet
    /// been emitted. Static reads can use the cache directly; dynamic reads
    /// and writes must materialize first.
    pub lazy_static_array_segments: HashSet<u32>,
    /// Structural-provenance table (see `src/provenance.rs` for the
    /// record types and the five invariants). Owns what used to be the
    /// ad-hoc `slice_views` cache; transferred onto the `IRGraph` at
    /// export so backends can consume it.
    pub provenance: crate::provenance::ProvenanceTable,
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
    /// Structured-event telemetry sink (verification-telemetry-sink card).
    /// `None` in production unless `ZINNIA_TELEMETRY_DIR` is set; an
    /// `Option`-check at each emit site keeps the off-path overhead to a
    /// single conditional. `Arc` so the sink is cheap to share across
    /// builder clones and helper threads, and `Send + Sync` via the
    /// inner `Mutex<File>`.
    pub telemetry: Option<std::sync::Arc<crate::optim::telemetry::TelemetrySink>>,
    /// Per-segment, per-cell bound facts captured at write time. When a
    /// `Value` whose `ask_bounds` is `Some((min, max))` is written into a
    /// segment via `alloc_and_write_values` / `write_typed`, the bounds
    /// are recorded here. When a downstream `ir_read_memory` reads the
    /// cell at a statically-pinned address, the stored bounds are
    /// republished as a `vid >= min AND vid < max+1` fact on the read
    /// output's value_id.
    ///
    /// Closes the segment-storage fact-laundering gap that otherwise
    /// loses structurally-bounded values (e.g. argwhere bbox bounds
    /// emitted by `dyn_aggregate_axis`'s fast path) when downstream
    /// kernels do `B.min(0)[0]`-style tuple-unpack extraction — the
    /// extraction emits `IR::ReadMemory` which mints a fresh value_id
    /// for the cell, and the original `acc_idx` fact never propagates.
    pub segment_cell_bounds: HashMap<u32, HashMap<usize, (i64, i64)>>,
    /// Segment-wide bound, used when every cell of a segment is
    /// structurally in the same `(min, max)` range. Set by producers
    /// like `np_nonzero` that write at runtime addresses (where the
    /// per-cell `segment_cell_bounds` capture in `ir_write_memory` can't
    /// fire — the address has no `int_val`). Consumed by `ir_read_memory`
    /// as a fallback when per-cell bounds aren't available.
    pub segment_default_bounds: HashMap<u32, (i64, i64)>,
    /// Bound facts on scalar SSA values that downstream discharge can
    /// consult but other passes do not see. Used when publishing a paired
    /// `(Ge, Le)` fact into the FactStack would trigger an unwanted
    /// constant-fold cascade (e.g. the `dyn_aggregate_all` Min/Max result
    /// of a `flatnonzero` output, where the bound is `(0, axis_len-1)`
    /// but treating it as a Fact pinned the downstream slice shape and
    /// broke 0509's output type).
    pub scalar_value_bounds: HashMap<crate::types::ValueId, (i64, i64)>,
    /// Which statements and segments hold values of NumPy's fixed-width
    /// integer types (`builder/fixed_width.rs`).
    fixed_width: fixed_width::FixedWidthValues,
    /// NumPy's `bool` dtype on arrays (see `builder/bool_dtype.rs`): the
    /// segments that store a bool array, and the integer-typed statements
    /// whose value is a NumPy bool.
    bool_segments: HashSet<u32>,
    bool_cells: HashSet<StmtId>,
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
            elided_publics: Vec::new(),
            elided_before_exposure: Vec::new(),
            write_predicate: None,
            guard_caches: guard::GuardCaches::default(),
            segment_capacity: HashMap::new(),
            dyn_array_capacity: HashMap::new(),
            predicate_exempt_segments: HashMap::new(),
            next_segment_id: 0,
            array_op_placeholders: HashMap::new(),
            next_array_id: 0,
            dim_table: crate::types::DimTable::new(),
            static_array_payload: HashMap::new(),
            nonzero_index_masks: HashMap::new(),
            known_permutations: Vec::new(),
            static_array_product_provenance: HashMap::new(),
            lazy_static_array_segments: HashSet::new(),
            provenance: crate::provenance::ProvenanceTable::new(),
            resolver: Box::new(StaticOnlyResolver::new()),
            facts: crate::optim::predicates::FactStack::new(),
            value_to_stmt: HashMap::new(),
            // Activated when `ZINNIA_TELEMETRY_DIR` is set; `None` otherwise.
            // Reading the env var at construction time means an
            // `IRBuilder::new()` call after the env var changes won't pick
            // up the change, but the tests scope the env var around the
            // builder construction so this matches the desired semantics.
            telemetry: crate::optim::telemetry::TelemetrySink::from_env(),
            segment_cell_bounds: HashMap::new(),
            segment_default_bounds: HashMap::new(),
            scalar_value_bounds: HashMap::new(),
            fixed_width: fixed_width::FixedWidthValues::default(),
            bool_segments: HashSet::new(),
            bool_cells: HashSet::new(),
        }
    }

    /// Borrow the structured-event telemetry sink, if one is active. Helper
    /// for the four `tracing::debug!` sites that pair structured events
    /// alongside the existing trace lines (verification-telemetry-sink).
    pub fn telemetry(&self) -> Option<&std::sync::Arc<crate::optim::telemetry::TelemetrySink>> {
        self.telemetry.as_ref()
    }

    /// Replace the structured-event sink (test hook). Production code uses
    /// the env-driven `from_env()` path in `new`; tests need to attach a
    /// sink pointed at a tmp directory after constructing the builder.
    pub fn set_telemetry(
        &mut self,
        sink: Option<std::sync::Arc<crate::optim::telemetry::TelemetrySink>>,
    ) {
        self.telemetry = sink;
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

    /// The compile-time value of statement `stmt` when it and every
    /// statement it depends on fold to constants (the value `ConstantFold`
    /// gives it, by the same folding rules), else `None`. Looks at most
    /// `depth` statements deep.
    pub fn constant_value_of(&self, stmt: StmtId, depth: usize) -> Option<Value> {
        let st = self.stmts.get(stmt as usize)?;
        let args: Vec<Value> = if st.arguments.is_empty() {
            Vec::new()
        } else if depth == 0 {
            return None;
        } else {
            st.arguments.iter().map(|&a| self.constant_value_of(a, depth - 1)).collect::<Option<_>>()?
        };
        let value = build_value(&st.ir, stmt, &args);
        let known = value.int_val().is_some() || value.float_val().is_some() || value.bool_val().is_some();
        known.then_some(value)
    }

    /// Record that `elements` are asserted (on every path) to be a
    /// permutation of `0..len`, with the range facts `0 <= e < len` for the
    /// index checks that follow.
    pub fn record_permutation(&mut self, elements: &[Value]) {
        use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
        let n = elements.len() as i64;
        for e in elements {
            let Some(vid) = e.value_id() else { continue };
            let var = || Box::new(ContractTerm::Var(ContractVar::Value(vid)));
            self.facts.insert_for(
                vid,
                ContractTerm::BoolComb {
                    op: BoolOp::And,
                    operands: vec![
                        ContractTerm::Cmp { op: CmpOp::Ge, lhs: var(), rhs: Box::new(ContractTerm::LitInt(0)) },
                        ContractTerm::Cmp { op: CmpOp::Lt, lhs: var(), rhs: Box::new(ContractTerm::LitInt(n)) },
                    ],
                },
            );
        }
        if let Some(stmts) = elements.iter().map(Value::stmt_id).collect::<Option<Vec<_>>>() {
            self.known_permutations.push(stmts);
        }
    }

    /// Whether `elements` (in order) were recorded as a permutation.
    pub fn is_known_permutation(&self, elements: &[Value]) -> bool {
        let Some(stmts) = elements.iter().map(Value::stmt_id).collect::<Option<Vec<_>>>() else { return false };
        self.known_permutations.contains(&stmts)
    }

    /// Reconstruct an integer `Value` wrapper for a recorded `ValueId`
    /// (static value recovered when the producer is a `ConstantInt`).
    /// Used by provenance-record consumers (e.g. compaction scans) that
    /// hold `ValueId`s and need IR-layer references to emit ops.
    pub fn value_of_vid(&self, vid: crate::types::ValueId) -> Option<Value> {
        let sid = self.value_to_stmt.get(&vid).copied()?;
        let static_val = match self.stmts.get(sid as usize).map(|st| &st.ir) {
            Some(IR::ConstantInt { value }) => Some(*value),
            _ => None,
        };
        let mut sv = crate::types::ScalarValue::new(static_val, Some(sid));
        sv.value_id = vid;
        Some(Value::Integer(sv))
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
    pub fn split_resolver_and_stmts(&mut self) -> (&mut dyn Resolver, &[IRStatement]) {
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
            if mn <= mx {
                return Some(self.narrow_dtype_wide_bounds(val, (mn, mx)));
            }
        }
        // Direct fact lookup on this value_id.
        if let Some(bounds) = val
            .value_id()
            .and_then(|vid| crate::optim::resolver::derive_bounds_from_facts(&self.facts, vid))
        {
            return Some(bounds);
        }
        // Private scalar-bounds sidetable. Holds bounds we don't want to
        // expose as Facts (which would trigger constant-fold cascades on
        // downstream shape derivation). Only consulted here.
        if let Some(bounds) = val
            .value_id()
            .and_then(|vid| self.scalar_value_bounds.get(&vid).copied())
        {
            return Some(bounds);
        }
        // One-level arithmetic propagation. The range pass doesn't consult
        // facts (its `interval_of` only reads IR), so a value like
        // `read_memory + 1` walks to `unbounded + (1,1) = unbounded` even
        // when the read output has a published bound fact. Recover here:
        // for the common shape `AddI(a, ConstantInt(c))` / `SubI`, recurse
        // ask_bounds on the non-constant operand and shift. Avoids
        // changing the Resolver trait shape just to thread facts in.
        if let Some(bounds) = self.ask_bounds_via_arith(val) {
            return Some(bounds);
        }
        self.deep_bounds_walk(val)
    }

    /// Narrow the resolver's bound `(min, max)` of `val` by the facts.
    ///
    /// The range pass reads only the IR, so for an input it knows just the
    /// dtype's domain: with `@requires(0 <= n <= 8)` it bounds `n - 1` by
    /// about `[-2^63, 2^63)`, and `ask_bounds` returned that, ignoring the
    /// precondition (`range(n - 1, 0, -1)` was rejected as unbounded, while
    /// `range(n + 1, 0, -1)`, whose resolver bound overflows and falls back
    /// to the facts, compiled). Both bounds hold for every value of `val`,
    /// so their intersection does. Only a bound wider than any array (2^32)
    /// is narrowed, which keeps the extra walk off the common path.
    fn narrow_dtype_wide_bounds(&mut self, val: &Value, resolved: (i64, i64)) -> (i64, i64) {
        const WIDE: i64 = 1 << 32;
        let (mn, mx) = resolved;
        if mx.saturating_sub(mn) < WIDE {
            return resolved;
        }
        let from_facts = val
            .value_id()
            .and_then(|vid| crate::optim::resolver::derive_bounds_from_facts(&self.facts, vid))
            .or_else(|| val.value_id().and_then(|vid| self.scalar_value_bounds.get(&vid).copied()))
            .or_else(|| self.ask_bounds_via_arith(val))
            .or_else(|| self.deep_bounds_walk(val));
        match from_facts {
            Some((lo, hi)) if mn.max(lo) <= mx.min(hi) => (mn.max(lo), mx.min(hi)),
            _ => resolved,
        }
    }

    /// Deep IR-walk bound estimator used as the last fallback in
    /// [`ask_bounds`]. Walks the SSA DAG, combining intervals via
    /// arithmetic and (for `SelectI`) union of branches. Consults
    /// [`Self::segment_cell_bounds`] / [`Self::segment_default_bounds`]
    /// on `ReadMemory` so values that originate from a bounded segment
    /// carry through chains the range resolver doesn't model (e.g. the
    /// Min/Max `SelectI` reduction of `flatnonzero` outputs feeding a
    /// `bbox.max() + 1` slice bound on 0507/0509/0510).
    ///
    /// Per-call memoization keeps cost O(stmts visited) instead of
    /// exponential on `SelectI` chains. Bounded recursion depth caps
    /// pathological cases.
    pub(crate) fn deep_bounds_walk(&self, val: &Value) -> Option<(i64, i64)> {
        let mut cache: std::collections::HashMap<crate::types::StmtId, Option<(i64, i64)>> =
            std::collections::HashMap::new();
        self.deep_bounds_walk_inner(val, &mut cache, 0)
    }

    fn deep_bounds_walk_inner(
        &self,
        val: &Value,
        cache: &mut std::collections::HashMap<crate::types::StmtId, Option<(i64, i64)>>,
        depth: u32,
    ) -> Option<(i64, i64)> {
        use crate::ir_defs::IR;
        if depth > 64 {
            return None;
        }
        if let Some(n) = val.int_val() {
            return Some((n, n));
        }
        // Direct facts on this value_id take precedence — they encode
        // tighter knowledge than a structural walk.
        if let Some(vid) = val.value_id() {
            if let Some(b) = crate::optim::resolver::derive_bounds_from_facts(&self.facts, vid) {
                return Some(b);
            }
        }
        let stmt_id = val.stmt_id()?;
        if let Some(cached) = cache.get(&stmt_id) {
            return *cached;
        }
        // Insert a `None` placeholder so cycles (shouldn't happen in
        // SSA, but defensive) don't infinite-recurse.
        cache.insert(stmt_id, None);
        let stmt = self.stmts.get(stmt_id as usize)?;
        let result = match &stmt.ir {
            IR::ConstantInt { value } => Some((*value, *value)),
            IR::ConstantBool { value } => {
                Some((if *value { 1 } else { 0 }, if *value { 1 } else { 0 }))
            }
            IR::ReadMemory { segment_id } => {
                // Try per-cell bound if the address is statically pinned;
                // otherwise fall through to segment-wide default.
                let addr_stmt_id = *stmt.arguments.first()?;
                let addr_stmt = self.stmts.get(addr_stmt_id as usize)?;
                let addr_val = if let IR::ConstantInt { value } = &addr_stmt.ir {
                    Some(*value)
                } else {
                    None
                };
                if let Some(addr_i) = addr_val {
                    if let Some(b) = self
                        .segment_cell_bounds
                        .get(segment_id)
                        .and_then(|m| m.get(&(addr_i as usize)))
                    {
                        return Some(*b);
                    }
                }
                self.segment_default_bounds.get(segment_id).copied()
            }
            IR::AddI if stmt.arguments.len() == 2 => {
                let a = self.deep_bounds_walk_stmt(stmt.arguments[0], cache, depth + 1)?;
                let b = self.deep_bounds_walk_stmt(stmt.arguments[1], cache, depth + 1)?;
                Some((a.0.saturating_add(b.0), a.1.saturating_add(b.1)))
            }
            IR::SubI if stmt.arguments.len() == 2 => {
                let a = self.deep_bounds_walk_stmt(stmt.arguments[0], cache, depth + 1)?;
                let b = self.deep_bounds_walk_stmt(stmt.arguments[1], cache, depth + 1)?;
                Some((a.0.saturating_sub(b.1), a.1.saturating_sub(b.0)))
            }
            // SelectI walking is intentionally omitted. Unioning Min/Max
            // reduction branches surfaces correct bounds on the Min/Max
            // scalar result (e.g. `flatnonzero(...).min() ∈ [0,
            // axis_len-1]`), but `ask_bounds` consumers further along
            // collapse the entire downstream slice expression — 0509 went
            // from `pub [Float; 100]` output to `pub Field` returning
            // `im[99]`. Reproduced and bisected but the constant-fold
            // path that misuses the bound is not yet identified. Closing
            // the flatnonzero-min/max bbox subfamily needs a targeted
            // fix once that path is found.
            _ => None,
        };
        cache.insert(stmt_id, result);
        result
    }

    fn deep_bounds_walk_stmt(
        &self,
        stmt_id: crate::types::StmtId,
        cache: &mut std::collections::HashMap<crate::types::StmtId, Option<(i64, i64)>>,
        depth: u32,
    ) -> Option<(i64, i64)> {
        use crate::ir_defs::IR;
        if let Some(cached) = cache.get(&stmt_id) {
            return *cached;
        }
        let stmt = self.stmts.get(stmt_id as usize)?;
        // Reconstruct a Value::Integer handle for the recursion. Only int_val
        // (for `ConstantInt`) and value_id are consulted in the inner walk.
        let static_val = if let IR::ConstantInt { value } = &stmt.ir {
            Some(*value)
        } else {
            None
        };
        let scalar = crate::types::ScalarValue {
            static_val,
            stmt_id: Some(stmt_id),
            value_id: stmt.value_id,
        };
        self.deep_bounds_walk_inner(&Value::Integer(scalar), cache, depth)
    }

    /// One-level fact-aware bound derivation through `AddI` / `SubI` /
    /// `MulI` when one operand is a constant. Used as a fallback in
    /// [`ask_bounds`] to bridge the gap between fact-on-leaf and
    /// arithmetic-on-leaf when the resolver's range pass doesn't see
    /// facts. Recursive but bounded — only fires when the inner operand
    /// itself recurses through this path, and each step strips a stmt.
    fn ask_bounds_via_arith(&mut self, val: &Value) -> Option<(i64, i64)> {
        use crate::ir_defs::IR;
        let stmt_id = val.stmt_id()?;
        let stmt = self.stmts.get(stmt_id as usize)?.clone();
        if stmt.arguments.len() != 2 || stmt.arg_values.len() != 2 {
            return None;
        }
        // Identify which arg (if any) is a ConstantInt so we can shift the
        // non-constant operand's bound by it.
        let arg0_stmt = self.stmts.get(stmt.arguments[0] as usize)?;
        let arg1_stmt = self.stmts.get(stmt.arguments[1] as usize)?;
        let arg0_const = match &arg0_stmt.ir {
            IR::ConstantInt { value } => Some(*value),
            _ => None,
        };
        let arg1_const = match &arg1_stmt.ir {
            IR::ConstantInt { value } => Some(*value),
            _ => None,
        };
        let (nonconst_vid, const_val, const_on_left) = match (arg0_const, arg1_const) {
            (None, Some(c)) => (stmt.arg_values[0], c, false),
            (Some(c), None) => (stmt.arg_values[1], c, true),
            _ => return None,
        };
        let (lo, hi) = crate::optim::resolver::derive_bounds_from_facts(&self.facts, nonconst_vid)
            .or_else(|| self.scalar_value_bounds.get(&nonconst_vid).copied())?;
        match &stmt.ir {
            IR::AddI => Some((lo + const_val, hi + const_val)),
            IR::SubI => {
                if const_on_left {
                    Some((const_val - hi, const_val - lo))
                } else {
                    Some((lo - const_val, hi - const_val))
                }
            }
            _ => None,
        }
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
        self.plant_contract_ensures(name, output_vid, formals);
    }

    /// The `requires` half of [`Self::fire_contract`] for a contract whose
    /// preconditions mention only the operands (formals), discharged
    /// before the operation is emitted: the check then precedes the
    /// computation it guards, so the lowering of the operation can use the
    /// range the check proves (a `sqrt` of an operand already asserted
    /// non-negative needs no sign handling). Follow the operation with
    /// [`Self::plant_contract_ensures`].
    pub fn discharge_contract_requires(
        &mut self,
        name: &str,
        formals: &HashMap<String, crate::types::ValueId>,
    ) {
        if formals.is_empty() {
            return; // a literal operand: nothing to discharge at runtime
        }
        let contract = crate::optim::predicates::op_contract_by_name(name);
        for req in &contract.requires {
            let term = crate::optim::predicates::instantiate_contract(&req.term, None, formals);
            self.discharge_requires(name, &term);
        }
    }

    /// The `ensures` half of [`Self::fire_contract`].
    pub fn plant_contract_ensures(
        &mut self,
        name: &str,
        output_vid: crate::types::ValueId,
        formals: &HashMap<String, crate::types::ValueId>,
    ) {
        let contract = crate::optim::predicates::op_contract_by_name(name);
        for ensures in &contract.ensures {
            let fact = crate::optim::predicates::instantiate_contract(
                &ensures.term,
                Some(output_vid),
                formals,
            );
            // Structured telemetry: record the per-ensures plant before
            // moving the fact into the stack (so `term_summary` doesn't
            // need a re-clone post-insert).
            if let Some(sink) = &self.telemetry {
                sink.emit(&crate::optim::telemetry::TelemetryEvent::FactEmit {
                    producer: name.to_string(),
                    output_value_id: output_vid.0,
                    term_summary: format!("{:?}", fact),
                });
            }
            self.facts.insert_for(output_vid, fact);
        }
    }

    /// Discharge a single, fully-instantiated `requires` term against the
    /// live fact stack. Policy:
    ///
    /// - **Proved**: the precondition follows from the visible facts; no
    ///   action besides telemetry.
    /// - **Disproved**: the visible facts contradict the precondition. On
    ///   code that runs on every path the program always fails: panic with
    ///   a diagnostic naming the op, the substituted term, and the
    ///   value-ids it references. Under a runtime execution predicate `p`
    ///   the circuit asserts `¬p` instead (the op is reached only where
    ///   Python would raise).
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
                if let Some(sink) = &self.telemetry {
                    let value_ids = crate::optim::predicates::collect_value_ids(term);
                    sink.emit(&crate::optim::telemetry::discharge_event(
                        op_name, "Proved", "none", false, &value_ids,
                    ));
                }
            }
            ProveOutcome::Disproved => {
                // Under a runtime execution predicate the op is only
                // reached where that predicate holds: the precondition
                // failing means the code must not run there.
                if self.assert_never_runs() {
                    return;
                }
                let value_ids = crate::optim::predicates::collect_value_ids(term);
                tracing::debug!(
                    target: "zinnia::op_contract",
                    "op_requires_discharge: op={} outcome=Disproved (compile error)",
                    op_name,
                );
                if let Some(sink) = &self.telemetry {
                    sink.emit(&crate::optim::telemetry::discharge_event(
                        op_name,
                        "Disproved",
                        "none",
                        false,
                        &value_ids,
                    ));
                }
                panic!(
                    "op `{op_name}` requires precondition disproved by visible facts: \
                     term={term:?}, value_ids={value_ids:?}"
                );
            }
            ProveOutcome::Unknown => {
                if op_requires_strict() {
                    let value_ids = crate::optim::predicates::collect_value_ids(term);
                    tracing::debug!(
                        target: "zinnia::op_contract",
                        "op_requires_discharge: op={} outcome=Unknown mode=strict (compile error)",
                        op_name,
                    );
                    if let Some(sink) = &self.telemetry {
                        sink.emit(&crate::optim::telemetry::discharge_event(
                            op_name, "Unknown", "strict", false, &value_ids,
                        ));
                    }
                    panic!(
                        "op `{op_name}` requires precondition not provable from visible facts \
                         (strict mode): term={term:?}, value_ids={value_ids:?}"
                    );
                }
                let emitted = self.emit_requires_witness_check(term);
                if !emitted {
                    let value_ids = crate::optim::predicates::collect_value_ids(term);
                    tracing::debug!(
                        target: "zinnia::op_contract",
                        "op_requires_discharge: op={} outcome=Unknown mode=lenient witness_emit=no (compile error)",
                        op_name,
                    );
                    if let Some(sink) = &self.telemetry {
                        sink.emit(&crate::optim::telemetry::discharge_event(
                            op_name, "Unknown", "lenient", false, &value_ids,
                        ));
                    }
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
                if let Some(sink) = &self.telemetry {
                    let value_ids = crate::optim::predicates::collect_value_ids(term);
                    sink.emit(&crate::optim::telemetry::discharge_event(
                        op_name, "Unknown", "lenient", true, &value_ids,
                    ));
                }
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
                let any_float = matches!(l, Value::Float(_)) || matches!(r, Value::Float(_));
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
                Some(self.value_from_stmt_id_for_value(sid, *vid))
            }
            // Unsubstituted formals / outputs / inputs at this layer are
            // template-shape bugs; the contract should have been
            // instantiated before reaching the discharge layer.
            ContractTerm::Var(_) => None,
            ContractTerm::Arith { op, lhs, rhs } => {
                let l = self.emit_term_as_num_value(lhs)?;
                let r = self.emit_term_as_num_value(rhs)?;
                let any_float = matches!(l, Value::Float(_)) || matches!(r, Value::Float(_));
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
    fn value_from_stmt_id_for_value(&self, sid: StmtId, value_id: crate::types::ValueId) -> Value {
        let stmt = match self.stmts.get(sid as usize) {
            Some(s) => s,
            None => {
                return Value::Integer(ScalarValue {
                    static_val: None,
                    stmt_id: Some(sid),
                    value_id,
                })
            }
        };
        match &stmt.ir {
            IR::ConstantFloat { .. }
            | IR::AddF
            | IR::SumF { .. }
            | IR::SubF
            | IR::MulF
            | IR::ProductSumF { .. }
            | IR::DivF
            | IR::FloorDivF
            | IR::ModF
            | IR::PowF
            | IR::PowFI { .. }
            | IR::AbsF
            | IR::SignF
            | IR::SinF
            | IR::SinHF
            | IR::CosF
            | IR::CosHF
            | IR::TanF
            | IR::TanHF
            | IR::SqrtF
            | IR::SqrtProductIF
            | IR::ExpF
            | IR::LogF
            | IR::ArcSinF
            | IR::ArcCosF
            | IR::ArcTan2F
            | IR::FloatCast
            | IR::ReadFloat { .. }
            | IR::SelectF
            | IR::SelectBitF
            | IR::SelectManyF { .. } => Value::Float(ScalarValue {
                static_val: None,
                stmt_id: Some(sid),
                value_id,
            }),
            IR::ConstantBool { .. }
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::LogicalNot
            | IR::BoolCast
            | IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LtIBounded
            | IR::IndexInBox { .. }
            | IR::LteI
            | IR::GtI
            | IR::GtIBounded
            | IR::GteI
            | IR::EqF
            | IR::NeF
            | IR::LtF
            | IR::LteF
            | IR::GtF
            | IR::GteF
            | IR::SelectB
            | IR::EqHash => Value::Boolean(ScalarValue {
                static_val: None,
                stmt_id: Some(sid),
                value_id,
            }),
            _ => Value::Integer(ScalarValue {
                static_val: None,
                stmt_id: Some(sid),
                value_id,
            }),
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
            // C12b: previously these arms silently coerced garbage to a
            // scalar — None/Class became `ir_constant_int(0)` and a
            // List/Tuple was replaced by its FIRST element. A caller passing
            // a composite (or None) to a scalar IR op then got a structurally
            // valid statement computing the WRONG thing, undermining the
            // hard-panic in `create_ir` two steps down. Fail loudly instead:
            // the correct fix is at the call site that passed the wrong value.
            Value::None | Value::Class(_) => {
                panic!(
                    "IRBuilder::ensure_ptr: cannot materialize {:?} as a scalar IR pointer. \
                     A None/Class value reached a scalar IR op; fix the call site that passed it.",
                    val
                );
            }
            Value::List(_) | Value::Tuple(_) => {
                panic!(
                    "IRBuilder::ensure_ptr: cannot materialize a composite (List/Tuple) as a \
                     scalar IR pointer; got {:?}. A composite reached a scalar IR op; the right \
                     fix is at the call site (e.g. dispatch a per-element loop or convert to a \
                     StaticArray), not silently taking the first element.",
                    val
                );
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
        //
        // Hard-fail on missing stmt_id rather than silently dropping the arg.
        // Silent-drop produces zero-arg / under-arg IR statements that crash
        // preprocess / synth later with opaque "index out of bounds" panics
        // (the 2026-05-30 fancy-index-assignment bug — see memory). If this
        // panics, the upstream caller is passing a non-scalar / unmaterialized
        // Value to an IR op that expects a scalar; the right fix is at the
        // call site (e.g. detect array-valued indices and dispatch a per-element
        // scatter loop), not here.
        for (i, a) in materialized.iter().enumerate() {
            if a.stmt_id().is_none() {
                panic!(
                    "IRBuilder::create_ir({:?}): arg[{}] has no stmt_id after ensure_ptr; raw={:?} materialized={:?}",
                    ir, i, args[i], a
                );
            }
        }
        let arguments: Vec<StmtId> = materialized.iter().filter_map(|a| a.stmt_id()).collect();
        let arg_values: Vec<crate::types::ValueId> =
            materialized.iter().filter_map(|a| a.value_id()).collect();
        let value_id = val.value_id().unwrap_or_else(crate::types::ValueId::next);
        let stmt = IRStatement::new(ir_id, value_id, ir.clone(), arguments, arg_values, None);
        self.stmts.push(stmt);
        self.record_value_id(&val);
        self.deposit_provenance(ir, &materialized, &val);
        self.track_statement(ir, ir_id, &materialized);
        val
    }

    /// Structural-provenance derivation chokepoint: every IR op deposits
    /// its affine/interval records here, ONCE, at construction time.
    /// Backends consume these instead of re-deriving the same algebra in
    /// per-backend caches (plan Step 2; the halo2 `AffineLin` and Noir
    /// `AffLin` machineries this replaces refused the same cases —
    /// >2 roots, coefficient overflow — so refusal here means "no
    /// record", never wrong math).
    fn deposit_provenance(&mut self, ir: &IR, args: &[Value], out: &Value) {
        use crate::provenance::{AffineView, Interval};

        fn static_int(v: &Value) -> Option<i64> {
            v.int_val()
                .or_else(|| v.bool_val().map(|b| if b { 1 } else { 0 }))
        }
        fn affine_of(me: &IRBuilder, v: &Value) -> Option<AffineView> {
            if let Some(c) = static_int(v) {
                return Some(AffineView::constant(c));
            }
            let vid = v.value_id()?;
            if let Some(a) = me.provenance.affine_of(vid) {
                return Some(a.clone());
            }
            Some(AffineView::root(vid))
        }
        fn interval_of(me: &IRBuilder, v: &Value) -> Option<Interval> {
            if let Some(c) = static_int(v) {
                return Some(Interval::point(c));
            }
            me.provenance.interval_of(v.value_id()?)
        }

        match ir {
            IR::ConstantInt { value } => {
                if let Some(vid) = out.value_id() {
                    self.provenance.set_affine(vid, AffineView::constant(*value));
                    self.provenance.set_interval(vid, Interval::point(*value));
                }
            }
            IR::ConstantBool { value } => {
                if let Some(vid) = out.value_id() {
                    let v = if *value { 1 } else { 0 };
                    self.provenance.set_interval(vid, Interval::point(v));
                }
            }
            IR::AddI | IR::SubI => {
                let Some(vid) = out.value_id() else { return };
                let sign = if matches!(ir, IR::AddI) { 1 } else { -1 };
                if let (Some(a), Some(b)) = (affine_of(self, &args[0]), affine_of(self, &args[1]))
                {
                    if let Some(c) = AffineView::combine(&a, &b, sign) {
                        self.provenance.set_affine(vid, c);
                    }
                }
                if let (Some(a), Some(b)) =
                    (interval_of(self, &args[0]), interval_of(self, &args[1]))
                {
                    if let Some(c) = Interval::add(a, b, sign) {
                        self.provenance.set_interval(vid, c);
                    }
                }
            }
            IR::MulI => {
                let Some(vid) = out.value_id() else { return };
                let (konst, other) = match (static_int(&args[0]), static_int(&args[1])) {
                    (Some(k), _) => (Some(k), &args[1]),
                    (_, Some(k)) => (Some(k), &args[0]),
                    _ => (None, &args[0]),
                };
                if let Some(k) = konst {
                    if let Some(a) = affine_of(self, other) {
                        if let Some(scaled) = a.scaled(k) {
                            self.provenance.set_affine(vid, scaled);
                        }
                    }
                    if let Some(iv) = interval_of(self, other) {
                        if let Some(scaled) = iv.scaled(k) {
                            self.provenance.set_interval(vid, scaled);
                        }
                    }
                }
            }
            IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LtIBounded
            | IR::LteI
            | IR::GtI
            | IR::GtIBounded
            | IR::GteI
            | IR::EqF
            | IR::NeF
            | IR::LtF
            | IR::LteF
            | IR::GtF
            | IR::GteF
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::LogicalNot
            | IR::BoolCast
            | IR::KeyLess { .. }
            | IR::PackedKeyLess { .. }
            | IR::EqHash => {
                if let Some(vid) = out.value_id() {
                    self.provenance.set_interval(vid, Interval::BIT);
                }
            }
            // What the operator's semantics bound a lane by holds for every
            // construction of it (`lower::op`).
            IR::ArrayOp(op) => {
                let Some(vid) = out.value_id() else { return };
                let operands: Vec<Option<Interval>> = args.iter().map(|a| interval_of(self, a)).collect();
                if let Some(iv) = op.kind.semantics().bounds(op, &operands) {
                    self.provenance.set_interval(vid, iv);
                }
            }
            IR::SelectI | IR::SelectBitI => {
                let Some(vid) = out.value_id() else { return };
                let ti = interval_of(self, &args[1]);
                let fi = interval_of(self, &args[2]);
                // Generic selects need both branch intervals for a hull;
                // min/max shapes (recognized below) can bound from ONE
                // known side, so don't bail early.
                let mut iv = match (ti, fi) {
                    (Some(a), Some(b)) => Interval::union(a, b),
                    _ => Interval {
                        lo: i64::MIN / 4,
                        hi: i64::MAX / 4,
                    },
                };
                let mut have_any = ti.is_some() && fi.is_some();
                // Min/max recognition: select(a<b, a, b) = min(a, b) →
                // hi tightens to min(hi_a, hi_b); select(a<b, b, a) =
                // max(a, b) → lo tightens to max(lo_a, lo_b). Same for
                // the > comparators with branches swapped.
                if let Some(cond_stmt) = args[0].stmt_id() {
                    if let Some(stmt) = self.stmts.get(cond_stmt as usize) {
                        let is_lt = matches!(stmt.ir, IR::LtI | IR::LtIBounded);
                        let is_gt = matches!(stmt.ir, IR::GtI | IR::GtIBounded);
                        if (is_lt || is_gt) && stmt.arg_values.len() == 2 {
                            let (x, y) = (stmt.arg_values[0], stmt.arg_values[1]);
                            let (t, f) = (args[1].value_id(), args[2].value_id());
                            let straight = t == Some(x) && f == Some(y);
                            let swapped = t == Some(y) && f == Some(x);
                            // lt + straight = min; lt + swapped = max;
                            // gt + straight = max; gt + swapped = min.
                            let is_min = (is_lt && straight) || (is_gt && swapped);
                            let is_max = (is_lt && swapped) || (is_gt && straight);
                            // Structural clamp record: min/max against a
                            // CONSTANT branch keeps the other branch's
                            // identity (backend indicator sharing).
                            if is_min || is_max {
                                let pairs =
                                    [(&args[1], &args[2]), (&args[2], &args[1])];
                                for (cv, sv) in pairs {
                                    if let (Some(k), Some(src)) =
                                        (cv.int_val(), sv.value_id())
                                    {
                                        self.provenance.set_clamp(
                                            vid,
                                            crate::provenance::ClampView {
                                                source: src,
                                                bound: k,
                                                is_min,
                                            },
                                        );
                                        break;
                                    }
                                }
                            }
                            // One known side suffices: min(a,b) <= hi of
                            // either; max(a,b) >= lo of either.
                            if is_min {
                                if let Some(h) = match (ti, fi) {
                                    (Some(a), Some(b)) => Some(a.hi.min(b.hi)),
                                    (Some(a), None) | (None, Some(a)) => Some(a.hi),
                                    _ => None,
                                } {
                                    iv.hi = iv.hi.min(h);
                                    have_any = true;
                                }
                            } else if is_max {
                                if let Some(l) = match (ti, fi) {
                                    (Some(a), Some(b)) => Some(a.lo.max(b.lo)),
                                    (Some(a), None) | (None, Some(a)) => Some(a.lo),
                                    _ => None,
                                } {
                                    iv.lo = iv.lo.max(l);
                                    have_any = true;
                                }
                            }
                        }
                    }
                }
                if have_any {
                    self.provenance.set_interval(vid, iv);
                }
            }
            IR::ReadMemory { segment_id } | IR::ReadTable { segment_id } | IR::DynamicNDArrayGetItem { segment_id, .. } => {
                if let (Some(vid), Some(hull)) =
                    (out.value_id(), self.provenance.segment_hull(*segment_id))
                {
                    self.provenance.set_interval(vid, hull);
                }
            }
            IR::AllocateMemory {
                segment_id,
                init_value,
                ..
            } => {
                self.provenance.init_segment_hull(*segment_id, *init_value);
            }
            IR::WriteMemory { segment_id } | IR::DynamicNDArraySetItem { segment_id, .. } => {
                // Invalidation chokepoint (provenance invariant #4): EVERY
                // write-shaped op bumps the segment version and eager-drops
                // segment-derived records here, so `ir_write_memory` and
                // `ir_dynamic_ndarray_set_item` (and any future write op)
                // get invalidation for free. Deposits made by helpers AFTER
                // their writes — e.g. `mask.rs` deposits its `MaskReadView`
                // after `alloc_and_write_values` — survive, because this
                // fires during the write, before the later deposit.
                self.provenance.on_segment_write(*segment_id);
                let iv = interval_of(self, &args[1]);
                self.provenance.widen_segment_hull(*segment_id, iv);
            }
            IR::AssertNonNegI => {
                // `x >= 0` over the signed-i64 domain (the halo2 check
                // enforces x in [0, 2^63)).
                if let Some(vid) = args[0].value_id() {
                    let bound = Interval { lo: 0, hi: i64::MAX };
                    let tightened = match self.provenance.interval_of(vid) {
                        Some(prev) => Interval::intersect(prev, bound),
                        None => bound,
                    };
                    self.provenance.set_interval(vid, tightened);
                }
            }
            IR::AssertU128Lt => {
                // `0 <= idx < hi` with a static hi tightens idx.
                if let (Some(idx_vid), Some(hi)) = (args[0].value_id(), static_int(&args[1])) {
                    if hi > 0 {
                        let bound = Interval { lo: 0, hi: hi - 1 };
                        let tightened = match self.provenance.interval_of(idx_vid) {
                            Some(prev) => Interval::intersect(prev, bound),
                            None => bound,
                        };
                        self.provenance.set_interval(idx_vid, tightened);
                    }
                }
            }
            _ => {}
        }
    }

    pub fn export_ir_graph(self) -> IRGraph {
        let elided = self.elided_output_slots();
        let provenance = self.provenance;
        let mut graph = IRGraph::new(self.stmts);
        graph.elided_publics = elided;
        graph.provenance = provenance;
        graph
    }

    // ── Convenience helpers used by optimization passes ──────────────

    pub fn ir_constant_int(&mut self, value: i64) -> Value {
        let out = self.create_ir(&IR::ConstantInt { value }, &[]);
        // Deposit a tight `out == value` fact so `derive_bounds_from_facts`
        // (and through it `interval_fact_for_int_binary`) can chain through
        // constants. Without this, arithmetic like `max(arr) + 1` fails
        // to propagate the bound on `max(arr)` to the sum, because the
        // constant operand has no per-value fact entries.
        // The fact defines the constant, so it holds on every path.
        if let Some(vid) = out.value_id() {
            use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
            self.facts.insert_unconditional(
                vid,
                ContractTerm::Cmp {
                    op: CmpOp::Eq,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
                    rhs: Box::new(ContractTerm::LitInt(value)),
                },
            );
        }
        out
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
    /// Deposit relational min/max facts: for a select-built max,
    /// `out >= a && out >= b` (min: `<=`). These let `prove()` discharge
    /// bound checks like `x < max(..., x, ...) + 1` at compile time,
    /// which the interval machinery cannot express (the bound is a
    /// variable, not a literal).
    pub fn deposit_minmax_facts(&mut self, out: &Value, a: &Value, b: &Value, is_max: bool) {
        use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
        let (Some(out_vid), Some(a_vid), Some(b_vid)) =
            (out.value_id(), a.value_id(), b.value_id())
        else {
            return;
        };
        let op = if is_max { CmpOp::Ge } else { CmpOp::Le };
        for rhs in [a_vid, b_vid] {
            self.facts.insert_for(
                out_vid,
                ContractTerm::Cmp {
                    op,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                    rhs: Box::new(ContractTerm::Var(ContractVar::Value(rhs))),
                },
            );
        }
    }

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
        {
            // Relational link for prove(): out == a + b. Lets variable-bound
            // chains (e.g. dims = max(...) + 1) discharge downstream
            // `idx < dims` checks that interval facts cannot express.
            use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
            if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
                (a.value_id(), b.value_id(), out.value_id())
            {
                // Volume gate: chain only through dynamic operands that
                // already carry facts (min/max outputs etc.). Depositing on
                // every integer add bloats the Z3 context of later proves.
                let chains = (a.int_val().is_none() && self.facts.has_facts_for(a_vid))
                    || (b.int_val().is_none() && self.facts.has_facts_for(b_vid));
                if chains {
                self.facts.insert_for(
                    out_vid,
                    ContractTerm::Cmp {
                        op: CmpOp::Eq,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                        rhs: Box::new(ContractTerm::Arith {
                            op: ArithOp::Add,
                            lhs: Box::new(ContractTerm::Var(ContractVar::Value(a_vid))),
                            rhs: Box::new(ContractTerm::Var(ContractVar::Value(b_vid))),
                        }),
                    },
                );
                }
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
        {
            // Relational link for prove(): out == a - b. Lets variable-bound
            // chains (e.g. dims = max(...) + 1) discharge downstream
            // `idx < dims` checks that interval facts cannot express.
            use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
            if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
                (a.value_id(), b.value_id(), out.value_id())
            {
                // Volume gate: chain only through dynamic operands that
                // already carry facts (min/max outputs etc.). Depositing on
                // every integer add bloats the Z3 context of later proves.
                let chains = (a.int_val().is_none() && self.facts.has_facts_for(a_vid))
                    || (b.int_val().is_none() && self.facts.has_facts_for(b_vid));
                if chains {
                self.facts.insert_for(
                    out_vid,
                    ContractTerm::Cmp {
                        op: CmpOp::Eq,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                        rhs: Box::new(ContractTerm::Arith {
                            op: ArithOp::Sub,
                            lhs: Box::new(ContractTerm::Var(ContractVar::Value(a_vid))),
                            rhs: Box::new(ContractTerm::Var(ContractVar::Value(b_vid))),
                        }),
                    },
                );
                }
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
    );
    ir_unary!(
        ir_inv_i  => IR::InvI, contract = "inv_i",
    );

    /// Integer `base ** exp`. The `pow_i` contract states the domain where
    /// the code runs; the halo2-base lowering of a runtime exponent also
    /// decomposes it into 6 bits, so where the code does not run the
    /// exponent is 0 (see `builder/guard.rs`).
    /// Integer power. NumPy raises `ValueError` for an integer to a negative
    /// integer power: a negative constant exponent is a compile error, and a
    /// runtime exponent not proven non-negative is asserted to be (no proof
    /// where the power runs with a negative exponent).
    pub fn ir_pow_i(&mut self, base: &Value, exp: &Value) -> Value {
        match exp.int_val() {
            Some(e) if e < 0 => panic!(
                "Integers to negative integer powers are not allowed (NumPy raises ValueError for an integer `** {e}`)"
            ),
            Some(_) => {}
            None => {
                if !self.ask_bounds(exp).is_some_and(|(lo, _)| lo >= 0) {
                    self.ir_assert_non_neg_i(exp);
                }
            }
        }
        self.power(&IR::PowI, "pow_i", base, exp)
    }

    fn power(&mut self, ir: &IR, contract: &str, base: &Value, exp: &Value) -> Value {
        let formals = Self::binary_formals(base, exp);
        self.discharge_contract_requires(contract, &formals);
        let exp_operand = match exp {
            Value::Integer(_) | Value::Boolean(_) => self.guard_operand(exp, 0),
            _ => self.guard_float_operand(exp, 0.0),
        };
        let out = self.create_ir(ir, &[base.clone(), exp_operand]);
        if let (Some(vid), false) = (out.value_id(), formals.is_empty()) {
            self.plant_contract_ensures(contract, vid, &formals);
        }
        out
    }

    /// Euclidean floor division with a producer-certified bit-width on the
    /// quotient/remainder/divisor (see [`crate::ir::IR::FloorDivIPinned`]).
    /// Semantically identical to [`Self::ir_floor_div_i`] — fires the same
    /// `floor_div_i` contract — but the Noir backend can lower it to a cheap
    /// bit-decomposition-range-checked division. `divisor_bits` must satisfy
    /// `2^divisor_bits > max(numerator, divisor)`; the caller guarantees this
    /// structurally (e.g. the reshape source total).
    pub fn ir_floor_div_i_pinned(&mut self, a: &Value, b: &Value, divisor_bits: u32) -> Value {
        let out = self.create_ir(&IR::FloorDivIPinned { divisor_bits }, &[a.clone(), b.clone()]);
        if let (Some(vid), Some(lhs_vid), Some(rhs_vid)) =
            (out.value_id(), a.value_id(), b.value_id())
        {
            let mut formals = HashMap::new();
            formals.insert("lhs".to_string(), lhs_vid);
            formals.insert("rhs".to_string(), rhs_vid);
            self.fire_contract("floor_div_i", vid, &formals);
        }
        out
    }

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
    // `ir_add_f`, `ir_sub_f`, `ir_mul_f`, `ir_div_f` are hand-rolled out
    // of the macro because they deposit a call-site-fact (input-fact-driven
    // interval bound on the output) on top of (or in place of) the
    // registered op contract. Mirrors the int-arith wiring at
    // `ir_add_i / sub_i / mul_i`.
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

    pub fn ir_sum_f(&mut self, cells: &[Value]) -> Value {
        assert!(!cells.is_empty(), "ir_sum_f requires operands");
        if cells.len() == 1 {
            return cells[0].clone();
        }
        if cells.len() == 2 {
            return self.ir_add_f(&cells[0], &cells[1]);
        }
        self.create_ir(
            &IR::SumF {
                len: cells.len() as u32,
            },
            cells,
        )
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

    /// Float product. Interval bounds via input facts (call-site-fact, no
    /// SMT). A factor that is an integer in float form is multiplied
    /// without a rescale, see [`Self::exact_product_with_integer`].
    pub fn ir_mul_f(&mut self, a: &Value, b: &Value) -> Value {
        let out = match self.exact_product_with_integer(a, b) {
            Some(product) => product,
            None => self.create_ir(&IR::MulF, &[a.clone(), b.clone()]),
        };
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

    /// `a * b` for Float operands of which one holds an integer `k`: the
    /// `FloatCast` of an integer cell, or an integral constant inside the
    /// Q32 window. Returns the Float-typed field product `k * raw(x)` of
    /// `k` and the other factor's raw fixed-point value, or `None` when
    /// neither factor is integer-valued (or both are constants, which
    /// folding handles).
    ///
    /// Exactness: `raw(k * x) = k * 2^P * x = k * raw(x)`, so this is the
    /// value `MulF` computes: its rescale divides `raw(k) * raw(x) =
    /// 2^P * k * raw(x)` by `2^P` with remainder zero. The product is a
    /// deterministic function of the operand cells, so no hint is added.
    /// What it omits is the rescale's signed 64-bit window check on the
    /// quotient: a product outside the Q32 window is now the exact field
    /// value instead of having no proof. That is not an obligation of the
    /// multiplication: every consumer that needs a bounded operand
    /// (comparisons, rescales, divisions, casts) checks it itself, as it
    /// must for unchecked circuit inputs.
    fn exact_product_with_integer(&mut self, a: &Value, b: &Value) -> Option<Value> {
        if a.float_val().is_some() && b.float_val().is_some() {
            return None;
        }
        let (k, x) = match self.integer_in_float(a) {
            Some(k) => (k, b),
            None => (self.integer_in_float(b)?, a),
        };
        if !matches!(x, Value::Float(_)) {
            return None;
        }
        match self.create_ir(&IR::MulI, &[k, x.clone()]) {
            Value::Integer(sv) => Some(Value::Float(ScalarValue {
                static_val: None,
                stmt_id: sv.stmt_id,
                value_id: sv.value_id,
            })),
            other => unreachable!("MulI built a non-integer value {:?}", other),
        }
    }

    /// The integer cell behind an integer-valued Float operand: the
    /// argument of the `FloatCast` statement that produced it, or a
    /// constant for an integral float constant `|c| < 2^31` (a Q32 value
    /// in the signed 64-bit window).
    fn integer_in_float(&mut self, v: &Value) -> Option<Value> {
        if !matches!(v, Value::Float(_)) {
            return None;
        }
        if let Some(c) = v.float_val() {
            let in_window = c.abs() < (1u64 << 31) as f64;
            return (c.fract() == 0.0 && in_window).then(|| self.ir_constant_int(c as i64));
        }
        let stmt = self.stmts.get(v.stmt_id()? as usize)?;
        if !matches!(stmt.ir, IR::FloatCast) {
            return None;
        }
        let (arg_stmt, arg_vid) = (stmt.arguments[0], stmt.arg_values[0]);
        let static_val = match self.stmts[arg_stmt as usize].ir {
            IR::ConstantInt { value } => Some(value),
            _ => None,
        };
        Some(Value::Integer(ScalarValue {
            static_val,
            stmt_id: Some(arg_stmt),
            value_id: arg_vid,
        }))
    }

    /// `sum_i a_i * b_i` over Float pairs, rescaled once: `ProductSumF` for
    /// two or more pairs (a product, `0.0` for none). Compile-time operands
    /// fold in f64, as `MulF`/`AddF` fold.
    pub fn ir_dot_f(&mut self, pairs: &[(Value, Value)]) -> Value {
        let folded: Option<f64> = pairs
            .iter()
            .map(|(a, b)| Some(a.float_val()? * b.float_val()?))
            .sum();
        if let Some(sum) = folded {
            return self.ir_constant_float(sum);
        }
        match pairs {
            [] => self.ir_constant_float(0.0),
            [(a, b)] => self.ir_mul_f(a, b),
            _ => self.ir_product_sum_f(pairs),
        }
    }

    /// ProductSumF operands are adjacent Float pairs:
    /// `[lhs0, rhs0, lhs1, rhs1, ...]`, with `len` equal to the pair count.
    /// Every backend computes `floor(sum_i raw(a_i) * raw(b_i) / 2^P)`
    /// (`kernel::fp_product_sum_rescale`): the exact field sum of the raw
    /// products, rescaled once, with the rescale's remainder and window
    /// checks. That is the real sum rounded once, where adding `MulF`
    /// results rounds every product.
    pub fn ir_product_sum_f(&mut self, pairs: &[(Value, Value)]) -> Value {
        let mut args: Vec<Value> = Vec::with_capacity(pairs.len() * 2);
        for (lhs, rhs) in pairs {
            args.push(lhs.clone());
            args.push(rhs.clone());
        }
        self.create_ir(
            &IR::ProductSumF {
                len: pairs.len() as u32,
            },
            &args,
        )
    }

    /// Float division. Fires the registered `div_f` op contract
    /// (`rhs != 0` requires) and, when input facts give bounds and the
    /// divisor interval is sign-stable, also deposits a call-site
    /// interval fact on the output via
    /// [`crate::optim::resolver::interval_fact_for_float_binary`].
    ///
    /// Zero-crossing or zero-touching divisor intervals skip the
    /// interval deposit (would expand to `±inf`); the registered
    /// contract's soundness check still fires.
    pub fn ir_div_f(&mut self, a: &Value, b: &Value) -> Value {
        let formals = Self::binary_formals(a, b);
        self.discharge_contract_requires("div_f", &formals);
        let out = self.create_ir(&IR::DivF, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            self.plant_contract_ensures("div_f", out_vid, &formals);

            if let Some(fact) = crate::optim::resolver::interval_fact_for_float_binary(
                &self.facts,
                crate::optim::predicates::formula::ArithOp::Div,
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
        ir_floor_div_f => IR::FloorDivF,  contract = "floor_div_f",
        ir_mod_f       => IR::ModF,       contract = "mod_f",
    );

    /// Float power. An integral compile-time exponent with
    /// `|e| <= MAX_UNROLLED_POWER` (`x ** 2.0`, `x ** -2.0`) is the exact
    /// fixed-point chain of [`Self::constant_power_f`]; any other exponent
    /// (fractional or runtime) is `PowF`, `exp(b * log a)` on every backend.
    ///
    /// A constant `0.5` is the square root (`SqrtF`: the exact floor root,
    /// whose `x >= 0` requirement leaves no proof for a negative base, where
    /// NumPy gives `nan`; `0 ** 0.5 == 0`).
    pub fn ir_pow_f(&mut self, base: &Value, exp: &Value) -> Value {
        match exp.float_val() {
            Some(e) if e.fract() == 0.0 && e.abs() <= MAX_UNROLLED_POWER as f64 => {
                self.constant_power_f(base, e as i64)
            }
            Some(e) if e == 0.5 => self.ir_sqrt_f(base),
            _ => self.power(&IR::PowF, "pow_f", base, exp),
        }
    }

    /// Float base raised to an Integer exponent. A compile-time exponent with
    /// `|e| <= MAX_UNROLLED_POWER` is the chain of
    /// [`Self::constant_power_f`]; any other is `PowFI`
    /// (`kernel::fp_pow_int`, the same chain over the exponent's bits, run
    /// by every backend), with as many bits as the exponent's proven range
    /// needs ([`Self::exponent_bits`]). The `pow_f` contract's requirement
    /// (`base != 0 or exp >= 0`) is NumPy's `inf` case.
    pub fn ir_pow_f_int(&mut self, base: &Value, exp: &Value) -> Value {
        match exp.int_val() {
            Some(e) if e.abs() <= MAX_UNROLLED_POWER => self.constant_power_f(base, e),
            _ => {
                let exponent_bits = self.exponent_bits(exp);
                self.power(&IR::PowFI { exponent_bits }, "pow_f", base, exp)
            }
        }
    }

    /// Bits of `|e|` for an integer exponent: from its proven range
    /// (`ask_bounds`: facts, preconditions, the dtype bound), else the full
    /// int64 magnitude (64 bits, `|i64::MIN| = 2^63`).
    fn exponent_bits(&mut self, exp: &Value) -> u32 {
        match self.ask_bounds(exp) {
            Some((lo, hi)) => {
                let magnitude = lo.unsigned_abs().max(hi.unsigned_abs());
                (u64::BITS - magnitude.leading_zeros()).max(1)
            }
            None => 64,
        }
    }

    /// `base ** e` for a compile-time integer `e` as float products, so the
    /// mock and every circuit compute the same value: square-and-multiply
    /// (`x ** 6` is three `MulF`, `x ** 0` is `1.0`), and for `e < 0` the
    /// same chain on the fixed-point reciprocal, `(1.0 / base) ** |e|`
    /// (`DivF`, whose `rhs != 0` requirement leaves no proof at `base == 0`,
    /// NumPy's `inf`). Taking the reciprocal first keeps the error at a few
    /// Q units: `1.0 / base ** |e|` would divide by a rounded small power
    /// when `|base| < 1` and amplify its rounding (thousands of units for
    /// `0.3 ** -3`). Each product rounds like any other `MulF`.
    fn constant_power_f(&mut self, base: &Value, e: i64) -> Value {
        if e >= 0 {
            return self.square_and_multiply_f(base, e as u32);
        }
        let one = self.ir_constant_float(1.0);
        let reciprocal = self.ir_div_f(&one, base);
        self.square_and_multiply_f(&reciprocal, e.unsigned_abs() as u32)
    }

    /// `base ** e` by square-and-multiply over the bits of `e`, most
    /// significant first.
    fn square_and_multiply_f(&mut self, base: &Value, e: u32) -> Value {
        if e == 0 {
            return self.ir_constant_float(1.0);
        }
        let mut acc = base.clone();
        for bit in (0..u32::BITS - 1 - e.leading_zeros()).rev() {
            acc = self.ir_mul_f(&acc, &acc);
            if (e >> bit) & 1 == 1 {
                acc = self.ir_mul_f(&acc, base);
            }
        }
        acc
    }
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
        ir_less_than_i_bounded      => IR::LtIBounded, contract = "lt_i",
        ir_greater_than_i_bounded   => IR::GtIBounded, contract = "gt_i",
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
        ir_select_bit_i => IR::SelectBitI,
        ir_select_bit_f => IR::SelectBitF,
    );

    pub fn ir_select_many_i(
        &mut self,
        cond: &Value,
        true_values: &[Value],
        false_values: &[Value],
    ) -> Vec<Value> {
        assert_eq!(
            true_values.len(),
            false_values.len(),
            "select_many_i: branch lengths must match"
        );
        let len = true_values.len() as u32;
        (0..len)
            .map(|lane| {
                let mut args = Vec::with_capacity(1 + true_values.len() + false_values.len());
                args.push(cond.clone());
                args.extend(true_values.iter().cloned());
                args.extend(false_values.iter().cloned());
                self.create_ir(&IR::SelectManyI { len, lane }, &args)
            })
            .collect()
    }

    pub fn ir_select_many_f(
        &mut self,
        cond: &Value,
        true_values: &[Value],
        false_values: &[Value],
    ) -> Vec<Value> {
        assert_eq!(
            true_values.len(),
            false_values.len(),
            "select_many_f: branch lengths must match"
        );
        let len = true_values.len() as u32;
        (0..len)
            .map(|lane| {
                let mut args = Vec::with_capacity(1 + true_values.len() + false_values.len());
                args.push(cond.clone());
                args.extend(true_values.iter().cloned());
                args.extend(false_values.iter().cloned());
                self.create_ir(&IR::SelectManyF { len, lane }, &args)
            })
            .collect()
    }

    // Casting
    ir_unary!(
        ir_bool_cast  => IR::BoolCast, contract = "bool_cast",
    );

    /// Cast to integer.
    ///
    /// The runtime `IR::IntCast` op is float→int (it DEQUANTIZES: cell /
    /// 2^32, truncating). Integer cells are already raw ints and Boolean
    /// cells already hold raw 0/1, so for those inputs this is a pure
    /// type retag with NO IR — emitting the runtime cast would divide
    /// the raw value by the fixed-point scale and truncate everything to
    /// 0 (the `int(flag)` / bool-into-int-array bug class, binding-audit
    /// cases 0325/0462).
    ///
    /// Fires `int_cast_from_bool` contract (Output ∈ {0,1}) when the
    /// input is `Value::Boolean`-typed.
    pub fn ir_int_cast(&mut self, a: &Value) -> Value {
        match a {
            Value::Integer(_) => a.clone(),
            Value::Boolean(sv) => {
                let out = Value::Integer(crate::types::ScalarValue {
                    static_val: sv.static_val.map(|b| if b { 1 } else { 0 }),
                    stmt_id: sv.stmt_id,
                    // Same cell, same identity: facts attached to the
                    // boolean (∈ {0,1}) remain valid for the integer.
                    value_id: sv.value_id,
                });
                self.fire_contract("int_cast_from_bool", sv.value_id, &HashMap::new());
                out
            }
            _ => self.create_ir(&IR::IntCast, &[a.clone()]),
        }
    }

    /// Cast to float. A Float input is already quantized — re-emitting
    /// the runtime cast would multiply by the scale a second time, so
    /// it's a no-op. Integer/Boolean inputs quantize (× 2^32) via the IR
    /// op. Fires `float_cast_from_bool` contract (Output ∈ {0.0, 1.0})
    /// when the input is `Value::Boolean`-typed.
    pub fn ir_float_cast(&mut self, a: &Value) -> Value {
        if matches!(a, Value::Float(_)) {
            return a.clone();
        }
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
        let formals = Self::unary_formals(a);
        self.discharge_contract_requires("arccos_f", &formals);
        let out = self.create_ir(&IR::ArcCosF, &[a.clone()]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            self.plant_contract_ensures("arccos_f", vid, &formals);
            crate::optim::resolver::relay_arccos_output_interval(self, x_vid, vid);
        }
        out
    }

    /// Float `arcsin`. Uses the same real-domain policy as `arccos`:
    /// `-1 <= x <= 1` is discharged if visible, otherwise emitted as a
    /// witness assertion in lenient mode.
    pub fn ir_arcsin_f(&mut self, a: &Value) -> Value {
        let formals = Self::unary_formals(a);
        self.discharge_contract_requires("arcsin_f", &formals);
        let out = self.create_ir(&IR::ArcSinF, &[a.clone()]);
        if let Some(vid) = out.value_id() {
            self.plant_contract_ensures("arcsin_f", vid, &formals);
        }
        out
    }

    /// Float `log`. Fires the `log_f` op contract so the input value's
    /// `requires(x > 0.0)` precondition is discharged against its fact
    /// bucket (formal `"x"` ↔ input ValueId), then relays any visible
    /// input interval `[lo, hi]` (with `lo > 0`) into
    /// `[log(lo), log(hi)]` on the output.
    pub fn ir_log_f(&mut self, a: &Value) -> Value {
        // The `log_f` contract checks `a > 0` where the code runs; the
        // lowering itself needs a positive operand on every path (Noir's
        // `qlog` asserts it), so where the code does not run it gets 1.0.
        let formals = Self::unary_formals(a);
        self.discharge_contract_requires("log_f", &formals);
        let operand = self.guard_float_operand(a, 1.0);
        let out = self.create_ir(&IR::LogF, &[operand]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            self.plant_contract_ensures("log_f", vid, &formals);
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

    /// `floor(sqrt(b * c) * 2^P)` of two integers as a float
    /// (`IR::SqrtProductIF`): no proof unless `0 <= b < 2^64` and
    /// `0 <= b * c < 2^(126 - 2P)`. For lowerings that need the root of an
    /// integer beyond the float window (`lower::stats`).
    pub fn ir_sqrt_product_i_f(&mut self, b: &Value, c: &Value) -> Value {
        self.create_ir(&IR::SqrtProductIF, &[b.clone(), c.clone()])
    }

    /// Float `sqrt`. Fires the `sqrt_f` op contract so the output value
    /// carries the `Output >= 0.0` fact for downstream resolvers, and so
    /// the `requires(x >= 0.0)` precondition is discharged against the
    /// input value's fact bucket (formal `"x"` ↔ input ValueId).
    pub fn ir_sqrt_f(&mut self, a: &Value) -> Value {
        let formals = Self::unary_formals(a);
        self.discharge_contract_requires("sqrt_f", &formals);
        let out = self.create_ir(&IR::SqrtF, &[a.clone()]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            self.plant_contract_ensures("sqrt_f", vid, &formals);
            crate::optim::resolver::relay_sqrt_output_interval(self, x_vid, vid);
        }
        out
    }

    /// The formal binding `x -> a` of a unary op contract (empty for a
    /// value without an id, e.g. a literal).
    fn unary_formals(a: &Value) -> HashMap<String, crate::types::ValueId> {
        a.value_id().map(|vid| HashMap::from([("x".to_string(), vid)])).unwrap_or_default()
    }

    /// The formal bindings `lhs -> a`, `rhs -> b` of a binary op contract
    /// (empty unless both operands have ids, as contracts fire only then).
    fn binary_formals(a: &Value, b: &Value) -> HashMap<String, crate::types::ValueId> {
        match (a.value_id(), b.value_id()) {
            (Some(lhs), Some(rhs)) => HashMap::from([("lhs".to_string(), lhs), ("rhs".to_string(), rhs)]),
            _ => HashMap::new(),
        }
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

    /// Float `arctan2`. Fires the `arctan2_f` op contract so the output
    /// value carries the static `[-π - slack, π + slack]` fact for
    /// downstream resolvers, then relays a tighter sign-stable interval
    /// bound from the inputs' visible facts when the bounding box does
    /// not straddle the negative-x-axis discontinuity.
    ///
    /// Hand-rolled out of the [`ir_binary!`] macro for the same reason
    /// as [`Self::ir_div_f`] — the relay deposit needs the call site to
    /// thread input/output `ValueId`s into
    /// [`crate::optim::resolver::relay_arctan2_output_interval`].
    pub fn ir_arctan2_f(&mut self, a: &Value, b: &Value) -> Value {
        let out = self.create_ir(&IR::ArcTan2F, &[a.clone(), b.clone()]);
        if let (Some(out_vid), Some(y_vid), Some(x_vid)) =
            (out.value_id(), a.value_id(), b.value_id())
        {
            let mut formals = HashMap::new();
            formals.insert("lhs".to_string(), y_vid);
            formals.insert("rhs".to_string(), x_vid);
            self.fire_contract("arctan2_f", out_vid, &formals);

            crate::optim::resolver::relay_arctan2_output_interval(self, y_vid, x_vid, out_vid);
        }
        out
    }

    // Assert & expose. The assert-emitting methods (`ir_assert`,
    // `ir_assert_bit`, `ir_assert_non_neg_i`, `ir_assert_u128_lt`,
    // `ir_assert_u128_lt_bits`) live in `builder/guard.rs`: they emit every
    // check conditional on the execution predicate.

    pub fn ir_expose_public_i(&mut self, a: &Value) -> Value {
        self.elided_before_exposure.push(self.elided_publics.len());
        self.create_ir(&IR::ExposePublicI, &[a.clone()])
    }

    pub fn ir_expose_public_f(&mut self, a: &Value) -> Value {
        self.elided_before_exposure.push(self.elided_publics.len());
        self.create_ir(&IR::ExposePublicF, &[a.clone()])
    }

    /// Place the elided constants among the output slots: exposure `k`
    /// occupies slot `k + (constants elided before it)`; the constants
    /// fill the remaining slots in order.
    fn elided_output_slots(&self) -> Vec<crate::ir::ElidedOutputSlot> {
        let total = self.elided_before_exposure.len() + self.elided_publics.len();
        let exposed: std::collections::HashSet<usize> = self
            .elided_before_exposure
            .iter()
            .enumerate()
            .map(|(k, before)| k + before)
            .collect();
        (0..total)
            .filter(|slot| !exposed.contains(slot))
            .zip(&self.elided_publics)
            .map(|(slot, value)| crate::ir::ElidedOutputSlot {
                slot,
                value: value.clone(),
            })
            .collect()
    }

    /// Multi-axis bounded-range mask. Returns Field 0/1 = 1 iff for
    /// every axis k: `lo_k <= idx_k < hi_k`. `triples` is a slice of
    /// `(idx_k, lo_k, hi_k)`; arguments are laid out flat as
    /// `[idx_0, lo_0, hi_0, idx_1, lo_1, hi_1, ...]`. Caller asserts
    /// every operand is non-negative and fits u128 (same contract as
    /// `LtIBounded` — see `IR::IndexInBox`). Used by the fused
    /// multi-axis dynamic slice path to emit one combined boolean
    /// check per output cell instead of the per-axis
    /// `LtIBounded + LogicalAnd` chain.
    pub fn ir_index_in_box(&mut self, triples: &[(Value, Value, Value)]) -> Value {
        let rank = triples.len();
        let mut args: Vec<Value> = Vec::with_capacity(rank * 3);
        for (idx, lo, hi) in triples {
            args.push(idx.clone());
            args.push(lo.clone());
            args.push(hi.clone());
        }
        self.create_ir(&IR::IndexInBox { rank }, &args)
    }

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
        self.create_ir(
            &IR::StructuralPredicate {
                kind,
                args,
                op,
                bound,
            },
            &[],
        )
    }

    /// Emit an `IR::ScalarPrecondition` atom carrying a serialized
    /// `ContractTerm` payload. Takes no stack operands. Returns
    /// `Value::None`.
    pub fn ir_scalar_precondition(&mut self, term_json: String) -> Value {
        self.create_ir(&IR::ScalarPrecondition { term_json }, &[])
    }

    // ── I/O ───────────────────────────────────────────────────────────

    pub fn ir_read_integer(
        &mut self,
        path: crate::circuit_input::InputPath,
        is_public: bool,
    ) -> Value {
        self.create_ir(&IR::ReadInteger { path, is_public }, &[])
    }

    pub fn ir_read_float(
        &mut self,
        path: crate::circuit_input::InputPath,
        is_public: bool,
    ) -> Value {
        self.create_ir(&IR::ReadFloat { path, is_public }, &[])
    }

    pub fn ir_read_hash(
        &mut self,
        path: crate::circuit_input::InputPath,
        is_public: bool,
    ) -> Value {
        self.create_ir(&IR::ReadHash { path, is_public }, &[])
    }

    pub fn ir_read_external_result(
        &mut self,
        store_idx: u32,
        output_idx: u32,
        is_float: bool,
    ) -> Value {
        self.create_ir(
            &IR::ReadExternalResult {
                store_idx,
                output_idx,
                is_float,
            },
            &[],
        )
    }

    // ── Memory ────────────────────────────────────────────────────────

    pub fn ir_allocate_memory(&mut self, segment_id: u32, size: u32, init_value: i64) -> Value {
        // A segment allocated while a conditional-write predicate is
        // active is branch-local: its contents are unobservable on the
        // not-taken path (it only escapes through a cond-gated merge), so
        // writes to it under the SAME predicate skip the read+select
        // predication. Keyed by the predicate's stmt identity — a nested
        // branch has a different (composed) predicate and still pays.
        if let Some(p) = &self.write_predicate {
            if let Some(pid) = p.stmt_id() {
                self.predicate_exempt_segments.insert(segment_id, pid);
            }
        }
        self.segment_capacity.insert(segment_id, size);
        self.create_ir(
            &IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            },
            &[],
        )
    }

    /// Current conditional-write predicate (`None` = unconditional).
    pub fn write_predicate(&self) -> Option<&Value> {
        self.write_predicate.as_ref()
    }

    /// Swap the conditional-write predicate, returning the previous one so
    /// callers can scope it (`let prev = swap(..); ...; swap(prev);`).
    /// The predicate passed in must already incorporate any enclosing
    /// branch conditions (IRGen uses the scope's calculated branching
    /// condition, which self-composes across nested `if`s).
    ///
    /// The predicate is also the execution predicate of every check emitted
    /// meanwhile and the guard of every fact learned meanwhile (see
    /// `builder/guard.rs`).
    pub fn swap_write_predicate(&mut self, p: Option<Value>) -> Option<Value> {
        let previous = std::mem::replace(&mut self.write_predicate, p);
        self.sync_fact_guard();
        previous
    }

    /// Resolve the write predicate to `Some(static bool)` when it's a
    /// compile-time constant, `None` when it's a runtime value.
    fn write_predicate_static(&self) -> Option<bool> {
        match &self.write_predicate {
            None => Some(true),
            Some(p) => p.bool_val().or_else(|| p.int_val().map(|v| v != 0)),
        }
    }

    /// Apply the conditional-write predicate to a value about to be stored
    /// at `(segment_id, address)`: returns `select(p, new, old)` (typed to
    /// match `value`), or `None` when the write is statically dead.
    fn apply_write_predicate(
        &mut self,
        segment_id: u32,
        address: &Value,
        value: &Value,
    ) -> Option<Value> {
        match self.write_predicate_static() {
            Some(true) => Some(value.clone()),
            Some(false) => None,
            None => {
                let p = self.write_predicate.clone().unwrap();
                // Branch-local segment under its own predicate: writes
                // are unobservable on the not-taken path; skip the
                // read+select.
                if p.stmt_id().is_some()
                    && self.predicate_exempt_segments.get(&segment_id) == p.stmt_id().as_ref()
                {
                    return Some(value.clone());
                }
                let old_raw = self.ir_read_memory(segment_id, address);
                Some(match value {
                    Value::Float(_) => {
                        // Storage cells are integer-typed at the IR layer;
                        // re-tag the old cell as Float so the select keeps
                        // float semantics (mirrors `segment::read_typed`).
                        let old_f = Value::Float(crate::types::ScalarValue::new(
                            None,
                            old_raw.stmt_id(),
                        ));
                        self.ir_select_f(&p, value, &old_f)
                    }
                    _ => {
                        let stored = self.ir_select_i(&p, value, &old_raw);
                        // A bool written into a bool array stays a bool.
                        if self.is_bool_cell(value) && self.is_bool_cell(&old_raw) {
                            self.record_bool_cell(&stored);
                        }
                        stored
                    }
                })
            }
        }
    }

    /// Write a value into a memory segment. Returns the value actually
    /// stored — equal to `value` for unconditional writes, or the
    /// predicated `select(p, value, old)` under a conditional-write
    /// predicate. Callers that mirror writes into compile-time caches
    /// (`static_array_payload`) must cache the RETURNED value, not the
    /// input, or the cached view diverges from the segment on the
    /// not-taken path.
    pub fn ir_write_memory(&mut self, segment_id: u32, address: &Value, value: &Value) -> Value {
        let capacity = self.segment_capacity.get(&segment_id).copied();
        let address = &self.guard_address(address, capacity);
        let Some(stored) = self.apply_write_predicate(segment_id, address, value) else {
            // Statically-false predicate: the write can never happen.
            return value.clone();
        };
        let value = &stored;
        // A write invalidates slice views of this segment and any views
        // reading through it as their base, and bumps the segment's
        // write version (provenance invariant #4). The invalidation now
        // lives in `deposit_provenance`'s `WriteMemory | …SetItem` arm —
        // the chokepoint that sees every write-shaped op — so the
        // `create_ir` call below triggers it. (Done there, not here, so
        // `ir_dynamic_ndarray_set_item` gets the same invalidation.)
        //
        // Per-cell bound invalidation (provenance invariant #4 applied to
        // the builder-owned `segment_cell_bounds` / `segment_default_bounds`
        // maps, which `on_segment_write` can't reach): a static-addr write
        // must drop the OLD cell's bound before the conditional re-capture
        // below (otherwise a re-write that doesn't re-capture leaves the
        // stale bound, and a later static read republishes it as a fact
        // about the NEW value). A dynamic-addr write could land on any
        // cell, so it clears the whole per-cell map and the segment-wide
        // default bound.
        match address.int_val() {
            Some(addr_i) => {
                if let Some(cells) = self.segment_cell_bounds.get_mut(&segment_id) {
                    cells.remove(&(addr_i as usize));
                }
            }
            None => {
                self.segment_cell_bounds.remove(&segment_id);
                self.segment_default_bounds.remove(&segment_id);
            }
        }
        // Capture per-cell bound facts at write time so a downstream
        // static-addr `ir_read_memory` of this cell can republish them on
        // the read output. Fires only for statically-pinned addresses
        // (e.g. `alloc_and_write_values` always uses constant addresses;
        // `np_nonzero` writes each cell at the runtime `true_count`
        // address, which doesn't have a static address and is skipped).
        // Fast gate: only call the (potentially-expensive) `ask_bounds`
        // when there's a plausible source of a fact-derived bound —
        // either a direct fact on `val.value_id()`, or `val` is an
        // `AddI`/`SubI` whose non-constant operand has a fact (the
        // `read + 1` pattern downstream of fact-aware reads). Avoids the
        // resolver walk on every cell of a 20K-cell array write (e.g.
        // patch-reshape 0390 on Float[100,200]).
        // Not under a runtime execution predicate: a bound learned there may
        // hold only where the code runs, while the captured cell bound is
        // republished to every later reader.
        let runs_everywhere = matches!(self.execution(), guard::Execution::Always);
        if let Some(addr_i) = address.int_val().filter(|_| runs_everywhere) {
            let should_capture = value.value_id().is_some_and(|vid| {
                if self.facts.has_facts_for(vid) {
                    return true;
                }
                if let Some(stmt_id) = value.stmt_id() {
                    if let Some(stmt) = self.stmts.get(stmt_id as usize) {
                        use crate::ir_defs::IR;
                        if matches!(stmt.ir, IR::AddI | IR::SubI) && stmt.arg_values.len() == 2 {
                            for &arg_vid in &stmt.arg_values {
                                if self.facts.has_facts_for(arg_vid) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            });
            if should_capture {
                if let Some(bounds) = self.ask_bounds(value) {
                    self.segment_cell_bounds
                        .entry(segment_id)
                        .or_default()
                        .insert(addr_i as usize, bounds);
                }
            }
        }
        self.create_ir(
            &IR::WriteMemory { segment_id },
            &[address.clone(), value.clone()],
        );
        stored
    }

    /// Read a memory cell. Under a runtime execution predicate a dynamic
    /// address that is not in the segment on every path is guarded (see
    /// `guard_address`): where the code does not run it reads cell 0.
    /// `IR::ReadTable`: the segment's slot at `address`, with no witness for
    /// an address outside the segment (the caller passes an index that is in
    /// range wherever the read does not run, as the frontend's guarded
    /// indices are). The segment must not be written afterwards.
    pub fn ir_read_table(&mut self, segment_id: u32, address: &Value) -> Value {
        let out = self.create_ir(&IR::ReadTable { segment_id }, &[address.clone()]);
        if self.is_bool_segment(segment_id) {
            self.record_bool_cell(&out);
        }
        out
    }

    pub fn ir_read_memory(&mut self, segment_id: u32, address: &Value) -> Value {
        let capacity = self.segment_capacity.get(&segment_id).copied();
        let address = &self.guard_address(address, capacity);
        let out = self.create_ir(&IR::ReadMemory { segment_id }, &[address.clone()]);
        if self.is_bool_segment(segment_id) {
            self.record_bool_cell(&out);
        }
        // Republish cell-bound facts captured at write time. Only fires for
        // statically-pinned addresses (the typical `B.min(0)[0]` pattern,
        // where the tuple-unpack index is a literal); dynamic reads keep
        // discharging via the existing bound-assert path.
        if let Some(addr_i) = address.int_val() {
            let cell_bound = self
                .segment_cell_bounds
                .get(&segment_id)
                .and_then(|m| m.get(&(addr_i as usize)))
                .copied();
            let default_bound = self.segment_default_bounds.get(&segment_id).copied();
            if let Some((lo, hi)) = cell_bound.or(default_bound) {
                if let Some(out_vid) = out.value_id() {
                    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
                    // Publish as two separate Cmp facts, not a combined
                    // BoolComb(And, ...). `derive_bounds_from_facts` only
                    // reads single Cmp shapes — a combined And-of-Cmps
                    // wouldn't match its pattern and the bound would be
                    // invisible. Mirrors how the registered contract
                    // ensures lists deposit one fact per formula.
                    let lo_fact = ContractTerm::Cmp {
                        op: CmpOp::Ge,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                        rhs: Box::new(ContractTerm::LitInt(lo)),
                    };
                    let hi_fact = ContractTerm::Cmp {
                        op: CmpOp::Le,
                        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                        rhs: Box::new(ContractTerm::LitInt(hi)),
                    };
                    self.facts.insert_for(out_vid, lo_fact);
                    self.facts.insert_for(out_vid, hi_fact);
                }
            }
        }
        out
    }

    // ── Dynamic NDArray ───────────────────────────────────────────────

    pub fn ir_allocate_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        dtype_name: String,
        max_length: u32,
        max_rank: u32,
        max_shape: Vec<u32>,
    ) -> Value {
        self.dyn_array_capacity.insert(array_id, max_length);
        self.create_ir(
            &IR::AllocateDynamicNDArrayMeta {
                array_id,
                dtype_name,
                max_length,
                max_rank,
                max_shape,
            },
            &[],
        )
    }

    pub fn ir_witness_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        param_name: String,
        args: &[Value],
    ) -> Value {
        self.create_ir(
            &IR::WitnessDynamicNDArrayMeta {
                array_id,
                max_rank,
                param_name,
            },
            args,
        )
    }

    pub fn ir_dynamic_ndarray_get_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        address: &Value,
    ) -> Value {
        let capacity = self.dyn_array_capacity.get(&array_id).copied();
        let address = self.guard_address(address, capacity);
        self.create_ir(
            &IR::DynamicNDArrayGetItem {
                array_id,
                segment_id,
            },
            &[address],
        )
    }

    pub fn ir_dynamic_ndarray_set_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        address: &Value,
        value: &Value,
    ) -> Value {
        // Same conditional-write semantics as `ir_write_memory`.
        let capacity = self.dyn_array_capacity.get(&array_id).copied();
        let address = &self.guard_address(address, capacity);
        let Some(stored) = self.apply_write_predicate(segment_id, address, value) else {
            return value.clone();
        };
        // Invalidate the builder-owned per-cell bound maps. A set_item is a
        // mutation that can change the addressed cell's value; we never
        // capture per-cell bounds for set_item, so the safe move is to drop
        // the whole segment's per-cell map and its default bound (otherwise
        // a later static read republishes a pre-set_item bound as a fact
        // about the post-set_item value — the `value_ops.rs` swap-and-read
        // shape). Segment-version / slice / mask-read invalidation is
        // handled by `deposit_provenance` when `create_ir` builds the op.
        self.segment_cell_bounds.remove(&segment_id);
        self.segment_default_bounds.remove(&segment_id);
        self.create_ir(
            &IR::DynamicNDArraySetItem {
                array_id,
                segment_id,
            },
            &[address.clone(), stored.clone()],
        );
        stored
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
            &IR::InvokeExternal {
                store_idx,
                func_name,
                args,
                kwargs,
            },
            &[],
        )
    }

    pub fn ir_poseidon_hash(&mut self, values: &[Value]) -> Value {
        self.create_ir(&IR::PoseidonHash, values)
    }

    /// `IR::Hint`: lane `rank` of `kind(elements)`, a value the prover
    /// supplies. It constrains nothing; the caller must pin it (see
    /// `lower::sort`).
    pub fn ir_hint(
        &mut self,
        kind: crate::ir_defs::HintKind,
        elements: &[Value],
        rank: &Value,
    ) -> Value {
        let mut args = elements.to_vec();
        args.push(rank.clone());
        self.create_ir(&IR::Hint { kind, len: elements.len() as u32 }, &args)
    }

    /// `IR::ArrayOp`: one lane of an array operator, over its shared
    /// operands `shared` (the same `Arc` for every lane of one instance:
    /// that is how the layer groups them) and the lane's own operands.
    pub fn ir_array_op(&mut self, op: crate::lower::ArrayOp, shared: &std::sync::Arc<[Value]>, own: &[Value]) -> Value {
        debug_assert_eq!(shared.len(), op.shared as usize, "ArrayOp: shared operand count");
        let mut args = shared.to_vec();
        args.extend(own.iter().cloned());
        let out = self.create_ir(&IR::ArrayOp(op.clone()), &args);
        if let (Some(stmt), Some(_)) = (out.stmt_id(), op.position) {
            let placeholder = ArrayOpPlaceholder { op, shared: shared.clone() };
            self.array_op_placeholders.insert(stmt, placeholder);
        }
        out
    }

    /// The placeholder behind `value`, when `value` is a lane of an array
    /// operator at a constant position.
    pub fn array_op_placeholder_of(&self, value: &Value) -> Option<&ArrayOpPlaceholder> {
        self.array_op_placeholder_at(value.stmt_id()?)
    }

    /// The placeholder statement `stmt` is, when it is a lane of an array
    /// operator at a constant position.
    pub fn array_op_placeholder_at(&self, stmt: StmtId) -> Option<&ArrayOpPlaceholder> {
        self.array_op_placeholders.get(&stmt)
    }

    /// Output `rank` of NumPy's stable sort of `elements` (the value, or with
    /// `index` its original position): a placeholder `lower::sort` lowers.
    pub fn ir_sorted(&mut self, elements: &std::sync::Arc<[Value]>, rank: u32, index: bool) -> Value {
        let op = Self::sort_lane(elements.len(), Some(rank), index);
        let out = self.ir_array_op(op, elements, &[]);
        self.publish_sorted_index_bounds(&out, index, elements.len());
        out
    }

    fn sort_lane(len: usize, position: Option<u32>, index: bool) -> crate::lower::ArrayOp {
        use crate::lower::sort::{INDICES, VALUES};
        crate::lower::ArrayOp {
            kind: crate::lower::OpKind::Sort,
            shared: len as u32,
            output: if index { INDICES } else { VALUES },
            position,
        }
    }

    /// An argsort output lies in `[0, len)` (as an `argmax` result does):
    /// published so that indexing with it needs no bounds check.
    fn publish_sorted_index_bounds(&mut self, out: &Value, index: bool, len: usize) {
        let Some(vid) = out.value_id().filter(|_| index) else { return };
        let len_value = self.ir_constant_int(len as i64);
        if let Some(len_vid) = len_value.value_id() {
            let mut formals = HashMap::new();
            formals.insert("len_arr".to_string(), len_vid);
            self.fire_contract("dyn_argextremum", vid, &formals);
        }
    }

    /// [`Self::ir_sorted`] at a runtime rank (a constant one takes the
    /// static form). A rank outside `[0, len)` has no proof.
    pub fn ir_sorted_at_runtime_rank(
        &mut self,
        elements: &std::sync::Arc<[Value]>,
        rank: &Value,
        index: bool,
    ) -> Value {
        if let Some(r) = rank.int_val().and_then(|r| u32::try_from(r).ok()) {
            if (r as usize) < elements.len() {
                return self.ir_sorted(elements, r, index);
            }
        }
        let op = Self::sort_lane(elements.len(), None, index);
        let out = self.ir_array_op(op, elements, std::slice::from_ref(rank));
        self.publish_sorted_index_bounds(&out, index, elements.len());
        out
    }

    /// `IR::KeyLess`: `[(v1, i1) < (v2, i2)]` lexicographically, for
    /// indices the caller guarantees in `[0, 2^index_bits)`.
    pub fn ir_key_less(&mut self, v1: &Value, i1: &Value, v2: &Value, i2: &Value, index_bits: u32) -> Value {
        self.create_ir(
            &IR::KeyLess { index_bits },
            &[v1.clone(), i1.clone(), v2.clone(), i2.clone()],
        )
    }

    /// `IR::PackKey`: the packed sort key `v * 2^index_bits + i`.
    pub fn ir_pack_key(&mut self, v: &Value, i: &Value, index_bits: u32) -> Value {
        self.create_ir(&IR::PackKey { index_bits }, &[v.clone(), i.clone()])
    }

    /// `IR::PackedKeyLess`: `[K1 < K2]` for packed keys.
    pub fn ir_packed_key_less(&mut self, k1: &Value, k2: &Value, index_bits: u32) -> Value {
        self.create_ir(&IR::PackedKeyLess { index_bits }, &[k1.clone(), k2.clone()])
    }

    /// `IR::PackedKeyIndex`: the index a packed key was packed from.
    pub fn ir_packed_key_index(&mut self, key: &Value, index_bits: u32) -> Value {
        self.create_ir(&IR::PackedKeyIndex { index_bits }, std::slice::from_ref(key))
    }

    /// `IR::IdentityCheck`: assert `identity` among `operands` (checked at
    /// verifier challenges on halo2-base; see `lower::challenge`).
    pub fn ir_identity_check(&mut self, identity: crate::ir_defs::Identity, operands: &[Value]) {
        assert_eq!(operands.len(), identity.operands(), "{} operands", identity.describe());
        self.create_ir(&IR::IdentityCheck(identity), operands);
    }

    /// `IR::RescaleRawF`: `floor(raw / 2^P)` of a raw product sum.
    pub fn ir_rescale_raw_f(&mut self, raw: &Value) -> Value {
        self.create_ir(&IR::RescaleRawF, std::slice::from_ref(raw))
    }

    /// Start allocating memory segment ids at `next` (a pass that rebuilds a
    /// graph and adds segments must not reuse the graph's ids).
    pub fn reserve_segment_ids_below(&mut self, next: u32) {
        self.next_segment_id = self.next_segment_id.max(next);
    }
}

// ---------------------------------------------------------------------------
// build_ir — the Rust equivalent of AbstractIR.build_ir()
// ---------------------------------------------------------------------------

/// Build an IR statement and compute the result value.
/// This implements the `build_ir(ir_id, args, dbg)` method for all 79 IR types.
fn build_value(ir: &IR, ir_id: StmtId, args: &[Value]) -> Value {
    if args.len() < min_required_arity(ir) {
        return unknown_value_for_ir(ir, ir_id);
    }

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
        // `DivI` is exact division (a field inverse in the backends): fold
        // only when the division is exact, where both readings agree.
        IR::DivI => int_binary_ir(ir_id, args, |a, b| {
            if b != 0 && a % b == 0 {
                Some(a / b)
            } else {
                None
            }
        }),
        // Same definition as the mock and the circuits
        // (`kernel::div_rem_or_zero`, Python floor semantics: `-7 // 2 ==
        // -4`, `7 % -2 == -1`). A zero divisor is left unfolded. Folding
        // used `div_euclid` and Rust's truncating `%`, so a constant
        // operand gave a different value from a runtime one.
        IR::FloorDivI | IR::FloorDivIPinned { .. } => int_binary_ir(ir_id, args, |a, b| {
            python_floor_div_rem(a, b).map(|(q, _)| q)
        }),
        IR::ModI => int_binary_ir(ir_id, args, |a, b| python_floor_div_rem(a, b).map(|(_, r)| r)),
        IR::PowI => int_binary_ir(ir_id, args, |a, b| {
            if b >= 0 {
                Some(a.pow(b as u32))
            } else {
                None
            }
        }),

        // ── Integer bitwise ───────────────────────────────────────
        IR::BitAndI => int_binary_ir(ir_id, args, |a, b| Some(a & b)),
        IR::BitOrI => int_binary_ir(ir_id, args, |a, b| Some(a | b)),
        IR::BitXorI => int_binary_ir(ir_id, args, |a, b| Some(a ^ b)),
        // NumPy's int64 shifts (`kernel::int64_shl`, `int64_shr`).
        IR::ShlI => int_binary_ir(ir_id, args, |a, b| Some(crate::prove::kernel::int64_shl(a, b))),
        IR::ShrI => int_binary_ir(ir_id, args, |a, b| Some(crate::prove::kernel::int64_shr(a, b))),
        IR::BitNotI => int_unary_ir(ir_id, args, |a| Some(!a)),

        // ── Integer unary arithmetic ──────────────────────────────
        // `|i64::MIN|` overflows: not folded (it has no proof).
        IR::AbsI => int_unary_ir(ir_id, args, |a| a.checked_abs()),
        IR::SignI => int_unary_ir(ir_id, args, |a| {
            Some(if a > 0 {
                1
            } else if a < 0 {
                -1
            } else {
                0
            })
        }),
        IR::InvI => Value::Integer(ScalarValue::new(None, Some(ir_id))),

        // ── Float binary arithmetic ───────────────────────────────
        IR::AddF => float_binary_ir(ir_id, args, |a, b| a + b),
        IR::SumF { .. } => {
            let mut acc = 0.0;
            for arg in args {
                match arg.float_val() {
                    Some(v) => acc += v,
                    None => return Value::Float(ScalarValue::new(None, Some(ir_id))),
                }
            }
            Value::Float(ScalarValue::new(Some(acc), Some(ir_id)))
        }
        IR::SubF => float_binary_ir(ir_id, args, |a, b| a - b),
        IR::MulF => float_binary_ir(ir_id, args, |a, b| a * b),
        IR::ProductSumF { .. } => {
            // ProductSumF semantics depend on proving-time `precision_bits`
            // because each pair uses fixed-point `MulF` rescaling. The builder
            // does not know that parameter, so avoid publishing a potentially
            // misleading compile-time float value here.
            Value::Float(ScalarValue::new(None, Some(ir_id)))
        }
        IR::DivF => float_binary_ir(ir_id, args, |a, b| a / b),
        IR::FloorDivF => float_binary_ir(ir_id, args, |a, b| (a / b).floor()),
        // Python/NumPy float remainder: the result has the divisor's sign
        // (`kernel::fp_float_mod`); Rust's `%` keeps the dividend's sign.
        IR::ModF => float_binary_ir(ir_id, args, |a, b| {
            let r = a % b;
            if r != 0.0 && (r < 0.0) != (b < 0.0) {
                r + b
            } else {
                r
            }
        }),
        IR::PowF => float_binary_ir(ir_id, args, |a, b| a.powf(b)),
        IR::PowFI { .. } => {
            // Float base ^ Integer exponent. Constant-fold from
            // (float_val(args[0]), int_val(args[1])).
            let la = args[0].float_val();
            let lb = args[1].int_val();
            let inferred = match (la, lb) {
                (Some(a), Some(b)) => Some(a.powi(b as i32)),
                _ => None,
            };
            Value::Float(ScalarValue::new(inferred, Some(ir_id)))
        }

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
        IR::LtIBounded => int_cmp_ir(ir_id, args, |a, b| a < b),
        IR::GtIBounded => int_cmp_ir(ir_id, args, |a, b| a > b),
        IR::NeI => int_cmp_ir(ir_id, args, |a, b| a != b),
        IR::LtI => int_cmp_ir(ir_id, args, |a, b| a < b),
        IR::LteI => int_cmp_ir(ir_id, args, |a, b| a <= b),
        IR::GtI => int_cmp_ir(ir_id, args, |a, b| a > b),
        IR::GteI => int_cmp_ir(ir_id, args, |a, b| a >= b),
        IR::AssertU128LtBit { hi, bit, .. } => {
            let inferred = args[0].int_val().and_then(|idx| {
                if idx >= 0 && (idx as u64) < *hi {
                    Some((((idx as u64) >> bit) & 1) != 0)
                } else {
                    None
                }
            });
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }

        IR::IndexInBox { rank } => {
            // Args laid out flat: [idx_0, lo_0, hi_0, idx_1, lo_1, hi_1, ...].
            // Fold to a constant boolean if every operand is known.
            let mut all_in = Some(true);
            for k in 0..*rank {
                let idx = args[k * 3].int_val();
                let lo = args[k * 3 + 1].int_val();
                let hi = args[k * 3 + 2].int_val();
                match (idx, lo, hi) {
                    (Some(i), Some(l), Some(h)) => {
                        if !(i >= l && i < h) {
                            all_in = Some(false);
                            break;
                        }
                    }
                    _ => {
                        all_in = None;
                        break;
                    }
                }
            }
            Value::Boolean(ScalarValue::new(all_in, Some(ir_id)))
        }

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
        // Not folded: the f64 root of a large radicand is not the floor root.
        IR::SqrtProductIF => Value::Float(ScalarValue::new(None, Some(ir_id))),
        IR::ExpF => float_unary_ir(ir_id, args, |a| a.exp()),
        IR::LogF => float_unary_ir(ir_id, args, |a| a.ln()),
        IR::ArcSinF => float_unary_ir(ir_id, args, |a| a.asin()),
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
        IR::SelectI | IR::SelectBitI => {
            let cond = args[0]
                .bool_val()
                .or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].int_val(),
                Some(false) => args[2].int_val(),
            };
            Value::Integer(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::SelectF | IR::SelectBitF => {
            let cond = args[0]
                .bool_val()
                .or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args[1].float_val(),
                Some(false) => args[2].float_val(),
            };
            Value::Float(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::SelectManyI { len, lane } => {
            let len = *len as usize;
            let lane = *lane as usize;
            let cond = args[0]
                .bool_val()
                .or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args.get(1 + lane).and_then(Value::int_val),
                Some(false) => args.get(1 + len + lane).and_then(Value::int_val),
            };
            Value::Integer(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::SelectManyF { len, lane } => {
            let len = *len as usize;
            let lane = *lane as usize;
            let cond = args[0]
                .bool_val()
                .or_else(|| args[0].int_val().map(|v| v != 0));
            let inferred = match cond {
                None => None,
                Some(true) => args.get(1 + lane).and_then(Value::float_val),
                Some(false) => args.get(1 + len + lane).and_then(Value::float_val),
            };
            Value::Float(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::SelectB => {
            let cond = args[0]
                .bool_val()
                .or_else(|| args[0].int_val().map(|v| v != 0));
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
            val: args[0].int_val().map(|v| v.to_string()).unwrap_or_default(),
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
        IR::ReadExternalResult { is_float, .. } => {
            if *is_float {
                Value::Float(ScalarValue::new(None, Some(ir_id)))
            } else {
                Value::Integer(ScalarValue::new(None, Some(ir_id)))
            }
        }
        IR::Print => Value::None,

        // ── Assert & expose (fixed, return None) ──────────────────
        IR::Assert | IR::AssertBit | IR::AssertNonNegI | IR::AssertU128Lt => Value::None,
        IR::StructuralPredicate { .. } => Value::None,
        IR::ScalarPrecondition { .. } => Value::None,
        IR::ExposePublicI | IR::ExposePublicF => Value::None,

        // ── Memory operations (fixed) ─────────────────────────────
        IR::AllocateMemory { .. } => Value::None,
        IR::WriteMemory { .. } => Value::None,
        IR::ReadMemory { .. } | IR::ReadTable { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::MemoryTraceEmit { .. } => Value::None,
        IR::MemoryTraceSeal => Value::None,

        // ── Dynamic NDArray (fixed) ───────────────────────────────
        IR::AllocateDynamicNDArrayMeta { .. } => Value::None,
        IR::WitnessDynamicNDArrayMeta { .. } | IR::AssertDynamicNDArrayMeta { .. } => Value::None,
        IR::DynamicNDArrayGetItem { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::DynamicNDArraySetItem { .. } => Value::None,

        // ── External calls (fixed) ────────────────────────────────
        IR::InvokeExternal { .. } => Value::None,
        IR::ExportExternalI { .. } | IR::ExportExternalF { .. } => Value::None,

        // ── Hash ──────────────────────────────────────────────────
        IR::PoseidonHash => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::EqHash => Value::Boolean(ScalarValue::new(None, Some(ir_id))),

        // ── Prover hints and sort placeholders ────────────────────
        // A hint is never folded: its value is the prover's to choose and
        // only its pins (separate statements) relate it to its operands.
        IR::Hint { kind, len } => match kind {
            crate::ir_defs::HintKind::StableArgsort
            | crate::ir_defs::HintKind::PositionOneHot
            | crate::ir_defs::HintKind::Negative
            | crate::ir_defs::HintKind::CountAtMost
            | crate::ir_defs::HintKind::RoutingBits => {
                Value::Integer(ScalarValue::new(None, Some(ir_id)))
            }
            crate::ir_defs::HintKind::SortedValues => {
                unknown_like_elements(&args[..*len as usize], ir_id)
            }
            crate::ir_defs::HintKind::MaskedSortedValues => {
                unknown_like_elements(&args[..*len as usize / 2], ir_id)
            }
            crate::ir_defs::HintKind::OrderedBy => {
                unknown_like_elements(&args[*len as usize / 2..*len as usize], ir_id)
            }
            // A raw product sum is not an int64 value: never folded.
            crate::ir_defs::HintKind::ProductSum => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        },
        IR::IdentityCheck(_) => Value::None,
        IR::RescaleRawF => Value::Float(ScalarValue::new(None, Some(ir_id))),
        // A packed key is not an int64 value: never folded.
        IR::PackKey { .. } | IR::PackedKeyIndex { .. } => {
            Value::Integer(ScalarValue::new(None, Some(ir_id)))
        }
        IR::PackedKeyLess { .. } => Value::Boolean(ScalarValue::new(None, Some(ir_id))),
        IR::KeyLess { .. } => {
            let key = |v: &Value, i: &Value| Some((v.int_val()?, i.int_val()?));
            let inferred = key(&args[0], &args[1]).zip(key(&args[2], &args[3])).map(|(x, y)| x < y);
            Value::Boolean(ScalarValue::new(inferred, Some(ir_id)))
        }
        IR::ArrayOp(op) => {
            let semantics = op.kind.semantics();
            match semantics.fold(op, args) {
                Some(Constant::Int(v)) => Value::Integer(ScalarValue::known(v, ir_id)),
                Some(Constant::Float(v)) => Value::Float(ScalarValue::known(v, ir_id)),
                Some(Constant::Bool(v)) => Value::Boolean(ScalarValue::known(v, ir_id)),
                None => match semantics.number_type(op, args) {
                    NumberType::Float => Value::Float(ScalarValue::new(None, Some(ir_id))),
                    _ => Value::Integer(ScalarValue::new(None, Some(ir_id))),
                },
            }
        }
    }
}

/// An unknown value of the type sorted `elements` have: Float when any is
/// a float, otherwise Integer.
fn unknown_like_elements(elements: &[Value], ir_id: StmtId) -> Value {
    if elements.iter().any(|v| matches!(v, Value::Float(_))) {
        Value::Float(ScalarValue::new(None, Some(ir_id)))
    } else {
        Value::Integer(ScalarValue::new(None, Some(ir_id)))
    }
}

fn min_required_arity(ir: &IR) -> usize {
    match ir {
        IR::ConstantInt { .. }
        | IR::ConstantFloat { .. }
        | IR::ConstantBool { .. }
        | IR::ConstantStr { .. }
        | IR::ReadInteger { .. }
        | IR::ReadFloat { .. }
        | IR::ReadHash { .. }
        | IR::ReadExternalResult { .. }
        | IR::Print
        | IR::StructuralPredicate { .. }
        | IR::ScalarPrecondition { .. }
        | IR::AllocateMemory { .. }
        | IR::MemoryTraceSeal
        | IR::AllocateDynamicNDArrayMeta { .. }
        | IR::InvokeExternal { .. } => 0,
        IR::AbsI
        | IR::SignI
        | IR::InvI
        | IR::BitNotI
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
        | IR::ArcSinF
        | IR::ArcCosF
        | IR::LogicalNot
        | IR::IntCast
        | IR::FloatCast
        | IR::BoolCast
        | IR::StrI
        | IR::StrF
        | IR::Assert
        | IR::AssertBit
        | IR::AssertNonNegI
        | IR::AssertU128LtBit { .. }
        | IR::AssertU128Lt
        | IR::ExposePublicI
        | IR::ExposePublicF
        | IR::ReadMemory { .. }
        | IR::ReadTable { .. }
        | IR::MemoryTraceEmit { .. }
        | IR::WitnessDynamicNDArrayMeta { .. }
        | IR::AssertDynamicNDArrayMeta { .. }
        | IR::DynamicNDArrayGetItem { .. }
        | IR::ExportExternalI { .. }
        | IR::ExportExternalF { .. }
        | IR::PoseidonHash => 1,
        IR::AddI
        | IR::SubI
        | IR::MulI
        | IR::DivI
        | IR::FloorDivI
        | IR::FloorDivIPinned { .. }
        | IR::ModI
        | IR::PowI
        | IR::BitAndI
        | IR::BitOrI
        | IR::BitXorI
        | IR::ShlI
        | IR::ShrI
        | IR::AddF
        | IR::SubF
        | IR::MulF
        | IR::DivF
        | IR::FloorDivF
        | IR::ModF
        | IR::PowF
        | IR::PowFI { .. }
        | IR::ArcTan2F
        | IR::SqrtProductIF
        | IR::EqI
        | IR::NeI
        | IR::LtI
        | IR::LtIBounded
        | IR::LteI
        | IR::GtI
        | IR::GtIBounded
        | IR::GteI
        | IR::EqF
        | IR::NeF
        | IR::LtF
        | IR::LteF
        | IR::GtF
        | IR::GteF
        | IR::LogicalAnd
        | IR::LogicalOr
        | IR::AddStr
        | IR::WriteMemory { .. }
        | IR::DynamicNDArraySetItem { .. }
        | IR::EqHash => 2,
        IR::SelectI | IR::SelectF | IR::SelectB | IR::SelectBitI | IR::SelectBitF => 3,
        IR::SelectManyI { len, .. } | IR::SelectManyF { len, .. } => {
            1 + (*len as usize).saturating_mul(2)
        }
        IR::SumF { len } => *len as usize,
        IR::ProductSumF { len } => (*len as usize).saturating_mul(2),
        IR::IndexInBox { rank } => rank.saturating_mul(3),
        IR::Hint { len, .. } => *len as usize + 1,
        IR::KeyLess { .. } => 4,
        IR::PackKey { .. } | IR::PackedKeyLess { .. } => 2,
        IR::PackedKeyIndex { .. } | IR::RescaleRawF => 1,
        IR::IdentityCheck(identity) => identity.operands(),
        IR::ArrayOp(op) => op.shared as usize,
    }
}

fn unknown_value_for_ir(ir: &IR, ir_id: StmtId) -> Value {
    match ir {
        IR::ConstantInt { value } => Value::Integer(ScalarValue::known(*value, ir_id)),
        IR::ConstantFloat { value } => Value::Float(ScalarValue::known(*value, ir_id)),
        IR::ConstantBool { value } => Value::Boolean(ScalarValue::known(*value, ir_id)),
        IR::ConstantStr { value } => Value::String(StringValue {
            val: value.clone(),
            stmt_id: ir_id,
        }),
        IR::AddF
        | IR::SumF { .. }
        | IR::SubF
        | IR::MulF
        | IR::ProductSumF { .. }
        | IR::RescaleRawF
        | IR::DivF
        | IR::FloorDivF
        | IR::ModF
        | IR::PowF
        | IR::PowFI { .. }
        | IR::AbsF
        | IR::SignF
        | IR::SinF
        | IR::SinHF
        | IR::CosF
        | IR::CosHF
        | IR::TanF
        | IR::TanHF
        | IR::SqrtF
        | IR::SqrtProductIF
        | IR::ExpF
        | IR::LogF
        | IR::ArcSinF
        | IR::ArcCosF
        | IR::ArcTan2F
        | IR::SelectF
        | IR::SelectBitF
        | IR::SelectManyF { .. }
        | IR::FloatCast
        | IR::ReadFloat { .. }
        | IR::ExposePublicF
        | IR::ExportExternalF { .. } => Value::Float(ScalarValue::new(None, Some(ir_id))),
        IR::EqI
        | IR::NeI
        | IR::AssertU128LtBit { .. }
        | IR::LtI
        | IR::LtIBounded
        | IR::LteI
        | IR::GtI
        | IR::GtIBounded
        | IR::GteI
        | IR::EqF
        | IR::NeF
        | IR::LtF
        | IR::LteF
        | IR::GtF
        | IR::GteF
        | IR::LogicalAnd
        | IR::LogicalOr
        | IR::LogicalNot
        | IR::SelectB
        | IR::BoolCast
        | IR::EqHash => Value::Boolean(ScalarValue::new(None, Some(ir_id))),
        IR::Print
        | IR::Assert
        | IR::AssertBit
        | IR::AssertNonNegI
        | IR::AssertU128Lt
        | IR::StructuralPredicate { .. }
        | IR::ScalarPrecondition { .. }
        | IR::AllocateMemory { .. }
        | IR::WriteMemory { .. }
        | IR::MemoryTraceEmit { .. }
        | IR::MemoryTraceSeal
        | IR::AllocateDynamicNDArrayMeta { .. }
        | IR::WitnessDynamicNDArrayMeta { .. }
        | IR::AssertDynamicNDArrayMeta { .. }
        | IR::DynamicNDArraySetItem { .. }
        | IR::InvokeExternal { .. }
        | IR::IdentityCheck(_)
        | IR::ExportExternalI { .. } => Value::None,
        IR::AddStr | IR::StrI | IR::StrF => Value::String(StringValue {
            val: String::new(),
            stmt_id: ir_id,
        }),
        _ => Value::Integer(ScalarValue::new(None, Some(ir_id))),
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Python `divmod` for a non-zero divisor (the kernel's `Rounding::Floor`),
/// `None` for a zero divisor or a quotient outside i64.
fn python_floor_div_rem(a: i64, b: i64) -> Option<(i64, i64)> {
    if b == 0 {
        return None;
    }
    let (q, r) = crate::prove::kernel::div_rem_or_zero(
        a as i128,
        b as i128,
        crate::prove::kernel::Rounding::Floor,
    );
    Some((i64::try_from(q).ok()?, i64::try_from(r).ok()?))
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

    /// Read back the `(lo, hi)` bound the read output carries via facts,
    /// the same way `derive_bounds_from_facts` does — i.e. exactly what
    /// downstream `prove()` / requires-discharge would see.
    fn read_bound_fact(b: &IRBuilder, read: &Value) -> Option<(i64, i64)> {
        let vid = read.value_id()?;
        crate::optim::resolver::derive_bounds_from_facts(&b.facts, vid)
    }

    /// Mint a fresh, dynamic (non-constant, fact-free) integer value with a
    /// real stmt_id — a read from a throwaway scratch segment. Used where a
    /// test needs an unbounded runtime scalar (a dynamic address, or a
    /// written value that must NOT trigger per-cell bound capture).
    fn fresh_dynamic_int(b: &mut IRBuilder) -> Value {
        let scratch = b.alloc_segment_id();
        b.ir_allocate_memory(scratch, 1, 0);
        let z = b.ir_constant_int(0);
        b.ir_read_memory(scratch, &z)
    }

    /// C06 hole 2 regression — the `value_ops.rs:1175` swap-and-read shape.
    /// Write bounded cells statically, overwrite one via dynamic set_item,
    /// then read every cell back. The read of the overwritten cell must NOT
    /// republish the pre-set_item cell bound as a fact about the new value.
    /// At `4b69c08` (before this fix) `set_item` invalidated nothing, so the
    /// overwritten cell's stale `(7, 7)` bound survived and was republished.
    #[test]
    fn test_set_item_invalidates_stale_cell_bound() {
        let mut b = IRBuilder::new();
        let seg = b.alloc_segment_id();
        let arr = b.alloc_array_id();
        b.ir_allocate_memory(seg, 2, 0);

        // Write constant 7 @ cell 0 and constant 3 @ cell 1. Constants
        // carry an `Eq` fact, so the per-cell bound capture fires.
        let v7 = b.ir_constant_int(7);
        let a0 = b.ir_constant_int(0);
        b.ir_write_memory(seg, &a0, &v7);
        let v3 = b.ir_constant_int(3);
        let a1 = b.ir_constant_int(1);
        b.ir_write_memory(seg, &a1, &v3);

        // Sanity: before any overwrite, a static read republishes (7, 7).
        {
            let mut probe = IRBuilder::new();
            let pseg = probe.alloc_segment_id();
            probe.ir_allocate_memory(pseg, 1, 0);
            let pv = probe.ir_constant_int(7);
            let pa = probe.ir_constant_int(0);
            probe.ir_write_memory(pseg, &pa, &pv);
            let pr = probe.ir_read_memory(pseg, &pa);
            assert_eq!(
                read_bound_fact(&probe, &pr),
                Some((7, 7)),
                "baseline: static write+read should republish the cell bound"
            );
        }

        // Overwrite cell 0 via dynamic set_item with a value whose runtime
        // address is not statically pinned (a fresh dynamic scalar).
        let idx = fresh_dynamic_int(&mut b);
        let v99 = b.ir_constant_int(99);
        b.ir_dynamic_ndarray_set_item(arr, seg, &idx, &v99);

        // Read cell 0 back. The stale (7, 7) bound must be gone.
        let r0 = b.ir_read_memory(seg, &a0);
        assert_eq!(
            read_bound_fact(&b, &r0),
            None,
            "set_item must invalidate the stale per-cell bound; \
             republishing (7,7) here is a fact about the new value 99"
        );
    }

    /// C06 hole 2 regression, static-write variant: a later static write to
    /// a cell whose new value does NOT trigger capture must still drop the
    /// previous cell's stale bound (unconditional remove before re-capture).
    #[test]
    fn test_restatic_write_drops_stale_cell_bound() {
        let mut b = IRBuilder::new();
        let seg = b.alloc_segment_id();
        b.ir_allocate_memory(seg, 1, 0);

        // First write: constant 7 captures bound (7, 7).
        let v7 = b.ir_constant_int(7);
        let a0 = b.ir_constant_int(0);
        b.ir_write_memory(seg, &a0, &v7);

        // Second write of a value with NO fact-derived bound: a fresh
        // dynamic scalar. `should_capture` is false, so nothing re-captures;
        // the unconditional remove must clear the old (7, 7).
        let fresh = fresh_dynamic_int(&mut b);
        b.ir_write_memory(seg, &a0, &fresh);

        let r0 = b.ir_read_memory(seg, &a0);
        assert_eq!(
            read_bound_fact(&b, &r0),
            None,
            "re-writing a cell with an unbounded value must drop its stale bound"
        );
    }

    /// C06 hole 1 regression — `set_item` must trigger the segment-version /
    /// slice-view invalidation that previously only `ir_write_memory` did.
    /// We deposit a `MaskReadView` (the `mask.rs` after-write deposit), then
    /// mutate the segment via `set_item`; the view must be dropped.
    #[test]
    fn test_set_item_invalidates_mask_read_view() {
        use crate::provenance::MaskReadView;
        let mut b = IRBuilder::new();
        let seg = b.alloc_segment_id();
        let arr = b.alloc_array_id();
        b.ir_allocate_memory(seg, 2, 0);
        let v = b.ir_constant_int(1);
        let a0 = b.ir_constant_int(0);
        b.ir_write_memory(seg, &a0, &v);

        // Deposit a mask-read view (mirrors mask.rs depositing AFTER the
        // alloc writes).
        b.provenance.set_mask_read_view(
            seg,
            MaskReadView {
                mask_cells: vec![],
                sources: vec![],
                count_stmt: None,
            },
        );
        assert!(b.provenance.mask_read_view(seg).is_some());

        // A set_item mutation must invalidate it (was a no-op at 4b69c08).
        let idx = fresh_dynamic_int(&mut b);
        let nv = b.ir_constant_int(5);
        b.ir_dynamic_ndarray_set_item(arr, seg, &idx, &nv);
        assert!(
            b.provenance.mask_read_view(seg).is_none(),
            "set_item must drop the segment's mask-read view (hole 1)"
        );
    }
}

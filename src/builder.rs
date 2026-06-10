use std::collections::{HashMap, HashSet};

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::optim::resolver::{Resolver, StaticOnlyResolver};
use crate::types::{NumberType, ScalarValue, StmtId, StringValue, Value};

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

/// Affine slice provenance: element `j` of the annotated array equals
/// `base[base_offset + (start + j*step) * stride]` (axis-domain map into a
/// flat base segment) for every in-range `j < runtime_length`. `start` is
/// an IR value (possibly dynamic); `step`/geometry are static.
#[derive(Clone, Debug)]
pub struct SliceView {
    pub base_segment: u32,
    pub axis_len: usize,
    pub stride: usize,
    pub base_offset: usize,
    pub start: Value,
    pub step: i64,
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
    /// Small expression-provenance side table for StaticArray results whose
    /// concrete payload can be reconstructed exactly, but may be consumed by
    /// a fused lowering before materializing every intermediate cell.
    pub static_array_product_provenance: HashMap<u32, StaticArrayProductProvenance>,
    /// Segment-backed StaticArrays whose logical payload is available in
    /// `static_array_payload` but whose per-cell memory writes have not yet
    /// been emitted. Static reads can use the cache directly; dynamic reads
    /// and writes must materialize first.
    pub lazy_static_array_segments: HashSet<u32>,
    /// Slice provenance for materialised dynamic 1-D slices: maps the
    /// output segment to the affine view of its source, so a slice of a
    /// slice composes index maps instead of re-materialising (and the
    /// composed start often folds to a constant — e.g. a reversal of a
    /// to-the-end slice cancels the dynamic start). Invalidated on any
    /// write to the view's segment or its base.
    pub slice_views: HashMap<u32, SliceView>,
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
    /// Provenance map for `np.argwhere` results, keyed by the result
    /// segment_id. Stores `(source_shape, per-cell keep predicates in
    /// row-major order)`. Lets `dyn_aggregate_axis` recognise the
    /// `argwhere(A).min/max(axis=0)` pattern and bypass the
    /// materialise-then-reduce O(N*rank) scan with an O(R + C) per-axis
    /// has-any + linear search — the algorithm the handwritten Noir
    /// baselines use for the nonzero-bbox kernels. Cleared lazily; entries
    /// are only consulted when an `axis=0` min/max reduces the matching
    /// dynamic array.
    pub argwhere_keep_indicators: HashMap<u32, (Vec<usize>, Vec<Value>)>,
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
            static_array_product_provenance: HashMap::new(),
            lazy_static_array_segments: HashSet::new(),
            slice_views: HashMap::new(),
            resolver: Box::new(StaticOnlyResolver::new()),
            facts: crate::optim::predicates::FactStack::new(),
            value_to_stmt: HashMap::new(),
            // Activated when `ZINNIA_TELEMETRY_DIR` is set; `None` otherwise.
            // Reading the env var at construction time means an
            // `IRBuilder::new()` call after the env var changes won't pick
            // up the change, but the tests scope the env var around the
            // builder construction so this matches the desired semantics.
            telemetry: crate::optim::telemetry::TelemetrySink::from_env(),
            argwhere_keep_indicators: HashMap::new(),
            segment_cell_bounds: HashMap::new(),
            segment_default_bounds: HashMap::new(),
            scalar_value_bounds: HashMap::new(),
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
                return Some((mn, mx));
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
                if let Some(sink) = &self.telemetry {
                    let value_ids = crate::optim::predicates::collect_value_ids(term);
                    sink.emit(&crate::optim::telemetry::discharge_event(
                        op_name, "Proved", "none", false, &value_ids,
                    ));
                }
            }
            ProveOutcome::Disproved => {
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
            | IR::PowFI
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
        val
    }

    pub fn export_ir_graph(self) -> IRGraph {
        IRGraph::new(self.stmts)
    }

    // ── Convenience helpers used by optimization passes ──────────────

    pub fn ir_constant_int(&mut self, value: i64) -> Value {
        let out = self.create_ir(&IR::ConstantInt { value }, &[]);
        // Deposit a tight `out == value` fact so `derive_bounds_from_facts`
        // (and through it `interval_fact_for_int_binary`) can chain through
        // constants. Without this, arithmetic like `max(arr) + 1` fails
        // to propagate the bound on `max(arr)` to the sum, because the
        // constant operand has no per-value fact entries.
        if let Some(vid) = out.value_id() {
            use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
            self.facts.insert_for(
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

    /// ProductSumF operands are adjacent Float pairs:
    /// `[lhs0, rhs0, lhs1, rhs1, ...]`, with `len` equal to the pair count.
    /// Backend semantics are scalar `MulF` per pair followed by existing
    /// float-add accumulation, preserving fixed-point rescale checks.
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
        let out = self.create_ir(&IR::DivF, &[a.clone(), b.clone()]);
        if let (Some(a_vid), Some(b_vid), Some(out_vid)) =
            (a.value_id(), b.value_id(), out.value_id())
        {
            let mut formals = HashMap::new();
            formals.insert("lhs".to_string(), a_vid);
            formals.insert("rhs".to_string(), b_vid);
            self.fire_contract("div_f", out_vid, &formals);

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
        ir_pow_f       => IR::PowF,       contract = "pow_f",
        ir_pow_f_int   => IR::PowFI,      contract = "pow_f",
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

    /// Float `arcsin`. Uses the same real-domain policy as `arccos`:
    /// `-1 <= x <= 1` is discharged if visible, otherwise emitted as a
    /// witness assertion in lenient mode.
    pub fn ir_arcsin_f(&mut self, a: &Value) -> Value {
        let out = self.create_ir(&IR::ArcSinF, &[a.clone()]);
        if let (Some(vid), Some(x_vid)) = (out.value_id(), a.value_id()) {
            let mut formals = HashMap::new();
            formals.insert("x".to_string(), x_vid);
            self.fire_contract("arcsin_f", vid, &formals);
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

    // Assert & expose
    ir_unary!(
        ir_assert          => IR::Assert,
        ir_assert_bit      => IR::AssertBit,
        ir_expose_public_i => IR::ExposePublicI,
        ir_expose_public_f => IR::ExposePublicF,
    );

    /// Two-operand assert: `0 <= idx < hi`. Backends pick the cheapest
    /// sound implementation; on Noir this lowers to a single
    /// `assert((idx as u128) < (hi as u128))` (~30 ACIR vs ~49 for the
    /// 2-cmp + and + assert decomposition). See `IR::AssertU128Lt`.
    pub fn ir_assert_u128_lt(&mut self, idx: &Value, hi: &Value) -> Value {
        self.create_ir(&IR::AssertU128Lt, &[idx.clone(), hi.clone()])
    }

    /// Constrain `0 <= idx < hi` and return the low decomposition bits.
    ///
    /// This is intentionally value-producing: users such as small static
    /// gathers can route directly on the constrained bits instead of
    /// rebuilding one-hot equality masks from the original index.
    pub fn ir_assert_u128_lt_bits(&mut self, idx: &Value, hi: usize) -> Vec<Value> {
        assert!(hi > 0, "assert_u128_lt_bits: hi must be positive");
        let bits = usize::BITS - (hi - 1).leading_zeros();
        (0..bits)
            .map(|bit| {
                self.create_ir(
                    &IR::AssertU128LtBit {
                        hi: hi as u64,
                        bit,
                        bits,
                    },
                    &[idx.clone()],
                )
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
        self.create_ir(
            &IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            },
            &[],
        )
    }

    pub fn ir_write_memory(&mut self, segment_id: u32, address: &Value, value: &Value) -> Value {
        // A write invalidates slice views of this segment and any views
        // reading through it as their base.
        self.slice_views
            .retain(|seg, view| *seg != segment_id && view.base_segment != segment_id);
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
        if let Some(addr_i) = address.int_val() {
            let should_capture = value.value_id().is_some_and(|vid| {
                if self.facts.per_value.contains_key(&vid) {
                    return true;
                }
                if let Some(stmt_id) = value.stmt_id() {
                    if let Some(stmt) = self.stmts.get(stmt_id as usize) {
                        use crate::ir_defs::IR;
                        if matches!(stmt.ir, IR::AddI | IR::SubI) && stmt.arg_values.len() == 2 {
                            for &arg_vid in &stmt.arg_values {
                                if self.facts.per_value.contains_key(&arg_vid) {
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
        )
    }

    pub fn ir_read_memory(&mut self, segment_id: u32, address: &Value) -> Value {
        let out = self.create_ir(&IR::ReadMemory { segment_id }, &[address.clone()]);
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
        self.create_ir(
            &IR::DynamicNDArrayGetItem {
                array_id,
                segment_id,
            },
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
            &IR::DynamicNDArraySetItem {
                array_id,
                segment_id,
            },
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
}

// ---------------------------------------------------------------------------
// build_ir — the Rust equivalent of AbstractIR.build_ir()
// ---------------------------------------------------------------------------

/// Build an IR statement and compute the result value.
/// This implements the `build_ir(ir_id, args, dbg)` method for all 79 IR types.
fn build_value(ir: &IR, ir_id: StmtId, args: &[Value]) -> Value {
    if let Some(min_arity) = min_required_arity(ir) {
        if args.len() < min_arity {
            return unknown_value_for_ir(ir, ir_id);
        }
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
        IR::DivI => int_binary_ir(ir_id, args, |a, b| if b != 0 { Some(a / b) } else { None }),
        IR::FloorDivI => int_binary_ir(ir_id, args, |a, b| {
            if b != 0 {
                Some(a.div_euclid(b))
            } else {
                None
            }
        }),
        IR::ModI => int_binary_ir(ir_id, args, |a, b| if b != 0 { Some(a % b) } else { None }),
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
        IR::ShlI => int_binary_ir(ir_id, args, |a, b| {
            let shift = b.max(0).min(63) as u32;
            Some(a.wrapping_shl(shift))
        }),
        IR::ShrI => int_binary_ir(ir_id, args, |a, b| {
            let shift = b.max(0).min(63) as u32;
            Some(a.wrapping_shr(shift))
        }),
        IR::BitNotI => int_unary_ir(ir_id, args, |a| Some(!a)),

        // ── Integer unary arithmetic ──────────────────────────────
        IR::AbsI => int_unary_ir(ir_id, args, |a| Some(a.abs())),
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
        IR::ModF => float_binary_ir(ir_id, args, |a, b| a % b),
        IR::PowF => float_binary_ir(ir_id, args, |a, b| a.powf(b)),
        IR::PowFI => {
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
        IR::Assert | IR::AssertBit | IR::AssertU128Lt => Value::None,
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
        IR::WitnessDynamicNDArrayMeta { .. } | IR::AssertDynamicNDArrayMeta { .. } => Value::None,
        IR::DynamicNDArrayGetItem { .. } => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::DynamicNDArraySetItem { .. } => Value::None,

        // ── External calls (fixed) ────────────────────────────────
        IR::InvokeExternal { .. } => Value::None,
        IR::ExportExternalI { .. } | IR::ExportExternalF { .. } => Value::None,

        // ── Hash ──────────────────────────────────────────────────
        IR::PoseidonHash => Value::Integer(ScalarValue::new(None, Some(ir_id))),
        IR::EqHash => Value::Boolean(ScalarValue::new(None, Some(ir_id))),
    }
}

fn min_required_arity(ir: &IR) -> Option<usize> {
    Some(match ir {
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
        | IR::AssertU128LtBit { .. }
        | IR::AssertU128Lt
        | IR::ExposePublicI
        | IR::ExposePublicF
        | IR::ReadMemory { .. }
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
        | IR::PowFI
        | IR::ArcTan2F
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
    })
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
        | IR::DivF
        | IR::FloorDivF
        | IR::ModF
        | IR::PowF
        | IR::PowFI
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

//! IR -> Noir source emitter.
//!
//! Implements [`Synthesizer`] with `CellRef = String` — each cell ref is a
//! Noir-language expression (typically a let-bound identifier). The
//! interpreter walks the IR in statement order and we emit one
//! `let vN = <expr>;` per produced value into the body of `fn main`.
//!
//! Inputs are bound to Noir function parameters whose names + shapes are
//! derived from the [`ResolvedWitness`]. Floats are quantized to Q32 at
//! emit time (so the field-element shape matches what the handwritten
//! baselines write into their `Prover.toml`). The accompanying
//! `Prover.toml` is also produced here from the witness so the two
//! artifacts stay in lockstep.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use crate::circuit_input::{InputNode, InputPath, PathSegment, ResolvedWitness};
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;

const Q_BITS: u32 = 32;
const Q_U64: u64 = 1u64 << 32;

/// Output of [`emit_circuit`].
pub struct EmitOutput {
    pub main_nr: String,
    pub prover_toml: String,
    /// Decimal-string field representation of the exposed public values,
    /// in declaration order. Mirrors the structure of `ProofArtifact.public_values`.
    pub public_values_decimal: Vec<String>,
    pub expose_count: usize,
}

/// Top-level entry point: emit a complete Noir package source pair.
pub fn emit_circuit(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    params: &ProvingParams,
) -> Result<EmitOutput, ProvingError> {
    // Honest cap: the Noir baselines (and `noir_fpc`'s `Q` global) hardcode
    // Q32 fixed-point. The float quantization here (`Q_BITS`) and the
    // library's rescale arithmetic both assume scale == 2^32, so a caller
    // requesting a different `precision_bits` would silently get the wrong
    // fixed-point semantics. Reject loudly rather than mis-quantize. True
    // parameterization would require code-generating the lib's `Q` global.
    if params.precision_bits != Q_BITS {
        return Err(ProvingError::synthesis(format!(
            "Noir backend only supports precision_bits == {} (Q32 fixed-point is \
             hardcoded in noir_fpc's `Q` global); got precision_bits == {}. \
             Re-run with precision_bits=32 or use the halo2 backend for other \
             precisions.",
            Q_BITS, params.precision_bits
        )));
    }

    let prover_toml = emit_prover_toml(witness);

    let input_params = derive_input_params(witness);
    let bounds = super::bound_analysis::run(ir);
    let intrinsic_count = super::bound_analysis::count_u32_intrinsic_ops(ir);
    let mut emitter = NoirEmitter::new(
        input_params.clone(),
        witness.clone(),
        params.clone(),
        bounds,
        intrinsic_count,
        &ir.provenance,
    );
    interpret_ir(ir, &mut emitter)?;

    let main_nr = render_main_nr(&input_params, &emitter)?;

    Ok(EmitOutput {
        main_nr,
        prover_toml,
        public_values_decimal: emitter.public_decimal.clone(),
        expose_count: emitter.exposes.len(),
    })
}

// ---------------------------------------------------------------------------
// Prover.toml emission
// ---------------------------------------------------------------------------

fn emit_prover_toml(witness: &ResolvedWitness) -> String {
    let mut out = String::new();
    for p in &witness.input.params {
        // Skip params the circuit declares but Zinnia never reads. The
        // safer default is to still write them — nargo errors out if a
        // declared `fn main` parameter has no Prover.toml entry, and the
        // function signature does declare every input. So we always emit.
        write_param(&mut out, &p.name, &p.value);
    }
    out
}

fn write_param(out: &mut String, name: &str, node: &InputNode) {
    match node {
        InputNode::Int(v) => {
            let _ = writeln!(out, "{} = \"{}\"", name, v);
        }
        InputNode::Float(v) => {
            let q = quantize_to_field_string(*v);
            let _ = writeln!(out, "{} = \"{}\"", name, q);
        }
        InputNode::Bool(v) => {
            // The parameter is declared `Field` in `fn main` (booleans
            // arrive as Field 0/1), so the Prover.toml value must be the
            // string-quoted field literal `"1"`/`"0"`. A bare TOML
            // `true`/`false` is a nargo type error against a Field param.
            let _ = writeln!(out, "{} = \"{}\"", name, if *v { 1u8 } else { 0 });
        }
        InputNode::Hex(s) => {
            let _ = writeln!(out, "{} = \"{}\"", name, s);
        }
        InputNode::Array { elements, .. } => {
            let _ = write!(out, "{} = [", name);
            let mut first = true;
            for e in flat_leaves(elements) {
                if !first {
                    let _ = write!(out, ", ");
                }
                first = false;
                let _ = write!(out, "\"{}\"", leaf_to_field_decimal(e));
            }
            let _ = writeln!(out, "]");
        }
        InputNode::Sequence(items) => {
            // Heterogeneous tuples — emit as inline-table with positional keys.
            // None of the 19 pilot kernels exercise this path; surface an
            // explicit comment so the failure case is visible at runtime.
            let _ = writeln!(
                out,
                "# {} = <Sequence — not yet supported in Prover.toml>",
                name
            );
            let _ = items;
        }
        InputNode::Hashed { .. } => {
            let _ = writeln!(
                out,
                "# {} = <Hashed — not yet supported in Prover.toml>",
                name
            );
        }
    }
}

fn flat_leaves(elements: &[InputNode]) -> Vec<&InputNode> {
    let mut out = Vec::new();
    for e in elements {
        match e {
            InputNode::Array { elements, .. } => out.extend(flat_leaves(elements)),
            InputNode::Sequence(items) => out.extend(flat_leaves(items)),
            leaf => out.push(leaf),
        }
    }
    out
}

fn leaf_to_field_decimal(node: &InputNode) -> String {
    match node {
        InputNode::Int(v) => v.to_string(),
        InputNode::Float(v) => quantize_to_field_string(*v),
        InputNode::Bool(v) => (if *v { 1u8 } else { 0 }).to_string(),
        InputNode::Hex(s) => format!("0x{}", s),
        _ => "0".to_string(),
    }
}

/// Q32 quantization: result is the *signed* Q-scaled integer rendered as
/// a decimal Noir Field literal. Negative values are encoded as
/// `BN254_PRIME - |v|`.
fn quantize_to_field_string(v: f64) -> String {
    let q = crate::prove::field::quantize(v, Q_BITS);
    signed_to_field_decimal(q)
}

/// BN254 scalar field modulus, base-10, for negative-value encoding.
/// `21888242871839275222246405745257275088548364400416034343698204186575808495617`.
fn bn254_modulus_dec() -> &'static str {
    "21888242871839275222246405745257275088548364400416034343698204186575808495617"
}

fn signed_to_field_decimal(q: i128) -> String {
    if q >= 0 {
        q.to_string()
    } else {
        // p - |q| via big-int subtraction expressed as string.
        sub_bigint_decimal(bn254_modulus_dec(), &q.unsigned_abs().to_string())
    }
}

fn sub_bigint_decimal(a: &str, b: &str) -> String {
    // Schoolbook subtraction, assuming a >= b (b is at most ~2^127 here,
    // and a is ~2^254). Both inputs are non-negative decimal digit strings.
    let a_bytes: Vec<u8> = a.bytes().collect();
    let b_bytes: Vec<u8> = b.bytes().collect();
    let mut result = Vec::with_capacity(a_bytes.len());
    let mut borrow: i32 = 0;
    let an = a_bytes.len();
    let bn = b_bytes.len();
    for i in 0..an {
        let a_digit = (a_bytes[an - 1 - i] - b'0') as i32;
        let b_digit = if i < bn {
            (b_bytes[bn - 1 - i] - b'0') as i32
        } else {
            0
        };
        let mut d = a_digit - b_digit - borrow;
        if d < 0 {
            d += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.push(b'0' + d as u8);
    }
    while result.len() > 1 && *result.last().unwrap() == b'0' {
        result.pop();
    }
    result.reverse();
    String::from_utf8(result).unwrap_or_else(|_| "0".to_string())
}

// ---------------------------------------------------------------------------
// Function-signature derivation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EmittedInput {
    pub name: String,
    pub kind: InputKind,
}

#[derive(Debug, Clone)]
pub enum InputKind {
    ScalarField,
    Bool,
    ArrayField { len: usize },
}

fn derive_input_params(witness: &ResolvedWitness) -> Vec<EmittedInput> {
    let mut out = Vec::new();
    for p in &witness.input.params {
        let kind = match &p.value {
            InputNode::Int(_) | InputNode::Float(_) | InputNode::Hex(_) => InputKind::ScalarField,
            InputNode::Bool(_) => InputKind::Bool,
            InputNode::Array { elements, .. } => InputKind::ArrayField {
                len: flat_leaves(elements).len(),
            },
            InputNode::Sequence(items) => InputKind::ArrayField {
                len: flat_leaves(items).len(),
            },
            InputNode::Hashed { .. } => InputKind::ScalarField,
        };
        out.push(EmittedInput {
            name: p.name.clone(),
            kind,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// NoirEmitter — Synthesizer impl
// ---------------------------------------------------------------------------

/// True iff `s` is a single Noir identifier (letters / digits / underscore,
/// leading non-digit). Used to detect passthrough expressions that don't need
/// to be rebound to a fresh SSA name.
fn is_identifier(s: &str) -> bool {
    let s = s.trim();
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Extract all identifier-shaped tokens from a snippet. Used by the DCE pass
/// to decide which let-bindings have at least one downstream reference.
/// Reserved keywords and the `let`/`assert` heads are filtered by simple
/// allow-list since they never reference an SSA name.
fn extract_idents(s: &str) -> Vec<&str> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < bytes.len() {
                let cc = bytes[i] as char;
                if cc.is_alphanumeric() || cc == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(&s[start..i]);
        } else {
            i += 1;
        }
    }
    out
}

/// Post-emit DCE: drop `let vN = expr;` lines whose `vN` is never referenced
/// in (a) any side-effecting line (`assert(...)`, function calls), or (b) any
/// other live let-binding, or (c) the exposed return tuple. Big-impact wins
/// for reshape-style kernels where the IR emits long runs of unused integer
/// constants (loop-iteration placeholders, envelope metadata).
fn dce_body(body: &str, exposes: &[String]) -> String {
    // Pass 1: parse lines into (Option<names>, raw_line, ref_idents). A line
    // can bind a single name (`let vN = expr;`) or a tuple destructure
    // (`let (vA, vB) = expr;` — used by `project_signed`); both forms are
    // kept alive iff ANY of their bound names is live.
    let mut entries: Vec<(Option<Vec<String>>, &str, Vec<String>)> = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("let ") {
            if let Some(eq) = rest.find(" = ") {
                let lhs = rest[..eq].trim();
                let expr = &rest[eq + 3..];
                // Drop an optional type annotation `vN: TYPE` so the name
                // matches downstream identifier refs. Both `let v: u32 =`
                // (u32-tagged cells from `bind_u32`) and `let (a, b) =`
                // tuple destructures (from `project_signed`) must be
                // tracked, otherwise DCE silently drops them and we get
                // "not found in this scope" errors at nargo time.
                let names: Vec<String> =
                    if let Some(inner) = lhs.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                        inner
                            .split(',')
                            .map(|s| {
                                let s = s.trim();
                                s.split(':').next().unwrap_or(s).trim().to_string()
                            })
                            .collect()
                    } else {
                        let name = lhs.split(':').next().unwrap_or(lhs).trim().to_string();
                        vec![name]
                    };
                let uses: Vec<String> =
                    extract_idents(expr).iter().map(|s| s.to_string()).collect();
                entries.push((Some(names), line, uses));
                continue;
            }
        }
        // Non-`let` line — side-effecting (assert / call / unsafe). Always live.
        let uses: Vec<String> = extract_idents(line).iter().map(|s| s.to_string()).collect();
        entries.push((None, line, uses));
    }

    // Pass 2: seed liveness with exposes + identifiers in side-effecting lines.
    let mut live: std::collections::HashSet<String> = std::collections::HashSet::new();
    for e in exposes {
        for id in extract_idents(e) {
            live.insert(id.to_string());
        }
    }
    for (names, _line, uses) in &entries {
        if names.is_none() {
            for u in uses {
                live.insert(u.clone());
            }
        }
    }

    // Pass 3: backward fixpoint — if any bound name on this line is live,
    // propagate liveness to its uses.
    let mut changed = true;
    while changed {
        changed = false;
        for (names, _line, uses) in entries.iter().rev() {
            if let Some(ns) = names {
                if ns.iter().any(|n| live.contains(n)) {
                    for u in uses {
                        if live.insert(u.clone()) {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Pass 4: emit only live lines.
    let mut out = String::new();
    for (names, line, _) in &entries {
        match names {
            Some(ns) if !ns.iter().any(|n| live.contains(n)) => continue,
            _ => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// SSA-style emitter. Each call returns the name of a Noir let-binding
/// (or a literal expression for cheap constants). The interpreter dispatches
/// one method per IR statement.
/// Affine view of an integer cell: `cell == sum(root_i * scale_i) + offset`,
/// with at most two dynamic roots and non-zero scales.
#[derive(Clone, Debug, Default)]
struct AffLin {
    terms: Vec<(String, i64)>,
    offset: i64,
}


pub struct NoirEmitter<'a> {
    /// Structural-provenance table (derived once in the IRBuilder; see
    /// src/provenance.rs). Read LIVE for intervals / affine views /
    /// segment hulls via the name ↔ ValueId maps below.
    prov: &'a crate::provenance::ProvenanceTable,
    vid_of_var: HashMap<String, crate::types::ValueId>,
    var_of_vid: HashMap<crate::types::ValueId, String>,
    /// Function arguments declared on `fn main`.
    pub inputs: Vec<EmittedInput>,
    /// Resolved witness — used to compute constant-folded values for any
    /// expose-public output (so we can populate `public_values_decimal`).
    witness: ResolvedWitness,
    /// Body of `main`, one let-binding per line.
    pub body: String,
    /// Monotonic counter for fresh SSA names.
    next_id: usize,
    /// SSA names that have been declared as `let` so far — used so we can
    /// constant-fold trivial expressions back to literals without
    /// double-binding.
    used_libs: BTreeMap<&'static str, ()>,
    /// IR-stmt-ordered list of expose targets. Each entry is the variable
    /// or expression to return (in declaration order) as part of the
    /// final `pub` tuple/array.
    pub exposes: Vec<String>,
    /// Decimal-field representation of each exposed value (matches what
    /// `bb verify` will read from the public_inputs file). Computed via
    /// the resolved witness — we constant-fold by tracking a parallel
    /// value cache keyed by variable name.
    pub public_decimal: Vec<String>,
    /// Variable-name -> known WITNESS-folded concrete value, when we
    /// constant-folded the expression at emit time. Used for public-input
    /// bookkeeping only; the proof itself derives values from witness
    /// execution by nargo. STRUCTURALLY private (see `wfold::WitnessFold`):
    /// it can only be read through the witness-read accessors, never by
    /// the structural `s_int_of`/`s_float_of` path. This is what makes a
    /// witness-dependent emission unrepresentable rather than convention.
    witness_vals: wfold::WitnessFold,
    /// Affine address lineage: var == sum of root_i * scale_i + offset,
    /// with at most two dynamic roots (mirrors the halo2 backend's
    /// AffineLin). Lets dynamic-address memory reads gather with shared
    /// per-root indicators instead of a per-read mux over the segment.
    affine_of: HashMap<String, AffLin>,
    /// eq(root, v) indicator bindings, shared across all reads/writes on
    /// the same root. Roots are SSA names (immutable), so no invalidation.
    ind_cache: HashMap<(String, i64), String>,
    /// (root, enumerated value set) one-hot memberships already asserted.
    membership_done: std::collections::HashSet<(String, Vec<i64>)>,
    /// Stage-1 gathered planes for factored two-root reads, keyed by
    /// (segment, write version, root, scale, residual offset).
    plane_cache: HashMap<(u32, u64, String, i64, i64), String>,
    /// Bumped on every write to a segment; invalidates plane_cache.
    write_version: HashMap<u32, u64>,
    /// Per-segment value hull: the interval union of every value written
    /// to the segment (plus its init). `None` once any written value
    /// lacks a derivable interval. Read results inherit this hull, which
    /// is what lets comparison chains over gathered values stay on the
    /// cheap unsigned path.
    segment_iv: HashMap<u32, Option<(i64, i64)>>,
    /// Constraint-derived signed value intervals (port of the halo2
    /// backend's value_interval): constants exact, select outputs take
    /// the hull of their branches, comparison outputs are [0,1], and
    /// add/sub/mul-const propagate. Used to shrink the feasible slot set
    /// of dynamic-address gathers. Witness-independent by construction.
    interval_of: HashMap<String, (i64, i64)>,
    /// Peephole provenance for the negative-index normalization pattern
    /// `select(lt(x, 0), x + N, x)`: maps lt(x,0) outputs to x, add-const
    /// outputs to (x, c), and recognized normalized indices to (x, N).
    /// `eq_i(norm, j)` then emits the signed-machinery-free identity
    /// `(x == j) | (x == j - N)` — exact for every x only when the compared
    /// constant satisfies `0 <= j < N` (see eq_i for the guard and proof).
    lt_zero_of: HashMap<String, String>,
    add_const_of: HashMap<String, (String, i64)>,
    norm_index_of: HashMap<String, (String, i64)>,
    /// Track whether the lookup-table-style cmp helper is used (records
    /// which noir_fpc symbols to `use ...`-import).
    needs_lib: HashMap<&'static str, bool>,
    /// Tracks how each statement was emitted so we can detect ID reuse.
    params: ProvingParams,
    /// In-Rust simulation of Zinnia's segment-keyed scratch memory.
    /// Each entry maps a (segment_id, addr) to either the most recent
    /// SSA variable written there (`Some(name)`) or the segment-wide
    /// init-value as a literal Field. Mirrors the halo2 synthesizer's
    /// `memories: HashMap<u32, Vec<Fp>>` exactly, but stores names so
    /// later reads can re-emit `let vN = <writer>` cheaply.
    memories: HashMap<u32, Vec<String>>,
    /// segment_id -> init-value-as-Field-decimal-string.
    memory_init: HashMap<u32, String>,
    /// Cache: cell -> (sign_var, mag_var) emitted by `project_signed`.
    /// `to_signed` does a 126-bit range check (~140 ACIR gates); when the
    /// same operand participates in many comparisons (slice masks, clamp
    /// chains) we project once and feed the pair into `qlt_signed` etc.
    signed_proj: HashMap<String, (String, String)>,
    /// Pairs `(idx, hi)` already emitted via `assert_u128_lt`. Zinnia's
    /// IR layer can re-issue the same bound discharge (e.g. multiple
    /// dynamic reads against the same axis length re-resolve through
    /// `discharge_index_in_range`); dropping the duplicate `assert((a
    /// as u128) < (b as u128))` line saves a per-cmp u128 cast + compare
    /// (~2 ACIR gates each, ~80 gates on the 0345 mux-indexed kernel).
    emitted_u128_asserts: HashSet<(String, String)>,
    /// CSE cache: expression text -> existing SSA name that binds it.
    /// Common in unrolled patterns where the IR builder re-emits
    /// `v_const_a + v_const_b` for the same constants. CSE here saves
    /// the duplicate let-binding plus any downstream cost (e.g. extra
    /// `assert_u128_lt(dup_idx, hi)` lines that the bound-dedupe set
    /// above can't catch because the names differ).
    cse_cache: HashMap<String, String>,
    /// Per-cell upper bound (inclusive) on the witness magnitude.
    /// Populated from `ir_constant_int(c)` (bound = c if c >= 0) and
    /// propagated through arithmetic via `checked_add` / `checked_mul`.
    /// When the bound provably fits `u32` AND every operand of an
    /// integer op is `u32`-typed (see `cell_type`), the op is emitted
    /// as `let v: u32 = ...;` — that elides the `BLACKBOX::RANGE` check
    /// every `Field as u32` cast would otherwise require (~258 RANGE
    /// blackboxes on a single 10×10 argwhere-bbox pattern).
    cell_bound: HashMap<String, u64>,
    /// Per-cell binding type. Defaults to `Field`. Cells whose
    /// `cell_bound` proves < 2^32 AND whose producing op kept all
    /// operands u32-typed are bound `u32`. A `u32` cell used in a
    /// `Field` context promotes via the zero-cost `(v as Field)`
    /// embedding (Noir generates no constraint for u32 → Field).
    /// See `bind_typed` / `as_field` for the call sites.
    cell_type: HashMap<String, CellType>,
    /// STRUCTURAL constant store: SSA name -> concrete value that is known
    /// at circuit-BUILD time (IR-level `ConstantInt`/`ConstantFloat`/
    /// `ConstantBool`, or values derived purely from such constants via
    /// the structural accessors). This is the ONLY map the structural
    /// accessors `s_int_of`/`s_float_of` read; it is wholly separate from
    /// `witness_vals`. A "value happens to be folded from the resolved
    /// witness" can therefore never reach a structural emission decision —
    /// inlining a witness-derived value would specialise the circuit to one
    /// witness (unsound for any other input). Replaces the former
    /// `literal_cells: HashSet` + shared `value_cache` arrangement.
    literal_vals: HashMap<String, ConcreteValue>,
    /// IR-level forward-dataflow `fits-u32` bounds, keyed by stmt_id.
    /// Populated by [`super::bound_analysis::run`] before emit. The
    /// interpreter's per-stmt hook (`on_stmt_begin`) advances
    /// `current_stmt_bound` to the upcoming statement's entry so the
    /// active synth method can promote results to `u32` even when
    /// the local emit-time tracking would have missed the bound
    /// (e.g. results that flow through `select` or come from constants
    /// the IR analysis can fold but emit cannot).
    stmt_bounds: super::bound_analysis::BoundMap,
    /// Bound for the statement currently being dispatched (set by
    /// `on_stmt_begin`). Read by the integer-result methods; not all
    /// methods consult it (Float ops + Q-functions always emit Field).
    current_stmt_bound: Option<u64>,
    /// Kernel-wide flag: enable Field→u32 promotion (one explicit cast
    /// per Field operand) when the IR analysis identifies enough
    /// u32-eligible cells that the per-cast RANGE checks are
    /// amortized against the savings on downstream u32 ops. Computed
    /// from the bound map at construction; see `new` for the
    /// threshold derivation.
    u32_amortizable: bool,
}

/// Noir-side binding type tracked per emitted cell. The default is
/// `Field`; the bound-driven emit upgrades to `U32` whenever it's safe
/// (operands are u32-typed and the result bound fits 2^32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellType {
    Field,
    U32,
}

#[derive(Debug, Clone)]
enum ConcreteValue {
    Int(i128),
    Float(f64),
    Bool(bool),
    /// Stored as a raw field-element decimal string; used for unresolvable
    /// values (e.g. complex expressions where we can't statically fold).
    Field(String),
}

/// Witness-folded value store — STRUCTURALLY isolated from circuit emission.
///
/// This module exists to make the project's hardest soundness invariant
/// — *a witness-folded value must never influence emitted circuit
/// structure* — UNREPRESENTABLE rather than convention-only. The inner
/// map is a private field, so the only way to read a witness value is
/// through one of the named accessor methods below (`int_of`/`float_of`/
/// `bool_of`/`decimal_of`). None of those can be confused with the
/// emitter's *structural* accessors (`s_int_of`/`s_float_of`, which read
/// the wholly separate `literal_vals` map on the emitter). A future
/// feature that wants to branch the circuit on a folded value would have
/// to call a method whose name advertises the witness read — the loud
/// visibility change R06 requires.
///
/// Whitelisted readers (all inside this module or the emitter's
/// fold/expose paths): `fold_*` (write+propagate), `expose_public` /
/// `public_decimal` rendering (via `decimal_of`), and
/// `resolve_path_concrete` (via `insert`).
mod wfold {
    use super::ConcreteValue;
    use std::collections::HashMap;

    /// Witness-derived constant-fold cache, keyed by SSA variable name.
    /// Populated during emit purely for public-input bookkeeping; the
    /// proof itself derives values from witness execution by nargo.
    #[derive(Default)]
    pub(super) struct WitnessFold {
        /// PRIVATE: never expose the map itself. Reads go through the
        /// typed accessors so a witness value cannot silently leak into a
        /// structural decision.
        vals: HashMap<String, ConcreteValue>,
    }

    impl WitnessFold {
        /// Insert a fully-resolved concrete value (used by the
        /// read-input / resolve-path and memory-propagation paths).
        pub(super) fn insert(&mut self, var: String, val: ConcreteValue) {
            self.vals.insert(var, val);
        }

        /// Witness read: integer view of the folded value, if any.
        /// Only `fold_*` and bookkeeping should consume this.
        pub(super) fn int_of(&self, var: &str) -> Option<i128> {
            match self.vals.get(var)? {
                ConcreteValue::Int(i) => Some(*i),
                ConcreteValue::Bool(b) => Some(if *b { 1 } else { 0 }),
                _ => None,
            }
        }

        /// Witness read: float view of the folded value, if any.
        pub(super) fn float_of(&self, var: &str) -> Option<f64> {
            match self.vals.get(var)? {
                ConcreteValue::Float(f) => Some(*f),
                ConcreteValue::Int(i) => Some(*i as f64),
                _ => None,
            }
        }

        /// Witness read: bool view of the folded value, if any.
        pub(super) fn bool_of(&self, var: &str) -> Option<bool> {
            match self.vals.get(var)? {
                ConcreteValue::Bool(b) => Some(*b),
                ConcreteValue::Int(i) => Some(*i != 0),
                _ => None,
            }
        }

        /// Whitelisted expose/public-decimal read: the raw concrete value
        /// behind a cell (cloned), for rendering the public-inputs file.
        pub(super) fn get(&self, var: &str) -> Option<ConcreteValue> {
            self.vals.get(var).cloned()
        }

        pub(super) fn fold_f(&mut self, v: &str, val: Option<f64>) {
            if let Some(f) = val {
                self.vals.insert(v.to_string(), ConcreteValue::Float(f));
            }
        }
        pub(super) fn fold_i(&mut self, v: &str, val: Option<i128>) {
            if let Some(i) = val {
                self.vals.insert(v.to_string(), ConcreteValue::Int(i));
            }
        }
        pub(super) fn fold_b(&mut self, v: &str, val: Option<bool>) {
            if let Some(b) = val {
                self.vals.insert(v.to_string(), ConcreteValue::Bool(b));
            }
        }
    }
}

impl<'a> NoirEmitter<'a> {
    fn lt_i_bounded_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x < y);
        // Pre-promote Field operands with provable u32 bounds (CSE-d):
        // u32→u128 widening downstream is free, so this trades a single
        // 32-bit RANGE per operand for one 128-bit RANGE per compare site.
        let a = self.maybe_promote_to_u32(a);
        let b = self.maybe_promote_to_u32(b);
        // u32 path when bound-prop has already typed both operands.
        if self.is_u32(&a) && self.is_u32(&b) {
            let v = self.bind_u32_with_bound(&format!("if {} < {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        // Otherwise the u128-cast path. Cast directly through
        // `as_u128` — for u32 operands this is a free widening; for
        // Field operands it emits the implicit < 2^128 range check
        // that catches the signed-Field negative encoding. Going
        // through `as_field` first would force a redundant Field→u128
        // RANGE on already-u32 cells.
        let v = self.bind(&format!(
            "if {} < {} {{ 1 }} else {{ 0 }}",
            self.as_u128(&a),
            self.as_u128(&b)
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn gt_i_bounded_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x > y);
        let a = self.maybe_promote_to_u32(a);
        let b = self.maybe_promote_to_u32(b);
        if self.is_u32(&a) && self.is_u32(&b) {
            let v = self.bind_u32_with_bound(&format!("if {} > {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!(
            "if {} > {} {{ 1 }} else {{ 0 }}",
            self.as_u128(&a),
            self.as_u128(&b)
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }

    fn gt_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.unsigned_cmp(a, b, ">") {
            let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x > y);
            self.fold_b(&v, folded);
            self.set_ival(&v, Some((0, 1)));
            return Ok(v);
        }
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x > y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} > {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        self.use_lib("qgt_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let v = self.bind(&format!(
            "if qgt_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }

    fn sub_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x - y);
        // u32 path is only sound if the result is provably non-negative
        // — u32 subtraction wraps on underflow. Gate on the bound
        // analysis for THIS statement (`current_stmt_bound`): a SubI
        // result only ever receives a bound through a provenance
        // interval with `lo >= 0`, i.e. a structural non-negativity
        // proof. (The previous gate keyed off the WITNESS SAMPLE's sign
        // (`folded >= 0`) — a witness-dependent emission shape.)
        if self.is_u32(a) && self.is_u32(b) {
            if let Some(bound) = self.current_stmt_bound {
                let v = self.bind_u32_with_bound(&format!("{} - {}", a, b), bound);
                self.fold_i(&v, folded);
                return Ok(v);
            }
        }
        let v = self.bind(&format!("{} - {}", self.as_field(a), self.as_field(b)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn mul_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x * y);
        if self.is_u32(a) && self.is_u32(b) {
            if let Some(prod) = self
                .bound_of(a)
                .zip(self.bound_of(b))
                .and_then(|(x, y)| x.checked_mul(y))
                .filter(|p| *p < (1u64 << 32))
            {
                let v = self.bind_u32_with_bound(&format!("{} * {}", a, b), prod);
                self.fold_i(&v, folded);
                return Ok(v);
            }
        }
        let v = self.bind(&format!("{} * {}", self.as_field(a), self.as_field(b)));
        self.fold_i(&v, folded);
        Ok(v)
    }

    /// Affine view of a cell: explicit lineage, a literal constant, or a
    /// fresh single-root term.
    fn aff_of(&self, var: &str) -> AffLin {
        if let Some(vid) = self.vid_of_var.get(var) {
            if let Some(av) = self.prov.affine_of(*vid) {
                if !av.terms.is_empty() {
                    let mut terms = Vec::with_capacity(av.terms.len());
                    let mut ok = true;
                    for (rvid, scale) in &av.terms {
                        match self.var_of_vid.get(rvid) {
                            Some(rv) => terms.push((rv.clone(), *scale)),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        return AffLin {
                            terms,
                            offset: av.offset,
                        };
                    }
                }
            }
        }
        if let Some(a) = self.affine_of.get(var) {
            return a.clone();
        }
        if let Some(c) = self.s_int_of(var) {
            return AffLin { terms: vec![], offset: c as i64 };
        }
        AffLin { terms: vec![(var.to_string(), 1)], offset: 0 }
    }

    /// Signed interval of a cell, if derivable. IR-level cells read the
    /// provenance table LIVE; emitter-internal cells (and, during the
    /// staged migration, locally-derived entries) fall back to the maps.
    fn ival(&self, var: &str) -> Option<(i64, i64)> {
        if let Some(vid) = self.vid_of_var.get(var) {
            if let Some(iv) = self.prov.interval_of(*vid) {
                return Some((iv.lo, iv.hi));
            }
        }
        if let Some(iv) = self.interval_of.get(var) {
            return Some(*iv);
        }
        if let Some(c) = self.s_int_of(var) {
            if let Ok(c) = i64::try_from(c) {
                return Some((c, c));
            }
        }
        // Cells with a known analysis bound are in [0, bound] — the
        // bound sources (AssertU128Lt discharges, LtIBounded, u32 typing)
        // all imply nonnegativity.
        if let Some(b) = self.bound_of(var) {
            if let Ok(b) = i64::try_from(b) {
                return Some((0, b));
            }
        }
        None
    }

    fn set_ival(&mut self, var: &str, iv: Option<(i64, i64)>) {
        if let Some(iv) = iv {
            self.interval_of.insert(var.to_string(), iv);
        }
    }


    /// Bind with an explicit non-Field type annotation (for narrow
    /// unsigned casts; plain bind() annotates Field).
    fn bind_typed(&mut self, expr: &str, ty: &str) -> String {
        let key = format!("{ty}:{expr}");
        if let Some(existing) = self.cse_cache.get(&key) {
            return existing.clone();
        }
        let v = self.fresh();
        let _ = writeln!(self.body, "    let {}: {} = {};", v, ty, expr);
        self.cse_cache.insert(key, v.clone());
        v
    }

    /// Tag a memory-read result with the segment's value hull (if known
    /// and the cell has no tighter interval already).
    fn apply_segment_iv(&mut self, segment_id: u32, out: &str) {
        if self.interval_of.contains_key(out) {
            return;
        }
        // Table-first (the builder maintains the hull at every write);
        // the local map stays as fallback during the staged migration.
        if let Some(h) = self.prov.segment_hull(segment_id) {
            self.interval_of.insert(out.to_string(), (h.lo, h.hi));
            return;
        }
        if let Some(Some(iv)) = self.segment_iv.get(&segment_id) {
            self.interval_of.insert(out.to_string(), *iv);
        }
    }

    /// Interval-driven unsigned comparison: when both operands are
    /// provably nonnegative (interval domain), the u128-cast comparison
    /// replaces the to_signed projection (~140 gates -> ~30; the casts
    /// CSE per operand). Sound regardless of the interval's correctness:
    /// the `as u128` cast itself constrains the operand nonnegative —
    /// a witness violating the derived interval simply fails to prove.
    fn unsigned_cmp(&mut self, a: &str, b: &str, op: &str) -> Option<String> {
        let (al, ah) = self.ival(a)?;
        let (bl, bh) = self.ival(b)?;
        if al < 0 || bl < 0 {
            return None;
        }
        // Narrowest unsigned width covering both hulls: a u8 comparison
        // is ~20 gates (vs ~130 for u128), and the cast range checks CSE
        // per (operand, width). The cast also enforces the hull's
        // nonnegativity claim, so this stays sound even if the derived
        // interval were wrong.
        let max_hi = ah.max(bh).max(1) as u128;
        let bits = 128 - max_hi.leading_zeros();
        let width = match bits {
            0..=8 => "u8",
            9..=16 => "u16",
            17..=32 => "u32",
            33..=64 => "u64",
            _ => "u128",
        };
        let (fa, fb) = (self.as_field(a), self.as_field(b));
        let ca = self.bind_typed(&format!("{fa} as {width}"), width);
        let cb = self.bind_typed(&format!("{fb} as {width}"), width);
        Some(self.bind(&format!("if {ca} {op} {cb} {{ 1 }} else {{ 0 }}")))
    }

    /// Bind a sum of product terms in chunks (Noir rejects monolithic
    /// expressions with hundreds of terms: "Try to simplify expressions").
    fn bind_sum(&mut self, parts: &[String]) -> String {
        const CHUNK: usize = 8;
        let mut acc: Option<String> = None;
        for chunk in parts.chunks(CHUNK) {
            let body = chunk.join(" + ");
            let expr = match &acc {
                Some(a) => format!("{a} + {body}"),
                None => body,
            };
            acc = Some(self.bind(&expr));
        }
        acc.unwrap_or_else(|| "0".to_string())
    }

    /// Cached eq(root, v) indicator, shared across all reads/writes on
    /// the same root. Structural provenance widens the sharing: a
    /// unit-scale single-root affine view shifts the request
    /// (`[x + c == v]` is `[x == v - c]`) and a constant clamp forwards
    /// strictly-inside values to its source (`[min(x,K) == v]` is
    /// `[x == v]` for `v < K`, and `[x >= K]` at the bound), so the
    /// per-row one-hots of a windowed gather all collapse onto the
    /// window start's single indicator family.
    fn indicator(&mut self, root: &str, v: i64) -> String {
        if let Some(i) = self.ind_cache.get(&(root.to_string(), v)) {
            return i.clone();
        }
        let mut base = root.to_string();
        let mut val = v;
        let mut boundary: Option<(String, i64, bool)> = None;
        for _ in 0..8 {
            let Some(vid) = self.vid_of_var.get(&base).copied() else {
                break;
            };
            if let Some(av) = self.prov.affine_of(vid) {
                if av.terms.len() == 1 && av.terms[0].1 == 1 && av.offset != 0 {
                    if let Some(b2) = self.var_of_vid.get(&av.terms[0].0).cloned() {
                        if b2 != base {
                            val -= av.offset;
                            base = b2;
                            continue;
                        }
                    }
                }
            }
            if let Some(cl) = self.prov.clamp_of(vid) {
                if let Some(b2) = self.var_of_vid.get(&cl.source).cloned() {
                    if b2 != base {
                        if (cl.is_min && val < cl.bound) || (!cl.is_min && val > cl.bound) {
                            base = b2;
                            continue;
                        }
                        if val == cl.bound && cl.bound >= 0 && self.is_u32(&b2) {
                            boundary = Some((b2, cl.bound, cl.is_min));
                        }
                    }
                }
            }
            break;
        }
        let ind = if let Some((src, k, is_min)) = boundary {
            // Saturated clamp slot: the comparison form. `src` is
            // u32-bound (nonnegative), so the integer comparison agrees
            // with the signed min/max semantics.
            let op = if is_min { ">=" } else { "<=" };
            self.bind(&format!("if {src} {op} {k} {{ 1 }} else {{ 0 }}"))
        } else if base != root || val != v {
            self.indicator(&base, val)
        } else {
            let rf = self.as_field(root);
            let expr = if v >= 0 {
                format!("if {rf} == {v} {{ 1 }} else {{ 0 }}")
            } else {
                format!("if {rf} + {} == 0 {{ 1 }} else {{ 0 }}", -v)
            };
            self.bind(&expr)
        };
        self.ind_cache.insert((root.to_string(), v), ind.clone());
        ind
    }

    /// Assert the indicator family over `values` is one-hot (exactly one
    /// fires). Replaces per-read bound checks: a witness root outside the
    /// enumerated set makes the proof fail instead of silently gathering
    /// the init value. Emitted once per (root, value set).
    fn assert_one_hot(&mut self, root: &str, values: &[i64]) {
        let key = (root.to_string(), values.to_vec());
        if self.membership_done.contains(&key) {
            return;
        }
        let inds: Vec<String> = values.iter().map(|&v| self.indicator(root, v)).collect();
        let parts: Vec<String> = inds.iter().map(|i| self.as_field(i)).collect();
        let sum = self.bind_sum(&parts);
        let _ = writeln!(self.body, "    assert({} == 1);", sum);
        self.membership_done.insert(key);
    }

    /// Feasible root values v with 0 <= v*scale + resid < len, clipped to
    /// the root's derived interval when known.
    fn feasible(scale: i64, resid: i64, len: usize, root_iv: Option<(i64, i64)>) -> Vec<i64> {
        let mut out = Vec::new();
        if scale == 0 {
            return out;
        }
        for slot in 0..len as i64 {
            let num = slot - resid;
            if num % scale == 0 {
                let v = num / scale;
                if let Some((lo, hi)) = root_iv {
                    if v < lo || v > hi {
                        continue;
                    }
                }
                out.push(v);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Stage-1 plane for factored two-root gathers: sum over feasible v of
    /// ind(root, v) * mem[v*scale + resid]. Cached by (segment, version,
    /// root, scale, resid).
    fn gather_plane(
        &mut self,
        segment_id: u32,
        root: &str,
        scale: i64,
        resid: i64,
        init: &str,
    ) -> String {
        let ver = self.write_version.get(&segment_id).copied().unwrap_or(0);
        let key = (segment_id, ver, root.to_string(), scale, resid);
        if let Some(p) = self.plane_cache.get(&key) {
            return p.clone();
        }
        let len = self.memories.get(&segment_id).map_or(0, |m| m.len());
        let root_iv = self.ival(root);
        let vs = Self::feasible(scale, resid, len, root_iv);
        let mut parts: Vec<String> = Vec::with_capacity(vs.len());
        for &v in &vs {
            let slot = (v * scale + resid) as usize;
            let cell = self.memories.get(&segment_id).unwrap()[slot].clone();
            let ind = self.indicator(root, v);
            parts.push(format!("{} * {}", self.as_field(&ind), self.as_field(&cell)));
        }
        let plane = if parts.is_empty() {
            init.to_string()
        } else {
            self.bind_sum(&parts)
        };
        self.plane_cache.insert(key, plane.clone());
        plane
    }

    fn add_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x + y);
        // u32 path: both operands already bound u32 AND sum still fits.
        // Bound-propagation lets axis-dim chains (`row_has`, sums, scan
        // accumulators) stay u32 without any RANGE cast.
        //
        // (Measured 2026-06-12: flipping these chains to Field adds +
        // endpoint promotion casts regressed the bbox family by ~1,000
        // gates each — on current bb a checked u32 add is cheaper than
        // a Field add plus its share of a Field→u32 truncation cast at
        // the compare sink. Keep the u32-chain policy.)
        if self.is_u32(a) && self.is_u32(b) {
            if let Some(sum) = self
                .bound_of(a)
                .zip(self.bound_of(b))
                .and_then(|(x, y)| x.checked_add(y))
                .filter(|s| *s < (1u64 << 32))
            {
                let v = self.bind_u32_with_bound(&format!("{} + {}", a, b), sum);
                self.fold_i(&v, folded);
                return Ok(v);
            }
        }
        // u32_amortizable + at-least-one-operand-u32: extend the u32
        // add-chain across a single Field neighbour by promoting it
        // (one CSE-d 32-bit RANGE) so accumulators keep flowing in
        // u32. The guard avoids breaking pure-Field chains where
        // neither operand is u32 yet — promoting both would pay two
        // RANGE casts per add and rarely pays back (cf. 0510).
        if self.u32_amortizable && (self.is_u32(a) || self.is_u32(b)) {
            let ap = self.maybe_promote_to_u32(a);
            let bp = self.maybe_promote_to_u32(b);
            if self.is_u32(&ap) && self.is_u32(&bp) {
                if let Some(sum) = self
                    .bound_of(&ap)
                    .zip(self.bound_of(&bp))
                    .and_then(|(x, y)| x.checked_add(y))
                    .filter(|s| *s < (1u64 << 32))
                {
                    let v = self.bind_u32_with_bound(&format!("{} + {}", ap, bp), sum);
                    self.fold_i(&v, folded);
                    return Ok(v);
                }
            }
        }
        let v = self.bind(&format!("{} + {}", self.as_field(a), self.as_field(b)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn lt_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.unsigned_cmp(a, b, "<") {
            let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x < y);
            self.fold_b(&v, folded);
            self.set_ival(&v, Some((0, 1)));
            return Ok(v);
        }
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x < y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} < {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        // Field path: signed-aware comparison via to_signed. Bind the
        // {0, 1} result as u32 so downstream stays u32-clean.
        self.use_lib("qlt_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let v = self.bind(&format!(
            "if qlt_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }

    pub fn new(
        inputs: Vec<EmittedInput>,
        witness: ResolvedWitness,
        params: ProvingParams,
        stmt_bounds: super::bound_analysis::BoundMap,
        intrinsic_count: usize,
        provenance: &'a crate::provenance::ProvenanceTable,
    ) -> NoirEmitter<'a> {
        // u32 amortization heuristic: instantiating bb's u32/u128 RANGE
        // lookup tables pulls in a fixed cost (~2700-3500 gates in the
        // tested bb 5.0 nightly). Each `(field as u32)` elimination
        // saves ~10-15 gates. We turn aggressive Field→u32 promotion
        // on only when the IR contains *enough* operations that
        // intrinsically lower to a u32/u128 RANGE check
        // (`AssertU128Lt`, `LtIBounded`, `GtIBounded`, `IndexInBox`)
        // for the table cost to be already sunk.
        //
        // Threshold derivation: 0508 (high intrinsic op count) saved
        // ~880 gates; 0507/0509/0510 with moderate counts saved
        // 1200-1400. 0355 (a handful of intrinsic ops) lost ~707 to
        // table overhead. 0503/0504/0505 (1 intrinsic op each, mostly
        // a single AssertU128Lt sitting in a memory-trace seal) lost
        // ~3500. The 20 threshold admits the moderate-count winners
        // while excluding the single-intrinsic regressors.
        let u32_amortizable = intrinsic_count >= 20;
        Self {
            prov: provenance,
            vid_of_var: HashMap::new(),
            var_of_vid: HashMap::new(),
            inputs,
            witness,
            body: String::new(),
            next_id: 0,
            used_libs: BTreeMap::new(),
            exposes: Vec::new(),
            public_decimal: Vec::new(),
            witness_vals: wfold::WitnessFold::default(),
            affine_of: HashMap::new(),
            ind_cache: HashMap::new(),
            membership_done: std::collections::HashSet::new(),
            plane_cache: HashMap::new(),
            write_version: HashMap::new(),
            interval_of: HashMap::new(),
            segment_iv: HashMap::new(),
            lt_zero_of: HashMap::new(),
            add_const_of: HashMap::new(),
            norm_index_of: HashMap::new(),
            needs_lib: HashMap::new(),
            params,
            memories: HashMap::new(),
            memory_init: HashMap::new(),
            signed_proj: HashMap::new(),
            emitted_u128_asserts: HashSet::new(),
            cse_cache: HashMap::new(),
            cell_bound: HashMap::new(),
            cell_type: HashMap::new(),
            literal_vals: HashMap::new(),
            stmt_bounds,
            current_stmt_bound: None,
            u32_amortizable,
        }
    }

    // ── Bound-driven u32 emit helpers ────────────────────────────────
    // These let the integer-arithmetic / compare ops decide whether to
    // bind a result as `u32` (avoiding a `Field as u32` RANGE blackbox)
    // or as `Field` (the default safe path). See `CellType` for the
    // policy; the savings show up on every cell that participates in
    // bbox / slice / index arithmetic against a u32-bound axis dim.

    fn bound_of(&self, cell: &str) -> Option<u64> {
        self.cell_bound.get(cell).copied()
    }

    fn is_u32(&self, cell: &str) -> bool {
        matches!(self.cell_type.get(cell), Some(CellType::U32))
    }

    fn set_bound(&mut self, cell: &str, bound: u64) {
        self.cell_bound.insert(cell.to_string(), bound);
    }

    fn mark_u32(&mut self, cell: &str) {
        self.cell_type.insert(cell.to_string(), CellType::U32);
    }

    /// Wrap `cell` so the expression is `Field`-typed at the call site.
    /// `u32 -> Field` is a zero-cost embedding in Noir; `Field` cells
    /// pass through unchanged.
    fn as_field(&self, cell: &str) -> String {
        if self.is_u32(cell) {
            format!("({} as Field)", cell)
        } else {
            cell.to_string()
        }
    }

    /// Wrap `cell` so the expression is `u128`-typed at the call site.
    /// For `u32` cells this is a free widening cast (no RANGE check);
    /// for `Field` cells it emits a single 128-bit RANGE check.
    /// Direct form — avoids the `(u32 as Field) as u128` chain whose
    /// Field intermediate would force a redundant Field→u128 RANGE
    /// check on a value Noir already knows fits 32 bits.
    fn as_u128(&self, cell: &str) -> String {
        format!("({} as u128)", cell)
    }

    /// Route a Field-typed cell with a known u32 bound through a one-time
    /// `u32` cast, so downstream u128 widenings become free and CSE can
    /// deduplicate the cast across many compare sites. Returns the new
    /// u32-typed SSA name; subsequent uses see `is_u32(returned) == true`.
    ///
    /// Gated on `u32_amortizable` — the per-cast 32-bit RANGE check is
    /// only worth paying when the kernel has enough u32-eligible cells
    /// (and existing u32 ops) for the lookup table to be amortized.
    /// In small kernels we leave the cell as Field and let the existing
    /// u128 path handle it (no extra table cost).
    fn maybe_promote_to_u32(&mut self, cell: &str) -> String {
        if self.is_u32(cell) {
            return cell.to_string();
        }
        if !self.u32_amortizable {
            return cell.to_string();
        }
        let bound = match self.bound_of(cell) {
            Some(b) if b <= u32::MAX as u64 => b,
            _ => return cell.to_string(),
        };
        // Bind `(cell as u32)` once; CSE collapses repeated promotions
        // of the same source cell so the 32-bit RANGE only fires once.
        let v = self.bind_u32_with_bound(&format!("{} as u32", cell), bound);
        // Value-preserving rebind: carry the affine lineage so address
        // chains through the promotion keep their root for the shared
        // indicator gathers.
        if let Some(aff) = self.affine_of.get(cell).cloned() {
            self.affine_of.insert(v.clone(), aff);
        } else if !self.literal_vals.contains_key(cell) {
            self.affine_of.insert(
                v.clone(),
                AffLin { terms: vec![(cell.to_string(), 1)], offset: 0 },
            );
        }
        if let Some(iv) = self.interval_of.get(cell).copied() {
            self.interval_of.insert(v.clone(), iv);
        }
        // TEST: vid copy across the promotion rebind
        if let Some(vid) = self.vid_of_var.get(cell).copied() {
            self.vid_of_var.insert(v.clone(), vid);
        }
        v
    }

    /// Apply the IR-analysis bound to a freshly-bound cell, regardless
    /// of how the producing method emitted it. This populates
    /// `cell_bound` so downstream emit-time tracking has richer info
    /// even when the producing method chose Field; it does NOT
    /// retroactively flip the cell's binding type.
    fn annotate_with_analysis_bound(&mut self, cell: &str) {
        if let Some(b) = self.current_stmt_bound {
            // Only overwrite if the new bound is tighter, or no bound exists.
            let keep = self.cell_bound.get(cell).copied();
            match keep {
                Some(prev) if prev <= b => {}
                _ => {
                    self.cell_bound.insert(cell.to_string(), b);
                }
            }
        }
    }

    /// Bind `expr` as a `u32`-typed cell with the given upper bound.
    /// Caller asserts the expression on the RHS evaluates within
    /// `0..=bound` (which the bound-propagation in the integer ops
    /// guarantees by construction).
    fn bind_u32_with_bound(&mut self, expr: &str, bound: u64) -> String {
        if let Some(existing) = self.cse_cache.get(expr) {
            return existing.clone();
        }
        let v = self.fresh();
        let _ = writeln!(self.body, "    let {}: u32 = {};", v, expr);
        self.cse_cache.insert(expr.to_string(), v.clone());
        self.mark_u32(&v);
        self.set_bound(&v, bound);
        v
    }

    fn fresh(&mut self) -> String {
        let n = self.next_id;
        self.next_id += 1;
        format!("v{}", n)
    }

    fn bind(&mut self, expr: &str) -> String {
        // CSE: if the same Noir expression text was already bound to an
        // SSA name in this body, reuse it. Sound because every binding is
        // a pure function of its operand names — same name string in,
        // same value out. Saves a fresh ID + a duplicate let-line, and
        // (transitively) any downstream assert/expose duplication.
        if let Some(existing) = self.cse_cache.get(expr) {
            return existing.clone();
        }
        let v = self.fresh();
        // Explicit Field annotation: nargo's type inference recurses
        // through chains of untyped lets unified via integer-literal type
        // variables and panics ("Type recursion limit reached") past a few
        // thousand bindings. Every bind() expression is Field-valued by
        // construction (bools are 0/1 Fields; typed lets go through
        // bind_u32_with_bound).
        let _ = writeln!(self.body, "    let {}: Field = {};", v, expr);
        self.cse_cache.insert(expr.to_string(), v.clone());
        // NOTE: the IR-analysis bound is NOT stamped here. `bind()` runs
        // for every *intermediate* cell produced while dispatching a
        // statement, but `current_stmt_bound` describes only the
        // statement's RESULT. Stamping it on intermediates (especially
        // indicator cells, which are cached in `ind_cache` and reused
        // across statements) could poison a later interval query with a
        // spuriously tight bound. The bound is applied to the actual
        // result cell in `on_stmt_result` instead.
        v
    }

    /// Bind a compile-time STRUCTURAL constant: the value is known at
    /// circuit-build time, so it lands in `literal_vals` (the only map the
    /// structural accessors read), never in the witness fold cache.
    fn bind_val(&mut self, expr: &str, val: ConcreteValue) -> String {
        let v = self.bind(expr);
        self.literal_vals.insert(v.clone(), val);
        v
    }

    // ── Structural-constant accessors (BUILD-time only) ──────────────
    // These read `literal_vals` exclusively. They are the ONLY accessors
    // an emission decision (a branch on a value, a gate elision) is
    // allowed to consult. By construction `literal_vals` never receives a
    // witness-folded value, so keying structure on these can never
    // specialise the circuit to one witness. The plain witness reads
    // (`int_of`/`float_of`/`bool_of`) live on `wfold::WitnessFold` and are
    // reachable only for bookkeeping (fold/expose).

    fn s_int_of(&self, var: &str) -> Option<i128> {
        match self.literal_vals.get(var)? {
            ConcreteValue::Int(i) => Some(*i),
            ConcreteValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }
    fn s_float_of(&self, var: &str) -> Option<f64> {
        match self.literal_vals.get(var)? {
            ConcreteValue::Float(f) => Some(*f),
            ConcreteValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    fn s_bool_of(&self, var: &str) -> Option<bool> {
        match self.literal_vals.get(var)? {
            ConcreteValue::Bool(b) => Some(*b),
            ConcreteValue::Int(i) => Some(*i != 0),
            _ => None,
        }
    }

    // ── Witness-fold accessors (bookkeeping only) ────────────────────
    // Thin forwarders to the privacy-enforced `witness_vals` store. The
    // results MUST flow only into `fold_*` / public-input rendering, never
    // into a structural decision — that is exactly what the `wfold` module
    // boundary + the two-witness diff test guard.

    fn float_of(&self, var: &str) -> Option<f64> {
        self.witness_vals.float_of(var)
    }
    fn int_of(&self, var: &str) -> Option<i128> {
        self.witness_vals.int_of(var)
    }
    fn bool_of(&self, var: &str) -> Option<bool> {
        self.witness_vals.bool_of(var)
    }
    fn fold_f(&mut self, v: &str, val: Option<f64>) {
        self.witness_vals.fold_f(v, val);
    }
    fn fold_i(&mut self, v: &str, val: Option<i128>) {
        self.witness_vals.fold_i(v, val);
    }
    fn fold_b(&mut self, v: &str, val: Option<bool>) {
        self.witness_vals.fold_b(v, val);
    }

    /// Register a STRUCTURAL literal value for a cell that was derived
    /// purely from other structural literals (e.g. `abs_f` of a literal,
    /// `float_cast` of a literal int). Mirrors what the old code did by
    /// pairing a `fold_*` write with a `literal_cells.insert`, but routes
    /// the value into `literal_vals` so the structural accessors can see
    /// it without ever consulting the witness cache.
    fn fold_literal(&mut self, v: &str, val: ConcreteValue) {
        self.literal_vals.insert(v.to_string(), val);
    }

    fn use_lib(&mut self, sym: &'static str) {
        self.needs_lib.insert(sym, true);
        self.used_libs.insert(sym, ());
    }

    /// Emit `let (sN, mN) = to_signed(cell);` once per unique `cell` and
    /// cache the resulting (sign_var, mag_var) for later comparisons.
    /// Returns the cached names on subsequent calls — this is the
    /// dedup that turns 23-way slice masks from ~3200 gates into ~250.
    ///
    /// Fast path: if the cell has a known constant-folded non-negative
    /// integer value, short-circuit by emitting literal `(false, N)`
    /// instead of a `to_signed` call. `to_signed` does a 126-bit RANGE
    /// check on the magnitude (~140 ACIR gates) — pure overhead when the
    /// value is a compile-time constant the prover already knows fits.
    /// Hits the np.insert / mux-indexed pattern where 100+ constant
    /// indices each compare against a single witness pos (cf. 0363/
    /// 0364/0365 each had 100 redundant to_signed projections).
    fn project_signed(&mut self, cell: &str) -> (String, String) {
        if let Some(p) = self.signed_proj.get(cell) {
            return p.clone();
        }
        // Callers commonly pass the `as_field`-wrapped form
        // `"(vN as Field)"` for u32 cells. Strip that wrapper so the
        // int_of constant-fold lookup (keyed on the raw SSA name)
        // hits — without this, the fast path never fires on u32
        // constants which is exactly the np.insert / mux-indexed
        // pattern we're trying to optimise.
        let raw = cell
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(" as Field)"))
            .unwrap_or(cell);
        // Fold IR-literal constants into a direct (sign, magnitude)
        // pair without going through `to_signed`. **Read via the
        // structural accessors `s_int_of`/`s_float_of`** — these read the
        // `literal_vals` store ONLY and so never surface witness-folded
        // values (which live in the privacy-enforced `witness_vals`).
        // Inlining a witness-derived value into the circuit would
        // specialise it to one witness — *unsound* for any other input.
        // `literal_vals` is populated only by `constant_int`/
        // `constant_float`/`constant_bool` and structural-literal
        // derivations, where the value really is a compile-time constant.
        //
        // `to_signed`'s convention is `(sign, mag)` with `sign == true`
        // for negative values and `mag == |v|` (see noir_fpc lib). We
        // mirror it for both signs:
        //
        //   - Int(n) literal: sign = n < 0, magnitude = |n| (the Field
        //     value is `n`, or `BN254_PRIME - |n|` when negative).
        //   - Float(f) literal: magnitude = |f| * 2^32 (Q-scaling — the
        //     Field value is the quantized form), sign = f < 0.
        //   Bool literals are non-negative {0,1} and fold via s_int_of.
        let folded: Option<(bool, i128)> = if let Some(n) = self.s_int_of(raw) {
            let sign = n < 0;
            // `i128::MIN.unsigned_abs()` would overflow i128 on cast;
            // guard it (no real literal hits this, but stay total).
            match i128::try_from(n.unsigned_abs()) {
                Ok(mag) => Some((sign, mag)),
                Err(_) => None,
            }
        } else if let Some(f) = self.s_float_of(raw) {
            if f.is_finite() {
                let scaled = (f.abs() * (Q_U64 as f64)).round();
                if scaled < (1i128 << 126) as f64 {
                    Some((f < 0.0, scaled as i128))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some((sign, mag)) = folded {
            if mag >= 0 && mag < (1i128 << 126) {
                let s_lit = if sign { "true" } else { "false" }.to_string();
                let m_lit = mag.to_string();
                self.signed_proj
                    .insert(cell.to_string(), (s_lit.clone(), m_lit.clone()));
                return (s_lit, m_lit);
            }
        }
        self.use_lib("to_signed");
        let s = self.fresh();
        let m = self.fresh();
        let _ = writeln!(self.body, "    let ({}, {}) = to_signed({});", s, m, cell);
        self.signed_proj
            .insert(cell.to_string(), (s.clone(), m.clone()));
        (s, m)
    }

    /// Look up an InputPath against the witness and return its decimal Field
    /// representation + concrete value. Used to populate the public-inputs
    /// decimal list for `ExposePublic` ops.
    fn resolve_path_concrete(&self, path: &InputPath) -> Option<(String, ConcreteValue)> {
        let param = self
            .witness
            .input
            .params
            .iter()
            .find(|p| p.name == path.param)?;
        let node = walk_path(&param.value, &path.segments)?;
        Some(input_node_concrete(node))
    }

    fn render_use_decls(&self) -> String {
        let mut syms: Vec<&'static str> = self.used_libs.keys().copied().collect();
        syms.sort();
        if syms.is_empty() {
            String::new()
        } else {
            format!("use noir_fpc::{{{}}};\n\n", syms.join(", "))
        }
    }
}

fn walk_path<'a>(node: &'a InputNode, segs: &[PathSegment]) -> Option<&'a InputNode> {
    if segs.is_empty() {
        return Some(node);
    }
    match (&segs[0], node) {
        (PathSegment::Index(i), InputNode::Array { elements, .. }) => {
            // Flat-leaf indexing.
            let leaves = flat_leaves(elements);
            leaves
                .get(*i as usize)
                .copied()
                .and_then(|n| walk_path(n, &segs[1..]))
        }
        (PathSegment::Index(i), InputNode::Sequence(items)) => items
            .get(*i as usize)
            .and_then(|n| walk_path(n, &segs[1..])),
        (PathSegment::Inner, InputNode::Hashed { inner, .. }) => walk_path(inner, &segs[1..]),
        _ => None,
    }
}

fn input_node_concrete(node: &InputNode) -> (String, ConcreteValue) {
    match node {
        InputNode::Int(v) => (
            signed_to_field_decimal(*v as i128),
            ConcreteValue::Int(*v as i128),
        ),
        InputNode::Float(v) => (quantize_to_field_string(*v), ConcreteValue::Float(*v)),
        InputNode::Bool(v) => (
            if *v { "1".into() } else { "0".into() },
            ConcreteValue::Bool(*v),
        ),
        InputNode::Hex(s) => (format!("0x{}", s), ConcreteValue::Field(format!("0x{}", s))),
        _ => ("0".into(), ConcreteValue::Field("0".into())),
    }
}

// ---------------------------------------------------------------------------
// Path-state for read_input
// ---------------------------------------------------------------------------
//
// Each call to read_input emits one of:
//   let vN = a;              (scalar)
//   let vN = a[5];           (array element)
// We do not bind redundant indexes — each read is a new SSA var so the
// IR's ValueId semantics are preserved.

fn input_index_expr(
    path: &InputPath,
    inputs: &[EmittedInput],
) -> Result<String, ProvingError> {
    let kind = inputs
        .iter()
        .find(|i| i.name == path.param)
        .map(|i| &i.kind);
    match kind {
        Some(InputKind::ScalarField) | Some(InputKind::Bool) | None => Ok(path.param.clone()),
        Some(InputKind::ArrayField { .. }) => {
            // Zinnia's IR-gen flattens `path.segments` to a single Index
            // for NDArray reads on every kernel we emit — the witness
            // Array reaching us is already flat-leaves. The single-axis
            // case is therefore the only sound one to index here.
            if path.segments.len() == 1 {
                if let Some(PathSegment::Index(i)) = path.segments.first() {
                    return Ok(format!("{}[{}]", path.param, i));
                }
            }
            // Multi-axis fallback removed: the previous code summed index
            // segments (`flat += i`) which is wrong for any genuine
            // multi-axis input (strides are unavailable in `EmittedInput`).
            // Fail loud rather than emit a silently-wrong flat index. If a
            // real kernel ever lands here, thread shape/stride info through
            // `EmittedInput` and compute the true flat index.
            Err(ProvingError::UnsupportedIR {
                op_name: format!(
                    "noir-emit:multi-axis input index ({} segments) on param `{}` — \
                     IR-gen did not flatten; flat index unavailable without strides",
                    path.segments.len(),
                    path.param
                ),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Synthesizer impl
// ---------------------------------------------------------------------------

type CellRef = String;

macro_rules! noop_unsupported {
    ($self:ident, $op:literal) => {{
        Err(ProvingError::UnsupportedIR {
            op_name: format!("noir-emit:{}", $op),
        })
    }};
}

impl Synthesizer for NoirEmitter<'_> {
    type CellRef = CellRef;

    fn on_stmt_result(
        &mut self,
        _stmt_id: u32,
        value_id: crate::types::ValueId,
        cell: &CellRef,
    ) {
        // Name ↔ ValueId correspondence for live provenance-table reads.
        self.vid_of_var.insert(cell.clone(), value_id);
        self.var_of_vid.insert(value_id, cell.clone());
        // Stamp the IR-analysis bound on the statement RESULT only. This
        // populates `cell_bound` so downstream emit-time u32-path
        // conditions see the richer info even when the producing method
        // chose Field. Doing it here (rather than inside `bind()`) keeps
        // the bound off shared intermediate cells such as cached
        // indicators, which would otherwise inherit a statement-result
        // bound that does not describe them and poison later interval
        // queries.
        self.annotate_with_analysis_bound(cell);
    }

    fn on_stmt_begin(&mut self, stmt_id: u32) {
        // Stash the IR-analysis bound for the result of this statement
        // so the producing method can choose `u32` over `Field` when
        // the analysis proves the value fits.
        self.current_stmt_bound = self.stmt_bounds.get(&stmt_id).copied();
    }

    fn constant_int(&mut self, value: i64) -> Result<Self::CellRef, ProvingError> {
        let expr = signed_to_field_decimal(value as i128);
        // Bind small non-negative constants as u32 so any chain of
        // integer ops they participate in (counters, axis dims, loop
        // indices, sentinel-tagged accumulators) stays in u32-land and
        // dodges the `Field as u32` RANGE blackbox on downstream
        // compares. Negative or large constants stay on Field.
        if value >= 0 && (value as u64) < (1u64 << 32) {
            let v = self.bind_u32_with_bound(&value.to_string(), value as u64);
            self.fold_literal(&v, ConcreteValue::Int(value as i128));
            return Ok(v);
        }
        // `bind_val` registers the structural literal in `literal_vals`.
        let v = self.bind_val(&expr, ConcreteValue::Int(value as i128));
        Ok(v)
    }

    fn constant_float(&mut self, value: f64) -> Result<Self::CellRef, ProvingError> {
        let expr = quantize_to_field_string(value);
        let v = self.bind_val(&expr, ConcreteValue::Float(value));
        Ok(v)
    }

    fn constant_bool(&mut self, value: bool) -> Result<Self::CellRef, ProvingError> {
        // Bool constants are also typed as Field for our purposes — the
        // logical-not / and / or routes treat 0/1 in Field.
        let expr = if value { "1" } else { "0" };
        let v = self.bind_val(expr, ConcreteValue::Bool(value));
        Ok(v)
    }

    fn add_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let out = self.add_i_inner(a, b)?;
        if let Some(c) = self.s_int_of(b) {
            self.add_const_of.insert(out.clone(), (a.clone(), c as i64));
        } else if let Some(c) = self.s_int_of(a) {
            self.add_const_of.insert(out.clone(), (b.clone(), c as i64));
        }
        return Ok(out);
    }

    fn sub_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.sub_i_inner(a, b)
    }

    fn mul_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.mul_i_inner(a, b)
    }

    fn div_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded =
            self.int_of(a)
                .zip(self.int_of(b))
                .and_then(|(x, y)| if y == 0 { None } else { Some(x / y) });
        // Native u32 division when both operands fit u32 — Plonkish u32 div is
        // sound by construction (the witness reconstruction `q*d + r == n`
        // with `r < d` is implicit) and emits no cast-side RANGE blackbox.
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self.bound_of(a).unwrap_or(u32::MAX as u64);
            let v = self.bind_u32_with_bound(&format!("{} / {}", a, b), bound);
            self.fold_i(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!(
            "(({} as u128) / ({} as u128)) as Field",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn floor_div_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).and_then(|(x, y)| {
            if y == 0 {
                None
            } else {
                Some(x.div_euclid(y))
            }
        });
        // u32 floor-div is sound by construction (see `div_i`).
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self.bound_of(a).unwrap_or(u32::MAX as u64);
            let v = self.bind_u32_with_bound(&format!("{} / {}", a, b), bound);
            self.fold_i(&v, folded);
            return Ok(v);
        }
        // Field-path floor division. Emit the self-sound `q_floor_div`
        // (asserts `q*d + r == n` with `0 <= r < |d|`, pinning a unique
        // quotient for any sign). User-level `a // b` lowers to a bare IR
        // FloorDivI with NO downstream circuit pin, so the previously-used
        // unconstrained `q_floor_div_hint` let a malicious prover substitute
        // any quotient (C04). The emitter has no witness-independent signal
        // here to distinguish the (rare) compiler-generated divisions whose
        // quotient is already pinned downstream, so it always emits the
        // self-contained sound form. See the commit message for the gate
        // cost of this on the pinned reshape cases (still < 0.5% of the
        // sweep) and the pinned-marker follow-up.
        self.use_lib("q_floor_div");
        let v = self.bind(&format!(
            "q_floor_div({}, {})",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn mod_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // Euclidean remainder (matches the IR's `rem_euclid` ground truth and
        // the sound `q_mod` circuit form below) -- NOT Rust's truncated `%`,
        // which would disagree for negative operands.
        let folded =
            self.int_of(a)
                .zip(self.int_of(b))
                .and_then(|(x, y)| if y == 0 { None } else { Some(x.rem_euclid(y)) });
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self
                .bound_of(b)
                .unwrap_or(u32::MAX as u64)
                .saturating_sub(1);
            let v = self.bind_u32_with_bound(&format!("{} % {}", a, b), bound);
            self.fold_i(&v, folded);
            return Ok(v);
        }
        // Field-path modulo. Emit the self-sound `q_mod` for the same
        // soundness reason as floor division (C04): user-level `a % b` lowers
        // to a bare IR ModI with no downstream pin, so the unconstrained
        // `q_mod_hint` was substitutable. `q_mod` asserts `q*d + r == n` with
        // `0 <= r < |d|`, pinning r uniquely.
        self.use_lib("q_mod");
        let v = self.bind(&format!(
            "q_mod({}, {})",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn pow_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "pow_i")
    }
    fn abs_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qabs_i");
        let folded = self.int_of(a).map(|x| x.abs());
        let v = self.bind(&format!("qabs_i({})", self.as_field(a)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn sign_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsign_i");
        let folded = self.int_of(a).map(|x| x.signum());
        let v = self.bind(&format!("qsign_i({})", self.as_field(a)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn inv_i(&mut self, _a: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "inv_i")
    }

    fn bit_and_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "bit_and_i")
    }
    fn bit_or_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "bit_or_i")
    }
    fn bit_xor_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "bit_xor_i")
    }
    fn shl_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "shl_i")
    }
    fn shr_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "shr_i")
    }
    fn bit_not_i(&mut self, _a: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "bit_not_i")
    }

    fn add_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qadd");
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x + y);
        let v = self.bind(&format!("qadd({}, {})", self.as_field(a), self.as_field(b)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn sub_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsub");
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x - y);
        let v = self.bind(&format!("qsub({}, {})", self.as_field(a), self.as_field(b)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn mul_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qmul");
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x * y);
        let v = self.bind(&format!("qmul({}, {})", self.as_field(a), self.as_field(b)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn product_sum_f(&mut self, pairs: &[(CellRef, CellRef)]) -> Result<CellRef, ProvingError> {
        self.use_lib("qscale_down_raw");
        let mut acc = format!(
            "({} * {})",
            self.as_field(&pairs[0].0),
            self.as_field(&pairs[0].1)
        );
        for (a, b) in &pairs[1..] {
            acc = format!("({} + ({} * {}))", acc, self.as_field(a), self.as_field(b));
        }
        let mut folded: Option<f64> = Some(0.0);
        for (a, b) in pairs {
            match (self.float_of(a), self.float_of(b), folded) {
                (Some(x), Some(y), Some(s)) => folded = Some(s + x * y),
                _ => {
                    folded = None;
                    break;
                }
            }
        }
        let v = self.bind(&format!("qscale_down_raw({})", acc));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn div_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qdiv");
        let folded = self.float_of(a).zip(self.float_of(b)).and_then(|(x, y)| {
            if y == 0.0 {
                None
            } else {
                Some(x / y)
            }
        });
        let v = self.bind(&format!("qdiv({}, {})", self.as_field(a), self.as_field(b)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn floor_div_f(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "floor_div_f")
    }
    fn mod_f(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "mod_f")
    }
    fn pow_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qpow");
        let folded = self
            .float_of(a)
            .zip(self.float_of(b))
            .map(|(x, y)| x.powf(y));
        let v = self.bind(&format!("qpow({}, {})", self.as_field(a), self.as_field(b)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn pow_f_int(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // Float base ^ Integer exponent: shared noir_fpc helper does 32-iter
        // square-and-multiply. The exponent is Zinnia's raw signed Field
        // encoding; cast through u128 (which Noir accepts from `Field as u128`
        // for non-negative values) up to u64 for the helper signature.
        // Mirrors the handwritten ds1000_case0329 / 0330 baselines.
        self.use_lib("qpow_int_q32");
        let folded = self
            .float_of(a)
            .zip(self.int_of(b))
            .map(|(x, e)| x.powi(e as i32));
        let v = self.bind(&format!(
            "qpow_int_q32({}, {} as u128 as u64)",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn abs_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qabs");
        let v = self.bind(&format!("qabs({})", self.as_field(a)));
        // Literal propagation: |literal| is itself a compile-time
        // constant. Fold the structural value and carry `literal_vals`
        // membership so downstream fast paths (e.g. project_signed,
        // select zero-branch) fire. Read the STRUCTURAL value only
        // (`s_float_of`), never the witness fold, per the
        // witness-independence rule.
        if let Some(f) = self.s_float_of(a) {
            self.fold_literal(&v, ConcreteValue::Float(f.abs()));
        }
        Ok(v)
    }
    fn sign_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsign");
        let folded = self.float_of(a).map(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
        let v = self.bind(&format!("qsign({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }

    fn eq_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x == y);
        // Normalized-index dual equality: for norm = select(lt(x,0), x+N, x),
        // eq(norm, j) == (x == j) | (x == j - N) — two Field equalities
        // instead of the signed-comparison machinery that the normalization
        // otherwise drags in (~600 ACIR gates per index). The identity is
        // exact ONLY for slot constants in range, i.e. 0 <= j < N: norm lands
        // in [0, N), so the two pre-images of j under normalization are
        // exactly {j, j - N}. For j >= N the `x == j - N` disjunct fires
        // spuriously on the non-negative pre-image j - N (which normalizes to
        // itself, not j); for j < 0 the `x == j` disjunct fires on negative
        // x = j (which normalizes to j + N, not j). Out of range we fall
        // through to the generic equality path.
        {
            let hit = self
                .norm_index_of
                .get(a)
                .cloned()
                .map(|nx| (nx, self.s_int_of(b)))
                .filter(|(_, j)| j.is_some())
                .or_else(|| {
                    self.norm_index_of
                        .get(b)
                        .cloned()
                        .map(|nx| (nx, self.s_int_of(a)))
                        .filter(|(_, j)| j.is_some())
                });
            if let Some(((x, n), Some(j))) = hit {
                let j = j as i64;
                if j >= 0 && j < n {
                    // In range: j - n < 0 always, so the second pre-image is
                    // j - n, written branch-free as (x + (n - j) == 0).
                    let xf = self.as_field(&x);
                    let v = self.bind(&format!(
                        "if ({} == {}) | ({} + {} == 0) {{ 1 }} else {{ 0 }}",
                        xf, j, xf, n - j
                    ));
                    self.fold_b(&v, folded);
                    return Ok(v);
                }
                // j out of [0, N): peephole identity does not hold, fall
                // through to the generic equality path below.
            }
        }
        // Only bind u32 when operands are already u32. Unconditional
        // u32-binding adds a `BLACKBOX::RANGE` per compare on
        // otherwise-Field-pure kernels (0441/0442/0443 regressed +2700
        // each in v10 when we forced u32 here).
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} == {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        // u32_amortizable + at-least-one-operand-u32: bind the {0,1}
        // result as u32 even on Field-vs-u32 compares. The conditional's
        // then/else branches are integer literals so Noir infers u32
        // freely. Bool kept in u32-land lets sum-chain accumulators
        // (`row_has[i]` against a u32 sentinel) stay u32 through add_i
        // without per-cell casts. The at-least-one-u32 guard skips
        // chains where both operands are pure-Field witness values —
        // those promote one accumulator cell per compare without
        // amortising savings downstream (cf. 0510 regression).
        if self.u32_amortizable && (self.is_u32(a) || self.is_u32(b)) {
            let v = self.bind_u32_with_bound(
                &format!(
                    "if {} == {} {{ 1 }} else {{ 0 }}",
                    self.as_field(a),
                    self.as_field(b)
                ),
                1,
            );
            self.fold_b(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!(
            "if {} == {} {{ 1 }} else {{ 0 }}",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn ne_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x != y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} == {} {{ 0 }} else {{ 1 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        if self.u32_amortizable && (self.is_u32(a) || self.is_u32(b)) {
            let v = self.bind_u32_with_bound(
                &format!(
                    "if {} == {} {{ 0 }} else {{ 1 }}",
                    self.as_field(a),
                    self.as_field(b)
                ),
                1,
            );
            self.fold_b(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!(
            "if {} == {} {{ 0 }} else {{ 1 }}",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn lt_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // min/max select recognition now happens at IR level (the
        // provenance table's SelectI deposit); only the lt-zero peephole
        // stays emitter-local.
        let out = self.lt_i_inner(a, b)?;
        if self.s_int_of(b) == Some(0) {
            self.lt_zero_of.insert(out.clone(), a.clone());
        }
        return Ok(out);
    }

    fn lt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        return self.lt_i_bounded_inner(a, b);
    }

    fn gt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        return self.gt_i_bounded_inner(a, b);
    }

    fn index_in_box(
        &mut self,
        triples: &[(CellRef, CellRef, CellRef)],
    ) -> Result<CellRef, ProvingError> {
        // Emit one combined boolean expression `((idx_0 as u128) >= (lo_0 as u128))
        // & ((idx_0 as u128) < (hi_0 as u128)) & ...`. Shares a single u128
        // RANGE table across all axes. Replaces the per-axis
        // `lt_i_bounded + ge + logical_and` chain (one Field-mul per axis
        // per cell) with one combined AND of bools — ~3 gates per axis per
        // cell saved on 2D dynamic slices.
        if triples.is_empty() {
            return self.constant_int(1);
        }
        let mut folded_all = Some(true);
        let mut parts: Vec<String> = Vec::with_capacity(triples.len() * 2);
        for (idx, lo, hi) in triples {
            // Fold per axis.
            let f = match (self.int_of(idx), self.int_of(lo), self.int_of(hi)) {
                (Some(i), Some(l), Some(h)) => Some(i >= l && i < h),
                _ => None,
            };
            folded_all = match (folded_all, f) {
                (Some(true), Some(b)) => Some(b),
                (Some(false), _) => Some(false),
                _ => None,
            };
            // Pre-promote per axis (CSE-d so same idx across axes shares
            // one cast). Index/lo/hi all bounded-by-axis-dim in practice.
            let idx_p = self.maybe_promote_to_u32(idx);
            let lo_p = self.maybe_promote_to_u32(lo);
            let hi_p = self.maybe_promote_to_u32(hi);
            parts.push(format!(
                "({} >= {})",
                self.as_u128(&idx_p),
                self.as_u128(&lo_p)
            ));
            parts.push(format!(
                "({} < {})",
                self.as_u128(&idx_p),
                self.as_u128(&hi_p)
            ));
        }
        let combined = parts.join(" & ");
        let v = self.bind(&format!("if {} {{ 1 }} else {{ 0 }}", combined));
        self.fold_b(&v, folded_all);
        Ok(v)
    }
    fn lte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.unsigned_cmp(a, b, "<=") {
            let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x <= y);
            self.fold_b(&v, folded);
            self.set_ival(&v, Some((0, 1)));
            return Ok(v);
        }
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x <= y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} <= {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        self.use_lib("qle_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let v = self.bind(&format!(
            "if qle_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn gt_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        return self.gt_i_inner(a, b);
    }

    fn gte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.unsigned_cmp(a, b, ">=") {
            let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x >= y);
            self.fold_b(&v, folded);
            self.set_ival(&v, Some((0, 1)));
            return Ok(v);
        }
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x >= y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} >= {} {{ 1 }} else {{ 0 }}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        self.use_lib("qge_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let v = self.bind(&format!(
            "if qge_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }

    fn eq_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x == y);
        let v = self.bind(&format!(
            "if {} == {} {{ 1 }} else {{ 0 }}",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn ne_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x != y);
        let v = self.bind(&format!(
            "if {} == {} {{ 0 }} else {{ 1 }}",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn lt_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qlt_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x < y);
        let v = self.bind(&format!(
            "if qlt_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn lte_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qle_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x <= y);
        let v = self.bind(&format!(
            "if qle_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn gt_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qgt_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x > y);
        let v = self.bind(&format!(
            "if qgt_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn gte_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qge_signed");
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let (sa, ma) = self.project_signed(&af);
        let (sb, mb) = self.project_signed(&bf);
        let folded = self.float_of(a).zip(self.float_of(b)).map(|(x, y)| x >= y);
        let v = self.bind(&format!(
            "if qge_signed({}, {}, {}, {}) {{ 1 }} else {{ 0 }}",
            sa, ma, sb, mb,
        ));
        self.fold_b(&v, folded);
        Ok(v)
    }

    fn sin_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsin");
        let folded = self.float_of(a).map(|x| x.sin());
        let v = self.bind(&format!("qsin({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn sinh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsinh");
        let folded = self.float_of(a).map(|x| x.sinh());
        let v = self.bind(&format!("qsinh({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn cos_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qcos");
        let folded = self.float_of(a).map(|x| x.cos());
        let v = self.bind(&format!("qcos({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn cosh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qcosh");
        let folded = self.float_of(a).map(|x| x.cosh());
        let v = self.bind(&format!("qcosh({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn tan_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qtan");
        let folded = self.float_of(a).map(|x| x.tan());
        let v = self.bind(&format!("qtan({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn tanh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qtanh");
        let folded = self.float_of(a).map(|x| x.tanh());
        let v = self.bind(&format!("qtanh({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn sqrt_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsqrt");
        let v = self.bind(&format!("qsqrt({})", self.as_field(a)));
        // Literal propagation: sqrt(literal) is a compile-time constant.
        // Structural read only (`s_float_of`); carry literal_vals so
        // downstream fast paths fire.
        if let Some(f) = self.s_float_of(a) {
            if f >= 0.0 {
                self.fold_literal(&v, ConcreteValue::Float(f.sqrt()));
            }
        }
        Ok(v)
    }
    fn exp_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qexp");
        let folded = self.float_of(a).map(|x| x.exp());
        let v = self.bind(&format!("qexp({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn log_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qlog");
        let folded = self.float_of(a).map(|x| x.ln());
        let v = self.bind(&format!("qlog({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn arcsin_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarcsin");
        let folded = self.float_of(a).map(|x| x.asin());
        let v = self.bind(&format!("qarcsin({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn arccos_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarccos");
        let folded = self.float_of(a).map(|x| x.acos());
        let v = self.bind(&format!("qarccos({})", self.as_field(a)));
        self.fold_f(&v, folded);
        Ok(v)
    }
    fn arctan2_f(&mut self, y: &CellRef, x: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarctan2");
        let folded = self
            .float_of(y)
            .zip(self.float_of(x))
            .map(|(y, x)| y.atan2(x));
        let v = self.bind(&format!(
            "qarctan2({}, {})",
            self.as_field(y),
            self.as_field(x)
        ));
        self.fold_f(&v, folded);
        Ok(v)
    }

    fn logical_and(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.bool_of(a).zip(self.bool_of(b)).map(|(x, y)| x & y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("{} * {}", a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!("{} * {}", self.as_field(a), self.as_field(b)));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn logical_or(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.bool_of(a).zip(self.bool_of(b)).map(|(x, y)| x | y);
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("{} + {} - {} * {}", a, b, a, b), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        let af = self.as_field(a);
        let bf = self.as_field(b);
        let v = self.bind(&format!("{} + {} - {} * {}", af, bf, af, bf));
        self.fold_b(&v, folded);
        Ok(v)
    }
    fn logical_not(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.bool_of(a).map(|x| !x);
        if self.is_u32(a) {
            let v = self.bind_u32_with_bound(&format!("1 - {}", a), 1);
            self.fold_b(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!("1 - {}", a));
        self.fold_b(&v, folded);
        Ok(v)
    }

    fn select(
        &mut self,
        cond: &CellRef,
        if_true: &CellRef,
        if_false: &CellRef,
    ) -> Result<CellRef, ProvingError> {
        // `cond * t + (1 - cond) * f` is the generic linear-combo
        // realisation of a Plonkish select. When `f == 0` (the common
        // dyn-slice / mask-then-default-zero pattern) the second half
        // is structurally dead and the binding becomes `cond * t` —
        // one mul instead of two muls plus an add.
        let is_zero_int = self.s_int_of(if_false) == Some(0);
        let is_zero_flt = self.s_float_of(if_false) == Some(0.0);
        let all_u32 = self.is_u32(cond)
            && self.is_u32(if_true)
            && (is_zero_int || is_zero_flt || self.is_u32(if_false));
        let v = if all_u32 {
            // Stay u32-clean: cond * t (zero-default fast path) or
            // cond * t + (1 - cond) * f. Bound is max of t/f bounds.
            let t_bound = self.bound_of(if_true).unwrap_or(u32::MAX as u64);
            let f_bound = if is_zero_int || is_zero_flt {
                0
            } else {
                self.bound_of(if_false).unwrap_or(u32::MAX as u64)
            };
            let bound = t_bound.max(f_bound).min(u32::MAX as u64);
            let expr = if is_zero_int || is_zero_flt {
                format!("{} * {}", cond, if_true)
            } else {
                format!("{} * {} + (1 - {}) * {}", cond, if_true, cond, if_false)
            };
            self.bind_u32_with_bound(&expr, bound)
        } else {
            // Field path. Promote cond / if_true / if_false as needed.
            let c = self.as_field(cond);
            let t = self.as_field(if_true);
            let expr = if is_zero_int || is_zero_flt {
                format!("{} * {}", c, t)
            } else {
                let f = self.as_field(if_false);
                format!("{} * {} + (1 - {}) * {}", c, t, c, f)
            };
            self.bind(&expr)
        };
        // Bookkeeping value propagation when the WITNESS knows the cond.
        // This mirrors the historical `value_cache` fold exactly: the
        // picked branch is chosen by a witness read (`bool_of`), so the
        // propagated value is witness-flavoured and lands ONLY in the
        // witness fold cache (public-input bookkeeping — never a structural
        // input). It is deliberately NOT carried into `literal_vals`:
        // making a select result structurally constant from a
        // witness-selected branch would specialise the circuit (the C03/
        // sub_i class of bug, caught by the two-witness diff test on 0442).
        if let Some(c) = self.bool_of(cond) {
            let picked = if c { if_true } else { if_false };
            if let Some(cv) = self.witness_vals.get(picked) {
                self.witness_vals.insert(v.clone(), cv);
            }
        }
        // STRUCTURAL fold: only when the cond is a BUILD-time literal may a
        // select of two structural literals stay structural. This never
        // depends on the witness, so it cannot specialise the circuit.
        if let Some(c) = self.s_bool_of(cond) {
            let picked = if c { if_true } else { if_false };
            if let Some(cv) = self.literal_vals.get(picked).cloned() {
                self.literal_vals.insert(v.clone(), cv);
            }
        }
        // Interval propagation (hulls, min/max recognition) comes from the
        // provenance table's SelectI records.
        // Normalized-index recognition: select(lt(x,0), x + N, x).
        if let (Some(x), Some((ax, n))) = (
            self.lt_zero_of.get(cond).cloned(),
            self.add_const_of.get(if_true).cloned(),
        ) {
            if ax == x && if_false == &x && n > 0 {
                self.norm_index_of.insert(v.clone(), (x, n));
            }
        }
        Ok(v)
    }

    fn int_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Float-to-int in Q32 is the inverse of float_cast: divide by Q. The
        // prior implementation passed through (a.clone()) which silently broke
        // round-trip casts (e.g. IIR filters that write a Float to memory as
        // Int and read it back via FloatCast) — each iteration scaled the
        // value by an extra Q, exploding the magnitude past noir_fpc's
        // MAG_BOUND = 2^126 after ~4 IIR steps and failing `to_signed`.
        // Default to the self-sound, sign-aware `q_div_q` (Euclidean floor of
        // v / Q, asserts `q*Q + r == v` with `0 <= r < Q`). User-level
        // `int(x)` lowers to a bare IR op with no downstream pin, so the
        // unconstrained `q_div_q_hint` would let a malicious prover substitute
        // any integer part (C04). The round-trip `int(float_cast(x))` still
        // cancels exactly because q_div_q(x*Q) == x.
        self.use_lib("q_div_q");
        // Truncation toward zero (Rust `as i128` on the f64 ratio), matching
        // the IR's int_cast convention and the circuit form for negatives.
        let folded = self.float_of(a).map(|f| (f / Q_U64 as f64) as i128);
        let v = self.bind(&format!("q_div_q({})", self.as_field(a)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn float_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Integer-to-float in Q32 means *= 2^32.
        let v = self.bind(&format!("{} * {}", self.as_field(a), Q_U64));
        // Literal propagation: float(literal int) is a compile-time
        // constant. `literal_vals` stores the *conceptual* float value
        // (the Q-scaling lives in the emitted Field expr, not the store),
        // so fold `a`'s integer value as a float into `literal_vals`.
        // Structural read only (`s_int_of`/`s_float_of`).
        let lit = self
            .s_int_of(a)
            .map(|n| n as f64)
            .or_else(|| self.s_float_of(a));
        if let Some(f) = lit {
            self.fold_literal(&v, ConcreteValue::Float(f));
        }
        Ok(v)
    }
    fn bool_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // `bool(x)` must map any nonzero `x` to 1 and 0 to 0 — matching the
        // mock backend and Python truthiness. `bound_analysis` stamps the
        // result `<= 1`, so the circuit MUST actually produce 0/1.
        //
        // Fast path (identity, 0 gates): the operand is provably already in
        // {0,1}. Two STRUCTURAL signals (witness-independent, per the
        // witness-independence rule — never `int_of`/`bound_of`-via-witness):
        //   - a structural literal 0/1 (`s_int_of`), or
        //   - a forward-dataflow analysis bound of <= 1 (`bound_of`, which
        //     is populated from IR constants + dataflow, not witness folding;
        //     e.g. the output of a prior comparison/BoolCast).
        let already_boolean = matches!(self.s_int_of(a), Some(0) | Some(1))
            || matches!(self.bound_of(a), Some(b) if b <= 1);

        let folded = self.int_of(a).map(|x| if x != 0 { 1i128 } else { 0 });

        if already_boolean {
            // Identity suffices; carry the (now provably-true) <= 1 bound and
            // any folded value forward.
            let v = self.bind(&self.as_field(a));
            self.set_bound(&v, 1);
            self.fold_i(&v, folded);
            return Ok(v);
        }

        // General Field path: real truthiness gadget. `if x == 0 { 0 } else
        // { 1 }` is ~2 ACIR gates (one equality, one select) and is sound for
        // any field value. The result is a genuine 0/1 cell.
        let v = self.bind(&format!("if {} == 0 {{ 0 }} else {{ 1 }}", self.as_field(a)));
        self.set_bound(&v, 1);
        self.fold_i(&v, folded);
        Ok(v)
    }

    fn read_input(&mut self, path: &InputPath, _is_public: bool) -> Result<CellRef, ProvingError> {
        let expr = input_index_expr(path, &self.inputs)?;
        let v = self.bind(&expr);
        if let Some((_decimal, concrete)) = self.resolve_path_concrete(path) {
            // Witness-derived value: bookkeeping store only.
            self.witness_vals.insert(v.clone(), concrete);
        }
        Ok(v)
    }

    fn read_external_result(
        &mut self,
        _store_idx: u32,
        _output_idx: u32,
    ) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "read_external_result")
    }

    fn expose_public(&mut self, a: &CellRef, _label: &str) -> Result<(), ProvingError> {
        // Stash the variable name; rendering wraps these into the
        // `pub` return value at the bottom of main. The pub return is
        // typed `[Field; N]`, so u32 cells need the free `(v as Field)`
        // embedding before they hit the array literal.
        self.exposes.push(self.as_field(a));
        // Compute the public-input decimal-field representation if we can
        // constant-fold. This is the whitelisted expose path: it may read
        // the witness fold cache (the exposed value is, by definition,
        // determined by the witness). Fall back to the structural literal
        // store when the exposed cell is a build-time constant.
        let folded = self.witness_vals.get(a).or_else(|| self.literal_vals.get(a).cloned());
        let dec = match folded {
            Some(ConcreteValue::Int(v)) => signed_to_field_decimal(v),
            Some(ConcreteValue::Float(v)) => quantize_to_field_string(v),
            Some(ConcreteValue::Bool(v)) => {
                if v {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            Some(ConcreteValue::Field(s)) => s.clone(),
            None => String::new(),
        };
        self.public_decimal.push(dec);
        Ok(())
    }

    fn assert_true(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        // u32 cells compare to literal `1` natively; Field cells need
        // their value compared as Field.
        let _ = writeln!(self.body, "    assert({} == 1);", a);
        Ok(())
    }

    fn assert_u128_lt(&mut self, idx: &CellRef, hi: &CellRef) -> Result<(), ProvingError> {
        // The assert proves idx in [0, hi): mine it for the interval
        // domain (shrinks dynamic-gather feasible sets).
        if let Some(h) = self.s_int_of(hi) {
            if let Ok(h) = i64::try_from(h) {
                if h > 0 {
                    let new_iv = match self.ival(idx) {
                        Some((lo, hi0)) => (lo.max(0), hi0.min(h - 1)),
                        None => (0, h - 1),
                    };
                    self.set_ival(idx, Some(new_iv));
                }
            }
        }
        // The `as u128` cast emits an implicit `< 2^128` range constraint that
        // catches Zinnia's signed-Field negative encoding (negatives ≈ `p - |x|`
        // are well above 2^128 for BN254). The trailing `<` then enforces
        // the upper bound. One `assert((a as u128) < (b as u128))` is ~30
        // ACIR — versus the default `qge + qlt + and + assert` chain at
        // ~49 ACIR. Used by `discharge_index_in_range` for the common
        // `lo == 0` case (slice bounds, internal index arithmetic).
        //
        // Dedupe: the IR layer's `discharge_index_in_range` (and similar)
        // can fire repeatedly for the same `(idx, hi)` pair when the same
        // dynamic index is re-read across an unrolled loop. The assertion
        // is idempotent — emitting it once is sound and saves the u128
        // cast/compare overhead on every duplicate.
        if !self.emitted_u128_asserts.insert((idx.clone(), hi.clone())) {
            return Ok(());
        }
        // Pre-promote bounded Field operands to u32 once (CSE-d) so
        // repeated bound-asserts against the same index/hi don't each
        // pay a separate 128-bit RANGE.
        let idx_p = self.maybe_promote_to_u32(idx);
        let hi_p = self.maybe_promote_to_u32(hi);
        // If both operands already fit u32 (and are u32-typed), emit
        // the native compare with no cast — zero RANGE blackboxes.
        if self.is_u32(&idx_p) && self.is_u32(&hi_p) {
            let _ = writeln!(self.body, "    assert({} < {});", idx_p, hi_p);
            return Ok(());
        }
        let _ = writeln!(
            self.body,
            "    assert({} < {});",
            self.as_u128(&idx_p),
            self.as_u128(&hi_p)
        );
        Ok(())
    }

    /// Override the trait default's per-bit `floor_div + mod` extraction.
    ///
    /// The default `assert_u128_lt_bit` extracts bit `bit` of `idx` with a
    /// `floor_div_i(idx, 2^bit)` followed by `mod_i(.., 2)`. On the Field
    /// Noir backend each of those lowers to a self-sound `q_floor_div` /
    /// `q_mod` library call (a quotient/remainder witness pinned by
    /// `q*d + r == n` with an internal `0 <= r < |d|` range check) — roughly
    /// 100+ ACIR per bit. The small-axis `take_along_axis` routing
    /// (`ir_assert_u128_lt_bits`, axis size 3..=8) calls this once PER BIT
    /// PER gathered cell, so a 13x13 axis-5 gather paid ~3 q-divisions x 169
    /// cells = ~1000 q-ops (the C19 51k-gate cost).
    ///
    /// Here, once `idx` is constrained to `[0, hi)` (the `as u32` promotion's
    /// 32-bit RANGE plus the explicit `< hi`), `idx` is a u32 in a small
    /// range and bit `bit` is the deterministic native `(idx >> bit) & 1` —
    /// no Field division, no extra range check. `assert_bit` on the result is
    /// then statically discharged (a u32 `& 1` is provably a bit), so we skip
    /// it. CSE collapses the shared `idx < hi` and `idx as u32` across the
    /// `bits` calls for the same cell, leaving just one cheap shift+mask per
    /// bit. Soundness is unchanged: the bound is asserted exactly as the
    /// default did (`assert_u128_lt`), and the extracted bit is a pure
    /// function of the bounded `idx`.
    fn assert_u128_lt_bit(
        &mut self,
        idx: &CellRef,
        hi: u64,
        bit: u32,
        _bits: u32,
    ) -> Result<CellRef, ProvingError> {
        // Constant-fold when the index is a compile-time literal.
        if let Some(i) = self.s_int_of(idx) {
            let bit_val = if i >= 0 && (i as u128) < hi as u128 {
                (((i as u128) >> bit) & 1) as i64
            } else {
                0
            };
            let v = self.constant_int(bit_val)?;
            return Ok(v);
        }
        // Enforce the `[0, hi)` bound (same soundness obligation as the
        // default, which calls `assert_u128_lt` first).
        let hi_cell = self.constant_int(hi as i64)?;
        self.assert_u128_lt(idx, &hi_cell)?;
        // Promote to u32 (bounded by hi-1) so the bit extraction is a native
        // u32 shift+mask rather than a Field division. `assert_u128_lt` set
        // the interval; pin the cell_bound so `maybe_promote_to_u32` fires.
        if self.bound_of(idx).is_none() && hi > 0 {
            self.set_bound(idx, hi - 1);
        }
        let idx_u = self.maybe_promote_to_u32(idx);
        let folded = self
            .int_of(idx)
            .map(|i| (((i as u128) >> bit) & 1) as i128);
        let out = if self.is_u32(&idx_u) {
            // Native u32 path: shift + mask is a single cheap blackbox-free op.
            self.bind_u32_with_bound(&format!("({} >> {}) & 1", idx_u, bit), 1)
        } else {
            // Fallback (u32 promotion declined, e.g. non-amortizable circuit):
            // keep the sound default's division-based extraction.
            let divisor = self.constant_int(1_i64 << bit)?;
            let shifted = self.floor_div_i(idx, &divisor)?;
            let two = self.constant_int(2)?;
            let m = self.mod_i(&shifted, &two)?;
            self.assert_bit(&m)?;
            m
        };
        if let Some(f) = folded {
            self.fold_i(&out, Some(f));
        }
        Ok(out)
    }

    fn allocate_memory(
        &mut self,
        segment_id: u32,
        size: u32,
        init: i64,
    ) -> Result<(), ProvingError> {
        // The init value is a signed Q-scaled int that the IR-gen will
        // also write into individual cells via WriteMemory before any
        // ReadMemory hits an unwritten cell, in every case we've seen.
        // We still need a defined fallback for the empty-cell case to
        // mirror halo2 synth semantics.
        let init_field = signed_to_field_decimal(init as i128);
        self.memory_init.insert(segment_id, init_field.clone());
        self.segment_iv.insert(segment_id, Some((init, init)));
        // Each cell starts as a literal; writes overwrite with SSA names.
        self.memories.insert(
            segment_id,
            (0..size as usize).map(|_| init_field.clone()).collect(),
        );
        Ok(())
    }
    fn write_memory(
        &mut self,
        segment_id: u32,
        addr: &CellRef,
        value: &CellRef,
    ) -> Result<(), ProvingError> {
        *self.write_version.entry(segment_id).or_insert(0) += 1;
        // Merge the written value into the segment hull.
        let viv = self.ival(value);
        let entry = self.segment_iv.entry(segment_id).or_insert(Some((0, 0)));
        *entry = match (*entry, viv) {
            (Some((sl, sh)), Some((vl, vh))) => Some((sl.min(vl), sh.max(vh))),
            _ => None,
        };
        if let Some(a) = self.s_int_of(addr) {
            let a = a as usize;
            let mem = self.memories.entry(segment_id).or_default();
            if a >= mem.len() {
                // Resize to mirror the halo2 mock behaviour: writes beyond
                // the allocated region are silently bounded.
                return Ok(());
            }
            mem[a] = value.clone();
            return Ok(());
        }
        // Dynamic address: honest in-circuit scatter. The address's affine
        // lineage names a root whose eq-indicators are cached and shared
        // with every other access on the same root; slot k updates with
        // the flat arithmetic form old + ind*(value - old).
        let len = self.memories.get(&segment_id).map_or(0, |m| m.len());
        let aff = self.aff_of(addr);
        let (root, scale, offset) = if aff.terms.len() == 1 {
            (aff.terms[0].0.clone(), aff.terms[0].1, aff.offset)
        } else {
            (addr.clone(), 1, 0)
        };
        let vf = self.as_field(value);
        let root_iv = self.ival(&root);
        for k in 0..len as i64 {
            let num = k - offset;
            if scale == 0 || num % scale != 0 {
                continue; // slot unreachable from this address
            }
            let v = num / scale;
            if let Some((lo, hi)) = root_iv {
                if v < lo || v > hi {
                    continue; // slot outside the address's value interval
                }
            }
            let ind = self.indicator(&root, v);
            let old_v = self.memories.get(&segment_id).unwrap()[k as usize].clone();
            let of = self.as_field(&old_v);
            let indf = self.as_field(&ind);
            let nv = self.bind(&format!("{of} + {indf} * ({vf} - {of})"));
            self.memories.get_mut(&segment_id).unwrap()[k as usize] = nv;
        }
        Ok(())
    }
    fn read_memory(&mut self, segment_id: u32, addr: &CellRef) -> Result<CellRef, ProvingError> {
        let init = self
            .memory_init
            .get(&segment_id)
            .cloned()
            .unwrap_or_else(|| "0".to_string());
        let Some(a) = self.s_int_of(addr) else {
            // Dynamic address: honest in-circuit gather via the address's
            // affine lineage (port of the halo2 backend's machinery).
            // Single root: flat sum of shared eq-indicators times the
            // reachable slots, with a one-hot membership assert replacing
            // per-read bound checks. Two roots: factored two-stage gather
            // with cached stage-1 planes. No lineage: the address itself
            // is the root (still flat + shared, never a sequential chain).
            let len = self.memories.get(&segment_id).map_or(0, |m| m.len());
            let aff = self.aff_of(addr);
            if aff.terms.len() == 2 && len > 0 {
                // Stage-1 over the larger-stride root, stage-2 over the
                // other; planes cached across reads.
                let (t0, t1) = (aff.terms[0].clone(), aff.terms[1].clone());
                let ((r1, s1), (r2, s2)) = if t0.1.abs() >= t1.1.abs() {
                    (t0, t1)
                } else {
                    (t1, t0)
                };
                let r1_iv = self.ival(&r1);
                let r2_iv = self.ival(&r2);
                let mut w_vals: Vec<i64> = Vec::new();
                for w in -(len as i64)..=(len as i64) {
                    if let Some((lo, hi)) = r2_iv {
                        if w < lo || w > hi {
                            continue;
                        }
                    }
                    let resid = w * s2 + aff.offset;
                    if !Self::feasible(s1, resid, len, r1_iv).is_empty() {
                        w_vals.push(w);
                    }
                }
                if !w_vals.is_empty() {
                    let mut v_all: Vec<i64> = Vec::new();
                    for &w in &w_vals {
                        v_all.extend(Self::feasible(s1, w * s2 + aff.offset, len, r1_iv));
                    }
                    v_all.sort_unstable();
                    v_all.dedup();
                    self.assert_one_hot(&r1.clone(), &v_all);
                    self.assert_one_hot(&r2.clone(), &w_vals);
                    let mut parts: Vec<String> = Vec::with_capacity(w_vals.len());
                    for &w in &w_vals {
                        let plane = self.gather_plane(
                            segment_id,
                            &r1,
                            s1,
                            w * s2 + aff.offset,
                            &init,
                        );
                        let ind = self.indicator(&r2, w);
                        parts.push(format!(
                            "{} * {}",
                            self.as_field(&ind),
                            self.as_field(&plane)
                        ));
                    }
                    let out = self.bind_sum(&parts);
                    self.apply_segment_iv(segment_id, &out);
                    return Ok(out);
                }
            }
            let (root, scale, offset) = if aff.terms.len() == 1 {
                (aff.terms[0].0.clone(), aff.terms[0].1, aff.offset)
            } else {
                (addr.clone(), 1, 0)
            };
            let root_iv = self.ival(&root);
            let vs = Self::feasible(scale, offset, len, root_iv);
            if vs.is_empty() {
                // No reachable slot: any witness fails the membership
                // assert below; surface it as an explicit contradiction.
                let _ = writeln!(self.body, "    assert(1 == 0);");
                return Ok(init);
            }
            self.assert_one_hot(&root.clone(), &vs);
            let mut parts: Vec<String> = Vec::with_capacity(vs.len());
            for &v in &vs {
                let slot = (v * scale + offset) as usize;
                let cell = self.memories.get(&segment_id).unwrap()[slot].clone();
                let ind = self.indicator(&root, v);
                parts.push(format!(
                    "{} * {}",
                    self.as_field(&ind),
                    self.as_field(&cell)
                ));
            }
            let out = self.bind_sum(&parts);
            self.apply_segment_iv(segment_id, &out);
            return Ok(out);
        };
        let a = a as usize;
        let mem = self.memories.get(&segment_id);
        let expr = match mem {
            Some(m) if a < m.len() => m[a].clone(),
            _ => init,
        };
        // If the stored expression is a single identifier (the memory cell
        // currently holds the SSA name of an existing binding), reuse that
        // name directly instead of emitting a vacuous `let vN = vM;`. The
        // common shape is a slice / reshape that re-reads cells previously
        // written verbatim — without this, every cell of a 20-element
        // reshape becomes a no-op rename, bloating both nargo compile time
        // and the proof's gate count.
        if is_identifier(&expr) {
            // Inherit the segment hull only when the cell itself carries
            // no tighter interval.
            if !self.interval_of.contains_key(&expr) {
                self.apply_segment_iv(segment_id, &expr.clone());
            }
            return Ok(expr);
        }
        // Propagate the witness fold across the memory read for
        // public-input bookkeeping (mirrors the historical `value_cache`
        // carry exactly). The structural literal store is deliberately NOT
        // carried: a value loaded out of segment memory is treated as a
        // fresh runtime cell, exactly as before the witness/literal split,
        // keeping emission gate-identical.
        let prior_witness = self.witness_vals.get(&expr);
        let v = self.bind(&expr);
        if let Some(val) = prior_witness {
            self.witness_vals.insert(v.clone(), val);
        }
        Ok(v)
    }
    fn memory_trace_emit(
        &mut self,
        _segment_id: u32,
        _is_write: bool,
        _args: &[CellRef],
    ) -> Result<(), ProvingError> {
        Ok(())
    }
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError> {
        Ok(())
    }

    fn allocate_dynamic_ndarray_meta(
        &mut self,
        _array_id: u32,
        _dtype_name: &str,
        _max_length: u32,
        _max_rank: u32,
        _max_shape: &[u32],
    ) -> Result<(), ProvingError> {
        noop_unsupported!(self, "allocate_dynamic_ndarray_meta")
    }
    fn witness_dynamic_ndarray_meta(
        &mut self,
        _array_id: u32,
        _max_rank: u32,
        _param_name: &str,
        _args: &[CellRef],
    ) -> Result<(), ProvingError> {
        noop_unsupported!(self, "witness_dynamic_ndarray_meta")
    }
    fn dyn_meta_axis_bounds(
        &self,
        _array_id: u32,
        max_rank: u32,
        max_length: u32,
    ) -> (u32, Vec<u32>) {
        // The Noir backend does not support DynamicNDArrays; the meta-assert
        // override below errors before this is consulted. Return the
        // allocation-default fallback for completeness.
        (max_length, vec![max_length; max_rank as usize])
    }
    fn assert_dynamic_ndarray_meta(
        &mut self,
        _array_id: u32,
        _max_rank: u32,
        _max_length: u32,
        _args: &[CellRef],
    ) -> Result<(), ProvingError> {
        // Overrides the trait default: dyn ndarrays are unsupported on Noir,
        // so signal that rather than emitting the default lowering.
        noop_unsupported!(self, "assert_dynamic_ndarray_meta")
    }
    fn dynamic_ndarray_get_item(
        &mut self,
        _array_id: u32,
        _segment_id: u32,
        _index: &CellRef,
    ) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "dynamic_ndarray_get_item")
    }
    fn dynamic_ndarray_set_item(
        &mut self,
        _array_id: u32,
        _segment_id: u32,
        _index: &CellRef,
        _value: &CellRef,
    ) -> Result<(), ProvingError> {
        noop_unsupported!(self, "dynamic_ndarray_set_item")
    }

    fn poseidon_hash(&mut self, _inputs: &[CellRef]) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "poseidon_hash")
    }
    fn eq_hash(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "eq_hash")
    }
}

// ---------------------------------------------------------------------------
// Rendering — assemble main.nr from emitted body + collected exposes
// ---------------------------------------------------------------------------

fn render_main_nr(inputs: &[EmittedInput], emitter: &NoirEmitter) -> Result<String, ProvingError> {
    let mut out = String::new();
    out.push_str(&emitter.render_use_decls());
    out.push_str("fn main(");
    let mut first = true;
    for inp in inputs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        match &inp.kind {
            InputKind::ScalarField => {
                let _ = write!(out, "{}: Field", inp.name);
            }
            InputKind::Bool => {
                // Booleans arrive as Field 0/1 — the body treats them as
                // plain Field values.
                let _ = write!(out, "{}: Field", inp.name);
            }
            InputKind::ArrayField { len } => {
                let _ = write!(out, "{}: [Field; {}]", inp.name, len);
            }
        }
    }
    // bb requires `witness.size() == max_witness_index + 1`. A trivially-empty
    // circuit (no exposes — e.g. a kernel whose result is a degenerate empty
    // array from `x[all_False]`) leaves declared input witnesses unused; bb
    // then aborts with an internal-invariant failure. Synthesize a single
    // public output referencing the last cell of the last input array so the
    // resulting circuit has at least one constraint that touches a witness.
    // nargo folds `let _ = x[N-1];` and `assert(x[N-1] == x[N-1])` to nothing,
    // so we promote to a pub return — that emission survives DCE.
    let pruned_body = dce_body(&emitter.body, &emitter.exposes);
    let mut synthetic_exposes: Vec<String> = Vec::new();
    if pruned_body.trim().is_empty() && emitter.exposes.is_empty() {
        if let Some(inp) = inputs
            .iter()
            .rfind(|i| !matches!(i.kind, InputKind::ScalarField))
        {
            if let InputKind::ArrayField { len } = &inp.kind {
                if *len > 0 {
                    synthetic_exposes.push(format!("{}[{}]", inp.name, len - 1));
                }
            }
        }
        if synthetic_exposes.is_empty() {
            if let Some(inp) = inputs.first() {
                synthetic_exposes.push(inp.name.clone());
            }
        }
    }
    let all_exposes: Vec<&str> = emitter
        .exposes
        .iter()
        .map(String::as_str)
        .chain(synthetic_exposes.iter().map(String::as_str))
        .collect();
    let n_exposes = all_exposes.len();
    if n_exposes == 0 {
        out.push_str(") {\n");
    } else if n_exposes == 1 {
        out.push_str(") -> pub Field {\n");
    } else {
        let _ = write!(out, ") -> pub [Field; {}] {{\n", n_exposes);
    }
    out.push_str(&pruned_body);
    if n_exposes == 1 {
        let _ = writeln!(out, "    {}", all_exposes[0]);
    } else if n_exposes > 1 {
        out.push_str("    [");
        let mut first = true;
        for e in &all_exposes {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(e);
        }
        out.push_str("]\n");
    }
    out.push_str("}\n");
    let _ = emitter.params; // params currently unused at render time
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::CircuitInputs;
    use crate::prove::traits::Synthesizer;

    /// Build a bare emitter with no inputs/witness, sufficient to drive
    /// the pure-string `eq_i` peephole in isolation.
    fn bare_emitter(prov: &crate::provenance::ProvenanceTable) -> NoirEmitter<'_> {
        let witness = ResolvedWitness::new(CircuitInputs::default(), Q_BITS);
        NoirEmitter::new(
            Vec::new(),
            witness,
            ProvingParams::default(),
            HashMap::new(),
            0,
            prov,
        )
    }

    /// Register a compile-time integer literal cell so `s_int_of` resolves it.
    fn lit_int(em: &mut NoirEmitter<'_>, name: &str, v: i128) {
        em.literal_vals
            .insert(name.to_string(), ConcreteValue::Int(v));
    }

    /// Register a recognized normalized index `name = select(lt(x,0), x+n, x)`.
    fn norm_index(em: &mut NoirEmitter<'_>, name: &str, x: &str, n: i64) {
        em.norm_index_of
            .insert(name.to_string(), (x.to_string(), n));
    }

    // C05 regression: the eq_i normalized-index dual-equality peephole
    // rewrites eq(norm, j) -> (x == j) | (x == j - N). That identity is exact
    // ONLY for 0 <= j < N. For out-of-range j the emitter must fall through to
    // the generic equality path, NOT emit the dual-disjunct form (which would
    // be a wrong circuit). These tests assert the guard on both boundaries.

    #[test]
    fn eq_i_norm_index_in_range_uses_peephole() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        // norm over axis of length N=5, x is some Field-typed cell.
        norm_index(&mut em, "norm", "x", 5);
        lit_int(&mut em, "j", 3); // in [0, 5)
        let _ = em.eq_i(&"norm".to_string(), &"j".to_string()).unwrap();
        // Expect the dual-disjunct peephole: (x == 3) | (x + 2 == 0).
        assert!(
            em.body.contains("(x == 3) | (x + 2 == 0)"),
            "in-range j=3 should use the dual-equality peephole, body was:\n{}",
            em.body
        );
    }

    #[test]
    fn eq_i_norm_index_j_equals_n_falls_through() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        norm_index(&mut em, "norm", "x", 5);
        lit_int(&mut em, "j", 5); // j == N, OUT of range
        let _ = em.eq_i(&"norm".to_string(), &"j".to_string()).unwrap();
        // The spurious second disjunct would be (x + 0 == 0) / (x == 0); the
        // first would be (x == 5). Neither dual form must appear.
        assert!(
            !em.body.contains(" | "),
            "j==N must NOT emit the dual-equality peephole, body was:\n{}",
            em.body
        );
    }

    #[test]
    fn eq_i_norm_index_j_negative_falls_through() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        norm_index(&mut em, "norm", "x", 5);
        lit_int(&mut em, "j", -1); // j < 0, OUT of range
        let _ = em.eq_i(&"norm".to_string(), &"j".to_string()).unwrap();
        assert!(
            !em.body.contains(" | "),
            "j<0 must NOT emit the dual-equality peephole, body was:\n{}",
            em.body
        );
    }

    // R06: the structural literal store (`literal_vals`) must never carry a
    // witness-folded value. The witness fold cache (`witness_vals`) is a
    // distinct, privacy-enforced store; the structural accessors read only
    // `literal_vals`. This test pins that separation: a value present only
    // in the witness fold cache is invisible to `s_int_of`/`s_float_of`.
    #[test]
    fn witness_fold_invisible_to_structural_accessors() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        // Simulate a witness-folded value (e.g. add_i of two read_input
        // cells): it lands ONLY in the witness cache.
        em.fold_i("wcell", Some(7));
        // The witness read sees it (bookkeeping path)...
        assert_eq!(em.int_of("wcell"), Some(7));
        // ...but the STRUCTURAL accessors must NOT — keying a gate on this
        // would specialise the circuit to the prover's witness.
        assert_eq!(em.s_int_of("wcell"), None);
        assert_eq!(em.s_float_of("wcell"), None);

        // A build-time literal, by contrast, is visible structurally and is
        // NOT in the witness cache.
        em.fold_literal("lcell", ConcreteValue::Int(3));
        assert_eq!(em.s_int_of("lcell"), Some(3));
        assert_eq!(em.int_of("lcell"), None);
    }

    // R06: `annotate_with_analysis_bound` moved out of `bind()` and into
    // `on_stmt_result`. Intermediate cells bound while a statement is
    // dispatched must NOT inherit the statement's RESULT bound; only the
    // actual result cell may. A shared indicator-style cell reused by a
    // later statement must keep its own correct bound.
    #[test]
    fn analysis_bound_lands_on_stmt_result_not_intermediates() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);

        // ── Statement 0: result bound is 1000. ──────────────────────────
        em.current_stmt_bound = Some(1000);
        // An *intermediate* cell bound mid-statement (e.g. a cached
        // indicator `if x == 3 { 1 } else { 0 }`, value in {0,1}).
        let shared_ind = em.bind("if x == 3 { 1 } else { 0 }");
        // The intermediate must NOT have been stamped with stmt 0's bound.
        assert_eq!(
            em.bound_of(&shared_ind),
            None,
            "intermediate cell wrongly inherited the statement-result bound"
        );
        // The statement RESULT cell (a different cell) gets the bound.
        let result0 = em.bind("x * 2");
        em.on_stmt_result(0, crate::types::ValueId(0), &result0);
        assert_eq!(em.bound_of(&result0), Some(1000));

        // ── Statement 1 reuses the shared indicator, with a tiny result
        //    bound (1). The reused indicator must keep ITS OWN bound (none
        //    set here, definitely not statement 1's spuriously tight 1 nor
        //    statement 0's 1000). on_stmt_result stamps only the result.
        em.current_stmt_bound = Some(1);
        // Reusing the indicator expression hits CSE -> same cell, no new bound.
        let reused = em.bind("if x == 3 { 1 } else { 0 }");
        assert_eq!(reused, shared_ind, "CSE should reuse the indicator cell");
        let result1 = em.bind(&format!("{} + 5", reused));
        em.on_stmt_result(1, crate::types::ValueId(1), &result1);
        // The shared indicator still carries no stmt-result bound — it was
        // never a statement result, so it is not poisoned by either stmt.
        assert_eq!(
            em.bound_of(&shared_ind),
            None,
            "shared indicator was poisoned by a later statement's result bound"
        );
        assert_eq!(em.bound_of(&result1), Some(1));
    }
}

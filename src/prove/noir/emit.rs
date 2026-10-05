//! IR -> Noir source emitter.
//!
//! Implements [`Synthesizer`] with `CellRef = String` — each cell ref is a
//! Noir-language expression (typically a let-bound identifier). The
//! interpreter walks the IR in statement order and we emit one
//! `let vN = <expr>;` per produced value into the body of `fn main`.
//!
//! Inputs are bound to Noir function parameters whose names and shapes come
//! from the [`ResolvedWitness`]'s parameter structure (`Public[...]`
//! parameters are declared `pub`); its values only go into `Prover.toml`.
//! The circuit text never depends on input values. Floats are quantized to
//! Q32.
//!
//! Soundness rules the lowering follows:
//! - A Field value reaches an unsigned Noir type only through a range-checked
//!   conversion (`checked_uN` in noir_fpc, see [`NoirEmitter::to_uint`]);
//!   `Field as uN` truncates and is never used as a check.
//! - Assert lowerings (`assert_u128_lt`, `assert_non_neg_i`, `assert_bit`)
//!   consult no interval or bound of the asserted value: the builder derives
//!   those intervals from the asserts themselves.
//! - Other lowerings may use intervals and bounds only to pick a cheaper
//!   form whose own checks enforce them (a wrong fact fails the proof), or
//!   facts guaranteed by a lowering (0/1 results of comparisons and logic
//!   ops, see [`NoirEmitter::is_boolean`]).
//! - Integer, logic and division semantics follow the mock backend
//!   (`kernel.rs`: Python `//` and `%`, a zero divisor gives 0, Python
//!   truthiness, the kernel's fixed-point sqrt/sin/cos/arcsin); see
//!   `benchmarking/noir_differential.py`.

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
    /// Field elements of the `pub` inputs of `main`. bb's public-inputs file
    /// lists these first, then the `return_count` returned values.
    pub public_input_count: usize,
    /// Public values returned by `main` (the exposed outputs, in IR order).
    pub return_count: usize,
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
        bounds,
        intrinsic_count,
        &ir.provenance,
    );
    interpret_ir(ir, &mut emitter)?;

    let main_nr = render_main_nr(&input_params, &emitter);
    let return_count = emitter.exposes.len();
    let public_input_count = input_params
        .iter()
        .filter(|i| i.is_public)
        .map(EmittedInput::field_count)
        .sum();

    Ok(EmitOutput {
        main_nr,
        prover_toml,
        public_input_count,
        return_count,
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
            // nargo reads field elements, so a negative integer is written
            // as p - |v| (a bare "-2" is rejected).
            let _ = writeln!(out, "{} = \"{}\"", name, signed_to_field_decimal(*v as i128));
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
        InputNode::Int(v) => signed_to_field_decimal(*v as i128),
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
    /// Declared `pub` on `fn main` (a `Public[...]` kernel parameter).
    pub is_public: bool,
}

impl EmittedInput {
    /// Number of field elements the parameter occupies.
    fn field_count(&self) -> usize {
        match self.kind {
            InputKind::ScalarField | InputKind::Bool => 1,
            InputKind::ArrayField { len } => len,
        }
    }
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
            is_public: p.is_public,
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
    // (`let (vA, vB) = expr;`, used by the division cells); both forms are
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
                // tuple destructures (`floor_divmod_cells`) must be
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

/// Affine view of an integer cell: `cell == sum(root_i * scale_i) + offset`,
/// with at most two dynamic roots and non-zero scales.
#[derive(Clone, Debug, Default)]
struct AffLin {
    terms: Vec<(String, i64)>,
    offset: i64,
}

/// SSA-style emitter. Each call returns the name of a Noir let-binding
/// (or a literal expression for cheap constants). The interpreter dispatches
/// one method per IR statement.
///
/// The emitter never sees the witness: the circuit text is a function of the
/// IR alone, so it cannot be specialized to one input. Public values are read
/// back from the proof (`bb`'s public-inputs file), not recomputed here.
pub struct NoirEmitter<'a> {
    /// Structural-provenance table (derived once in the IRBuilder; see
    /// src/provenance.rs). Read LIVE for intervals / affine views /
    /// segment hulls via the name ↔ ValueId maps below.
    prov: &'a crate::provenance::ProvenanceTable,
    vid_of_var: HashMap<String, crate::types::ValueId>,
    var_of_vid: HashMap<crate::types::ValueId, String>,
    /// Function arguments declared on `fn main`.
    pub inputs: Vec<EmittedInput>,
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
    /// In-Rust simulation of Zinnia's segment-keyed scratch memory.
    /// Each entry maps a (segment_id, addr) to either the most recent
    /// SSA variable written there (`Some(name)`) or the segment-wide
    /// init-value as a literal Field. Mirrors the halo2 synthesizer's
    /// `memories: HashMap<u32, Vec<Fp>>` exactly, but stores names so
    /// later reads can re-emit `let vN = <writer>` cheaply.
    memories: HashMap<u32, Vec<String>>,
    /// segment_id -> init-value-as-Field-decimal-string.
    memory_init: HashMap<u32, String>,
    /// Cache: division call text -> its (quotient, remainder) names.
    divmod_cache: HashMap<String, (String, String)>,
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
    /// Per-cell inclusive upper bound on the (non-negative) value.
    /// Populated from non-negative constants, the IR bound analysis and
    /// checked u32 arithmetic. When every operand of an integer op is a u32
    /// cell and the result bound fits `u32`, the op is emitted as
    /// `let v: u32 = ...;` (Noir checks u32 overflow).
    cell_bound: HashMap<String, u64>,
    /// Cells bound with type `u32` (all others are `Field`). Every entry
    /// holds a genuine u32 value: it is a non-negative literal, the result
    /// of checked u32 arithmetic, or a range-checked `checked_u32`
    /// conversion. A u32 cell used in a Field context takes the free
    /// `(v as Field)` embedding (see `as_field`).
    u32_cells: HashSet<String>,
    /// Compile-time constant store: SSA name -> value known at circuit-build
    /// time (IR `ConstantInt`/`ConstantFloat`/`ConstantBool`, or values
    /// derived only from such constants). Read by `s_int_of`/`s_float_of`/
    /// `s_bool_of`.
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

/// A compile-time constant known to the emitter (an IR literal, or a value
/// derived only from IR literals).
#[derive(Debug, Clone)]
enum ConcreteValue {
    Int(i128),
    Float(f64),
    Bool(bool),
}

/// Order comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn op(self) -> &'static str {
        match self {
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
        }
    }
}

/// Unsigned Noir integer types the emitter converts Field cells into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UintWidth {
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl UintWidth {
    /// Narrowest width whose range contains `max`.
    fn covering(max: u128) -> Self {
        match 128 - max.leading_zeros() {
            0..=8 => UintWidth::U8,
            9..=16 => UintWidth::U16,
            17..=32 => UintWidth::U32,
            33..=64 => UintWidth::U64,
            _ => UintWidth::U128,
        }
    }

    fn ty(self) -> &'static str {
        match self {
            UintWidth::U8 => "u8",
            UintWidth::U16 => "u16",
            UintWidth::U32 => "u32",
            UintWidth::U64 => "u64",
            UintWidth::U128 => "u128",
        }
    }

    /// The `noir_fpc` range-checked conversion into this type.
    fn checked_fn(self) -> &'static str {
        match self {
            UintWidth::U8 => "checked_u8",
            UintWidth::U16 => "checked_u16",
            UintWidth::U32 => "checked_u32",
            UintWidth::U64 => "checked_u64",
            UintWidth::U128 => "checked_u128",
        }
    }
}

impl<'a> NoirEmitter<'a> {


    fn sub_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
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
                return Ok(v);
            }
        }
        let v = self.bind(&format!("{} - {}", self.as_field(a), self.as_field(b)));
        Ok(v)
    }
    fn mul_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if self.is_u32(a) && self.is_u32(b) {
            if let Some(prod) = self
                .bound_of(a)
                .zip(self.bound_of(b))
                .and_then(|(x, y)| x.checked_mul(y))
                .filter(|p| *p < (1u64 << 32))
            {
                let v = self.bind_u32_with_bound(&format!("{} * {}", a, b), prod);
                return Ok(v);
            }
        }
        let v = self.bind(&format!("{} * {}", self.as_field(a), self.as_field(b)));
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

    /// Unsigned comparison for operands whose (enforced) intervals are
    /// non-negative: both sides are converted to the narrowest unsigned type
    /// covering the hulls and compared natively (~20 gates for u8 against
    /// ~30 for the signed projections). The conversions are range-checked,
    /// so an operand outside the chosen width fails the proof instead of
    /// being truncated: the interval only selects the width, and a wrong
    /// interval can cost completeness but never soundness.
    fn unsigned_cmp(&mut self, a: &str, b: &str, cmp: Cmp) -> Option<String> {
        let (al, ah) = self.ival(a)?;
        let (bl, bh) = self.ival(b)?;
        if al < 0 || bl < 0 {
            return None;
        }
        let width = UintWidth::covering(ah.max(bh).max(1) as u128);
        let ca = self.to_uint(a, width);
        let cb = self.to_uint(b, width);
        Some(self.bind_flag(&format!("if {} {} {} {{ 1 }} else {{ 0 }}", ca, cmp.op(), cb)))
    }

    /// Signed integer comparison `a cmp b`: the unsigned fast path when the
    /// intervals allow it, a native u32 compare for u32 cells, else the
    /// library's gap-pinned `qlt` (`signed_cmp`).
    fn compare_i(&mut self, a: &str, b: &str, cmp: Cmp) -> String {
        if let Some(v) = self.unsigned_cmp(a, b, cmp) {
            return v;
        }
        if self.is_u32(a) && self.is_u32(b) {
            return self.bind_u32_with_bound(
                &format!("if {} {} {} {{ 1 }} else {{ 0 }}", a, cmp.op(), b),
                1,
            );
        }
        self.signed_cmp(a, b, cmp)
    }

    /// Signed comparison `a cmp b` through the library's `qlt`, which pins
    /// its answer with one 64-bit range check of the gap between the
    /// operands: it is the mock's `lt_i` (the sign of `a - b`) for every pair
    /// with |a - b| < 2^64, which includes any two values of the signed
    /// 64-bit window, and fails the proof for any other pair. `a <= b` is
    /// `!(b < a)`, `a > b` is `b < a` and `a >= b` is `!(a < b)`, so a
    /// comparison and its negation share one `qlt`.
    fn signed_cmp(&mut self, a: &str, b: &str, cmp: Cmp) -> String {
        let (af, bf) = (self.as_field(a), self.as_field(b));
        let (lhs, rhs, negate) = match cmp {
            Cmp::Lt => (af, bf, false),
            Cmp::Le => (bf, af, true),
            Cmp::Gt => (bf, af, false),
            Cmp::Ge => (af, bf, true),
        };
        self.use_lib("qlt");
        let less = self.bind_typed(&format!("qlt({lhs}, {rhs})"), "bool");
        let flag = if negate { format!("(!{less}) as Field") } else { format!("{less} as Field") };
        self.bind_flag(&flag)
    }

    /// `a cmp b` for `LtIBounded`/`GtIBounded`, whose operands the IR
    /// promises are non-negative. The promise is not trusted: the operands
    /// go through range-checked conversions (u32 when their bound allows,
    /// else u64), so a negative operand fails the proof.
    fn bounded_cmp(&mut self, a: &str, b: &str, cmp: Cmp) -> String {
        let a = self.maybe_promote_to_u32(a);
        let b = self.maybe_promote_to_u32(b);
        if self.is_u32(&a) && self.is_u32(&b) {
            return self.bind_u32_with_bound(
                &format!("if {} {} {} {{ 1 }} else {{ 0 }}", a, cmp.op(), b),
                1,
            );
        }
        let ua = self.to_uint(&a, UintWidth::U64);
        let ub = self.to_uint(&b, UintWidth::U64);
        self.bind_flag(&format!("if {} {} {} {{ 1 }} else {{ 0 }}", ua, cmp.op(), ub))
    }

    /// Expression of unsigned type `width` holding the value of `cell`.
    /// A u32 cell widens for free (or narrows through a range check); a
    /// Field cell goes through the library's range-checked `checked_uN`, so
    /// a value outside [0, 2^N) fails the proof. The emitter never writes a
    /// bare `Field as uN`: that cast truncates to the low N bits and is not
    /// a range check.
    fn to_uint(&mut self, cell: &str, width: UintWidth) -> String {
        if self.is_u32(cell) {
            match width {
                UintWidth::U32 => return cell.to_string(),
                UintWidth::U64 | UintWidth::U128 => {
                    return format!("({} as {})", cell, width.ty())
                }
                UintWidth::U8 | UintWidth::U16 => {}
            }
        }
        let f = width.checked_fn();
        self.use_lib(f);
        let arg = self.as_field(cell);
        self.bind_typed(&format!("{}({})", f, arg), width.ty())
    }

    /// `c` as a 0/1 value with Python truthiness (non-zero is 1), the mock's
    /// reading of logic-op operands and select conditions. Cells already
    /// known to be 0/1 (`is_boolean`) pass through; others cost one
    /// equality gadget.
    fn booleanize(&mut self, c: &str) -> String {
        if self.is_boolean(c) {
            return c.to_string();
        }
        let expr = format!("if {} == 0 {{ 0 }} else {{ 1 }}", self.as_field(c));
        self.bind_flag(&expr)
    }

    /// True when `c` is 0 or 1 for every accepted witness: a 0/1 literal, a
    /// cell bound to `<= 1` (comparison and logic results, bool casts,
    /// extracted bits), or a cell whose interval lies in [0, 1]. Every such
    /// fact is produced by a lowering that guarantees it (`bind_flag`,
    /// enforced asserts), never by an unchecked claim.
    fn is_boolean(&self, c: &str) -> bool {
        matches!(self.s_int_of(c), Some(0) | Some(1))
            || matches!(self.bound_of(c), Some(b) if b <= 1)
            || matches!(self.ival(c), Some((lo, hi)) if lo >= 0 && hi <= 1)
    }

    /// Bind a 0/1-valued Field expression and record that it is boolean.
    fn bind_flag(&mut self, expr: &str) -> String {
        let v = self.bind(expr);
        self.set_bound(&v, 1);
        v
    }

    /// Python `(a // b, a % b)` as two cells, shared between the integer and
    /// float floor division and remainder on the same operands. A positive
    /// compile-time divisor below 2^120 uses `divmod_by_positive` (two small
    /// range checks on the remainder); any other divisor the general
    /// `floor_divmod` (one sign projection of b; a zero divisor gives
    /// (0, 0), as the mock). Both pin the unique quotient and remainder.
    fn floor_divmod_cells(&mut self, a: &str, b: &str) -> (String, String) {
        let (af, bf) = (self.as_field(a), self.as_field(b));
        let call = match self.s_int_of(b) {
            Some(d) if d > 0 && d < (1i128 << 120) => {
                let bits = (128 - ((d - 1) as u128).leading_zeros()).max(1);
                self.use_lib("divmod_by_positive");
                format!("divmod_by_positive::<{}>({}, {})", bits, af, bf)
            }
            _ => {
                self.use_lib("floor_divmod");
                format!("floor_divmod({}, {})", af, bf)
            }
        };
        if let Some(pair) = self.divmod_cache.get(&call) {
            return pair.clone();
        }
        let (q, r) = (self.fresh(), self.fresh());
        let _ = writeln!(self.body, "    let ({}, {}) = {};", q, r, call);
        self.divmod_cache.insert(call, (q.clone(), r.clone()));
        (q, r)
    }

    /// Operands for a native u32 `/` or `%`. A divisor that may be zero is
    /// replaced by 1 and the dividend by 0, so a zero divisor gives 0 (the
    /// mock's convention) instead of failing the proof.
    fn u32_division_operands(&mut self, a: &str, b: &str) -> (String, String) {
        if matches!(self.s_int_of(b), Some(d) if d > 0) {
            return (a.to_string(), b.to_string());
        }
        let zero = self.bind_flag(&format!("if {} == 0 {{ 1 }} else {{ 0 }}", self.as_field(b)));
        let n = self.bind_u32_with_bound(&format!("if {} == 1 {{ 0 }} else {{ {} }}", zero, a), self.bound_of(a).unwrap_or(u32::MAX as u64));
        let d = self.bind_u32_with_bound(&format!("if {} == 1 {{ 1 }} else {{ {} }}", zero, b), self.bound_of(b).unwrap_or(u32::MAX as u64).max(1));
        (n, d)
    }

    /// Emit `0 <= cell < 2^bits` as a standalone range check on the Field
    /// value (never removed by `dce_body`, which only drops `let` lines).
    fn emit_range_check(&mut self, cell: &str, bits: u32) {
        let target = if is_identifier(cell) && !self.is_u32(cell) {
            cell.to_string()
        } else {
            let t = self.fresh();
            let _ = writeln!(self.body, "    let {}: Field = {};", t, self.as_field(cell));
            t
        };
        let _ = writeln!(self.body, "    {}.assert_max_bit_size::<{}>();", target, bits);
    }

    /// Emit `0 <= cell < hi` for a compile-time bound `hi`: two range checks
    /// of `bits = bit_length(hi - 1)` bits on `cell` and on `hi - 1 - cell`.
    /// A negative cell or one above `hi - 1` turns one of them into a field
    /// element of about 254 bits, which fails. `hi <= 0` admits nothing.
    fn assert_index_in_range(&mut self, cell: &str, hi: i128) {
        if hi <= 0 {
            let _ = writeln!(self.body, "    assert(1 == 0);");
            return;
        }
        if self.is_u32(cell) && hi <= u32::MAX as i128 {
            let _ = writeln!(self.body, "    assert({} < {});", cell, hi);
            return;
        }
        let bits = (128 - ((hi - 1) as u128).leading_zeros()).max(1);
        let t = self.fresh();
        let _ = writeln!(self.body, "    let {}: Field = {};", t, self.as_field(cell));
        let _ = writeln!(self.body, "    {}.assert_max_bit_size::<{}>();", t, bits);
        let u = self.fresh();
        let _ = writeln!(self.body, "    let {}: Field = {} - {};", u, hi - 1, t);
        let _ = writeln!(self.body, "    {}.assert_max_bit_size::<{}>();", u, bits);
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

    /// True when `s1*v + s2*w + offset` lies in [0, len) for every v in
    /// `vs` and w in `ws` (extremes of an affine form over a box).
    fn all_combinations_in_range(
        s1: i64,
        vs: &[i64],
        s2: i64,
        ws: &[i64],
        offset: i64,
        len: usize,
    ) -> bool {
        let extremes = |s: i64, xs: &[i64]| {
            let (lo, hi) = (xs[0] * s, xs[xs.len() - 1] * s);
            (lo.min(hi), lo.max(hi))
        };
        let (a_lo, a_hi) = extremes(s1, vs);
        let (b_lo, b_hi) = extremes(s2, ws);
        a_lo + b_lo + offset >= 0 && a_hi + b_hi + offset < len as i64
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
                    return Ok(v);
                }
            }
        }
        let v = self.bind(&format!("{} + {}", self.as_field(a), self.as_field(b)));
        Ok(v)
    }

    pub fn new(
        inputs: Vec<EmittedInput>,
        stmt_bounds: super::bound_analysis::BoundMap,
        intrinsic_count: usize,
        provenance: &'a crate::provenance::ProvenanceTable,
    ) -> NoirEmitter<'a> {
        // u32 promotion (`maybe_promote_to_u32`) range-checks a Field cell
        // into u32 so later integer ops run natively. The first range check
        // of 14+ bits pulls bb's range table into the circuit (~2.7k gates),
        // so promotion only pays off when the kernel already has enough ops
        // that need such checks; `count_u32_intrinsic_ops` counts them. The
        // threshold affects cost only, never soundness (every promotion is
        // range-checked). Measured 2026-10-06 on the performance set:
        // promoting unconditionally costs 1.9% more gates (geomean) than
        // this threshold.
        let u32_amortizable = intrinsic_count >= 20;
        Self {
            prov: provenance,
            vid_of_var: HashMap::new(),
            var_of_vid: HashMap::new(),
            inputs,
            body: String::new(),
            next_id: 0,
            used_libs: BTreeMap::new(),
            exposes: Vec::new(),
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
            memories: HashMap::new(),
            memory_init: HashMap::new(),
            divmod_cache: HashMap::new(),
            emitted_u128_asserts: HashSet::new(),
            cse_cache: HashMap::new(),
            cell_bound: HashMap::new(),
            u32_cells: HashSet::new(),
            literal_vals: HashMap::new(),
            stmt_bounds,
            current_stmt_bound: None,
            u32_amortizable,
        }
    }

    // ── Bound-driven u32 emit helpers ────────────────────────────────
    // These let the integer-arithmetic / compare ops decide whether to
    // bind a result as `u32` or as `Field` (see `u32_cells`).

    fn bound_of(&self, cell: &str) -> Option<u64> {
        self.cell_bound.get(cell).copied()
    }

    fn is_u32(&self, cell: &str) -> bool {
        self.u32_cells.contains(cell)
    }

    fn set_bound(&mut self, cell: &str, bound: u64) {
        self.cell_bound.insert(cell.to_string(), bound);
    }

    fn mark_u32(&mut self, cell: &str) {
        self.u32_cells.insert(cell.to_string());
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

    /// Route a Field-typed cell with a known u32 bound through a one-time
    /// range-checked `checked_u32` conversion, so downstream u128 widenings
    /// become free and CSE can deduplicate the conversion across many
    /// compare sites. Returns the new u32-typed SSA name; subsequent uses
    /// see `is_u32(returned) == true`. The range check makes the u32 copy
    /// equal to the Field value for every accepted witness, so u32 cells
    /// are genuine u32 values; a wrong bound fails the proof.
    ///
    /// Gated on `u32_amortizable`: the 32-bit range check pulls in bb's
    /// range table, which is only worth paying when the kernel has enough
    /// u32-eligible cells. In small kernels the cell stays Field.
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
        self.use_lib("checked_u32");
        let v = self.bind_u32_with_bound(&format!("checked_u32({})", cell), bound);
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
        // Same value, same provenance records.
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

    // ── Compile-time constant accessors ──────────────────────────────
    // `literal_vals` holds IR literals and values derived only from them;
    // these are the only values an emission decision may branch on.

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

    /// Register a compile-time constant for a cell derived only from other
    /// constants (e.g. `abs_f` of a literal, `float_cast` of a literal int).
    fn fold_literal(&mut self, v: &str, val: ConcreteValue) {
        self.literal_vals.insert(v.to_string(), val);
    }

    fn use_lib(&mut self, sym: &'static str) {
        self.used_libs.insert(sym, ());
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
        // `DivI` is field division a * b^-1 (it equals integer division when b
        // divides a), and 0 for b == 0, as the mock's `div_i`.
        self.use_lib("field_div_or_zero");
        let (af, bf) = (self.as_field(a), self.as_field(b));
        Ok(self.bind(&format!("field_div_or_zero({}, {})", af, bf)))
    }
    fn floor_div_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // u32 cells are non-negative, where Noir's u32 division is the floor
        // quotient.
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self.bound_of(a).unwrap_or(u32::MAX as u64);
            let (n, d) = self.u32_division_operands(a, b);
            return Ok(self.bind_u32_with_bound(&format!("{} / {}", n, d), bound));
        }
        let (q, _r) = self.floor_divmod_cells(a, b);
        Ok(q)
    }

    fn floor_div_i_pinned(
        &mut self,
        a: &CellRef,
        b: &CellRef,
        divisor_bits: u32,
    ) -> Result<CellRef, ProvingError> {
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self.bound_of(a).unwrap_or(u32::MAX as u64);
            return Ok(self.bind_u32_with_bound(&format!("{} / {}", a, b), bound));
        }
        // The producer (e.g. reshape `-1` dim inference) promises a, b, the
        // quotient and the remainder are non-negative `divisor_bits`-bit
        // values. `q_floor_div_bounded` checks exactly that (so a false
        // promise fails the proof) and pins the unique quotient.
        let bits = divisor_bits.max(1);
        self.use_lib("q_floor_div_bounded");
        let (af, bf) = (self.as_field(a), self.as_field(b));
        Ok(self.bind(&format!("q_floor_div_bounded::<{}>({}, {})", bits, af, bf)))
    }
    fn mod_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // Python remainder (the divisor's sign; 0 for a zero divisor).
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self
                .bound_of(b)
                .unwrap_or(u32::MAX as u64)
                .saturating_sub(1);
            let (n, d) = self.u32_division_operands(a, b);
            return Ok(self.bind_u32_with_bound(&format!("{} % {}", n, d), bound));
        }
        let (_q, r) = self.floor_divmod_cells(a, b);
        Ok(r)
    }
    fn pow_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "pow_i")
    }
    fn abs_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qabs_i");
        let v = self.bind(&format!("qabs_i({})", self.as_field(a)));
        Ok(v)
    }
    fn sign_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsign_i");
        let v = self.bind(&format!("qsign_i({})", self.as_field(a)));
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
        let v = self.bind(&format!("qadd({}, {})", self.as_field(a), self.as_field(b)));
        Ok(v)
    }
    fn sub_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsub");
        let v = self.bind(&format!("qsub({}, {})", self.as_field(a), self.as_field(b)));
        Ok(v)
    }
    fn mul_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qmul");
        let v = self.bind(&format!("qmul({}, {})", self.as_field(a), self.as_field(b)));
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
        let v = self.bind(&format!("qscale_down_raw({})", acc));
        Ok(v)
    }
    fn div_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qdiv");
        let v = self.bind(&format!("qdiv({}, {})", self.as_field(a), self.as_field(b)));
        Ok(v)
    }
    fn floor_div_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // floor(a / b) as a float: the Python floor quotient of the raw Q32
        // integers, re-scaled (kernel::fp_float_floor_div).
        let (q, _r) = self.floor_divmod_cells(a, b);
        Ok(self.bind(&format!("{} * {}", self.as_field(&q), Q_U64)))
    }
    fn mod_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // a % b with the divisor's sign: the floor remainder of the raw Q32
        // integers is already Q-scaled (kernel::fp_float_mod).
        let (_q, r) = self.floor_divmod_cells(a, b);
        Ok(r)
    }
    fn pow_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qpow");
        let v = self.bind(&format!("qpow({}, {})", self.as_field(a), self.as_field(b)));
        Ok(v)
    }
    fn pow_f_int(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // Float base ^ integer exponent by square-and-multiply. The exponent
        // must be a non-negative value below 2^32: the range-checked
        // conversion fails the proof otherwise (a truncating cast would have
        // used the low bits of a negative exponent's field encoding).
        self.use_lib("qpow_int_q32");
        let e = self.to_uint(b, UintWidth::U32);
        let af = self.as_field(a);
        Ok(self.bind(&format!("qpow_int_q32({}, {} as u64)", af, e)))
    }
    fn abs_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qabs");
        let v = self.bind(&format!("qabs({})", self.as_field(a)));
        // Literal propagation: |literal| is itself a compile-time
        // constant. Fold the structural value and carry `literal_vals`
        // membership so downstream fast paths (e.g. the select
        // zero-branch) fire. Read the STRUCTURAL value only
        // (`s_float_of`), never the witness fold, per the
        // witness-independence rule.
        if let Some(f) = self.s_float_of(a) {
            self.fold_literal(&v, ConcreteValue::Float(f.abs()));
        }
        Ok(v)
    }
    fn sign_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsign");
        let v = self.bind(&format!("qsign({})", self.as_field(a)));
        Ok(v)
    }

    fn eq_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
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
                    let v = self.bind_flag(&format!(
                        "if ({} == {}) | ({} + {} == 0) {{ 1 }} else {{ 0 }}",
                        xf, j, xf, n - j
                    ));
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
            return Ok(v);
        }
        let v = self.bind_flag(&format!(
            "if {} == {} {{ 1 }} else {{ 0 }}",
            self.as_field(a),
            self.as_field(b)
        ));
        Ok(v)
    }
    fn ne_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if self.is_u32(a) && self.is_u32(b) {
            let v = self.bind_u32_with_bound(&format!("if {} == {} {{ 0 }} else {{ 1 }}", a, b), 1);
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
            return Ok(v);
        }
        let v = self.bind_flag(&format!(
            "if {} == {} {{ 0 }} else {{ 1 }}",
            self.as_field(a),
            self.as_field(b)
        ));
        Ok(v)
    }
    fn lt_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // min/max select recognition happens at IR level (the provenance
        // table's SelectI deposit); only the lt-zero peephole stays here.
        let out = self.compare_i(a, b, Cmp::Lt);
        if self.s_int_of(b) == Some(0) {
            self.lt_zero_of.insert(out.clone(), a.clone());
        }
        Ok(out)
    }

    fn lt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.bounded_cmp(a, b, Cmp::Lt))
    }

    fn gt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.bounded_cmp(a, b, Cmp::Gt))
    }

    fn index_in_box(
        &mut self,
        triples: &[(CellRef, CellRef, CellRef)],
    ) -> Result<CellRef, ProvingError> {
        // One combined boolean `(idx_0 >= lo_0) & (idx_0 < hi_0) & ...` over
        // unsigned views of the operands (the IR promises they are
        // non-negative; the range-checked conversions enforce it).
        if triples.is_empty() {
            return self.constant_int(1);
        }
        let mut parts: Vec<String> = Vec::with_capacity(triples.len() * 2);
        for (idx, lo, hi) in triples {
            // Promote per axis (CSE-d, so the same idx across axes shares
            // one conversion).
            let idx_p = self.maybe_promote_to_u32(idx);
            let lo_p = self.maybe_promote_to_u32(lo);
            let hi_p = self.maybe_promote_to_u32(hi);
            let i = self.to_uint(&idx_p, UintWidth::U64);
            let l = self.to_uint(&lo_p, UintWidth::U64);
            let h = self.to_uint(&hi_p, UintWidth::U64);
            parts.push(format!("({} >= {})", i, l));
            parts.push(format!("({} < {})", i, h));
        }
        let combined = parts.join(" & ");
        Ok(self.bind_flag(&format!("if {} {{ 1 }} else {{ 0 }}", combined)))
    }
    fn lte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare_i(a, b, Cmp::Le))
    }
    fn gt_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare_i(a, b, Cmp::Gt))
    }
    fn gte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare_i(a, b, Cmp::Ge))
    }

    fn eq_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (af, bf) = (self.as_field(a), self.as_field(b));
        Ok(self.bind_flag(&format!("if {} == {} {{ 1 }} else {{ 0 }}", af, bf)))
    }
    fn ne_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (af, bf) = (self.as_field(a), self.as_field(b));
        Ok(self.bind_flag(&format!("if {} == {} {{ 0 }} else {{ 1 }}", af, bf)))
    }
    // Float comparisons are signed comparisons of the Q32 integers (the
    // mock's `lt_f` is its `lt_i`).
    fn lt_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.signed_cmp(a, b, Cmp::Lt))
    }
    fn lte_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.signed_cmp(a, b, Cmp::Le))
    }
    fn gt_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.signed_cmp(a, b, Cmp::Gt))
    }
    fn gte_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.signed_cmp(a, b, Cmp::Ge))
    }

    fn sin_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsin");
        let v = self.bind(&format!("qsin({})", self.as_field(a)));
        Ok(v)
    }
    fn sinh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsinh");
        let v = self.bind(&format!("qsinh({})", self.as_field(a)));
        Ok(v)
    }
    fn cos_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qcos");
        let v = self.bind(&format!("qcos({})", self.as_field(a)));
        Ok(v)
    }
    fn cosh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qcosh");
        let v = self.bind(&format!("qcosh({})", self.as_field(a)));
        Ok(v)
    }
    fn tan_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qtan");
        let v = self.bind(&format!("qtan({})", self.as_field(a)));
        Ok(v)
    }
    fn tanh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qtanh");
        let v = self.bind(&format!("qtanh({})", self.as_field(a)));
        Ok(v)
    }
    fn sqrt_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // (No literal propagation: the circuit's floor root of a constant
        // need not equal the f64 sqrt the literal store would record.)
        self.use_lib("qsqrt");
        Ok(self.bind(&format!("qsqrt({})", self.as_field(a))))
    }
    fn exp_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qexp");
        let v = self.bind(&format!("qexp({})", self.as_field(a)));
        Ok(v)
    }
    fn log_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qlog");
        let v = self.bind(&format!("qlog({})", self.as_field(a)));
        Ok(v)
    }
    fn arcsin_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarcsin");
        let v = self.bind(&format!("qarcsin({})", self.as_field(a)));
        Ok(v)
    }
    fn arccos_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarccos");
        let v = self.bind(&format!("qarccos({})", self.as_field(a)));
        Ok(v)
    }
    fn arctan2_f(&mut self, y: &CellRef, x: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarctan2");
        let v = self.bind(&format!(
            "qarctan2({}, {})",
            self.as_field(y),
            self.as_field(x)
        ));
        Ok(v)
    }

    fn logical_and(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (a, b) = (self.booleanize(a), self.booleanize(b));
        if self.is_u32(&a) && self.is_u32(&b) {
            return Ok(self.bind_u32_with_bound(&format!("{} * {}", a, b), 1));
        }
        let (af, bf) = (self.as_field(&a), self.as_field(&b));
        Ok(self.bind_flag(&format!("{} * {}", af, bf)))
    }
    fn logical_or(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (a, b) = (self.booleanize(a), self.booleanize(b));
        if self.is_u32(&a) && self.is_u32(&b) {
            return Ok(self.bind_u32_with_bound(&format!("{} + {} - {} * {}", a, b, a, b), 1));
        }
        let (af, bf) = (self.as_field(&a), self.as_field(&b));
        Ok(self.bind_flag(&format!("{} + {} - {} * {}", af, bf, af, bf)))
    }
    fn logical_not(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let a = self.booleanize(a);
        if self.is_u32(&a) {
            return Ok(self.bind_u32_with_bound(&format!("1 - {}", a), 1));
        }
        Ok(self.bind_flag(&format!("1 - {}", a)))
    }

    fn select(
        &mut self,
        cond: &CellRef,
        if_true: &CellRef,
        if_false: &CellRef,
    ) -> Result<CellRef, ProvingError> {
        // `c * t + (1 - c) * f` picks a branch only for c in {0, 1}; any
        // other condition is reduced to 0/1 first (Python truthiness, as the
        // mock). When `f == 0` (the common dyn-slice / mask-then-default-zero
        // pattern) the second half is dead and the binding is `c * t`.
        let c = self.booleanize(cond);
        let is_zero_int = self.s_int_of(if_false) == Some(0);
        let is_zero_flt = self.s_float_of(if_false) == Some(0.0);
        let all_u32 = self.is_u32(&c)
            && self.is_u32(if_true)
            && (is_zero_int || is_zero_flt || self.is_u32(if_false));
        let v = if all_u32 {
            // Stay u32-clean: c * t (zero-default fast path) or
            // c * t + (1 - c) * f. Bound is max of t/f bounds.
            let t_bound = self.bound_of(if_true).unwrap_or(u32::MAX as u64);
            let f_bound = if is_zero_int || is_zero_flt {
                0
            } else {
                self.bound_of(if_false).unwrap_or(u32::MAX as u64)
            };
            let bound = t_bound.max(f_bound).min(u32::MAX as u64);
            let expr = if is_zero_int || is_zero_flt {
                format!("{} * {}", c, if_true)
            } else {
                format!("{} * {} + (1 - {}) * {}", c, if_true, c, if_false)
            };
            self.bind_u32_with_bound(&expr, bound)
        } else {
            let cf = self.as_field(&c);
            let t = self.as_field(if_true);
            let expr = if is_zero_int || is_zero_flt {
                format!("{} * {}", cf, t)
            } else {
                let f = self.as_field(if_false);
                format!("{} * {} + (1 - {}) * {}", cf, t, cf, f)
            };
            self.bind(&expr)
        };
        // A literal condition picking a literal branch keeps the result a
        // compile-time constant.
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
        // Float -> int: trunc(a / Q) toward zero, the mock's `fp_int_cast`.
        self.use_lib("q_div_q");
        Ok(self.bind(&format!("q_div_q({})", self.as_field(a))))
    }
    fn float_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Integer-to-float in Q32 means *= 2^32.
        let v = self.bind(&format!("{} * {}", self.as_field(a), Q_U64));
        // float(literal int) is a compile-time constant; `literal_vals`
        // stores the real value (the Q-scaling lives in the Noir text).
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
        // `bool(x)`: any non-zero x is 1 (Python truthiness, as the mock).
        Ok(self.booleanize(a))
    }

    fn read_input(&mut self, path: &InputPath, _is_public: bool) -> Result<CellRef, ProvingError> {
        // Public inputs are declared `pub` on `fn main` (see render_main_nr).
        let expr = input_index_expr(path, &self.inputs)?;
        Ok(self.bind(&expr))
    }

    fn read_external_result(
        &mut self,
        _store_idx: u32,
        _output_idx: u32,
    ) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "read_external_result")
    }

    fn expose_public(&mut self, a: &CellRef, _label: &str) -> Result<(), ProvingError> {
        // Exposed values become the `pub` return array of `main`, which is
        // typed `[Field; N]`, so u32 cells take the free `as Field`.
        self.exposes.push(self.as_field(a));
        Ok(())
    }

    fn assert_true(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        // Python truthiness, as the mock: any non-zero value passes.
        if self.is_boolean(a) {
            let _ = writeln!(self.body, "    assert({} == 1);", a);
        } else {
            let _ = writeln!(self.body, "    assert({} != 0);", self.as_field(a));
        }
        Ok(())
    }

    /// `0 <= idx < hi`. Self-standing: the lowering consults no interval or
    /// bound of `idx`, because the IR builder derives the interval
    /// `idx in [0, hi)` from this very assert (`deposit_provenance`), and a
    /// check lowered with the fact it is meant to establish checks nothing.
    fn assert_u128_lt(&mut self, idx: &CellRef, hi: &CellRef) -> Result<(), ProvingError> {
        // The assertion is idempotent: emit each (idx, hi) pair once.
        if !self.emitted_u128_asserts.insert((idx.clone(), hi.clone())) {
            return Ok(());
        }
        match self.s_int_of(hi) {
            Some(h) => {
                self.assert_index_in_range(idx, h);
                // The (now enforced) bound narrows later gathers on idx.
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
            None => {
                // Dynamic bound: both sides range-checked to u64 (no shape
                // reaches 2^64), then a native compare.
                let i = self.to_uint(idx, UintWidth::U64);
                let h = self.to_uint(hi, UintWidth::U64);
                let _ = writeln!(self.body, "    assert({} < {});", i, h);
            }
        }
        Ok(())
    }

    /// Constrain `0 <= idx < hi` and return bit `bit` of `idx`.
    fn assert_u128_lt_bit(
        &mut self,
        idx: &CellRef,
        hi: u64,
        bit: u32,
        _bits: u32,
    ) -> Result<CellRef, ProvingError> {
        if let Some(i) = self.s_int_of(idx) {
            let bit_val = if i >= 0 && (i as u128) < hi as u128 {
                (((i as u128) >> bit) & 1) as i64
            } else {
                0
            };
            return self.constant_int(bit_val);
        }
        let hi_i64 = i64::try_from(hi).map_err(|_| {
            ProvingError::synthesis(format!("AssertU128LtBit bound {} exceeds i64", hi))
        })?;
        let hi_cell = self.constant_int(hi_i64)?;
        self.assert_u128_lt(idx, &hi_cell)?;
        // idx is now known to lie in [0, hi), so the range-checked
        // conversion below cannot fail; the bit is a native shift and mask.
        if hi <= 1u64 << 32 {
            let iu = self.to_uint(idx, UintWidth::U32);
            Ok(self.bind_u32_with_bound(&format!("({} >> {}) & 1", iu, bit), 1))
        } else {
            let iu = self.to_uint(idx, UintWidth::U64);
            Ok(self.bind_flag(&format!("(({} >> {}) & 1) as Field", iu, bit)))
        }
    }

    /// `a >= 0` over the signed-i64 domain: one 63-bit range check on the
    /// Field value (as halo2-base). Self-standing for the same reason as
    /// `assert_u128_lt`: the builder records `a >= 0` from this assert.
    fn assert_non_neg_i(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        if self.is_u32(a) {
            // A u32 cell holds a genuine u32 value (every Field -> u32 entry
            // is range-checked), so it is non-negative by construction.
            return Ok(());
        }
        self.emit_range_check(a, 63);
        Ok(())
    }

    /// `a` in {0, 1}, checked directly (no bound or interval consulted).
    fn assert_bit(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        let af = self.as_field(a);
        let _ = writeln!(self.body, "    assert({} * ({} - 1) == 0);", af, af);
        Ok(())
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
            let mem = self.memories.entry(segment_id).or_default();
            let slot = usize::try_from(a).ok().filter(|&slot| slot < mem.len());
            let Some(slot) = slot else {
                // The mock rejects an out-of-range write; silently dropping
                // it would prove a different program.
                return Err(ProvingError::synthesis(format!(
                    "memory write out of bounds: segment {} address {} length {}",
                    segment_id,
                    a,
                    mem.len()
                )));
            };
            mem[slot] = value.clone();
            return Ok(());
        }
        // Dynamic address: in-circuit scatter. The address's affine lineage
        // names a root whose eq-indicators are cached and shared with every
        // other access on the same root; each reachable slot k updates as
        // old + ind * (value - old). The one-hot assert over the reachable
        // root values makes an address outside the segment (or outside the
        // root's interval) fail the proof, as the mock rejects it, instead
        // of writing nothing.
        let len = self.memories.get(&segment_id).map_or(0, |m| m.len());
        let aff = self.aff_of(addr);
        let (root, scale, offset) = if aff.terms.len() == 1 {
            (aff.terms[0].0.clone(), aff.terms[0].1, aff.offset)
        } else {
            (addr.clone(), 1, 0)
        };
        let root_iv = self.ival(&root);
        let vs = Self::feasible(scale, offset, len, root_iv);
        if vs.is_empty() {
            let _ = writeln!(self.body, "    assert(1 == 0);");
            return Ok(());
        }
        self.assert_one_hot(&root, &vs);
        let vf = self.as_field(value);
        for &v in &vs {
            let slot = (v * scale + offset) as usize;
            let ind = self.indicator(&root, v);
            let old_v = self.memories.get(&segment_id).unwrap()[slot].clone();
            let of = self.as_field(&old_v);
            let indf = self.as_field(&ind);
            let nv = self.bind(&format!("{of} + {indf} * ({vf} - {of})"));
            self.memories.get_mut(&segment_id).unwrap()[slot] = nv;
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
                    // Each root is now in its set, but a combination of two
                    // members need not be a valid slot (it would gather 0).
                    // When some combination falls outside the segment, check
                    // the full address too.
                    if !Self::all_combinations_in_range(s1, &v_all, s2, &w_vals, aff.offset, len) {
                        self.assert_index_in_range(addr, len as i128);
                    }
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
        // Otherwise the slot holds a literal (the segment's init value).
        Ok(self.bind(&expr))
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

/// Render `main.nr`. Returns the source text; `main` returns the exposed
/// values in IR order (plus, for a circuit with no output and no body, one
/// synthetic return value; see below).
fn render_main_nr(inputs: &[EmittedInput], emitter: &NoirEmitter) -> String {
    let mut out = String::new();
    out.push_str(&emitter.render_use_decls());
    out.push_str("fn main(");
    let mut first = true;
    for inp in inputs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        let visibility = if inp.is_public { "pub " } else { "" };
        // Booleans arrive as Field 0/1; the body treats them as integers.
        match &inp.kind {
            InputKind::ScalarField | InputKind::Bool => {
                let _ = write!(out, "{}: {}Field", inp.name, visibility);
            }
            InputKind::ArrayField { len } => {
                let _ = write!(out, "{}: {}[Field; {}]", inp.name, visibility, len);
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prove::traits::Synthesizer;

    /// Build a bare emitter with no inputs, sufficient to drive single
    /// lowering methods in isolation.
    fn bare_emitter(prov: &crate::provenance::ProvenanceTable) -> NoirEmitter<'_> {
        NoirEmitter::new(Vec::new(), HashMap::new(), 0, prov)
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

    /// Emitter with u32 promotion enabled (the intrinsic-op threshold met).
    fn promoting_emitter(prov: &crate::provenance::ProvenanceTable) -> NoirEmitter<'_> {
        NoirEmitter::new(Vec::new(), HashMap::new(), 20, prov)
    }

    // B4: a Field cell never reaches an unsigned type through a truncating
    // `as uN` cast; the interval only picks the width of a checked conversion.
    #[test]
    fn unsigned_cmp_converts_through_range_checks() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        em.interval_of.insert("x".to_string(), (0, 200));
        em.interval_of.insert("y".to_string(), (0, 200));
        em.lt_i(&"x".to_string(), &"y".to_string()).unwrap();
        assert!(em.body.contains("checked_u8(x)") && em.body.contains("checked_u8(y)"), "{}", em.body);
        assert!(!em.body.contains("x as u8"), "{}", em.body);
    }

    // Signed comparisons go through the gap-pinned `qlt` (one 64-bit range
    // check), and a comparison and its negation share one `qlt` call.
    #[test]
    fn signed_comparisons_share_one_gap_pinned_qlt() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let (x, y) = ("x".to_string(), "y".to_string());
        em.lt_i(&x, &y).unwrap();
        em.gte_i(&x, &y).unwrap();
        em.gt_i(&y, &x).unwrap();
        assert_eq!(em.body.matches("qlt(x, y)").count(), 1, "{}", em.body);
        assert!(!em.body.contains("to_signed"), "{}", em.body);
        em.lte_i(&x, &y).unwrap();
        assert!(em.body.contains("qlt(y, x)"), "{}", em.body);
    }

    #[test]
    fn u32_promotion_is_range_checked() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = promoting_emitter(&prov);
        em.set_bound("x", 9);
        let p = em.maybe_promote_to_u32("x");
        assert!(em.is_u32(&p));
        assert!(em.body.contains("checked_u32(x)"), "{}", em.body);
        assert!(!em.body.contains("x as u32"), "{}", em.body);
    }

    // B4(b): the index of `AssertU128Lt` carries the interval the builder
    // derives from this assert; the lowering must not use it (no u32
    // promotion of idx, a direct range check on the Field value instead).
    #[test]
    fn assert_u128_lt_does_not_use_its_own_interval() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = promoting_emitter(&prov);
        em.set_bound("idx", 9);
        em.interval_of.insert("idx".to_string(), (0, 9));
        lit_int(&mut em, "hi", 10);
        em.assert_u128_lt(&"idx".to_string(), &"hi".to_string()).unwrap();
        assert!(!em.body.contains("checked_u32(idx)"), "{}", em.body);
        assert!(em.body.contains("= idx;"), "{}", em.body);
        assert_eq!(em.body.matches("assert_max_bit_size::<4>()").count(), 2, "{}", em.body);
    }

    // B4(a): `AssertNonNegI` is a 63-bit range check on the raw value, even
    // when the cell's interval already says `>= 0` (from this assert).
    #[test]
    fn assert_non_neg_is_a_range_check_on_the_value() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        em.interval_of.insert("x".to_string(), (0, i64::MAX));
        em.assert_non_neg_i(&"x".to_string()).unwrap();
        assert!(em.body.contains("x.assert_max_bit_size::<63>();"), "{}", em.body);
        assert!(!em.body.contains(" as u"), "{}", em.body);
    }

    // M6: logic ops read their operands with Python truthiness.
    #[test]
    fn logic_ops_booleanize_operands_not_known_to_be_bits() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        em.logical_not(&"x".to_string()).unwrap();
        assert!(em.body.contains("if x == 0 { 0 } else { 1 }"), "{}", em.body);
        // A cell known to be 0/1 is used as is.
        em.set_bound("flag", 1);
        let before = em.body.matches("== 0 { 0 } else { 1 }").count();
        em.logical_not(&"flag".to_string()).unwrap();
        assert_eq!(em.body.matches("== 0 { 0 } else { 1 }").count(), before);
        assert!(em.body.contains("1 - flag"), "{}", em.body);
    }

    // M5: a dynamic-address write asserts that the address is one of the
    // reachable slots.
    #[test]
    fn dynamic_write_asserts_membership() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        em.allocate_memory(0, 3, 0).unwrap();
        em.write_memory(0, &"i".to_string(), &"v".to_string()).unwrap();
        assert!(em.body.contains("== 1);"), "{}", em.body);
        for slot in 0..3 {
            assert!(em.body.contains(&format!("if i == {} {{ 1 }} else {{ 0 }}", slot)));
        }
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

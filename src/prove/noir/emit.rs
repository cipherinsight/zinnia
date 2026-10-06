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
/// bb gates of one `if x == v { 1 } else { 0 }` indicator (an inverse hint
/// and two arithmetic gates).
const INDICATOR_GATES: usize = 3;
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
    let mut emitter = NoirEmitter::new(input_params.clone(), bounds, &ir.provenance);
    emitter.dynamic_reads_per_epoch = count_dynamic_reads_per_epoch(ir);
    emitter.wrapped_index_ranges = wrapped_index_ranges(ir);
    emitter.asserted_comparisons = asserted_comparisons(ir);
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

/// Comparison statements whose result is used only by `Assert` statements:
/// their lowering checks the asserted outcome directly
/// (`NoirEmitter::asserted_compare`) and the asserts become trivial.
fn asserted_comparisons(ir: &IRGraph) -> HashSet<u32> {
    use crate::ir_defs::IR;
    let mut assert_uses: HashMap<u32, usize> = HashMap::new();
    let mut other_uses: HashSet<u32> = HashSet::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            if matches!(stmt.ir, IR::Assert) {
                *assert_uses.entry(*arg).or_insert(0) += 1;
            } else {
                other_uses.insert(*arg);
            }
        }
    }
    ir.stmts
        .iter()
        .filter(|s| {
            matches!(
                s.ir,
                IR::LtI | IR::LteI | IR::GtI | IR::GteI | IR::LtF | IR::LteF | IR::GtF | IR::GteF
                    | IR::LtIBounded | IR::GtIBounded | IR::IndexInBox { .. }
            )
        })
        .map(|s| s.stmt_id)
        .filter(|id| assert_uses.contains_key(id) && !other_uses.contains(id))
        .collect()
}

/// Ranges that NumPy's index wrap-around proves for raw indices. A runtime
/// index x into an axis of length N is lowered to `y = x + N if x < 0 else
/// x` (`LtI(x, 0)`, `AddI(x, N)`, `SelectI`) followed by `AssertU128Lt(y,
/// hi)` with constants N and hi <= N. Every witness the assert accepts has
/// x in [-N, hi - 1], so for those witnesses the comparison `x < 0` can use
/// a range check sized by that range (`comparison_width`): it fails only on
/// an x the assert rejects anyway, and is never wrong where it passes.
/// Returns that range per raw index value.
fn wrapped_index_ranges(ir: &IRGraph) -> HashMap<crate::types::ValueId, (i64, i64)> {
    use crate::ir_defs::IR;
    let by_id: HashMap<u32, &crate::ir::IRStatement> =
        ir.stmts.iter().map(|s| (s.stmt_id, s)).collect();
    let constant = |id: u32| match by_id.get(&id).map(|s| &s.ir) {
        Some(IR::ConstantInt { value }) => Some(*value),
        _ => None,
    };
    let mut ranges = HashMap::new();
    for stmt in &ir.stmts {
        let (IR::AssertU128Lt, [sel, hi]) = (&stmt.ir, stmt.arguments.as_slice()) else {
            continue;
        };
        let (Some(hi), Some(select)) = (constant(*hi), by_id.get(sel)) else {
            continue;
        };
        let (IR::SelectI, [cond, wrapped, raw]) = (&select.ir, select.arguments.as_slice()) else {
            continue;
        };
        let (Some(cond), Some(wrapped)) = (by_id.get(cond), by_id.get(wrapped)) else {
            continue;
        };
        let is_negative_test = matches!(cond.ir, IR::LtI)
            && cond.arguments.first() == Some(raw)
            && cond.arguments.get(1).and_then(|z| constant(*z)) == Some(0);
        let shift = match (&wrapped.ir, wrapped.arguments.as_slice()) {
            (IR::AddI, [a, b]) if a == raw => constant(*b),
            (IR::AddI, [a, b]) if b == raw => constant(*a),
            _ => None,
        };
        if let (true, Some(n)) = (is_negative_test, shift) {
            if hi >= 1 && n >= hi {
                let raw_vid = select.arg_values[2];
                ranges.insert(raw_vid, (-n, hi - 1));
            }
        }
    }
    ranges
}

/// For every memory segment and write epoch (the writes to the segment seen
/// so far, the emitter's `write_version`), the number of `ReadMemory`
/// statements whose address is not an IR constant. A native array read
/// shares the cost of materializing the epoch's array with these reads.
fn count_dynamic_reads_per_epoch(ir: &IRGraph) -> HashMap<(u32, u64), usize> {
    let constants: HashSet<u32> = ir
        .stmts
        .iter()
        .filter(|s| matches!(s.ir, crate::ir_defs::IR::ConstantInt { .. }))
        .map(|s| s.stmt_id)
        .collect();
    let mut epoch: HashMap<u32, u64> = HashMap::new();
    let mut counts: HashMap<(u32, u64), usize> = HashMap::new();
    for stmt in &ir.stmts {
        match &stmt.ir {
            crate::ir_defs::IR::WriteMemory { segment_id } => {
                *epoch.entry(*segment_id).or_insert(0) += 1;
            }
            crate::ir_defs::IR::ReadMemory { segment_id } => {
                let dynamic = stmt.arguments.first().is_some_and(|a| !constants.contains(a));
                if dynamic {
                    let e = epoch.get(segment_id).copied().unwrap_or(0);
                    *counts.entry((*segment_id, e)).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }
    counts
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
                // matches downstream identifier refs. Both `let v: u8 =`
                // (typed cells from `bind_typed`) and `let (a, b) =`
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
    /// Comparisons used only by asserts (see `asserted_comparisons`).
    asserted_comparisons: HashSet<u32>,
    /// Statement being dispatched (set by `on_stmt_begin`).
    current_stmt: Option<u32>,
    /// Ranges a check emitted earlier in this circuit proves for a cell
    /// (`asserted_compare` against a constant): every accepted witness lies
    /// in them, so they may decide later comparisons of the cell.
    proven_range: HashMap<String, (i128, i128)>,
    /// Ranges of raw indices that every accepted witness satisfies, from
    /// NumPy's wrap-around lowering (see `wrapped_index_ranges`). Used only
    /// to size comparisons.
    wrapped_index_ranges: HashMap<crate::types::ValueId, (i64, i64)>,
    /// Dynamic reads per (segment, write epoch), from the IR (see
    /// `count_dynamic_reads_per_epoch`).
    dynamic_reads_per_epoch: HashMap<(u32, u64), usize>,
    /// Native Noir arrays holding a segment's slots, per (segment, epoch).
    segment_arrays: HashMap<(u32, u64), String>,
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
    /// Per-cell inclusive upper bound on the (non-negative) value, from the
    /// IR bound analysis (statement results, see `on_stmt_result`) and from
    /// lowerings whose results are 0/1 (`bind_flag`). Read through `ival`
    /// and `is_boolean`.
    cell_bound: HashMap<String, u64>,
    /// Compile-time constant store: SSA name -> value known at circuit-build
    /// time (IR `ConstantInt`/`ConstantFloat`/`ConstantBool`, or values
    /// derived only from such constants). Read by `s_int_of`/`s_float_of`/
    /// `s_bool_of`.
    literal_vals: HashMap<String, ConcreteValue>,
    /// Float cells that are exactly `n * 2^32` for an integer `n`, with the
    /// Field expression of `n`: `FloatCast` results and integral float
    /// constants. A Q32 product with such a cell is exact (see
    /// [`NoirEmitter::exact_product`]).
    integral_float: HashMap<String, String>,
    /// Cells that are non-negative by construction (bit compositions of
    /// `low_bits`), so comparing them with 0 is a zero test.
    nonneg_by_construction: HashSet<String>,
    /// Cells equal to `2^k * bit` for a Noir `bool` expression `bit` (an AND
    /// with a one-bit mask): their zero test is the bit itself.
    scaled_bit: HashMap<String, String>,
    /// `low_bits::<NB>(x)` decompositions already emitted: x -> (NB, name).
    low_bits_of: HashMap<String, (u32, String)>,
    /// `sign_abs(x)` pairs already emitted: x -> (negative: bool, |x|).
    sign_abs_of: HashMap<String, (String, String)>,
    /// Bit decompositions of bounded indices: (idx, width) -> name.
    index_bits_of: HashMap<(String, u32), String>,
    /// IR-level forward-dataflow bounds, keyed by stmt_id, from
    /// [`super::bound_analysis::run`]. The interpreter's per-stmt hook
    /// (`on_stmt_begin`) sets `current_stmt_bound` to the upcoming
    /// statement's entry, which `on_stmt_result` stamps on its result.
    stmt_bounds: super::bound_analysis::BoundMap,
    /// Bound for the statement currently being dispatched (set by
    /// `on_stmt_begin`).
    current_stmt_bound: Option<u64>,
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
    /// The comparison with its operands swapped: `a cmp b` is
    /// `b cmp.mirrored() a`.
    fn mirrored(self) -> Self {
        match self {
            Cmp::Lt => Cmp::Gt,
            Cmp::Le => Cmp::Ge,
            Cmp::Gt => Cmp::Lt,
            Cmp::Ge => Cmp::Le,
        }
    }

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UintWidth {
    U8,
    U32,
    U64,
}

impl UintWidth {
    fn ty(self) -> &'static str {
        match self {
            UintWidth::U8 => "u8",
            UintWidth::U32 => "u32",
            UintWidth::U64 => "u64",
        }
    }

    /// The `noir_fpc` range-checked conversion into this type.
    fn checked_fn(self) -> &'static str {
        match self {
            UintWidth::U8 => "checked_u8",
            UintWidth::U32 => "checked_u32",
            UintWidth::U64 => "checked_u64",
        }
    }
}

impl<'a> NoirEmitter<'a> {


    /// `a op b` for an integer `+`, `-` or `*` as a Field expression, with
    /// the interval of the result when both operands have one (exact
    /// integer arithmetic: the values involved stay far below p).
    fn integer_arith(&mut self, a: &str, b: &str, op: char) -> String {
        let v = self.bind(&format!("{a} {op} {b}"));
        if let (Some((al, ah)), Some((bl, bh))) = (self.ival(a), self.ival(b)) {
            let (al, ah, bl, bh) = (al as i128, ah as i128, bl as i128, bh as i128);
            let (lo, hi) = match op {
                '+' => (al + bl, ah + bh),
                '-' => (al - bh, ah - bl),
                _ => {
                    let corners = [al * bl, al * bh, ah * bl, ah * bh];
                    (*corners.iter().min().unwrap(), *corners.iter().max().unwrap())
                }
            };
            if let (Ok(lo), Ok(hi)) = (i64::try_from(lo), i64::try_from(hi)) {
                if !self.interval_of.contains_key(&v) {
                    self.interval_of.insert(v.clone(), (lo, hi));
                }
            }
        }
        v
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
        // Cells with a known analysis bound are in [0, bound]: the bound
        // analysis only bounds non-negative values.
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

    /// `a cmp b` as a 0/1 cell, the mock's signed comparison (the sign of
    /// `a - b`), for every IR comparison (`LtI`..`GteI`, `LtF`..`GteF`,
    /// `LtIBounded`, `GtIBounded`, the axes of `IndexInBox`). In order:
    /// a value non-negative by construction against 0 is a zero test; a
    /// value against 0 reads the sign of its `sign_abs` pair when that pair
    /// exists or costs no more than the comparison (`compare_with_zero_by_sign`);
    /// operands with non-negative intervals below 2^8 compare as u8
    /// (`unsigned_compare`); anything else goes through the library's
    /// gap-pinned `qlt_bits` (`gap_compare`).
    fn compare(&mut self, a: &str, b: &str, cmp: Cmp) -> String {
        if let Some(v) = self.compare_nonneg_with_zero(a, b, cmp) {
            return v;
        }
        if let Some(outcome) = self.decide_from_proven_ranges(a, b, cmp) {
            return self.literal_int(outcome as i64);
        }
        let asserted = self
            .current_stmt
            .is_some_and(|id| self.asserted_comparisons.contains(&id));
        if asserted && !self.has_cached_sign_for_zero_test(a, b) {
            self.asserted_compare(a, b, cmp);
            return self.literal_int(1);
        }
        if let Some(v) = self.compare_with_zero_by_sign(a, b, cmp) {
            return v;
        }
        if let Some(v) = self.unsigned_compare(a, b, cmp) {
            return v;
        }
        self.gap_compare(a, b, cmp)
    }

    /// `a cmp b` for operands whose intervals lie in [0, 2^8): both are
    /// converted to u8 by range-checked conversions and compared natively.
    /// A conversion is shared by every comparison of the same cell (and
    /// free for a constant), which makes this cheaper than `gap_compare`
    /// when one operand meets many others (e.g. `i < pos` for every i).
    /// The interval only picks the path; an operand outside it fails the
    /// proof. Wider operands use `gap_compare`: a u32 conversion would cost
    /// two 32-bit checks (and pull bb's 14-bit lookup table into circuits
    /// that may not need it) where `gap_compare` costs one.
    fn unsigned_compare(&mut self, a: &str, b: &str, cmp: Cmp) -> Option<String> {
        let (al, ah) = self.ival(a)?;
        let (bl, bh) = self.ival(b)?;
        if al < 0 || bl < 0 || ah.max(bh) > 0xff {
            return None;
        }
        let ca = self.to_uint(a, UintWidth::U8);
        let cb = self.to_uint(b, UintWidth::U8);
        Some(self.bind_flag(&format!("if {} {} {} {{ 1 }} else {{ 0 }}", ca, cmp.op(), cb)))
    }

    /// `a cmp b` through `qlt_bits::<W>`, which pins its answer with one
    /// W-bit range check of the gap between the operands: it is the mock's
    /// comparison for every pair with a - b in [-2^W, 2^W), and fails the
    /// proof for any other pair (`comparison_width` picks W). `a <= b` is
    /// `!(b < a)`, `a > b` is `b < a` and `a >= b` is `!(a < b)`, so a
    /// comparison and its negation share one call.
    fn gap_compare(&mut self, a: &str, b: &str, cmp: Cmp) -> String {
        let (lhs, rhs, negate) = match cmp {
            Cmp::Lt => (a, b, false),
            Cmp::Le => (b, a, true),
            Cmp::Gt => (b, a, false),
            Cmp::Ge => (a, b, true),
        };
        let width = self.comparison_width(lhs, rhs);
        let (lf, rf) = (lhs, rhs);
        self.use_lib("qlt_bits");
        let less = self.bind_typed(&format!("qlt_bits::<{width}>({lf}, {rf})"), "bool");
        let flag = if negate { format!("(!{less}) as Field") } else { format!("{less} as Field") };
        self.bind_flag(&flag)
    }

    /// The gap width W for comparing `a` with `b`: 64 (any two values of the
    /// signed 64-bit window), or less when the operands' intervals bound
    /// a - b inside [-2^W, 2^W). The intervals only choose the width: they
    /// come from constants, enforced asserts and lowerings (see `ival`) or
    /// from the bounds check of a wrapped index (`wrapped_index_ranges`),
    /// and a pair outside them fails the proof rather than getting a wrong
    /// answer. W is rounded up to 8 or 28 bits, whose lookup tables bb
    /// shares with the 64-bit checks (a 14-bit limb table and an 8-bit
    /// one), so a narrow comparison never adds a table of its own.
    fn comparison_width(&self, a: &str, b: &str) -> u32 {
        const WINDOW_WIDTH: u32 = 64;
        let range = |cell: &str| {
            self.ival(cell).or_else(|| {
                let vid = self.vid_of_var.get(cell)?;
                self.wrapped_index_ranges.get(vid).copied()
            })
        };
        let (Some((al, ah)), Some((bl, bh))) = (range(a), range(b)) else {
            return WINDOW_WIDTH;
        };
        let lo = al as i128 - bh as i128;
        let hi = ah as i128 - bl as i128;
        // 2^W must be at least max(-lo, hi + 1).
        let need = (-lo).max(hi + 1).max(1) as u128;
        let bits = 128 - (need - 1).leading_zeros();
        match bits {
            0..=8 => 8,
            9..=28 => 28,
            _ => WINDOW_WIDTH,
        }
    }

    /// Expression of unsigned type `width` holding the value of `cell`,
    /// through the library's range-checked `checked_uN`, so a value outside
    /// [0, 2^N) fails the proof. The emitter never writes a bare
    /// `Field as uN`: that cast truncates to the low N bits and is not a
    /// range check.
    fn to_uint(&mut self, cell: &str, width: UintWidth) -> String {
        let f = width.checked_fn();
        self.use_lib(f);
        self.bind_typed(&format!("{}({})", f, cell), width.ty())
    }

    /// `c` as a 0/1 value with Python truthiness (non-zero is 1), the mock's
    /// reading of logic-op operands and select conditions. Cells already
    /// known to be 0/1 (`is_boolean`) pass through; others cost one
    /// equality gadget.
    fn booleanize(&mut self, c: &str) -> String {
        if self.is_boolean(c) {
            return c.to_string();
        }
        let expr = format!("if {} == 0 {{ 0 }} else {{ 1 }}", c);
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

    /// `a cmp b` when one side is the literal 0 and the other is
    /// non-negative by construction (`nonneg_by_construction`): `x < 0` is
    /// 0, `x >= 0` is 1, `x > 0` is `x != 0` and `x <= 0` is `x == 0`. Only
    /// facts a lowering guarantees qualify, never an interval.
    fn compare_nonneg_with_zero(&mut self, a: &str, b: &str, cmp: Cmp) -> Option<String> {
        let (x, cmp) = if self.is_zero_literal(b) {
            (a, cmp)
        } else if self.is_zero_literal(a) {
            (b, cmp.mirrored())
        } else {
            return None;
        };
        if !self.nonneg_by_construction.contains(x) {
            return None;
        }
        let x = x.to_string();
        Some(match cmp {
            Cmp::Lt => self.literal_int(0),
            Cmp::Ge => self.literal_int(1),
            Cmp::Gt => self.nonzero_flag(&x, true),
            Cmp::Le => self.nonzero_flag(&x, false),
        })
    }

    /// `a != b` (`nonzero`) or `a == b` when one side is the literal 0 and
    /// the other a scaled bit: the bit or its negation.
    fn scaled_bit_zero_test(&mut self, a: &str, b: &str, nonzero: bool) -> Option<String> {
        let x = if self.s_int_of(b) == Some(0) {
            a
        } else if self.s_int_of(a) == Some(0) {
            b
        } else {
            return None;
        };
        if !self.scaled_bit.contains_key(x) {
            return None;
        }
        let x = x.to_string();
        Some(self.nonzero_flag(&x, nonzero))
    }

    /// `a cmp b` for a comparison whose outcome is asserted (its only uses
    /// are `Assert`s, so every accepted witness makes it true): one range
    /// check of the gap that must be non-negative, `b - a - 1` for `<`,
    /// `b - a` for `<=` (and mirrored for `>`, `>=`), at the width
    /// `comparison_width` picks. It accepts exactly the pairs `qlt_bits`
    /// would accept with this outcome, without the hinted answer and its
    /// select. Against a constant c it proves a range for the other operand
    /// (`x >= c`: x in [c, c + 2^W)), kept in `proven_range`.
    fn asserted_compare(&mut self, a: &str, b: &str, cmp: Cmp) {
        let (low, high, strict) = match cmp {
            Cmp::Lt => (a, b, true),
            Cmp::Le => (a, b, false),
            Cmp::Gt => (b, a, true),
            Cmp::Ge => (b, a, false),
        };
        let width = self.comparison_width(low, high);
        let gap = if strict { format!("{high} - {low} - 1") } else { format!("{high} - {low}") };
        let t = self.fresh();
        let _ = writeln!(self.body, "    let {t}: Field = {gap};");
        let _ = writeln!(self.body, "    {t}.assert_max_bit_size::<{width}>();");
        // low + strict <= high < low + strict + 2^W.
        let span = 1i128 << width;
        let offset = strict as i128;
        if let Some(c) = self.raw_constant(high) {
            self.narrow_proven_range(low, c - offset - span + 1, c - offset);
        }
        if let Some(c) = self.raw_constant(low) {
            self.narrow_proven_range(high, c + offset, c + offset + span - 1);
        }
    }

    fn narrow_proven_range(&mut self, cell: &str, lo: i128, hi: i128) {
        let entry = self.proven_range.entry(cell.to_string()).or_insert((lo, hi));
        *entry = (entry.0.max(lo), entry.1.min(hi));
    }

    /// The outcome of `a cmp b` when one operand is a constant and a check
    /// emitted earlier proves the other's range on one side of it.
    fn decide_from_proven_ranges(&self, a: &str, b: &str, cmp: Cmp) -> Option<bool> {
        let range = |cell: &str| {
            self.raw_constant(cell)
                .map(|c| (c, c))
                .or_else(|| self.proven_range.get(cell).copied())
        };
        let ((al, ah), (bl, bh)) = (range(a)?, range(b)?);
        let decide = |always: bool, never: bool| match (always, never) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        };
        let less = decide(ah < bl, al >= bh);
        let less_or_equal = decide(ah <= bl, al > bh);
        match cmp {
            Cmp::Lt => less,
            Cmp::Ge => less.map(|v| !v),
            Cmp::Le => less_or_equal,
            Cmp::Gt => less_or_equal.map(|v| !v),
        }
    }

    /// Whether `a` or `b` is a zero literal whose other operand already has
    /// a `sign_abs` pair (then the comparison reads its sign for free).
    fn has_cached_sign_for_zero_test(&self, a: &str, b: &str) -> bool {
        (self.is_zero_literal(b) && self.sign_abs_of.contains_key(a))
            || (self.is_zero_literal(a) && self.sign_abs_of.contains_key(b))
    }

    /// `x cmp 0` (or `0 cmp x`) from the sign of x's `sign_abs` pair: `x < 0`
    /// is the sign, `x >= 0` its negation, and with a pair already emitted
    /// also `x > 0` (not negative and not zero) and `x <= 0`. A new pair is
    /// emitted only for `<` and `>=` at the full 64-bit width, where its one
    /// 64-bit check costs what `qlt_bits::<64>` costs, and later sign users
    /// (abs, sign, sqrt, int casts) of x then share it. `sign_abs` accepts
    /// x in [-2^64, 2^64), the domain of `qlt_bits::<64>(x, 0)`.
    fn compare_with_zero_by_sign(&mut self, a: &str, b: &str, cmp: Cmp) -> Option<String> {
        let (x, zero, cmp) = if self.is_zero_literal(b) {
            (a, b, cmp)
        } else if self.is_zero_literal(a) {
            (b, a, cmp.mirrored())
        } else {
            return None;
        };
        let x = x.to_string();
        let cached = self.sign_abs_of.contains_key(&x);
        let cheap = matches!(cmp, Cmp::Lt | Cmp::Ge) && self.comparison_width(&x, zero) == 64;
        if !cached && !cheap {
            return None;
        }
        let (negative, mag) = self.sign_abs(&x);
        let flag = match cmp {
            Cmp::Lt => format!("{negative} as Field"),
            Cmp::Ge => format!("(!{negative}) as Field"),
            Cmp::Gt => format!("((!{negative}) & ({mag} != 0)) as Field"),
            Cmp::Le => format!("({negative} | ({mag} == 0)) as Field"),
        };
        Some(self.bind_flag(&flag))
    }

    /// The literal integer 0 or float 0.0 (both the field element 0).
    fn is_zero_literal(&self, cell: &str) -> bool {
        self.s_int_of(cell) == Some(0) || self.s_float_of(cell) == Some(0.0)
    }

    /// `(x < 0, |x|)` through the library's `sign_abs` (one 64-bit check;
    /// x must lie in [-2^64, 2^64)), emitted once per cell and shared by
    /// comparisons with 0, abs, sign, sqrt and int casts of x.
    fn sign_abs(&mut self, x: &str) -> (String, String) {
        if let Some(pair) = self.sign_abs_of.get(x) {
            return pair.clone();
        }
        self.use_lib("sign_abs");
        let (negative, mag) = (self.fresh(), self.fresh());
        let _ = writeln!(self.body, "    let ({negative}, {mag}) = sign_abs({x});");
        self.sign_abs_of
            .insert(x.to_string(), (negative.clone(), mag.clone()));
        (negative, mag)
    }

    /// `x != 0` (`nonzero`) or `x == 0` as a 0/1 cell; for a scaled bit
    /// `2^k * bit` it is the bit (or its negation).
    fn nonzero_flag(&mut self, x: &str, nonzero: bool) -> String {
        if let Some(bit) = self.scaled_bit.get(x).cloned() {
            let expr = if nonzero { format!("{bit} as Field") } else { format!("(!{bit}) as Field") };
            return self.bind_flag(&expr);
        }
        let (t, f) = if nonzero { (0, 1) } else { (1, 0) };
        let xf = x;
        self.bind_flag(&format!("if {xf} == 0 {{ {t} }} else {{ {f} }}"))
    }

    /// The `8 * nb` low bits of the two's complement of `x` (library
    /// `low_bits`, which also proves x lies in the signed 64-bit window),
    /// emitted once per cell and widened when a later mask needs more.
    fn low_bits(&mut self, x: &str, nb: u32) -> String {
        if let Some((have, name)) = self.low_bits_of.get(x) {
            if *have >= nb {
                return name.clone();
            }
        }
        self.use_lib("low_bits");
        let name = self.fresh();
        let xf = x;
        let _ = writeln!(self.body, "    let {}: [bool; {}] = low_bits::<{}>({});", name, nb * 8, nb, xf);
        self.low_bits_of.insert(x.to_string(), (nb, name.clone()));
        name
    }

    /// `x & mask` for a constant mask, as the mock's i64 `&`. A non-negative
    /// mask below 2^56 selects bits of x's two's complement: the result is
    /// the sum of the selected bits times their weights (non-negative by
    /// construction; a one-bit mask gives a scaled bit). A negative mask
    /// uses `x & m = x - (x & !m)` with `!m = -m - 1 >= 0`. Wider masks go
    /// through the library's `bitand`.
    fn and_with_mask(&mut self, x: &str, mask: i64) -> String {
        if mask == 0 {
            return self.literal_int(0);
        }
        if mask == -1 {
            return x.to_string();
        }
        if mask < 0 {
            let complement = self.and_with_mask(x, !mask);
            let xf = x;
            return self.bind(&format!("{} - {}", xf, &complement));
        }
        let width = 64 - mask.leading_zeros();
        if width > 56 {
            self.use_lib("bitand");
            return self.bind(&format!("bitand({}, {})", x, mask));
        }
        let bits = self.low_bits(x, width.div_ceil(8));
        let terms: Vec<String> = (0..width)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| format!("({bits}[{i}] as Field) * {}", 1u64 << i))
            .collect();
        let out = self.bind_sum(&terms);
        self.nonneg_by_construction.insert(out.clone());
        if mask.count_ones() == 1 {
            let k = mask.trailing_zeros();
            self.scaled_bit.insert(out.clone(), format!("{bits}[{k}]"));
        }
        out
    }

    /// The constant operand of a binary op and the other operand, if exactly
    /// one operand is a compile-time integer within i64.
    fn constant_and_other(&self, a: &str, b: &str) -> Option<(i64, String)> {
        match (self.s_int_of(a), self.s_int_of(b)) {
            (None, Some(m)) => i64::try_from(m).ok().map(|m| (m, a.to_string())),
            (Some(m), None) => i64::try_from(m).ok().map(|m| (m, b.to_string())),
            _ => None,
        }
    }

    /// Both operands as compile-time i64 constants.
    fn constant_pair(&self, a: &str, b: &str) -> Option<(i64, i64)> {
        let x = i64::try_from(self.s_int_of(a)?).ok()?;
        let y = i64::try_from(self.s_int_of(b)?).ok()?;
        Some((x, y))
    }

    /// `a | b` (`times == 1`) or `a ^ b` (`times == 2`) from the two's
    /// complement identity `a + b - times * (a & b)`.
    fn or_xor_from_and(&mut self, a: &str, b: &str, and: &str, times: u32) -> String {
        let (af, bf, nf) = (a, b, and);
        if times == 1 {
            self.bind(&format!("{af} + {bf} - {nf}"))
        } else {
            self.bind(&format!("{af} + {bf} - 2 * {nf}"))
        }
    }

    /// The Q32 product `floor(a * b / 2^32)` as an exact Field expression
    /// when one factor is an integral float `n * 2^32` (`integral_float`):
    /// `floor(n * 2^32 * x / 2^32) = n * x`, the mock's value bit for bit,
    /// with no rescale and so no range check. `None` otherwise.
    fn exact_product(&self, a: &str, b: &str) -> Option<String> {
        if let Some(n) = self.integral_float.get(a) {
            return Some(format!("{} * {}", n, b));
        }
        let n = self.integral_float.get(b)?;
        Some(format!("{} * {}", a, n))
    }

    /// Q32 `a / c` for a constant raw divisor c: the Euclidean quotient of
    /// a * 2^32 by c (`kernel::fp_div_prescale`), i.e. floor(a * 2^32 / |c|)
    /// negated for c < 0, and 0 for c = 0. An integral float c = n * 2^32
    /// cancels the scale: floor(a / |n|), exactly the same quotient.
    fn div_f_by_constant(&mut self, a: &str, c: i128) -> String {
        if c == 0 {
            return self.literal_int(0);
        }
        let q_scale = 1i128 << Q_BITS;
        let (numerator, divisor) = if c % q_scale == 0 {
            (a.to_string(), (c / q_scale).abs())
        } else {
            (format!("{a} * {Q_U64}"), c.abs())
        };
        let (q, _r) = self.divide_by_positive_constant(&numerator, divisor);
        if c < 0 {
            self.bind(&format!("0 - {q}"))
        } else {
            q
        }
    }

    /// Python `(a // b, a % b)` of two compile-time integers, with the
    /// mock's zero-divisor convention (`kernel::div_rem_or_zero`).
    fn constant_floor_divmod(&self, a: &str, b: &str) -> Option<(i64, i64)> {
        let (x, y) = self.constant_pair(a, b)?;
        if y == 0 {
            return Some((0, 0));
        }
        let (q, r) = (x.checked_div(y)?, x % y);
        Some(if r != 0 && (r < 0) != (y < 0) { (q - 1, r + y) } else { (q, r) })
    }

    /// Python `(a // b, a % b)` of the raw values as two cells, shared
    /// between the integer and float floor division and remainder on the
    /// same operands. A positive compile-time divisor uses
    /// `divide_by_positive_constant`; any other divisor the general
    /// `floor_divmod` (one sign projection of b; a zero divisor gives
    /// (0, 0), as the mock). Both pin the unique quotient and remainder.
    fn floor_divmod_cells(&mut self, a: &str, b: &str) -> (String, String) {
        if let Some(d) = self.raw_constant(b).filter(|d| *d > 0) {
            return self.divide_by_positive_constant(a, d);
        }
        self.use_lib("floor_divmod");
        self.divmod_call(format!("floor_divmod({a}, {b})"))
    }

    /// The raw field value of a compile-time constant as a signed integer:
    /// an integer literal, or the Q32 encoding of a float literal.
    fn raw_constant(&self, cell: &str) -> Option<i128> {
        match self.literal_vals.get(cell)? {
            ConcreteValue::Int(i) => Some(*i),
            ConcreteValue::Bool(b) => Some(*b as i128),
            ConcreteValue::Float(f) => Some(crate::prove::field::quantize(*f, Q_BITS)),
        }
    }

    /// `(n // d, n % d)` (floor, 0 <= r < d) for a numerator expression `n`
    /// and a constant `1 <= d < 2^120`. A power of two 2^k is one k-bit
    /// check of the remainder (`divmod_by_pow2`); any other d two checks of
    /// `r` and `d - 1 - r` (`divmod_by_positive`), at a width rounded up to
    /// a multiple of bb's 14-bit limb (the same limbs, no extra lookup
    /// table; any width of at least bit_length(d - 1) is sound). The
    /// quotient is checked into the signed 64-bit window either way.
    fn divide_by_positive_constant(&mut self, n: &str, d: i128) -> (String, String) {
        debug_assert!(d >= 1 && d < 1i128 << 120);
        if d == 1 {
            return (n.to_string(), self.literal_int(0));
        }
        let call = if d & (d - 1) == 0 {
            self.use_lib("divmod_by_pow2");
            format!("divmod_by_pow2::<{}>({n})", d.trailing_zeros())
        } else {
            let bits = 128 - ((d - 1) as u128).leading_zeros();
            let width = if bits <= 8 { bits } else { bits.div_ceil(14) * 14 }.min(120);
            self.use_lib("divmod_by_positive");
            format!("divmod_by_positive::<{width}>({n}, {d})")
        };
        self.divmod_call(call)
    }

    /// Bind `let (q, r) = <call>;` once per call text.
    fn divmod_call(&mut self, call: String) -> (String, String) {
        if let Some(pair) = self.divmod_cache.get(&call) {
            return pair.clone();
        }
        let (q, r) = (self.fresh(), self.fresh());
        let _ = writeln!(self.body, "    let ({}, {}) = {};", q, r, call);
        self.divmod_cache.insert(call, (q.clone(), r.clone()));
        (q, r)
    }

    /// Emit `0 <= cell < 2^bits` as a standalone range check on the Field
    /// value (never removed by `dce_body`, which only drops `let` lines).
    fn emit_range_check(&mut self, cell: &str, bits: u32) {
        let target = if is_identifier(cell) {
            cell.to_string()
        } else {
            let t = self.fresh();
            let _ = writeln!(self.body, "    let {}: Field = {};", t, cell);
            t
        };
        let _ = writeln!(self.body, "    {}.assert_max_bit_size::<{}>();", target, bits);
    }

    /// Emit `0 <= cell < hi` for a compile-time bound `hi`: two range checks
    /// of `bits = bit_length(hi - 1)` bits on `cell` and on `hi - 1 - cell`.
    /// A negative cell or one above `hi - 1` turns one of them into a field
    /// element of about 254 bits, which fails. For `hi = 2^bits` the first
    /// check alone is exact, so the second is left out. `hi <= 0` admits
    /// nothing.
    fn assert_index_in_range(&mut self, cell: &str, hi: i128) {
        if hi <= 0 {
            let _ = writeln!(self.body, "    assert(1 == 0);");
            return;
        }
        let bits = (128 - ((hi - 1) as u128).leading_zeros()).max(1);
        let t = self.fresh();
        let _ = writeln!(self.body, "    let {}: Field = {};", t, cell);
        let _ = writeln!(self.body, "    {}.assert_max_bit_size::<{}>();", t, bits);
        if hi == 1i128 << bits {
            return;
        }
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
                    }
                }
            }
            break;
        }
        let ind = if base != root || val != v {
            self.indicator(&base, val)
        } else {
            let rf = root;
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
        let sum = self.bind_sum(&inds);
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

    /// Whether a single-root dynamic read whose one-hot gather costs about
    /// `one_hot_gates` is cheaper through a native array (`native_read`).
    /// Measured costs (bb gates): about 10 per native read (the ROM read and
    /// `array_index`'s two range checks) plus 3 per element to build the
    /// epoch's array, which the epoch's dynamic reads share. Two-root reads
    /// always gather: their stage-1 planes are shared across reads, which
    /// measured cheaper than native reads on every case that has them.
    fn prefer_native_read(&self, segment_id: u32, one_hot_gates: usize) -> bool {
        const NATIVE_READ_GATES: usize = 10;
        const ARRAY_ELEMENT_GATES: usize = 3;
        let epoch = self.write_version.get(&segment_id).copied().unwrap_or(0);
        let len = self.memories.get(&segment_id).map_or(0, |m| m.len());
        let build = if self.segment_arrays.contains_key(&(segment_id, epoch)) {
            0
        } else {
            (ARRAY_ELEMENT_GATES * len).div_ceil(self.epoch_dynamic_reads(segment_id))
        };
        len > 0 && len < (1 << 27) && build + NATIVE_READ_GATES < one_hot_gates
    }

    /// Dynamic reads of the segment's current write epoch (at least 1).
    fn epoch_dynamic_reads(&self, segment_id: u32) -> usize {
        let epoch = self.write_version.get(&segment_id).copied().unwrap_or(0);
        self.dynamic_reads_per_epoch
            .get(&(segment_id, epoch))
            .copied()
            .unwrap_or(1)
            .max(1)
    }

    /// `mem[addr]` through a native Noir array (a ROM block in bb). The
    /// library's `array_index` proves `0 <= addr < len` on its own (an
    /// out-of-range address fails the proof, as the mock rejects it); it
    /// does not rely on bb's memory check, which the pinned bb version does
    /// not enforce for index p - 1 (see `array_index`).
    fn native_read(&mut self, segment_id: u32, addr: &str) -> String {
        let array = self.segment_array(segment_id);
        let len = self.memories[&segment_id].len() as u64;
        let index = self.array_index(addr, len);
        let out = self.bind(&format!("{array}[{index}]"));
        self.apply_segment_iv(segment_id, &out);
        out
    }

    /// `addr` as a checked `u32` index into an array of `len >= 1` elements
    /// (library `array_index`), once per (addr, len).
    fn array_index(&mut self, addr: &str, len: u64) -> String {
        self.use_lib("array_index");
        let af = addr;
        let last = len - 1;
        let width = if last < 1 << 7 { 8 } else { 28 };
        self.bind_typed(&format!("array_index::<{width}>({af}, {last})"), "u32")
    }

    /// The segment's current slots as a Noir array, built once per write
    /// epoch.
    fn segment_array(&mut self, segment_id: u32) -> String {
        let epoch = self.write_version.get(&segment_id).copied().unwrap_or(0);
        if let Some(name) = self.segment_arrays.get(&(segment_id, epoch)) {
            return name.clone();
        }
        let cells = self.memories[&segment_id].clone();
        let name = self.fresh();
        let _ = writeln!(
            self.body,
            "    let {}: [Field; {}] = [{}];",
            name,
            cells.len(),
            cells.join(", ")
        );
        self.segment_arrays.insert((segment_id, epoch), name.clone());
        name
    }

    /// The `width` low bits of an index known to satisfy `0 <= idx < hi`
    /// (with `hi <= 2^width`), emitted once per (idx, width) together with
    /// the bound: `to_le_bits` proves `idx < 2^width`, and a `width`-bit
    /// check of `hi - 1 - idx` the rest. The pair is recorded as asserted,
    /// so a later `AssertU128Lt(idx, hi)` is not repeated.
    fn index_bits(&mut self, idx: &str, hi: i64, width: u32) -> String {
        if let Some(name) = self.index_bits_of.get(&(idx.to_string(), width)) {
            return name.clone();
        }
        let name = self.fresh();
        let idx_field = idx;
        let _ = writeln!(self.body, "    let {}: [bool; {}] = {}.to_le_bits();", name, width, idx_field);
        if (hi as u128) < (1u128 << width) {
            let gap = self.fresh();
            let _ = writeln!(self.body, "    let {}: Field = {} - {};", gap, hi - 1, idx_field);
            let _ = writeln!(self.body, "    {}.assert_max_bit_size::<{}>();", gap, width);
        }
        let hi_cell = self.literal_int(hi);
        self.emitted_u128_asserts.insert((idx.to_string(), hi_cell));
        self.index_bits_of.insert((idx.to_string(), width), name.clone());
        name
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
            parts.push(format!("{} * {}", &ind, &cell));
        }
        let plane = if parts.is_empty() {
            init.to_string()
        } else {
            self.bind_sum(&parts)
        };
        self.plane_cache.insert(key, plane.clone());
        plane
    }

    pub fn new(
        inputs: Vec<EmittedInput>,
        stmt_bounds: super::bound_analysis::BoundMap,
        provenance: &'a crate::provenance::ProvenanceTable,
    ) -> NoirEmitter<'a> {
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
            wrapped_index_ranges: HashMap::new(),
            asserted_comparisons: HashSet::new(),
            current_stmt: None,
            proven_range: HashMap::new(),
            dynamic_reads_per_epoch: HashMap::new(),
            segment_arrays: HashMap::new(),
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
            literal_vals: HashMap::new(),
            integral_float: HashMap::new(),
            nonneg_by_construction: HashSet::new(),
            scaled_bit: HashMap::new(),
            low_bits_of: HashMap::new(),
            index_bits_of: HashMap::new(),
            sign_abs_of: HashMap::new(),
            stmt_bounds,
            current_stmt_bound: None,
        }
    }

    // ── Bounds ────────────────────────────────────────────────────────

    fn bound_of(&self, cell: &str) -> Option<u64> {
        self.cell_bound.get(cell).copied()
    }

    fn set_bound(&mut self, cell: &str, bound: u64) {
        self.cell_bound.insert(cell.to_string(), bound);
    }

    /// Apply the IR-analysis bound to a statement's result cell, so that
    /// later interval queries (`ival`) see it.
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
        // bind_typed).
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

    /// A cell holding the integer constant `value`, registered in
    /// `literal_vals`.
    fn literal_int(&mut self, value: i64) -> String {
        let expr = signed_to_field_decimal(value as i128);
        self.bind_val(&expr, ConcreteValue::Int(value as i128))
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
        // Stamp the IR-analysis bound on the statement RESULT only, for
        // later interval queries. Doing it here (rather than inside `bind()`) keeps
        // the bound off shared intermediate cells such as cached
        // indicators, which would otherwise inherit a statement-result
        // bound that does not describe them and poison later interval
        // queries.
        self.annotate_with_analysis_bound(cell);
    }

    fn on_stmt_begin(&mut self, stmt_id: u32) {
        // Stash the IR-analysis bound for the result of this statement.
        self.current_stmt_bound = self.stmt_bounds.get(&stmt_id).copied();
        self.current_stmt = Some(stmt_id);
    }

    fn constant_int(&mut self, value: i64) -> Result<Self::CellRef, ProvingError> {
        Ok(self.literal_int(value))
    }

    fn constant_float(&mut self, value: f64) -> Result<Self::CellRef, ProvingError> {
        let expr = quantize_to_field_string(value);
        let v = self.bind_val(&expr, ConcreteValue::Float(value));
        // An integral constant inside the int64 window is exactly n * 2^32.
        if value.fract() == 0.0 && value.abs() < (1u64 << 31) as f64 {
            self.integral_float
                .insert(v.clone(), signed_to_field_decimal(value as i128));
        }
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
        let out = self.integer_arith(a, b, '+');
        if let Some(c) = self.s_int_of(b) {
            self.add_const_of.insert(out.clone(), (a.clone(), c as i64));
        } else if let Some(c) = self.s_int_of(a) {
            self.add_const_of.insert(out.clone(), (b.clone(), c as i64));
        }
        return Ok(out);
    }

    fn sub_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.integer_arith(a, b, '-'))
    }

    fn mul_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.integer_arith(a, b, '*'))
    }

    fn div_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // `DivI` is field division a * b^-1 (it equals integer division when b
        // divides a), and 0 for b == 0, as the mock's `div_i`.
        self.use_lib("field_div_or_zero");
        let (af, bf) = (a, b);
        Ok(self.bind(&format!("field_div_or_zero({}, {})", af, bf)))
    }
    fn floor_div_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some((q, _r)) = self.constant_floor_divmod(a, b) {
            return Ok(self.literal_int(q));
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
        if let Some((q, _r)) = self.constant_floor_divmod(a, b) {
            return Ok(self.literal_int(q));
        }
        // The producer (e.g. reshape `-1` dim inference) promises a, b, the
        // quotient and the remainder are non-negative `divisor_bits`-bit
        // values. `q_floor_div_bounded` checks exactly that (so a false
        // promise fails the proof) and pins the unique quotient.
        let bits = divisor_bits.max(1);
        self.use_lib("q_floor_div_bounded");
        let (af, bf) = (a, b);
        Ok(self.bind(&format!("q_floor_div_bounded::<{}>({}, {})", bits, af, bf)))
    }
    fn mod_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // Python remainder (the divisor's sign; 0 for a zero divisor).
        if let Some((_q, r)) = self.constant_floor_divmod(a, b) {
            return Ok(self.literal_int(r));
        }
        let (_q, r) = self.floor_divmod_cells(a, b);
        Ok(r)
    }
    fn pow_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "pow_i")
    }
    fn abs_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let (_negative, mag) = self.sign_abs(a);
        Ok(mag)
    }
    fn sign_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let (negative, mag) = self.sign_abs(a);
        self.use_lib("sign_of");
        Ok(self.bind(&format!("sign_of({negative}, {mag})")))
    }
    fn inv_i(&mut self, _a: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "inv_i")
    }

    // Bitwise operators follow the mock's i64 two's complement operators.
    // A constant mask reads bits of the other operand (`and_with_mask`);
    // `|` and `^` reduce to `&` by `a | b = a + b - (a & b)` and
    // `a ^ b = a + b - 2 (a & b)`; two dynamic operands use the library's
    // `bitand` (native u64 AND on the window-checked operands).
    fn bit_and_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some((x, y)) = self.constant_pair(a, b) {
            return Ok(self.literal_int(x & y));
        }
        if let Some((mask, x)) = self.constant_and_other(a, b) {
            return Ok(self.and_with_mask(&x, mask));
        }
        self.use_lib("bitand");
        Ok(self.bind(&format!("bitand({}, {})", a, b)))
    }
    fn bit_or_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some((x, y)) = self.constant_pair(a, b) {
            return Ok(self.literal_int(x | y));
        }
        let and = self.bit_and_i(a, b)?;
        Ok(self.or_xor_from_and(a, b, &and, 1))
    }
    fn bit_xor_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some((x, y)) = self.constant_pair(a, b) {
            return Ok(self.literal_int(x ^ y));
        }
        let and = self.bit_and_i(a, b)?;
        Ok(self.or_xor_from_and(a, b, &and, 2))
    }
    fn shl_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "shl_i")
    }
    fn shr_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "shr_i")
    }
    fn bit_not_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Two's complement: !a = -a - 1.
        if let Some(x) = self.s_int_of(a).and_then(|x| i64::try_from(x).ok()) {
            return Ok(self.literal_int(!x));
        }
        Ok(self.bind(&format!("0 - {} - 1", a)))
    }

    fn add_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qadd");
        let v = self.bind(&format!("qadd({}, {})", a, b));
        Ok(v)
    }
    fn sub_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsub");
        let v = self.bind(&format!("qsub({}, {})", a, b));
        Ok(v)
    }
    fn mul_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(exact) = self.exact_product(a, b) {
            return Ok(self.bind(&exact));
        }
        self.use_lib("qmul");
        let v = self.bind(&format!("qmul({}, {})", a, b));
        Ok(v)
    }
    fn product_sum_f(&mut self, pairs: &[(CellRef, CellRef)]) -> Result<CellRef, ProvingError> {
        // floor((E * 2^32 + R) / 2^32) = E + floor(R / 2^32) for an integer
        // E: the exact products leave the rescale, which then covers only
        // the other pairs (none at all when every pair is exact).
        let mut exact: Vec<String> = Vec::new();
        let mut raw: Vec<String> = Vec::new();
        for (a, b) in pairs {
            match self.exact_product(a, b) {
                Some(term) => exact.push(format!("({term})")),
                None => raw.push(format!("({} * {})", a, b)),
            }
        }
        if !raw.is_empty() {
            self.use_lib("qscale_down_raw");
            let rescaled = self.bind(&format!("qscale_down_raw({})", raw.join(" + ")));
            exact.push(rescaled);
        }
        if let [single] = exact.as_slice() {
            if is_identifier(single) {
                return Ok(single.clone());
            }
        }
        Ok(self.bind_sum(&exact))
    }
    fn div_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(c) = self.raw_constant(b) {
            if c.unsigned_abs() < 1u128 << 100 {
                return Ok(self.div_f_by_constant(a, c));
            }
        }
        self.use_lib("qdiv");
        let v = self.bind(&format!("qdiv({}, {})", a, b));
        Ok(v)
    }
    fn floor_div_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // floor(a / b) as a float: the Python floor quotient of the raw Q32
        // integers, re-scaled (kernel::fp_float_floor_div).
        let (q, _r) = self.floor_divmod_cells(a, b);
        Ok(self.bind(&format!("{} * {}", &q, Q_U64)))
    }
    fn mod_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // a % b with the divisor's sign: the floor remainder of the raw Q32
        // integers is already Q-scaled (kernel::fp_float_mod).
        let (_q, r) = self.floor_divmod_cells(a, b);
        Ok(r)
    }
    fn pow_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qpow");
        let v = self.bind(&format!("qpow({}, {})", a, b));
        Ok(v)
    }
    fn pow_f_int(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // Float base ^ integer exponent by square-and-multiply. The exponent
        // must be a non-negative value below 2^32: the range-checked
        // conversion fails the proof otherwise (a truncating cast would have
        // used the low bits of a negative exponent's field encoding).
        self.use_lib("qpow_int_q32");
        let e = self.to_uint(b, UintWidth::U32);
        let af = a;
        Ok(self.bind(&format!("qpow_int_q32({}, {} as u64)", af, e)))
    }
    fn abs_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let (_negative, v) = self.sign_abs(a);
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
        // NumPy's float sign: -1.0, 0.0 or 1.0 (Q32).
        let (negative, mag) = self.sign_abs(a);
        self.use_lib("sign_of");
        Ok(self.bind(&format!("sign_of({negative}, {mag}) * {Q_U64}")))
    }

    fn eq_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.scaled_bit_zero_test(a, b, false) {
            return Ok(v);
        }
        // Normalized-index dual equality: for norm = select(lt(x,0), x+N, x),
        // eq(norm, j) == (x == j) | (x == j - N), one zero test of
        // (x - j)(x - j + N), instead of the comparison the normalization
        // otherwise drags in. The identity is
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
                    // j - n. x is j or j - n exactly when
                    // (x - j) * (x - j + n) is zero (a field has no zero
                    // divisors): one product and one zero test.
                    let v = self.bind_flag(&format!(
                        "if ({x} - {j}) * ({x} + {}) == 0 {{ 1 }} else {{ 0 }}",
                        n - j
                    ));
                    return Ok(v);
                }
                // j out of [0, N): peephole identity does not hold, fall
                // through to the generic equality path below.
            }
        }
        let v = self.bind_flag(&format!(
            "if {} == {} {{ 1 }} else {{ 0 }}",
            a,
            b
        ));
        Ok(v)
    }
    fn ne_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.scaled_bit_zero_test(a, b, true) {
            return Ok(v);
        }
        let v = self.bind_flag(&format!(
            "if {} == {} {{ 0 }} else {{ 1 }}",
            a,
            b
        ));
        Ok(v)
    }
    fn lt_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        // min/max select recognition happens at IR level (the provenance
        // table's SelectI deposit); only the lt-zero peephole stays here.
        let out = self.compare(a, b, Cmp::Lt);
        if self.s_int_of(b) == Some(0) {
            self.lt_zero_of.insert(out.clone(), a.clone());
        }
        Ok(out)
    }

    // The IR promises non-negative operands for the bounded comparisons;
    // the lowering does not rely on it (the mock compares them signed).
    fn lt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Lt))
    }

    fn gt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Gt))
    }

    fn index_in_box(
        &mut self,
        triples: &[(CellRef, CellRef, CellRef)],
    ) -> Result<CellRef, ProvingError> {
        // The product of the per-axis flags `lo_k <= idx_k` and
        // `idx_k < hi_k` (the mock's default lowering: two comparisons and a
        // logical and per axis).
        if triples.is_empty() {
            return self.constant_int(1);
        }
        let mut parts: Vec<String> = Vec::with_capacity(triples.len() * 2);
        for (idx, lo, hi) in triples {
            let above = self.compare(idx, lo, Cmp::Ge);
            let below = self.compare(idx, hi, Cmp::Lt);
            parts.push(above);
            parts.push(below);
        }
        Ok(self.bind_flag(&parts.join(" * ")))
    }
    fn lte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Le))
    }
    fn gt_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Gt))
    }
    fn gte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Ge))
    }

    fn eq_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (af, bf) = (a, b);
        Ok(self.bind_flag(&format!("if {} == {} {{ 1 }} else {{ 0 }}", af, bf)))
    }
    fn ne_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (af, bf) = (a, b);
        Ok(self.bind_flag(&format!("if {} == {} {{ 0 }} else {{ 1 }}", af, bf)))
    }
    // Float comparisons are signed comparisons of the Q32 integers (the
    // mock's `lt_f` is its `lt_i`).
    fn lt_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Lt))
    }
    fn lte_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Le))
    }
    fn gt_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Gt))
    }
    fn gte_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.compare(a, b, Cmp::Ge))
    }

    fn sin_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsin");
        let v = self.bind(&format!("qsin({})", a));
        Ok(v)
    }
    fn sinh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsinh");
        let v = self.bind(&format!("qsinh({})", a));
        Ok(v)
    }
    fn cos_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qcos");
        let v = self.bind(&format!("qcos({})", a));
        Ok(v)
    }
    fn cosh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qcosh");
        let v = self.bind(&format!("qcosh({})", a));
        Ok(v)
    }
    fn tan_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qtan");
        let v = self.bind(&format!("qtan({})", a));
        Ok(v)
    }
    fn tanh_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qtanh");
        let v = self.bind(&format!("qtanh({})", a));
        Ok(v)
    }
    fn sqrt_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // (No literal propagation: the circuit's floor root of a constant
        // need not equal the f64 sqrt the literal store would record.)
        let (negative, mag) = self.sign_abs(a);
        self.use_lib("qsqrt_of");
        Ok(self.bind(&format!("qsqrt_of({negative}, {mag})")))
    }
    fn exp_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qexp");
        let v = self.bind(&format!("qexp({})", a));
        Ok(v)
    }
    fn log_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qlog");
        let v = self.bind(&format!("qlog({})", a));
        Ok(v)
    }
    fn arcsin_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarcsin");
        let v = self.bind(&format!("qarcsin({})", a));
        Ok(v)
    }
    fn arccos_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarccos");
        let v = self.bind(&format!("qarccos({})", a));
        Ok(v)
    }
    fn arctan2_f(&mut self, y: &CellRef, x: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qarctan2");
        let v = self.bind(&format!(
            "qarctan2({}, {})",
            y,
            x
        ));
        Ok(v)
    }

    fn logical_and(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (a, b) = (self.booleanize(a), self.booleanize(b));
        let (af, bf) = (&a, &b);
        Ok(self.bind_flag(&format!("{} * {}", af, bf)))
    }
    fn logical_or(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (a, b) = (self.booleanize(a), self.booleanize(b));
        let (af, bf) = (&a, &b);
        Ok(self.bind_flag(&format!("{} + {} - {} * {}", af, bf, af, bf)))
    }
    fn logical_not(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let a = self.booleanize(a);
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
        let expr = if is_zero_int || is_zero_flt {
            format!("{} * {}", c, if_true)
        } else {
            format!("{} * {} + (1 - {}) * {}", c, if_true, c, if_false)
        };
        let v = self.bind(&expr);
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
        let (negative, mag) = self.sign_abs(a);
        self.use_lib("q_div_q_of");
        Ok(self.bind(&format!("q_div_q_of({negative}, {mag})")))
    }
    fn float_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Integer-to-float in Q32 means *= 2^32.
        let v = self.bind(&format!("{} * {}", a, Q_U64));
        self.integral_float.insert(v.clone(), a.clone());
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
        // Exposed values become the `pub` return array of `main`.
        self.exposes.push(a.clone());
        Ok(())
    }

    fn assert_true(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        // Python truthiness, as the mock: any non-zero value passes. A
        // literal true (an asserted comparison checked where it was
        // lowered, see `asserted_compare`) needs nothing.
        if matches!(self.s_int_of(a), Some(v) if v != 0) {
            return Ok(());
        }
        if self.is_boolean(a) {
            let _ = writeln!(self.body, "    assert({} == 1);", a);
        } else {
            let _ = writeln!(self.body, "    assert({} != 0);", a);
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
    /// `0 <= idx < hi` and bit `bit` of idx, from one decomposition of idx
    /// into `width` bits (shared by every bit of the same index) plus, when
    /// `hi` is not a power of two, a `width`-bit check of `hi - 1 - idx`.
    /// The decomposition itself proves `0 <= idx < 2^width` (Noir's
    /// `to_le_bits` fails for a larger value), so no other check is needed.
    fn assert_u128_lt_bit(
        &mut self,
        idx: &CellRef,
        hi: u64,
        bit: u32,
        bits: u32,
    ) -> Result<CellRef, ProvingError> {
        if let Some(i) = self.s_int_of(idx) {
            if i < 0 || i as u128 >= hi as u128 {
                // The mock rejects an out-of-range constant index.
                let _ = writeln!(self.body, "    assert(1 == 0);");
                return Ok(self.literal_int(0));
            }
            return Ok(self.literal_int((((i as u128) >> bit) & 1) as i64));
        }
        let hi_i64 = i64::try_from(hi).map_err(|_| {
            ProvingError::synthesis(format!("AssertU128LtBit bound {} exceeds i64", hi))
        })?;
        if hi == 0 {
            let _ = writeln!(self.body, "    assert(1 == 0);");
            return Ok(self.literal_int(0));
        }
        let width = bits.max(64 - (hi - 1).leading_zeros()).max(1);
        let decomposition = self.index_bits(idx, hi_i64, width);
        if bit >= width {
            return Ok(self.literal_int(0));
        }
        Ok(self.bind_flag(&format!("{decomposition}[{bit}] as Field")))
    }

    /// `a >= 0` over the signed-i64 domain: one 63-bit range check on the
    /// Field value (as halo2-base). Self-standing for the same reason as
    /// `assert_u128_lt`: the builder records `a >= 0` from this assert.
    fn assert_non_neg_i(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        self.emit_range_check(a, 63);
        Ok(())
    }

    /// `a` in {0, 1}, checked directly (no bound or interval consulted).
    fn assert_bit(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        let af = a;
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
        let vf = value;
        for &v in &vs {
            let slot = (v * scale + offset) as usize;
            let ind = self.indicator(&root, v);
            let old_v = self.memories.get(&segment_id).unwrap()[slot].clone();
            let of = &old_v;
            let indf = &ind;
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
                            &ind,
                            &plane
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
            // One-hot cost of this read: one product per candidate slot, and
            // one equality per indicator of the root not built yet, which
            // the epoch's other dynamic reads may share (counted as shared
            // evenly; reads on another root do not share them, so this
            // leans toward the one-hot gather).
            let new_indicators = vs
                .iter()
                .filter(|&&v| !self.ind_cache.contains_key(&(root.clone(), v)))
                .count();
            let indicator_share =
                (INDICATOR_GATES * new_indicators).div_ceil(self.epoch_dynamic_reads(segment_id));
            if self.prefer_native_read(segment_id, indicator_share + vs.len()) {
                return Ok(self.native_read(segment_id, addr));
            }
            self.assert_one_hot(&root.clone(), &vs);
            let mut parts: Vec<String> = Vec::with_capacity(vs.len());
            for &v in &vs {
                let slot = (v * scale + offset) as usize;
                let cell = self.memories.get(&segment_id).unwrap()[slot].clone();
                let ind = self.indicator(&root, v);
                parts.push(format!(
                    "{} * {}",
                    &ind,
                    &cell
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
        NoirEmitter::new(Vec::new(), HashMap::new(), prov)
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
        // Expect the dual-equality peephole: (x - 3) * (x + 2) == 0.
        assert!(
            em.body.contains("(x - 3) * (x + 2) == 0"),
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
        // The spurious second factor would be (x + 0), the first (x - 5).
        // The dual form must not appear.
        assert!(
            !em.body.contains(") * ("),
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
            !em.body.contains(") * ("),
            "j<0 must NOT emit the dual-equality peephole, body was:\n{}",
            em.body
        );
    }

    // Comparisons size their gap check by the operands' intervals: the
    // interval only picks the width of `qlt_bits` (a pair outside it fails
    // the proof), never a truncating cast.
    // Non-negative intervals compare through shared checked conversions.
    #[test]
    fn non_negative_intervals_compare_unsigned() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        em.interval_of.insert("x".to_string(), (0, 200));
        em.interval_of.insert("y".to_string(), (0, 200));
        em.lt_i(&"x".to_string(), &"y".to_string()).unwrap();
        assert!(em.body.contains("checked_u8(x)") && em.body.contains("checked_u8(y)"), "{}", em.body);
        assert!(!em.body.contains("x as u8"), "{}", em.body);
    }

    #[test]
    fn comparison_width_follows_the_intervals() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        for (name, iv) in [("x", (-1, 200)), ("y", (-50, 200)), ("z", (0, 1 << 20)), ("w", (0, 1 << 40))] {
            em.interval_of.insert(name.to_string(), iv);
        }
        let s = |n: &str| n.to_string();
        em.lt_i(&s("x"), &s("y")).unwrap();
        em.lt_i(&s("x"), &s("z")).unwrap();
        em.lt_i(&s("x"), &s("w")).unwrap();
        em.lt_f(&s("x"), &s("u")).unwrap();
        assert!(em.body.contains("qlt_bits::<8>(x, y)"), "{}", em.body);
        assert!(em.body.contains("qlt_bits::<28>(x, z)"), "{}", em.body);
        assert!(em.body.contains("qlt_bits::<64>(x, w)"), "{}", em.body);
        assert!(em.body.contains("qlt_bits::<64>(x, u)"), "{}", em.body);
        assert!(!em.body.contains(" as u"), "{}", em.body);
    }

    // Signed comparisons go through the gap-pinned `qlt_bits` (one range
    // check), and a comparison and its negation share one call.
    #[test]
    fn signed_comparisons_share_one_gap_pinned_qlt() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let (x, y) = ("x".to_string(), "y".to_string());
        em.lt_i(&x, &y).unwrap();
        em.gte_i(&x, &y).unwrap();
        em.gt_i(&y, &x).unwrap();
        assert_eq!(em.body.matches("qlt_bits::<64>(x, y)").count(), 1, "{}", em.body);
        assert!(!em.body.contains("to_signed"), "{}", em.body);
        em.lte_i(&x, &y).unwrap();
        assert!(em.body.contains("qlt_bits::<64>(y, x)"), "{}", em.body);
    }

    // A product with a float made from an integer (`FloatCast`, or an
    // integral constant) is exact: no rescale.
    #[test]
    fn products_with_integral_floats_skip_the_rescale() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let n = em.float_cast(&"n".to_string()).unwrap();
        let two = em.constant_float(2.0).unwrap();
        let half = em.constant_float(0.5).unwrap();
        let (x, y) = ("x".to_string(), "y".to_string());
        em.mul_f(&n, &x).unwrap();
        em.mul_f(&x, &two).unwrap();
        assert!(!em.body.contains("qmul"), "{}", em.body);
        assert!(em.body.contains("n * x") && em.body.contains("x * 2"), "{}", em.body);
        // A non-integral constant still rescales; in a product sum only the
        // inexact pairs do.
        em.mul_f(&x, &half).unwrap();
        assert!(em.body.contains("qmul(x, "), "{}", em.body);
        em.product_sum_f(&[(n.clone(), x.clone()), (x.clone(), y.clone())]).unwrap();
        assert!(em.body.contains("qscale_down_raw((x * y))"), "{}", em.body);
    }

    // Bitwise ops with constant masks read the low bits of the operand once;
    // a one-bit test against 0 is the bit itself (no zero test).
    #[test]
    fn constant_masks_share_one_bit_decomposition() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let x = "x".to_string();
        let mut masked = Vec::new();
        for k in (0..11).rev() {
            let mask = em.constant_int(1 << k).unwrap();
            masked.push(em.bit_and_i(&x, &mask).unwrap());
        }
        assert_eq!(em.body.matches("low_bits::<2>(x)").count(), 1, "{}", em.body);
        let zero = em.constant_int(0).unwrap();
        em.gt_i(&masked[0], &zero).unwrap();
        assert!(em.body.contains("[10] as Field;"), "{}", em.body);
        assert!(!em.body.contains("qlt") && !em.body.contains("== 0"), "{}", em.body);
        // A negative mask goes through the complement; two dynamic operands
        // through the library's `bitand`.
        let neg = em.constant_int(-8).unwrap();
        em.bit_and_i(&x, &neg).unwrap();
        em.bit_or_i(&x, &"y".to_string()).unwrap();
        assert!(em.body.contains("bitand(x, y)"), "{}", em.body);
    }

    // `AssertU128LtBit` on the same index shares one bit decomposition,
    // which with one check of `hi - 1 - idx` is the whole bound check.
    #[test]
    fn index_bits_share_one_decomposition() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let idx = "idx".to_string();
        for bit in 0..3 {
            em.assert_u128_lt_bit(&idx, 5, bit, 3).unwrap();
        }
        assert_eq!(em.body.matches("idx.to_le_bits()").count(), 1, "{}", em.body);
        assert_eq!(em.body.matches("assert_max_bit_size::<3>()").count(), 1, "{}", em.body);
        // The bound is recorded: a later AssertU128Lt(idx, 5) adds nothing.
        let before = em.body.clone();
        let hi = em.constant_int(5).unwrap();
        em.assert_u128_lt(&idx, &hi).unwrap();
        assert_eq!(em.body, before);
    }

    // Comparisons with 0, abs, sign, sqrt and int casts of one value share
    // one `sign_abs`.
    #[test]
    fn sign_users_share_one_sign_abs() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let x = "x".to_string();
        let zero = em.constant_float(0.0).unwrap();
        em.lt_f(&x, &zero).unwrap();
        em.sqrt_f(&x).unwrap();
        em.abs_f(&x).unwrap();
        em.int_cast(&x).unwrap();
        em.gt_f(&x, &zero).unwrap();
        em.lte_f(&zero, &x).unwrap();
        em.sign_f(&x).unwrap();
        assert_eq!(em.body.matches("sign_abs(x)").count(), 1, "{}", em.body);
        assert!(!em.body.contains("qlt_bits"), "{}", em.body);
    }

    // A float division by a constant divides by the constant directly: an
    // integral float n by n (a power of two by one remainder check), any
    // other constant c by its raw value after scaling the numerator.
    #[test]
    fn division_by_a_constant_skips_the_sign_projection() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let x = "x".to_string();
        for value in [2.0, 30.0, 0.6, -4.0] {
            let c = em.constant_float(value).unwrap();
            em.div_f(&x, &c).unwrap();
        }
        assert!(em.body.contains("divmod_by_pow2::<1>(x)"), "{}", em.body);
        assert!(em.body.contains("divmod_by_positive::<5>(x, 30)"), "{}", em.body);
        assert!(em.body.contains("divmod_by_positive::<42>(x * 4294967296, 2576980378)"), "{}", em.body);
        assert!(em.body.contains("divmod_by_pow2::<2>(x)"), "{}", em.body);
        assert!(!em.body.contains("qdiv") && !em.body.contains("sign_abs"), "{}", em.body);
    }

    // B4(b): the index of `AssertU128Lt` carries the interval the builder
    // derives from this assert; the lowering must not use it (a direct
    // range check on the Field value).
    #[test]
    fn assert_u128_lt_does_not_use_its_own_interval() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
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

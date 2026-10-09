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

use crate::circuit_input::{InputNode, InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;

use super::inputs::{leaf_expr, noir_param_identifier, LeafLayout};

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

/// Sums of more terms than this are bound in chunks (`NoirEmitter::bind_sum`)
/// before a library call reads them: nargo 1.0.0-beta.20 stops with
/// "Reached the recursion limit during elaboration" on one expression of a
/// few hundred terms. Shorter sums stay one expression (no change in gates).
const LONG_SUM_TERMS: usize = 64;

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
    emitter.wrapped_index_ranges = wrapped_index_ranges(ir);
    emitter.asserted_comparisons = asserted_comparisons(ir);
    emitter.implied_comparisons = implied_comparisons(ir, &emitter.asserted_comparisons);
    emitter.multiplied = multiplied(ir);
    emitter.reads = reads(ir);
    emitter.root_guards = crate::prove::guards::root_guards(ir);
    emitter.divisor_guards = crate::prove::guards::divisor_guards(ir);
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
                    | IR::LtIBounded | IR::GtIBounded | IR::IndexInBox { .. } | IR::NeI | IR::NeF
                    | IR::EqI | IR::EqF | IR::KeyLess { .. }
                    | IR::PackedKeyLess { .. }
            )
        })
        .map(|s| s.stmt_id)
        .filter(|id| assert_uses.contains_key(id) && !other_uses.contains(id))
        .collect()
}

/// How many `MulI` operands each statement's value is, not counting
/// products with a constant (those scale an expression's terms; they add
/// no product term).
fn multiplied(ir: &IRGraph) -> HashMap<u32, usize> {
    use crate::ir_defs::IR;
    let constant = |id: u32| matches!(ir.stmts.get(id as usize).map(|s| &s.ir), Some(IR::ConstantInt { .. }));
    let mut uses: HashMap<u32, usize> = HashMap::new();
    for stmt in ir.stmts.iter().filter(|s| matches!(s.ir, IR::MulI)) {
        if let [a, b] = stmt.arguments.as_slice() {
            if constant(*a) || constant(*b) {
                continue;
            }
            for arg in [a, b] {
                *uses.entry(*arg).or_insert(0) += 1;
            }
        }
    }
    uses
}

/// How many statement operands each statement's value is.
fn reads(ir: &IRGraph) -> HashMap<u32, usize> {
    let mut reads: HashMap<u32, usize> = HashMap::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            *reads.entry(*arg).or_insert(0) += 1;
        }
    }
    reads
}

/// Asserted comparisons of a value with a constant that a bounds check of
/// the same value elsewhere in the circuit implies: `AssertU128Lt(x, hi)`
/// with a constant hi, applied to x itself (so on every path), puts x in
/// [0, hi - 1], which decides e.g. the input-domain checks
/// `x >= i64::MIN` and `x <= i64::MAX`. The circuit enforces the bounds
/// check, which is the stronger constraint, so these comparisons need no
/// constraint of their own.
fn implied_comparisons(ir: &IRGraph, asserted: &HashSet<u32>) -> HashSet<u32> {
    use crate::ir_defs::IR;
    let constant = |id: u32| match ir.stmts.get(id as usize).map(|s| &s.ir) {
        Some(IR::ConstantInt { value }) => Some(*value as i128),
        _ => None,
    };
    let mut bounds: HashMap<u32, (i128, i128)> = HashMap::new();
    for stmt in &ir.stmts {
        if let (IR::AssertU128Lt, [x, hi]) = (&stmt.ir, stmt.arguments.as_slice()) {
            if let Some(hi) = constant(*hi).filter(|h| *h >= 1) {
                let entry = bounds.entry(*x).or_insert((0, hi - 1));
                entry.1 = entry.1.min(hi - 1);
            }
        }
    }
    let mut implied = HashSet::new();
    for stmt in &ir.stmts {
        if !asserted.contains(&stmt.stmt_id) {
            continue;
        }
        let [a, b] = stmt.arguments.as_slice() else { continue };
        let range = |id: u32| constant(id).map(|c| (c, c)).or_else(|| bounds.get(&id).copied());
        let (Some((al, ah)), Some((bl, bh))) = (range(*a), range(*b)) else { continue };
        let holds = match stmt.ir {
            IR::LtI | IR::LtF => ah < bl,
            IR::LteI | IR::LteF => ah <= bl,
            IR::GtI | IR::GtF => al > bh,
            IR::GteI | IR::GteF => al >= bh,
            _ => false,
        };
        if holds {
            implied.insert(stmt.stmt_id);
        }
    }
    implied
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
        write_param(&mut out, &noir_param_identifier(&p.name), &p.value);
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
        // A tuple or a complex number is written like an array: its leaves in
        // order (`inputs::LeafLayout`).
        InputNode::Array { elements, .. } | InputNode::Sequence(elements) => {
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
    /// The kernel's parameter name (`InputPath::param`).
    pub param: String,
    /// The `fn main` parameter name ([`noir_param_identifier`]).
    pub name: String,
    pub kind: InputKind,
    /// Where each scalar leaf sits in the flat parameter.
    pub layout: LeafLayout,
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
            param: p.name.clone(),
            name: noir_param_identifier(&p.name),
            kind,
            layout: LeafLayout::of(&p.value),
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
                // A memory (`IR::RamInit`) is `let mut`; its writes are
                // statements of their own, which keep it live.
                let lhs = rest[..eq].trim();
                let lhs = lhs.strip_prefix("mut ").unwrap_or(lhs);
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
    /// Comparisons used only by asserts (see `asserted_comparisons`).
    asserted_comparisons: HashSet<u32>,
    /// Asserted `x >= 0` with a later `sqrt(x)` (`crate::prove::guards`).
    root_guards: HashSet<u32>,
    /// Asserted `d != 0` with a later division by `d` (`crate::prove::guards`):
    /// emitted as nothing, `d` recorded in `nonzero` (see `asserted_distinct`).
    divisor_guards: HashSet<u32>,
    /// Cells an emitted `assert(x != 0)` proves non-zero.
    nonzero: HashSet<String>,
    /// `out -> (a, b)` for every emitted sum `out = a + b`.
    sum_of: HashMap<String, (String, String)>,
    /// The number of linear terms of the expression Noir builds for a sum
    /// or difference (see `linear_terms`); absent for every other cell.
    sum_terms: HashMap<String, u32>,
    /// How many `MulI` operands each statement's value is (see
    /// `linear_terms`).
    multiplied: HashMap<u32, usize>,
    /// Cells bound to the cast of one value (`b as Field`, `bs[i] as
    /// Field`, typically a comparison's `bool`), which Noir keeps as a single
    /// witness (see `select`).
    cast_cells: HashSet<String>,
    /// Asserted comparisons a bounds check elsewhere implies (see
    /// `implied_comparisons`): they emit nothing.
    implied_comparisons: HashSet<u32>,
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
    /// The read-only tables (`IR::RomInit`): the Noir array and its cells.
    roms: HashMap<crate::types::StmtId, (String, Vec<String>)>,
    /// The read-write memories (`IR::RamInit`): each memory version's
    /// mutable Noir array and length.
    rams: HashMap<crate::types::StmtId, (String, usize)>,
    /// Hint arrays already computed, by their call expression (see
    /// `prover_hint`).
    hint_arrays: HashMap<String, String>,
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
    /// Cache: division call text -> its (quotient, remainder) names.
    divmod_cache: HashMap<String, (String, String)>,
    /// Pairs `(idx, hi)` already emitted via `assert_u128_lt`. Zinnia's
    /// IR layer can re-issue the same bound discharge (e.g. multiple
    /// dynamic reads against the same axis length re-resolve through
    /// `discharge_index_in_range`); dropping the duplicate `assert((a
    /// as u128) < (b as u128))` line saves a per-cmp u128 cast + compare
    /// (~2 ACIR gates each, ~80 gates on the 0345 mux-indexed kernel).
    emitted_u128_asserts: HashSet<(String, String)>,
    /// Range checks of `0 <= x < hi` whose exact width would need a lookup
    /// table of its own, their width settled when the circuit is rendered
    /// (`settle_range_widths`).
    deferred_widths: Vec<DeferredWidth>,
    /// Each packed key's value cell (`pack_key`), whose interval sizes the
    /// keys' comparisons (`packed_gap_bits`).
    packed_values: HashMap<String, String>,
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
    /// Offset windows an asserted comparison against a constant checked
    /// (`asserted_compare`): x -> (t, c, W) with `t = x - c` a cell whose
    /// `W`-bit range check (`W <= 64`) every accepted witness passes. Later
    /// comparisons of x with a constant read t natively
    /// (`compare_in_checked_window`).
    window_of: HashMap<String, (String, i128, u32)>,
    /// Flags bound to the product of two flags (`logical_and`) that Noir
    /// still holds as a degree-2 expression (see `logical_or`).
    product_flags: HashSet<String>,
    /// How many statement operands each statement's value is.
    reads: HashMap<u32, usize>,
    /// How many statement operands each cell is: the reads of every
    /// statement whose result it is (see `witness_shared_summands`).
    var_reads: HashMap<String, usize>,
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
        if op == '*' && self.linear_terms(a) > 0 && self.linear_terms(b) > 0 {
            for factor in [a, b] {
                if self.linear_terms(factor) > 2 {
                    self.materialize_sum(factor);
                }
            }
        }
        if op != '*' {
            self.witness_shared_summands(a, b);
        }
        let v = self.bind(&format!("{a} {op} {b}"));
        if op == '+' {
            self.record_sum(&v, a, b);
        }
        if op != '*' {
            self.track_linear(&v, a, b);
        }
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
        if self.current_stmt.is_some_and(|id| self.implied_comparisons.contains(&id)) {
            return self.literal_int(1);
        }
        if self.nonneg_by_later_root(a, b, cmp) {
            return self.literal_int(1);
        }
        let asserted = self
            .current_stmt
            .is_some_and(|id| self.asserted_comparisons.contains(&id));
        if asserted && !self.has_cached_sign_for_zero_test(a, b) {
            self.asserted_compare(a, b, cmp);
            return self.literal_int(1);
        }
        if let Some(v) = self.compare_in_checked_window(a, b, cmp) {
            return v;
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

    /// `x cmp k` (or `k cmp x`) for a constant `k` and a cell `x` that an
    /// asserted comparison already checked into a window: `t = x - c` with
    /// `t` in `[0, 2^W)`, `W <= 64` (`window_of`; the input-domain check
    /// `x >= i64::MIN` gives `t = x + 2^63`, the signed 64-bit window).
    /// Then `x < k` exactly when `t < k - c` as integers, which Noir decides
    /// natively on `t as u64`: the cast is exact because the range check of
    /// `t` holds for every accepted witness (it was emitted unconditionally:
    /// the comparison's only uses are asserts, see `asserted_comparisons`),
    /// and Noir's own `u64` comparison is sound. This replaces the hinted
    /// `qlt_bits` or `sign_abs` (a hint, a select and a range check of their
    /// own) with the one comparison the baselines' `window(x) < K` make.
    /// Applies when `0 < k - c` and `k - c + 1 < 2^W` (otherwise the range
    /// of `t` already decides the comparison), and not when a `sign_abs`
    /// pair of x already exists for a comparison with 0 (reading its sign
    /// is free).
    fn compare_in_checked_window(&mut self, a: &str, b: &str, cmp: Cmp) -> Option<String> {
        let (x, k, cmp) = match (self.raw_constant(a), self.raw_constant(b)) {
            (None, Some(k)) => (a, k, cmp),
            (Some(k), None) => (b, k, cmp.mirrored()),
            _ => return None,
        };
        if k == 0 && self.sign_abs_of.contains_key(x) {
            return None;
        }
        // qmul/qscale_down_raw already constrain x + 2^63 to 64 bits
        // inside the library. Reuse that exact expression lazily instead
        // of emitting another sign hint and range check for its result.
        if !self.window_of.contains_key(x)
            && self.known_range(x) == Some((-(1i128 << 63), (1i128 << 63) - 1))
        {
            let t = self.bind(&format!("{x} + {}", 1u64 << 63));
            self.window_of.insert(x.to_string(), (t, -(1i128 << 63), 64));
        }
        let (t, c, width) = self.window_of.get(x).cloned()?;
        let bound = k - c;
        if bound <= 0 || bound + 1 >= 1i128 << width {
            return None;
        }
        // At the signed zero threshold, native > keeps the power-of-two
        // boundary and saves gates. Retain the existing lowering at other
        // thresholds: native > is not uniformly cheaper there.
        let signed_zero = k == 0 && c == -(1i128 << 63) && width == 64;
        let (operator, limit, negate) = match cmp {
            Cmp::Lt => ("<", bound, false),
            Cmp::Ge => ("<", bound, true),
            Cmp::Le if signed_zero => (">", bound, true),
            Cmp::Gt if signed_zero => (">", bound, false),
            Cmp::Le => ("<", bound + 1, false),
            Cmp::Gt => ("<", bound + 1, true),
        };
        let comparison = self.bind_typed(&format!("({t} as u64) {operator} {limit}"), "bool");
        let flag = if negate { format!("(!{comparison}) as Field") } else { format!("{comparison} as Field") };
        Some(self.bind_flag(&flag))
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

    /// Bits `w` with `|K1 - K2| < 2^w` for two packed keys whose values have
    /// known intervals (for example values read at hinted indices that the
    /// sort bounded into its elements' range,
    /// `lower::sort::emit::Elements::bound`): the values differ by at most
    /// the span `s` of the intervals, the indices by less than `2^b`, so
    /// `|K1 - K2| < (s + 1) 2^b`. The intervals only pick the width; a pair
    /// outside them fails the check.
    fn packed_gap_bits(&self, k1: &CellRef, k2: &CellRef, index_bits: u32) -> Option<u32> {
        let (l1, h1) = self.ival(self.packed_values.get(k1)?)?;
        let (l2, h2) = self.ival(self.packed_values.get(k2)?)?;
        let span = (h1.max(h2) as i128) - (l1.min(l2) as i128);
        let bits = 128 - ((span + 1) as u128).leading_zeros();
        Some(bits + index_bits)
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

    /// An asserted `x >= 0` that a later `sqrt(x)` proves
    /// (`crate::prove::guards`): when an earlier check puts `x` in the
    /// signed 64-bit window, nothing is emitted and `x`'s range becomes
    /// `[0, hi]`, so `sqrt_f` takes `qsqrt_of(false, x)`, whose
    /// `isqrt_floor(x·Q)` has no witness for a negative `x` (`y` and
    /// `x·Q - y^2` are range-checked to 64 bits, far below p).
    fn nonneg_by_later_root(&mut self, a: &str, b: &str, cmp: Cmp) -> bool {
        if !self.current_stmt.is_some_and(|id| self.root_guards.contains(&id)) {
            return false;
        }
        let x = match cmp {
            Cmp::Ge if self.raw_constant(b) == Some(0) => a,
            Cmp::Le if self.raw_constant(a) == Some(0) => b,
            _ => return false,
        };
        let Some((lo, hi)) = self.proven_range.get(x).copied() else { return false };
        if lo < -(1i128 << 63) || hi >= 1i128 << 63 {
            return false;
        }
        self.narrow_proven_range(x, 0, hi);
        true
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
        // `x >= c` (or `x > c - 1`) checked as t = x - c in [0, 2^W): later
        // comparisons of x with a constant read t (`compare_in_checked_window`).
        if let (Some(c), None) = (self.raw_constant(low), self.raw_constant(high)) {
            if width <= 64 && !self.window_of.contains_key(high) {
                self.window_of.insert(high.to_string(), (t.clone(), c + strict as i128, width));
            }
        }
        // low + strict <= high < low + strict + 2^W: an operand with a known
        // range (a constant, or a range an earlier check proves) bounds the
        // other one, an integer then too.
        let span = 1i128 << width;
        let offset = strict as i128;
        if let Some((lo, hi)) = self.known_range(high) {
            self.narrow_proven_range(low, lo - offset - span + 1, hi - offset);
        }
        if let Some((lo, hi)) = self.known_range(low) {
            self.narrow_proven_range(high, lo + offset, hi + offset + span - 1);
        }
    }

    /// The range of a constant or of a cell whose range an emitted check proves.
    fn known_range(&self, cell: &str) -> Option<(i128, i128)> {
        self.raw_constant(cell).map(|c| (c, c)).or_else(|| self.proven_range.get(cell).copied())
    }

    fn record_signed64(&mut self, cell: &str) {
        self.narrow_proven_range(cell, -(1i128 << 63), (1i128 << 63) - 1);
    }

    /// Remember that `out = a + b`, so a select between `out` and one of
    /// its summands is one product of the other summand (see `select`).
    fn record_sum(&mut self, out: &str, a: &str, b: &str) {
        self.sum_of.insert(out.to_string(), (a.to_string(), b.to_string()));
    }

    /// The number of linear terms of the expression Noir builds for `var`:
    /// 0 for a literal, the recorded count for a sum or difference, else 1
    /// (a witness).
    ///
    /// Noir does not give an intermediate sum a witness of its own: `s = a +
    /// b + c` stays the expression `a + b + c`, and `s * x` becomes one
    /// product term per term of `s`, one gate each. A running sum whose
    /// every prefix is multiplied (a thermometer `g[k] = g[k-1] + h[k]`
    /// times each element) then costs a gate per term per product, quadratic
    /// in its length. So a sum the IR multiplies gets a witness where it is
    /// defined (`track_linear`) when that saves gates: when it has three or
    /// more terms, or two and more than one product reads it. The next
    /// prefix then starts from that witness.
    fn linear_terms(&self, var: &str) -> u32 {
        if self.literal_vals.contains_key(var) {
            return 0;
        }
        self.sum_terms.get(var).copied().unwrap_or(1)
    }

    /// Record the terms of `out = a ± b`, and give it a witness when
    /// products read it (see `linear_terms`).
    fn track_linear(&mut self, out: &str, a: &str, b: &str) {
        if self.literal_vals.contains_key(out) || self.sum_terms.contains_key(out) || out == a || out == b {
            return;
        }
        let terms = self.linear_terms(a) + self.linear_terms(b);
        self.sum_terms.insert(out.to_string(), terms);
        if self.needs_witness(terms) {
            self.materialize_sum(out);
        }
    }

    /// Whether the sum of `terms` linear terms that the current statement
    /// computes gets a witness: products read it, and it has three or more
    /// terms, or two and more than one product reads it (`linear_terms`).
    fn needs_witness(&self, terms: u32) -> bool {
        let products = self.current_stmt.and_then(|id| self.multiplied.get(&id)).copied().unwrap_or(0);
        products > 0 && (terms > 2 || (terms == 2 && products > 1))
    }

    /// Before `out = a ± b` is bound: when `out` would get a witness
    /// (`needs_witness`), an operand that is itself a sum of three or more
    /// terms read by several statements gets one first. A Barretenberg
    /// arithmetic gate has four wires, so `out = x + s` fits one gate when
    /// `s` has at most two terms; with three (`x + c0 + c1 + c2`, five
    /// wires) every such `out` takes two, while with the operand's witness,
    /// paid once, each is `x + s` (one gate): a sum of constants shared by
    /// many elements (the Horner coefficients of a membership test) is the
    /// typical case. `std::as_witness` only tells Noir where to cut an
    /// expression; the constraints still say `out = a ± b`, so this changes
    /// gates, never what is proved.
    fn witness_shared_summands(&mut self, a: &str, b: &str) {
        if !self.needs_witness(self.linear_terms(a) + self.linear_terms(b)) {
            return;
        }
        for x in [a, b] {
            if self.linear_terms(x) > 2 && self.var_reads.get(x).copied().unwrap_or(0) > 1 {
                self.materialize_sum(x);
            }
        }
    }

    /// Give a sum of three or more terms its own witness before a product
    /// reads it (two terms cost two product terms either way).
    fn materialize_sum(&mut self, var: &str) {
        if self.linear_terms(var) > 1 {
            let _ = writeln!(self.body, "    std::as_witness({var});");
            self.sum_terms.insert(var.to_string(), 1);
        }
    }

    /// `a != b` for a `!=` whose only uses are asserts: the assertion
    /// itself (`assert(a != b)`, an inverse constraint), and the literal
    /// true for the asserts. Against the constant 0 the other operand is
    /// recorded as non-zero, so divisions by it skip the zero-divisor
    /// mapping (`nonzero`).
    fn asserted_distinct(&mut self, a: &str, b: &str) -> Option<String> {
        let asserted = self.current_stmt.is_some_and(|id| self.asserted_comparisons.contains(&id));
        if !asserted {
            return None;
        }
        // `d != 0` before a division by `d` (`crate::prove::guards`): the
        // division of a divisor in `nonzero` bounds its remainder by `|d|`
        // (`d_mag - 1 - r_mag` or `d - 1 - r` range-checked), which has no
        // solution for `d = 0`, so the assertion emits nothing.
        let guarded_divisor = self.current_stmt.is_some_and(|id| self.divisor_guards.contains(&id))
            && [(a, b), (b, a)].into_iter().any(|(d, zero)| self.raw_constant(zero) == Some(0) && self.raw_constant(d).is_none());
        if !guarded_divisor {
            let _ = writeln!(self.body, "    assert({a} != {b});");
        }
        for (x, other) in [(a, b), (b, a)] {
            if self.raw_constant(other) == Some(0) {
                self.nonzero.insert(x.to_string());
            }
        }
        Some(self.literal_int(1))
    }

    /// Whether every accepted witness has `cell >= 0`: an emitted check
    /// (`proven_range`) or the static interval of its IR value (`ival`).
    fn known_non_negative(&self, cell: &str) -> bool {
        self.proven_range.get(cell).is_some_and(|(lo, _)| *lo >= 0) || self.ival(cell).is_some_and(|(lo, _)| lo >= 0)
    }

    /// `a == b` for an `==` whose only uses are asserts: the assertion
    /// itself (`assert(a == b)`, one constraint) and the literal true for
    /// the asserts, instead of an equality bit.
    fn asserted_equal(&mut self, a: &str, b: &str) -> Option<String> {
        let asserted = self.current_stmt.is_some_and(|id| self.asserted_comparisons.contains(&id));
        if !asserted {
            return None;
        }
        let _ = writeln!(self.body, "    assert({a} == {b});");
        Some(self.literal_int(1))
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
        if self.known_range(x).is_some_and(|(lo, hi)| lo >= 0 && hi < 1i128 << 64) {
            return ("false".to_string(), x.to_string());
        }
        self.use_lib("sign_abs");
        let (negative, mag) = (self.fresh(), self.fresh());
        let _ = writeln!(self.body, "    let ({negative}, {mag}) = sign_abs({x});");
        self.sign_abs_of
            .insert(x.to_string(), (negative.clone(), mag.clone()));
        (negative, mag)
    }

    /// `|x|` (library `abs_int64`, one 64-bit range check): no proof for
    /// `x = i64::MIN`, whose magnitude overflows int64 (the raw Q32 value of
    /// a float lives in the same window). `sign_abs`'s magnitude is not used
    /// here because it admits `2^63`.
    /// `|x|` (`noir_fpc::abs_int64`); its check `2|x| + sign < 2^64` puts the
    /// result in `[0, 2^63)`, recorded in `proven_range`.
    fn abs_int64(&mut self, x: &str) -> String {
        self.use_lib("abs_int64");
        let mag = self.bind(&format!("abs_int64({x})"));
        self.narrow_proven_range(&mag, 0, (1i128 << 63) - 1);
        mag
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
        self.witness_divisor(b);
        if self.nonzero.contains(b) {
            let (negative, mag) = self.sign_abs(b);
            self.use_lib("floor_divmod_nonzero_of");
            return self.divmod_call(format!("floor_divmod_nonzero_of({a}, {b}, {negative}, {mag})"));
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

    /// An integer divisor (or the operand of `>>`) must be an int64: `b` in
    /// `[-2^63, 2^63)`, with no proof otherwise (`assert_in_window`, one 64-bit check of `b + 2^63`;
    /// nothing when a constant or an earlier check already proves the range,
    /// e.g. an input's domain check).
    ///
    /// Why. The IR's integer `+ - *` are exact in the field, NumPy's int64
    /// arithmetic wraps; the two agree modulo 2^64, so a value that left the
    /// window (a row sum of `|x|` reaching 2^63) must not be observed by an
    /// operation that does not commute with the wrap. The division gadgets
    /// bound the divisor only by about 2^64 (`d - 1 - r` or `sign_abs`), so
    /// `x / sum(|x|)` divided by the unwrapped 2^63 and proved `0.5` where
    /// NumPy gives `-0.5`. Inside the window the field value's signed reading
    /// is NumPy's int64, so the division is NumPy's.
    /// A dynamic divisor that is a sum gets a witness before its first
    /// division: the division gadgets read it in several constraints
    /// (`q * d + r == n`, the remainder bound `d - 1 - r`, a window check),
    /// and Noir would otherwise expand every term of the sum into each of
    /// them (0452: two gates per remainder bound instead of one). The
    /// constraints are unchanged; only where Noir cuts the expression.
    fn witness_divisor(&mut self, b: &str) {
        self.materialize_sum(b);
    }

    fn ensure_int64(&mut self, b: &str) {
        self.witness_divisor(b);
        let window = (-(1i128 << 63), (1i128 << 63) - 1);
        if self.known_range(b).is_some_and(|(lo, hi)| lo >= window.0 && hi <= window.1) {
            return;
        }
        self.use_lib("assert_in_window");
        let _ = writeln!(self.body, "    assert_in_window({b});");
        self.narrow_proven_range(b, window.0, window.1);
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
        if needs_table_of_its_own(bits) {
            let id = self.deferred_widths.len();
            let _ = writeln!(self.body, "    {t}.assert_max_bit_size::<__W{id}__>();");
            let _ = writeln!(self.body, "    __TOP{id}__;");
            self.deferred_widths.push(DeferredWidth {
                bits,
                cell: t,
                top: hi - 1,
                exact_power: hi == 1i128 << bits,
                one_sided: false,
            });
            return;
        }
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

    /// The mutable array and length of a memory version.
    fn ram_version(&self, version: crate::types::StmtId) -> Result<(String, usize), ProvingError> {
        self.rams
            .get(&version)
            .cloned()
            .ok_or_else(|| ProvingError::synthesis(format!("statement {version} is not a memory version")))
    }

    /// An index into a memory of `len` cells: a constant must be in range at
    /// compile time; a runtime address is checked by `array_index` (an
    /// address outside fails the proof, as the mock rejects it).
    fn ram_index(&mut self, address: &CellRef, len: usize) -> Result<String, ProvingError> {
        if let Some(a) = self.s_int_of(address) {
            return usize::try_from(a)
                .ok()
                .filter(|&a| a < len)
                .map(|a| a.to_string())
                .ok_or_else(|| ProvingError::synthesis(format!("memory access out of bounds: address {a}")));
        }
        Ok(self.array_index(address, len as u64))
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
            wrapped_index_ranges: HashMap::new(),
            asserted_comparisons: HashSet::new(),
            root_guards: HashSet::new(),
            divisor_guards: HashSet::new(),
            nonzero: HashSet::new(),
            sum_of: HashMap::new(),
            sum_terms: HashMap::new(),
            multiplied: HashMap::new(),
            cast_cells: HashSet::new(),
            implied_comparisons: HashSet::new(),
            current_stmt: None,
            proven_range: HashMap::new(),
            roms: HashMap::new(),
            rams: HashMap::new(),
            hint_arrays: HashMap::new(),
            interval_of: HashMap::new(),
            lt_zero_of: HashMap::new(),
            add_const_of: HashMap::new(),
            norm_index_of: HashMap::new(),
            divmod_cache: HashMap::new(),
            emitted_u128_asserts: HashSet::new(),
            deferred_widths: Vec::new(),
            packed_values: HashMap::new(),
            cse_cache: HashMap::new(),
            cell_bound: HashMap::new(),
            literal_vals: HashMap::new(),
            integral_float: HashMap::new(),
            nonneg_by_construction: HashSet::new(),
            scaled_bit: HashMap::new(),
            low_bits_of: HashMap::new(),
            index_bits_of: HashMap::new(),
            sign_abs_of: HashMap::new(),
            window_of: HashMap::new(),
            product_flags: HashSet::new(),
            reads: HashMap::new(),
            var_reads: HashMap::new(),
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
        if is_single_cast(expr) {
            self.cast_cells.insert(v.clone());
        }
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

    /// A lane of `HintKind::PositionOneHot` (operands `[x, n]`): with a
    /// constant `n` and rank, an entry of `position_one_hot_hint`'s array of
    /// `n + 1` lanes (one unconstrained call per operand pair); otherwise
    /// the `_at` form. No constraint: the IR pins every lane it uses.
    fn position_one_hot_hint(&mut self, operands: &[CellRef], rank: &CellRef) -> Result<CellRef, ProvingError> {
        let [x, n] = operands else {
            return Err(ProvingError::synthesis("position_one_hot hint expects the operands [x, n]"));
        };
        let lanes = self.s_int_of(n).and_then(|n| usize::try_from(n).ok()).map(|n| n + 1);
        match (lanes, self.s_int_of(rank)) {
            (Some(lanes), Some(rank)) => {
                let Some(lane) = usize::try_from(rank).ok().filter(|&r| r < lanes) else {
                    return Ok(self.literal_int(0));
                };
                let key = format!("position_one_hot_hint({x}, {n})");
                let array = match self.hint_arrays.get(&key) {
                    Some(name) => name.clone(),
                    None => {
                        self.use_lib("position_one_hot_hint");
                        let name = self.fresh();
                        let _ = writeln!(self.body, "    let {name}: [Field; {lanes}] = {key};");
                        self.hint_arrays.insert(key, name.clone());
                        name
                    }
                };
                Ok(self.bind(&format!("{array}[{lane}]")))
            }
            _ => {
                self.use_lib("position_one_hot_hint_at");
                Ok(self.bind(&format!("position_one_hot_hint_at({x}, {n}, {rank})")))
            }
        }
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
    let input = inputs.iter().find(|i| i.param == path.param).ok_or_else(|| {
        ProvingError::other(format!("noir-emit: read of `{}`, which is no circuit input", path.param))
    })?;
    leaf_expr(&input.name, &input.layout, path).ok_or_else(|| ProvingError::UnsupportedIR {
        op_name: format!("noir-emit: input path `{path}` does not address a scalar of `{}`", path.param),
    })
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
        stmt_id: u32,
        value_id: crate::types::ValueId,
        cell: &CellRef,
    ) {
        // Name ↔ ValueId correspondence for live provenance-table reads.
        self.vid_of_var.insert(cell.clone(), value_id);
        let reads = self.reads.get(&stmt_id).copied().unwrap_or(0);
        *self.var_reads.entry(cell.clone()).or_insert(0) += reads;
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
        // The sum of two values with proven ranges (integers far below p/2)
        // is the integer sum: its range is the sum of the ranges.
        if let (Some((al, ah)), Some((bl, bh))) =
            (self.proven_range.get(a.as_str()).copied(), self.proven_range.get(b.as_str()).copied())
        {
            if let (Some(lo), Some(hi)) = (al.checked_add(bl), ah.checked_add(bh)) {
                if lo > -(1i128 << 120) && hi < 1i128 << 120 {
                    self.narrow_proven_range(&out, lo, hi);
                }
            }
        }
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
        self.ensure_int64(b);
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
        self.ensure_int64(b);
        let (_q, r) = self.floor_divmod_cells(a, b);
        Ok(r)
    }
    fn pow_i(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "pow_i")
    }
    fn abs_i(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        Ok(self.abs_int64(a))
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
    /// NumPy's int64 `a << b` (`kernel::fp_shl`): `a * 2^b` wrapped into the
    /// int64 window for `0 <= b < 64`, 0 for any other `b`; `a` must be an
    /// int64 (`ensure_int64`). A constant count is resolved here (the wrap,
    /// `wrap_int64`, is skipped when the product's known range lies in the
    /// window); a dynamic one is `shl_int64` (noir_fpc).
    fn shl_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some((x, y)) = self.constant_pair(a, b) {
            return Ok(self.literal_int(crate::prove::kernel::int64_shl(x, y)));
        }
        if let Some(count) = self.s_int_of(b).and_then(|c| i64::try_from(c).ok()) {
            let c = match crate::prove::kernel::int64_shift_count(count) {
                None => return Ok(self.literal_int(0)),
                Some(0) => return Ok(a.clone()),
                Some(c) => c,
            };
            self.ensure_int64(a);
            let power = 1i128 << c;
            let fits = self.known_range(a).is_some_and(|(lo, hi)| {
                let window = -(1i128 << 63)..(1i128 << 63);
                window.contains(&(lo << c)) && window.contains(&(hi << c))
            });
            let power = self.bind_val(&power.to_string(), ConcreteValue::Int(power));
            let product = self.integer_arith(a, &power, '*');
            if fits {
                return Ok(product);
            }
            self.use_lib("wrap_int64");
            return Ok(self.bind(&format!("wrap_int64({product})")));
        }
        self.use_lib("shl_int64");
        Ok(self.bind(&format!("shl_int64({a}, {b})")))
    }
    /// NumPy's int64 `a >> b` (`kernel::fp_shr`): `floor(a / 2^c)` with
    /// `c = b` for `0 <= b < 64`, else 63 (the sign fill); `a` must be an
    /// int64 (`ensure_int64`). A constant count is the floor division by the
    /// constant `2^c` (`divmod_by_pow2`: a c-bit remainder check, the quotient
    /// window-checked); a dynamic one is `shr_int64` (noir_fpc).
    fn shr_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some((x, y)) = self.constant_pair(a, b) {
            return Ok(self.literal_int(crate::prove::kernel::int64_shr(x, y)));
        }
        if let Some(count) = self.s_int_of(b).and_then(|c| i64::try_from(c).ok()) {
            self.ensure_int64(a);
            return Ok(match crate::prove::kernel::int64_shift_count(count).unwrap_or(63) {
                0 => a.clone(),
                c => self.divide_by_positive_constant(a, 1i128 << c).0,
            });
        }
        self.use_lib("shr_int64");
        Ok(self.bind(&format!("shr_int64({a}, {b})")))
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
        self.witness_shared_summands(a, b);
        let v = self.bind(&format!("qadd({}, {})", a, b));
        self.record_sum(&v, a, b);
        self.track_linear(&v, a, b);
        Ok(v)
    }
    fn sub_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsub");
        self.witness_shared_summands(a, b);
        let v = self.bind(&format!("qsub({}, {})", a, b));
        self.track_linear(&v, a, b);
        Ok(v)
    }
    fn mul_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(exact) = self.exact_product(a, b) {
            return Ok(self.bind(&exact));
        }
        self.use_lib("qmul");
        let v = self.bind(&format!("qmul({}, {})", a, b));
        self.record_signed64(&v);
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
            // A long sum is bound in chunks first (`bind_sum`): nargo's
            // elaborator hits its recursion limit on one expression of a
            // few hundred terms (`np.dot` of length 256), and the chunks are
            // the same linear combination.
            let sum = if raw.len() > LONG_SUM_TERMS { self.bind_sum(&raw) } else { raw.join(" + ") };
            let rescaled = self.bind(&format!("qscale_down_raw({sum})"));
            self.record_signed64(&rescaled);
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
        if self.nonzero.contains(b.as_str()) {
            // A divisor known non-negative: `divmod_by_positive` bounds the
            // remainder by the divisor itself (`d - 1 - r` 64 bits, so
            // `d >= 1`; a wrong sign would only fail the proof), with no sign
            // projection; floor and Euclid agree for `d > 0`. An integral
            // float `b = n * 2^32` cancels the scale, as halo2-base's
            // `fixed_div` does: the Euclidean quotient of `a` by `n` equals
            // that of `a * 2^32` by `b`.
            let (numerator, divisor) = match self.integral_float.get(b.as_str()).cloned() {
                Some(n) => {
                    self.ensure_int64(&n);
                    (a.to_string(), n)
                }
                None => (format!("{a} * {Q_U64}"), b.clone()),
            };
            if self.known_non_negative(&divisor) {
                self.use_lib("divmod_by_positive");
                return Ok(self.bind(&format!("divmod_by_positive::<64>({numerator}, {divisor}).0")));
            }
            let (_, mag) = self.sign_abs(b);
            self.use_lib("qdiv_nonzero_of");
            return Ok(self.bind(&format!("qdiv_nonzero_of({a}, {b}, {mag})")));
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
    /// `kernel::fp_pow_int` (`qpow_int`): square-and-multiply over the
    /// `exponent_bits` low bits of `|b|`.
    fn pow_f_int(&mut self, a: &CellRef, b: &CellRef, exponent_bits: u32) -> Result<CellRef, ProvingError> {
        self.use_lib("qpow_int");
        Ok(self.bind(&format!("qpow_int::<{exponent_bits}>({a}, {b})")))
    }
    fn abs_f(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        let v = self.abs_int64(a);
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
        if let Some(v) = self.asserted_equal(a, b) {
            return Ok(v);
        }
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
        if let Some(v) = self.asserted_distinct(a, b) {
            return Ok(v);
        }
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
        if let Some(v) = self.asserted_equal(a, b) {
            return Ok(v);
        }
        let (af, bf) = (a, b);
        Ok(self.bind_flag(&format!("if {} == {} {{ 1 }} else {{ 0 }}", af, bf)))
    }
    fn ne_f(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        if let Some(v) = self.asserted_distinct(a, b) {
            return Ok(v);
        }
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
        // An operand a check emitted earlier proves non-negative (the
        // `x >= 0` precondition of `np.sqrt`) is its own magnitude, below
        // 2^64, inside `isqrt_floor`'s domain.
        let root = if self.proven_range.get(a.as_str()).is_some_and(|(lo, hi)| *lo >= 0 && *hi < 1i128 << 64) {
            self.use_lib("qsqrt_of");
            self.bind(&format!("qsqrt_of(false, {a})"))
        } else {
            let (negative, mag) = self.sign_abs(a);
            self.use_lib("qsqrt_of");
            self.bind(&format!("qsqrt_of({negative}, {mag})"))
        };
        // Both paths clamp negatives to zero and apply floor sqrt to an
        // unsigned64 magnitude times Q=2^32, so the root is <2^48.
        self.narrow_proven_range(&root, 0, (1i128 << 48) - 1);
        Ok(root)
    }
    /// `floor(sqrt(b * c) * 2^32)` of integers with `0 <= b < 2^64` and
    /// `0 <= b * c < 2^62` (`qsqrt_product_int`'s checks; `noir_fpc`
    /// hardcodes P = 32). Its floor root `y` satisfies
    /// `y^2 <= b * c * 2^64 < 2^126`, so `y` lies in `[0, 2^63)`.
    fn sqrt_product_i_f(&mut self, b: &CellRef, c: &CellRef) -> Result<CellRef, ProvingError> {
        self.use_lib("qsqrt_product_int");
        let root = self.bind(&format!("qsqrt_product_int({b}, {c})"));
        self.narrow_proven_range(&root, 0, (1i128 << 63) - 1);
        Ok(root)
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
    /// `kernel::fp_atan2` (`qarctan2`, bit for bit); its domain, the int64
    /// window of both operands, is checked first (`ensure_int64`: nothing
    /// when a known range proves it). `qarctan2`'s own `sign_abs` admits
    /// magnitudes up to 2^64.
    fn arctan2_f(&mut self, y: &CellRef, x: &CellRef) -> Result<CellRef, ProvingError> {
        self.ensure_int64(y);
        self.ensure_int64(x);
        self.use_lib("qarctan2");
        Ok(self.bind(&format!("qarctan2({y}, {x})")))
    }

    fn logical_and(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (a, b) = (self.booleanize(a), self.booleanize(b));
        if a == b {
            return Ok(a);
        }
        let (af, bf) = (&a, &b);
        let v = self.bind_flag(&format!("{} * {}", af, bf));
        if self.literal_vals.get(&v).is_none() {
            self.product_flags.insert(v.clone());
        }
        Ok(v)
    }
    fn logical_or(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let (a, b) = (self.booleanize(a), self.booleanize(b));
        if a == b {
            return Ok(a);
        }
        // An operand that is an AND (`p * q`, still an expression in Noir)
        // gets a witness first: `a + (1 - a) * b` of two such expressions
        // otherwise makes Noir bind `1 - p * q` and keep two product terms
        // (four gates for `(p & q) | (r & s)`, against three with both ANDs
        // as witnesses; measured on DS-1000 0444). `std::as_witness` changes
        // only where Noir cuts the expression, not the constraints.
        for x in [&a, &b] {
            if self.product_flags.remove(x.as_str()) {
                let _ = writeln!(self.body, "    std::as_witness({x});");
            }
        }
        // `a + (1 - a) * b`, the same polynomial as `a + b - a * b`: in a
        // chain of ORs (an `any` over a mask) Noir lowers this form with
        // fewer gates (measured with nargo 1.0.0-beta.20 / bb 5.0.0).
        Ok(self.bind_flag(&format!("{a} + (1 - {a}) * {b}")))
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
        // `f - c * f + c * t` picks a branch only for c in {0, 1}; any
        // other condition is reduced to 0/1 first (Python truthiness, as the
        // mock). When `f == 0` (the common dyn-slice / mask-then-default-zero
        // pattern) the second half is dead and the binding is `c * t`.
        // When c is the cast of one bool (a comparison result), the select
        // is written `f - c * f + c * t`, without a `1 - c` factor: for a
        // chain of selects (a running argmax) Noir binds `1 - c` to a witness
        // of its own, one more gate per select. For any other condition (a
        // product, `1 - x`, a sum) that form costs more gates than the
        // `(1 - c)` form on the DS-1000 set, so it keeps that form. Both are
        // the same polynomial; only Noir's lowering of the text differs (the
        // gate counts were measured with nargo 1.0.0-beta.20 / bb 5.0.0).
        let c = self.booleanize(cond);
        let is_zero_int = self.s_int_of(if_false) == Some(0);
        let is_zero_flt = self.s_float_of(if_false) == Some(0.0);
        let difference = |x: &str, base: &str| -> Option<String> {
            let (p, q) = self.sum_of.get(x)?;
            if p == base {
                Some(q.clone())
            } else if q == base {
                Some(p.clone())
            } else {
                None
            }
        };
        let expr = if is_zero_int || is_zero_flt {
            format!("{} * {}", c, if_true)
        } else if let Some(x) = difference(if_true, if_false) {
            // t = f + x: one product, f + c * x.
            format!("{if_false} + {c} * {x}")
        } else if let Some(x) = difference(if_false, if_true) {
            // f = t + x: t + (1 - c) * x.
            format!("{if_true} + (1 - {c}) * {x}")
        } else if self.cast_cells.contains(&c) {
            // c is one witness: no `1 - c` factor (see above).
            format!("{if_false} - {c} * {if_false} + {c} * {if_true}")
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
    /// `a ∈ {0, 1}`; every accepted witness then has `a` boolean (the
    /// circuit has no control flow), so later lowerings may use it as a
    /// flag without a zero test.
    fn assert_bit(&mut self, a: &CellRef) -> Result<(), ProvingError> {
        let _ = writeln!(self.body, "    assert({a} * ({a} - 1) == 0);");
        self.set_bound(a, 1);
        Ok(())
    }

    /// A native array holding the cells (a ROM block in bb).
    fn rom_init(&mut self, table: crate::types::StmtId, cells: &[CellRef]) -> Result<(), ProvingError> {
        let name = self.fresh();
        let _ = writeln!(self.body, "    let {}: [Field; {}] = [{}];", name, cells.len(), cells.join(", "));
        self.roms.insert(table, (name, cells.to_vec()));
        Ok(())
    }

    /// A read of the native array at an index `array_index` checks into the
    /// table (an address outside fails the proof, as the mock rejects it;
    /// it does not rely on bb's memory check, which the pinned bb version
    /// does not enforce for index p - 1); a constant address must be in
    /// range at compile time and reads its cell.
    fn rom_read(&mut self, table: crate::types::StmtId, address: &CellRef) -> Result<CellRef, ProvingError> {
        let (name, cells) = self
            .roms
            .get(&table)
            .cloned()
            .ok_or_else(|| ProvingError::synthesis(format!("statement {table} is not a table")))?;
        if let Some(a) = self.s_int_of(address) {
            return usize::try_from(a)
                .ok()
                .and_then(|a| cells.get(a).cloned())
                .ok_or_else(|| ProvingError::synthesis(format!("table read out of bounds: address {a}")));
        }
        if cells.is_empty() {
            return Err(ProvingError::synthesis("a read of an empty table"));
        }
        let index = self.array_index(address, cells.len() as u64);
        Ok(self.bind(&format!("{name}[{index}]")))
    }

    /// A mutable native array holding the cells (a RAM block in bb). The
    /// interpreter checks the RAM discipline, so every version of a memory
    /// is the array's state where it is read.
    fn ram_init(&mut self, memory: crate::types::StmtId, cells: &[CellRef]) -> Result<(), ProvingError> {
        if cells.is_empty() {
            return Err(ProvingError::synthesis("an empty memory"));
        }
        let name = self.fresh();
        let _ = writeln!(self.body, "    let mut {}: [Field; {}] = [{}];", name, cells.len(), cells.join(", "));
        self.rams.insert(memory, (name, cells.len()));
        Ok(())
    }

    /// `array[index] = value`, the index checked into the memory by
    /// `array_index` (or a constant in range).
    fn ram_write(&mut self, from: crate::types::StmtId, to: crate::types::StmtId, address: &CellRef, value: &CellRef) -> Result<(), ProvingError> {
        let (name, len) = self.ram_version(from)?;
        let index = self.ram_index(address, len)?;
        let _ = writeln!(self.body, "    {name}[{index}] = {value};");
        self.rams.insert(to, (name, len));
        Ok(())
    }

    /// `array[index]`, the index checked into the memory by `array_index`
    /// (or a constant in range). Not a cached binding: a later write changes
    /// the array.
    fn ram_read(&mut self, version: crate::types::StmtId, address: &CellRef) -> Result<CellRef, ProvingError> {
        let (name, len) = self.ram_version(version)?;
        let index = self.ram_index(address, len)?;
        let out = self.fresh();
        let _ = writeln!(self.body, "    let {out} = {name}[{index}];");
        Ok(out)
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

    fn poseidon_hash(&mut self, _inputs: &[CellRef]) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "poseidon_hash")
    }
    fn eq_hash(&mut self, _a: &CellRef, _b: &CellRef) -> Result<CellRef, ProvingError> {
        noop_unsupported!(self, "eq_hash")
    }

    /// `[(v1, i1) < (v2, i2)]` by one gap check of the packed keys
    /// `K = v·2^b + i`: the producer guarantees `i` in `[0, 2^b)`, so
    /// `K1 - K2 = (v1 - v2)·2^b + (i1 - i2)` has the sign of the
    /// lexicographic order (`|i1 - i2| < 2^b`), and `qlt_bits::<W>` pins
    /// that sign whenever `|K1 - K2| < 2^W` and fails the proof otherwise
    /// (as `gap_compare` does for values). `W` covers every pair of window
    /// values (`64 + b + 1` bits), rounded up to whole 14-bit limbs so bb
    /// builds no table of its own. Used only by asserts, the comparison is
    /// the check itself: `K2 - K1 - 1` in `[0, 2^W)`.
    fn key_less(
        &mut self,
        v1: &CellRef,
        i1: &CellRef,
        v2: &CellRef,
        i2: &CellRef,
        index_bits: u32,
    ) -> Result<CellRef, ProvingError> {
        let k1 = self.pack_key(v1, i1, index_bits)?;
        let k2 = self.pack_key(v2, i2, index_bits)?;
        self.packed_key_less(&k1, &k2, index_bits)
    }

    /// `v * 2^index_bits + i`; no check (`IR::PackKey`: the value lies in
    /// the window, as every Noir integer, and the index is a constant).
    fn pack_key(&mut self, v: &CellRef, i: &CellRef, index_bits: u32) -> Result<CellRef, ProvingError> {
        let scale = 1u128 << index_bits;
        let key = self.bind(&format!("{v} * {scale} + {i}"));
        self.packed_values.insert(key.clone(), v.clone());
        Ok(key)
    }

    /// `[K1 < K2]` for packed keys: `|K1 - K2| < 2^(64 + index_bits)`, so
    /// `qlt_bits` at `64 + index_bits + 1` bits (rounded up to whole 14-bit
    /// limbs) decides it, or, when asserted, one range check of the gap.
    /// Keys whose values have known intervals take the narrower width of
    /// `packed_gap_bits`.
    fn packed_key_less(&mut self, k1: &CellRef, k2: &CellRef, index_bits: u32) -> Result<CellRef, ProvingError> {
        let window = (64 + index_bits + 1).div_ceil(14) * 14;
        let narrow = self.packed_gap_bits(k1, k2, index_bits).filter(|&w| w < 64 + index_bits + 1);
        let asserted = self
            .current_stmt
            .is_some_and(|id| self.asserted_comparisons.contains(&id));
        if asserted {
            // `K2 - K1 - 1` in `[0, 2^w)` for any `w` at which the honest
            // gap fits: a negative gap is a field element of about 254 bits.
            let t = self.fresh();
            let _ = writeln!(self.body, "    let {t}: Field = {k2} - {k1} - 1;");
            match narrow {
                Some(bits) if needs_table_of_its_own(bits) => {
                    let id = self.deferred_widths.len();
                    let _ = writeln!(self.body, "    {t}.assert_max_bit_size::<__W{id}__>();");
                    self.deferred_widths.push(DeferredWidth { bits, cell: t, top: 0, exact_power: true, one_sided: true });
                }
                Some(bits) => {
                    let _ = writeln!(self.body, "    {t}.assert_max_bit_size::<{}>();", bits.max(8));
                }
                None => {
                    let _ = writeln!(self.body, "    {t}.assert_max_bit_size::<{window}>();");
                }
            }
            return Ok(self.literal_int(1));
        }
        // `qlt_bits::<W>` at the narrow widths whose tables bb shares (as
        // `comparison_width` picks them).
        let width = match narrow {
            Some(bits) if bits <= 8 => 8,
            Some(bits) if bits <= 28 => 28,
            _ => window,
        };
        self.use_lib("qlt_bits");
        let less = self.bind_typed(&format!("qlt_bits::<{width}>({k1}, {k2})"), "bool");
        Ok(self.bind_flag(&format!("{less} as Field")))
    }

    /// The index of a packed key, by `noir_fpc::packed_key_index` (a hinted
    /// decomposition pinned by range checks and a reconstruction).
    fn packed_key_index(&mut self, key: &CellRef, index_bits: u32) -> Result<CellRef, ProvingError> {
        self.use_lib("packed_key_index");
        Ok(self.bind(&format!("packed_key_index::<{index_bits}>({key})")))
    }

    /// A hint lane from `noir_fpc`'s unconstrained implementation of
    /// `kernel::hint_lane` (same order, same algorithm, same lanes). A
    /// constant rank indexes the hint's array, computed once per operand
    /// list (constant indices into the witnesses an unconstrained call
    /// returns cost no gates); a runtime rank calls the `_at` form, which
    /// returns the lane itself (a witness index would be a memory read).
    /// No constraint: the IR pins every hint it uses.
    fn prover_hint(
        &mut self,
        kind: crate::ir_defs::HintKind,
        operands: &[CellRef],
        rank: &CellRef,
    ) -> Result<CellRef, ProvingError> {
        use crate::ir_defs::HintKind;
        if kind == HintKind::PositionOneHot {
            return self.position_one_hot_hint(operands, rank);
        }
        if kind == HintKind::Negative {
            let [x] = operands else {
                return Err(ProvingError::synthesis("negative hint expects one operand"));
            };
            if self.s_int_of(rank) != Some(0) {
                return Ok(self.literal_int(0));
            }
            self.use_lib("negative_hint");
            return Ok(self.bind(&format!("negative_hint({x})")));
        }
        let (lanes_fn, lane_at_fn) = match kind {
            HintKind::StableArgsort => ("argsort_hint", "argsort_hint_at"),
            HintKind::SortedValues => ("sorted_hint", "sorted_hint_at"),
            HintKind::MaskedSortedValues => ("masked_sorted_hint", "masked_sorted_hint_at"),
            HintKind::PositionOneHot | HintKind::Negative => unreachable!("handled above"),
            HintKind::RoutingBits => {
                return Err(ProvingError::synthesis(
                    "routing-bit hints (the routed Waksman sort) are not emitted for Noir; the sort family offers that plan on halo2-base only",
                ));
            }
            HintKind::OrderedBy | HintKind::ProductSum => {
                return Err(ProvingError::synthesis(format!(
                    "{} hints are pinned by identity checks at verifier challenges, which Noir has not; \
                     the array lowering offers those constructions on halo2-base only",
                    kind.name()
                )));
            }
            HintKind::CountAtMost => {
                self.use_lib("count_at_most_hint");
                return Ok(self.bind(&format!("count_at_most_hint([{}], {rank})", operands.join(", "))));
            }
        };
        // A masked hint takes the values and their activity flags as two
        // arrays; its lanes are as many as the values.
        let (operand_list, lanes) = match kind {
            HintKind::MaskedSortedValues => {
                let (values, active) = operands.split_at(operands.len() / 2);
                (format!("{}], [{}", values.join(", "), active.join(", ")), values.len())
            }
            _ => (operands.join(", "), operands.len()),
        };
        if let Some(rank) = self.s_int_of(rank) {
            let Some(lane) = usize::try_from(rank).ok().filter(|&r| r < lanes) else {
                return Ok(self.literal_int(0));
            };
            let key = format!("{lanes_fn}([{operand_list}])");
            let array = match self.hint_arrays.get(&key) {
                Some(name) => name.clone(),
                None => {
                    self.use_lib(lanes_fn);
                    let name = self.fresh();
                    let _ = writeln!(self.body, "    let {}: [Field; {}] = {};", name, lanes, key);
                    self.hint_arrays.insert(key, name.clone());
                    name
                }
            };
            return Ok(self.bind(&format!("{array}[{lane}]")));
        }
        self.use_lib(lane_at_fn);
        Ok(self.bind(&format!("{lane_at_fn}([{operand_list}], {rank})")))
    }
}

// ---------------------------------------------------------------------------
// Rendering — assemble main.nr from emitted body + collected exposes
// ---------------------------------------------------------------------------

/// Render `main.nr`. Returns the source text; `main` returns the exposed
/// values in IR order (plus, for a circuit with no output and no body, one
/// synthetic return value; see below).
#[cfg(test)]
pub(super) use self::{settle_range_widths as settle_range_widths_for_tests, DeferredWidth as DeferredWidthForTests};

/// A range check `0 <= x < top + 1` emitted with its width left open
/// (`NoirEmitter::assert_index_in_range`).
pub(super) struct DeferredWidth {
    /// The exact width, `bit_length(top)`.
    pub(super) bits: u32,
    /// The checked cell.
    pub(super) cell: String,
    pub(super) top: i128,
    /// `top + 1 == 2^bits`: one check of the exact width suffices.
    pub(super) exact_power: bool,
    /// Only `0 <= x` matters (a gap check): any width at least `bits`
    /// admits the honest value, so a rounded width needs no second check.
    pub(super) one_sided: bool,
}

/// Whether a range check of `bits` bits needs a lookup table that no
/// usual check shares. bb checks a value of more than 14 bits in 14-bit
/// limbs plus a final limb of the remaining bits, and a value of at most 14
/// bits against a table of that width; each width's table costs about
/// `2^w / 6` gates once (measured: 8 bits 43, 11 bits 340, 13 bits 1,350,
/// 14 bits 2,700). Every circuit with a 64-bit check (`4 * 14 + 8`) has the
/// 14-bit and the 8-bit tables; widths up to 8 need small ones.
fn needs_table_of_its_own(bits: u32) -> bool {
    bits > 8 && bits % 14 != 0 && bits < 64
}

/// The body with every deferred range check's width settled: when the
/// circuit already uses the 14-bit table (`uses_wide_table`), the check is
/// made at the next multiple of 14 bits, on the value and on `top - value`
/// (each check fails for a value outside `[0, top]`: a negative one is a
/// field element of about 254 bits); otherwise at its exact width, as
/// before (one check when `top + 1` is a power of two). Both forms admit
/// exactly `[0, top]`; the first adds no table.
pub(super) fn settle_range_widths(body: &str, deferred: &[DeferredWidth]) -> String {
    if deferred.is_empty() {
        return body.to_string();
    }
    let wide = uses_wide_table(body);
    // Per exact width: round its checks up when the table it would add costs
    // more than the second checks rounding adds (one per check whose bound
    // is a power of two; a check of another bound has two either way), and
    // the body has no check of that width already (its table is paid).
    let rounded = |bits: u32| {
        let own_table_cost = 1u64 << bits.saturating_sub(3);
        let extra_checks = deferred.iter().filter(|d| d.bits == bits && d.exact_power).count() as u64;
        wide && extra_checks < own_table_cost && !body.contains(&format!("assert_max_bit_size::<{bits}>"))
    };
    let width = |d: &DeferredWidth| if rounded(d.bits) { d.bits.div_ceil(14) * 14 } else { d.bits };
    let mut out = String::with_capacity(body.len());
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(id) = trimmed.strip_prefix("__TOP").and_then(|r| r.strip_suffix("__;")).and_then(|n| n.parse::<usize>().ok()) {
            let d = &deferred[id];
            let width = width(d);
            if !d.one_sided && (width != d.bits || !d.exact_power) {
                let _ = writeln!(out, "    let {}_top: Field = {} - {};", d.cell, d.top, d.cell);
                let _ = writeln!(out, "    {}_top.assert_max_bit_size::<{}>();", d.cell, width);
            }
            continue;
        }
        if let Some(start) = line.find("::<__W") {
            let rest = &line[start + 6..];
            if let Some(end) = rest.find("__>") {
                if let Ok(id) = rest[..end].parse::<usize>() {
                    let _ = writeln!(out, "{}::<{}>{}", &line[..start], width(&deferred[id]), &rest[end + 3..]);
                    continue;
                }
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Whether the circuit range-checks some value of 14 bits or more, which
/// makes bb build its 14-bit table: a literal width of 14 or more in the
/// body (`assert_max_bit_size`, `qlt_bits`), a checked conversion to 16 bits
/// or more, or a call of a library function that does either
/// ([`wide_library_functions`]).
fn uses_wide_table(body: &str) -> bool {
    let wide_literal = |marker: &str| {
        body.match_indices(marker).any(|(at, _)| {
            let digits: String = body[at + marker.len()..].chars().take_while(char::is_ascii_digit).collect();
            digits.parse::<u32>().is_ok_and(|w| w >= 14)
        })
    };
    if wide_literal("assert_max_bit_size::<") || wide_literal("qlt_bits::<") {
        return true;
    }
    if ["checked_u16(", "checked_u32(", "checked_u64("].iter().any(|c| body.contains(c)) {
        return true;
    }
    wide_library_functions().iter().any(|name| {
        body.match_indices(name.as_str()).any(|(at, _)| {
            let before = body[..at].chars().next_back();
            let after = body[at + name.len()..].chars().next();
            !before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') && after == Some('(')
        })
    })
}

/// The functions of `noir_fpc` that range-check a value at 14 bits or more:
/// a literal width of 14 or more, a checked conversion to 16 bits or more,
/// or a call of such a function (read off the library's source once).
fn wide_library_functions() -> &'static Vec<String> {
    static WIDE: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    WIDE.get_or_init(|| {
        let source = include_str!("../../../benchmarking/noir_fpc/src/lib.nr");
        // (name, body) of every function, a body running to the next `fn`.
        let mut functions: Vec<(String, String)> = Vec::new();
        let starts: Vec<usize> = source.match_indices("fn ").map(|(at, _)| at).collect();
        for (k, &at) in starts.iter().enumerate() {
            let head = &source[at + 3..];
            let name: String = head.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            if name.is_empty() {
                continue;
            }
            let end = starts.get(k + 1).copied().unwrap_or(source.len());
            functions.push((name, source[at..end].to_string()));
        }
        let literal_wide = |text: &str| {
            text.match_indices("assert_max_bit_size::<").any(|(at, m)| {
                let digits: String = text[at + m.len()..].chars().take_while(char::is_ascii_digit).collect();
                digits.parse::<u32>().is_ok_and(|w| w >= 14)
            }) || ["checked_u16(", "checked_u32(", "checked_u64("].iter().any(|c| text.contains(c))
        };
        let mut wide: Vec<String> = functions.iter().filter(|(_, b)| literal_wide(b)).map(|(n, _)| n.clone()).collect();
        loop {
            let more: Vec<String> = functions
                .iter()
                .filter(|(n, _)| !wide.contains(n))
                .filter(|(_, b)| wide.iter().any(|w| b.contains(&format!("{w}("))))
                .map(|(n, _)| n.clone())
                .collect();
            if more.is_empty() {
                break wide;
            }
            wide.extend(more);
        }
    })
}

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
    let pruned_body = settle_range_widths(&dce_body(&emitter.body, &emitter.exposes), &emitter.deferred_widths);
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

/// Whether `expr` is `x as Field` for one identifier or one indexed
/// identifier `x`.
fn is_single_cast(expr: &str) -> bool {
    expr.strip_suffix(" as Field").is_some_and(|x| {
        !x.is_empty() && x.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '[' | ']'))
    })
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

    // These tests inspect emitted text/metadata only: no constraint or
    // witness execution is involved.
    #[test]
    fn primitive_result_facts_sqrt_reuses_sign_and_window() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let x = "x".to_string();
        let root = em.sqrt_f(&x).unwrap();
        assert_eq!(em.known_range(&root), Some((0, (1i128 << 48) - 1)));
        assert_eq!(em.body.matches("= sign_abs(x)").count(), 1);
        // The original input still goes through sign handling/clamping.
        assert!(!em.body.contains("qsqrt_of(false, x)"));
        em.nonzero.insert(root.clone());
        em.div_f(&"numerator".to_string(), &root).unwrap();
        em.arctan2_f(&root, &root).unwrap();
        assert!(em.body.contains("divmod_by_positive::<64>"), "{}", em.body);
        assert_eq!(em.body.matches("= sign_abs(").count(), 1, "{}", em.body);
        assert!(!em.body.contains("assert_in_window"), "{}", em.body);
    }

    #[test]
    fn primitive_result_facts_sqrt_requires_checked_input_range() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        em.interval_of.insert("unverified".to_string(), (0, 100));
        em.sqrt_f(&"unverified".to_string()).unwrap();
        assert!(em.body.contains("sign_abs(unverified)"));
        em.narrow_proven_range("checked", 0, (1i128 << 64) - 1);
        let root = em.sqrt_f(&"checked".to_string()).unwrap();
        assert!(em.body.contains("qsqrt_of(false, checked)"));
        assert_eq!(em.known_range(&root), Some((0, (1i128 << 48) - 1)));
    }

    #[test]
    fn primitive_result_facts_narrow_root_does_not_create_shifted_window() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let root = em.sqrt_f(&"x".to_string()).unwrap();
        let epsilon = em.constant_float(0.000001).unwrap();
        em.lt_f(&root, &epsilon).unwrap();
        assert!(!em.window_of.contains_key(&root));
        assert!(!em.body.contains("as u64"), "{}", em.body);
        assert!(em.body.contains("qlt_bits"), "{}", em.body);
    }

    #[test]
    fn primitive_result_facts_rounded_product_reuses_window() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let product = em.mul_f(&"x".to_string(), &"y".to_string()).unwrap();
        let zero = em.constant_int(0).unwrap();
        em.gt_f(&product, &zero).unwrap();
        em.lte_f(&product, &zero).unwrap();
        assert!(em.body.contains("as u64) > 9223372036854775808"), "{}", em.body);
        assert!(!em.body.contains("qlt_bits") && !em.body.contains("sign_abs"), "{}", em.body);
        assert!(!em.body.contains("assert_in_window"), "{}", em.body);
    }

    #[test]
    fn primitive_result_facts_nonzero_threshold_keeps_complement_lowering() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let product = em.mul_f(&"x".to_string(), &"y".to_string()).unwrap();
        let threshold = em.constant_int(7).unwrap();
        em.gt_f(&product, &threshold).unwrap();
        em.lte_f(&product, &threshold).unwrap();
        assert_eq!(em.body.matches("as u64) < 9223372036854775816").count(), 1, "{}", em.body);
        assert!(!em.body.contains("as u64) >"), "{}", em.body);
    }

    #[test]
    fn primitive_result_facts_exact_products_do_not_gain_windows() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let integer = em.float_cast(&"n".to_string()).unwrap();
        let x = "x".to_string();
        let exact = em.mul_f(&integer, &x).unwrap();
        assert_eq!(em.known_range(&exact), None);
        let zero = em.constant_int(0).unwrap();
        em.gt_f(&exact, &zero).unwrap();
        assert!(em.body.contains("qlt_bits::<64>"), "{}", em.body);
        assert!(!em.body.contains("as u64"), "{}", em.body);
        let y = "y".to_string();
        let rounded = em.product_sum_f(&[(x.clone(), y.clone())]).unwrap();
        assert_eq!(em.known_range(&rounded), Some((-(1i128 << 63), (1i128 << 63) - 1)));
        let mixed = em.product_sum_f(&[(integer, x.clone()), (x, y)]).unwrap();
        assert_eq!(em.known_range(&mixed), None);
    }

    #[test]
    fn primitive_result_facts_boolean_idempotence_keeps_truth_conversion() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = bare_emitter(&prov);
        let x = "x".to_string();
        let and = em.logical_and(&x, &x).unwrap();
        let or = em.logical_or(&x, &x).unwrap();
        assert_eq!(and, or);
        assert_ne!(and, x);
        assert_eq!(em.body.matches("if x == 0 { 0 } else { 1 }").count(), 1);
        assert!(!em.body.contains(" * "), "{}", em.body);
        let before = em.body.clone();
        assert_eq!(em.logical_and(&and, &and).unwrap(), and);
        assert_eq!(em.logical_or(&or, &or).unwrap(), or);
        assert_eq!(em.body, before);
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


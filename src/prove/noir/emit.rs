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
            let _ = writeln!(out, "{} = {}", name, v);
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
pub struct NoirEmitter {
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
    /// Variable-name -> known concrete value, when we constant-folded the
    /// expression at emit time. Used for public-input bookkeeping only;
    /// the proof itself derives values from witness execution by nargo.
    value_cache: HashMap<String, ConcreteValue>,
    /// Peephole provenance for the negative-index normalization pattern
    /// `select(lt(x, 0), x + N, x)`: maps lt(x,0) outputs to x, add-const
    /// outputs to (x, c), and recognized normalized indices to (x, N).
    /// `eq_i(norm, j)` then emits the signed-machinery-free identity
    /// `(x == j) | (x == j - N)` — exact for every x.
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
    /// SSA names that were bound from an IR-level *constant* op
    /// (`ConstantInt` / `ConstantFloat` / `ConstantBool`). Distinguishes
    /// "literal known at circuit-build time" from "value happens to be
    /// in the value_cache because the resolved witness was propagated"
    /// — the latter must NEVER be inlined into the circuit, or the
    /// circuit becomes witness-specific (unsound for any other witness).
    literal_cells: HashSet<String>,
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

impl NoirEmitter {
    fn add_i_inner(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x + y);
        // u32 path: both operands already bound u32 AND sum still fits.
        // Bound-propagation lets axis-dim chains (`row_has`, sums, scan
        // accumulators) stay u32 without any RANGE cast.
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
    ) -> Self {
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
            inputs,
            witness,
            body: String::new(),
            next_id: 0,
            used_libs: BTreeMap::new(),
            exposes: Vec::new(),
            public_decimal: Vec::new(),
            value_cache: HashMap::new(),
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
            literal_cells: HashSet::new(),
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
        self.bind_u32_with_bound(&format!("{} as u32", cell), bound)
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
        let _ = writeln!(self.body, "    let {} = {};", v, expr);
        self.cse_cache.insert(expr.to_string(), v.clone());
        // Forward-annotate the IR-analysis bound for the current
        // statement so downstream emit-time u32-path conditions see
        // the richer info even when this method chose Field.
        self.annotate_with_analysis_bound(&v);
        v
    }

    fn bind_val(&mut self, expr: &str, val: ConcreteValue) -> String {
        let v = self.bind(expr);
        self.value_cache.insert(v.clone(), val);
        v
    }

    fn float_of(&self, var: &str) -> Option<f64> {
        match self.value_cache.get(var)? {
            ConcreteValue::Float(f) => Some(*f),
            ConcreteValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    fn int_of(&self, var: &str) -> Option<i128> {
        match self.value_cache.get(var)? {
            ConcreteValue::Int(i) => Some(*i),
            ConcreteValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }
    fn bool_of(&self, var: &str) -> Option<bool> {
        match self.value_cache.get(var)? {
            ConcreteValue::Bool(b) => Some(*b),
            ConcreteValue::Int(i) => Some(*i != 0),
            _ => None,
        }
    }
    fn fold_f(&mut self, v: &str, val: Option<f64>) {
        if let Some(f) = val {
            self.value_cache
                .insert(v.to_string(), ConcreteValue::Float(f));
        }
    }
    fn fold_i(&mut self, v: &str, val: Option<i128>) {
        if let Some(i) = val {
            self.value_cache
                .insert(v.to_string(), ConcreteValue::Int(i));
        }
    }
    fn fold_b(&mut self, v: &str, val: Option<bool>) {
        if let Some(b) = val {
            self.value_cache
                .insert(v.to_string(), ConcreteValue::Bool(b));
        }
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
        // pair without going through `to_signed`. **Gated on
        // `literal_cells`** — value_cache also holds witness-resolved
        // values propagated by `bind_val` during emit (e.g. the result
        // of `add_i` on two constant args stores its computed value).
        // Inlining a witness-derived value into the circuit would
        // specialise it to one witness — *unsound* for any other
        // input. `literal_cells` is populated only by `constant_int`/
        // `constant_float`/`constant_bool`, where the value really is
        // a compile-time constant.
        //
        //   - Int(n) literal: magnitude = n (Field value already = n)
        //   - Bool(b) literal: magnitude = 0 or 1
        //   - Float(f) literal: magnitude = f * 2^32 (Q-scaling — the
        //     Field value is the quantized form). Negatives skipped;
        //     they need a signed-Field wrap that the runtime
        //     `to_signed` handles cleanly.
        let folded_magnitude: Option<i128> = if self.literal_cells.contains(raw) {
            match self.value_cache.get(raw) {
                Some(ConcreteValue::Int(n)) if *n >= 0 => Some(*n),
                Some(ConcreteValue::Bool(b)) => Some(if *b { 1 } else { 0 }),
                Some(ConcreteValue::Float(f)) if *f >= 0.0 && f.is_finite() => {
                    let scaled = (*f * (Q_U64 as f64)).round();
                    if scaled >= 0.0 && scaled < (1i128 << 126) as f64 {
                        Some(scaled as i128)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some(v) = folded_magnitude {
            if v >= 0 && v < (1i128 << 126) {
                let s_lit = "false".to_string();
                let m_lit = v.to_string();
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

fn input_index_expr(path: &InputPath, inputs: &[EmittedInput]) -> String {
    let kind = inputs
        .iter()
        .find(|i| i.name == path.param)
        .map(|i| &i.kind);
    match kind {
        Some(InputKind::ScalarField) | Some(InputKind::Bool) | None => path.param.clone(),
        Some(InputKind::ArrayField { .. }) => {
            // Compute the flat index from the segments.
            let mut flat: u64 = 0;
            for s in &path.segments {
                if let PathSegment::Index(i) = s {
                    flat = flat.saturating_mul(1).saturating_add(*i as u64);
                }
            }
            // Single-axis case: just use the first Index segment.
            if path.segments.len() == 1 {
                if let Some(PathSegment::Index(i)) = path.segments.first() {
                    return format!("{}[{}]", path.param, i);
                }
            }
            // Multi-axis: the witness Array we get is already flat-leaves,
            // so the flat index is just the LAST Index segment + rolling
            // sum scaled by per-segment widths — but Zinnia's IR-gen
            // already flattens path.segments to a single Index for
            // NDArray reads on most kernels we care about. As a fallback,
            // use the flat composition above.
            format!("{}[{}]", path.param, flat)
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

impl Synthesizer for NoirEmitter {
    type CellRef = CellRef;

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
            self.value_cache
                .insert(v.clone(), ConcreteValue::Int(value as i128));
            self.literal_cells.insert(v.clone());
            return Ok(v);
        }
        let v = self.bind_val(&expr, ConcreteValue::Int(value as i128));
        self.literal_cells.insert(v.clone());
        Ok(v)
    }

    fn constant_float(&mut self, value: f64) -> Result<Self::CellRef, ProvingError> {
        let expr = quantize_to_field_string(value);
        let v = self.bind_val(&expr, ConcreteValue::Float(value));
        self.literal_cells.insert(v.clone());
        Ok(v)
    }

    fn constant_bool(&mut self, value: bool) -> Result<Self::CellRef, ProvingError> {
        // Bool constants are also typed as Field for our purposes — the
        // logical-not / and / or routes treat 0/1 in Field.
        let expr = if value { "1" } else { "0" };
        let v = self.bind_val(expr, ConcreteValue::Bool(value));
        self.literal_cells.insert(v.clone());
        Ok(v)
    }

    fn add_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let out = self.add_i_inner(a, b)?;
        if let Some(c) = self.int_of(b) {
            self.add_const_of.insert(out.clone(), (a.clone(), c as i64));
        } else if let Some(c) = self.int_of(a) {
            self.add_const_of.insert(out.clone(), (b.clone(), c as i64));
        }
        return Ok(out);
    }

    fn sub_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded = self.int_of(a).zip(self.int_of(b)).map(|(x, y)| x - y);
        // u32 path is only sound if we can prove the result is
        // non-negative — u32 subtraction wraps on underflow. The
        // strongest evidence we have is folded value: if the witness
        // sample is non-negative AND both operands are u32, take the
        // path. Otherwise fall back to Field (signed-encoded).
        if self.is_u32(a) && self.is_u32(b) && folded.map_or(false, |v| v >= 0) {
            let bound = self.bound_of(a).unwrap_or(u32::MAX as u64);
            let v = self.bind_u32_with_bound(&format!("{} - {}", a, b), bound);
            self.fold_i(&v, folded);
            return Ok(v);
        }
        let v = self.bind(&format!("{} - {}", self.as_field(a), self.as_field(b)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn mul_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
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
        // Witnessed-quotient pattern via noir_fpc::q_floor_div_hint.
        // Hint runs unconstrained; downstream IR-gen pins the quotient.
        self.use_lib("q_floor_div_hint");
        let v = self.bind(&format!(
            "q_floor_div_hint({}, {})",
            self.as_field(a),
            self.as_field(b)
        ));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn mod_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
        let folded =
            self.int_of(a)
                .zip(self.int_of(b))
                .and_then(|(x, y)| if y == 0 { None } else { Some(x % y) });
        if self.is_u32(a) && self.is_u32(b) {
            let bound = self
                .bound_of(b)
                .unwrap_or(u32::MAX as u64)
                .saturating_sub(1);
            let v = self.bind_u32_with_bound(&format!("{} % {}", a, b), bound);
            self.fold_i(&v, folded);
            return Ok(v);
        }
        self.use_lib("q_mod_hint");
        let v = self.bind(&format!(
            "q_mod_hint({}, {})",
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
        Ok(self.bind(&format!("qabs({})", self.as_field(a))))
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
        // eq(norm, j) == (x == j) | (x == j - N) for every x — two Field
        // equalities instead of the signed-comparison machinery that the
        // normalization otherwise drags in (~600 ACIR gates per index).
        {
            let hit = self
                .norm_index_of
                .get(a)
                .cloned()
                .map(|nx| (nx, self.int_of(b)))
                .filter(|(_, j)| j.is_some())
                .or_else(|| {
                    self.norm_index_of
                        .get(b)
                        .cloned()
                        .map(|nx| (nx, self.int_of(a)))
                        .filter(|(_, j)| j.is_some())
                });
            if let Some(((x, n), Some(j))) = hit {
                let xf = self.as_field(&x);
                let j = j as i64;
                let second = if j - n >= 0 {
                    format!("({} == {})", xf, j - n)
                } else {
                    format!("({} + {} == 0)", xf, n - j)
                };
                let v = self.bind(&format!(
                    "if ({} == {}) | {} {{ 1 }} else {{ 0 }}",
                    xf, j, second
                ));
                self.fold_b(&v, folded);
                return Ok(v);
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
        let out = self.lt_i_inner(a, b)?;
        if self.int_of(b) == Some(0) {
            self.lt_zero_of.insert(out.clone(), a.clone());
        }
        return Ok(out);
    }

    fn lt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
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
    fn gt_i_bounded(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
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
    fn gte_i(&mut self, a: &CellRef, b: &CellRef) -> Result<CellRef, ProvingError> {
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
        Ok(self.bind(&format!("qsqrt({})", self.as_field(a))))
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
        let is_zero_int = self.int_of(if_false) == Some(0);
        let is_zero_flt = self.float_of(if_false) == Some(0.0);
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
        // Fold value through cond. We propagate either side based on
        // concrete cond; falls back to no-fold if cond unknown.
        if let Some(c) = self.bool_of(cond) {
            let picked = if c { if_true } else { if_false };
            if let Some(cv) = self.value_cache.get(picked).cloned() {
                self.value_cache.insert(v.clone(), cv);
            }
        }
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
        // Use u128 floor division (signed values are wrapped Field elements
        // so this matches Zinnia's signed Q-floor semantic for the kernels
        // in scope; full negative-aware floor needs a sign-aware path).
        self.use_lib("q_div_q_hint");
        let folded = self.float_of(a).map(|f| (f / Q_U64 as f64) as i128);
        let v = self.bind(&format!("q_div_q_hint({})", self.as_field(a)));
        self.fold_i(&v, folded);
        Ok(v)
    }
    fn float_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Integer-to-float in Q32 means *= 2^32.
        Ok(self.bind(&format!("{} * {}", self.as_field(a), Q_U64)))
    }
    fn bool_cast(&mut self, a: &CellRef) -> Result<CellRef, ProvingError> {
        // Treat any nonzero as 1 — we lack a generic field-to-bool. For
        // the kernels in scope (0/1 already), identity suffices.
        Ok(a.clone())
    }

    fn read_input(&mut self, path: &InputPath, _is_public: bool) -> Result<CellRef, ProvingError> {
        let expr = input_index_expr(path, &self.inputs);
        let v = self.bind(&expr);
        if let Some((_decimal, concrete)) = self.resolve_path_concrete(path) {
            self.value_cache.insert(v.clone(), concrete);
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
        // Compute the public-input decimal-field representation if we
        // can constant-fold.
        let dec = match self.value_cache.get(a) {
            Some(ConcreteValue::Int(v)) => signed_to_field_decimal(*v),
            Some(ConcreteValue::Float(v)) => quantize_to_field_string(*v),
            Some(ConcreteValue::Bool(v)) => {
                if *v {
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
        let a = self.int_of(addr).unwrap_or(0) as usize;
        let mem = self.memories.entry(segment_id).or_default();
        if a >= mem.len() {
            // Resize to mirror the halo2 mock behaviour: writes beyond
            // the allocated region are silently bounded.
            return Ok(());
        }
        mem[a] = value.clone();
        Ok(())
    }
    fn read_memory(&mut self, segment_id: u32, addr: &CellRef) -> Result<CellRef, ProvingError> {
        let a = self.int_of(addr).unwrap_or(0) as usize;
        let init = self
            .memory_init
            .get(&segment_id)
            .cloned()
            .unwrap_or_else(|| "0".to_string());
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
            return Ok(expr);
        }
        let prior_val = self.value_cache.get(&expr).cloned();
        let v = self.bind(&expr);
        if let Some(val) = prior_val {
            self.value_cache.insert(v.clone(), val);
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
    fn assert_dynamic_ndarray_meta(
        &mut self,
        _array_id: u32,
        _max_rank: u32,
        _max_length: u32,
        _args: &[CellRef],
    ) -> Result<(), ProvingError> {
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

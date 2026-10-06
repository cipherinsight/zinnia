use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::circuit_input::InputPath;

// ---------------------------------------------------------------------------
// IR enum — all 78 IR instruction types as a single Rust enum
// ---------------------------------------------------------------------------

/// The unified IR instruction enum. Each variant corresponds to one of the
/// 78 Python IR classes (e.g., `ConstantIntIR`, `AddIIR`, etc.).
///
/// Variants that carry no data are unit variants. Variants that carry
/// IR-specific parameters use named fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IR {
    // ── Constants ──────────────────────────────────────────────────────
    ConstantInt {
        value: i64,
    },
    ConstantFloat {
        value: f64,
    },
    ConstantBool {
        value: bool,
    },
    ConstantStr {
        value: String,
    },

    // ── Integer arithmetic ────────────────────────────────────────────
    AddI,
    SubI,
    MulI,
    DivI,
    FloorDivI,
    /// Euclidean floor division, semantically identical to [`IR::FloorDivI`],
    /// but additionally certifying that the quotient, remainder, and divisor
    /// all fit in `divisor_bits` bits — a witness-INDEPENDENT structural fact
    /// the producer guarantees (e.g. a reshape `-1` dim: the divisor is a
    /// product of dims that divides the source total, so it is bounded by the
    /// total). Backends that only care about the value treat this exactly like
    /// `FloorDivI`; the Noir backend uses the bound to emit a cheap
    /// bit-decomposition-range-checked division (`q_floor_div_bounded`) instead
    /// of the general `q_floor_div`'s two ~126-bit `to_signed` projections.
    /// Soundness is NOT weakened: the bit-size checks still pin a unique
    /// Euclidean quotient and reject any out-of-range (e.g. field-inverse)
    /// substitution.
    FloorDivIPinned {
        divisor_bits: u32,
    },
    ModI,
    PowI,
    AbsI,
    SignI,
    InvI,

    // ── Integer bitwise ───────────────────────────────────────────────
    BitAndI,
    BitOrI,
    BitXorI,
    ShlI,
    ShrI,
    BitNotI,

    // ── Float arithmetic ──────────────────────────────────────────────
    AddF,
    /// Static float sum aggregate.
    ///
    /// Arguments are same-scale fixed-point float operands. Semantics are
    /// exactly repeated `AddF`, but backends may lower small chunks to a
    /// denser native addition gate.
    SumF {
        len: u32,
    },
    SubF,
    MulF,
    /// Static float product-sum aggregate.
    ///
    /// Arguments are `2 * len` scalar float operands in adjacent pair order:
    /// `[a0, b0, a1, b1, ...]`. Semantics are exactly the scalar fixed-point
    /// path per pair:
    /// `sum_i fp_mul_rescale(ai, bi, precision_bits)`, accumulated with
    /// existing float-add semantics. This is not raw field `mul_add`.
    ProductSumF {
        len: u32,
    },
    DivF,
    FloorDivF,
    ModF,
    PowF,
    /// Float base raised to an Integer exponent. The exponent stays in
    /// integer representation (not Q-scaled) so backends can lower to
    /// integer-exponent square-and-multiply instead of `exp(b*log(a))`.
    PowFI,
    AbsF,
    SignF,

    // ── Integer comparison ────────────────────────────────────────────
    EqI,
    NeI,
    /// `lt_i` specialized for the case where BOTH operands are provably
    /// non-negative and fit in u128 (e.g. internally-generated slice
    /// indices, axis lengths, row/col coords). Lowers to a single ACIR
    /// constraint via Noir's native `<` on u128, rather than the full
    /// sign-aware `qlt` decomposition. Caller is responsible for the
    /// bound; mis-use produces wrong answers on Zinnia's signed-Field
    /// negative encoding.
    LtIBounded,
    /// `gt_i` companion to `LtIBounded` — same bounded-operand contract.
    GtIBounded,
    /// Single-shot `0 <= idx < hi` discharge. Noir lowers to one
    /// `assert((idx as u128) < (hi as u128))` (~30 ACIR), saving the
    /// `qge(idx, 0) && qlt(idx, hi) && assert` chain (~49 ACIR). Sound
    /// because the Noir `as u128` cast emits an implicit `< 2^128` range
    /// constraint that catches Zinnia's signed-Field negative encoding
    /// (negatives are encoded as `p - |x|`, well above `2^128`). Used by
    /// `discharge_index_in_range` for the common `lo == 0` case.
    AssertU128Lt,
    /// Value-producing bounded-index decomposition bit. Returns bit `bit`
    /// of `idx`, while also constraining `0 <= idx < hi`. Multiple bits for
    /// the same `(idx, hi, bits)` may be fused/cached by backends.
    AssertU128LtBit {
        hi: u64,
        bit: u32,
        bits: u32,
    },
    /// Multi-axis bounded-range mask. Returns Field 0/1 = 1 iff for
    /// every axis k: `lo_k <= idx_k < hi_k`. `arguments` is laid out as
    /// `[idx_0, lo_0, hi_0, idx_1, lo_1, hi_1, ...]` for `rank` axes.
    /// Same operand contract as `LtIBounded`: every operand must be a
    /// non-negative u128-safe value (slice indices, axis lengths). The
    /// Noir backend lowers to one combined boolean expression sharing
    /// a single u128 RANGE table — replaces the multi-multiplication
    /// AND chain a generic `LtIBounded + LogicalAnd` lowering would
    /// emit, saving ~3 gates per axis per cell on 2D dynamic slices.
    IndexInBox {
        rank: usize,
    },
    LtI,
    LteI,
    GtI,
    GteI,

    // ── Float comparison ──────────────────────────────────────────────
    EqF,
    NeF,
    LtF,
    LteF,
    GtF,
    GteF,

    // ── Math functions (float) ────────────────────────────────────────
    SinF,
    SinHF,
    CosF,
    CosHF,
    TanF,
    TanHF,
    SqrtF,
    ExpF,
    LogF,
    ArcSinF,
    ArcCosF,
    ArcTan2F,

    // ── Logical ───────────────────────────────────────────────────────
    LogicalAnd,
    LogicalOr,
    LogicalNot,

    // ── Selection (conditional) ───────────────────────────────────────
    SelectI,
    SelectF,
    SelectB,
    /// Select where the condition is also constrained to be a bit.
    /// Backends may fuse `assert_bit(cond)` and the mux equation.
    SelectBitI,
    SelectBitF,
    /// Lane of a vector select with one shared condition:
    /// `out[lane] = cond ? true_values[lane] : false_values[lane]`.
    SelectManyI {
        len: u32,
        lane: u32,
    },
    SelectManyF {
        len: u32,
        lane: u32,
    },

    // ── Casting ───────────────────────────────────────────────────────
    IntCast,
    FloatCast,
    BoolCast,

    // ── String operations ─────────────────────────────────────────────
    AddStr,
    StrI,
    StrF,

    // ── I/O ───────────────────────────────────────────────────────────
    ReadInteger {
        path: InputPath,
        is_public: bool,
    },
    ReadFloat {
        path: InputPath,
        is_public: bool,
    },
    ReadHash {
        path: InputPath,
        is_public: bool,
    },
    /// Read an external function result (resolved during preprocessing).
    ReadExternalResult {
        store_idx: u32,
        output_idx: u32,
        is_float: bool,
    },
    Print,

    // ── Assertions & public exposure ──────────────────────────────────
    Assert,
    AssertBit,
    /// Assert the (signed-i64-domain) operand is non-negative. Backends
    /// may lower this far cheaper than a general comparison (a single
    /// 63-bit range check in halo2-base).
    AssertNonNegI,
    ExposePublicI,
    ExposePublicF,

    // ── Structural-predicate precondition (inert) ─────────────────────
    // Compile-time fact emitted by @requires / @zk_circuit(requires=[...]).
    // No stack operands, no result; later SMT-foundation cards consume the
    // payload. Treated as `fixed` (never DCE'd) so the surface card's
    // IR-dump round-trip can observe it.
    StructuralPredicate {
        kind: String,
        args: Vec<String>,
        op: Option<String>,
        bound: Option<String>,
    },

    // ── Scalar precondition (inert) ───────────────────────────────────
    // Compile-time fact carrying a serialized `ContractTerm` payload.
    // Emitted by ir-gen for scalar / arithmetic / logical preconditions
    // (`@requires(lambda k: 0 <= k <= 1024)` etc.). The discharger
    // lowers the term via `formula::lower_bool` and asserts on the Z3
    // solver; the contract-to-IR lowering via
    // `witness::lower_precondition_to_ir` wraps the result in `IR::Assert`.
    // The IR atom itself is inert (no stack args, no result) and
    // `is_fixed` so DCE leaves it alone.
    ScalarPrecondition {
        term_json: String,
    },

    // ── Memory operations ─────────────────────────────────────────────
    AllocateMemory {
        segment_id: u32,
        size: u32,
        init_value: i64,
    },
    WriteMemory {
        segment_id: u32,
    },
    ReadMemory {
        segment_id: u32,
    },
    MemoryTraceEmit {
        segment_id: u32,
        is_write: bool,
    },
    MemoryTraceSeal,

    // ── Dynamic NDArray ───────────────────────────────────────────────
    AllocateDynamicNDArrayMeta {
        array_id: u32,
        dtype_name: String,
        max_length: u32,
        max_rank: u32,
        #[serde(default)]
        max_shape: Vec<u32>,
    },
    WitnessDynamicNDArrayMeta {
        array_id: u32,
        max_rank: u32,
        param_name: String,
    },
    AssertDynamicNDArrayMeta {
        array_id: u32,
        max_rank: u32,
        max_length: u32,
    },
    DynamicNDArrayGetItem {
        array_id: u32,
        segment_id: u32,
    },
    DynamicNDArraySetItem {
        array_id: u32,
        segment_id: u32,
    },

    // ── External function calls ───────────────────────────────────────
    InvokeExternal {
        store_idx: u32,
        func_name: String,
        /// Serialized DTDescriptor dicts for positional args
        args: Vec<serde_json::Value>,
        /// Serialized DTDescriptor dicts for keyword args
        kwargs: HashMap<String, serde_json::Value>,
    },
    ExportExternalI {
        for_which: u32,
        key: ExternalKey,
        indices: Vec<u32>,
    },
    ExportExternalF {
        for_which: u32,
        key: ExternalKey,
        indices: Vec<u32>,
    },

    // ── Hash ──────────────────────────────────────────────────────────
    PoseidonHash,
    EqHash,

    // ── Prover hints and their compile-time placeholders ─────────────
    /// A value the prover supplies: lane `rank` of `kind` applied to the
    /// `len` leading operands, where `rank` is the last operand (`len + 1`
    /// operands in all). Every backend computes it with
    /// [`crate::prove::kernel::hint_lane`] (the Noir backend with the same
    /// algorithm in `noir_fpc`), and a rank outside `[0, len)` gives 0.
    ///
    /// A hint constrains nothing: the circuit accepts ANY value in its
    /// place. Every producer must therefore follow it with IR constraints
    /// under which the honest lane is the only admissible value, and state
    /// that argument next to the code (see `lower::sort`). Not
    /// fixed: a hint nothing reads is dead, and its pins read it.
    Hint {
        kind: HintKind,
        len: u32,
    },
    /// One live output lane of an array operator (`lower::op`): the
    /// operator, how many leading operands are its shared inputs, and the
    /// lane's output and position (a runtime position is given by the
    /// lane's own operands).
    ///
    /// A compile-time placeholder only: the array lowering layer
    /// (`lower::ArrayLowering`) replaces every one with the constructions it
    /// selects for the operator's live lanes after dead-code elimination.
    /// No backend accepts it.
    ArrayOp(crate::lower::ArrayOp),
    /// `[(v1, i1) < (v2, i2)]` for the operands `[v1, i1, v2, i2]`, compared
    /// lexicographically: `v1 < v2`, or `v1 == v2` and `i1 < i2`. The values
    /// are signed (int64 or Q32: a backend rejects values outside the 64-bit
    /// window, as for `LtI`); the producer guarantees the indices lie in
    /// `[0, 2^index_bits)` (asserted bounds or constants). Since
    /// `|i1 - i2| < 2^index_bits`, the order is the sign of the packed
    /// difference `(v1 - v2) * 2^index_bits + (i1 - i2)`, which backends test
    /// with one comparison instead of three and two logic gates.
    KeyLess {
        index_bits: u32,
    },
}

/// The deterministic function an [`IR::Hint`] takes a lane of. Both order
/// the operands by their signed reading (the integer of smallest magnitude
/// a field element represents; on the signed 64-bit window where Zinnia's
/// integers and Q32 floats live, the integer order of the values), see
/// [`crate::prove::kernel::signed_cmp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HintKind {
    /// The stable ascending argsort: lane `r` is the original position of
    /// the element of rank `r`, equal elements in their original order
    /// (NumPy's `kind="stable"`).
    StableArgsort,
    /// The ascending sorted values: lane `r` is the value of rank `r`.
    SortedValues,
    /// The ascending sorted values of the active elements of a runtime-length
    /// array: the operands are `n` values and then their `n` activity flags
    /// (1: the slot holds an element), so `len = 2n`; lane `r` is the value
    /// of rank `r` among the active ones (0 past their number).
    MaskedSortedValues,
    /// The position NumPy's index `x` names on an axis of length `n`, as a
    /// one-hot vector: the operands are `[x, n]` (`n` a small constant), and
    /// lane `r` (for `r` in `[0, n]`) is 1 when `r` is `x` (`0 <= x <= n`) or
    /// `x + n` (`-n <= x < 0`), else 0. Lane `n` is the end of the axis, an
    /// insertion point; for any other `x` every lane is 0.
    PositionOneHot,
}

impl HintKind {
    pub fn name(self) -> &'static str {
        match self {
            HintKind::StableArgsort => "stable_argsort",
            HintKind::SortedValues => "sorted_values",
            HintKind::MaskedSortedValues => "masked_sorted_values",
            HintKind::PositionOneHot => "position_one_hot",
        }
    }

    pub fn from_name(name: &str) -> Option<HintKind> {
        match name {
            "stable_argsort" => Some(HintKind::StableArgsort),
            "sorted_values" => Some(HintKind::SortedValues),
            "masked_sorted_values" => Some(HintKind::MaskedSortedValues),
            "position_one_hot" => Some(HintKind::PositionOneHot),
            _ => None,
        }
    }
}

/// Key type for ExportExternal IRs — can be either an integer or a string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExternalKey {
    Int(u32),
    Str(String),
}

impl IR {
    /// Returns the signature string for this IR instruction.
    /// Mirrors the Python `get_signature()` method.
    pub fn signature(&self) -> String {
        match self {
            // Constants
            IR::ConstantInt { value } => format!("constant_int[{}]", value),
            IR::ConstantFloat { value } => format!("constant_float[{}]", value),
            IR::ConstantBool { value } => format!("constant_bool[{}]", value),
            IR::ConstantStr { value } => format!("constant_str[{}]", value),

            // Integer arithmetic
            IR::AddI => "add_i".to_string(),
            IR::SubI => "sub_i".to_string(),
            IR::MulI => "mul_i".to_string(),
            IR::DivI => "div_i".to_string(),
            IR::FloorDivI => "floor_divide_i".to_string(),
            IR::FloorDivIPinned { divisor_bits } => format!("floor_divide_i_pinned[{}]", divisor_bits),
            IR::ModI => "mod_i".to_string(),
            IR::PowI => "pow_i".to_string(),
            IR::AbsI => "abs_i".to_string(),
            IR::SignI => "sign_i".to_string(),
            IR::InvI => "inv_i".to_string(),
            IR::BitAndI => "bit_and_i".to_string(),
            IR::BitOrI => "bit_or_i".to_string(),
            IR::BitXorI => "bit_xor_i".to_string(),
            IR::ShlI => "shl_i".to_string(),
            IR::ShrI => "shr_i".to_string(),
            IR::BitNotI => "bit_not_i".to_string(),

            // Float arithmetic
            IR::AddF => "add_f".to_string(),
            IR::SumF { len } => format!("sum_f[{}]", len),
            IR::SubF => "sub_f".to_string(),
            IR::MulF => "mul_f".to_string(),
            IR::ProductSumF { len } => format!("product_sum_f[{}]", len),
            IR::DivF => "div_f".to_string(),
            IR::FloorDivF => "floor_divide_f".to_string(),
            IR::ModF => "mod_f".to_string(),
            IR::PowF => "pow_f".to_string(),
            IR::PowFI => "pow_f_int".to_string(),
            IR::AbsF => "abs_f".to_string(),
            IR::SignF => "sign_f".to_string(),

            // Integer comparison
            IR::EqI => "eq_i".to_string(),
            IR::NeI => "ne_i".to_string(),
            IR::LtIBounded => "lt_i_bounded".to_string(),
            IR::GtIBounded => "gt_i_bounded".to_string(),
            IR::AssertU128Lt => "assert_u128_lt".to_string(),
            IR::AssertU128LtBit { hi, bit, bits } => {
                format!("assert_u128_lt_bit[{},{},{}]", hi, bit, bits)
            }
            IR::IndexInBox { rank } => format!("index_in_box[{}]", rank),
            IR::LtI => "lt_i".to_string(),
            IR::LteI => "lte_i".to_string(),
            IR::GtI => "gt_i".to_string(),
            IR::GteI => "gte_i".to_string(),

            // Float comparison
            IR::EqF => "eq_f".to_string(),
            IR::NeF => "ne_f".to_string(),
            IR::LtF => "lt_f".to_string(),
            IR::LteF => "lte_f".to_string(),
            IR::GtF => "gt_f".to_string(),
            IR::GteF => "gte_f".to_string(),

            // Math
            IR::SinF => "sin_f".to_string(),
            IR::SinHF => "sinh_f".to_string(),
            IR::CosF => "cos_f".to_string(),
            IR::CosHF => "cosh_f".to_string(),
            IR::TanF => "tan_f".to_string(),
            IR::TanHF => "tanh_f".to_string(),
            IR::SqrtF => "sqrt_f".to_string(),
            IR::ExpF => "exp_f".to_string(),
            IR::LogF => "log_f".to_string(),
            IR::ArcSinF => "arcsin_f".to_string(),
            IR::ArcCosF => "arccos_f".to_string(),
            IR::ArcTan2F => "arctan2_f".to_string(),

            // Logical
            IR::LogicalAnd => "logical_and".to_string(),
            IR::LogicalOr => "logical_or".to_string(),
            IR::LogicalNot => "logical_not".to_string(),

            // Selection
            IR::SelectI => "select_i".to_string(),
            IR::SelectF => "select_f".to_string(),
            IR::SelectB => "select_b".to_string(),
            IR::SelectBitI => "select_bit_i".to_string(),
            IR::SelectBitF => "select_bit_f".to_string(),
            IR::SelectManyI { len, lane } => format!("select_many_i[{},{}]", len, lane),
            IR::SelectManyF { len, lane } => format!("select_many_f[{},{}]", len, lane),

            // Cast
            IR::IntCast => "int_cast".to_string(),
            IR::FloatCast => "float_cast".to_string(),
            IR::BoolCast => "bool_cast".to_string(),

            // String
            IR::AddStr => "add_str".to_string(),
            IR::StrI => "str_i".to_string(),
            IR::StrF => "str_f".to_string(),

            // I/O
            IR::ReadInteger { path, is_public } => {
                format!("read_integer[{}][{}]", path.display(), is_public)
            }
            IR::ReadFloat { path, is_public } => {
                format!("read_float[{}][{}]", path.display(), is_public)
            }
            IR::ReadHash { path, is_public } => {
                format!("read_hash[{}][{}]", path.display(), is_public)
            }
            IR::ReadExternalResult {
                store_idx,
                output_idx,
                is_float,
            } => {
                format!(
                    "read_external_result[{}.{}][float={}]",
                    store_idx, output_idx, is_float
                )
            }
            IR::Print => "print".to_string(),

            // Assert & expose
            IR::Assert => "assert".to_string(),
            IR::AssertBit => "assert_bit".to_string(),
            IR::AssertNonNegI => "assert_non_neg_i".to_string(),
            IR::ExposePublicI => "expose_public_i".to_string(),
            IR::ExposePublicF => "expose_public_f".to_string(),

            // Structural-predicate precondition
            IR::StructuralPredicate {
                kind,
                args,
                op,
                bound,
            } => {
                let args_str = args.join(",");
                let op_str = op.as_deref().unwrap_or("");
                let bound_str = bound.as_deref().unwrap_or("");
                format!(
                    "structural_predicate[{}][{}][{}][{}]",
                    kind, args_str, op_str, bound_str
                )
            }

            // Scalar precondition (term-tree carrying the payload)
            IR::ScalarPrecondition { term_json } => {
                format!("scalar_precondition[{}]", term_json)
            }

            // Memory
            IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            } => format!("allocate_memory[{}][{}][{}]", segment_id, size, init_value),
            IR::WriteMemory { segment_id } => format!("write_memory[{}]", segment_id),
            IR::ReadMemory { segment_id } => format!("read_memory[{}]", segment_id),
            IR::MemoryTraceEmit {
                segment_id,
                is_write,
            } => format!("memory_trace_emit[{}][{}]", segment_id, is_write),
            IR::MemoryTraceSeal => "memory_trace_seal".to_string(),

            // Dynamic NDArray
            IR::AllocateDynamicNDArrayMeta {
                array_id,
                dtype_name,
                max_length,
                max_rank,
                max_shape,
            } => format!(
                "alloc_dynamic_ndarray_meta[{}][{}][{}][{}][{:?}]",
                array_id, dtype_name, max_length, max_rank, max_shape
            ),
            IR::WitnessDynamicNDArrayMeta {
                array_id,
                max_rank,
                param_name,
            } => {
                format!(
                    "witness_dynamic_ndarray_meta[{}][{}][{}]",
                    array_id, max_rank, param_name
                )
            }
            IR::AssertDynamicNDArrayMeta {
                array_id,
                max_rank,
                max_length,
            } => format!(
                "assert_dynamic_ndarray_meta[{}][{}][{}]",
                array_id, max_rank, max_length
            ),
            IR::DynamicNDArrayGetItem {
                array_id,
                segment_id,
            } => format!("dynamic_ndarray_get_item[{}][{}]", array_id, segment_id),
            IR::DynamicNDArraySetItem {
                array_id,
                segment_id,
            } => format!("dynamic_ndarray_set_item[{}][{}]", array_id, segment_id),

            // External
            IR::InvokeExternal { .. } => "invoke_external".to_string(),
            IR::ExportExternalI {
                for_which,
                key,
                indices,
            } => {
                let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!(
                    "export_external_i[{}][{:?}][{}]",
                    for_which,
                    key,
                    idx_str.join(", ")
                )
            }
            IR::ExportExternalF {
                for_which,
                key,
                indices,
            } => {
                let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!(
                    "export_external_i[{}][{:?}][{}]",
                    for_which,
                    key,
                    idx_str.join(", ")
                )
            }

            // Hash
            IR::PoseidonHash => "poseidon_hash".to_string(),
            IR::EqHash => "eq_hash".to_string(),
            // (StructuralPredicate handled in the Assert/expose block above)

            // Prover hints and their placeholders
            IR::Hint { kind, len } => format!("hint[{}][{}]", kind.name(), len),
            IR::KeyLess { index_bits } => format!("key_less[{}]", index_bits),
            IR::ArrayOp(op) => op.describe(),
        }
    }

    /// Returns `true` if this IR is a "fixed" IR that should not be eliminated
    /// by dead code elimination or constant folding.
    /// Mirrors the Python `is_fixed_ir()` method.
    ///
    /// `ReadInteger { is_public }` / `ReadFloat { is_public }` are fixed
    /// only when `is_public == true`: public inputs are part of the
    /// verifier-visible interface and must be preserved. Private reads
    /// with no downstream consumers can be safely eliminated by DCE —
    /// they would otherwise commit witness values the circuit never
    /// constrains (e.g. case 0402 declares a 36 864-cell
    /// `Float[8,9,8,8,8]` parameter but only uses `dims` and `index`).
    pub fn is_fixed(&self) -> bool {
        match self {
            IR::Assert
            | IR::AssertBit
            | IR::AssertNonNegI
            | IR::AssertU128Lt
            // Returns a bit but also asserts `0 <= idx < hi`; without being
            // fixed, DCE deleted the range check whenever the bit was unused.
            | IR::AssertU128LtBit { .. }
            | IR::StructuralPredicate { .. }
            | IR::ScalarPrecondition { .. }
            | IR::Print
            | IR::ExposePublicI
            | IR::ExposePublicF
            | IR::ReadHash { .. }
            | IR::AllocateMemory { .. }
            | IR::WriteMemory { .. }
            | IR::ReadMemory { .. }
            | IR::MemoryTraceEmit { .. }
            | IR::MemoryTraceSeal
            | IR::AllocateDynamicNDArrayMeta { .. }
            | IR::WitnessDynamicNDArrayMeta { .. }
            | IR::AssertDynamicNDArrayMeta { .. }
            | IR::DynamicNDArrayGetItem { .. }
            | IR::DynamicNDArraySetItem { .. }
            | IR::InvokeExternal { .. }
            | IR::ExportExternalI { .. }
            | IR::ExportExternalF { .. }
            | IR::EqHash => true,
            IR::ReadInteger { is_public, .. } | IR::ReadFloat { is_public, .. } => *is_public,
            _ => false,
        }
    }

    /// Returns the Python IR class name for serialization compatibility.
    /// Used by `IRFactory.export()` / `IRFactory.import_from()`.
    pub fn class_name(&self) -> &'static str {
        match self {
            IR::ConstantInt { .. } => "ConstantIntIR",
            IR::ConstantFloat { .. } => "ConstantFloatIR",
            IR::ConstantBool { .. } => "ConstantBoolIR",
            IR::ConstantStr { .. } => "ConstantStrIR",
            IR::AddI => "AddIIR",
            IR::SubI => "SubIIR",
            IR::MulI => "MulIIR",
            IR::DivI => "DivIIR",
            IR::FloorDivI => "FloorDivIIR",
            IR::FloorDivIPinned { .. } => "FloorDivIPinnedIR",
            IR::ModI => "ModIIR",
            IR::PowI => "PowIIR",
            IR::AbsI => "AbsIIR",
            IR::SignI => "SignIIR",
            IR::InvI => "InvIIR",
            IR::BitAndI => "BitAndIIR",
            IR::BitOrI => "BitOrIIR",
            IR::BitXorI => "BitXorIIR",
            IR::ShlI => "ShlIIR",
            IR::ShrI => "ShrIIR",
            IR::BitNotI => "BitNotIIR",
            IR::AddF => "AddFIR",
            IR::SumF { .. } => "SumFIR",
            IR::SubF => "SubFIR",
            IR::MulF => "MulFIR",
            IR::ProductSumF { .. } => "ProductSumFIR",
            IR::DivF => "DivFIR",
            IR::FloorDivF => "FloorDivFIR",
            IR::ModF => "ModFIR",
            IR::PowF => "PowFIR",
            IR::PowFI => "PowFIIR",
            IR::AbsF => "AbsFIR",
            IR::SignF => "SignFIR",
            IR::EqI => "EqualIIR",
            IR::NeI => "NotEqualIIR",
            IR::LtIBounded => "LessThanIBoundedIR",
            IR::GtIBounded => "GreaterThanIBoundedIR",
            IR::AssertU128Lt => "AssertU128LtIR",
            IR::AssertU128LtBit { .. } => "AssertU128LtBitIR",
            IR::IndexInBox { .. } => "IndexInBoxIR",
            IR::LtI => "LessThanIIR",
            IR::LteI => "LessThanOrEqualIIR",
            IR::GtI => "GreaterThanIIR",
            IR::GteI => "GreaterThanOrEqualIIR",
            IR::EqF => "EqualFIR",
            IR::NeF => "NotEqualFIR",
            IR::LtF => "LessThanFIR",
            IR::LteF => "LessThanOrEqualFIR",
            IR::GtF => "GreaterThanFIR",
            IR::GteF => "GreaterThanOrEqualFIR",
            IR::SinF => "SinFIR",
            IR::SinHF => "SinHFIR",
            IR::CosF => "CosFIR",
            IR::CosHF => "CosHFIR",
            IR::TanF => "TanFIR",
            IR::TanHF => "TanHFIR",
            IR::SqrtF => "SqrtFIR",
            IR::ExpF => "ExpFIR",
            IR::LogF => "LogFIR",
            IR::ArcSinF => "ArcSinFIR",
            IR::ArcCosF => "ArcCosFIR",
            IR::ArcTan2F => "ArcTan2FIR",
            IR::LogicalAnd => "LogicalAndIR",
            IR::LogicalOr => "LogicalOrIR",
            IR::LogicalNot => "LogicalNotIR",
            IR::SelectI => "SelectIIR",
            IR::SelectF => "SelectFIR",
            IR::SelectB => "SelectBIR",
            IR::SelectBitI => "SelectBitIIR",
            IR::SelectBitF => "SelectBitFIR",
            IR::SelectManyI { .. } => "SelectManyIIR",
            IR::SelectManyF { .. } => "SelectManyFIR",
            IR::IntCast => "IntCastIR",
            IR::FloatCast => "FloatCastIR",
            IR::BoolCast => "BoolCastIR",
            IR::AddStr => "AddStrIR",
            IR::StrI => "StrIIR",
            IR::StrF => "StrFIR",
            IR::ReadInteger { .. } => "ReadIntegerIR",
            IR::ReadFloat { .. } => "ReadFloatIR",
            IR::ReadHash { .. } => "ReadHashIR",
            IR::ReadExternalResult { .. } => "ReadExternalResultIR",
            IR::Print => "PrintIR",
            IR::Assert => "AssertIR",
            IR::AssertBit => "AssertBitIR",
            IR::AssertNonNegI => "AssertNonNegIIR",
            IR::StructuralPredicate { .. } => "StructuralPredicateIR",
            IR::ScalarPrecondition { .. } => "ScalarPreconditionIR",
            IR::ExposePublicI => "ExposePublicIIR",
            IR::ExposePublicF => "ExposePublicFIR",
            IR::AllocateMemory { .. } => "AllocateMemoryIR",
            IR::WriteMemory { .. } => "WriteMemoryIR",
            IR::ReadMemory { .. } => "ReadMemoryIR",
            IR::MemoryTraceEmit { .. } => "MemoryTraceEmitIR",
            IR::MemoryTraceSeal => "MemoryTraceSealIR",
            IR::AllocateDynamicNDArrayMeta { .. } => "AllocateDynamicNDArrayMetaIR",
            IR::WitnessDynamicNDArrayMeta { .. } => "WitnessDynamicNDArrayMetaIR",
            IR::AssertDynamicNDArrayMeta { .. } => "AssertDynamicNDArrayMetaIR",
            IR::DynamicNDArrayGetItem { .. } => "DynamicNDArrayGetItemIR",
            IR::DynamicNDArraySetItem { .. } => "DynamicNDArraySetItemIR",
            IR::InvokeExternal { .. } => "InvokeExternalIR",
            IR::ExportExternalI { .. } => "ExportExternalIIR",
            IR::ExportExternalF { .. } => "ExportExternalFIR",
            IR::PoseidonHash => "PoseidonHashIR",
            IR::EqHash => "EqualHashIR",
            IR::Hint { .. } => "HintIR",
            IR::ArrayOp(_) => "ArrayOpIR",
            IR::KeyLess { .. } => "KeyLessIR",
        }
    }

    /// Serialize the IR-specific data to a JSON value.
    /// Mirrors the Python `export()` method on each IR class.
    ///
    /// # Determinism invariant (load-bearing for `ProofArtifact.ir_hash`)
    ///
    /// `compiler.halo2-cross-process-vk-serialization` (D' direction)
    /// ships the IR through `ProofArtifact.vk_bytes` and identifies it via
    /// a BLAKE2b-256 `ir_hash` over the canonical JSON bytes produced by
    /// this path. That hash must be byte-stable across processes for the
    /// same IR — otherwise the fuzzer's auto-filer sees spurious
    /// `IrHashMismatch` errors and the VK cache loses its hit rate.
    ///
    /// Most variants here use `serde_json::json!` with fixed source-order
    /// keys; the resulting `serde_json::Map` is `BTreeMap`-backed (no
    /// `preserve_order` feature on `serde_json` in this project, confirmed
    /// via `Cargo.lock` — no `indexmap` transitive dep), so keys serialize
    /// lexicographically and the byte output is deterministic.
    ///
    /// **The one variant that embeds a `HashMap<String, serde_json::Value>`
    /// (`IR::InvokeExternal::kwargs`) relies on the same `BTreeMap`-backed
    /// `Map` to sort keys at serialize time.** `HashMap → serde_json::Value`
    /// via `From<HashMap<K, V>> for Value` collects into that map, so
    /// `HashMap` iteration randomness is normalized away. **If anyone ever
    /// enables `serde_json`'s `preserve_order` feature (which switches
    /// `Map` to `IndexMap`, preserving insertion order), this guarantee
    /// silently breaks and `ir_hash` will drift across processes for the
    /// same IR.** Guarded by:
    /// 1. The `ir_export_byte_stable_*` determinism regression tests in
    ///    `prove/halo2/tests.rs`.
    /// 2. @Sage's `keygen_vk` byte-equality test (a second-layer guard
    ///    that catches a determinism break one layer up).
    pub fn export_data(&self) -> serde_json::Value {
        match self {
            IR::ConstantInt { value } => serde_json::json!({ "value": value }),
            IR::ConstantFloat { value } => serde_json::json!({ "value": value }),
            IR::ConstantBool { value } => serde_json::json!({ "value": value }),
            IR::ConstantStr { value } => serde_json::json!({ "value": value }),

            IR::ReadInteger { path, is_public } => {
                serde_json::json!({ "path": path, "is_public": is_public })
            }
            IR::ReadFloat { path, is_public } => {
                serde_json::json!({ "path": path, "is_public": is_public })
            }
            IR::ReadHash { path, is_public } => {
                serde_json::json!({ "path": path, "is_public": is_public })
            }
            IR::ReadExternalResult {
                store_idx,
                output_idx,
                is_float,
            } => {
                serde_json::json!({ "store_idx": store_idx, "output_idx": output_idx, "is_float": is_float })
            }
            IR::ProductSumF { len } => serde_json::json!({ "len": len }),
            IR::SumF { len } => serde_json::json!({ "len": len }),
            IR::FloorDivIPinned { divisor_bits } => serde_json::json!({ "divisor_bits": divisor_bits }),
            IR::AssertU128LtBit { hi, bit, bits } => {
                serde_json::json!({ "hi": hi, "bit": bit, "bits": bits })
            }
            IR::SelectManyI { len, lane } | IR::SelectManyF { len, lane } => {
                serde_json::json!({ "len": len, "lane": lane })
            }
            IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            } => serde_json::json!({
                "segment_id": segment_id,
                "size": size,
                "init_value": init_value,
            }),
            IR::WriteMemory { segment_id } => serde_json::json!({ "segment_id": segment_id }),
            IR::ReadMemory { segment_id } => serde_json::json!({ "segment_id": segment_id }),
            IR::IndexInBox { rank } => serde_json::json!({ "rank": rank }),
            IR::MemoryTraceEmit {
                segment_id,
                is_write,
            } => serde_json::json!({ "segment_id": segment_id, "is_write": is_write }),

            IR::AllocateDynamicNDArrayMeta {
                array_id,
                dtype_name,
                max_length,
                max_rank,
                max_shape,
            } => serde_json::json!({
                "array_id": array_id,
                "dtype_name": dtype_name,
                "max_length": max_length,
                "max_rank": max_rank,
                "max_shape": max_shape,
            }),
            IR::WitnessDynamicNDArrayMeta {
                array_id,
                max_rank,
                param_name,
            } => {
                serde_json::json!({ "array_id": array_id, "max_rank": max_rank, "param_name": param_name })
            }
            IR::AssertDynamicNDArrayMeta {
                array_id,
                max_rank,
                max_length,
            } => serde_json::json!({
                "array_id": array_id,
                "max_rank": max_rank,
                "max_length": max_length,
            }),
            IR::DynamicNDArrayGetItem {
                array_id,
                segment_id,
            } => serde_json::json!({ "array_id": array_id, "segment_id": segment_id }),
            IR::DynamicNDArraySetItem {
                array_id,
                segment_id,
            } => serde_json::json!({ "array_id": array_id, "segment_id": segment_id }),

            IR::InvokeExternal {
                store_idx,
                func_name,
                args,
                kwargs,
            } => serde_json::json!({
                "store_idx": store_idx,
                "func_name": func_name,
                "args": args,
                "kwargs": kwargs,
            }),
            IR::ExportExternalI {
                for_which,
                key,
                indices,
            } => serde_json::json!({
                "for_which": for_which,
                "key": key,
                "indices": indices,
            }),
            IR::ExportExternalF {
                for_which,
                key,
                indices,
            } => serde_json::json!({
                "for_which": for_which,
                "key": key,
                "indices": indices,
            }),

            IR::StructuralPredicate {
                kind,
                args,
                op,
                bound,
            } => serde_json::json!({
                "kind": kind,
                "args": args,
                "op": op,
                "bound": bound,
            }),

            IR::ScalarPrecondition { term_json } => serde_json::json!({
                "term_json": term_json,
            }),

            IR::Hint { kind, len } => serde_json::json!({ "kind": kind.name(), "len": len }),
            IR::ArrayOp(op) => serde_json::to_value(op).expect("an ArrayOp serializes"),
            IR::KeyLess { index_bits } => serde_json::json!({ "index_bits": index_bits }),

            // All parameterless IRs export empty dicts
            _ => serde_json::json!({}),
        }
    }

    /// Export to the Python IRFactory.export() format:
    /// `{"__class__": "ClassName", "ir_data": {...}}`
    pub fn export(&self) -> serde_json::Value {
        serde_json::json!({
            "__class__": self.class_name(),
            "ir_data": self.export_data(),
        })
    }

    /// Import from the Python IRFactory.export() format.
    pub fn import_from(data: &serde_json::Value) -> Result<IR, String> {
        let class_name = data["__class__"]
            .as_str()
            .ok_or("Missing __class__ field")?;
        let ir_data = &data["ir_data"];
        IR::from_class_and_data(class_name, ir_data)
    }

    /// Construct an IR variant from class name and data dict.
    fn from_class_and_data(class_name: &str, data: &serde_json::Value) -> Result<IR, String> {
        match class_name {
            "ConstantIntIR" => Ok(IR::ConstantInt {
                value: data["value"].as_i64().ok_or("ConstantInt: missing value")?,
            }),
            "ConstantFloatIR" => Ok(IR::ConstantFloat {
                value: data["value"]
                    .as_f64()
                    .ok_or("ConstantFloat: missing value")?,
            }),
            "ConstantBoolIR" => Ok(IR::ConstantBool {
                value: data["value"]
                    .as_bool()
                    .ok_or("ConstantBool: missing value")?,
            }),
            "ConstantStrIR" => Ok(IR::ConstantStr {
                value: data["value"]
                    .as_str()
                    .ok_or("ConstantStr: missing value")?
                    .to_string(),
            }),

            "AddIIR" => Ok(IR::AddI),
            "SubIIR" => Ok(IR::SubI),
            "MulIIR" => Ok(IR::MulI),
            "DivIIR" => Ok(IR::DivI),
            "FloorDivIIR" => Ok(IR::FloorDivI),
            "FloorDivIPinnedIR" => Ok(IR::FloorDivIPinned {
                divisor_bits: data["divisor_bits"].as_u64().unwrap_or(0) as u32,
            }),
            "ModIIR" => Ok(IR::ModI),
            "PowIIR" => Ok(IR::PowI),
            "AbsIIR" => Ok(IR::AbsI),
            "SignIIR" => Ok(IR::SignI),
            "InvIIR" => Ok(IR::InvI),

            "AddFIR" => Ok(IR::AddF),
            "SumFIR" => Ok(IR::SumF {
                len: data["len"].as_u64().unwrap_or(0) as u32,
            }),
            "SubFIR" => Ok(IR::SubF),
            "MulFIR" => Ok(IR::MulF),
            "ProductSumFIR" => Ok(IR::ProductSumF {
                len: data["len"].as_u64().unwrap_or(0) as u32,
            }),
            "DivFIR" => Ok(IR::DivF),
            "FloorDivFIR" => Ok(IR::FloorDivF),
            "ModFIR" => Ok(IR::ModF),
            "PowFIR" => Ok(IR::PowF),
            "PowFIIR" => Ok(IR::PowFI),
            "AbsFIR" => Ok(IR::AbsF),
            "SignFIR" => Ok(IR::SignF),

            "EqualIIR" => Ok(IR::EqI),
            "NotEqualIIR" => Ok(IR::NeI),
            "LessThanIIR" => Ok(IR::LtI),
            "LessThanIBoundedIR" => Ok(IR::LtIBounded),
            "GreaterThanIBoundedIR" => Ok(IR::GtIBounded),
            "AssertU128LtIR" => Ok(IR::AssertU128Lt),
            "AssertU128LtBitIR" => Ok(IR::AssertU128LtBit {
                hi: data["hi"].as_u64().ok_or("AssertU128LtBit: missing hi")?,
                bit: data["bit"]
                    .as_u64()
                    .ok_or("AssertU128LtBit: missing bit")? as u32,
                bits: data["bits"]
                    .as_u64()
                    .ok_or("AssertU128LtBit: missing bits")? as u32,
            }),
            "IndexInBoxIR" => Ok(IR::IndexInBox {
                rank: data["rank"].as_u64().ok_or("IndexInBox: missing rank")? as usize,
            }),
            "LessThanOrEqualIIR" => Ok(IR::LteI),
            "GreaterThanIIR" => Ok(IR::GtI),
            "GreaterThanOrEqualIIR" => Ok(IR::GteI),

            "EqualFIR" => Ok(IR::EqF),
            "NotEqualFIR" => Ok(IR::NeF),
            "LessThanFIR" => Ok(IR::LtF),
            "LessThanOrEqualFIR" => Ok(IR::LteF),
            "GreaterThanFIR" => Ok(IR::GtF),
            "GreaterThanOrEqualFIR" => Ok(IR::GteF),

            "SinFIR" => Ok(IR::SinF),
            "SinHFIR" => Ok(IR::SinHF),
            "CosFIR" => Ok(IR::CosF),
            "CosHFIR" => Ok(IR::CosHF),
            "TanFIR" => Ok(IR::TanF),
            "TanHFIR" => Ok(IR::TanHF),
            "SqrtFIR" => Ok(IR::SqrtF),
            "ExpFIR" => Ok(IR::ExpF),
            "LogFIR" => Ok(IR::LogF),
            "ArcSinFIR" => Ok(IR::ArcSinF),
            "ArcCosFIR" => Ok(IR::ArcCosF),
            "ArcTan2FIR" => Ok(IR::ArcTan2F),

            "LogicalAndIR" => Ok(IR::LogicalAnd),
            "LogicalOrIR" => Ok(IR::LogicalOr),
            "LogicalNotIR" => Ok(IR::LogicalNot),

            "SelectIIR" => Ok(IR::SelectI),
            "SelectFIR" => Ok(IR::SelectF),
            "SelectBIR" => Ok(IR::SelectB),
            "SelectBitIIR" => Ok(IR::SelectBitI),
            "SelectBitFIR" => Ok(IR::SelectBitF),
            "SelectManyIIR" => Ok(IR::SelectManyI {
                len: data["len"].as_u64().ok_or("SelectManyI: missing len")? as u32,
                lane: data["lane"].as_u64().ok_or("SelectManyI: missing lane")? as u32,
            }),
            "SelectManyFIR" => Ok(IR::SelectManyF {
                len: data["len"].as_u64().ok_or("SelectManyF: missing len")? as u32,
                lane: data["lane"].as_u64().ok_or("SelectManyF: missing lane")? as u32,
            }),
            "IntCastIR" => Ok(IR::IntCast),
            "FloatCastIR" => Ok(IR::FloatCast),
            "BoolCastIR" => Ok(IR::BoolCast),

            "AddStrIR" => Ok(IR::AddStr),
            "StrIIR" => Ok(IR::StrI),
            "StrFIR" => Ok(IR::StrF),

            "ReadIntegerIR" => {
                let path: InputPath = serde_json::from_value(data["path"].clone())
                    .map_err(|e| format!("ReadInteger: invalid path: {}", e))?;
                let is_public = data["is_public"]
                    .as_bool()
                    .ok_or("ReadInteger: missing is_public")?;
                Ok(IR::ReadInteger { path, is_public })
            }
            "ReadFloatIR" => {
                let path: InputPath = serde_json::from_value(data["path"].clone())
                    .map_err(|e| format!("ReadFloat: invalid path: {}", e))?;
                let is_public = data["is_public"]
                    .as_bool()
                    .ok_or("ReadFloat: missing is_public")?;
                Ok(IR::ReadFloat { path, is_public })
            }
            "ReadHashIR" => {
                let path: InputPath = serde_json::from_value(data["path"].clone())
                    .map_err(|e| format!("ReadHash: invalid path: {}", e))?;
                let is_public = data["is_public"]
                    .as_bool()
                    .ok_or("ReadHash: missing is_public")?;
                Ok(IR::ReadHash { path, is_public })
            }
            "ReadExternalResultIR" => Ok(IR::ReadExternalResult {
                store_idx: data["store_idx"].as_u64().unwrap_or(0) as u32,
                output_idx: data["output_idx"].as_u64().unwrap_or(0) as u32,
                is_float: data["is_float"].as_bool().unwrap_or(false),
            }),
            "PrintIR" => Ok(IR::Print),

            "AssertIR" => Ok(IR::Assert),
            "AssertBitIR" => Ok(IR::AssertBit),
            "AssertNonNegIIR" => Ok(IR::AssertNonNegI),
            "StructuralPredicateIR" => {
                let kind = data["kind"].as_str().unwrap_or("").to_string();
                let args: Vec<String> = data["args"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let op = data.get("op").and_then(|v| v.as_str()).map(String::from);
                let bound = data.get("bound").and_then(|v| v.as_str()).map(String::from);
                Ok(IR::StructuralPredicate {
                    kind,
                    args,
                    op,
                    bound,
                })
            }
            "ScalarPreconditionIR" => {
                let term_json = data["term_json"].as_str().unwrap_or("").to_string();
                Ok(IR::ScalarPrecondition { term_json })
            }
            "ExposePublicIIR" => Ok(IR::ExposePublicI),
            "ExposePublicFIR" => Ok(IR::ExposePublicF),

            "AllocateMemoryIR" => Ok(IR::AllocateMemory {
                segment_id: data["segment_id"].as_u64().unwrap_or(0) as u32,
                size: data["size"].as_u64().unwrap_or(0) as u32,
                init_value: data.get("init_value").and_then(|v| v.as_i64()).unwrap_or(0),
            }),
            "WriteMemoryIR" => Ok(IR::WriteMemory {
                segment_id: data["segment_id"].as_u64().unwrap_or(0) as u32,
            }),
            "ReadMemoryIR" => Ok(IR::ReadMemory {
                segment_id: data["segment_id"].as_u64().unwrap_or(0) as u32,
            }),
            "MemoryTraceEmitIR" => Ok(IR::MemoryTraceEmit {
                segment_id: data["segment_id"].as_u64().unwrap_or(0) as u32,
                is_write: data["is_write"].as_bool().unwrap_or(false),
            }),
            "MemoryTraceSealIR" => Ok(IR::MemoryTraceSeal),

            "AllocateDynamicNDArrayMetaIR" => Ok(IR::AllocateDynamicNDArrayMeta {
                array_id: data["array_id"].as_u64().unwrap_or(0) as u32,
                dtype_name: data["dtype_name"].as_str().unwrap_or("").to_string(),
                max_length: data["max_length"].as_u64().unwrap_or(0) as u32,
                max_rank: data["max_rank"].as_u64().unwrap_or(0) as u32,
                max_shape: data["max_shape"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
                    .unwrap_or_default(),
            }),
            "WitnessDynamicNDArrayMetaIR" => Ok(IR::WitnessDynamicNDArrayMeta {
                array_id: data["array_id"].as_u64().unwrap_or(0) as u32,
                max_rank: data["max_rank"].as_u64().unwrap_or(0) as u32,
                param_name: data["param_name"].as_str().unwrap_or("").to_string(),
            }),
            "AssertDynamicNDArrayMetaIR" => Ok(IR::AssertDynamicNDArrayMeta {
                array_id: data["array_id"].as_u64().unwrap_or(0) as u32,
                max_rank: data["max_rank"].as_u64().unwrap_or(0) as u32,
                max_length: data["max_length"].as_u64().unwrap_or(0) as u32,
            }),
            "DynamicNDArrayGetItemIR" => Ok(IR::DynamicNDArrayGetItem {
                array_id: data["array_id"].as_u64().unwrap_or(0) as u32,
                segment_id: data["segment_id"].as_u64().unwrap_or(0) as u32,
            }),
            "DynamicNDArraySetItemIR" => Ok(IR::DynamicNDArraySetItem {
                array_id: data["array_id"].as_u64().unwrap_or(0) as u32,
                segment_id: data["segment_id"].as_u64().unwrap_or(0) as u32,
            }),

            "InvokeExternalIR" => {
                let args: Vec<serde_json::Value> =
                    data["args"].as_array().cloned().unwrap_or_default();
                let kwargs: HashMap<String, serde_json::Value> = data["kwargs"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                Ok(IR::InvokeExternal {
                    store_idx: data["store_idx"].as_u64().unwrap_or(0) as u32,
                    func_name: data["func_name"].as_str().unwrap_or("").to_string(),
                    args,
                    kwargs,
                })
            }
            "ExportExternalIIR" => {
                let key = parse_external_key(&data["key"])?;
                let indices: Vec<u32> = data["indices"]
                    .as_array()
                    .ok_or("ExportExternalI: missing indices")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                Ok(IR::ExportExternalI {
                    for_which: data["for_which"].as_u64().unwrap_or(0) as u32,
                    key,
                    indices,
                })
            }
            "ExportExternalFIR" => {
                let key = parse_external_key(&data["key"])?;
                let indices: Vec<u32> = data["indices"]
                    .as_array()
                    .ok_or("ExportExternalF: missing indices")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                Ok(IR::ExportExternalF {
                    for_which: data["for_which"].as_u64().unwrap_or(0) as u32,
                    key,
                    indices,
                })
            }

            "PoseidonHashIR" => Ok(IR::PoseidonHash),
            "EqualHashIR" => Ok(IR::EqHash),

            "HintIR" => {
                let name = data["kind"].as_str().ok_or("Hint: missing kind")?;
                Ok(IR::Hint {
                    kind: HintKind::from_name(name)
                        .ok_or_else(|| format!("Hint: unknown kind {}", name))?,
                    len: data["len"].as_u64().ok_or("Hint: missing len")? as u32,
                })
            }
            "KeyLessIR" => Ok(IR::KeyLess {
                index_bits: data["index_bits"].as_u64().ok_or("KeyLess: missing index_bits")? as u32,
            }),
            "ArrayOpIR" => serde_json::from_value(data.clone())
                .map(IR::ArrayOp)
                .map_err(|e| format!("ArrayOp: {e}")),

            other => Err(format!("Unknown IR class: {}", other)),
        }
    }
}

fn parse_external_key(val: &serde_json::Value) -> Result<ExternalKey, String> {
    if let Some(n) = val.as_u64() {
        Ok(ExternalKey::Int(n as u32))
    } else if let Some(s) = val.as_str() {
        Ok(ExternalKey::Str(s.to_string()))
    } else {
        Err("ExportExternal key must be int or string".to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::PathSegment;

    #[test]
    fn test_signatures() {
        assert_eq!(IR::AddI.signature(), "add_i");
        assert_eq!(
            IR::ConstantInt { value: 42 }.signature(),
            "constant_int[42]"
        );
        assert_eq!(
            IR::AllocateMemory {
                segment_id: 1,
                size: 100,
                init_value: 0,
            }
            .signature(),
            "allocate_memory[1][100][0]"
        );
        assert_eq!(
            IR::ReadInteger {
                path: InputPath::new("x", vec![PathSegment::Index(1)]),
                is_public: true,
            }
            .signature(),
            "read_integer[x.1][true]"
        );
        assert_eq!(IR::ProductSumF { len: 3 }.signature(), "product_sum_f[3]");
    }

    #[test]
    fn test_is_fixed() {
        assert!(!IR::AddI.is_fixed());
        assert!(!IR::ConstantInt { value: 0 }.is_fixed());
        assert!(IR::Assert.is_fixed());
        assert!(IR::ReadInteger {
            path: InputPath::new("x", vec![]),
            is_public: true,
        }
        .is_fixed());
        assert!(IR::AllocateMemory {
            segment_id: 0,
            size: 0,
            init_value: 0,
        }
        .is_fixed());
        assert!(IR::Print.is_fixed());
        assert!(IR::MemoryTraceSeal.is_fixed());
    }

    #[test]
    fn test_class_names() {
        assert_eq!(IR::AddI.class_name(), "AddIIR");
        assert_eq!(IR::ConstantInt { value: 0 }.class_name(), "ConstantIntIR");
        assert_eq!(IR::EqI.class_name(), "EqualIIR");
        assert_eq!(IR::LogicalNot.class_name(), "LogicalNotIR");
        assert_eq!(IR::ProductSumF { len: 2 }.class_name(), "ProductSumFIR");
        assert_eq!(IR::PoseidonHash.class_name(), "PoseidonHashIR");
        assert_eq!(IR::EqHash.class_name(), "EqualHashIR");
    }

    #[test]
    fn test_export_import_round_trip_simple() {
        let irs = vec![
            IR::AddI,
            IR::SubF,
            IR::ProductSumF { len: 2 },
            IR::EqI,
            IR::LogicalAnd,
            IR::SelectI,
            IR::IntCast,
            IR::Assert,
            IR::Print,
            IR::PoseidonHash,
            IR::EqHash,
            IR::MemoryTraceSeal,
            IR::ExposePublicI,
        ];
        for ir in irs {
            let exported = ir.export();
            let imported = IR::import_from(&exported)
                .unwrap_or_else(|e| panic!("Failed to import {:?}: {}", ir, e));
            assert_eq!(ir, imported, "Round-trip failed for {:?}", ir);
        }
    }

    #[test]
    fn test_export_import_round_trip_parametric() {
        let irs = vec![
            IR::FloorDivIPinned { divisor_bits: 6 },
            IR::ConstantInt { value: 42 },
            IR::ConstantFloat { value: 3.14 },
            IR::ConstantBool { value: true },
            IR::ConstantStr {
                value: "hello".to_string(),
            },
            IR::ReadInteger {
                path: InputPath::new("x", vec![PathSegment::Index(1)]),
                is_public: true,
            },
            IR::ReadFloat {
                path: InputPath::new("y", vec![PathSegment::Index(2)]),
                is_public: false,
            },
            IR::ReadHash {
                path: InputPath::new("h", vec![PathSegment::Hash]),
                is_public: true,
            },
            IR::ReadExternalResult {
                store_idx: 1,
                output_idx: 0,
                is_float: false,
            },
            IR::AllocateMemory {
                segment_id: 1,
                size: 100,
                init_value: 0,
            },
            IR::WriteMemory { segment_id: 2 },
            IR::ReadMemory { segment_id: 3 },
            IR::MemoryTraceEmit {
                segment_id: 4,
                is_write: true,
            },
            IR::AllocateDynamicNDArrayMeta {
                array_id: 0,
                dtype_name: "Integer".to_string(),
                max_length: 50,
                max_rank: 3,
                max_shape: vec![10, 5, 1],
            },
            IR::WitnessDynamicNDArrayMeta {
                array_id: 1,
                max_rank: 2,
                param_name: "x".to_string(),
            },
            IR::AssertDynamicNDArrayMeta {
                array_id: 2,
                max_rank: 3,
                max_length: 100,
            },
            IR::DynamicNDArrayGetItem {
                array_id: 0,
                segment_id: 1,
            },
            IR::DynamicNDArraySetItem {
                array_id: 0,
                segment_id: 1,
            },
            IR::ExportExternalI {
                for_which: 0,
                key: ExternalKey::Int(1),
                indices: vec![0, 1],
            },
            IR::ExportExternalF {
                for_which: 1,
                key: ExternalKey::Str("x".to_string()),
                indices: vec![2],
            },
            IR::Hint { kind: HintKind::StableArgsort, len: 7 },
            IR::Hint { kind: HintKind::SortedValues, len: 1 },
            IR::Hint { kind: HintKind::MaskedSortedValues, len: 6 },
            IR::ArrayOp(crate::lower::ArrayOp {
                kind: crate::lower::OpKind::Sort,
                shared: 5,
                output: 1,
                position: Some(3),
            }),
            IR::ArrayOp(crate::lower::ArrayOp {
                kind: crate::lower::OpKind::Sort,
                shared: 5,
                output: 0,
                position: None,
            }),
            IR::KeyLess { index_bits: 7 },
        ];
        for ir in irs {
            let exported = ir.export();
            let imported = IR::import_from(&exported)
                .unwrap_or_else(|e| panic!("Failed to import {:?}: {}", ir, e));
            assert_eq!(ir, imported, "Round-trip failed for {:?}", ir);
        }
    }

    #[test]
    fn test_invoke_external_round_trip() {
        let ir = IR::InvokeExternal {
            store_idx: 5,
            func_name: "my_func".to_string(),
            args: vec![serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}})],
            kwargs: HashMap::from([(
                "key1".to_string(),
                serde_json::json!({"__class__": "FloatDTDescriptor", "dt_data": {}}),
            )]),
        };
        let exported = ir.export();
        let imported = IR::import_from(&exported).unwrap();
        assert_eq!(ir, imported);
    }
}

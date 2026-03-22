use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    ConstantInt { value: i64 },
    ConstantFloat { value: f64 },
    ConstantBool { value: bool },
    ConstantStr { value: String },

    // ── Integer arithmetic ────────────────────────────────────────────
    AddI,
    SubI,
    MulI,
    DivI,
    FloorDivI,
    ModI,
    PowI,
    AbsI,
    SignI,
    InvI,

    // ── Float arithmetic ──────────────────────────────────────────────
    AddF,
    SubF,
    MulF,
    DivF,
    FloorDivF,
    ModF,
    PowF,
    AbsF,
    SignF,

    // ── Integer comparison ────────────────────────────────────────────
    EqI,
    NeI,
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

    // ── Logical ───────────────────────────────────────────────────────
    LogicalAnd,
    LogicalOr,
    LogicalNot,

    // ── Selection (conditional) ───────────────────────────────────────
    SelectI,
    SelectF,
    SelectB,

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
        indices: Vec<u32>,
        is_public: bool,
    },
    ReadFloat {
        indices: Vec<u32>,
        is_public: bool,
    },
    ReadHash {
        indices: Vec<u32>,
        is_public: bool,
    },
    Print,

    // ── Assertions & public exposure ──────────────────────────────────
    Assert,
    ExposePublicI,
    ExposePublicF,

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
    },
    WitnessDynamicNDArrayMeta {
        array_id: u32,
        max_rank: u32,
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
            IR::ModI => "mod_i".to_string(),
            IR::PowI => "pow_i".to_string(),
            IR::AbsI => "abs_i".to_string(),
            IR::SignI => "sign_i".to_string(),
            IR::InvI => "inv_i".to_string(),

            // Float arithmetic
            IR::AddF => "add_f".to_string(),
            IR::SubF => "sub_f".to_string(),
            IR::MulF => "mul_f".to_string(),
            IR::DivF => "div_f".to_string(),
            IR::FloorDivF => "floor_divide_f".to_string(),
            IR::ModF => "mod_f".to_string(),
            IR::PowF => "pow_f".to_string(),
            IR::AbsF => "abs_f".to_string(),
            IR::SignF => "sign_f".to_string(),

            // Integer comparison
            IR::EqI => "eq_i".to_string(),
            IR::NeI => "ne_i".to_string(),
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

            // Logical
            IR::LogicalAnd => "logical_and".to_string(),
            IR::LogicalOr => "logical_or".to_string(),
            IR::LogicalNot => "logical_not".to_string(),

            // Selection
            IR::SelectI => "select_i".to_string(),
            IR::SelectF => "select_f".to_string(),
            IR::SelectB => "select_b".to_string(),

            // Cast
            IR::IntCast => "int_cast".to_string(),
            IR::FloatCast => "float_cast".to_string(),
            IR::BoolCast => "bool_cast".to_string(),

            // String
            IR::AddStr => "add_str".to_string(),
            IR::StrI => "str_i".to_string(),
            IR::StrF => "str_f".to_string(),

            // I/O
            IR::ReadInteger { indices, is_public } => {
                let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!("read_integer[{}][{}]", idx_str.join(", "), is_public)
            }
            IR::ReadFloat { indices, is_public } => {
                let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!("read_float[{}][{}]", idx_str.join(", "), is_public)
            }
            IR::ReadHash { indices, is_public } => {
                let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!("read_hash[({},)][{}]", idx_str.join(", "), is_public)
            }
            IR::Print => "print".to_string(),

            // Assert & expose
            IR::Assert => "assert".to_string(),
            IR::ExposePublicI => "expose_public_i".to_string(),
            IR::ExposePublicF => "expose_public_f".to_string(),

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
            } => format!(
                "alloc_dynamic_ndarray_meta[{}][{}][{}][{}]",
                array_id, dtype_name, max_length, max_rank
            ),
            IR::WitnessDynamicNDArrayMeta { array_id, max_rank } => {
                format!("witness_dynamic_ndarray_meta[{}][{}]", array_id, max_rank)
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
        }
    }

    /// Returns `true` if this IR is a "fixed" IR that should not be eliminated
    /// by dead code elimination or constant folding.
    /// Mirrors the Python `is_fixed_ir()` method.
    pub fn is_fixed(&self) -> bool {
        matches!(
            self,
            IR::Assert
                | IR::Print
                | IR::ExposePublicI
                | IR::ExposePublicF
                | IR::ReadInteger { .. }
                | IR::ReadFloat { .. }
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
                | IR::EqHash
        )
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
            IR::ModI => "ModIIR",
            IR::PowI => "PowIIR",
            IR::AbsI => "AbsIIR",
            IR::SignI => "SignIIR",
            IR::InvI => "InvIIR",
            IR::AddF => "AddFIR",
            IR::SubF => "SubFIR",
            IR::MulF => "MulFIR",
            IR::DivF => "DivFIR",
            IR::FloorDivF => "FloorDivFIR",
            IR::ModF => "ModFIR",
            IR::PowF => "PowFIR",
            IR::AbsF => "AbsFIR",
            IR::SignF => "SignFIR",
            IR::EqI => "EqualIIR",
            IR::NeI => "NotEqualIIR",
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
            IR::LogicalAnd => "LogicalAndIR",
            IR::LogicalOr => "LogicalOrIR",
            IR::LogicalNot => "LogicalNotIR",
            IR::SelectI => "SelectIIR",
            IR::SelectF => "SelectFIR",
            IR::SelectB => "SelectBIR",
            IR::IntCast => "IntCastIR",
            IR::FloatCast => "FloatCastIR",
            IR::BoolCast => "BoolCastIR",
            IR::AddStr => "AddStrIR",
            IR::StrI => "StrIIR",
            IR::StrF => "StrFIR",
            IR::ReadInteger { .. } => "ReadIntegerIR",
            IR::ReadFloat { .. } => "ReadFloatIR",
            IR::ReadHash { .. } => "ReadHashIR",
            IR::Print => "PrintIR",
            IR::Assert => "AssertIR",
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
        }
    }

    /// Serialize the IR-specific data to a JSON value.
    /// Mirrors the Python `export()` method on each IR class.
    pub fn export_data(&self) -> serde_json::Value {
        match self {
            IR::ConstantInt { value } => serde_json::json!({ "value": value }),
            IR::ConstantFloat { value } => serde_json::json!({ "value": value }),
            IR::ConstantBool { value } => serde_json::json!({ "value": value }),
            IR::ConstantStr { value } => serde_json::json!({ "value": value }),

            IR::ReadInteger { indices, is_public } => {
                serde_json::json!({ "indices": indices, "is_public": is_public })
            }
            IR::ReadFloat { indices, is_public } => {
                serde_json::json!({ "indices": indices, "is_public": is_public })
            }
            IR::ReadHash { indices, is_public } => {
                serde_json::json!({ "indices": indices, "is_public": is_public })
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
            IR::MemoryTraceEmit {
                segment_id,
                is_write,
            } => serde_json::json!({ "segment_id": segment_id, "is_write": is_write }),

            IR::AllocateDynamicNDArrayMeta {
                array_id,
                dtype_name,
                max_length,
                max_rank,
            } => serde_json::json!({
                "array_id": array_id,
                "dtype_name": dtype_name,
                "max_length": max_length,
                "max_rank": max_rank,
            }),
            IR::WitnessDynamicNDArrayMeta { array_id, max_rank } => {
                serde_json::json!({ "array_id": array_id, "max_rank": max_rank })
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
    fn from_class_and_data(
        class_name: &str,
        data: &serde_json::Value,
    ) -> Result<IR, String> {
        match class_name {
            "ConstantIntIR" => Ok(IR::ConstantInt {
                value: data["value"].as_i64().ok_or("ConstantInt: missing value")?,
            }),
            "ConstantFloatIR" => Ok(IR::ConstantFloat {
                value: data["value"].as_f64().ok_or("ConstantFloat: missing value")?,
            }),
            "ConstantBoolIR" => Ok(IR::ConstantBool {
                value: data["value"].as_bool().ok_or("ConstantBool: missing value")?,
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
            "ModIIR" => Ok(IR::ModI),
            "PowIIR" => Ok(IR::PowI),
            "AbsIIR" => Ok(IR::AbsI),
            "SignIIR" => Ok(IR::SignI),
            "InvIIR" => Ok(IR::InvI),

            "AddFIR" => Ok(IR::AddF),
            "SubFIR" => Ok(IR::SubF),
            "MulFIR" => Ok(IR::MulF),
            "DivFIR" => Ok(IR::DivF),
            "FloorDivFIR" => Ok(IR::FloorDivF),
            "ModFIR" => Ok(IR::ModF),
            "PowFIR" => Ok(IR::PowF),
            "AbsFIR" => Ok(IR::AbsF),
            "SignFIR" => Ok(IR::SignF),

            "EqualIIR" => Ok(IR::EqI),
            "NotEqualIIR" => Ok(IR::NeI),
            "LessThanIIR" => Ok(IR::LtI),
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

            "LogicalAndIR" => Ok(IR::LogicalAnd),
            "LogicalOrIR" => Ok(IR::LogicalOr),
            "LogicalNotIR" => Ok(IR::LogicalNot),

            "SelectIIR" => Ok(IR::SelectI),
            "SelectFIR" => Ok(IR::SelectF),
            "SelectBIR" => Ok(IR::SelectB),

            "IntCastIR" => Ok(IR::IntCast),
            "FloatCastIR" => Ok(IR::FloatCast),
            "BoolCastIR" => Ok(IR::BoolCast),

            "AddStrIR" => Ok(IR::AddStr),
            "StrIIR" => Ok(IR::StrI),
            "StrFIR" => Ok(IR::StrF),

            "ReadIntegerIR" => {
                let indices: Vec<u32> = data["indices"]
                    .as_array()
                    .ok_or("ReadInteger: missing indices")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                let is_public = data["is_public"]
                    .as_bool()
                    .ok_or("ReadInteger: missing is_public")?;
                Ok(IR::ReadInteger { indices, is_public })
            }
            "ReadFloatIR" => {
                let indices: Vec<u32> = data["indices"]
                    .as_array()
                    .ok_or("ReadFloat: missing indices")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                let is_public = data["is_public"]
                    .as_bool()
                    .ok_or("ReadFloat: missing is_public")?;
                Ok(IR::ReadFloat { indices, is_public })
            }
            "ReadHashIR" => {
                let indices: Vec<u32> = data["indices"]
                    .as_array()
                    .ok_or("ReadHash: missing indices")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                let is_public = data["is_public"]
                    .as_bool()
                    .ok_or("ReadHash: missing is_public")?;
                Ok(IR::ReadHash { indices, is_public })
            }
            "PrintIR" => Ok(IR::Print),

            "AssertIR" => Ok(IR::Assert),
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
                dtype_name: data["dtype_name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                max_length: data["max_length"].as_u64().unwrap_or(0) as u32,
                max_rank: data["max_rank"].as_u64().unwrap_or(0) as u32,
            }),
            "WitnessDynamicNDArrayMetaIR" => Ok(IR::WitnessDynamicNDArrayMeta {
                array_id: data["array_id"].as_u64().unwrap_or(0) as u32,
                max_rank: data["max_rank"].as_u64().unwrap_or(0) as u32,
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
                let args: Vec<serde_json::Value> = data["args"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let kwargs: HashMap<String, serde_json::Value> = data["kwargs"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                Ok(IR::InvokeExternal {
                    store_idx: data["store_idx"].as_u64().unwrap_or(0) as u32,
                    func_name: data["func_name"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
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
                indices: vec![0, 1],
                is_public: true,
            }
            .signature(),
            "read_integer[0, 1][true]"
        );
    }

    #[test]
    fn test_is_fixed() {
        assert!(!IR::AddI.is_fixed());
        assert!(!IR::ConstantInt { value: 0 }.is_fixed());
        assert!(IR::Assert.is_fixed());
        assert!(IR::ReadInteger {
            indices: vec![],
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
        assert_eq!(IR::PoseidonHash.class_name(), "PoseidonHashIR");
        assert_eq!(IR::EqHash.class_name(), "EqualHashIR");
    }

    #[test]
    fn test_export_import_round_trip_simple() {
        let irs = vec![
            IR::AddI,
            IR::SubF,
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
            IR::ConstantInt { value: 42 },
            IR::ConstantFloat { value: 3.14 },
            IR::ConstantBool { value: true },
            IR::ConstantStr {
                value: "hello".to_string(),
            },
            IR::ReadInteger {
                indices: vec![0, 1],
                is_public: true,
            },
            IR::ReadFloat {
                indices: vec![2],
                is_public: false,
            },
            IR::ReadHash {
                indices: vec![3],
                is_public: true,
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
            },
            IR::WitnessDynamicNDArrayMeta {
                array_id: 1,
                max_rank: 2,
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

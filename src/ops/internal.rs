//! Internal operators: select, assert, input, expose_public, export_external,
//! sign, poseidon_hash, implicit_type_cast, implicit_type_align, placeholder_value.
//! Ports `zinnia/op_def/internal/`.

#![allow(clippy::cloned_ref_to_slice_refs)]

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::{Value, ZinniaType};

// ═══════════════════════════════════════════════════════════════════════════
// SelectOp — conditional value selection
// ═══════════════════════════════════════════════════════════════════════════

pub struct SelectOp;

impl SelectOp {
    const PARAMS: [ParamEntry; 3] = [
        ParamEntry::required("cond"),
        ParamEntry::required("tv"),
        ParamEntry::required("fv"),
    ];
}

impl Op for SelectOp {
    fn name(&self) -> &'static str { "select" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let cond = args.require("cond");
        let tv = args.require("tv");
        let fv = args.require("fv");

        match (tv, fv) {
            (Value::Boolean(_), Value::Boolean(_)) => builder.ir_select_b(cond, tv, fv),
            (Value::Integer(_), Value::Integer(_))
            | (Value::Boolean(_), Value::Integer(_))
            | (Value::Integer(_), Value::Boolean(_)) => builder.ir_select_i(cond, tv, fv),
            (Value::Float(_), Value::Float(_)) => builder.ir_select_f(cond, tv, fv),
            (Value::None, Value::None) => Value::None,
            _ => panic!(
                "select: unsupported types {:?} and {:?}",
                tv.zinnia_type(),
                fv.zinnia_type()
            ),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AssertOp
// ═══════════════════════════════════════════════════════════════════════════

pub struct AssertOp;

impl AssertOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("test"),
        ParamEntry::optional("condition"),
    ];
}

impl Op for AssertOp {
    fn name(&self) -> &'static str { "assert" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let test = args.require("test");
        let asserted = match test {
            Value::Boolean(_) | Value::Integer(_) => test.clone(),
            _ => panic!("assert: test must be boolean or integer"),
        };
        builder.ir_assert(&asserted)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ExposePublicOp
// ═══════════════════════════════════════════════════════════════════════════

pub struct ExposePublicOp;

impl ExposePublicOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for ExposePublicOp {
    fn name(&self) -> &'static str { "expose_public" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_expose_public_i(x),
            Value::Float(_) => builder.ir_expose_public_f(x),
            _ => panic!("expose_public: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ExportExternalOp — stateful, carries for_which/key/indices
// ═══════════════════════════════════════════════════════════════════════════

pub struct ExportExternalOp {
    pub for_which: u32,
    pub key: ExternalKeyValue,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub enum ExternalKeyValue {
    Int(u32),
    Str(String),
}

impl ExportExternalOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for ExportExternalOp {
    fn name(&self) -> &'static str { "export_external" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        let key = match &self.key {
            ExternalKeyValue::Int(n) => crate::ir_defs::ExternalKey::Int(*n),
            ExternalKeyValue::Str(s) => crate::ir_defs::ExternalKey::Str(s.clone()),
        };
        match x {
            Value::Integer(_) | Value::Boolean(_) => {
                builder.create_ir(
                    &crate::ir_defs::IR::ExportExternalI {
                        for_which: self.for_which,
                        key,
                        indices: self.indices.clone(),
                    },
                    &[x.clone()],
                )
            }
            Value::Float(_) => {
                builder.create_ir(
                    &crate::ir_defs::IR::ExportExternalF {
                        for_which: self.for_which,
                        key,
                        indices: self.indices.clone(),
                    },
                    &[x.clone()],
                )
            }
            _ => panic!("export_external: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// InputOp — stateful, carries indices/dtype/kind
// ═══════════════════════════════════════════════════════════════════════════

pub struct InputOp {
    pub indices: Vec<u32>,
    pub dtype: ZinniaType,
    pub kind: String, // "PUBLIC", "PRIVATE", etc.
}

impl InputOp {
    const PARAMS: [ParamEntry; 0] = [];
}

impl Op for InputOp {
    fn name(&self) -> &'static str { "input" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, _args: &OpArgs) -> Value {
        let is_public = self.kind == "PUBLIC";
        match &self.dtype {
            ZinniaType::Integer => {
                builder.ir_read_integer(self.indices.clone(), is_public)
            }
            ZinniaType::Float => {
                builder.ir_read_float(self.indices.clone(), is_public)
            }
            ZinniaType::PoseidonHashed { .. } => {
                builder.ir_read_hash(self.indices.clone(), is_public)
            }
            _ => panic!("input: unsupported type {:?}", self.dtype),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SignOp
// ═══════════════════════════════════════════════════════════════════════════

pub struct SignOp;

impl SignOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for SignOp {
    fn name(&self) -> &'static str { "sign" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_sign_i(x),
            Value::Float(_) => builder.ir_sign_f(x),
            _ => panic!("sign: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PoseidonHashOp
// ═══════════════════════════════════════════════════════════════════════════

pub struct PoseidonHashOp;

impl PoseidonHashOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for PoseidonHashOp {
    fn name(&self) -> &'static str { "poseidon_hash" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) | Value::Float(_) => {
                builder.ir_poseidon_hash(&[x.clone()])
            }
            _ => panic!("poseidon_hash: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PlaceholderValueOp — creates a placeholder value for a given type
// ═══════════════════════════════════════════════════════════════════════════

pub struct PlaceholderValueOp {
    pub dtype: ZinniaType,
}

impl PlaceholderValueOp {
    const PARAMS: [ParamEntry; 0] = [];
}

impl Op for PlaceholderValueOp {
    fn name(&self) -> &'static str { "placeholder_value" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, _args: &OpArgs) -> Value {
        match &self.dtype {
            ZinniaType::Integer => builder.ir_constant_int(0),
            ZinniaType::Float => builder.ir_constant_float(0.0),
            ZinniaType::Boolean => builder.ir_constant_bool(false),
            _ => panic!("placeholder_value: unsupported type {:?}", self.dtype),
        }
    }
}

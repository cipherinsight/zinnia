#![allow(clippy::cloned_ref_to_slice_refs)]

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

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

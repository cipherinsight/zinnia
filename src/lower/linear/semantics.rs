//! What passes may know about a linear lane before it is lowered: a float,
//! a constant only where the frontend's own sum would have folded (a matrix
//! product whose pairs are all constants, `IRBuilder::ir_dot_f`; the
//! frontend emits no placeholder then, so this is a safeguard).

use std::ops::Range;

use crate::provenance::Interval;
use crate::types::{NumberType, Value};

use crate::lower::op::{ArrayOp, Constant, OperatorSemantics};

pub struct LinearSemantics;

impl OperatorSemantics for LinearSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        if !matches!(op.kind, crate::lower::OpKind::MatMul { .. }) {
            return None;
        }
        let pairs = super::lane_pairs(op.kind, op.position? as usize);
        let sum: Option<f64> = pairs.iter().map(|&(x, y)| Some(args[x].float_val()? * args[y].float_val()?)).sum();
        sum.map(Constant::Float)
    }

    fn number_type(&self, _op: &ArrayOp, _args: &[Value]) -> NumberType {
        NumberType::Float
    }

    fn bounds(&self, _op: &ArrayOp, _operands: &[Option<Interval>]) -> Option<Interval> {
        None
    }

    fn picks_from(&self, _op: &ArrayOp) -> Option<Range<usize>> {
        None
    }
}

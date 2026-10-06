//! What passes may know about a read lane before it is lowered.

use std::ops::Range;

use crate::lower::op::{ArrayOp, Constant, OperatorSemantics};
use crate::provenance::Interval;
use crate::types::{NumberType, Value};

/// `OpKind::Read`: the lane is one of the shared cells (the one at the
/// lane's own index operand).
pub struct ReadSemantics;

impl OperatorSemantics for ReadSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let n = op.shared as usize;
        let index = usize::try_from(args.get(n)?.int_val()?).ok().filter(|&i| i < n)?;
        match &args[index] {
            v @ Value::Float(_) => v.float_val().map(Constant::Float),
            v => v.int_val().map(Constant::Int).or_else(|| v.bool_val().map(Constant::Bool)),
        }
    }

    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType {
        dtype_of(&args[..op.shared as usize])
    }

    /// The read is one of the cells.
    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval> {
        operands[..op.shared as usize].iter().copied().reduce(|x, y| Some(Interval::union(x?, y?))).flatten()
    }

    fn picks_from(&self, op: &ArrayOp) -> Option<Range<usize>> {
        Some(0..op.shared as usize)
    }
}

/// The number type of a read of `cells`: float when any cell is a float.
pub(super) fn dtype_of(cells: &[Value]) -> NumberType {
    if cells.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

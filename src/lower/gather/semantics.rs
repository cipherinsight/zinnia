//! What passes may know about a gather-family lane before it is lowered.

use std::ops::Range;

use crate::lower::op::{ArrayOp, Constant, OpKind, OperatorSemantics};
use crate::provenance::Interval;
use crate::types::{NumberType, Value};

pub struct GatherSemantics;

impl OperatorSemantics for GatherSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let n = op.shared as usize;
        match op.kind {
            OpKind::FirstSet | OpKind::LastSet | OpKind::ArgmaxOfFlags => {
                let flags: Vec<bool> = args[..n].iter().map(constant_flag).collect::<Option<_>>()?;
                let set: Vec<usize> = (0..n).filter(|&i| flags[i]).collect();
                let position = match op.kind {
                    OpKind::FirstSet => *set.first()?,
                    OpKind::LastSet => *set.last()?,
                    _ => set.first().copied().unwrap_or(0),
                };
                Some(Constant::Int(position as i64))
            }
            OpKind::Window { rank, dims } => {
                let shape = &dims[..rank as usize];
                let cells = n - 2 * shape.len();
                let bounds: Vec<i64> = args[cells..n].iter().map(Value::int_val).collect::<Option<_>>()?;
                match window_source(shape, &bounds, op.position? as usize) {
                    Some(flat) => constant_of(&args[flat]),
                    None => Some(zero_like(&args[..cells])),
                }
            }
            OpKind::PermutationScatter => {
                let half = n / 2;
                let j = op.position? as i64;
                let k = (0..half).rev().find(|&k| args[k].int_val() == Some(j))?;
                args[..half].iter().all(|p| p.int_val().is_some()).then_some(())?;
                constant_of(&args[half + k])
            }
            _ => None,
        }
    }

    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType {
        let n = op.shared as usize;
        match op.kind {
            OpKind::Window { rank, .. } if is_float(&args[..n - 2 * rank as usize]) => NumberType::Float,
            OpKind::PermutationScatter if is_float(&args[n / 2..n]) => NumberType::Float,
            _ => NumberType::Integer,
        }
    }

    /// A position lies in `[0, n)`; a window cell is a source cell or 0; a
    /// scattered cell one of the values.
    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval> {
        let n = op.shared as usize;
        let union = |xs: &[Option<Interval>]| xs.iter().copied().reduce(|x, y| Some(Interval::union(x?, y?))).flatten();
        match op.kind {
            OpKind::FirstSet | OpKind::LastSet | OpKind::ArgmaxOfFlags => Some(Interval { lo: 0, hi: n as i64 - 1 }),
            OpKind::Window { rank, .. } => {
                let cells = union(&operands[..n - 2 * rank as usize])?;
                Some(Interval::union(cells, Interval::point(0)))
            }
            OpKind::PermutationScatter => union(&operands[n / 2..n]),
            _ => None,
        }
    }

    fn picks_from(&self, op: &ArrayOp) -> Option<Range<usize>> {
        match op.kind {
            OpKind::PermutationScatter => Some(op.shared as usize / 2..op.shared as usize),
            _ => None,
        }
    }
}

/// The flat source position read by output position `out` of the window
/// with `bounds = [start_0, stop_0, start_1, stop_1, ...]` over `shape`
/// (row-major), or `None` for padding.
pub(super) fn window_source(shape: &[u32], bounds: &[i64], out: usize) -> Option<usize> {
    let mut rest = out;
    let mut flat = 0usize;
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        let n = shape[axis] as usize;
        let u = rest % n;
        rest /= n;
        let (start, stop) = (bounds[2 * axis], bounds[2 * axis + 1]);
        let source = start + u as i64;
        if source >= stop.min(n as i64) {
            return None;
        }
        flat += source as usize * stride;
        stride *= n;
    }
    Some(flat)
}

fn constant_flag(v: &Value) -> Option<bool> {
    v.bool_val().or_else(|| v.int_val().map(|i| i != 0))
}

fn constant_of(v: &Value) -> Option<Constant> {
    match v {
        Value::Float(_) => v.float_val().map(Constant::Float),
        _ => v.int_val().map(Constant::Int).or_else(|| v.bool_val().map(Constant::Bool)),
    }
}

fn is_float(cells: &[Value]) -> bool {
    cells.iter().any(|v| matches!(v, Value::Float(_)))
}

fn zero_like(cells: &[Value]) -> Constant {
    if is_float(cells) {
        Constant::Float(0.0)
    } else {
        Constant::Int(0)
    }
}

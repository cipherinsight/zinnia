//! What passes may know about a sort lane before it is lowered.

use std::ops::Range;

use crate::provenance::Interval;
use crate::types::{NumberType, Value};

use crate::lower::op::{ArrayOp, Constant, OperatorSemantics};

/// The output of a sort that holds the stable argsort.
pub const INDICES: u8 = 1;
/// The output of a sort that holds the sorted values.
pub const VALUES: u8 = 0;

pub struct SortSemantics;

impl OperatorSemantics for SortSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let n = op.shared as usize;
        let rank = op.position.map(i64::from).or_else(|| args.get(n)?.int_val());
        fold_sorted(&args[..n], rank, op.output == INDICES)
    }

    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType {
        let elements = &args[..op.shared as usize];
        if op.output == VALUES && elements.iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        }
    }

    /// An index lies in `[0, n)`; a value is one of the elements.
    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval> {
        let n = op.shared as usize;
        if op.output == INDICES {
            return Some(Interval { lo: 0, hi: n as i64 - 1 });
        }
        operands[..n].iter().copied().reduce(|x, y| Some(Interval::union(x?, y?))).flatten()
    }

    fn picks_from(&self, op: &ArrayOp) -> Option<Range<usize>> {
        (op.output == VALUES).then_some(0..op.shared as usize)
    }
}

/// Output `rank` of NumPy's stable sort of constant `elements` (the value,
/// or with `index` its original position). `None` unless every element and
/// the rank are known and the rank is in range.
fn fold_sorted(elements: &[Value], rank: Option<i64>, index: bool) -> Option<Constant> {
    let rank = usize::try_from(rank?).ok().filter(|&r| r < elements.len())?;
    let as_int = |v: &Value| v.int_val().or_else(|| v.bool_val().map(i64::from));
    if elements.iter().any(|v| matches!(v, Value::Float(_))) {
        let keys: Vec<f64> = elements
            .iter()
            .map(|v| v.float_val().or_else(|| as_int(v).map(|i| i as f64)))
            .collect::<Option<_>>()?;
        if keys.iter().any(|k| k.is_nan()) {
            return None;
        }
        let position = stable_sort_position(&keys, rank, |a, b| a.partial_cmp(b).expect("no NaN"));
        Some(if index { Constant::Int(position as i64) } else { Constant::Float(keys[position]) })
    } else {
        let keys: Vec<i64> = elements.iter().map(as_int).collect::<Option<_>>()?;
        let position = stable_sort_position(&keys, rank, i64::cmp);
        Some(Constant::Int(if index { position as i64 } else { keys[position] }))
    }
}

/// The original position of the element of rank `rank` in a stable sort.
fn stable_sort_position<T>(keys: &[T], rank: usize, cmp: impl Fn(&T, &T) -> std::cmp::Ordering) -> usize {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&i, &j| cmp(&keys[i], &keys[j]));
    order[rank]
}

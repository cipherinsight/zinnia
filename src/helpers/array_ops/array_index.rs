//! Array-valued indices: a boolean mask, or a list of positions.
//!
//! NumPy reads `a[idx]` with an array-valued `idx` (an array, or a list,
//! which NumPy converts to one) by `idx`'s dtype: a bool `idx` is a mask that
//! keeps the elements where it is true; an integer `idx` lists the positions
//! to gather (a negative one counts from the end; each must be in range);
//! any other dtype is an `IndexError`. Every indexing site decides through
//! [`array_index_kind`]. The dtype comes from the builder's record of bool
//! arrays (`builder/bool_dtype.rs`), never from the index's values: a bool
//! array and an integer array of zeros and ones look alike in the circuit.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

/// How NumPy reads `a[idx]` for an array-valued `idx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayIndexKind {
    /// A bool array: keep the elements where it is true.
    Mask,
    /// An integer array: gather the elements at these positions.
    Positions,
}

/// How NumPy reads `a[idx]`, or `None` when `idx` is not array-valued (a
/// scalar, a slice, or a tuple, which indexes one axis per entry). Panics
/// with NumPy's `IndexError` for an array that is neither bool nor integer.
pub fn array_index_kind(b: &IRBuilder, idx: &Value) -> Option<ArrayIndexKind> {
    let kind = match idx {
        Value::List(_) => list_index_kind(b, idx)?,
        Value::StaticArray { dtype, .. } => array_kind(*dtype, is_bool_array(b, idx)),
        Value::DynamicNDArray(d) => array_kind(d.dtype, is_bool_array(b, idx)),
        _ => return None,
    };
    Some(kind)
}

/// `a[m, :, ..., :]` and `a[m, ...]` are `a[m]`: the leading index when it is
/// a boolean mask or a runtime-shaped index array and every other index is a
/// full slice or an ellipsis. (Integer positions with trailing slices keep
/// the several-index gather of `static_array::read`.)
pub fn leading_index_array<'a>(
    b: &IRBuilder,
    indices: &'a [crate::types::SliceIndex],
) -> Option<&'a Value> {
    use crate::types::SliceIndex;
    let (SliceIndex::Single(first), rest) = indices.split_first()? else {
        return None;
    };
    let absent = |v: &Option<Value>| matches!(v, None | Some(Value::None));
    let all_full = rest.iter().all(|idx| match idx {
        SliceIndex::Range(start, stop, step) => absent(start) && absent(stop) && absent(step),
        SliceIndex::Ellipsis => true,
        _ => false,
    });
    let reads_alone = match array_index_kind(b, first)? {
        ArrayIndexKind::Mask => true,
        ArrayIndexKind::Positions => matches!(first, Value::DynamicNDArray(_)),
    };
    (!rest.is_empty() && all_full && reads_alone).then_some(first)
}

/// Reject index arrays among several indices of a static array that the
/// gather of `static_array::read` cannot read: a boolean mask (NumPy reads
/// `a[m, 1]` as `a[np.nonzero(m)[0], 1]`) and a runtime-shaped index array.
/// Without this check both were read as something else (a mask as integer
/// positions once its cells are integers).
pub fn reject_unsupported_index_arrays(b: &IRBuilder, indices: &[crate::types::SliceIndex]) {
    if indices.len() < 2 {
        return;
    }
    for idx in indices {
        let crate::types::SliceIndex::Single(v) = idx else { continue };
        match array_index_kind(b, v) {
            Some(ArrayIndexKind::Mask) => {
                panic!("a boolean mask combined with other indices is not supported")
            }
            Some(_) if matches!(v, Value::DynamicNDArray(_)) => panic!(
                "a runtime-shaped index array combined with other indices is not supported"
            ),
            _ => {}
        }
    }
}

/// Whether `v` is a NumPy bool array: a list whose every leaf is a bool cell
/// (`np.array` of it is bool), or an array stored in a bool segment. (An
/// array's dtype is its segment's, whatever its cells hold now: an integer
/// array stays one after bools are assigned into it.)
pub fn is_bool_array(b: &IRBuilder, v: &Value) -> bool {
    match v {
        Value::List(_) | Value::Tuple(_) => {
            let leaves = crate::helpers::composite::flatten_composite(v);
            !leaves.is_empty() && leaves.iter().all(|leaf| b.is_bool_cell(leaf))
        }
        Value::StaticArray { segment_id, .. } => b.is_bool_segment(*segment_id),
        Value::DynamicNDArray(d) => b.is_bool_segment(d.segment_id),
        _ => false,
    }
}

fn array_kind(dtype: NumberType, is_bool: bool) -> ArrayIndexKind {
    match dtype {
        NumberType::Integer if is_bool => ArrayIndexKind::Mask,
        NumberType::Integer => ArrayIndexKind::Positions,
        NumberType::Float | NumberType::Complex => not_an_index_array(),
    }
}

/// A list index: NumPy converts it to an array, whose dtype is bool when
/// every element is a bool and integer when the elements are integers or
/// bools (`[True, 2]` is the integer array `[1, 2]`). A list of anything
/// but numbers (e.g. of arrays) is not handled here.
fn list_index_kind(b: &IRBuilder, idx: &Value) -> Option<ArrayIndexKind> {
    let leaves = crate::helpers::composite::flatten_composite(idx);
    if !leaves
        .iter()
        .all(|v| matches!(v, Value::Integer(_) | Value::Boolean(_) | Value::Float(_)))
    {
        return None;
    }
    if leaves.iter().any(|v| matches!(v, Value::Float(_))) {
        not_an_index_array();
    }
    if !leaves.is_empty() && leaves.iter().all(|leaf| b.is_bool_cell(leaf)) {
        Some(ArrayIndexKind::Mask)
    } else {
        Some(ArrayIndexKind::Positions)
    }
}

fn not_an_index_array() -> ! {
    panic!("IndexError: arrays used as indices must be of integer (or boolean) type")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::InputPath;
    use crate::types::CompositeData;

    fn list(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, values))
    }

    fn runtime_int(b: &mut IRBuilder, name: &str) -> Value {
        b.ir_read_integer(InputPath::new(name, vec![]), false)
    }

    #[test]
    fn list_dtype_decides() {
        let mut b = IRBuilder::new();
        let x = runtime_int(&mut b, "x");
        let zero = b.ir_constant_int(0);
        let flag = b.ir_less_than_i(&x, &zero);
        let one = b.ir_constant_int(1);
        assert_eq!(array_index_kind(&b, &list(vec![flag.clone(), flag.clone()])), Some(ArrayIndexKind::Mask));
        // Integers that happen to be 0 or 1 are positions, never a mask.
        assert_eq!(array_index_kind(&b, &list(vec![one.clone(), zero])), Some(ArrayIndexKind::Positions));
        assert_eq!(array_index_kind(&b, &list(vec![x])), Some(ArrayIndexKind::Positions));
        // NumPy reads `[True, 1]` as the integer array `[1, 1]`.
        assert_eq!(array_index_kind(&b, &list(vec![flag, one])), Some(ArrayIndexKind::Positions));
        let three = b.ir_constant_int(3);
        assert_eq!(array_index_kind(&b, &three), None);
    }

    #[test]
    #[should_panic(expected = "must be of integer (or boolean) type")]
    fn float_list_is_not_an_index() {
        let mut b = IRBuilder::new();
        let f = b.ir_constant_float(1.0);
        array_index_kind(&b, &list(vec![f]));
    }

    #[test]
    fn bool_dtype_survives_storage() {
        let mut b = IRBuilder::new();
        let x = runtime_int(&mut b, "x");
        let zero = b.ir_constant_int(0);
        let flags: Vec<Value> = (0..3).map(|_| b.ir_less_than_i(&x, &zero)).collect();
        let mask = crate::helpers::static_array::base::to_static_array(&mut b, &list(flags)).unwrap();
        assert_eq!(array_index_kind(&b, &mask), Some(ArrayIndexKind::Mask));

        // Cells read back from the bool segment (no cached payload) are
        // integers in the IR but still bools.
        let Value::StaticArray { segment_id, .. } = &mask else { unreachable!() };
        b.static_array_payload.remove(segment_id);
        let reread = crate::helpers::static_array::base::to_value_list(&mut b, &mask);
        assert_eq!(array_index_kind(&b, &reread), Some(ArrayIndexKind::Mask));
        let copy = crate::helpers::static_array::base::to_static_array(&mut b, &reread).unwrap();
        assert_eq!(array_index_kind(&b, &copy), Some(ArrayIndexKind::Mask));

        // A runtime-shaped copy keeps the dtype too.
        let promoted = crate::helpers::promote::promote_static_to_dynamic(&mut b, &reread);
        assert_eq!(array_index_kind(&b, &promoted), Some(ArrayIndexKind::Mask));

        // Integer arrays stay positions however they are stored.
        let ints: Vec<Value> = (0..3).map(|i| runtime_int(&mut b, &format!("i{i}"))).collect();
        let ints = crate::helpers::static_array::base::to_static_array(&mut b, &list(ints)).unwrap();
        assert_eq!(array_index_kind(&b, &ints), Some(ArrayIndexKind::Positions));
        let ints = crate::helpers::static_array::base::to_value_list(&mut b, &ints);
        let promoted = crate::helpers::promote::promote_static_to_dynamic(&mut b, &ints);
        assert_eq!(array_index_kind(&b, &promoted), Some(ArrayIndexKind::Positions));
    }

    #[test]
    fn integer_conversion_ends_the_bool_dtype() {
        let mut b = IRBuilder::new();
        let x = runtime_int(&mut b, "x");
        let zero = b.ir_constant_int(0);
        let flag = b.ir_less_than_i(&x, &zero);
        let mask = crate::helpers::static_array::base::to_static_array(&mut b, &list(vec![flag.clone(); 2])).unwrap();
        let Value::StaticArray { segment_id, .. } = &mask else { unreachable!() };
        b.static_array_payload.remove(segment_id);
        // Cells read back from memory: integers in the IR, bools in NumPy.
        let reread = crate::helpers::static_array::base::to_value_list(&mut b, &mask);
        let cells = crate::helpers::composite::flatten_composite(&reread);
        assert!(cells.iter().all(|c| matches!(c, Value::Integer(_)) && b.is_bool_cell(c)));
        // `astype(int)` of them lists positions.
        let ints: Vec<Value> = cells.iter().map(|c| b.int_dtype_cell(c)).collect();
        assert_eq!(array_index_kind(&b, &list(ints)), Some(ArrayIndexKind::Positions));
        // The integer view of a `Boolean` (`ir_int_cast`) is an integer, even
        // when something tries to record it; as a bool array's cell it is
        // the `Boolean` again.
        let view = b.ir_int_cast(&flag);
        b.record_bool_cell(&view);
        assert!(!b.is_bool_cell(&view));
        assert!(matches!(b.bool_dtype_cell(&view), Value::Boolean(_)));
    }
}

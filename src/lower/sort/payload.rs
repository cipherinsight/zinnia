//! Payload lanes: an array gathered by a stable argsort (`b[np.argsort(a)]`,
//! `b[np.argsort(a, axis=0), j, k]`) as an output of the sort itself.
//!
//! Construction (inside every sort plan, see `cost::Demand::payloads`):
//! * Preconditions: the payload has one entry per sorted element.
//! * Postconditions: payload lane `r` is `p[argsort[r]]`.
//! * Emission: a network exchanges the payload's slots on the same
//!   comparisons as the keys (two selects per comparator in the cone of the
//!   live payload ranks; the network is a stable one); a hinted plan reads
//!   the payload at the pinned argsort index of rank `r`.
//! * Cost: halo2-base 12 cells, Noir 2 gates per carried comparator; a
//!   hinted plan pays one read per live rank.
//! * Soundness: a network's comparisons are exact and every select moves a
//!   payload slot together with its key, so slot `r` holds the payload entry
//!   of the element whose key ends at `r`, which a stable network makes the
//!   element of rank `r` in NumPy's stable order. A hinted plan's index is
//!   pinned to that element (`emit::pinned_run`) and bounded to `[0, n)`.
//!   No new hint.

use crate::builder::IRBuilder;
use crate::lower::{ArrayOp, OpKind};
use crate::types::Value;

use super::semantics::{INDICES, PAYLOAD};

/// `payload[index]` as a payload lane of the sort, when `index` is the
/// stable argsort's lane of a constant rank `r` of a sort of as many
/// elements as `payload` has entries; `None` otherwise.
pub fn payload_at_argsort_lane(b: &mut IRBuilder, index: &Value, payload: &[Value]) -> Option<Value> {
    let placeholder = b.array_op_placeholder_of(index)?;
    if placeholder.op.kind != OpKind::Sort || placeholder.op.output != INDICES {
        return None;
    }
    let rank = placeholder.op.position?;
    let elements = placeholder.shared.clone();
    if elements.len() != payload.len() {
        return None;
    }
    let payload: Vec<Value> = payload.iter().map(|v| b.ensure_ptr(v)).collect();
    let op = ArrayOp { kind: OpKind::Sort, shared: elements.len() as u32, output: PAYLOAD, position: Some(rank) };
    Some(b.ir_array_op(op, &elements, &payload))
}

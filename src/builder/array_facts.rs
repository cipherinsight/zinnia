//! Facts about an array's elements, and when they stop holding.
//!
//! A fact such as "every element is 0" (`forall_eq_const`), "the array is
//! sorted" (`is_sorted`), "the array is the identity" (`is_identity`) or
//! "every element lies in `[lo, hi]`" (`array_in_range`) is a `FactStack`
//! fact keyed by the array's value id. A write through the name it is bound
//! to gives the array a new value id (`after_write`), but a write through
//! another name for the same array (`y = x; y[0] = 5`, a view, a loop
//! variable) changes `x`'s elements and keeps its value id. The fact then
//! describes elements the array no longer holds.
//!
//! So whoever states such a fact also records the array's storages and the
//! store's write clock ([`ArrayContents`]). A fact about an array whose
//! storages were written since is not used: `optim::prove` leaves it out of
//! every query ([`IRBuilder::fact_is_current`]), and the direct readers of
//! `array_in_range` check [`IRBuilder::array_facts_are_current`]. Store writes reach every storage
//! linked to the written one, so writes through views count as well.
//!
//! The first statement about an array sets its record and a later one does
//! not replace it. A later fact stated after a write is then left unused
//! too, which only costs the optimization: replacing the record would bring
//! back the earlier facts, which describe the elements before the write.

use crate::optim::predicates::formula::{ContractTerm, ContractVar};
use crate::types::{Value, ValueId};

use super::IRBuilder;

/// What an array held at some point: its storages (a complex static array
/// has two) and the store's write clock then.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ArrayContents {
    storages: Vec<u32>,
    clock: u64,
}

impl ArrayContents {
    /// The contents of `arr` now; `None` for a value without storage (a
    /// list or a scalar).
    pub(crate) fn of(b: &IRBuilder, arr: &Value) -> Option<ArrayContents> {
        let storage = match arr {
            Value::StaticArray { segment_id, .. } => *segment_id,
            Value::DynamicNDArray(d) => d.segment_id,
            _ => return None,
        };
        let storages = std::iter::once(storage).chain(b.imag_storage(storage)).collect();
        Some(ArrayContents { storages, clock: b.storage_write_clock() })
    }

    /// The array still holds what it held when this was recorded.
    pub(crate) fn unchanged(&self, b: &IRBuilder) -> bool {
        self.storages.iter().all(|&s| !b.storage_written_since(s, self.clock))
    }
}

impl IRBuilder {
    /// Fire `contract`, whose postconditions are facts about the elements
    /// of its output, on the array `arr`, and record what `arr` holds now.
    pub(crate) fn state_array_fact(&mut self, arr: &Value, contract: &str) {
        let Some(vid) = arr.value_id() else { return };
        self.record_array_contents(arr);
        self.fire_contract(contract, vid, &std::collections::HashMap::new());
    }

    /// Record what `arr` holds now, before stating a fact about its
    /// elements, unless a fact was stated about it before (see the module
    /// docs). Nothing to record for a value without storage: a list is
    /// rebound, not written in place.
    pub(crate) fn record_array_contents(&mut self, arr: &Value) {
        let (Some(vid), Some(contents)) = (arr.value_id(), ArrayContents::of(self, arr)) else {
            return;
        };
        self.array_fact_records.entry(vid).or_insert(contents);
    }

    /// Whether facts about the elements of the array with value id `vid`
    /// still hold: true when none was recorded (a value that is not an
    /// array with storage), else whether its storages are unwritten since.
    pub(crate) fn array_facts_are_current(&self, vid: ValueId) -> bool {
        self.array_fact_records.get(&vid).is_none_or(|contents| contents.unchanged(self))
    }

    /// Whether `fact` still holds as far as array writes go: every
    /// predicate in it (`forall_eq_const(x, 0)`, `is_sorted(x)`, ...) is
    /// about arrays whose element facts are current. Facts about scalars
    /// never lapse: a scalar is not written in place.
    pub(crate) fn fact_is_current(&self, fact: &ContractTerm) -> bool {
        match fact {
            ContractTerm::PredicateApp { args, .. } => args.iter().all(|arg| match arg {
                ContractTerm::Var(ContractVar::Value(vid)) => self.array_facts_are_current(*vid),
                _ => true,
            }),
            ContractTerm::BoolComb { operands, .. } => operands.iter().all(|o| self.fact_is_current(o)),
            ContractTerm::Not(inner) => self.fact_is_current(inner),
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::IRBuilder;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::types::{NumberType, ScalarValue, Value};

    /// A one-dimensional runtime-shaped integer array of `len` zeros.
    fn zeros(b: &mut IRBuilder, len: usize) -> Value {
        let cells = vec![ScalarValue::new(Some(0), None); len];
        let length = b.ir_constant_int(len as i64);
        crate::ops::dyn_ndarray::constructors::dyn_from_values_with_active_nd(
            b,
            cells,
            vec![len],
            vec![ScalarValue::new(Some(len as i64), None)],
            length,
            NumberType::Integer,
        )
    }

    fn all_equal(arr: &Value, k: i64) -> ContractTerm {
        ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![ContractTerm::Var(ContractVar::Value(arr.value_id().unwrap())), ContractTerm::LitInt(k)],
        }
    }

    /// Write 5 into the first cell, as a write through another name for the
    /// array does: the storage changes, the value id stays.
    fn write_through_alias(b: &mut IRBuilder, arr: &Value) {
        let Value::DynamicNDArray(data) = arr else { unreachable!() };
        let address = b.ir_constant_int(0);
        let five = b.ir_constant_int(5);
        b.storage_write_at(data.segment_id, &address, &five);
    }

    #[test]
    fn a_fact_about_a_written_array_is_not_proved() {
        let mut b = IRBuilder::new();
        let x = zeros(&mut b, 4);
        b.state_array_fact(&x, "zeros_content");
        assert_eq!(b.prove(&all_equal(&x, 0)), ProveOutcome::Proved);
        write_through_alias(&mut b, &x);
        assert_eq!(b.prove(&all_equal(&x, 0)), ProveOutcome::Unknown);
    }

    #[test]
    fn a_relay_from_a_written_array_states_nothing() {
        let mut b = IRBuilder::new();
        let x = zeros(&mut b, 4);
        b.state_array_fact(&x, "zeros_content");
        write_through_alias(&mut b, &x);
        let copy = zeros(&mut b, 4);
        assert!(!crate::optim::resolver::relay_forall_eq_const_from_input(&mut b, &x, &copy));
        assert_eq!(b.prove(&all_equal(&copy, 0)), ProveOutcome::Unknown);
    }

    #[test]
    fn a_relayed_fact_lapses_when_its_own_array_is_written() {
        let mut b = IRBuilder::new();
        let x = zeros(&mut b, 4);
        b.state_array_fact(&x, "zeros_content");
        let copy = zeros(&mut b, 4);
        assert!(crate::optim::resolver::relay_forall_eq_const_from_input(&mut b, &x, &copy));
        write_through_alias(&mut b, &copy);
        assert_eq!(b.prove(&all_equal(&copy, 0)), ProveOutcome::Unknown);
        assert_eq!(b.prove(&all_equal(&x, 0)), ProveOutcome::Proved, "the source was not written");
    }

    #[test]
    fn a_later_fact_does_not_bring_back_an_earlier_one() {
        let mut b = IRBuilder::new();
        let x = zeros(&mut b, 4);
        b.state_array_fact(&x, "zeros_content");
        write_through_alias(&mut b, &x);
        b.state_array_fact(&x, "ones_content");
        assert_eq!(b.prove(&all_equal(&x, 0)), ProveOutcome::Unknown);
        assert_eq!(b.prove(&all_equal(&x, 1)), ProveOutcome::Unknown);
    }
}

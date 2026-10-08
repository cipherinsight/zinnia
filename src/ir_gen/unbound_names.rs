//! Checks made where a program uses a name.
//!
//! Python binds a local name exactly where one of its assignments ran;
//! reading it elsewhere raises `UnboundLocalError`. Zinnia's scopes keep a
//! name bound after the scope that first bound it, which is exact except
//! for a first binding in a loop body that the loop may leave before it (a
//! `break` or `continue` earlier in it or in an enclosing loop): after it
//! the name is bound only on some paths. `IRContext` records the execution
//! predicates of the assignments to such a name, and a read of it asserts
//! that one of them held, so no proof exists for an input on which Python
//! raises (review 3, B4). The assertion is left out where the read's own
//! predicate implies an assignment's, as for a read right after the
//! assignment in the same iteration.

use crate::types::Value;

use super::IRGenerator;

impl IRGenerator {
    /// Assert, where the current code runs, that `name` is bound (see the
    /// module documentation). Nothing is emitted for a name bound on every
    /// path, or where the read's predicate implies an assignment's.
    pub(crate) fn check_bound(&mut self, name: &str) {
        let Some(predicates) = self.ctx.possibly_unbound(name).map(<[Value]>::to_vec) else { return };
        if let Some(here) = self.builder.write_predicate().cloned() {
            if predicates.iter().any(|p| self.builder.predicate_implies(&here, p)) {
                return;
            }
        }
        let bound = predicates[1..]
            .iter()
            .fold(predicates[0].clone(), |any, p| self.builder.ir_logical_or(&any, p));
        self.builder.ir_assert(&bound);
    }

    /// The value of the variable `name` for a use that does not read it
    /// through a bare name (a method or attribute of it, a write to an
    /// element of it, `name op= value`): the name must be bound, and its
    /// array must have a known shape here. A stand-in for arrays of
    /// different shapes is resolved to the array the current path selects
    /// (see `detached_copies::resolve_different_shapes`); otherwise the use
    /// is a compile error. The cells may be stale; a read of them is
    /// rejected by the store. `None` when the name is not bound at all.
    pub(crate) fn usable_name_value(&mut self, name: &str) -> Option<Value> {
        let value = self.ctx.get(name)?;
        self.check_bound(name);
        let value = self.resolve_different_shapes(value);
        if let Some(reason) = self.builder.unknown_shape_reason_in(&value) {
            panic!("Using `{}` is not supported here: {}", name, reason);
        }
        Some(value)
    }
}

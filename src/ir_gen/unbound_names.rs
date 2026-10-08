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

use std::collections::{BTreeMap, BTreeSet};

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

    /// The records of possibly unbound names after the join of a runtime
    /// `if` that runs under `predicate` (`None`: every path): `before` the
    /// `if`, after each arm (`arms`, each with its condition and the names
    /// it assigns). A name bound in one arm only and not before is dropped
    /// at the join (Zinnia does not keep it), so its record goes too, unless
    /// the join keeps the first arm's names (`keeps_first_arm_names`). Each
    /// arm contributes the predicates it added; an arm that binds the name
    /// on every path through it (an assignment whose predicate is the arm's
    /// own) contributes that. Where both arms do, the name is bound wherever
    /// the `if` ran: the join's predicate replaces the arms' (review 4, FR4:
    /// an arm's plain assignment was not counted when the other arm started
    /// the record in a loop that may stop early).
    pub(crate) fn join_assignment_records(
        &mut self,
        predicate: Option<&Value>,
        before: &BTreeMap<String, Vec<Value>>,
        arms: [(&Value, &BTreeMap<String, Vec<Value>>, &BTreeSet<String>); 2],
        keeps_first_arm_names: bool,
    ) -> BTreeMap<String, Vec<Value>> {
        let mut joined = before.clone();
        let names: BTreeSet<String> = arms.iter().flat_map(|(_, records, _)| records.keys().cloned()).collect();
        for name in names {
            let bound_before = self.ctx.exists(&name);
            if !bound_before && !arms.iter().all(|(_, _, assigned)| assigned.contains(&name)) {
                // Kept where the first arm's assignments ran (`visit_branches`).
                match arms[0].1.get(&name) {
                    Some(predicates) if keeps_first_arm_names && arms[0].2.contains(&name) => {
                        joined.insert(name, predicates.clone());
                    }
                    _ => {
                        joined.remove(&name);
                    }
                }
                continue;
            }
            if bound_before && !before.contains_key(&name) {
                continue; // bound on every path before the `if`
            }
            let earlier = before.get(&name).cloned().unwrap_or_default();
            let mut predicates = earlier.clone();
            let mut everywhere = true;
            for (condition, records, assigned) in arms {
                let added: Vec<Value> = records.get(&name).map(|p| p[earlier.len().min(p.len())..].to_vec()).unwrap_or_default();
                everywhere &= assigned.contains(&name)
                    && added.iter().any(|p| self.builder.conjunction_implies(&[predicate, Some(condition)], p));
                predicates.extend(added);
            }
            if everywhere {
                let on_every_path = predicate.is_none_or(|p| p.bool_val() == Some(true) || p.int_val().is_some_and(|v| v != 0));
                if on_every_path && earlier.is_empty() {
                    joined.remove(&name);
                    continue;
                }
                predicates = earlier;
                if let Some(p) = predicate {
                    predicates.push(p.clone());
                }
            }
            joined.insert(name, predicates);
        }
        joined
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

    /// Whether `name` is bound to a stand-in for arrays of different shapes
    /// that [`Self::usable_name_value`] resolved to `resolved`. A write
    /// through the name then goes into `resolved` in place, and the name
    /// stays bound to the stand-in: rebinding it to `resolved` would drop
    /// the other paths' arrays.
    pub(crate) fn binds_stand_in(&self, name: &str, resolved: &Value) -> bool {
        let binding = self.ctx.get(name).as_ref().and_then(super::detached_copies::array_segment);
        binding.is_some() && binding != super::detached_copies::array_segment(resolved)
    }
}

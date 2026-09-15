//! The record invariants of `implementation.md` §5a, checked on a whole
//! [`Phase0Result`].
//!
//! Every walk test goes through [`super::fixtures::run_full`], which calls
//! [`assert_record_invariants`] on the record it produces, so a rule or a
//! handler that forgets a producer, an operand or a subject fails the test
//! that exercises it. `testing/phase0/record.py` is the same four checks over
//! the exported JSON, which the coverage harness runs on every accepted
//! circuit of the parity set.

use std::collections::HashSet;

use crate::phase0::desc::{Desc, ValueKey};
use crate::phase0::phi::ConstraintKind;
use crate::phase0::source::Producer;
use crate::phase0::Phase0Result;

/// Check the four invariants of $\Delta$ and panic with the offending key if
/// one fails.
///
/// - I1 every entry has a producer; an `Applied` producer of a scalar or an
///   array names at least one rule; `Opaque` stands only for a value with no
///   descriptor content.
/// - I2 every operand and every alias resolves to an entry recorded *earlier*
///   in $\Delta$, so the traversal order is a topological order of the source
///   graph.
/// - I3 every check and every obligation has a subject that resolves to an
///   entry.
/// - I4 keys are unique.
pub(super) fn assert_record_invariants(result: &Phase0Result) {
    let mut seen: HashSet<&ValueKey> = HashSet::new();
    for entry in &result.descriptors {
        check_producer(entry.key.encode(), &entry.desc, &entry.producer);
        // Checked before the key joins `seen`, so a value that names itself
        // is a violation and not a cycle of length one.
        for input in entry.producer.inputs() {
            assert!(
                seen.contains(input),
                "I2: the producer of {} names {}, which Δ does not record before it",
                entry.key.encode(),
                input.encode()
            );
        }
        assert!(
            seen.insert(&entry.key),
            "I4: two values share the key {}",
            entry.key.encode()
        );
    }
    for constraint in result.phi.constraints() {
        if let ConstraintKind::Check { subject, .. } = &constraint.kind {
            assert!(
                seen.contains(subject),
                "I3: the check from {} points at {}, which Δ does not record",
                constraint.provenance,
                subject.encode()
            );
        }
    }
    for obligation in result.obligations() {
        assert!(
            seen.contains(&obligation.subject),
            "I3: the {} obligation from {} points at {}, which Δ does not record",
            obligation.kind.as_str(),
            obligation.provenance,
            obligation.subject.encode()
        );
    }
}

/// I1 for one entry.
fn check_producer(key: String, desc: &Desc, producer: &Producer) {
    let opaque = matches!(desc, Desc::Class(_) | Desc::None);
    match producer {
        Producer::Opaque => assert!(
            opaque,
            "I1: {} is an entry of kind {} and may not be opaque",
            key,
            desc.kind_str()
        ),
        Producer::Applied { rules, .. } => assert!(
            !rules.is_empty() || matches!(desc, Desc::Composite(_)) || opaque,
            "I1: {} names no rule",
            key
        ),
        Producer::Input | Producer::Alias(_) => {}
    }
}

// ---------------------------------------------------------------------------
// The record itself: the tests of what step 7 added
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::fixtures::*;
    use crate::phase0::phi::ConstraintKind;
    use crate::phase0::source::Producer;
    use crate::phase0::Phase0Result;

    /// The entry recorded under `key`, by its canonical string.
    fn entry_at<'r>(
        result: &'r Phase0Result,
        key: &str,
    ) -> &'r crate::phase0::DescriptorEntry {
        result
            .descriptors
            .iter()
            .find(|entry| entry.key.encode() == key)
            .unwrap_or_else(|| {
                panic!(
                    "no entry at {}; Δ has {:?}",
                    key,
                    result
                        .descriptors
                        .iter()
                        .map(|e| e.key.encode())
                        .collect::<Vec<_>>()
                )
            })
    }

    /// §1: Δ maps every SSA value, so an expression that binds no name has an
    /// entry too — here `a[1:3]`, which is the right-hand side of `s`.
    #[test]
    fn an_intermediate_expression_has_an_entry() {
        // s = a[1:3] + 1; return s
        let sliced = json!({"__class__": "ASTSubscriptExp", "val": load("a"),
                            "slicing": {"__class__": "ASTSlice",
                                        "data": [[int(1), int(3), {"__class__": "ASTConstantNone"}]]}});
        let result = run(
            vec![array_input("a", &[8])],
            vec![
                assign(name_target("s"), binop("add", sliced, int(1))),
                ret(load("s")),
            ],
        )
        .expect("a slice of a static array");
        // Statement 0: expression 0 is `a[1:3] + 1`, expression 1 is `a[1:3]`.
        let entry = entry_at(&result, "0|||1");
        assert_eq!(entry.name, None, "an intermediate has no name");
        let array = entry.desc.as_array().expect("a rank-1 array");
        assert_eq!(result.terms.constant_value(array.dims[0]), Some(2));
        match &entry.producer {
            Producer::Applied { rules, operands, .. } => {
                assert_eq!(rules.first().map(|r| r.name()), Some("E-Slice"));
                assert_eq!(
                    operands.first().map(|k| k.encode()),
                    Some("0|||2".to_string()),
                    "the sliced array is the first operand"
                );
            }
            other => panic!("expected a rule application, got {:?}", other),
        }
    }

    /// §5a rule 5: the merge takes one key per merged name, and its producer
    /// names both arms and the condition.
    #[test]
    fn the_merge_of_an_if_names_both_arms_and_the_condition() {
        // if k > 0: x = a[0:2] else: x = a[0:3]; return x
        let slice = |hi: i64| {
            json!({"__class__": "ASTSubscriptExp", "val": load("a"),
                   "slicing": {"__class__": "ASTSlice",
                               "data": [[int(0), int(hi), {"__class__": "ASTConstantNone"}]]}})
        };
        let result = run(
            vec![array_input("a", &[8]), scalar_input("k")],
            vec![
                cond(
                    binop("gt", load("k"), int(0)),
                    vec![assign(name_target("x"), slice(2))],
                    vec![assign(name_target("x"), slice(3))],
                ),
                ret(load("x")),
            ],
        )
        .expect("both arms give a rank-1 integer array");
        // The merge keys follow the two arms, one per merged name in sorted
        // order; the merge of `x` is the one that names an array merge.
        let merged = result
            .descriptors
            .iter()
            .find(|entry| {
                entry.name.as_deref() == Some("x")
                    && matches!(&entry.producer,
                        Producer::Applied { rules, .. }
                            if rules.iter().any(|r| r.name() == "merge-array"))
            })
            .expect("the merge of `x` applies the array merge of §6.1");
        let Producer::Applied {
            operands,
            condition,
            ..
        } = &merged.producer
        else {
            panic!("expected a merge, got {:?}", merged.producer)
        };
        let keys: Vec<String> = operands.iter().map(|k| k.encode()).collect();
        assert_eq!(keys.len(), 2, "one operand per arm, got {:?}", keys);
        assert!(
            keys[0].starts_with("0.then.") && keys[1].starts_with("0.else."),
            "the operands are the two arms' bindings, got {:?}",
            keys
        );
        assert!(condition.is_some(), "a merge records its condition");
        for key in operands {
            assert!(
                result.descriptor(key).is_some(),
                "each arm's binding is an entry of Δ"
            );
        }
    }

    /// §5a: a check points at the value whose operation it belongs to — here
    /// the indexing expression `a[k]`.
    #[test]
    fn a_check_points_at_the_indexing_expression() {
        // s = a[k]; return s
        let result = run(
            vec![array_input("a", &[4]), scalar_input("k")],
            vec![
                assign(name_target("s"), subscript(load("a"), load("k"))),
                ret(load("s")),
            ],
        )
        .expect("an element read of a static array");
        let subject = result
            .phi
            .constraints()
            .iter()
            .find_map(|c| match (&c.kind, c.provenance.as_str()) {
                (ConstraintKind::Check { subject, .. }, "E-Elem") => Some(subject.clone()),
                _ => None,
            })
            .expect("the element read records a bound check");
        assert_eq!(subject.encode(), "0|||0", "the subscript expression");
        assert!(
            result.descriptor(&subject).is_some(),
            "the subject resolves to an entry"
        );
    }

    /// §5a: a list method's rebinding of its receiver takes one key of its
    /// own, and the new version points at the previous one.
    #[test]
    fn a_list_method_rebinds_the_receiver_under_a_new_key() {
        // v = [1, 2]; v.append(3); return v
        let literal = json!({"__class__": "ASTSquareBrackets", "values": [int(1), int(2)]});
        let append = json!({"__class__": "ASTExprStatement",
                            "expr": {"__class__": "ASTNamedAttribute", "target": "v",
                                     "member": "append", "args": [int(3)], "kwargs": {}}});
        let result = run(
            vec![array_input("a", &[2])],
            vec![
                assign(name_target("v"), literal),
                append,
                ret(load("v")),
            ],
        )
        .expect("a list literal and an append");
        // Statement 1: expression 0 is the call, 1 its argument, and the
        // receiver's new version takes the key after them.
        let rebound = entry_at(&result, "1|||2");
        let composite = rebound.desc.as_composite().expect("a list");
        assert_eq!(composite.elems.len(), 3, "append added an element");
        match &rebound.producer {
            Producer::Applied { rules, operands, .. } => {
                assert_eq!(rules.iter().map(|r| r.name()).collect::<Vec<_>>(),
                           vec!["S-ListMethod"]);
                assert_eq!(
                    operands.first().map(|k| k.encode()),
                    Some("0|||3".to_string()),
                    "the receiver's previous version"
                );
            }
            other => panic!("expected S-ListMethod, got {:?}", other),
        }
    }

    /// §5a: a use of a name is the bound value under a new key.
    #[test]
    fn a_name_reference_is_an_alias_of_its_binding() {
        // s = a; return s
        let result = run(
            vec![array_input("a", &[4])],
            vec![assign(name_target("s"), load("a")), ret(load("s"))],
        )
        .expect("an alias of an input");
        match &entry_at(&result, "0|||0").producer {
            Producer::Alias(key) => assert_eq!(
                key.encode(),
                "|||0",
                "the input's own key"
            ),
            other => panic!("expected an alias, got {:?}", other),
        }
    }
}

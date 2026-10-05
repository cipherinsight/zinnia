//! The name environment and the `if` merge of `definition.md` §6.1.
//!
//! `Env` is the walker's half of the abstract state: one binding per name in
//! scope. `definition.md` §2 keeps it out of `Analysis`, because a transfer
//! rule sees descriptors, never names.
//!
//! A `Frame` is one call: the circuit, or an inlined chip body. A chip body
//! cannot see the caller's names, so lookup consults the innermost frame
//! only.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::phase0::desc::{Desc, ValueKey};
use crate::phase0::rules::merge_desc;
use crate::phase0::source::Producer;
use crate::phase0::term::BoolId;
use crate::phase0::Reject;

use super::loops::Element;
use super::Walker;

/// One value a `return` statement handed back, with the path condition it was
/// taken under (§6.1's `ite` merge of several returns).
#[derive(Debug, Clone)]
pub(crate) struct Returned {
    pub value: Element,
    pub path: BoolId,
}

/// One name binding: its descriptor, the key of the current version, and its
/// SSA version, which counts the assignments to the name and gives fresh
/// variables readable names.
///
/// The key is what makes a use of the name an [`Producer::Alias`] of the
/// value it was bound to (`implementation.md` §5a). The dtype names the
/// walker binds in every scope are not values and have none.
#[derive(Debug, Clone)]
pub(crate) struct Binding {
    pub desc: Desc,
    pub key: Option<ValueKey>,
    pub version: u32,
}

/// The names of one frame. A `BTreeMap` so that a merge visits names in a
/// fixed order and the variables it mints get the same names on every run.
pub(crate) type Scope = BTreeMap<String, Binding>;

/// One call frame: the circuit body, or an inlined chip body (§5.10).
#[derive(Debug, Default)]
pub(crate) struct Frame {
    pub names: Scope,
    /// The returned values, each with the path condition it was taken under
    /// (§6.1's `ite` merge of several returns).
    pub returns: Vec<Returned>,
    /// The disjunction of those path conditions: `φ` excludes it, so a
    /// statement after an unconditional `return` is dead.
    pub returned: Option<BoolId>,
    /// Names that a merge left undefined, with the reason, so that a later
    /// use reports why rather than "not found" (§6.1).
    pub undefined: HashMap<String, String>,
}

impl Frame {
    pub fn new() -> Self {
        Frame::default()
    }

    pub fn get(&self, name: &str) -> Option<&Desc> {
        self.names.get(name).map(|b| &b.desc)
    }

    pub fn binding(&self, name: &str) -> Option<&Binding> {
        self.names.get(name)
    }

    /// The key of the name's current version, when it has one.
    pub fn key_of(&self, name: &str) -> Option<ValueKey> {
        self.names.get(name).and_then(|b| b.key.clone())
    }

    pub fn version(&self, name: &str) -> u32 {
        self.names.get(name).map_or(0, |b| b.version)
    }

    /// Bind `name` to the value `desc` at `key`, bumping its version.
    ///
    /// `key` is `None` only for a value that is not one of $\Delta$'s: the
    /// dtype and module names bound in every scope (`GLOBAL_DTYPES`).
    pub fn set(&mut self, name: &str, desc: Desc, key: Option<ValueKey>) {
        let version = self.version(name).saturating_add(1);
        self.undefined.remove(name);
        self.names
            .insert(name.to_string(), Binding { desc, key, version });
    }

    /// Why a name is not bound: either a merge dropped it (§6.1) or it was
    /// never assigned.
    pub fn missing(&self, name: &str) -> Reject {
        match self.undefined.get(name) {
            Some(reason) => Reject::Unsupported {
                construct: format!("use of `{}`, which is {}", name, reason),
            },
            None => Reject::Unsupported {
                construct: format!("use of the unbound name `{}`", name),
            },
        }
    }
}

/// `Env' = Env1 ⊔_b Env2` (§6.1), taken name by name.
///
/// A name whose two bindings *differ* merges with [`merge_desc`] under one
/// further key (§5a rule 5), which becomes the merged value's entry in
/// $\Delta$. Two bindings are the same binding when they are the same value:
/// every version of a name has a key of its own (§5a rule 3), so equal keys
/// mean neither arm rebound the name, and the merge is `ite(b, d, d) = d` —
/// the identity fold the term constructors already apply to dimensions,
/// applied here to the binding itself. Such a name keeps its binding
/// unchanged, takes no key and gets no entry. The dtype and module names
/// bound in every scope have no key on either side and are identical by the
/// same test.
///
/// A name bound on one side only was born inside that branch — a name that
/// existed before the branch is in both scopes, because each branch starts
/// from a copy of the scope before it — and is undefined after the merge,
/// which §6.1 makes a static error at its next use.
pub(crate) fn merge_scopes(
    w: &mut Walker,
    b: BoolId,
    then_scope: Scope,
    else_scope: Scope,
) -> Result<(), Reject> {
    let mut merged = Scope::new();
    let mut undefined = HashMap::new();
    for name in union_of_names(&then_scope, &else_scope) {
        match (then_scope.get(&name), else_scope.get(&name)) {
            (Some(t), Some(e)) if t.key == e.key => {
                merged.insert(name, t.clone());
            }
            (Some(t), Some(e)) => {
                let version = t.version.max(e.version).saturating_add(1);
                let key = w.an.keys.next_expr();
                let desc = merge_binding(w, b, t, e, &key, &name)?;
                merged.insert(
                    name,
                    Binding {
                        desc,
                        key: Some(key),
                        version,
                    },
                );
            }
            (Some(_), None) | (None, Some(_)) => {
                undefined.insert(name, "assigned in only one branch of an `if`".to_string());
            }
            (None, None) => unreachable!("the name came from one of the two scopes"),
        }
    }
    let frame = w.frame_mut();
    frame.names = merged;
    frame.undefined.extend(undefined);
    Ok(())
}

/// Every name either scope binds, in sorted order — the order §5a rule 5
/// allocates the merge keys in.
fn union_of_names(then_scope: &Scope, else_scope: &Scope) -> BTreeSet<String> {
    then_scope
        .keys()
        .chain(else_scope.keys())
        .cloned()
        .collect()
}

/// One name's merge: the descriptor, its entry in $\Delta$, and the producer
/// that names the two arms it came from (`implementation.md` §5a).
fn merge_binding(
    w: &mut Walker,
    b: BoolId,
    then_binding: &Binding,
    else_binding: &Binding,
    key: &ValueKey,
    name: &str,
) -> Result<Desc, Reject> {
    let (t, e) = (then_binding.desc.clone(), else_binding.desc.clone());
    let (desc, frame) = w.in_value(key, |w| merge_one(w, b, &t, &e, key, name))?;
    let operands = [then_binding.key.clone(), else_binding.key.clone()]
        .into_iter()
        .flatten()
        .collect();
    let producer = Producer::of(&desc, frame.into_rules(), operands, Some(b));
    // The merge binds the name to a new version, so the entry carries it.
    w.record(key.clone(), Some(name), &desc, producer.clone());
    w.record_composite_elements(key, &desc, &producer);
    Ok(desc)
}

/// One name's merge, with the name in the rejection message.
fn merge_one(
    w: &mut Walker,
    b: BoolId,
    d1: &Desc,
    d2: &Desc,
    key: &ValueKey,
    name: &str,
) -> Result<Desc, Reject> {
    merge_desc(&mut w.ctx(), b, d1, d2, key, name).map_err(|reject| match reject {
        Reject::MergeMismatch { detail } => Reject::MergeMismatch {
            detail: format!("`{}`: {}", name, detail),
        },
        other => other,
    })
}

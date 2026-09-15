//! The transfer rules of `definition.md` §5 and the merge rules of §6.1.
//!
//! One function per rule, named after the rule (`e_slice`, `e_zip`,
//! `e_filter_1d`, `merge_array`, ...) and split into submodules by the
//! section it belongs to. Each rule is a pure function of the input
//! descriptors plus a mutable [`RuleCtx`]; nothing else in phase 0 knows
//! operator semantics (§5, opening paragraph).
//!
//! # The rule format
//!
//! §4 fixes the shape of a rule: premises on the input descriptors (a static
//! error if violated, hence a [`Reject`]), the variables it introduces with
//! their ranges, the constraints it adds, the output descriptor, and the
//! effect line `Φ += (φ ⇒ defcond)`. The effect line is not written out in
//! each rule: every rule builds its descriptor through [`RuleCtx::array`],
//! [`RuleCtx::scalar`] or [`RuleCtx::data_scalar`], and those are the single
//! place that records the check. A rule therefore cannot forget it.
//!
//! # What is here and what is the walker's
//!
//! Every rule that is a pure function from descriptors to a descriptor lives
//! here. The S-rules that need statement context live in `walk`:
//! `S-Assign`, `S-If`, `S-SetItem`, `S-Unpack` and `S-Assert` in
//! `walk::stmts`, the loops of §6.2 in `walk::loops`, `S-Call` in
//! `walk::calls`, `S-ListMethod` in `walk::values`. They use [`merge`] and
//! the rules below.
//!
//! Every rule name phase 0 records is a variant of [`Rule`]; a name that is
//! in no document is not one.

use crate::phase0::desc::KeyCursor;
use crate::phase0::desc::{ArrayDesc, Desc, Dtype, Origin, ScalarDesc, ScalarTerm, ValueKey};
use crate::phase0::phi::{ConstraintKind, ObligationKind, ObligationTerm, Phi, VarOrigin};
use crate::phase0::source::ValueStack;
use crate::phase0::term::{BoolId, Sort, TermArena, TermId};

pub mod composites;
pub mod constructors;
pub mod elementwise;
pub mod helpers;
pub mod indexing;
pub mod inputs;
pub mod masks;
pub mod merge;
pub mod reductions;
pub mod rule;
pub mod scalars;
pub mod shapes;

pub use composites::{
    e_composite, e_composite_binop, e_composite_index, e_composite_slice, e_external, e_property,
    CompositeBinop, Property,
};
pub use constructors::{
    e_arange, e_arange_range, e_repeat, e_value_sized, e_zeros, ValueSizedLength,
};
pub use elementwise::{e_map, e_zip};
pub use helpers::{bjoin, broadcast_dims, norm, product_term, range_len, size_term, wrap_index};
pub use indexing::{e_index, e_slice, e_take, slice_extent};
pub use inputs::{e_const, e_const_bool, e_const_opaque, e_input, ContractSymbols};
pub use masks::{e_filter_1d, e_filter_axis0, e_filter_same, e_nonzero, e_where3};
pub use merge::{merge_array, merge_composite, merge_desc, merge_scalar};
pub use reductions::{
    e_cumsum, e_matmul, e_reduce_axis, e_reduce_scalar, e_reverse, e_sort, e_unique, ReduceKind,
};
pub use rule::{Provenance, Rule};
pub use scalars::{e_arith, e_cmp, e_elem, e_len, e_shape, ArithOp, CmpOp};
pub use shapes::{
    e_concat, e_expand_dims, e_moveaxis, e_reshape, e_squeeze, e_stack, e_transpose, ReshapeExtent,
};

use super::Reject;

/// An inclusive range for a fresh variable (§3.4).
#[derive(Debug, Clone, Copy, Default)]
pub struct VarRange {
    pub lo: Option<TermId>,
    pub hi: Option<TermId>,
}

impl VarRange {
    /// The dtype range of §3.4. `Idx` is an unbounded Python integer and so
    /// has none: "intermediate values have no default upper bound".
    pub fn of_dtype(terms: &mut TermArena, dtype: Dtype) -> Self {
        let (lo, hi) = dtype.value_range();
        VarRange {
            lo: lo.map(|n| terms.constant(n)),
            hi: hi.map(|n| terms.constant(n)),
        }
    }

    /// A structural range such as the $[0, d)$ of an `argmax` (§3.4).
    ///
    /// The upper bound is `max(lo, exclusive_hi - 1)`, not `exclusive_hi - 1`:
    /// when `d` is 0 the operation did not run (its `defcond` demands
    /// `size(a) > 0`), but the range constraint is unconditional, and an
    /// empty range would leave no value for $\theta(u)$ to assign and so
    /// break G1. For every `d > 0` the two agree exactly.
    pub fn structural(terms: &mut TermArena, lo: TermId, exclusive_hi: TermId) -> Self {
        let one = terms.constant(1);
        let last = terms.sub(exclusive_hi, one);
        let hi = terms.max(lo, last);
        VarRange {
            lo: Some(lo),
            hi: Some(hi),
        }
    }
}

/// The mutable half of the abstract state that a rule may touch: the term
/// arena, $\Phi$, the current path condition $\varphi$, the key cursor, and
/// the value being built.
///
/// `Env` is deliberately absent — a rule sees descriptors, not names.
pub struct RuleCtx<'a> {
    pub terms: &'a mut TermArena,
    pub phi: &'a mut Phi,
    pub keys: &'a mut KeyCursor,
    /// $\varphi$, the branch conditions currently in force (§2).
    pub path: BoolId,
    /// The largest number of elements a rule will accept as a declared
    /// maximum, i.e. [`crate::phase0::Phase0Config::work_budget`]. Only
    /// `E-ValueSized` reads it (§5.9, through
    /// [`crate::phase0::interval::declared_maximum`]).
    pub materialisable: i128,
    /// The values under construction (`implementation.md` §5a). The innermost
    /// open frame is the value this rule is producing: its key is the subject
    /// of every check and obligation recorded here, and the rules applied to
    /// it are collected there. A `RuleCtx` cannot be built without one, so a
    /// check cannot be recorded without a subject.
    pub(crate) values: &'a mut ValueStack,
}

impl RuleCtx<'_> {
    /// The value this rule is producing: the subject of the conditions it
    /// records (`implementation.md` §5a).
    pub fn subject(&self) -> &ValueKey {
        self.values.subject()
    }

    /// Record that `rule` was applied to the value being produced.
    ///
    /// The descriptor constructors below call this with the rule they are
    /// given, so a rule that builds its output through them is tagged once
    /// and cannot forget. A rule that returns a descriptor it did not build —
    /// an identity such as `E-Copy`, a composite constructor — calls this
    /// itself.
    pub fn applied(&mut self, rule: Rule) {
        self.values.applied(rule);
    }
    /// $\psi$, the execution condition of a value created now: the path
    /// condition at creation (§2).
    pub fn psi(&self) -> BoolId {
        self.path
    }

    pub fn tt(&self) -> BoolId {
        self.terms.tt()
    }

    /// `Φ += (φ ⇒ defcond)`, the effect line of §4. Every descriptor
    /// constructor below funnels through here, and a rule that returns a
    /// descriptor it did not build calls [`RuleCtx::check`] instead, so the
    /// effect line is recorded in exactly one place.
    pub fn check(&mut self, rule: impl Into<Provenance>, defcond: BoolId) {
        let provenance = rule.into().to_string();
        self.phi.add_check(
            self.terms,
            self.path,
            defcond,
            self.values.subject(),
            &provenance,
        );
    }

    /// Record the CONST question of `implementation.md` §4: phase 1 has to
    /// know whether `term` is constant.
    ///
    /// The subjects are the lengths a representation decision turns on — the
    /// trip count of a bounded loop, the inferred extent of a reshape, the
    /// declared length of `E-ValueSized`, every count variable. A term that
    /// is already a literal raises no question and is dropped.
    pub fn record_const(&mut self, rule: impl Into<Provenance>, term: TermId) {
        if self.terms.constant_value(term).is_some() {
            return;
        }
        self.phi.record_obligation(
            ObligationKind::Const,
            ObligationTerm::Int(term),
            self.path,
            self.values.subject(),
            &rule.into().to_string(),
        );
    }

    /// Record the PROVE question of `implementation.md` §4: a premise-like
    /// condition the rule would have discharged by folding, but could not.
    ///
    /// `definition.md` §4 splits premises, which are decided here, from
    /// definedness conditions, which are decided at run time. A few
    /// conditions sit between the two — broadcast compatibility, a reshape's
    /// size equality, a matmul's inner dimensions — because they fold to
    /// `true` on constant shapes and are a question for phase 1 otherwise.
    /// That question is recorded here; the same condition is also a check, so
    /// that a run which violates it has no proof (§2).
    pub fn prove(&mut self, rule: impl Into<Provenance>, premise: BoolId) {
        if self.terms.is_true(premise) {
            return;
        }
        self.phi.record_obligation(
            ObligationKind::Prove,
            ObligationTerm::Bool(premise),
            self.path,
            self.values.subject(),
            &rule.into().to_string(),
        );
    }

    /// A fact a rule declares rather than checks: `E-ValueSized`'s
    /// `ℓ = max(0, t_n)` (§5.9), `E-Unique`'s `size(a) > 0 ⇒ c' ≥ 1` (§5.7).
    ///
    /// It is recorded as `φ ⇒ formula`, guarded exactly like a check (§2).
    /// §5.9 writes the declaration unguarded, but an unguarded one would make
    /// $\Phi$ unsatisfiable for a run that never reaches the operation — for
    /// `E-ValueSized`, whenever `t_n` exceeds the declared maximum inside an
    /// untaken branch — and G1 says every successful run satisfies $\Phi$.
    pub fn declare(&mut self, rule: impl Into<Provenance>, formula: BoolId) {
        if self.terms.is_true(formula) {
            return;
        }
        let guarded = self.terms.implies(self.path, formula);
        self.phi.add_fact(
            self.terms,
            guarded,
            ConstraintKind::Declaration,
            &rule.into().to_string(),
        );
    }

    /// Build `Array⟨τ, r, dims, defcond, ψ⟩`, record the rule's effect, and
    /// tag the rule on the value being produced.
    pub fn array(
        &mut self,
        rule: impl Into<Provenance>,
        dtype: Dtype,
        dims: Vec<TermId>,
        defcond: BoolId,
    ) -> Desc {
        let rule = rule.into();
        self.applied(rule.rule);
        self.check(rule, defcond);
        Desc::Array(ArrayDesc {
            dtype,
            dims,
            defcond,
            psi: self.path,
        })
    }

    /// Build `Scalar⟨τ, t, origin, defcond, ψ⟩`, record the rule's effect, and
    /// tag the rule on the value being produced.
    pub fn scalar(
        &mut self,
        rule: impl Into<Provenance>,
        dtype: Dtype,
        term: ScalarTerm,
        origin: Origin,
        defcond: BoolId,
    ) -> Desc {
        let rule = rule.into();
        self.applied(rule.rule);
        self.check(rule, defcond);
        Desc::Scalar(ScalarDesc {
            dtype,
            term,
            origin,
            defcond,
            psi: self.path,
        })
    }

    /// A scalar of origin `data` (§3.2): its term is a fresh variable whose
    /// range comes from the dtype or from the structure of the source.
    ///
    /// Dtypes that carry no term ([`Dtype::carries_term`]) introduce no
    /// variable; their descriptor is opaque.
    pub fn data_scalar(
        &mut self,
        rule: impl Into<Provenance>,
        dtype: Dtype,
        base: &str,
        value: &ValueKey,
        range: VarRange,
        defcond: BoolId,
    ) -> Desc {
        let term = self.fresh_data_term(dtype, base, value, range);
        self.scalar(rule, dtype, term, Origin::Data, defcond)
    }

    /// The unconstrained Boolean that stands for "the operation at `value`
    /// was defined" ([`VarOrigin::Definedness`]).
    ///
    /// `E-Take`'s `inrange` (§5.4) is the canonical case: index values are
    /// element contents, unknown to $\Phi$, so only an unconstrained Boolean
    /// can be introduced. This keeps soundness — $\Phi$ excludes no real
    /// state — and loses precision, so CANFAIL answers "may fail".
    pub fn definedness(&mut self, base: &str, value: &ValueKey) -> BoolId {
        let var = self.phi.fresh_var(
            self.terms,
            base,
            Sort::Bool,
            VarOrigin::Definedness {
                value: value.clone(),
            },
            None,
            None,
        );
        self.terms.bool_var(var)
    }

    /// The variable behind [`RuleCtx::data_scalar`], without building the
    /// descriptor. Used where a rule needs the variable itself.
    pub fn fresh_data_term(
        &mut self,
        dtype: Dtype,
        base: &str,
        value: &ValueKey,
        range: VarRange,
    ) -> ScalarTerm {
        if !dtype.carries_term() {
            return ScalarTerm::Opaque;
        }
        if dtype == Dtype::Bool {
            let v = self.phi.fresh_var(
                self.terms,
                base,
                Sort::Bool,
                VarOrigin::DataBool {
                    value: value.clone(),
                },
                None,
                None,
            );
            return ScalarTerm::Bool(self.terms.bool_var(v));
        }
        let sort = if dtype == Dtype::I64 {
            Sort::I64
        } else {
            Sort::Idx
        };
        let v = self.phi.fresh_var(
            self.terms,
            base,
            sort,
            VarOrigin::DataScalar {
                value: value.clone(),
            },
            range.lo,
            range.hi,
        );
        ScalarTerm::Int(self.terms.var(v))
    }
}

/// The descriptor argument of a rule that requires an array.
pub(crate) fn require_array(rule: Rule, d: &Desc) -> Result<&ArrayDesc, Reject> {
    d.as_array().ok_or_else(|| Reject::Premise {
        rule,
        detail: format!("applied to a {}", d.kind_str()),
    })
}

/// The descriptor argument of a rule that requires a scalar: the bounds of a
/// `range` (§6.2), and the walker's own scalar arguments.
pub(crate) fn require_scalar(rule: Rule, d: &Desc) -> Result<&ScalarDesc, Reject> {
    d.as_scalar().ok_or_else(|| Reject::Premise {
        rule,
        detail: format!("applied to a {}", d.kind_str()),
    })
}

/// The integer term of a scalar used as an index, a length or a bound.
pub(crate) fn require_int_term(rule: Rule, s: &ScalarDesc) -> Result<TermId, Reject> {
    s.term.as_int().ok_or_else(|| Reject::Premise {
        rule,
        detail: format!("applied to a {} scalar with no term", s.dtype.as_str()),
    })
}

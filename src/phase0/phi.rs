//! The relation system $\Phi = (V, C)$ of `definition.md` §1, §3.4 and §5.5.
//!
//! $\Phi$ holds every variable phase 0 introduces, with its range constraint
//! and its provenance, plus every constraint: range constraints, the checks
//! $\varphi \Rightarrow \mathrm{defcond}$ of §2, the declared facts read off
//! `scalar_requires`, and the product facts of §3.4. It also carries the
//! `Count` memo of §5.5 and the obligations of `implementation.md` §4, which
//! phase 0 records and does not answer.
//!
//! # Invariants
//!
//! - Every variable gets its range constraint at the moment it is introduced
//!   (§3.4). [`Phi::fresh_var`] is the only way to introduce one, and it is
//!   the only caller of `TermArena::declare_var`, so `phi.vars.len()` always
//!   equals `terms.var_count()` and a `VarId` indexes both.
//! - $\Phi$ only grows (§2, the table of the abstract state).
//! - `Count` is keyed by the value key of the mask and returns the same
//!   variable for the same mask (§5.5). That is how $\Phi$ remembers that one
//!   mask was used twice.

use std::collections::{HashMap, HashSet};

use crate::phase0::desc::ValueKey;
use crate::phase0::term::{BoolId, Sort, TermArena, TermId, VarId};

/// Why a variable exists (`implementation.md` §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarOrigin {
    /// $n_x$: the logical length of a variable-length input axis (§5.1).
    InputLength { input: String, axis: usize },
    /// $c$: the count of one mask used as a selector (§5.5).
    Count { mask: ValueKey },
    /// A data integer scalar (§3.2).
    DataScalar { value: ValueKey },
    /// A data Boolean: a branch condition or a mask atom (§3.4).
    DataBool { value: ValueKey },
    /// A Boolean witness that the operation at `value` was *defined*: the
    /// `inrange` $\beta_I$ of `E-Take` (§5.4), the domain condition of an
    /// `E-Map` whose operand $\Phi$ cannot see (§5.3), `list.remove`'s "the
    /// element is present" (§5.11). $\Phi$ does not model the values these
    /// rest on, so the variable is unconstrained; under $\theta(u)$ it is
    /// true exactly when the run produced a value at `value`.
    Definedness { value: ValueKey },
    /// A `SymbolicNDArray` shape symbol, shared by name across inputs (§5.1).
    ShapeSymbol { symbol: String },
    /// $c'$: the count of `unique` (§5.7). Not the count of any mask, so it
    /// is an independent variable.
    UniqueCount { value: ValueKey },
    /// $\ell$: the declared length of a value-sized construct (§5.9).
    ValueSized { value: ValueKey },
    /// $\sigma$: stands for a size whose factors are two or more
    /// non-constant extents (§3.4, end).
    SizeProduct { value: ValueKey },
}

impl VarOrigin {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            VarOrigin::InputLength { input, axis } => {
                serde_json::json!({"kind": "input_length", "input": input, "axis": axis})
            }
            VarOrigin::Count { mask } => {
                serde_json::json!({"kind": "count", "mask": mask.to_json()})
            }
            VarOrigin::DataScalar { value } => {
                serde_json::json!({"kind": "data_scalar", "value": value.to_json()})
            }
            VarOrigin::DataBool { value } => {
                serde_json::json!({"kind": "data_bool", "value": value.to_json()})
            }
            VarOrigin::Definedness { value } => {
                serde_json::json!({"kind": "definedness", "value": value.to_json()})
            }
            VarOrigin::ShapeSymbol { symbol } => {
                serde_json::json!({"kind": "shape_symbol", "symbol": symbol})
            }
            VarOrigin::UniqueCount { value } => {
                serde_json::json!({"kind": "unique_count", "value": value.to_json()})
            }
            VarOrigin::ValueSized { value } => {
                serde_json::json!({"kind": "value_sized", "value": value.to_json()})
            }
            VarOrigin::SizeProduct { value } => {
                serde_json::json!({"kind": "size_product", "value": value.to_json()})
            }
        }
    }
}

/// A variable of $\Phi$. Its name and sort live in the term arena, which is
/// what every printer consults; this is the rest of the record.
#[derive(Debug, Clone)]
pub struct VarInfo {
    /// Inclusive lower bound, when the variable has one (§3.4).
    pub lo: Option<TermId>,
    /// Inclusive upper bound, when the variable has one.
    pub hi: Option<TermId>,
    pub origin: VarOrigin,
}

/// Why a constraint is in $\Phi$.
#[derive(Debug, Clone)]
pub enum ConstraintKind {
    /// The range constraint a variable gets at introduction (§3.4).
    VarRange(VarId),
    /// A check $\varphi \Rightarrow \mathrm{defcond}$ (§2), with the value
    /// whose operation it belongs to (`implementation.md` §5a): phase 1's
    /// ENFORCE_CHECK and the acyclicity condition of `../README.md` §6.1 both
    /// need to know which operation instance a check is about.
    Check {
        path: BoolId,
        defcond: BoolId,
        subject: ValueKey,
    },
    /// A `scalar_requires` fact asserted at circuit start (`coverage.md` §4).
    Precondition,
    /// A fact that is part of the source semantics rather than something the
    /// run must satisfy: `E-ValueSized`'s `ℓ = max(0, t_n)` (§5.9) and
    /// `E-Unique`'s `size(a) > 0 ⇒ c' ≥ 1` (§5.7).
    Declaration,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub formula: BoolId,
    pub kind: ConstraintKind,
    /// Human-readable source of the constraint: the rule name, or the input
    /// name for a range.
    pub provenance: String,
}

/// $\sigma = \prod_k \mathrm{dims}[k]$ for a size that is not a term (§3.4,
/// end). A product of non-constant terms is outside the term language, so
/// this fact is recorded structurally instead of as a constraint; phase 1
/// bounds each factor separately and multiplies.
#[derive(Debug, Clone)]
pub struct ProductFact {
    pub var: VarId,
    pub factors: Vec<TermId>,
    pub provenance: String,
}

/// The kinds of question phase 1 answers (`definition.md` §1, "decided here
/// versus deferred"). MAX and MIN are asked by phase 1 itself and are not
/// recorded here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationKind {
    /// "Is this term constant?"
    Const,
    /// "Does $\Phi$ entail this premise?"
    Prove,
    /// "Can this condition be false?"
    CanFail,
}

impl ObligationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ObligationKind::Const => "CONST",
            ObligationKind::Prove => "PROVE",
            ObligationKind::CanFail => "CANFAIL",
        }
    }
}

/// The subject of an obligation: an integer term for CONST, a Boolean
/// formula for PROVE and CANFAIL.
#[derive(Debug, Clone, Copy)]
pub enum ObligationTerm {
    Int(TermId),
    Bool(BoolId),
}

/// A question phase 0 records and phase 1 answers (`implementation.md` §4).
#[derive(Debug, Clone)]
pub struct Obligation {
    pub kind: ObligationKind,
    pub term: ObligationTerm,
    pub path: BoolId,
    /// The value whose operation asks the question (`implementation.md` §5a).
    pub subject: ValueKey,
    pub provenance: String,
}

/// The relation system.
#[derive(Debug, Default)]
pub struct Phi {
    vars: Vec<VarInfo>,
    names: HashSet<String>,
    /// The next numeric suffix to try for each base name, so that minting the
    /// `k`-th variable of a base costs `O(1)` and not `O(k)`. An unrolled
    /// loop mints thousands of variables from the same base, which made the
    /// naive scan quadratic.
    next_suffix: HashMap<String, u32>,
    constraints: Vec<Constraint>,
    counts: HashMap<ValueKey, VarId>,
    products: Vec<ProductFact>,
    obligations: Vec<Obligation>,
}

impl Phi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vars(&self) -> &[VarInfo] {
        &self.vars
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub fn products(&self) -> &[ProductFact] {
        &self.products
    }

    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }

    pub fn var_info(&self, v: VarId) -> &VarInfo {
        &self.vars[v.0 as usize]
    }

    /// Introduce a variable with its range constraint (§3.4).
    ///
    /// `base` is a suggestion; a numeric suffix is appended if it is already
    /// taken, so names stay unique and stable across a run. `lo` and `hi`
    /// are inclusive bounds; a Boolean variable has neither (§3.4).
    pub fn fresh_var(
        &mut self,
        terms: &mut TermArena,
        base: &str,
        sort: Sort,
        origin: VarOrigin,
        lo: Option<TermId>,
        hi: Option<TermId>,
    ) -> VarId {
        let name = self.unique_name(base);
        let id = terms.declare_var(name.clone(), sort);
        debug_assert_eq!(id.0 as usize, self.vars.len(), "Phi and the arena agree");
        self.vars.push(VarInfo { lo, hi, origin });
        self.add_range_constraint(terms, id, sort, lo, hi, &name);
        id
    }

    fn unique_name(&mut self, base: &str) -> String {
        if self.names.insert(base.to_string()) {
            return base.to_string();
        }
        let start = self.next_suffix.get(base).copied().unwrap_or(1);
        for n in start.. {
            let candidate = format!("{}_{}", base, n);
            if self.names.insert(candidate.clone()) {
                self.next_suffix.insert(base.to_string(), n + 1);
                return candidate;
            }
        }
        // `start..` is unbounded in practice; the loop always returns.
        unreachable!("a unique variable name always exists")
    }

    fn add_range_constraint(
        &mut self,
        terms: &mut TermArena,
        id: VarId,
        sort: Sort,
        lo: Option<TermId>,
        hi: Option<TermId>,
        name: &str,
    ) {
        if sort.is_bool() || (lo.is_none() && hi.is_none()) {
            return;
        }
        let v = terms.var(id);
        let mut bounds = Vec::with_capacity(2);
        if let Some(l) = lo {
            bounds.push(terms.le(l, v));
        }
        if let Some(h) = hi {
            bounds.push(terms.le(v, h));
        }
        let formula = terms.all(&bounds);
        if terms.is_true(formula) {
            return;
        }
        self.constraints.push(Constraint {
            formula,
            kind: ConstraintKind::VarRange(id),
            provenance: format!("range of {}", name),
        });
    }

    /// `Count(m)` of §5.5: the same mask value key returns the same
    /// variable, with range `0 <= c <= upper`.
    pub fn count_var(
        &mut self,
        terms: &mut TermArena,
        mask: &ValueKey,
        upper: TermId,
        base: &str,
    ) -> VarId {
        if let Some(&v) = self.counts.get(mask) {
            return v;
        }
        let zero = terms.constant(0);
        let v = self.fresh_var(
            terms,
            base,
            Sort::Idx,
            VarOrigin::Count { mask: mask.clone() },
            Some(zero),
            Some(upper),
        );
        self.counts.insert(mask.clone(), v);
        v
    }

    /// Record `path ⇒ defcond` (§2, the "effect" line of §4's rule format)
    /// and the CANFAIL obligation that goes with it.
    ///
    /// A trivially true `defcond` adds nothing: the guard of §2 already makes
    /// a check inside an untaken branch vacuous, and a check that folded to
    /// `true` constrains nothing.
    pub fn add_check(
        &mut self,
        terms: &mut TermArena,
        path: BoolId,
        defcond: BoolId,
        subject: &ValueKey,
        provenance: &str,
    ) {
        if terms.is_true(defcond) {
            return;
        }
        let formula = terms.implies(path, defcond);
        if terms.is_true(formula) {
            return;
        }
        self.constraints.push(Constraint {
            formula,
            kind: ConstraintKind::Check {
                path,
                defcond,
                subject: subject.clone(),
            },
            provenance: provenance.to_string(),
        });
        self.obligations.push(Obligation {
            kind: ObligationKind::CanFail,
            term: ObligationTerm::Bool(defcond),
            path,
            subject: subject.clone(),
            provenance: provenance.to_string(),
        });
    }

    /// Assert a fact that holds by declaration rather than by a check: a
    /// `scalar_requires` precondition, or a rule's own declaration. A fact
    /// that folded to `true` constrains nothing and is dropped.
    pub fn add_fact(
        &mut self,
        terms: &mut TermArena,
        formula: BoolId,
        kind: ConstraintKind,
        provenance: &str,
    ) {
        if terms.is_true(formula) {
            return;
        }
        self.constraints.push(Constraint {
            formula,
            kind,
            provenance: provenance.to_string(),
        });
    }

    /// Record $\sigma = \prod_k \mathrm{dims}[k]$ (§3.4, end).
    pub fn add_product_fact(&mut self, var: VarId, factors: Vec<TermId>, provenance: &str) {
        self.products.push(ProductFact {
            var,
            factors,
            provenance: provenance.to_string(),
        });
    }

    /// Record a question for phase 1. CANFAIL obligations come from
    /// [`Phi::add_check`]; CONST and PROVE come from
    /// [`crate::phase0::rules::RuleCtx::record_const`] and
    /// [`crate::phase0::rules::RuleCtx::prove`], where a rule needs a fact it
    /// cannot decide (`implementation.md` §4).
    pub fn record_obligation(
        &mut self,
        kind: ObligationKind,
        term: ObligationTerm,
        path: BoolId,
        subject: &ValueKey,
        provenance: &str,
    ) {
        self.obligations.push(Obligation {
            kind,
            term,
            path,
            subject: subject.clone(),
            provenance: provenance.to_string(),
        });
    }
}

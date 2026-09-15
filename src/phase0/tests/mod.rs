//! Unit tests for phase 0.
//!
//! Four groups: the term simplifier and the arena's printers; `norm` and the
//! slice length against CPython's own slicing (`definition.md` §7 demands an
//! exact match, including negative indices and negative steps); broadcasting;
//! and the two worked examples of §8, reproduced by calling the rules by hand
//! in sequence, with no AST walking.
//!
//! The first three groups and the fixture they share are here; the two
//! worked examples are in [`examples`].
//!
//! `../testdata/slice_lengths.txt` is generated ground truth: every row is
//! `lo hi step d start stop len` as CPython's `slice.indices` and
//! `len(range(...))` report it, with `*` for an omitted bound.

mod defects;
mod examples;
mod fixtures;
mod invariants;
mod walk;

use super::desc::{ArrayDesc, Desc, Dtype, Origin, PathStep, ScalarDesc, ScalarTerm, ValueKey};
use super::source::Producer;
use super::phi::ConstraintKind;
use super::rules::composites::python_slice_indices;
use super::rules::helpers::{constant_norm, constant_range_len};
use super::rules::{broadcast_dims, e_filter_1d, e_slice, e_zip, norm, range_len, RuleCtx};
use super::term::{floor_div, floor_mod, Assignment, BoolId, Sort, TermArena, TermId};
use super::{Analysis, Phase0Config};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// A phase-0 state to call rules against, standing in for the walker.
///
/// It is `pub(crate)` so that the `#[cfg(test)] mod tests` of each
/// `rules/*.rs` file can call its rule directly: a transfer rule is a pure
/// function of descriptors plus a [`RuleCtx`] (§5), and this is the smallest
/// thing that provides one.
pub(crate) struct Fixture {
    pub analysis: Analysis,
    /// $\varphi$. A test that wants a guarded rule sets it.
    pub path: BoolId,
    next_expr: u32,
}

impl Fixture {
    pub fn new() -> Self {
        let analysis = Analysis::new(Phase0Config::default());
        let path = analysis.terms.tt();
        let mut fixture = Fixture {
            analysis,
            path,
            next_expr: 0,
        };
        fixture.open_root_value();
        fixture
    }

    pub fn ctx(&mut self) -> RuleCtx<'_> {
        let path = self.path;
        self.analysis.ctx(path)
    }

    /// The record the fixture built, with its root value frame closed: the
    /// walk balances its frames and [`Analysis::finish`] asserts it.
    pub fn finish(mut self) -> crate::phase0::Phase0Result {
        self.analysis.close_value();
        self.analysis.finish()
    }

    /// The fixture stands in for the walker, which opens a value frame around
    /// every value it builds (`implementation.md` §5a); one frame stays open
    /// for the whole fixture, so a rule called by hand has a subject.
    fn open_root_value(&mut self) {
        let key = ValueKey {
            stmt_path: vec![PathStep::Stmt(0)],
            iters: vec![],
            calls: vec![],
            expr_index: 0,
        };
        self.analysis.open_value(&key);
    }

    /// A distinct value key per call, standing in for the walker's cursor.
    pub fn key(&mut self) -> ValueKey {
        self.next_expr += 1;
        ValueKey {
            stmt_path: vec![PathStep::Stmt(0)],
            iters: vec![],
            calls: vec![],
            expr_index: self.next_expr,
        }
    }

    /// An input array with constant extents (`E-Input-Fixed`).
    pub fn input(&mut self, dtype: Dtype, shape: &[i64]) -> ArrayDesc {
        let dims = shape
            .iter()
            .map(|&n| self.analysis.terms.constant(n as i128))
            .collect();
        ArrayDesc {
            dtype,
            dims,
            defcond: self.path,
            psi: self.path,
        }
    }

    /// An array one of whose extents is a variable, for the rules whose
    /// answer is a term rather than a number.
    pub fn symbolic_input(&mut self, dtype: Dtype, name: &str, bound: i64) -> ArrayDesc {
        let zero = self.analysis.terms.constant(0);
        let hi = self.analysis.terms.constant(bound as i128);
        let var = self.analysis.phi.fresh_var(
            &mut self.analysis.terms,
            name,
            Sort::Idx,
            crate::phase0::phi::VarOrigin::InputLength {
                input: name.to_string(),
                axis: 0,
            },
            Some(zero),
            Some(hi),
        );
        let dim = self.analysis.terms.var(var);
        ArrayDesc {
            dtype,
            dims: vec![dim],
            defcond: self.path,
            psi: self.path,
        }
    }

    /// A shape-origin integer literal (`E-Const`).
    pub fn literal(&mut self, n: i64) -> ScalarDesc {
        let t = self.analysis.terms.constant(n as i128);
        ScalarDesc {
            dtype: Dtype::Idx,
            term: ScalarTerm::Int(t),
            origin: Origin::Shape,
            defcond: self.path,
            psi: self.path,
        }
    }

    /// A scalar of origin `data`: a value read out of an array, whose term
    /// is a fresh variable (§3.2).
    pub fn data_scalar(&mut self, dtype: Dtype) -> ScalarDesc {
        let key = self.key();
        let range = crate::phase0::rules::VarRange::of_dtype(&mut self.analysis.terms, dtype);
        let tt = self.analysis.terms.tt();
        let desc = self.ctx().data_scalar(
            crate::phase0::rules::Rule::EElem,
            dtype,
            "v",
            &key,
            range,
            tt,
        );
        self.scalar_of(&desc)
    }

    pub fn array_of(&self, d: &Desc) -> ArrayDesc {
        d.as_array().expect("the rule returned an array").clone()
    }

    pub fn scalar_of(&self, d: &Desc) -> ScalarDesc {
        d.as_scalar().expect("the rule returned a scalar").clone()
    }

    /// The constant extents of a rule's output, or `None` where an extent is
    /// not a literal.
    pub fn dims_of(&self, d: &Desc) -> Vec<Option<i128>> {
        self.array_of(d)
            .dims
            .iter()
            .map(|&t| self.analysis.terms.constant_value(t))
            .collect()
    }

    pub fn pretty(&self, t: TermId) -> String {
        self.analysis.terms.pretty(t)
    }

    /// The checks `Φ` holds, i.e. the `φ ⇒ defcond` constraints.
    pub fn checks(&self) -> Vec<String> {
        self.analysis
            .phi
            .constraints()
            .iter()
            .filter(|c| matches!(c.kind, ConstraintKind::Check { .. }))
            .map(|c| {
                format!(
                    "{}: {}",
                    c.provenance,
                    self.analysis.terms.pretty_bool(c.formula)
                )
            })
            .collect()
    }

    /// The rule names the checks of `Φ` carry, which is what a per-rule test
    /// asserts when the rule's content is its `defcond`.
    pub fn check_rules(&self) -> Vec<String> {
        self.analysis
            .phi
            .constraints()
            .iter()
            .filter(|c| matches!(c.kind, ConstraintKind::Check { .. }))
            .map(|c| c.provenance.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The term simplifier (§3.1)
// ---------------------------------------------------------------------------

#[test]
fn hash_consing_makes_equal_terms_one_object() {
    let mut a = TermArena::new();
    let v = a.declare_var("n".into(), Sort::Idx);
    let x = a.var(v);
    let three = a.constant(3);
    assert_eq!(a.add(x, three), a.add(x, three));
    // Argument order is not canonicalised, so these stay distinct terms.
    assert_ne!(a.add(x, three), a.add(three, x));
}

#[test]
fn constant_folding() {
    let mut a = TermArena::new();
    let (two, three) = (a.constant(2), a.constant(3));
    let folded = |a: &TermArena, t: TermId| a.constant_value(t);
    let sum = a.add(two, three);
    assert_eq!(folded(&a, sum), Some(5));
    let difference = a.sub(two, three);
    assert_eq!(folded(&a, difference), Some(-1));
    let product = a.mul_const(4, three);
    assert_eq!(folded(&a, product), Some(12));
    let smaller = a.min(two, three);
    assert_eq!(folded(&a, smaller), Some(2));
    let larger = a.max(two, three);
    assert_eq!(folded(&a, larger), Some(3));
    let ordered = a.le(two, three);
    assert!(a.is_true(ordered));
    let reversed = a.lt(three, two);
    assert!(a.is_false(reversed));
}

#[test]
fn identity_folds_on_equal_children() {
    let mut a = TermArena::new();
    let v = a.declare_var("n".into(), Sort::Idx);
    let x = a.var(v);
    let zero = a.constant(0);
    assert_eq!(a.add(x, zero), x);
    assert_eq!(a.sub(x, zero), x);
    let self_difference = a.sub(x, x);
    assert_eq!(a.constant_value(self_difference), Some(0));
    assert_eq!(a.mul_const(1, x), x);
    let scaled_by_zero = a.mul_const(0, x);
    assert_eq!(a.constant_value(scaled_by_zero), Some(0));
    assert_eq!(a.min(x, x), x);
    let equal = a.eq(x, x);
    assert!(a.is_true(equal));
    let ordered = a.le(x, x);
    assert!(a.is_true(ordered));
    let strict = a.lt(x, x);
    assert!(a.is_false(strict));
}

#[test]
fn ite_folds() {
    let mut a = TermArena::new();
    let v = a.declare_var("n".into(), Sort::Idx);
    let x = a.var(v);
    let (tt, ff) = (a.tt(), a.ff());
    let one = a.constant(1);
    assert_eq!(a.ite(tt, x, one), x);
    assert_eq!(a.ite(ff, x, one), one);
    let b = a.declare_var("b".into(), Sort::Bool);
    let beta = a.bool_var(b);
    assert_eq!(a.ite(beta, x, x), x);
}

#[test]
fn min_max_absorb_constants() {
    let mut a = TermArena::new();
    let v = a.declare_var("c".into(), Sort::Idx);
    let c = a.var(v);
    let zero = a.constant(0);
    let three = a.constant(3);
    // Lattice absorption: this is `norm(0, c)` of §8.1, which must be 0.
    let inner = a.min(c, zero);
    let absorbed = a.max(zero, inner);
    assert_eq!(a.constant_value(absorbed), Some(0));
    // The mirror image under min.
    let inner = a.max(c, zero);
    let absorbed = a.min(zero, inner);
    assert_eq!(a.constant_value(absorbed), Some(0));
    // A constant merged into a nested same-operator node: the doubled
    // `max(0, ·)` that E-Slice builds collapses.
    let bounded = a.min(c, three);
    let clamped = a.max(zero, bounded);
    assert_eq!(a.max(zero, clamped), clamped);
}

#[test]
fn boolean_folds() {
    let mut a = TermArena::new();
    let v = a.declare_var("b".into(), Sort::Bool);
    let b = a.bool_var(v);
    let (tt, ff) = (a.tt(), a.ff());
    assert_eq!(a.and(tt, b), b);
    let conjunction = a.and(ff, b);
    assert!(a.is_false(conjunction));
    let disjunction = a.or(tt, b);
    assert!(a.is_true(disjunction));
    assert_eq!(a.or(ff, b), b);
    assert_eq!(a.and(b, b), b);
    let negated = a.not(b);
    assert_eq!(a.not(negated), b);
    let vacuous = a.implies(ff, b);
    assert!(a.is_true(vacuous));
    // A Boolean `ite` is `(c ∧ a) ∨ (¬c ∧ b)` and folds on a constant guard.
    assert_eq!(a.bool_ite(tt, b, ff), b);
}

#[test]
fn floor_division_matches_python() {
    // Ground truth from CPython: `//` and `%` floor, they do not truncate.
    let cases = [
        (7i128, 2i128, 3i128, 1i128),
        (-7, 2, -4, 1),
        (7, -2, -4, -1),
        (-7, -2, 3, -1),
        (6, 3, 2, 0),
        (-6, 3, -2, 0),
    ];
    for (a, b, q, r) in cases {
        assert_eq!(floor_div(a, b), q, "{} // {}", a, b);
        assert_eq!(floor_mod(a, b), r, "{} % {}", a, b);
    }
    let mut arena = TermArena::new();
    for (a, b, q, r) in cases {
        let t = arena.constant(a);
        let quotient = arena.div_const(t, b).unwrap();
        assert_eq!(arena.constant_value(quotient), Some(q));
        let remainder = arena.mod_const(t, b).unwrap();
        assert_eq!(arena.constant_value(remainder), Some(r));
    }
    let t = arena.constant(1);
    assert!(arena.div_const(t, 0).is_none());
    assert!(arena.mod_const(t, 0).is_none());
}

#[test]
fn wrap64_folds_and_evaluates() {
    let mut a = TermArena::new();
    let max = a.constant(super::term::I64_MAX);
    let one = a.constant(1);
    let overflow = a.add(max, one);
    let folded = a.wrap64(overflow);
    assert_eq!(a.constant_value(folded), Some(super::term::I64_MIN));
    let v = a.declare_var("x".into(), Sort::I64);
    let x = a.var(v);
    let incremented = a.add(x, one);
    let wrapped = a.wrap64(incremented);
    let mut theta = Assignment::default();
    theta.set_int(v, super::term::I64_MAX);
    assert_eq!(a.eval(wrapped, &theta), Ok(super::term::I64_MIN));
}

#[test]
fn evaluation_under_an_assignment() {
    let mut a = TermArena::new();
    let n = a.declare_var("n".into(), Sort::Idx);
    let b = a.declare_var("b".into(), Sort::Bool);
    let (nt, bt) = (a.var(n), a.bool_var(b));
    let five = a.constant(5);
    let capped = a.min(nt, five);
    let doubled = a.mul_const(2, nt);
    let term = a.ite(bt, capped, doubled);
    let mut theta = Assignment::default();
    theta.set_int(n, 7).set_bool(b, true);
    assert_eq!(a.eval(term, &theta), Ok(5));
    theta.set_bool(b, false);
    assert_eq!(a.eval(term, &theta), Ok(14));
    let unbound = Assignment::default();
    assert!(a.eval(term, &unbound).is_err());
}

#[test]
fn pretty_printing_is_stable() {
    let mut a = TermArena::new();
    let c = a.declare_var("c".into(), Sort::Idx);
    let ct = a.var(c);
    let three = a.constant(3);
    let zero = a.constant(0);
    let bounded = a.min(ct, three);
    let len = a.max(zero, bounded);
    assert_eq!(a.pretty(len), "max(0, min(c, 3))");
    let cond = a.le(zero, ct);
    assert_eq!(a.pretty_bool(cond), "(0 <= c)");
}

#[test]
fn smtlib_expresses_floor_division() {
    let mut a = TermArena::new();
    let v = a.declare_var("n".into(), Sort::Idx);
    let t = a.var(v);
    // A positive constant divisor: SMT-LIB's Euclidean `div` already floors.
    let pos = a.div_const(t, 2).unwrap();
    // A negative one: floor(a / k) = floor(-a / |k|) = (div (- a) |k|).
    let neg = a.div_const(t, -2).unwrap();
    let neg_mod = a.mod_const(t, -2).unwrap();
    let text = a.to_smtlib(&[]);
    assert!(text.contains("(declare-const n Int)"));
    assert!(text.contains(&format!(
        "(define-fun t{} () Int (div t{} 2))",
        pos.index(),
        t.index()
    )));
    assert!(text.contains(&format!(
        "(define-fun t{} () Int (div (- t{}) 2))",
        neg.index(),
        t.index()
    )));
    assert!(text.contains(&format!(
        "(define-fun t{} () Int (+ t{} (* 2 (div (- t{}) 2))))",
        neg_mod.index(),
        t.index(),
        t.index()
    )));
    // min and max become one `ite`, which SMT-LIB does have.
    let m = a.min(t, pos);
    let text = a.to_smtlib(&[]);
    assert!(text.contains(&format!(
        "(define-fun t{} () Int (ite (<= t{} t{}) t{} t{}))",
        m.index(),
        t.index(),
        pos.index(),
        t.index(),
        pos.index()
    )));
}

#[test]
fn term_json_lists_every_node_with_smaller_arguments() {
    let mut a = TermArena::new();
    let v = a.declare_var("n".into(), Sort::Idx);
    let t = a.var(v);
    let three = a.constant(3);
    let _ = a.max(t, three);
    let json = a.to_json();
    let nodes = json["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), a.node_count());
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(node["id"].as_u64().unwrap() as usize, i);
        if let Some(args) = node["args"].as_array() {
            for arg in args {
                assert!(
                    (arg.as_u64().unwrap() as usize) < i,
                    "argument precedes its node"
                );
            }
        }
    }
    assert_eq!(json["vars"][0]["name"], "n");
}

// ---------------------------------------------------------------------------
// norm and slice lengths against CPython (§5.4, §7)
// ---------------------------------------------------------------------------

/// One row of `testdata/slice_lengths.txt`.
struct SliceCase {
    lo: Option<i64>,
    hi: Option<i64>,
    step: i64,
    d: i64,
    start: i64,
    stop: i64,
    len: i64,
}

fn slice_cases() -> Vec<SliceCase> {
    include_str!("../testdata/slice_lengths.txt")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let bound = |s: &str| {
                if s == "*" {
                    None
                } else {
                    Some(s.parse().unwrap())
                }
            };
            SliceCase {
                lo: bound(f[0]),
                hi: bound(f[1]),
                step: f[2].parse().unwrap(),
                d: f[3].parse().unwrap(),
                start: f[4].parse().unwrap(),
                stop: f[5].parse().unwrap(),
                len: f[6].parse().unwrap(),
            }
        })
        .collect()
}

/// Build `norm` for both endpoints of one case, given the extent term `d`.
fn slice_terms(a: &mut TermArena, case: &SliceCase, d: TermId) -> (TermId, TermId, TermId) {
    let negative = case.step < 0;
    let one = a.constant(1);
    let default_lo = if negative {
        a.sub(d, one)
    } else {
        a.constant(0)
    };
    let default_hi = if negative {
        let neg_d = a.mul_const(-1, d);
        a.sub(neg_d, one)
    } else {
        d
    };
    let raw = |a: &mut TermArena, b: Option<i64>, default: TermId| match b {
        Some(n) => a.constant(n as i128),
        None => default,
    };
    let lo = raw(a, case.lo, default_lo);
    let hi = raw(a, case.hi, default_hi);
    let l = norm(a, lo, d, negative);
    let u = norm(a, hi, d, negative);
    let len = range_len(a, l, u, case.step).unwrap();
    (l, u, len)
}

#[test]
fn norm_and_slice_length_match_python_on_constants() {
    for case in slice_cases() {
        let mut a = TermArena::new();
        let d = a.constant(case.d as i128);
        let (l, u, len) = slice_terms(&mut a, &case, d);
        let what = format!(
            "[{:?}:{:?}:{}] on d={}",
            case.lo, case.hi, case.step, case.d
        );
        assert_eq!(
            a.constant_value(l),
            Some(case.start as i128),
            "start of {}",
            what
        );
        assert_eq!(
            a.constant_value(u),
            Some(case.stop as i128),
            "stop of {}",
            what
        );
        assert_eq!(
            a.constant_value(len),
            Some(case.len as i128),
            "len of {}",
            what
        );
    }
}

#[test]
fn norm_and_slice_length_match_python_on_a_symbolic_extent() {
    for case in slice_cases() {
        let mut a = TermArena::new();
        let v = a.declare_var("d".into(), Sort::Idx);
        let d = a.var(v);
        let (l, u, len) = slice_terms(&mut a, &case, d);
        let mut theta = Assignment::default();
        theta.set_int(v, case.d as i128);
        let what = format!(
            "[{:?}:{:?}:{}] on d={}",
            case.lo, case.hi, case.step, case.d
        );
        assert_eq!(
            a.eval(l, &theta),
            Ok(case.start as i128),
            "start of {}",
            what
        );
        assert_eq!(a.eval(u, &theta), Ok(case.stop as i128), "stop of {}", what);
        assert_eq!(a.eval(len, &theta), Ok(case.len as i128), "len of {}", what);
    }
}

/// The same table drives the *constant* forms of the two formulas, which
/// the rules that work on literals use: `constant_norm` for a composite
/// slice (§5.11) and `constant_range_len` for a `range` whose bounds are all
/// constants (§6.2). The three transcriptions cannot drift apart while this
/// passes.
#[test]
fn the_constant_forms_of_norm_and_the_slice_length_match_python_too() {
    for case in slice_cases() {
        let negative = case.step < 0;
        let d = case.d as i128;
        let default_lo = if negative { d - 1 } else { 0 };
        let default_hi = if negative { -d - 1 } else { d };
        let start = constant_norm(case.lo.map_or(default_lo, |n| n as i128), d, negative);
        let stop = constant_norm(case.hi.map_or(default_hi, |n| n as i128), d, negative);
        let what = format!(
            "[{:?}:{:?}:{}] on d={}",
            case.lo, case.hi, case.step, case.d
        );
        assert_eq!(start, case.start as i128, "start of {}", what);
        assert_eq!(stop, case.stop as i128, "stop of {}", what);
        assert_eq!(
            constant_range_len(start, stop, case.step as i128),
            Some(case.len as i128),
            "len of {}",
            what
        );
        // The composite slice enumerates the same indices, so its count is
        // the same length (§5.11's `E-Composite-Index`).
        assert_eq!(
            python_slice_indices(case.lo, case.hi, case.step, case.d).len() as i128,
            case.len as i128,
            "composite slice of {}",
            what
        );
    }
}

#[test]
fn e_slice_uses_the_same_formula() {
    let mut f = Fixture::new();
    let a = f.input(Dtype::I64, &[7]);
    let lo = f.literal(1);
    let hi = f.literal(6);
    let sliced = e_slice(&mut f.ctx(), &a, 0, Some(&lo), Some(&hi), 2).unwrap();
    let dims = f.array_of(&sliced).dims;
    // Python: len(range(7)[1:6:2]) == 3.
    assert_eq!(f.analysis.terms.constant_value(dims[0]), Some(3));
    assert!(f.checks().is_empty(), "a slice cannot fail");
}

// ---------------------------------------------------------------------------
// Broadcasting (§5.3)
// ---------------------------------------------------------------------------

#[test]
fn broadcasting_equalises_ranks_and_joins_extents() {
    let mut a = TermArena::new();
    let (three, four, one) = (a.constant(3), a.constant(4), a.constant(1));
    let (dims, defcond) = broadcast_dims(&mut a, &[&[three, one], &[four]]);
    assert_eq!(dims.len(), 2);
    assert_eq!(a.constant_value(dims[0]), Some(3));
    assert_eq!(a.constant_value(dims[1]), Some(4));
    assert!(a.is_true(defcond), "3x1 against 4 broadcasts");
}

#[test]
fn broadcasting_rejects_incompatible_constant_extents() {
    let mut a = TermArena::new();
    let (three, four) = (a.constant(3), a.constant(4));
    let (_, defcond) = broadcast_dims(&mut a, &[&[three], &[four]]);
    assert!(a.is_false(defcond), "3 against 4 cannot broadcast");
}

#[test]
fn broadcasting_keeps_a_symbolic_condition() {
    let mut a = TermArena::new();
    let v = a.declare_var("n".into(), Sort::Idx);
    let n = a.var(v);
    let four = a.constant(4);
    let (dims, defcond) = broadcast_dims(&mut a, &[&[n], &[four]]);
    // The extent is `bjoin(n, 4)`, an `ite` that neither operand folds away,
    // and `defcond` is the compatibility condition. Both are asserted by
    // what they *mean* under θ rather than by how the printer renders them:
    // a change to `bjoin`'s argument order is not a change to the rule.
    assert_eq!(a.constant_value(dims[0]), None, "the extent stays symbolic");
    for (value, extent, ok) in [(4i128, 4i128, true), (1, 4, true), (3, 3, false)] {
        let mut theta = Assignment::default();
        theta.set_int(v, value);
        assert_eq!(
            a.eval(dims[0], &theta),
            Ok(extent),
            "bjoin(n, 4) at n={}",
            value
        );
        assert_eq!(
            a.eval_bool(defcond, &theta),
            Ok(ok),
            "defcond at n={}",
            value
        );
    }
}

#[test]
fn e_zip_broadcasts_a_scalar_against_an_array() {
    let mut f = Fixture::new();
    let arr = Desc::Array(f.input(Dtype::I64, &[5]));
    let scalar = Desc::Scalar(f.literal(2));
    let out = e_zip(&mut f.ctx(), &arr, &scalar, Dtype::I64).unwrap();
    let dims = f.array_of(&out).dims;
    assert_eq!(dims.len(), 1);
    assert_eq!(f.analysis.terms.constant_value(dims[0]), Some(5));
    assert!(f.checks().is_empty(), "scalar broadcasting cannot fail");
}

// ---------------------------------------------------------------------------
// The end-to-end entry point and the JSON export
// ---------------------------------------------------------------------------

#[test]
fn analyze_builds_input_descriptors_before_walking_the_body() {
    let ast = serde_json::json!({
        "__class__": "ASTCircuit",
        "block": [{"__class__": "ASTPassStatement"}],
        "inputs": [
            {"name": "x",
             "annotation": {"dt": {"NDArray": {"shape": [8], "dtype": "Integer"}}}},
            {"name": "v",
             "annotation": {"dt": {"DynamicNDArray": {
                 "dtype": "Integer", "max_length": 16, "max_rank": 1}}}}
        ]
    });
    let circuit: crate::ast::ASTCircuit = serde_json::from_value(ast).unwrap();
    let empty = std::collections::HashMap::new();
    let config = Phase0Config::default();
    let result = super::analyze(&circuit, &empty, &empty, &config).expect("a body of `pass`");
    invariants::assert_record_invariants(&result);
    let named: Vec<&str> = result
        .descriptors
        .iter()
        .filter_map(|entry| entry.name.as_deref())
        .collect();
    assert_eq!(named, vec!["x", "v"], "Δ carries both inputs, in order");
    // `E-Input-VarLen` introduced exactly one variable, for `v`'s length.
    assert_eq!(result.phi.vars().len(), 1);
    assert!(
        result.delivery.is_empty(),
        "a circuit with no return delivers nothing"
    );
}

#[test]
fn config_follows_the_json_defaults() {
    let empty = serde_json::json!({});
    assert_eq!(
        Phase0Config::from_json(&empty),
        Phase0Config {
            loop_limit: 1000,
            recursion_limit: 16,
            work_budget: 4_000_000
        }
    );
    let given = serde_json::json!({
        "loop_limit": 32, "recursion_limit": 4, "phase0_work_budget": 64
    });
    assert_eq!(
        Phase0Config::from_json(&given),
        Phase0Config {
            loop_limit: 32,
            recursion_limit: 4,
            work_budget: 64
        }
    );
}

#[test]
fn export_carries_the_documented_schema() {
    let mut f = Fixture::new();
    let x = f.input(Dtype::I64, &[8]);
    let mask = f.input(Dtype::Bool, &[8]);
    let mask_key = f.key();
    let filtered = e_filter_1d(&mut f.ctx(), &x, &mask, &mask_key, "c").unwrap();
    let key = f.key();
    let producer = crate::phase0::source::Producer::Applied {
        rules: vec![crate::phase0::rules::Rule::EFilter1D],
        operands: vec![mask_key.clone()],
        condition: None,
    };
    f.analysis
        .record(mask_key.clone(), None, Desc::None, Producer::Opaque);
    f.analysis
        .record(key.clone(), Some("s".into()), filtered, producer);
    f.analysis.delivery.push(key);
    let result = f.finish();
    let json = super::export::to_json(&result, &super::export::ExportOptions::default());

    assert_eq!(json["schema"], super::export::SCHEMA);
    assert_eq!(json["descriptors"][1]["name"], "s");
    assert_eq!(json["descriptors"][1]["desc"]["kind"], "array");
    assert_eq!(json["descriptors"][1]["desc"]["dims_pretty"][0], "c");
    // The producer of §5a: the rule chain and the operands it read. A
    // reference to another value is its canonical key string, not the key
    // object the entry itself carries.
    assert_eq!(json["descriptors"][1]["producer"]["kind"], "applied");
    assert_eq!(json["descriptors"][1]["producer"]["rules"][0], "E-Filter-1D");
    assert_eq!(
        json["descriptors"][1]["producer"]["operands"][0],
        mask_key.encode()
    );
    // Every check names the value whose operation it belongs to.
    let checks = json["constraints"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "check")
        .count();
    assert!(json["obligations"].as_array().unwrap().len() >= checks);
    // A subject is the canonical key string too.
    for obligation in json["obligations"].as_array().unwrap() {
        assert!(obligation["subject"].is_string(), "{:?}", obligation);
    }
    assert_eq!(json["delivery"].as_array().unwrap().len(), 1);
    assert_eq!(json["variables"][0]["name"], "c");
    assert_eq!(json["variables"][0]["origin"]["kind"], "count");
    assert_eq!(json["variables"][0]["hi_pretty"], "8");
    assert_eq!(json["constraints"][0]["kind"], "var_range");
    assert!(json["smtlib"]
        .as_str()
        .unwrap()
        .contains("(declare-const c Int)"));
    assert!(json["terms"]["nodes"].as_array().unwrap().len() > 2);
    // The canonical key string of acceptance.md §2.2.
    assert_eq!(json["descriptors"][1]["key"]["key"], "0||\u{7c}2");

    // Without the pretty strings the same record carries the same terms.
    let plain = super::export::to_json(
        &result,
        &super::export::ExportOptions { pretty: false },
    );
    assert_eq!(plain["descriptors"][1]["desc"]["dims"], json["descriptors"][1]["desc"]["dims"]);
    assert!(plain["descriptors"][1]["desc"]["dims_pretty"][0].is_null());
}

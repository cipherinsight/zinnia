//! Syntactic interval evaluation of a term over the ranges $\Phi$ records.
//!
//! Two rules need an upper bound of a term without calling a solver:
//!
//! - `E-ValueSized` (`definition.md` §5.9) needs the declared maximum `M` of
//!   a length that depends on element values.
//! - `S-For-Range-Bounded` (`implementation.md` §8) needs the cap `C` of a
//!   `range` whose trip count is not a constant.
//!
//! Both are the same question, so both ask it here. The evaluation is
//! syntactic, in the sense of §1's "phase 0 makes no solver call, computes no
//! numeric bound": it reads the range every variable carries from its
//! introduction (§3.4) and pushes those bounds through the term language of
//! §3.1. A variable with no upper bound, or an operation that cannot keep
//! one, yields no bound, which the callers turn into a rejection.
//!
//! The result is an over-approximation: the interval contains every value the
//! term can take under $\Phi$, and may contain more. That is the direction
//! the callers need — a cap that is too large costs work, one that is too
//! small would be unsound.
//!
//! # The usable maximum
//!
//! §5.9 observes that a bound of $2^{63}$, the int64 dtype range, "is
//! useless": it is what $\Phi$ knows about *any* element value, so accepting
//! it as a declaration would accept every `np.zeros(a[0])`. A bound counts as
//! a declaration only if phase 0 could materialise that many iterations or
//! elements, which is [`crate::phase0::Phase0Config::work_budget`]; see
//! [`declared_maximum`].

use std::collections::HashMap;

use crate::phase0::phi::Phi;
use crate::phase0::term::{floor_div, IntNode, TermArena, TermId, I64_MAX};

/// An inclusive integer interval. `None` is unbounded on that side:
/// `lo = None` is $-\infty$, `hi = None` is $+\infty$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub lo: Option<i128>,
    pub hi: Option<i128>,
}

impl Interval {
    /// $(-\infty, +\infty)$: everything the term language can produce.
    pub const UNBOUNDED: Interval = Interval { lo: None, hi: None };

    pub fn exact(n: i128) -> Interval {
        Interval {
            lo: Some(n),
            hi: Some(n),
        }
    }

    pub fn new(lo: Option<i128>, hi: Option<i128>) -> Interval {
        Interval { lo, hi }
    }
}

/// The interval of `t` over the ranges of $\Phi$ (§3.4).
pub fn interval_of(terms: &TermArena, phi: &Phi, t: TermId) -> Interval {
    Eval {
        terms,
        phi,
        memo: HashMap::new(),
    }
    .of(t)
}

/// The declared maximum of `t`: its upper bound, when that bound is one
/// phase 0 could materialise.
///
/// `limit` is [`crate::phase0::Phase0Config::work_budget`], the largest
/// number of iterations or elements phase 0 will produce. A larger bound —
/// the int64 dtype range above all, which §5.9 calls useless — is reported
/// as [`NoMaximum::TooLarge`] rather than as a declaration, so that
/// `np.zeros(a[0])` and `range(a[0])` are rejected for the reason §5.9 and
/// §6.2 give.
pub fn declared_maximum(
    terms: &TermArena,
    phi: &Phi,
    t: TermId,
    limit: i128,
) -> Result<i128, NoMaximum> {
    match interval_of(terms, phi, t).hi {
        None => Err(NoMaximum::Unbounded),
        Some(hi) if hi > limit => Err(NoMaximum::TooLarge { bound: hi, limit }),
        // A negative upper bound is a maximum like any other: the length or
        // trip count clamps to zero.
        Some(hi) => Ok(hi),
    }
}

/// Why a term has no usable maximum, in the words a rejection uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoMaximum {
    /// $\Phi$ bounds the term nowhere above.
    Unbounded,
    /// The bound exists but is larger than phase 0 will materialise.
    TooLarge { bound: i128, limit: i128 },
}

impl NoMaximum {
    /// The clause a rejection detail appends after naming the term.
    ///
    /// The int64 case is named separately because it is the one §5.9
    /// describes: the bound exists, it is the dtype range, and it says
    /// nothing. What the source has to add is a declaration — an
    /// `@requires` bound on the scalar, or a `size=` on the operation.
    pub fn describe(self) -> String {
        match self {
            NoMaximum::Unbounded => "has no upper bound in Φ".to_string(),
            NoMaximum::TooLarge { bound, .. } if bound >= I64_MAX => {
                "has no declared maximum: Φ knows only the int64 dtype range for it, \
                 so declare one (`@requires` on the scalar, or `size=` here)"
                    .to_string()
            }
            NoMaximum::TooLarge { bound, limit } => format!(
                "has upper bound {}, past the {} elements phase 0 will materialise; \
                 declare a smaller maximum for it, or raise `phase0_work_budget`",
                bound, limit
            ),
        }
    }
}

/// One evaluation, with a memo over the term DAG.
///
/// The memo is what keeps the cost linear in the DAG rather than in the tree,
/// which matters because an unrolled loop shares sub-terms heavily. It doubles
/// as the cycle guard: a term is entered as unbounded before it is computed,
/// so a range that mentions its own variable through another one terminates
/// with no bound, which is sound.
struct Eval<'a> {
    terms: &'a TermArena,
    phi: &'a Phi,
    memo: HashMap<TermId, Interval>,
}

impl Eval<'_> {
    fn of(&mut self, t: TermId) -> Interval {
        if let Some(&cached) = self.memo.get(&t) {
            return cached;
        }
        // Reserve the entry so that a cyclic range — a variable bounded by a
        // term that mentions it — returns the unbounded interval instead of
        // recursing for ever.
        self.memo.insert(t, Interval::UNBOUNDED);
        let value = self.compute(t);
        self.memo.insert(t, value);
        value
    }

    fn compute(&mut self, t: TermId) -> Interval {
        match self.terms.int_node(t) {
            IntNode::Const(n) => Interval::exact(n),
            IntNode::Var(v) => {
                let info = self.phi.var_info(v);
                let lo = info.lo.and_then(|b| self.of(b).lo);
                let hi = info.hi.and_then(|b| self.of(b).hi);
                Interval::new(lo, hi)
            }
            IntNode::Add(a, b) => {
                let (a, b) = (self.of(a), self.of(b));
                Interval::new(add(a.lo, b.lo), add(a.hi, b.hi))
            }
            IntNode::Sub(a, b) => {
                let (a, b) = (self.of(a), self.of(b));
                Interval::new(sub(a.lo, b.hi), sub(a.hi, b.lo))
            }
            IntNode::Mul(k, t) => scale(self.of(t), k),
            IntNode::Div(t, k) => divide(self.of(t), k),
            IntNode::Mod(_, k) => modulus(k),
            IntNode::Min(a, b) => {
                let (a, b) = (self.of(a), self.of(b));
                Interval::new(min_lo(a.lo, b.lo), min_hi(a.hi, b.hi))
            }
            IntNode::Max(a, b) => {
                let (a, b) = (self.of(a), self.of(b));
                Interval::new(max_lo(a.lo, b.lo), max_hi(a.hi, b.hi))
            }
            // An `ite` takes either arm, so the interval is their hull.
            IntNode::Ite(a, b) => {
                let (a, b) = (self.of(a), self.of(b));
                Interval::new(min_lo(a.lo, b.lo), max_hi(a.hi, b.hi))
            }
            IntNode::Opaque => Interval::UNBOUNDED,
        }
    }
}

// ---------------------------------------------------------------------------
// The lattice operations
//
// `None` is the infinity on the side it appears: `-∞` for a `lo`, `+∞` for a
// `hi`. Overflow of the exact `i128` arithmetic is reported the same way, so
// a bound that cannot be represented is simply absent.
// ---------------------------------------------------------------------------

fn add(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    a?.checked_add(b?)
}

fn sub(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    a?.checked_sub(b?)
}

/// `min` of two lower bounds: $-\infty$ on either side wins.
fn min_lo(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    Some(a?.min(b?))
}

/// `min` of two upper bounds: $+\infty$ on one side leaves the other.
fn min_hi(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (value, None) | (None, value) => value,
    }
}

/// `max` of two lower bounds: $-\infty$ on one side leaves the other.
fn max_lo(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (value, None) | (None, value) => value,
    }
}

/// `max` of two upper bounds: $+\infty$ on either side wins.
fn max_hi(a: Option<i128>, b: Option<i128>) -> Option<i128> {
    Some(a?.max(b?))
}

/// `k * t`: the endpoints swap when `k` is negative.
fn scale(t: Interval, k: i128) -> Interval {
    let (lo, hi) = if k >= 0 { (t.lo, t.hi) } else { (t.hi, t.lo) };
    Interval::new(
        lo.and_then(|n| n.checked_mul(k)),
        hi.and_then(|n| n.checked_mul(k)),
    )
}

/// `t div k`, floor semantics. Division by a constant is monotone: ascending
/// for `k > 0`, descending for `k < 0`. `k` is never zero (§3.1).
fn divide(t: Interval, k: i128) -> Interval {
    if k == 0 {
        return Interval::UNBOUNDED;
    }
    let (lo, hi) = if k > 0 { (t.lo, t.hi) } else { (t.hi, t.lo) };
    Interval::new(lo.map(|n| floor_div(n, k)), hi.map(|n| floor_div(n, k)))
}

/// `t mod k`, floor semantics: the result takes the sign of the divisor, so
/// it lies in `[0, k)` for `k > 0` and in `(k, 0]` for `k < 0`.
fn modulus(k: i128) -> Interval {
    if k > 0 {
        Interval::new(Some(0), Some(k - 1))
    } else if k < 0 {
        Interval::new(Some(k + 1), Some(0))
    } else {
        Interval::UNBOUNDED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::desc::ValueKey;
    use crate::phase0::phi::VarOrigin;
    use crate::phase0::term::I64_MAX as DTYPE_MAX;
    use crate::phase0::term::{Sort, I64_MIN};

    /// The value key a fixture variable is introduced under; the interval
    /// evaluator never reads it.
    fn key() -> ValueKey {
        ValueKey {
            stmt_path: Vec::new(),
            iters: Vec::new(),
            calls: Vec::new(),
            expr_index: 0,
        }
    }

    /// A term arena and a $\Phi$ with one variable per declared range.
    struct Fixture {
        terms: TermArena,
        phi: Phi,
    }

    impl Fixture {
        fn new() -> Self {
            Fixture {
                terms: TermArena::new(),
                phi: Phi::new(),
            }
        }

        /// A variable with the inclusive range `[lo, hi]`, either side open.
        fn var(&mut self, name: &str, lo: Option<i128>, hi: Option<i128>) -> TermId {
            let lo = lo.map(|n| self.terms.constant(n));
            let hi = hi.map(|n| self.terms.constant(n));
            let v = self.phi.fresh_var(
                &mut self.terms,
                name,
                Sort::Idx,
                VarOrigin::DataScalar { value: key() },
                lo,
                hi,
            );
            self.terms.var(v)
        }

        fn interval(&self, t: TermId) -> Interval {
            interval_of(&self.terms, &self.phi, t)
        }
    }

    #[test]
    fn constant_is_exact() {
        let mut f = Fixture::new();
        let five = f.terms.constant(5);
        assert_eq!(f.interval(five), Interval::exact(5));
    }

    #[test]
    fn variable_takes_its_range() {
        let mut f = Fixture::new();
        let n = f.var("n", Some(0), Some(1024));
        assert_eq!(f.interval(n), Interval::new(Some(0), Some(1024)));
    }

    #[test]
    fn a_range_bound_that_is_itself_a_term_is_followed() {
        let mut f = Fixture::new();
        // `0 <= n <= 1024`, then `0 <= c <= n`: the count of a mask over an
        // input of variable length (§5.5).
        let n = f.var("n", Some(0), Some(1024));
        let lo = f.terms.constant(0);
        let v = f.phi.fresh_var(
            &mut f.terms,
            "c",
            Sort::Idx,
            VarOrigin::Count { mask: key() },
            Some(lo),
            Some(n),
        );
        let c = f.terms.var(v);
        assert_eq!(f.interval(c), Interval::new(Some(0), Some(1024)));
    }

    #[test]
    fn arithmetic_propagates_both_ends() {
        let mut f = Fixture::new();
        let n = f.var("n", Some(2), Some(10));
        let m = f.var("m", Some(1), Some(3));
        let sum = f.terms.add(n, m);
        assert_eq!(f.interval(sum), Interval::new(Some(3), Some(13)));
        let difference = f.terms.sub(n, m);
        assert_eq!(f.interval(difference), Interval::new(Some(-1), Some(9)));
        let scaled = f.terms.mul_const(-2, n);
        assert_eq!(f.interval(scaled), Interval::new(Some(-20), Some(-4)));
    }

    #[test]
    fn division_and_modulo_are_floor() {
        let mut f = Fixture::new();
        let n = f.var("n", Some(-7), Some(7));
        let positive = f.terms.div_const(n, 2).expect("2 is non-zero");
        assert_eq!(f.interval(positive), Interval::new(Some(-4), Some(3)));
        // `n div -2` is decreasing: 7 div -2 = -4 and -7 div -2 = 3.
        let negative = f.terms.div_const(n, -2).expect("-2 is non-zero");
        assert_eq!(f.interval(negative), Interval::new(Some(-4), Some(3)));
        let residue = f.terms.mod_const(n, 3).expect("3 is non-zero");
        assert_eq!(f.interval(residue), Interval::new(Some(0), Some(2)));
    }

    #[test]
    fn min_max_and_ite_mix_the_open_sides() {
        let mut f = Fixture::new();
        let open = f.var("d", None, None);
        let bounded = f.var("n", Some(0), Some(8));
        let lower = f.terms.min(open, bounded);
        assert_eq!(f.interval(lower), Interval::new(None, Some(8)));
        let upper = f.terms.max(open, bounded);
        assert_eq!(f.interval(upper), Interval::new(Some(0), None));
        let condition = f.terms.lt(open, bounded);
        let either = f.terms.ite(condition, open, bounded);
        assert_eq!(f.interval(either), Interval::UNBOUNDED);
    }

    #[test]
    fn a_trip_count_over_a_declared_bound_has_a_cap() {
        let mut f = Fixture::new();
        // `for i in range(2, n)` with `0 <= n <= 1024`: the trip count is
        // `max(0, (n - 2 + 0) div 1)` (§6.2, `range_len`).
        let n = f.var("n", Some(0), Some(1024));
        let two = f.terms.constant(2);
        let span = f.terms.sub(n, two);
        let zero = f.terms.constant(0);
        let trip = f.terms.max(zero, span);
        assert_eq!(f.interval(trip), Interval::new(Some(0), Some(1022)));
        assert_eq!(
            declared_maximum(&f.terms, &f.phi, trip, 4_000_000),
            Ok(1022)
        );
    }

    #[test]
    fn the_int64_dtype_range_is_not_a_declaration() {
        let mut f = Fixture::new();
        // `np.zeros(a[0])`: all Φ knows is the dtype range of §3.4, which
        // §5.9 calls useless.
        let element = f.var("v_elem", Some(I64_MIN), Some(DTYPE_MAX));
        assert_eq!(
            declared_maximum(&f.terms, &f.phi, element, 4_000_000),
            Err(NoMaximum::TooLarge {
                bound: DTYPE_MAX,
                limit: 4_000_000
            })
        );
    }

    #[test]
    fn an_unbounded_variable_has_no_maximum() {
        let mut f = Fixture::new();
        let open = f.var("t", Some(0), None);
        assert_eq!(
            declared_maximum(&f.terms, &f.phi, open, 4_000_000),
            Err(NoMaximum::Unbounded)
        );
    }
}

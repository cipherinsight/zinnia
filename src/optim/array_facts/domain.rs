//! The value domain of array facts: what every cell of a segment satisfies.
//!
//! A segment's [`ElemFacts`] combine three facts about the integer cells
//! `a[p]`, `p` in the segment (design: `docs/array-facts.md`, section 1):
//!
//! * a [`Range`] `lo <= a[p] <= hi`, whose ends are constants or a runtime
//!   integer plus a constant ([`Bound`]);
//! * an [`Order`] along the positions (`a[p] <= a[p + 1]`, ...);
//! * an [`Affine`] formula `a[p] = base + step * p`.
//!
//! Everything here is over the mathematical integers the circuit computes
//! (field elements far below the field size), like the rest of the fact
//! layer. Interval arithmetic uses checked `i128` and states nothing on
//! overflow; a range end that does not fit in `i64` is dropped.

use crate::types::StmtId;

/// What the cheap reasoning tiers (constants, intervals, scalar facts; never
/// a solver) know about a runtime integer that a symbolic [`Bound`] names.
pub trait Scalars {
    /// An interval `[min, max]` containing the statement's value on every
    /// accepted witness, if one is known.
    fn interval(&self, stmt: StmtId) -> Option<(i64, i64)>;
}

/// Knows nothing about any runtime integer.
pub struct NoScalars;

impl Scalars for NoScalars {
    fn interval(&self, _stmt: StmtId) -> Option<(i64, i64)> {
        None
    }
}

/// One end of a range or of a segment: a constant, or the value of a runtime
/// integer statement plus a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Bound {
    Const(i64),
    Stmt { stmt: StmtId, offset: i64 },
}

impl Bound {
    /// The value of `stmt` itself.
    pub fn stmt(stmt: StmtId) -> Bound {
        Bound::Stmt { stmt, offset: 0 }
    }

    /// The smallest value this bound takes on any accepted witness.
    pub fn least(&self, s: &dyn Scalars) -> Option<i64> {
        match *self {
            Bound::Const(c) => Some(c),
            Bound::Stmt { stmt, offset } => s.interval(stmt).and_then(|(min, _)| min.checked_add(offset)),
        }
    }

    /// The largest value this bound takes on any accepted witness.
    pub fn greatest(&self, s: &dyn Scalars) -> Option<i64> {
        match *self {
            Bound::Const(c) => Some(c),
            Bound::Stmt { stmt, offset } => s.interval(stmt).and_then(|(_, max)| max.checked_add(offset)),
        }
    }

    /// This bound plus `delta`, if it does not overflow.
    pub fn shifted(&self, delta: i64) -> Option<Bound> {
        match *self {
            Bound::Const(c) => c.checked_add(delta).map(Bound::Const),
            Bound::Stmt { stmt, offset } => offset.checked_add(delta).map(|offset| Bound::Stmt { stmt, offset }),
        }
    }

    /// `self <= other` on every accepted witness (`false` when not known).
    pub fn le(&self, other: &Bound, s: &dyn Scalars) -> bool {
        match (self, other) {
            (Bound::Const(a), Bound::Const(b)) => a <= b,
            (Bound::Stmt { stmt: a, offset: x }, Bound::Stmt { stmt: b, offset: y }) if a == b => x <= y,
            _ => match (self.greatest(s), other.least(s)) {
                (Some(a), Some(b)) => a <= b,
                _ => false,
            },
        }
    }

    /// `self < other` on every accepted witness.
    pub fn lt(&self, other: &Bound, s: &dyn Scalars) -> bool {
        self.shifted(1).is_some_and(|b| b.le(other, s))
    }

    /// The smaller of two bounds when one is known to be below the other,
    /// else the least constant either can take.
    fn lower_of(a: &Bound, b: &Bound, s: &dyn Scalars) -> Option<Bound> {
        if a.le(b, s) {
            Some(*a)
        } else if b.le(a, s) {
            Some(*b)
        } else {
            Some(Bound::Const(a.least(s)?.min(b.least(s)?)))
        }
    }

    /// The larger of two bounds when one is known to be above the other,
    /// else the greatest constant either can take.
    fn upper_of(a: &Bound, b: &Bound, s: &dyn Scalars) -> Option<Bound> {
        if a.le(b, s) {
            Some(*b)
        } else if b.le(a, s) {
            Some(*a)
        } else {
            Some(Bound::Const(a.greatest(s)?.max(b.greatest(s)?)))
        }
    }
}

/// `lo <= x <= hi` for every cell `x` (both ends inclusive; an absent end
/// bounds nothing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Range {
    pub lo: Option<Bound>,
    pub hi: Option<Bound>,
}

impl Range {
    /// Bounds nothing.
    pub const TOP: Range = Range { lo: None, hi: None };

    /// `lo <= x <= hi` with constant ends.
    pub fn new(lo: i64, hi: i64) -> Range {
        Range { lo: Some(Bound::Const(lo)), hi: Some(Bound::Const(hi)) }
    }

    /// Exactly `c`.
    pub fn point(c: i64) -> Range {
        Range::new(c, c)
    }

    /// 0 or 1.
    pub fn boolean() -> Range {
        Range::new(0, 1)
    }

    /// `x >= lo`.
    pub fn at_least(lo: i64) -> Range {
        Range { lo: Some(Bound::Const(lo)), hi: None }
    }

    pub fn is_top(&self) -> bool {
        self.lo.is_none() && self.hi.is_none()
    }

    /// A range holding every value of either (pointwise union, used for a
    /// cell that is one of two values).
    pub fn join(&self, other: &Range, s: &dyn Scalars) -> Range {
        let lo = match (&self.lo, &other.lo) {
            (Some(a), Some(b)) => Bound::lower_of(a, b, s),
            _ => None,
        };
        let hi = match (&self.hi, &other.hi) {
            (Some(a), Some(b)) => Bound::upper_of(a, b, s),
            _ => None,
        };
        Range { lo, hi }
    }

    /// Both ranges hold, kept as one range: each end is the tighter one when
    /// the two are comparable, else this range's (dropping the other's
    /// information is sound).
    pub fn meet(&self, other: &Range, s: &dyn Scalars) -> Range {
        let lo = match (&self.lo, &other.lo) {
            (Some(a), Some(b)) => Some(if a.le(b, s) { *b } else { *a }),
            (a, b) => a.or(*b),
        };
        let hi = match (&self.hi, &other.hi) {
            (Some(a), Some(b)) => Some(if a.le(b, s) { *a } else { *b }),
            (a, b) => a.or(*b),
        };
        Range { lo, hi }
    }

    /// Constant ends: the least value `lo` can take and the greatest `hi` can.
    pub fn interval(&self, s: &dyn Scalars) -> (Option<i64>, Option<i64>) {
        (self.lo.and_then(|b| b.least(s)), self.hi.and_then(|b| b.greatest(s)))
    }

    /// Every cell is `>= bound`.
    pub fn implies_ge(&self, bound: &Bound, s: &dyn Scalars) -> bool {
        self.lo.is_some_and(|lo| bound.le(&lo, s))
    }

    /// Every cell is `<= bound`.
    pub fn implies_le(&self, bound: &Bound, s: &dyn Scalars) -> bool {
        self.hi.is_some_and(|hi| hi.le(bound, s))
    }

    /// Every cell is 0 or 1.
    pub fn is_boolean(&self, s: &dyn Scalars) -> bool {
        self.implies_ge(&Bound::Const(0), s) && self.implies_le(&Bound::Const(1), s)
    }

    /// The range shifted by a constant: symbolic ends stay symbolic.
    pub fn shifted(&self, delta: i64) -> Range {
        Range { lo: self.lo.and_then(|b| b.shifted(delta)), hi: self.hi.and_then(|b| b.shifted(delta)) }
    }

    fn wide(&self, s: &dyn Scalars) -> Wide {
        let (lo, hi) = self.interval(s);
        Wide { lo: lo.map(i128::from), hi: hi.map(i128::from) }
    }
}

/// Interval arithmetic on constant ends in `i128` (a product of two `i64`
/// fits), back to a [`Range`] dropping any end outside `i64`.
#[derive(Clone, Copy, Debug)]
struct Wide {
    lo: Option<i128>,
    hi: Option<i128>,
}

impl Wide {
    fn range(self) -> Range {
        let fit = |v: Option<i128>| v.and_then(|v| i64::try_from(v).ok()).map(Bound::Const);
        Range { lo: fit(self.lo), hi: fit(self.hi) }
    }
}

/// Interval transfer functions of the integer operations (design 2.5).
/// Each returns a range containing the operation's result for every pair of
/// operand values in the operands' ranges.
pub mod transfer {
    use super::{Bound, Range, Scalars, Wide};

    /// `x + y`. A constant operand shifts symbolic ends.
    pub fn add(x: &Range, y: &Range, s: &dyn Scalars) -> Range {
        if let Some(c) = constant(y) {
            return x.shifted(c);
        }
        if let Some(c) = constant(x) {
            return y.shifted(c);
        }
        let (a, b) = (x.wide(s), y.wide(s));
        Wide { lo: sum(a.lo, b.lo), hi: sum(a.hi, b.hi) }.range()
    }

    /// `-x`.
    pub fn neg(x: &Range, s: &dyn Scalars) -> Range {
        let a = x.wide(s);
        Wide { lo: a.hi.map(|v| -v), hi: a.lo.map(|v| -v) }.range()
    }

    /// `x - y`.
    pub fn sub(x: &Range, y: &Range, s: &dyn Scalars) -> Range {
        if let Some(c) = constant(y) {
            return c.checked_neg().map_or(Range::TOP, |d| x.shifted(d));
        }
        add(x, &neg(y, s), s)
    }

    /// `x * y` for independent operands (the four corner products).
    pub fn mul(x: &Range, y: &Range, s: &dyn Scalars) -> Range {
        let (a, b) = (x.wide(s), y.wide(s));
        match (a.lo, a.hi, b.lo, b.hi) {
            (Some(al), Some(ah), Some(bl), Some(bh)) => {
                let corners = [al * bl, al * bh, ah * bl, ah * bh];
                Wide { lo: corners.iter().min().copied(), hi: corners.iter().max().copied() }.range()
            }
            _ => Range::TOP,
        }
    }

    /// `x * x` (one value times itself): never negative.
    pub fn square(x: &Range, s: &dyn Scalars) -> Range {
        let a = x.wide(s);
        let hi = match (a.lo, a.hi) {
            (Some(l), Some(h)) => Some((l * l).max(h * h)),
            _ => None,
        };
        let lo = match (a.lo, a.hi) {
            (Some(l), _) if l >= 0 => Some(l * l),
            (_, Some(h)) if h <= 0 => Some(h * h),
            _ => Some(0),
        };
        Wide { lo, hi }.range()
    }

    /// `|x|`.
    pub fn abs(x: &Range, s: &dyn Scalars) -> Range {
        let a = x.wide(s);
        match (a.lo, a.hi) {
            (Some(l), Some(h)) if l >= 0 => Wide { lo: Some(l), hi: Some(h) }.range(),
            (Some(l), Some(h)) if h <= 0 => Wide { lo: Some(-h), hi: Some(-l) }.range(),
            (Some(l), Some(h)) => Wide { lo: Some(0), hi: Some((-l).max(h)) }.range(),
            _ => Range::at_least(0),
        }
    }

    /// `min(x, y)`.
    pub fn min(x: &Range, y: &Range, s: &dyn Scalars) -> Range {
        let lo = match (&x.lo, &y.lo) {
            (Some(a), Some(b)) => Bound::lower_of(a, b, s),
            _ => None,
        };
        // Below both upper ends; either one is a sound upper end.
        let hi = match (&x.hi, &y.hi) {
            (Some(a), Some(b)) => Some(if a.le(b, s) { *a } else { *b }),
            (a, b) => a.or(*b),
        };
        Range { lo, hi }
    }

    /// `max(x, y)`.
    pub fn max(x: &Range, y: &Range, s: &dyn Scalars) -> Range {
        let hi = match (&x.hi, &y.hi) {
            (Some(a), Some(b)) => Bound::upper_of(a, b, s),
            _ => None,
        };
        let lo = match (&x.lo, &y.lo) {
            (Some(a), Some(b)) => Some(if a.le(b, s) { *b } else { *a }),
            (a, b) => a.or(*b),
        };
        Range { lo, hi }
    }

    /// `x // c` (Python's floor division) by a positive constant.
    pub fn floor_div_by(x: &Range, c: i64, s: &dyn Scalars) -> Range {
        if c <= 0 {
            return Range::TOP;
        }
        let a = x.wide(s);
        let c = i128::from(c);
        Wide { lo: a.lo.map(|v| v.div_euclid(c)), hi: a.hi.map(|v| v.div_euclid(c)) }.range()
    }

    /// `x % c` (Python's modulo) by a positive constant: in `[0, c - 1]`
    /// whatever `x` is.
    pub fn mod_by(c: i64) -> Range {
        if c <= 0 {
            return Range::TOP;
        }
        Range::new(0, c - 1)
    }

    fn constant(x: &Range) -> Option<i64> {
        match (x.lo, x.hi) {
            (Some(Bound::Const(a)), Some(Bound::Const(b))) if a == b => Some(a),
            _ => None,
        }
    }

    fn sum(a: Option<i128>, b: Option<i128>) -> Option<i128> {
        Some(a? + b?)
    }
}

/// How the cells of a segment are ordered along increasing positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    /// `a[p] <= a[p + 1]`.
    NonDecreasing,
    /// `a[p] < a[p + 1]`.
    Increasing,
    /// `a[p] >= a[p + 1]`.
    NonIncreasing,
    /// `a[p] > a[p + 1]`.
    Decreasing,
}

impl Order {
    /// The order of the same cells read from the last position to the first.
    pub fn reversed(self) -> Order {
        match self {
            Order::NonDecreasing => Order::NonIncreasing,
            Order::Increasing => Order::Decreasing,
            Order::NonIncreasing => Order::NonDecreasing,
            Order::Decreasing => Order::Increasing,
        }
    }

    /// The same order without strictness (`<` becomes `<=`).
    pub fn non_strict(self) -> Order {
        match self {
            Order::Increasing => Order::NonDecreasing,
            Order::Decreasing => Order::NonIncreasing,
            other => other,
        }
    }

    /// Whether `self` implies `other` (`<` implies `<=`).
    pub fn implies(self, other: Order) -> bool {
        self == other || self.non_strict() == other
    }
}

/// `a[p] = base + step * p` for every position `p` of the segment (positions
/// counted from the start of the array, not of the segment).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Affine {
    pub base: i64,
    pub step: i64,
}

impl Affine {
    /// The cell at position `p`, if it does not overflow.
    pub fn at(&self, p: usize) -> Option<i64> {
        self.step.checked_mul(i64::try_from(p).ok()?)?.checked_add(self.base)
    }

    /// The range of the cells at positions `[from, to]`.
    pub fn range(&self, from: usize, to: usize) -> Range {
        match (self.at(from), self.at(to)) {
            (Some(a), Some(b)) => Range::new(a.min(b), a.max(b)),
            _ => Range::TOP,
        }
    }
}

/// What every cell of one segment satisfies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ElemFacts {
    pub range: Range,
    pub order: Option<Order>,
    pub affine: Option<Affine>,
}

impl ElemFacts {
    /// Nothing known.
    pub const TOP: ElemFacts = ElemFacts { range: Range::TOP, order: None, affine: None };

    /// Only a range.
    pub fn range(range: Range) -> ElemFacts {
        ElemFacts { range, ..ElemFacts::TOP }
    }

    pub fn is_top(&self) -> bool {
        self.range.is_top() && self.order.is_none() && self.affine.is_none()
    }

    /// Facts of the cells of a segment that is, as a whole, one of two
    /// segments (a merge `select(c, a, b)` under one condition `c` for every
    /// cell: the merged array is `a` or it is `b`). Each side's facts hold
    /// for the side chosen, so their join holds: the ranges' join, the order
    /// both sides have, the formula both share.
    pub fn either(&self, other: &ElemFacts, s: &dyn Scalars) -> ElemFacts {
        let order = match (self.order, other.order) {
            (Some(a), Some(b)) if a.implies(b) => Some(b),
            (Some(a), Some(b)) if b.implies(a) => Some(a),
            _ => None,
        };
        ElemFacts {
            range: self.range.join(&other.range, s),
            order,
            affine: if self.affine == other.affine { self.affine } else { None },
        }
    }

    /// Facts of a cell that is one of two cells at the same position, chosen
    /// independently of its neighbours (a weak update): the ranges' join. A position-by-position choice
    /// between two ordered arrays need not be ordered, so order survives
    /// only with the formula: when both sides have the same affine formula,
    /// both cells at `p` are `base + step * p`, so the chosen one is too.
    pub fn join(&self, other: &ElemFacts, s: &dyn Scalars) -> ElemFacts {
        let same_formula = self.affine.is_some() && self.affine == other.affine;
        ElemFacts {
            range: self.range.join(&other.range, s),
            order: if same_formula { self.order } else { None },
            affine: if same_formula { self.affine } else { None },
        }
    }
}

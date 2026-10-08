//! Segmentations: an array's positions cut into runs that share one
//! abstract value (FunArray: P. Cousot, R. Cousot, F. Logozzo, "A Parametric
//! Segmentation Functor for Fully Automatic and Scalable Array Content
//! Analysis", POPL 2011).
//!
//! `{0} S1 {b1} S2 ... Sk {bk}`: segment `Si` holds the positions
//! `[b(i-1), bi)` and every cell there satisfies `Si`'s [`ElemFacts`].
//! Bounds are [`Bound`]s: constants, or a runtime integer plus a constant
//! (the live length of a compacted array: `{0} live {count} padding {n}`).
//! A segment whose bounds are not known to differ is possibly empty; a
//! position whose place relative to a symbolic bound is unknown may lie in
//! either neighbour, and gets the join of their facts.
//!
//! Not adopted from FunArray: widening (loops are unrolled, so nothing is
//! computed to a fixpoint) and sets of equal expressions per bound (one
//! expression per bound suffices for the producers we have).

use super::domain::{Bound, ElemFacts, Range, Scalars};


/// The positions `[previous end, end)` and what their cells satisfy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub end: Bound,
    pub facts: ElemFacts,
}

/// The facts of an array of `len` cells, as consecutive segments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segmentation {
    /// Number of positions (the capacity of a bounded-dynamic array).
    len: usize,
    /// Non-empty; the last segment ends at `len`.
    segments: Vec<Segment>,
    /// The cells are a permutation of `0, 1, ..., len - 1` (a fact about the
    /// whole array, meaningful only without symbolic bounds).
    permutation: bool,
}

impl Segmentation {
    /// One segment over all `len` positions.
    pub fn uniform(len: usize, facts: ElemFacts) -> Segmentation {
        Segmentation { len, segments: vec![Segment { end: Bound::Const(len as i64), facts }], permutation: false }
    }

    /// `{0} live {count} padding {len}`: the first `count` cells (a runtime
    /// length, `0 <= count <= len`) satisfy `live`, the rest `padding`.
    pub fn live_then_padding(len: usize, count: Bound, live: ElemFacts, padding: ElemFacts) -> Segmentation {
        Segmentation {
            len,
            segments: vec![Segment { end: count, facts: live }, Segment { end: Bound::Const(len as i64), facts: padding }],
            permutation: false,
        }
    }

    /// The same segmentation, also stating that the cells are a permutation
    /// of `0..len`.
    pub fn with_permutation(mut self) -> Segmentation {
        self.permutation = true;
        self
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn is_permutation(&self) -> bool {
        self.permutation
    }

    /// The same segmentation with only its order facts: what a list of float
    /// cells may carry (a range or formula would be about their fixed-point
    /// encoding, which no consumer reads).
    pub fn order_only(&self) -> Segmentation {
        let segments = self
            .segments
            .iter()
            .map(|seg| Segment { end: seg.end, facts: ElemFacts { order: seg.facts.order, ..ElemFacts::TOP } })
            .collect();
        Segmentation { len: self.len, segments, permutation: false }
    }

    /// Whether the segmentation states more than one range for all cells:
    /// an order, a formula, a permutation, or several segments.
    pub fn is_relational(&self) -> bool {
        self.permutation
            || self.segments.len() > 1
            || self.segments.iter().any(|seg| seg.facts.order.is_some() || seg.facts.affine.is_some())
    }

    /// Whether nothing is stated.
    pub fn is_top(&self) -> bool {
        !self.permutation && self.segments.iter().all(|seg| seg.facts.is_top())
    }

    /// The start of segment `k`.
    fn start(&self, k: usize) -> Bound {
        if k == 0 {
            Bound::Const(0)
        } else {
            self.segments[k - 1].end
        }
    }

    /// Indices of the segments that may contain a position in `[from, to]`.
    fn reaching(&self, from: usize, to: usize, s: &dyn Scalars) -> impl Iterator<Item = usize> + '_ {
        let (from, to) = (from as i64, to as i64);
        let ends: Vec<(Bound, Bound)> = (0..self.segments.len()).map(|k| (self.start(k), self.segments[k].end)).collect();
        let s_ref: Vec<bool> = ends
            .iter()
            .map(|(start, end)| {
                // Out of reach: the whole window lies before the start or at
                // or after the end, on every witness.
                let before = Bound::Const(to).lt(start, s);
                let after = end.le(&Bound::Const(from), s);
                !(before || after)
            })
            .collect();
        (0..self.segments.len()).filter(move |&k| s_ref[k])
    }

    /// What every cell at a position in `[from, to]` satisfies (the join of
    /// the facts of the segments that window may reach). For one position,
    /// `from == to`.
    pub fn over(&self, from: usize, to: usize, s: &dyn Scalars) -> ElemFacts {
        let mut reached = self.reaching(from, to, s).map(|k| self.segment_facts_over(k, from, to));
        let Some(first) = reached.next() else { return ElemFacts::TOP };
        reached.fold(first, |acc, f| acc.join(&f, s))
    }

    /// Segment `k`'s facts, with the range sharpened by its affine formula
    /// over the window.
    fn segment_facts_over(&self, k: usize, from: usize, to: usize) -> ElemFacts {
        let mut facts = self.segments[k].facts;
        if let Some(affine) = facts.affine {
            facts.range = affine.range(from, to);
        }
        if self.permutation && self.segments.len() == 1 && facts.range.is_top() && self.len > 0 {
            facts.range = Range::new(0, self.len as i64 - 1);
        }
        facts
    }

    /// The facts at one position.
    pub fn at(&self, p: usize, s: &dyn Scalars) -> ElemFacts {
        self.over(p, p, s)
    }

    /// What every cell satisfies.
    pub fn hull(&self, s: &dyn Scalars) -> ElemFacts {
        if self.len == 0 {
            return ElemFacts::TOP;
        }
        let facts = self.over(0, self.len - 1, s);
        // A hull over several segments keeps only the range; one segment
        // keeps its order (positions `0..len` are all of it).
        if self.segments.len() > 1 {
            return ElemFacts::range(facts.range);
        }
        facts
    }

    /// The facts of the positions `[from, to)` taken as an array of their
    /// own (a slice): a segmentation over `to - from` positions. Symbolic
    /// bounds are kept, shifted to the slice's positions; an affine formula
    /// is rebased.
    pub fn slice(&self, from: usize, to: usize, s: &dyn Scalars) -> Segmentation {
        if to <= from {
            return Segmentation::uniform(0, ElemFacts::TOP);
        }
        let mut segments: Vec<Segment> = Vec::new();
        for k in self.reaching(from, to - 1, s) {
            let mut facts = self.segments[k].facts;
            if let Some(affine) = facts.affine.as_mut() {
                match affine.at(from) {
                    Some(base) => affine.base = base,
                    None => facts.affine = None,
                }
            }
            let end = match self.segments[k].end {
                Bound::Const(e) => Bound::Const((e - from as i64).clamp(0, (to - from) as i64)),
                symbolic => match symbolic.shifted(-(from as i64)) {
                    Some(b) => b,
                    None => return Segmentation::uniform(to - from, self.over(from, to - 1, s)),
                },
            };
            segments.push(Segment { end, facts });
        }
        let len = to - from;
        match segments.last_mut() {
            Some(last) => last.end = Bound::Const(len as i64),
            None => return Segmentation::uniform(len, ElemFacts::TOP),
        }
        Segmentation { len, segments, permutation: self.permutation && from == 0 && to == self.len }
    }

    /// The sharpest range segment `k`'s facts give its cells: its stated
    /// range, met with its formula's values over its positions (all
    /// positions when a bound is symbolic) and, for a permutation stated as
    /// one segment, `[0, len - 1]`.
    fn segment_range(&self, k: usize, s: &dyn Scalars) -> Range {
        let facts = self.segments[k].facts;
        let mut range = facts.range;
        if let Some(affine) = facts.affine {
            let (from, to) = match (self.start(k), self.segments[k].end) {
                (Bound::Const(a), Bound::Const(b)) if a < b => (a as usize, b as usize - 1),
                _ if self.len > 0 => (0, self.len - 1),
                _ => return range,
            };
            range = range.meet(&affine.range(from, to), s);
        }
        if self.permutation && self.segments.len() == 1 && self.len > 0 {
            range = range.meet(&Range::new(0, self.len as i64 - 1), s);
        }
        range
    }

    /// The write rule: a value in `written` stored at a position in `reach`
    /// (inclusive; `None`: any position), possibly not at all (a runtime
    /// position, or a write under a runtime predicate). Every cell the write
    /// may reach is its old value or the written one, so each segment that
    /// window may reach has its range joined with `written`; it loses its
    /// order and formula, which relate cells the write may change. The
    /// permutation, a fact about all cells together, is dropped.
    pub fn weak_write(&self, written: &Range, reach: Option<(usize, usize)>, s: &dyn Scalars) -> Segmentation {
        let reached: Vec<usize> = match reach {
            Some((from, to)) => self.reaching(from, to, s).collect(),
            None => (0..self.segments.len()).collect(),
        };
        let segments = self
            .segments
            .iter()
            .enumerate()
            .map(|(k, seg)| {
                if reached.contains(&k) {
                    Segment { end: seg.end, facts: ElemFacts::range(self.segment_range(k, s).join(written, s)) }
                } else {
                    seg.clone()
                }
            })
            .collect();
        Segmentation { len: self.len, segments, permutation: false }
    }

    /// The write rule for a write that happens on every path at the constant
    /// position `p` (a strong update, FunArray's split): the cell at `p` is
    /// the written value, with the facts `written`, and every other cell
    /// keeps its value, so the segment holding `p` splits into the run before
    /// `p`, the cell `p` and the run after it, the two runs keeping their
    /// segment's facts (a run of an ordered run is ordered, and a formula
    /// counts positions from the start of the array). When the segment
    /// holding `p` is not known (a symbolic bound) it is a weak write at `p`.
    pub fn strong_write(&self, p: usize, written: &Range, s: &dyn Scalars) -> Segmentation {
        if p >= self.len {
            return self.clone();
        }
        let at = Bound::Const(p as i64);
        let holder = (0..self.segments.len())
            .find(|&k| self.start(k).le(&at, s) && at.lt(&self.segments[k].end, s));
        let Some(k) = holder else {
            return self.weak_write(written, Some((p, p)), s);
        };
        let seg = &self.segments[k];
        // The runs keep the segment's facts, with the range its formula or
        // the permutation gives made explicit (the permutation is dropped).
        let kept = ElemFacts { range: self.segment_range(k, s), ..seg.facts };
        let mut segments: Vec<Segment> = self.segments[..k].to_vec();
        if self.start(k) != at {
            segments.push(Segment { end: at, facts: kept });
        }
        segments.push(Segment { end: Bound::Const(p as i64 + 1), facts: ElemFacts::range(*written) });
        if seg.end != Bound::Const(p as i64 + 1) {
            segments.push(Segment { end: seg.end, facts: kept });
        }
        segments.extend_from_slice(&self.segments[k + 1..]);
        Segmentation { len: self.len, segments, permutation: false }
    }

    /// Both segmentations hold for the same cells: with the same segment
    /// bounds, each segment's ranges met and the order and formula either
    /// has; otherwise the one with more structure, its segments' ranges met
    /// with the other's range of all cells.
    pub fn meet(&self, other: &Segmentation, s: &dyn Scalars) -> Segmentation {
        assert_eq!(self.len, other.len, "two segmentations of one array");
        let same_bounds = self.segments.len() == other.segments.len()
            && self.segments.iter().zip(&other.segments).all(|(a, b)| a.end == b.end);
        if same_bounds {
            let segments = self
                .segments
                .iter()
                .zip(&other.segments)
                .enumerate()
                .map(|(k, (a, b))| {
                    let range = self.segment_range(k, s).meet(&other.segment_range(k, s), s);
                    let facts = ElemFacts { range, order: a.facts.order.or(b.facts.order), affine: a.facts.affine.or(b.facts.affine) };
                    Segment { end: a.end, facts }
                })
                .collect();
            return Segmentation { len: self.len, segments, permutation: self.permutation || other.permutation };
        }
        let (rich, poor) = if self.segments.len() >= other.segments.len() { (self, other) } else { (other, self) };
        let all = poor.hull(s).range;
        let segments = rich
            .segments
            .iter()
            .enumerate()
            .map(|(k, seg)| Segment { end: seg.end, facts: ElemFacts { range: rich.segment_range(k, s).meet(&all, s), ..seg.facts } })
            .collect();
        Segmentation { len: self.len, segments, permutation: rich.permutation }
    }

    /// The facts of an array that is, as a whole, this array or `other` (a
    /// merge `select(c, a, b)` with one condition for every cell). With the
    /// same segment bounds each segment gets [`ElemFacts::either`] of its two
    /// facts, and a permutation stays one; otherwise only what holds for
    /// all cells of each side is kept.
    pub fn either(&self, other: &Segmentation, s: &dyn Scalars) -> Segmentation {
        assert_eq!(self.len, other.len, "a merge of arrays of one length");
        let same_bounds = self.segments.len() == other.segments.len()
            && self.segments.iter().zip(&other.segments).all(|(a, b)| a.end == b.end);
        if !same_bounds {
            return Segmentation::uniform(self.len, self.hull(s).either(&other.hull(s), s));
        }
        let segments = self
            .segments
            .iter()
            .zip(&other.segments)
            .enumerate()
            .map(|(k, (a, b))| {
                let a_facts = ElemFacts { range: self.segment_range(k, s), ..a.facts };
                let b_facts = ElemFacts { range: other.segment_range(k, s), ..b.facts };
                Segment { end: a.end, facts: a_facts.either(&b_facts, s) }
            })
            .collect();
        Segmentation { len: self.len, segments, permutation: self.permutation && other.permutation }
    }

    /// The same cells read from the last position to the first.
    pub fn reversed(&self, s: &dyn Scalars) -> Segmentation {
        if self.segments.len() != 1 {
            // Reversing symbolic cuts would need `len - count` bounds; keep
            // the range only.
            return Segmentation::uniform(self.len, ElemFacts::range(self.hull(s).range));
        }
        let mut facts = self.segments[0].facts;
        facts.order = facts.order.map(|o| o.reversed());
        facts.affine = facts.affine.and_then(|a| {
            let last = a.at(self.len.saturating_sub(1))?;
            Some(super::domain::Affine { base: last, step: a.step.checked_neg()? })
        });
        Segmentation { len: self.len, segments: vec![Segment { end: Bound::Const(self.len as i64), facts }], permutation: self.permutation }
    }
}

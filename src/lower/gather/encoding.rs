//! Index encodings: a runtime integer index read as indicator bits.
//!
//! A gather by a runtime index `v` needs, per candidate value `k`, the bit
//! `[v == k]` (a one-hot) and, for a window, the bits `[v <= t]` (a
//! thermometer). Built from scratch, each indicator is an equality test
//! (7-11 halo2-base cells) and each thermometer bit a comparison. Some
//! constructions produce these bits as a by-product: a first/last scan over
//! flags (`scan.rs`) has the one-hot of its result and its running products
//! are a thermometer. Such a construction records an [`IndexEncoding`] on
//! its result value (`IRBuilder::index_encodings`, a postcondition fact),
//! and a gather asks for it before building indicators of its own
//! (`encoding_of`, which also follows `v + c` for a constant `c`).
//!
//! Every bit of an encoding is a 0/1 value that equals the stated indicator
//! on every witness the circuit accepts, under the producing construction's
//! precondition (for the scans: at least one flag is set; see `scan.rs`).

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{Value, ValueId};

/// A 0/1 value, or its complement `1 - value` (kept symbolic so a consumer
/// can fold the complement into its own gate: `h * (1 - b) = h - h * b`).
#[derive(Clone, Debug)]
pub struct Bit {
    pub value: Value,
    pub negated: bool,
}

impl Bit {
    pub fn plain(value: Value) -> Bit {
        Bit { value, negated: false }
    }

    pub fn negated(value: Value) -> Bit {
        Bit { value, negated: true }
    }

    /// The bit as one 0/1 value (emits `1 - value` when negated).
    pub fn materialize(&self, b: &mut IRBuilder) -> Value {
        if self.negated {
            let one = b.ir_constant_int(1);
            b.ir_sub_i(&one, &self.value)
        } else {
            self.value.clone()
        }
    }

    /// `h * self` for a 0/1 value `h`: one product, plus one subtraction
    /// (`h - h * value`) when the bit is negated.
    pub fn and(&self, b: &mut IRBuilder, h: &Value) -> Value {
        if let Some(c) = self.value.int_val() {
            let set = (c != 0) != self.negated;
            return if set { h.clone() } else { b.ir_constant_int(0) };
        }
        let product = b.ir_mul_i(h, &self.value);
        if self.negated {
            b.ir_sub_i(h, &product)
        } else {
            product
        }
    }
}

/// Indicator bits of a runtime integer `v` whose value lies in
/// `[lo, lo + onehot.len())`.
#[derive(Clone, Debug)]
pub struct IndexEncoding {
    /// The value of the first candidate.
    pub lo: i64,
    /// `onehot[k] = [v == lo + k]`: exactly one bit is 1.
    pub onehot: Vec<Value>,
    /// `below[k] = [v < lo + k]` for `k` in `1..onehot.len()` (index 0 is
    /// unused: `[v < lo]` is 0), when the producer has it.
    pub below: Option<Vec<Bit>>,
    /// The flags when `v - lo` is their last set position (`scan::last_set`):
    /// every flag at a position `>= v - lo + 1` is 0.
    pub last_of: Option<Vec<Value>>,
}

impl IndexEncoding {
    /// The number of candidate values.
    pub fn len(&self) -> usize {
        self.onehot.len()
    }

    /// The same bits as an encoding of `v + offset`.
    pub fn shifted(&self, offset: i64) -> IndexEncoding {
        IndexEncoding { lo: self.lo + offset, ..self.clone() }
    }

    /// `[v == t]` (the constant 0 outside the candidates).
    pub fn equals(&self, b: &mut IRBuilder, t: i64) -> Value {
        match usize::try_from(t - self.lo).ok().filter(|k| *k < self.len()) {
            Some(k) => self.onehot[k].clone(),
            None => b.ir_constant_int(0),
        }
    }

    /// `[v < t]` as a [`Bit`], when the encoding has a thermometer or `t`
    /// lies outside the candidates (`None` otherwise).
    pub fn less_than(&self, b: &mut IRBuilder, t: i64) -> Option<Bit> {
        let k = t - self.lo;
        if k <= 0 {
            return Some(Bit::plain(b.ir_constant_int(0)));
        }
        if k >= self.len() as i64 {
            return Some(Bit::plain(b.ir_constant_int(1)));
        }
        self.below.as_ref().map(|below| below[k as usize].clone())
    }
}

/// The encoding recorded on `v`, or on some `w` with `v = w + c` for a
/// constant `c`. Values are matched by their affine provenance
/// `sum a_i r_i + o`: two values with the same terms differ by a constant.
pub fn encoding_of(b: &IRBuilder, v: &Value) -> Option<IndexEncoding> {
    let vid = v.value_id()?;
    if let Some(enc) = b.index_encodings.by_value.get(&vid) {
        return Some(enc.clone());
    }
    let (root, offset) = affine_signature(b, vid)?;
    let (recorded_offset, enc) = b.index_encodings.by_root.get(&root)?;
    Some(enc.shifted(offset - recorded_offset))
}

/// Record `encoding` as a postcondition fact of `v`.
pub fn record_encoding(b: &mut IRBuilder, v: &Value, encoding: IndexEncoding) {
    let Some(vid) = v.value_id() else { return };
    if let Some((root, offset)) = affine_signature(b, vid) {
        b.index_encodings.by_root.insert(root, (offset, encoding.clone()));
    }
    b.index_encodings.by_value.insert(vid, encoding);
}

/// `(terms, o)` for a value whose affine provenance is `sum a_i r_i + o`
/// (a value without one is its own single root).
fn affine_signature(b: &IRBuilder, vid: ValueId) -> Option<(AffineTerms, i64)> {
    match b.provenance.affine_of(vid) {
        Some(view) if !view.terms.is_empty() => {
            let mut terms = view.terms.clone();
            terms.sort_unstable();
            Some((terms, view.offset))
        }
        Some(_) => None,
        None => Some((vec![(vid, 1)], 0)),
    }
}

/// The bits `[v == lo + k]` for `k` in `0..len`: read off the recorded
/// encoding when there is one (`v` lies in its candidates, so a value
/// outside them has the bit 0), else one equality test each.
pub fn onehot_over(b: &mut IRBuilder, v: &Value, lo: i64, len: usize) -> Vec<Value> {
    if let Some(enc) = encoding_of(b, v) {
        return (0..len).map(|k| enc.equals(b, lo + k as i64)).collect();
    }
    (0..len)
        .map(|k| {
            let c = b.ir_constant_int(lo + k as i64);
            b.ir_equal_i(v, &c)
        })
        .collect()
}

/// The builder's side table of recorded encodings.
#[derive(Default)]
pub struct EncodingTable {
    by_value: HashMap<ValueId, IndexEncoding>,
    /// By the affine terms of the recorded value: its offset and encoding.
    by_root: HashMap<AffineTerms, (i64, IndexEncoding)>,
}

/// The roots and scales of an affine view, sorted.
type AffineTerms = Vec<(ValueId, i64)>;

//! Index encodings: a runtime integer index read as indicator bits.
//!
//! A gather by a runtime index `v` needs, per candidate value `k`, the bit
//! `[v == k]` (a one-hot) and, for a window, the bits `[v < t]` (a
//! thermometer). Built from scratch, each indicator is an equality test
//! (7-11 halo2-base cells) and each thermometer bit a comparison. A scan by
//! running products (`scan.rs`) has both as by-products; it hands them to the
//! window or element read that consumes its result as the shared value
//! `ENCODING` of its group (`constructions.rs`).
//!
//! Every bit of an encoding is a 0/1 value that equals the stated indicator
//! on every witness the circuit accepts, under the producing scan's
//! precondition (see `scan.rs`).

use crate::builder::IRBuilder;
use crate::types::Value;

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

    /// The complement `1 - self`.
    pub fn not(&self) -> Bit {
        Bit { value: self.value.clone(), negated: !self.negated }
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
    /// (`h - h * value`) when the bit is negated; nothing for a constant.
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

    /// `[v < t]`, when the encoding has a thermometer or `t` lies outside
    /// the candidates (`None` otherwise).
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

/// The bits `[v == lo + k]` for `k` in `0..len`: read off `encoding` when
/// there is one (`v` lies in its candidates, so a value outside them has
/// the bit 0), else one equality test each.
pub fn onehot_over(b: &mut IRBuilder, v: &Value, encoding: Option<&IndexEncoding>, lo: i64, len: usize) -> Vec<Value> {
    if let Some(enc) = encoding {
        return (0..len).map(|k| enc.equals(b, lo + k as i64)).collect();
    }
    (0..len)
        .map(|k| {
            let c = b.ir_constant_int(lo + k as i64);
            b.ir_equal_i(v, &c)
        })
        .collect()
}

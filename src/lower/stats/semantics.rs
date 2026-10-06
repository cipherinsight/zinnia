//! What passes may know about a corrcoef lane before it is lowered.

use std::ops::Range;

use crate::prove::kernel;
use crate::provenance::Interval;
use crate::types::{NumberType, Value};

use crate::lower::op::{ArrayOp, Constant, OpKind, OperatorSemantics};

pub struct CorrcoefSemantics;

impl OperatorSemantics for CorrcoefSemantics {
    /// Integer elements fold to the exact-sums construction's value, bit for
    /// bit (`exact_integer_r`); float elements are not folded (the centred
    /// fixed-point chain is left to the IR's own folding).
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let OpKind::Corrcoef { n, integer: true, .. } = op.kind else { return None };
        let n = n as usize;
        let ints = |vs: &[Value]| -> Option<Vec<i64>> {
            vs.iter().map(|v| v.int_val().or_else(|| v.bool_val().map(i64::from))).collect()
        };
        let raw = exact_integer_r(&ints(&args[..n])?, &ints(&args[n..2 * n])?, PRECISION_BITS)?;
        // |raw| <= 2^32: exact in an f64, and quantizing it gives `raw` back.
        Some(Constant::Float(raw as f64 / (1u64 << PRECISION_BITS) as f64))
    }

    fn number_type(&self, _op: &ArrayOp, _args: &[Value]) -> NumberType {
        NumberType::Float
    }

    fn bounds(&self, _op: &ArrayOp, _operands: &[Option<Interval>]) -> Option<Interval> {
        None
    }

    fn picks_from(&self, _op: &ArrayOp) -> Option<Range<usize>> {
        None
    }
}

/// The fixed-point precision the folded constant is computed at (the
/// compiler's only supported precision; the Noir backend hardcodes it).
const PRECISION_BITS: u32 = 32;

/// The raw fixed-point `r` the exact-sums construction proves for integer
/// `x`, `y`: `floor(A · 2^(2P) / floor(sqrt(B·C) · 2^P))`, or `None` where it
/// has no proof (`B·C` outside `[1, 2^62)`, or `B` outside `[0, 2^64)`).
pub(crate) fn exact_integer_r(x: &[i64], y: &[i64], precision_bits: u32) -> Option<i128> {
    let n = x.len() as i128;
    let sum = |v: &[i64]| v.iter().map(|&e| e as i128).sum::<i128>();
    let dot = |u: &[i64], v: &[i64]| -> Option<i128> {
        u.iter().zip(v).try_fold(0i128, |acc, (&a, &b)| acc.checked_add((a as i128).checked_mul(b as i128)?))
    };
    let (sx, sy) = (sum(x), sum(y));
    let a = n.checked_mul(dot(x, y)?)?.checked_sub(sx.checked_mul(sy)?)?;
    let var_x = n.checked_mul(dot(x, x)?)?.checked_sub(sx.checked_mul(sx)?)?;
    let var_y = n.checked_mul(dot(y, y)?)?.checked_sub(sy.checked_mul(sy)?)?;
    if !(0..1i128 << 64).contains(&var_x) {
        return None;
    }
    let radicand = var_x.checked_mul(var_y)?;
    if !(1..1i128 << kernel::sqrt_int_bits(precision_bits)).contains(&radicand) {
        return None;
    }
    let root = kernel::isqrt_u128((radicand as u128) << (2 * precision_bits)) as i128;
    Some((a << (2 * precision_bits)).div_euclid(root))
}

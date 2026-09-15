//! Shape formulas shared by several rules of `definition.md` §5.
//!
//! Broadcasting appears in `E-Zip`, in `E-Where3`, in the batch axes of
//! `E-Matmul` and in `S-SetItem`; `norm` and the slice length appear in
//! `E-Slice` and in `arange(lo, hi, step)`; the size of an array appears in
//! `E-Reshape`, `E-Filter-Same`, `E-Nonzero`, `E-Reduce-Scalar` and
//! `E-Unique`. Each formula is written once, here.

use super::{Rule, RuleCtx};
use crate::phase0::desc::ValueKey;
use crate::phase0::phi::VarOrigin;
use crate::phase0::term::{BoolId, Sort, TermArena, TermId};
use crate::phase0::Reject;

/// `norm(b, d)` of `E-Slice` (§5.4): a Python slice endpoint normalised
/// against the axis extent `d`.
///
/// ```text
/// norm(b, d) = max(lower, min(upper, ite(b < 0, d + b, b)))
/// ```
///
/// with `(lower, upper) = (0, d)` for a positive step and `(-1, d - 1)` for a
/// negative one. §5.4 gives the positive-step form and says the clamp bound
/// for a negative step is `d - 1`; the lower clamp must then be `-1`, not
/// `0`, because Python's omitted upper bound for a negative step is `-d-1`,
/// which normalises to `-1` ("one before the start") and would otherwise
/// clamp to `0` and drop the first element. This matches CPython's
/// `slice.indices`, which uses `lower = -1` exactly when `step < 0`.
pub fn norm(terms: &mut TermArena, b: TermId, d: TermId, negative_step: bool) -> TermId {
    let zero = terms.constant(0);
    let below = terms.lt(b, zero);
    let shifted = terms.add(d, b);
    let wrapped = terms.ite(below, shifted, b);
    let (lower, upper) = if negative_step {
        let one = terms.constant(1);
        (terms.constant(-1), terms.sub(d, one))
    } else {
        (zero, d)
    };
    let clamped_high = terms.min(upper, wrapped);
    terms.max(lower, clamped_high)
}

/// The number of elements of `range(lo, hi, step)`, which is also the length
/// of a slice with normalised endpoints (§5.4):
///
/// ```text
/// len = max(0, (hi - lo + step - 1) div step)          step > 0
/// len = max(0, (lo - hi - step - 1) div (-step))       step < 0
/// ```
///
/// `None` when `step` is zero, which Python rejects.
pub fn range_len(terms: &mut TermArena, lo: TermId, hi: TermId, step: i64) -> Option<TermId> {
    let (span, magnitude) = if step > 0 {
        (terms.sub(hi, lo), step as i128)
    } else if step < 0 {
        (terms.sub(lo, hi), -(step as i128))
    } else {
        return None;
    };
    let bias = terms.constant(magnitude - 1);
    let biased = terms.add(span, bias);
    let quotient = terms.div_const(biased, magnitude)?;
    let zero = terms.constant(0);
    Some(terms.max(zero, quotient))
}

/// `bjoin(p, q) = ite(p = q, p, ite(p = 1, q, p))` (§5.3).
///
/// When `p != q` and neither is 1 the result is `p` arbitrarily: the
/// broadcasting `defcond` is false there and the dimension term is
/// irrelevant.
///
/// Two identities are applied before building the `ite`, both provable by
/// substituting into the definition: `bjoin(p, p) = p`, and `bjoin(p, 1) =
/// p` when the `1` is the literal (then the inner `ite` returns `p` on both
/// arms). The second is what makes broadcasting a scalar against an array
/// leave the array's extents untouched (§8.1's `s - p`).
pub fn bjoin(terms: &mut TermArena, p: TermId, q: TermId) -> TermId {
    if p == q {
        return p;
    }
    let one = terms.constant(1);
    if q == one {
        return p;
    }
    if p == one {
        return q;
    }
    let same = terms.eq(p, q);
    let p_is_one = terms.eq(p, one);
    let inner = terms.ite(p_is_one, q, p);
    terms.ite(same, p, inner)
}

/// Broadcast `operands` together (§5.3), right-aligning the dimension
/// vectors and prepending 1s to equalise ranks.
///
/// Returns the output dimensions and the broadcasting condition
/// `⋀_k ⋀_{i<j} (d_i = d_j ∨ d_i = 1 ∨ d_j = 1)`. For two operands this is
/// exactly `E-Zip`'s `defcond`; for three it is `E-Where3`'s, and requiring
/// every pair to be compatible is the same as requiring all non-1 extents on
/// an axis to be equal, which is NumPy's rule.
pub fn broadcast_dims(terms: &mut TermArena, operands: &[&[TermId]]) -> (Vec<TermId>, BoolId) {
    let rank = operands.iter().map(|d| d.len()).max().unwrap_or(0);
    let one = terms.constant(1);
    let mut dims = Vec::with_capacity(rank);
    let mut conditions = Vec::new();
    for axis in 0..rank {
        let extents: Vec<TermId> = operands
            .iter()
            .map(|d| {
                // Right-aligned: axis `rank - 1` is each operand's last.
                let offset = rank - d.len();
                if axis < offset {
                    one
                } else {
                    d[axis - offset]
                }
            })
            .collect();
        for i in 0..extents.len() {
            for j in i + 1..extents.len() {
                conditions.push(compatible(terms, extents[i], extents[j]));
            }
        }
        let joined = extents
            .iter()
            .copied()
            .reduce(|acc, e| bjoin(terms, acc, e))
            .unwrap_or(one);
        dims.push(joined);
    }
    let defcond = terms.all(&conditions);
    (dims, defcond)
}

/// `p = q ∨ p = 1 ∨ q = 1`, one axis of a broadcasting condition (§5.3).
fn compatible(terms: &mut TermArena, p: TermId, q: TermId) -> BoolId {
    let one = terms.constant(1);
    let same = terms.eq(p, q);
    let p_is_one = terms.eq(p, one);
    let q_is_one = terms.eq(q, one);
    let a = terms.or(same, p_is_one);
    terms.or(a, q_is_one)
}

/// `dims_a = dims_b` axis by axis. The caller has already checked that the
/// ranks agree (that is a premise, not a `defcond`).
pub fn dims_equal(terms: &mut TermArena, a: &[TermId], b: &[TermId]) -> BoolId {
    let per_axis: Vec<BoolId> = a
        .iter()
        .zip(b.iter())
        .map(|(&p, &q)| terms.eq(p, q))
        .collect();
    terms.all(&per_axis)
}

/// `-d ≤ i < d`, the validity of one integer index on an axis of extent `d`
/// (`E-Index` and `E-Elem`, §5.2 and §5.4).
pub fn in_bounds(terms: &mut TermArena, i: TermId, d: TermId) -> BoolId {
    let neg_d = terms.mul_const(-1, d);
    let low = terms.le(neg_d, i);
    let high = terms.lt(i, d);
    terms.and(low, high)
}

/// Python's negative index resolved against a length: `ite(i < 0, i + d, i)`.
///
/// A rule that has to say *which* element an index names — a composite's, not
/// an array's, whose elements share one descriptor — compares the resolved
/// form, or `v[-1]` would name no element at all.
pub fn wrap_index(terms: &mut TermArena, i: TermId, d: TermId) -> TermId {
    let zero = terms.constant(0);
    let negative = terms.lt(i, zero);
    let wrapped = terms.add(i, d);
    terms.ite(negative, wrapped, i)
}

/// $\mathrm{size}(D) = \prod_k \mathrm{dims}[k]$ (§3.3).
///
/// The product is a term when at most one axis is non-constant, since §3.1
/// allows `k * t` but no product of two non-constants. Otherwise a variable
/// $\sigma \ge 0$ stands for it and the product fact is recorded for phase 1
/// (§3.4, end).
pub fn size_term(ctx: &mut RuleCtx, dims: &[TermId], value: &ValueKey, base: &str) -> TermId {
    if let Some(t) = product_term(ctx.terms, dims) {
        return t;
    }
    let (constant_factor, symbolic) = split_factors(ctx.terms, dims);
    let zero = ctx.terms.constant(0);
    let sigma = ctx.phi.fresh_var(
        ctx.terms,
        base,
        Sort::Idx,
        VarOrigin::SizeProduct {
            value: value.clone(),
        },
        Some(zero),
        None,
    );
    let mut factors = symbolic;
    if constant_factor != 1 {
        let k = ctx.terms.constant(constant_factor);
        factors.insert(0, k);
    }
    ctx.phi
        .add_product_fact(sigma, factors, "size of a multi-axis dynamic array");
    ctx.terms.var(sigma)
}

/// The product of `dims` as a term, when at most one factor is non-constant.
/// `None` otherwise: §3.1 has no product of two non-constant terms.
pub fn product_term(terms: &mut TermArena, dims: &[TermId]) -> Option<TermId> {
    let (constant_factor, symbolic) = split_factors(terms, dims);
    match symbolic.len() {
        0 => Some(terms.constant(constant_factor)),
        1 => Some(terms.mul_const(constant_factor, symbolic[0])),
        _ => None,
    }
}

/// Separate `dims` into the product of its constant factors and the list of
/// its non-constant ones. An overflowing constant product is kept symbolic so
/// that no wrong numeral enters $\Phi$.
fn split_factors(terms: &TermArena, dims: &[TermId]) -> (i128, Vec<TermId>) {
    let mut constant_factor: i128 = 1;
    let mut symbolic = Vec::new();
    for &d in dims {
        match terms.constant_value(d) {
            Some(n) => match constant_factor.checked_mul(n) {
                Some(p) => constant_factor = p,
                None => symbolic.push(d),
            },
            None => symbolic.push(d),
        }
    }
    (constant_factor, symbolic)
}

/// Validate a constant permutation of `0..rank` (`E-Transpose`, §5.6). A
/// non-permutation is a static error, which the front end rejects.
pub fn permutation(axes: &[i64], rank: usize) -> Result<Vec<usize>, Reject> {
    if axes.len() != rank {
        return Err(Reject::Premise {
            rule: Rule::ETranspose,
            detail: format!("with {} axes for a rank-{} array", axes.len(), rank),
        });
    }
    let mut seen = vec![false; rank];
    let mut out = Vec::with_capacity(rank);
    for &a in axes {
        let k = normalise_axis(Rule::ETranspose, a, rank)?;
        if seen[k] {
            return Err(Reject::Premise {
                rule: Rule::ETranspose,
                detail: format!("with a repeated axis {}", a),
            });
        }
        seen[k] = true;
        out.push(k);
    }
    Ok(out)
}

/// Turn a possibly negative constant axis index into `0..rank`.
/// [`norm`] evaluated on literals, for the rules that work on a composite
/// whose length is a compile-time constant (§3.3): `E-Composite-Index`'s
/// slice form. The two must agree exactly, which
/// `tests::norm_and_slice_length_match_python_on_constants` drives from the
/// same CPython table.
pub fn constant_norm(b: i128, d: i128, negative_step: bool) -> i128 {
    let wrapped = if b < 0 { d + b } else { b };
    let (lower, upper) = if negative_step { (-1, d - 1) } else { (0, d) };
    lower.max(upper.min(wrapped))
}

/// [`range_len`] evaluated on literals, for the callers that have numbers
/// rather than terms: a `range` whose bounds are all constants (§6.2's
/// `S-For-Unrolled`) and the slice of a composite (§5.11). `None` for a
/// step of zero, as [`range_len`].
pub fn constant_range_len(lo: i128, hi: i128, step: i128) -> Option<i128> {
    let (span, magnitude) = match step.cmp(&0) {
        std::cmp::Ordering::Greater => (hi - lo, step),
        std::cmp::Ordering::Less => (lo - hi, -step),
        std::cmp::Ordering::Equal => return None,
    };
    Some(0.max(crate::phase0::term::floor_div(
        span + magnitude - 1,
        magnitude,
    )))
}

pub fn normalise_axis(rule: Rule, axis: i64, rank: usize) -> Result<usize, Reject> {
    let r = rank as i64;
    let k = if axis < 0 { axis + r } else { axis };
    if k < 0 || k >= r {
        return Err(Reject::Premise {
            rule,
            detail: format!("with axis {} on a rank-{} array", axis, rank),
        });
    }
    Ok(k as usize)
}

/// Like [`normalise_axis`] but for an insertion point, where `rank` itself is
/// a legal position (`E-Stack`, `expand_dims`).
pub fn normalise_insert_axis(rule: Rule, axis: i64, rank: usize) -> Result<usize, Reject> {
    let r = rank as i64;
    let k = if axis < 0 { axis + r + 1 } else { axis };
    if k < 0 || k > r {
        return Err(Reject::Premise {
            rule,
            detail: format!("with insertion axis {} on a rank-{} array", axis, rank),
        });
    }
    Ok(k as usize)
}

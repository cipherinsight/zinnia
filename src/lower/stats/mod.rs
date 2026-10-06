//! The statistics family: `np.corrcoef` of two 1-D arrays (`OpKind::Corrcoef`).
//!
//! The frontend emits one placeholder for Pearson's `r`, the off-diagonal
//! entry of `np.corrcoef(x, y)` (the diagonal is the constant 1). Its shared
//! operands are the `n` elements of `x`, the `n` elements of `y`, and, when
//! the call runs under a runtime execution predicate, that predicate (the
//! checks a construction emits hold only where the call runs; see
//! `builder/guard.rs`).
//!
//! Exactly one construction applies to a group, chosen by the element type,
//! never by the target: the two compute different fixed-point roundings of
//! the same real number, and a program must prove the same value on the mock
//! and on every backend.
//!
//! * [`ExactIntegerSums`] (integer or Boolean elements): the centred sums
//!   scaled by `n^2` are exact integers, `r = A / sqrt(B * C)`.
//! * [`CentredFixedPoint`] (any float element): the deviations from the
//!   fixed-point means, three rescaled sums of products, a rescaled product,
//!   its root and one division.

mod semantics;
#[cfg(test)]
mod tests;

pub use semantics::CorrcoefSemantics;

use crate::builder::IRBuilder;
use crate::helpers::value_ops::ensure_float;
use crate::types::{StmtId, Value};

use super::{Candidate, Construction, Context, EmitCx, Emitter, Group, Lane, OpKind, Requirement, Target};

/// The statistics family's constructions.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&ExactIntegerSums, &CentredFixedPoint]
}

/// `np.corrcoef` of integer (or Boolean) arrays by exact integer sums.
///
/// With `Sx = Σx`, `Sxx = Σx²`, `Sxy = Σxy` (and likewise for `y`), the
/// centred sums scaled by `n²` are the integers
/// `A = n·Sxy − Sx·Sy`, `B = n·Sxx − Sx²`, `C = n·Syy − Sy²`, and NumPy's
/// `r = cov / sqrt(var_x · var_y)` (its population or sample factors cancel)
/// is `A / sqrt(B · C)`. Emitted as: the sums (exact integer adds and
/// products); the root `y = floor(sqrt(B·C) · 2^32)` (`IR::SqrtProductIF`,
/// which admits `0 <= B < 2^64` and `0 <= B·C < 2^62`); and the float
/// quotient `float(A) / y` (`floor(A · 2^64 / y)`; the division's contract
/// asserts `y != 0`, which the division's own remainder bound implies, see
/// `prove::guards::divisor_guards`).
///
/// Preconditions: elements are int64 (in the signed 64-bit window, as every
/// integer the IR computes); `n < 2^31`.
/// Postconditions: the lane `r`; no fact.
///
/// Soundness and completeness. The elements are below `2^63` in magnitude,
/// so every sum and `A`, `B`, `C` lie far below `p / 2` (`|B|, |C| <=
/// n²·2^126`): they are the exact integers, and the true `B`, `C` are
/// non-negative (Cauchy–Schwarz). The root's domain check `B < 2^64` bounds
/// `|B·C| < n²·2^190 < p / 2`, so its field product is the integer `B·C`,
/// admitted exactly when `B·C < 2^62`, and the root is pinned (`isqrt`;
/// `kernel::fp_sqrt_product_int`). Then `|A| <= sqrt(B·C) < 2^31`, so
/// `float(A)` is in the window, and the division's quotient is pinned by its
/// remainder bound. `B < 2^64` never binds beyond `B·C < 2^62`: `C = 0`
/// gives `y = 0`, which has no proof (NumPy returns nan for a zero
/// variance). Admitted domain: `1 <= B·C < 2^62` (inputs within ±11585
/// always fit), the handwritten baselines' domain.
///
/// Accuracy: `|r − r_exact| <= 2^-31` (one floor in the root, relative
/// error below `2^-32 / sqrt(B·C)`, and one in the quotient), and `|r| <= 1`
/// holds exactly (`A² <= B·C − 1` unless `A² = B·C`, where the root is
/// exact), so NumPy's clip to `[-1, 1]` never acts.
///
/// Cost (measured on n = 4, `costs_halo2 --profile`): halo2-base ≈
/// `12·n + 440` cells at lookup width 6 (the root's three range checks and
/// `B`'s, the division's three); Noir ≈ `4·n + 60` gates. Only this
/// construction applies to integer elements, so the cost steers no choice.
pub struct ExactIntegerSums;

/// `np.corrcoef` of float arrays in fixed point: `dx = x − mean(x)` (one
/// fixed-point division by `n` per mean), the three sums of products each
/// rescaled once (`IRBuilder::ir_dot_f`), the rescaled product of the two
/// variances, its root (`np.sqrt`'s `x >= 0` contract) and one division
/// (its `!= 0` contract). Every step is a scalar fixed-point operation of
/// the IR, pinned by its own gadget. Preconditions: none. Postconditions:
/// the lane `r`. Cost: ≈ `60·n + 600` halo2-base cells, `10·n + 90` Noir
/// gates (unused for selection: the only construction for float inputs).
pub struct CentredFixedPoint;

impl Construction for ExactIntegerSums {
    fn name(&self) -> &'static str {
        "corrcoef: exact integer sums"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        match group.kind {
            OpKind::Corrcoef { n, integer: true, .. } if n < 1 << 31 => {
                let cost = match cx.target {
                    Target::Halo2Base => 12 * n as u64 + 440,
                    Target::Noir => 4 * n as u64 + 60,
                };
                vec![candidate(group, cost, Formula::ExactIntegerSums)]
            }
            _ => Vec::new(),
        }
    }
}

impl Construction for CentredFixedPoint {
    fn name(&self) -> &'static str {
        "corrcoef: centred fixed point"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        match group.kind {
            OpKind::Corrcoef { n, integer, .. } if !integer || n >= 1 << 31 => {
                let cost = match cx.target {
                    Target::Halo2Base => 60 * n as u64 + 600,
                    Target::Noir => 10 * n as u64 + 90,
                };
                vec![candidate(group, cost, Formula::CentredFixedPoint)]
            }
            _ => Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
enum Formula {
    ExactIntegerSums,
    CentredFixedPoint,
}

fn candidate(group: &Group, cost: u64, formula: Formula) -> Candidate {
    Candidate {
        pre: Vec::new(),
        post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
        cost,
        emitter: Box::new(CorrcoefEmitter { kind: group.kind, shared: group.shared.clone(), formula, r: None }),
    }
}

/// Emits `r` once, before the first lane; every lane is that value.
struct CorrcoefEmitter {
    kind: OpKind,
    shared: Vec<StmtId>,
    formula: Formula,
    r: Option<Value>,
}

impl Emitter for CorrcoefEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let OpKind::Corrcoef { n, guarded, .. } = self.kind else { unreachable!("a corrcoef group") };
        let n = n as usize;
        let operands = cx.values(&self.shared);
        let (x, y) = (&operands[..n], &operands[n..2 * n]);
        let formula = self.formula;
        let emit = |b: &mut IRBuilder| match formula {
            Formula::ExactIntegerSums => exact_integer_corrcoef(b, x, y),
            Formula::CentredFixedPoint => centred_fixed_point_corrcoef(b, x, y),
        };
        self.r = Some(if guarded { b.under_condition(&operands[2 * n], emit) } else { emit(b) });
    }

    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _lane: &Lane) -> Value {
        self.r.clone().expect("begin ran")
    }
}

/// `A / sqrt(B · C)` from exact integer sums ([`ExactIntegerSums`]).
fn exact_integer_corrcoef(b: &mut IRBuilder, x: &[Value], y: &[Value]) -> Value {
    let count = b.ir_constant_int(x.len() as i64);
    let (sx, sy) = (integer_sum(b, x), integer_sum(b, y));
    let (sxy, sxx, syy) = (integer_dot(b, x, y), integer_dot(b, x, x), integer_dot(b, y, y));
    let mut centred = |s: &Value, t: &Value, u: &Value| {
        let scaled = b.ir_mul_i(&count, s);
        let product = b.ir_mul_i(t, u);
        b.ir_sub_i(&scaled, &product)
    };
    let a = centred(&sxy, &sx, &sy);
    let var_x = centred(&sxx, &sx, &sx);
    let var_y = centred(&syy, &sy, &sy);
    // Where the call does not run, factors 1 · 1 keep the root's domain
    // checks satisfiable.
    let var_x = b.guard_operand(&var_x, 1);
    let var_y = b.guard_operand(&var_y, 1);
    let root = b.ir_sqrt_product_i_f(&var_x, &var_y);
    let numerator = b.ir_float_cast(&a);
    b.ir_div_f(&numerator, &root)
}

fn integer_sum(b: &mut IRBuilder, values: &[Value]) -> Value {
    let mut acc = b.ir_constant_int(0);
    for v in values {
        acc = b.ir_add_i(&acc, v);
    }
    acc
}

fn integer_dot(b: &mut IRBuilder, u: &[Value], v: &[Value]) -> Value {
    let mut acc = b.ir_constant_int(0);
    for (l, r) in u.iter().zip(v) {
        let term = b.ir_mul_i(l, r);
        acc = b.ir_add_i(&acc, &term);
    }
    acc
}

/// `cov / sqrt(var_x · var_y)` on fixed-point deviations ([`CentredFixedPoint`]).
fn centred_fixed_point_corrcoef(b: &mut IRBuilder, x: &[Value], y: &[Value]) -> Value {
    let deviations = |b: &mut IRBuilder, cells: &[Value]| -> Vec<Value> {
        let cells: Vec<Value> = cells.iter().map(|c| ensure_float(b, c)).collect();
        let sum = b.ir_sum_f(&cells);
        let count = b.ir_constant_float(cells.len() as f64);
        let mean = b.ir_div_f(&sum, &count);
        cells.iter().map(|c| b.ir_sub_f(c, &mean)).collect()
    };
    let dx = deviations(b, x);
    let dy = deviations(b, y);
    let pairs = |u: &[Value], v: &[Value]| -> Vec<(Value, Value)> { u.iter().cloned().zip(v.iter().cloned()).collect() };
    let cov = b.ir_dot_f(&pairs(&dx, &dy));
    let var_x = b.ir_dot_f(&pairs(&dx, &dx));
    let var_y = b.ir_dot_f(&pairs(&dy, &dy));
    let denom_sq = b.ir_mul_f(&var_x, &var_y);
    let denom = b.ir_sqrt_f(&denom_sq);
    b.ir_div_f(&cov, &denom)
}

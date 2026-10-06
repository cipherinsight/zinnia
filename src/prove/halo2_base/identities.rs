//! The second-phase checks of `IR::IdentityCheck`: each tests an exact
//! identity among first-phase cells at verifier challenges
//! (`circuit::SecondPhaseCheck`).
//!
//! **Soundness.** The cells a check reads are committed in the first phase,
//! before the challenges are drawn (`circuit.rs`). Each check computes a
//! polynomial in the challenges from those cells and constrains it to
//! vanish; the polynomial is the zero polynomial exactly when the identity
//! holds (stated per check), and a nonzero polynomial of total degree `d`
//! vanishes at a uniformly drawn point with probability at most `d / |F|`
//! (Schwartz-Zippel), `|F| ≈ 2^254` (`Identity::degree` gives `d`). A
//! false identity is therefore accepted with probability at most
//! `d / |F|`; a true one always (completeness: the honest cells make every
//! constraint hold whatever the challenges).
//!
//! **Shape.** Every check lays out the same cells for any values (the
//! circuit's shape is fixed at key generation): the loops depend only on
//! the identity's dimensions.
//!
//! **Cost** (second-phase advice cells, halo2-base's `a + b·c = d` gate;
//! a gate whose first cell is the previous gate's last shares it):
//! * multiset equality of `len` tuples of `width`: per tuple and side
//!   `3·(width − 1)` to compress (`width > 1`) and 6 for its factor of the
//!   grand product; once, the powers of `α` and `−(γ + 1)`;
//! * polynomial product: 4 per coefficient of each of the three
//!   polynomials (Horner), plus 4;
//! * matrix product (Freivalds): `3·(inner·cols + rows·inner + rows·cols)`
//!   for the three families of inner products, plus 4 per power of `γ`.

use halo2_base::gates::flex_gate::{GateChip, GateInstructions};
use halo2_base::halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_base::{AssignedValue, Context, QuantumCell};

use super::circuit::SecondPhaseCheck;

use QuantumCell::{Constant, Existing, Witness};

/// The two lists of `width`-tuples are equal as multisets.
///
/// With `width = 1`: `Π_j (x_j − γ) = Π_j (y_j − γ)`. Two lists of field
/// elements are equal as multisets exactly when these two monic
/// polynomials in `γ` are equal (unique factorization), and two different
/// ones agree on at most `len` points: degree `len`.
///
/// With `width > 1` each tuple is first compressed with the powers of a
/// second, independent challenge `α`: `t(α) = Σ_l c_l·α^l`. As polynomials
/// in `(α, γ)`, the factors `γ − t(α)` are irreducible and pairwise distinct
/// for distinct tuples, so `Π (γ − t_j(α)) = Π (γ − t'_j(α))` as
/// polynomials exactly when the tuple multisets are equal; their difference
/// has total degree at most `len·max(1, width − 1)`. (A single challenge
/// for both roles would not do: `γ − t(γ)` can coincide for different
/// tuples.)
pub(super) struct MultisetEquality {
    pub left: Vec<AssignedValue<Fp>>,
    pub right: Vec<AssignedValue<Fp>>,
    pub width: usize,
}

impl SecondPhaseCheck for MultisetEquality {
    fn challenges(&self) -> usize {
        if self.width > 1 {
            2
        } else {
            1
        }
    }

    fn reads(&self) -> Vec<AssignedValue<Fp>> {
        self.left.iter().chain(&self.right).copied().collect()
    }

    fn build(&self, ctx: &mut Context<Fp>, gate: &GateChip<Fp>, challenges: &[AssignedValue<Fp>]) {
        let gamma = challenges[0];
        let powers = if self.width > 1 { powers(ctx, gate, challenges[1], self.width - 1) } else { Vec::new() };
        // `−(γ + 1)`, the multiplier of the product's first gate.
        let shift = gate.mul_add(ctx, gamma, Constant(-Fp::one()), Constant(-Fp::one()));
        let product = |ctx: &mut Context<Fp>, values: &[AssignedValue<Fp>]| {
            let factors: Vec<AssignedValue<Fp>> =
                values.chunks_exact(self.width).map(|tuple| compress(ctx, tuple, &powers)).collect();
            grand_product(ctx, &factors, shift, gamma)
        };
        let left = product(ctx, &self.left);
        let right = product(ctx, &self.right);
        ctx.constrain_equal(&left, &right);
    }
}

/// `α, α^2, ..., α^count` (`count − 1` multiplications).
fn powers(ctx: &mut Context<Fp>, gate: &GateChip<Fp>, alpha: AssignedValue<Fp>, count: usize) -> Vec<AssignedValue<Fp>> {
    let mut powers = vec![alpha];
    while powers.len() < count {
        let next = gate.mul(ctx, *powers.last().expect("non-empty"), alpha);
        powers.push(next);
    }
    powers
}

/// `t = c_0 + Σ_{l ≥ 1} c_l·α^l` for the tuple `c` and `powers[l − 1] =
/// α^l`: a chain of gates `s + c_l·α^l = s'` sharing each partial sum
/// (`3·(width − 1)` cells; none for one component).
fn compress(ctx: &mut Context<Fp>, tuple: &[AssignedValue<Fp>], powers: &[AssignedValue<Fp>]) -> AssignedValue<Fp> {
    let (first, rest) = tuple.split_first().expect("a tuple has a component");
    let mut sum = *first;
    for (l, (c, power)) in rest.iter().zip(powers).enumerate() {
        let value = *sum.value() + *c.value() * power.value();
        sum = if l == 0 {
            ctx.assign_region_last([Existing(sum), Existing(*c), Existing(*power), Witness(value)], [0])
        } else {
            ctx.assign_region_last([Existing(*c), Existing(*power), Witness(value)], [-1])
        };
    }
    sum
}

/// `Π_j (t_j − γ)`, six cells per factor: with `shift = −(γ + 1)`, the
/// gates `P + P·shift = u` (so `u = −γ·P`) and `u + P·t = P'` (so
/// `P' = P·(t − γ)`), each sharing its first cell with the previous gate's
/// last. The empty product is 1.
fn grand_product(
    ctx: &mut Context<Fp>,
    factors: &[AssignedValue<Fp>],
    shift: AssignedValue<Fp>,
    gamma: AssignedValue<Fp>,
) -> AssignedValue<Fp> {
    let mut product: Option<AssignedValue<Fp>> = None;
    for t in factors {
        let p = product.map_or(Fp::one(), |p| *p.value());
        let u = -p * gamma.value();
        let next = p * (*t.value() - gamma.value());
        product = Some(match product {
            None => ctx.assign_region_last(
                [Constant(Fp::one()), Constant(Fp::one()), Existing(shift), Witness(u), Constant(Fp::one()), Existing(*t), Witness(next)],
                [0, 3],
            ),
            Some(p) => ctx.assign_region_last(
                [Existing(p), Existing(shift), Witness(u), Existing(p), Existing(*t), Witness(next)],
                [-1, 2],
            ),
        });
    }
    product.unwrap_or_else(|| ctx.load_constant(Fp::one()))
}

/// `a(γ)·b(γ) = c(γ)` for coefficient lists `a` (`n`), `b` (`m`) and `c`
/// (`n + m − 1`), lowest degree first. `a(x)·b(x) − c(x)` is the zero
/// polynomial exactly when `c_k = Σ_{i+j=k} a_i·b_j` for every `k`; else it
/// has degree at most `n + m − 2` and vanishes at that many points at most.
pub(super) struct PolynomialProduct {
    pub a: Vec<AssignedValue<Fp>>,
    pub b: Vec<AssignedValue<Fp>>,
    pub c: Vec<AssignedValue<Fp>>,
}

impl SecondPhaseCheck for PolynomialProduct {
    fn challenges(&self) -> usize {
        1
    }

    fn reads(&self) -> Vec<AssignedValue<Fp>> {
        self.a.iter().chain(&self.b).chain(&self.c).copied().collect()
    }

    fn build(&self, ctx: &mut Context<Fp>, gate: &GateChip<Fp>, challenges: &[AssignedValue<Fp>]) {
        let gamma = challenges[0];
        let a = horner(ctx, gate, &self.a, gamma);
        let b = horner(ctx, gate, &self.b, gamma);
        let c = horner(ctx, gate, &self.c, gamma);
        let product = gate.mul(ctx, a, b);
        ctx.constrain_equal(&product, &c);
    }
}

/// `Σ_k coefficients[k]·γ^k` by Horner's rule: one gate `c_k + acc·γ` per
/// coefficient below the leading one (4 cells each).
fn horner(ctx: &mut Context<Fp>, gate: &GateChip<Fp>, coefficients: &[AssignedValue<Fp>], gamma: AssignedValue<Fp>) -> AssignedValue<Fp> {
    let (leading, rest) = coefficients.split_last().expect("a polynomial has a coefficient");
    rest.iter().rev().fold(*leading, |acc, c| gate.mul_add(ctx, acc, gamma, *c))
}

/// `C = A·B` for `A` (`rows × inner`), `B` (`inner × cols`) and `C` (`rows ×
/// cols`), row-major, by Freivalds' check with the vector
/// `r = (1, γ, ..., γ^(cols − 1))`: `A·(B·r) = C·r`, row by row.
///
/// Row `i` of `A·(B·r) − C·r` is `Σ_j (A·B − C)_ij·γ^j`, a polynomial in `γ`
/// that is zero exactly when row `i` of `A·B − C` is; a nonzero one has
/// degree at most `cols − 1`. So if `C ≠ A·B` some row's equality fails
/// except with probability `(cols − 1) / |F|`. Powers of one challenge
/// suffice: each row is a univariate identity (no tuple is compressed).
pub(super) struct MatrixProduct {
    pub a: Vec<AssignedValue<Fp>>,
    pub b: Vec<AssignedValue<Fp>>,
    pub c: Vec<AssignedValue<Fp>>,
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
}

impl SecondPhaseCheck for MatrixProduct {
    fn challenges(&self) -> usize {
        1
    }

    fn reads(&self) -> Vec<AssignedValue<Fp>> {
        self.a.iter().chain(&self.b).chain(&self.c).copied().collect()
    }

    fn build(&self, ctx: &mut Context<Fp>, gate: &GateChip<Fp>, challenges: &[AssignedValue<Fp>]) {
        let (rows, inner, cols) = (self.rows, self.inner, self.cols);
        let r: Vec<QuantumCell<Fp>> = std::iter::once(Constant(Fp::one()))
            .chain(powers(ctx, gate, challenges[0], cols.saturating_sub(1)).into_iter().take(cols - 1).map(Existing))
            .collect();
        let dot = |ctx: &mut Context<Fp>, values: &[AssignedValue<Fp>], weights: Vec<QuantumCell<Fp>>| {
            gate.inner_product(ctx, values.iter().copied().map(Existing), weights)
        };
        let br: Vec<AssignedValue<Fp>> = self.b.chunks_exact(cols).map(|row| dot(ctx, row, r.clone())).collect();
        for i in 0..rows {
            let lhs = dot(ctx, &self.a[i * inner..(i + 1) * inner], br.iter().copied().map(Existing).collect());
            let rhs = dot(ctx, &self.c[i * cols..(i + 1) * cols], r.clone());
            ctx.constrain_equal(&lhs, &rhs);
        }
    }
}

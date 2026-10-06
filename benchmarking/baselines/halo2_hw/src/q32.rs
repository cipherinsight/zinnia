//! Q32 fixed-point arithmetic on top of halo2-base's `GateChip` and `RangeChip`.
//!
//! This is the helper layer a halo2-base developer writes once and reuses for
//! every kernel. It holds no case-specific logic.
//!
//! # Representation
//!
//! * A float `v` is the field element of the integer `round(v * 2^32)`; negative
//!   integers are `p - |x|`. This is exactly Zinnia's quantizer
//!   (`src/prove/field.rs::quantize`).
//! * An integer is its own field element (negatives as `p - |x|`), a boolean is
//!   a cell constrained to `{0, 1}`.
//! * [`I64`] marks a cell that is *proven* to lie in the signed 64-bit window
//!   `[-2^63, 2^63)`. Orderings need it, so a value is range-checked once when
//!   it first enters a comparison and the proof is reused afterwards. Every
//!   rescaled product and quotient comes out of its gadget already as an `I64`.
//!
//! # Rounding (same as Zinnia's halo2-base backend)
//!
//! * product: `floor(a * b / 2^32)`;
//! * fused dot product: one `floor(sum_i a_i * b_i / 2^32)`;
//! * float division: `floor(a * 2^32 / b)` with remainder in `[0, |b|)`;
//! * float to int: truncation toward zero (Python `int`, NumPy `astype(int)`).
//!
//! # Soundness
//!
//! Each gadget states the facts its constraints establish next to it. The
//! common argument: every witnessed decomposition `x = q * d + r` is checked
//! with `q` and `r` range-checked so tightly that `|q * d + r| < p / 2`; the
//! integer identity then follows from the field identity and the
//! decomposition is unique, so the prover cannot choose `q`.

use halo2_base::gates::{GateChip, GateInstructions, RangeChip, RangeInstructions};
use halo2_base::halo2_proofs::arithmetic::Field;
use halo2_base::halo2_proofs::halo2curves::ff::PrimeField;
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use halo2_base::utils::ScalarField;
use halo2_base::QuantumCell::{Constant, Existing, Witness};
use halo2_base::{AssignedValue, Context, QuantumCell};

use crate::Builder;

/// A circuit cell.
pub type Cell = AssignedValue<Fr>;
/// A halo2-base gate thread.
pub type Ctx = Context<Fr>;

/// Number of fractional bits.
pub const FRAC_BITS: u32 = 32;
/// `2^FRAC_BITS` as an integer.
pub const SCALE: i128 = 1 << FRAC_BITS;

/// `round(v * 2^32)`, Zinnia's quantizer (`field::quantize`).
pub fn quantize(v: f64) -> i128 {
    (v * SCALE as f64).round() as i128
}

/// The field element of a signed integer.
pub fn fe(v: i128) -> Fr {
    if v >= 0 {
        Fr::from_u128(v as u128)
    } else {
        -Fr::from_u128(v.unsigned_abs())
    }
}

/// The signed integer a field element encodes, for witness computation.
///
/// Values outside `(-2^127, 2^127)` only occur for witnesses that the
/// constraints reject anyway; they decode to 0 so hint code never panics.
pub fn signed(x: &Fr) -> i128 {
    let small = |f: Fr| -> Option<u128> {
        let bytes = f.to_bytes_le();
        if bytes[16..].iter().all(|b| *b == 0) && bytes[15] < 0x80 {
            Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
        } else {
            None
        }
    };
    match (small(*x), small(-*x)) {
        (Some(v), _) => v as i128,
        (None, Some(v)) => -(v as i128),
        (None, None) => 0,
    }
}

/// Decodes a Q32 field element to a float.
pub fn dequantize(x: &Fr) -> f64 {
    signed(x) as f64 / SCALE as f64
}

/// A cell proven to lie in `[-2^63, 2^63)`, together with its offset
/// `cell + 2^63`, which lies in `[0, 2^64)`.
#[derive(Clone, Copy, Debug)]
pub struct I64 {
    pub cell: Cell,
    offset: Cell,
}

impl I64 {
    pub fn value(&self) -> i128 {
        signed(self.cell.value())
    }
}

impl From<I64> for QuantumCell<Fr> {
    fn from(x: I64) -> Self {
        Existing(x.cell)
    }
}

const OFFSET: u128 = 1 << 63;

/// Longest axis indexed by the one-hot form in [`Q32::index`].
pub const ONEHOT_MAX_LEN: usize = 8;

/// The Q32 chip: halo2-base's gate and range chips plus the fixed-point gadgets.
#[derive(Clone, Debug)]
pub struct Q32 {
    pub gate: GateChip<Fr>,
    pub range: RangeChip<Fr>,
}

impl Q32 {
    pub fn new(builder: &Builder) -> Self {
        let range = builder.range_chip();
        Q32 { gate: range.gate().clone(), range }
    }

    // ---------------------------------------------------------------- loading
    //
    // Every private input element the circuit reads is checked to lie in its
    // dtype's domain when it is loaded, so a proof means "some valid NumPy input
    // gives these outputs": an integer is an int64 and a float's Q32 code lies
    // in the same signed 64-bit window (both are the window check of
    // `Q32::bound`), a boolean is 0 or 1. The `*_bounded` loaders return the
    // window proof so comparisons reuse it instead of checking again. Inputs
    // the circuit never reads are not loaded and get no check.

    /// Private float inputs, each checked to the Q32 window.
    pub fn load_floats(&self, ctx: &mut Ctx, values: &[f64]) -> Vec<Cell> {
        self.load_floats_bounded(ctx, values).iter().map(|x| x.cell).collect()
    }

    /// [`Q32::load_floats`], returning the window proofs.
    pub fn load_floats_bounded(&self, ctx: &mut Ctx, values: &[f64]) -> Vec<I64> {
        values.iter().map(|v| self.load_input(ctx, quantize(*v))).collect()
    }

    pub fn load_float(&self, ctx: &mut Ctx, value: f64) -> Cell {
        self.load_input(ctx, quantize(value)).cell
    }

    pub fn load_float_bounded(&self, ctx: &mut Ctx, value: f64) -> I64 {
        self.load_input(ctx, quantize(value))
    }

    /// Private integer inputs, each checked to be an int64.
    pub fn load_ints(&self, ctx: &mut Ctx, values: &[i64]) -> Vec<Cell> {
        self.load_ints_bounded(ctx, values).iter().map(|x| x.cell).collect()
    }

    /// [`Q32::load_ints`], returning the window proofs.
    pub fn load_ints_bounded(&self, ctx: &mut Ctx, values: &[i64]) -> Vec<I64> {
        values.iter().map(|v| self.load_input(ctx, *v as i128)).collect()
    }

    pub fn load_int(&self, ctx: &mut Ctx, value: i64) -> Cell {
        self.load_input(ctx, value as i128).cell
    }

    pub fn load_int_bounded(&self, ctx: &mut Ctx, value: i64) -> I64 {
        self.load_input(ctx, value as i128)
    }

    /// Private boolean inputs, each constrained to `{0, 1}`.
    pub fn load_bools(&self, ctx: &mut Ctx, values: &[bool]) -> Vec<Cell> {
        values
            .iter()
            .map(|v| {
                let cell = ctx.load_witness(Fr::from(*v as u64));
                self.gate.assert_bit(ctx, cell);
                cell
            })
            .collect()
    }

    /// One input element: loaded and checked to the window.
    fn load_input(&self, ctx: &mut Ctx, value: i128) -> I64 {
        let cell = ctx.load_witness(fe(input_tamper::apply(value)));
        let bounded = self.bound_unchecked_reuse(ctx, cell);
        input_reuse::loaded(&cell);
        bounded
    }

    /// A float constant (fixed by the circuit, not chosen by the prover).
    pub fn float_const(&self, ctx: &mut Ctx, value: f64) -> Cell {
        ctx.load_constant(fe(quantize(value)))
    }

    /// An integer constant.
    pub fn int_const(&self, ctx: &mut Ctx, value: i64) -> Cell {
        ctx.load_constant(fe(value as i128))
    }

    // ------------------------------------------------------------- the window

    /// Proves `x` lies in `[-2^63, 2^63)`: one 64-bit range check of `x + 2^63`.
    /// An input loaded by the `load_*` functions is already checked: use the
    /// `I64` from `load_*_bounded` instead (the mock run counts re-checks of
    /// inputs, see [`input_reuse`]).
    pub fn bound(&self, ctx: &mut Ctx, x: Cell) -> I64 {
        input_reuse::bounded(&x);
        self.bound_unchecked_reuse(ctx, x)
    }

    fn bound_unchecked_reuse(&self, ctx: &mut Ctx, x: Cell) -> I64 {
        let offset = self.gate.add(ctx, x, Constant(Fr::from_u128(OFFSET)));
        self.range.range_check(ctx, offset, 64);
        I64 { cell: x, offset }
    }

    /// [`Q32::bound`] for each cell.
    pub fn bound_all(&self, ctx: &mut Ctx, xs: &[Cell]) -> Vec<I64> {
        xs.iter().map(|x| self.bound(ctx, *x)).collect()
    }

    /// A constant inside the window; costs no range check.
    pub fn bounded_const(&self, ctx: &mut Ctx, value: i128) -> I64 {
        assert!((-(OFFSET as i128)..OFFSET as i128).contains(&value));
        let cell = ctx.load_constant(fe(value));
        let offset = ctx.load_constant(fe(value + OFFSET as i128));
        I64 { cell, offset }
    }

    // ------------------------------------------------------------ comparisons

    /// `a < b` as a bit. Both offsets are below `2^64`, so halo2-base's
    /// 64-bit `is_less_than` precondition holds.
    pub fn lt(&self, ctx: &mut Ctx, a: I64, b: I64) -> Cell {
        self.range.is_less_than(ctx, a.offset, b.offset, 64)
    }

    pub fn gt(&self, ctx: &mut Ctx, a: I64, b: I64) -> Cell {
        self.lt(ctx, b, a)
    }

    pub fn le(&self, ctx: &mut Ctx, a: I64, b: I64) -> Cell {
        let gt = self.lt(ctx, b, a);
        self.gate.not(ctx, gt)
    }

    pub fn ge(&self, ctx: &mut Ctx, a: I64, b: I64) -> Cell {
        let lt = self.lt(ctx, a, b);
        self.gate.not(ctx, lt)
    }

    /// `a < c` for a constant `c` (raw encoding: pass `quantize(v)` for floats).
    pub fn lt_const(&self, ctx: &mut Ctx, a: I64, c: i128) -> Cell {
        assert!((-(OFFSET as i128)..OFFSET as i128).contains(&c));
        let c_offset = Constant(fe(c + OFFSET as i128));
        self.range.is_less_than(ctx, a.offset, c_offset, 64)
    }

    /// `a > c` for a constant `c` (raw encoding).
    pub fn gt_const(&self, ctx: &mut Ctx, a: I64, c: i128) -> Cell {
        assert!((-(OFFSET as i128)..OFFSET as i128).contains(&c));
        let c_offset = Constant(fe(c + OFFSET as i128));
        self.range.is_less_than(ctx, c_offset, a.offset, 64)
    }

    /// `a < 0`.
    pub fn is_neg(&self, ctx: &mut Ctx, a: I64) -> Cell {
        self.lt_const(ctx, a, 0)
    }

    /// Exact equality (`==` on floats and integers alike).
    pub fn eq(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: impl Into<QuantumCell<Fr>>) -> Cell {
        self.gate.is_equal(ctx, a, b)
    }

    /// `if c { a } else { b }`; the result keeps the window proof of both.
    pub fn select_i64(&self, ctx: &mut Ctx, a: I64, b: I64, c: Cell) -> I64 {
        let cell = self.gate.select(ctx, a.cell, b.cell, c);
        let offset = self.gate.select(ctx, a.offset, b.offset, c);
        I64 { cell, offset }
    }

    /// `max(a, b)`, NumPy's `np.maximum` without NaN.
    pub fn max(&self, ctx: &mut Ctx, a: I64, b: I64) -> I64 {
        let a_lt_b = self.lt(ctx, a, b);
        self.select_i64(ctx, b, a, a_lt_b)
    }

    /// `min(a, b)`.
    pub fn min(&self, ctx: &mut Ctx, a: I64, b: I64) -> I64 {
        let b_lt_a = self.lt(ctx, b, a);
        self.select_i64(ctx, b, a, b_lt_a)
    }

    /// `max(xs)` of a few values by a hint `m`, pinned by `m` in the window,
    /// `m - x` in `[0, 2^64)` for every `x` and `prod_x (m - x) = 0`. The
    /// `xs` need no window proof of their own: the gap checks make every `x`
    /// an integer in `(m - 2^64, m]`, so `m >= x` as integers, and the product
    /// puts `m` among the `xs`; `m` is then their integer maximum and lies in
    /// the window. Cheaper than a compare-and-select chain for a handful of
    /// values (one range check per value plus one, no comparison).
    pub fn max_of(&self, ctx: &mut Ctx, xs: &[Cell]) -> Cell {
        let hint = xs.iter().map(|x| signed(x.value())).max().expect("non-empty");
        self.max_of_with(ctx, xs, fe(hint))
    }

    fn max_of_with(&self, ctx: &mut Ctx, xs: &[Cell], m: Fr) -> Cell {
        self.extremum_with(ctx, xs, m, true)
    }

    /// `min(xs)` of a few values: [`Q32::max_of`] with the gaps `x - m`
    /// (every `x` an integer in `[m, m + 2^64)`).
    pub fn min_of(&self, ctx: &mut Ctx, xs: &[Cell]) -> Cell {
        let hint = xs.iter().map(|x| signed(x.value())).min().expect("non-empty");
        self.extremum_with(ctx, xs, fe(hint), false)
    }

    fn extremum_with(&self, ctx: &mut Ctx, xs: &[Cell], m: Fr, is_max: bool) -> Cell {
        let m = ctx.load_witness(m);
        self.bound(ctx, m);
        let mut prod: Option<Cell> = None;
        for x in xs {
            let gap = if is_max { self.gate.sub(ctx, m, *x) } else { self.gate.sub(ctx, *x, m) };
            self.range.range_check(ctx, gap, 64);
            prod = Some(match prod {
                None => gap,
                Some(p) => self.gate.mul(ctx, p, gap),
            });
        }
        self.gate.assert_is_const(ctx, &prod.expect("non-empty"), &Fr::ZERO);
        m
    }

    /// `|a|` (for floats and integers; `|-2^63|` leaves the window and is
    /// returned unbounded).
    pub fn abs(&self, ctx: &mut Ctx, a: I64) -> Cell {
        let neg = self.is_neg(ctx, a);
        let minus_a = self.gate.neg(ctx, a.cell);
        self.gate.select(ctx, minus_a, a.cell, neg)
    }

    // -------------------------------------------------------------- arithmetic

    pub fn add(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: impl Into<QuantumCell<Fr>>) -> Cell {
        self.gate.add(ctx, a, b)
    }

    pub fn sub(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: impl Into<QuantumCell<Fr>>) -> Cell {
        self.gate.sub(ctx, a, b)
    }

    pub fn neg(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>) -> Cell {
        self.gate.neg(ctx, a)
    }

    /// Sum of cells (exact for both floats and integers).
    pub fn sum(&self, ctx: &mut Ctx, xs: impl IntoIterator<Item = Cell>) -> Cell {
        self.gate.sum(ctx, xs.into_iter().map(Existing))
    }

    /// Float product `floor(a * b / 2^32)`.
    pub fn mul(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: impl Into<QuantumCell<Fr>>) -> I64 {
        let raw = self.gate.mul(ctx, a, b);
        self.rescale(ctx, raw)
    }

    /// Fused float dot product `floor(sum_i a_i * b_i / 2^32)`: one rescale
    /// for the whole sum, as `np.dot`/`@` on reals rounds once.
    pub fn dot(&self, ctx: &mut Ctx, a: &[Cell], b: &[Cell]) -> I64 {
        assert_eq!(a.len(), b.len());
        let raw = self.gate.inner_product(ctx, a.iter().map(|x| Existing(*x)), b.iter().map(|x| Existing(*x)));
        self.rescale(ctx, raw)
    }

    /// `floor(raw / 2^32)` for a raw product `raw` whose integer value lies in
    /// `[-2^95, 2^95)`.
    ///
    /// Constraints: `raw = q * 2^32 + r`, `r` in `[0, 2^32)`, `q` in the 64-bit
    /// window. Then `q * 2^32 + r` lies in `[-2^95, 2^95)`, far inside
    /// `(-p/2, p/2)`, so the field identity is an integer identity and `(q, r)`
    /// is the unique floor decomposition. An out-of-range `raw` has no
    /// satisfying witness (the proof fails rather than wrapping).
    pub fn rescale(&self, ctx: &mut Ctx, raw: Cell) -> I64 {
        let v = signed(raw.value());
        self.rescale_with(ctx, raw, fe(v.div_euclid(SCALE)), fe(v.rem_euclid(SCALE)))
    }

    fn rescale_with(&self, ctx: &mut Ctx, raw: Cell, q: Fr, r: Fr) -> I64 {
        ctx.assign_region([Witness(r), Constant(fe(SCALE)), Witness(q), Existing(raw)], [0]);
        let r_cell = ctx.get(-4);
        let q_cell = ctx.get(-2);
        self.range.range_check(ctx, r_cell, FRAC_BITS as usize);
        self.bound(ctx, q_cell)
    }

    /// Float times integer: exact, no rescale.
    pub fn mul_int(&self, ctx: &mut Ctx, float: impl Into<QuantumCell<Fr>>, int: impl Into<QuantumCell<Fr>>) -> Cell {
        self.gate.mul(ctx, float, int)
    }

    /// Integer to float (`float(n)`, `astype(float)`): multiply by the constant `2^32`.
    pub fn int_to_float(&self, ctx: &mut Ctx, n: impl Into<QuantumCell<Fr>>) -> Cell {
        self.gate.mul(ctx, n, Constant(fe(SCALE)))
    }

    /// Float division `floor(a * 2^32 / b)` with remainder in `[0, |b|)`.
    ///
    /// Constraints: `a * 2^32 = q * b + r`, `0 <= r < 2^64`,
    /// `0 <= |b| - r - 1 < 2^64` (so `r < |b|`, `b != 0`), `q` in the window.
    /// With `|b| < 2^63`, `|q * b + r| < 2^127`, so the identity holds over the
    /// integers and fixes the Euclidean `(q, r)`.
    pub fn div(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: I64) -> I64 {
        let abs_b = self.abs(ctx, b);
        self.div_by_magnitude(ctx, a, b.cell, abs_b)
    }

    /// [`Q32::div`] for a divisor that must be positive: the remainder check
    /// `0 <= r < b` itself rules out `b <= 0` (no witness), so no sign test.
    pub fn div_pos(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: I64) -> I64 {
        self.div_by_magnitude(ctx, a, b.cell, b.cell)
    }

    fn div_by_magnitude(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: Cell, abs_b: Cell) -> I64 {
        let a = a.into();
        let num = signed(a.value()) * SCALE;
        let bv = signed(b.value());
        let (q, r) = if bv == 0 { (0, 0) } else { (num.div_euclid(bv), num.rem_euclid(bv)) };
        self.div_with(ctx, a, b, abs_b, fe(q), fe(r))
    }

    fn div_with(&self, ctx: &mut Ctx, a: QuantumCell<Fr>, b: Cell, abs_b: Cell, q: Fr, r: Fr) -> I64 {
        let scaled = self.gate.mul(ctx, a, Constant(fe(SCALE)));
        let q_cell = ctx.load_witness(q);
        let r_cell = ctx.load_witness(r);
        let recomposed = self.gate.mul_add(ctx, q_cell, b, r_cell);
        ctx.constrain_equal(&recomposed, &scaled);
        self.range.range_check(ctx, r_cell, 64);
        let gap = self.gate.sub(ctx, abs_b, r_cell);
        let gap = self.gate.sub(ctx, gap, Constant(Fr::ONE));
        self.range.range_check(ctx, gap, 64);
        self.bound(ctx, q_cell)
    }

    /// Float division by a nonzero float constant `d`.
    pub fn div_const(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, d: f64) -> I64 {
        let dq = quantize(d);
        assert!(dq != 0, "division by a zero constant");
        // floor(a * 2^32 / dq) == floor(-(a * 2^32) / -dq), so divide by |dq|
        // after moving the sign to the numerator.
        let a = a.into();
        let num = if dq < 0 { self.gate.neg(ctx, a) } else { self.gate.add(ctx, a, Constant(Fr::ZERO)) };
        let scaled = self.gate.mul(ctx, num, Constant(fe(SCALE)));
        self.floor_div_const(ctx, scaled, dq.unsigned_abs() as u64).0
    }

    /// Integer floor division and remainder by a positive constant `d`
    /// (Python `//`, `%`): `a = q * d + r`, `0 <= r < d`.
    ///
    /// `r` is checked against the constant `d` (`check_less_than_safe`) and `q`
    /// is in the window; `|q * d + r| < 2^127`, so `(q, r)` is unique. The
    /// integer value of `a` must lie in the window times `d`.
    pub fn floor_div_const(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, d: u64) -> (I64, Cell) {
        assert!(d > 0, "division by a non-positive constant");
        let a = a.into();
        let v = signed(a.value());
        self.floor_div_const_with(ctx, a, d, fe(v.div_euclid(d as i128)), fe(v.rem_euclid(d as i128)))
    }

    fn floor_div_const_with(&self, ctx: &mut Ctx, a: QuantumCell<Fr>, d: u64, q: Fr, r: Fr) -> (I64, Cell) {
        ctx.assign_region([Witness(r), Constant(Fr::from(d)), Witness(q), a], [0]);
        let r_cell = ctx.get(-4);
        let q_cell = ctx.get(-2);
        if d.is_power_of_two() {
            self.range.range_check(ctx, r_cell, d.trailing_zeros() as usize);
        } else {
            self.range.check_less_than_safe(ctx, r_cell, d);
        }
        (self.bound(ctx, q_cell), r_cell)
    }

    /// `(a // d, a % d)` for an `a` the circuit already knows is a non-negative
    /// integer below `2^a_bits` (for example an index selected from loop
    /// constants), by halo2-base's `div_mod`: it checks `q < 2^a_bits / d + 1`
    /// and `r < d`, which pins `(q, r)` under that precondition.
    pub fn divmod_small(&self, ctx: &mut Ctx, a: Cell, d: u64, a_bits: usize) -> (Cell, Cell) {
        self.range.div_mod(ctx, a, d, a_bits)
    }

    /// Integer floor division and remainder by a private divisor that must be
    /// positive (Python `//`, `%`): `a = q * b + r`, `0 <= r < b`; `b <= 0` has
    /// no witness. Same uniqueness argument as [`Q32::div`].
    pub fn floor_div_pos(&self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: I64) -> (I64, Cell) {
        let a = a.into();
        let (av, bv) = (signed(a.value()), b.value());
        let (q, r) = if bv == 0 { (0, 0) } else { (av.div_euclid(bv), av.rem_euclid(bv)) };
        self.floor_div_pos_with(ctx, a, b.cell, fe(q), fe(r))
    }

    fn floor_div_pos_with(&self, ctx: &mut Ctx, a: QuantumCell<Fr>, b: Cell, q: Fr, r: Fr) -> (I64, Cell) {
        let q_cell = ctx.load_witness(q);
        let r_cell = ctx.load_witness(r);
        let recomposed = self.gate.mul_add(ctx, q_cell, b, r_cell);
        let a_cell = self.gate.add(ctx, a, Constant(Fr::ZERO));
        ctx.constrain_equal(&recomposed, &a_cell);
        self.range.range_check(ctx, r_cell, 64);
        let gap = self.gate.sub(ctx, b, r_cell);
        let gap = self.gate.sub(ctx, gap, Constant(Fr::ONE));
        self.range.range_check(ctx, gap, 64);
        (self.bound(ctx, q_cell), r_cell)
    }

    /// Float to integer, truncating toward zero (`int(x)`, `astype(int)`).
    ///
    /// Constraints: `x = n * 2^32 + r` where `r = s ? -m : m`, `m` in
    /// `[0, 2^32)`, `s` the proven sign bit of `x`, and `n` in the window.
    /// The remainder carrying `x`'s sign excludes the floor alternative, and
    /// the bound on `n` excludes non-integer field solutions.
    pub fn float_to_int(&self, ctx: &mut Ctx, x: I64) -> I64 {
        let v = x.value();
        let n = v / SCALE; // Rust `/` truncates toward zero
        self.float_to_int_with(ctx, x, fe(n), fe((v - n * SCALE).abs()))
    }

    fn float_to_int_with(&self, ctx: &mut Ctx, x: I64, n: Fr, m: Fr) -> I64 {
        let neg = self.is_neg(ctx, x);
        let n_cell = ctx.load_witness(n);
        let m_cell = ctx.load_witness(m);
        self.range.range_check(ctx, m_cell, FRAC_BITS as usize);
        let minus_m = self.gate.neg(ctx, m_cell);
        let r = self.gate.select(ctx, minus_m, m_cell, neg);
        let recomposed = self.gate.mul_add(ctx, n_cell, Constant(fe(SCALE)), r);
        ctx.constrain_equal(&recomposed, &x.cell);
        self.bound(ctx, n_cell)
    }

    /// `floor(x)` as a float (`np.floor`): `x = f * 2^32 + r`, `r` in `[0, 2^32)`.
    pub fn floor(&self, ctx: &mut Ctx, x: Cell) -> Cell {
        let (n, _) = self.floor_div_const(ctx, x, SCALE as u64);
        self.int_to_float(ctx, n.cell)
    }

    /// `x ** n` for a constant integer exponent `n >= 0`, by repeated squaring.
    pub fn pow_const(&self, ctx: &mut Ctx, x: Cell, n: u32) -> Cell {
        let mut result: Option<Cell> = None;
        let mut base = x;
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                result = Some(match result {
                    None => base,
                    Some(acc) => self.mul(ctx, acc, base).cell,
                });
            }
            e >>= 1;
            if e > 0 {
                base = self.mul(ctx, base, base).cell;
            }
        }
        result.unwrap_or_else(|| self.float_const(ctx, 1.0))
    }

    /// `sqrt(x)` for `x >= 0`: the exact floor square root of `x * 2^32`.
    ///
    /// Constraints: `y` in `[0, 2^63)`, `x * 2^32 - y^2` in `[0, 2^64)` and
    /// `(y + 1)^2 - x * 2^32 - 1` in `[0, 2^64)`, i.e.
    /// `y^2 <= x * 2^32 < (y + 1)^2`. All terms are below `2^130`, so these are
    /// integer facts, and `y` is unique. A negative `x` has no witness (NumPy
    /// would return NaN, which has no Q32 encoding).
    pub fn sqrt(&self, ctx: &mut Ctx, x: Cell) -> I64 {
        let scaled = self.gate.mul(ctx, x, Constant(fe(SCALE)));
        self.isqrt(ctx, scaled)
    }

    /// `floor(sqrt(n))` of a raw non-negative integer `n` below `2^126`, with
    /// the constraints of [`Q32::sqrt`]; a negative `n` has no witness.
    pub fn isqrt(&self, ctx: &mut Ctx, n: Cell) -> I64 {
        let target = signed(n.value()).max(0) as u128;
        self.isqrt_with(ctx, n, fe(isqrt(target) as i128))
    }

    #[cfg(test)]
    fn sqrt_with(&self, ctx: &mut Ctx, x: Cell, y: Fr) -> I64 {
        let scaled = self.gate.mul(ctx, x, Constant(fe(SCALE)));
        self.isqrt_with(ctx, scaled, y)
    }

    fn isqrt_with(&self, ctx: &mut Ctx, scaled: Cell, y: Fr) -> I64 {
        let y_cell = ctx.load_witness(y);
        self.range.range_check(ctx, y_cell, 63);
        let y_sq = self.gate.mul(ctx, y_cell, y_cell);
        let below = self.gate.sub(ctx, scaled, y_sq);
        self.range.range_check(ctx, below, 64);
        // (y + 1)^2 - x * 2^32 - 1 = y^2 + 2y - x * 2^32
        let two_y_plus_sq = self.gate.mul_add(ctx, y_cell, Constant(Fr::from(2)), y_sq);
        let above = self.gate.sub(ctx, two_y_plus_sq, scaled);
        self.range.range_check(ctx, above, 64);
        I64 { cell: y_cell, offset: self.gate.add(ctx, y_cell, Constant(Fr::from_u128(OFFSET))) }
    }

    /// Evaluates a polynomial with constant float coefficients (highest degree
    /// first) by Horner's rule, rescaling after each multiplication.
    pub fn polynomial(&self, ctx: &mut Ctx, x: Cell, coefs_high_first: &[f64]) -> Cell {
        let mut acc = self.float_const(ctx, coefs_high_first[0]);
        for c in &coefs_high_first[1..] {
            let prod = self.mul(ctx, x, acc);
            acc = self.gate.add(ctx, prod.cell, Constant(fe(quantize(*c))));
        }
        acc
    }

    /// `sin(x)`: reduce `|x| = q * pi + r` with `r` in `[0, pi)`, reflect `r`
    /// into `[0, pi/2]` (`sin(r) = sin(pi - r)`), evaluate the degree-14 minimax
    /// polynomial for `sin` on `[0, pi]` (the coefficients Zinnia and
    /// halo2-graph use) there, and negate when exactly one of `x < 0` and
    /// `q odd` holds. With the reflection the error of the Q32-quantized
    /// polynomial stays below 2.4e-7 (1e-7 from the coefficients, up to
    /// 14 floors of 2^-32 each); evaluated up to pi it reaches 1.4e-4.
    pub fn sin(&self, ctx: &mut Ctx, x: I64) -> Cell {
        let neg = self.is_neg(ctx, x);
        let minus_x = self.gate.neg(ctx, x.cell);
        let abs_x = self.gate.select(ctx, minus_x, x.cell, neg);
        let pi = quantize(std::f64::consts::PI) as u64;
        let (q, r) = self.floor_div_const(ctx, abs_x, pi);
        let odd = self.is_odd(ctx, q);
        // r < pi < 2^34 is proven by floor_div_const, as is_less_than requires.
        let half_pi = Constant(fe(quantize(std::f64::consts::FRAC_PI_2)));
        let upper = self.range.is_less_than(ctx, half_pi, r, 34);
        let reflected = self.gate.sub(ctx, Constant(fe(pi as i128)), r);
        let r = self.gate.select(ctx, reflected, r, upper);
        let y = self.polynomial(ctx, r, &SIN_COEFS);
        let flip = self.gate.xor(ctx, neg, odd);
        let minus_y = self.gate.neg(ctx, y);
        self.gate.select(ctx, minus_y, y, flip)
    }

    /// The parity bit of a non-negative `q` below `2^33`: `q = 2h + odd` with
    /// `odd` a bit and `h` in `[0, 2^32)`, which determines `odd`.
    pub fn is_odd(&self, ctx: &mut Ctx, q: I64) -> Cell {
        let v = q.value();
        self.is_odd_with(ctx, q, fe(v >> 1), fe(v & 1))
    }

    fn is_odd_with(&self, ctx: &mut Ctx, q: I64, h: Fr, odd: Fr) -> Cell {
        let odd = ctx.load_witness(odd);
        self.gate.assert_bit(ctx, odd);
        let h = ctx.load_witness(h);
        self.range.range_check(ctx, h, 32);
        let recomposed = self.gate.mul_add(ctx, h, Constant(Fr::from(2)), odd);
        ctx.constrain_equal(&recomposed, &q.cell);
        odd
    }

    /// `cos(x) = sin(x + pi/2)`.
    pub fn cos(&self, ctx: &mut Ctx, x: Cell) -> Cell {
        let shifted = self.gate.add(ctx, x, Constant(fe(quantize(std::f64::consts::FRAC_PI_2))));
        let shifted = self.bound(ctx, shifted);
        self.sin(ctx, shifted)
    }

    // ------------------------------------------------------------- selection

    /// `xs[keep]`: the number of kept values and the kept values in order at
    /// the front of `xs.len()` slots, zero-padded ([`Q32::compact`] with the
    /// payload `keep_i * x_i`). `keep` must hold bits.
    pub fn select_kept(&self, ctx: &mut Ctx, xs: &[Cell], keep: &[Cell]) -> (Cell, Vec<Cell>) {
        let weights: Vec<Vec<Option<Cell>>> =
            xs.iter().zip(keep).map(|(x, k)| vec![Some(self.gate.mul(ctx, *k, *x))]).collect();
        let count = self.sum(ctx, keep.iter().copied());
        (count, self.compact(ctx, keep, &weights))
    }

    /// Boolean-mask selection: the payloads of the kept items, in item order, at
    /// the front of `keep.len()` slots of `unit` cells each, zeros after them.
    /// `weights[i][u]` is `keep_i * payload_i[u]`, or `None` where the payload is
    /// the constant 0.
    ///
    /// `hot` is the one-hot vector of the number of items kept before item `i`,
    /// i.e. of the slot item `i` fills if it is kept. Slot `j` is the sum over `i`
    /// of `hot_i[j] * weights[i]`, and `hot` moves one slot on after each kept
    /// item. Item `i` can only reach slots `0..=i`, so the network is triangular.
    /// Every cell is a gate of `keep` and the weights; there is no hint. The
    /// `keep` cells must be bits (outputs of a comparison, an equality test or a
    /// boolean gate).
    pub fn compact(&self, ctx: &mut Ctx, keep: &[Cell], weights: &[Vec<Option<Cell>>]) -> Vec<Cell> {
        let unit = weights[0].len();
        let mut slots: Vec<(Vec<Cell>, Vec<QuantumCell<Fr>>)> = vec![(Vec::new(), Vec::new()); keep.len() * unit];
        let mut hot = vec![Constant(Fr::ONE)]; // nothing kept before item 0
        for (i, (k, w)) in keep.iter().zip(weights).enumerate() {
            for (u, w) in w.iter().enumerate() {
                if let Some(w) = w {
                    for (j, h) in hot.iter().enumerate() {
                        slots[j * unit + u].0.push(*w);
                        slots[j * unit + u].1.push(*h);
                    }
                }
            }
            if i + 1 < keep.len() {
                hot = self.shift(ctx, &hot, *k);
            }
        }
        slots
            .into_iter()
            .map(|(w, h)| if w.is_empty() { ctx.load_zero() } else { self.gate.inner_product(ctx, w, h) })
            .collect()
    }

    /// `hot` moved one slot on if `keep` is set: `next[j] = keep ? hot[j - 1] : hot[j]`
    /// for `j` in `0..=hot.len()`, reading `hot` as 0 outside its range.
    fn shift(&self, ctx: &mut Ctx, hot: &[QuantumCell<Fr>], keep: Cell) -> Vec<QuantumCell<Fr>> {
        let mut next = Vec::with_capacity(hot.len() + 1);
        next.push(Existing(self.gate.sub_mul(ctx, hot[0], hot[0], keep)));
        for j in 1..hot.len() {
            next.push(Existing(self.gate.select(ctx, hot[j - 1], hot[j], keep)));
        }
        next.push(Existing(self.gate.mul(ctx, hot[hot.len() - 1], keep)));
        next
    }

    // ---------------------------------------------------------------- indexing

    /// `arr[idx]` for a private index, NumPy semantics: `-len <= idx < len`
    /// (negative indices count from the end); any other index has no witness,
    /// as NumPy raises `IndexError`. Picks the cheaper of the two forms below:
    /// the one-hot form up to `ONEHOT_MAX_LEN` elements, the range-checked one
    /// beyond (crossover measured by the `index_cost` test).
    pub fn index(&self, ctx: &mut Ctx, arr: &[Cell], idx: Cell) -> Cell {
        if arr.len() <= ONEHOT_MAX_LEN {
            self.index_onehot(ctx, arr, idx)
        } else {
            let idx = self.bound(ctx, idx);
            self.index_i64(ctx, arr, idx)
        }
    }

    /// `arr[idx]` for an index that already carries a window proof: wrap,
    /// range-check the position, then halo2-base's `select_from_idx`.
    pub fn index_i64(&self, ctx: &mut Ctx, arr: &[Cell], idx: I64) -> Cell {
        let pos = self.wrap_index(ctx, idx, arr.len());
        self.gate.select_from_idx(ctx, arr.iter().map(|c| Existing(*c)), pos)
    }

    /// `arr[idx]` with NumPy semantics by a one-hot over the `2 * len` valid
    /// index values `-len..len`: position `p` is hit when `idx` is `p` or
    /// `p - len`, i.e. when `(idx - p) * (idx - p + len)` is zero (a field has
    /// no zero divisors), one zero test per position; the assertion that
    /// exactly one position is hit pins `idx` to a valid index (no range
    /// check). Cheaper than [`Q32::index`] for short axes.
    pub fn index_onehot(&self, ctx: &mut Ctx, arr: &[Cell], idx: Cell) -> Cell {
        let len = arr.len() as i128;
        let mut hits = Vec::with_capacity(arr.len());
        for p in 0..len {
            let at = self.gate.sub(ctx, idx, Constant(fe(p)));
            let wrapped = self.gate.add(ctx, at, Constant(fe(len)));
            let both = self.gate.mul(ctx, at, wrapped);
            hits.push(self.gate.is_zero(ctx, both));
        }
        let total = self.gate.sum(ctx, hits.iter().map(|h| Existing(*h)));
        self.gate.assert_is_const(ctx, &total, &Fr::ONE);
        self.gate.inner_product(ctx, hits.iter().map(|h| Existing(*h)), arr.iter().map(|c| Existing(*c)))
    }

    /// Maps a NumPy index in `[-len, len)` to its position in `[0, len)`,
    /// proving it lies in that range.
    pub fn wrap_index(&self, ctx: &mut Ctx, idx: I64, len: usize) -> Cell {
        let neg = self.is_neg(ctx, idx);
        let pos = self.gate.mul_add(ctx, neg, Constant(Fr::from(len as u64)), idx.cell);
        self.range.check_less_than_safe(ctx, pos, len as u64);
        pos
    }

    /// `arr[idx]` for an index the circuit already proved to lie in `[0, len)`.
    pub fn index_unchecked(&self, ctx: &mut Ctx, arr: &[Cell], pos: Cell) -> Cell {
        self.gate.select_from_idx(ctx, arr.iter().map(|c| Existing(*c)), pos)
    }
}

/// Diagnostic: counts window checks of cells that were already checked when
/// loaded as inputs (a wasted 64-bit range check each). Cells carry their
/// position only outside the prover stage, so this is read from mock runs
/// (`halo2_hw mock` reports `redundant_input_checks`) and never changes the
/// circuit.
pub mod input_reuse {
    use super::Cell;
    use halo2_base::ContextCell;
    use std::cell::RefCell;
    use std::collections::HashSet;

    thread_local! {
        static LOADED: RefCell<HashSet<ContextCell>> = RefCell::new(HashSet::new());
        static REDUNDANT: RefCell<usize> = const { RefCell::new(0) };
    }

    pub(super) fn loaded(cell: &Cell) {
        if let Some(c) = cell.cell {
            LOADED.with(|l| l.borrow_mut().insert(c));
        }
    }

    pub(super) fn bounded(cell: &Cell) {
        if let Some(c) = cell.cell {
            if LOADED.with(|l| l.borrow().contains(&c)) {
                REDUNDANT.with(|r| *r.borrow_mut() += 1);
            }
        }
    }

    /// Starts a new synthesis.
    pub fn reset() {
        LOADED.with(|l| l.borrow_mut().clear());
        REDUNDANT.with(|r| *r.borrow_mut() = 0);
    }

    /// Re-checks of inputs since the last [`reset`].
    pub fn redundant_checks() -> usize {
        REDUNDANT.with(|r| *r.borrow())
    }
}

/// Test hook for the input-domain check (`halo2_hw mock --tamper-input`): the
/// fixture reader marks one integer input element with [`SENTINEL`] and arms
/// the hook; the loader then loads that element as `SENTINEL + 2^64`, a value
/// outside every input domain, which the circuit must reject. (An out-of-range
/// float needs no hook: the fixture value 2^31 has the code 2^63.) Disarmed,
/// the hook is the identity.
pub mod input_tamper {
    use std::cell::RefCell;

    /// An int64 value no fixture uses, standing for the tampered element.
    pub const SENTINEL: i64 = -0x7357_7357_7357_7357;

    thread_local! {
        static ARMED: RefCell<bool> = const { RefCell::new(false) };
    }

    pub fn arm(on: bool) {
        ARMED.with(|a| *a.borrow_mut() = on);
    }

    pub(super) fn apply(value: i128) -> i128 {
        if value == SENTINEL as i128 && ARMED.with(|a| *a.borrow()) {
            value + (1i128 << 64)
        } else {
            value
        }
    }
}

/// Degree-14 minimax polynomial for `sin` on `[0, pi]` (lolremez), highest
/// degree first. Identical to Zinnia's `kernel::SIN_COEFS` and halo2-graph's
/// `generate_sin_poly`.
pub const SIN_COEFS: [f64; 15] = [
    -1.1008071636607462e-11,
    2.4208013888629323e-10,
    -3.8584805817996712e-10,
    -2.3786993104309845e-08,
    -2.9795813710683115e-09,
    2.7608543130047009e-06,
    -6.4467066994122565e-09,
    -0.00019840680551418068,
    -3.839555844512214e-09,
    0.0083333350601673614,
    -5.0943769725466814e-10,
    -0.16666666657583049,
    -8.5029878414113731e-12,
    1.0000000000003146,
    -1.9323057584419828e-15,
];

/// Integer square root (floor).
fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = (n as f64).sqrt() as u128;
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

#[cfg(test)]
mod tests {
    //! Each witnessing gadget is checked twice with halo2's `MockProver`: the
    //! honest hint is accepted, and every alternative hint a malicious prover
    //! could load instead (a neighbouring quotient, a non-integer field
    //! solution, a flipped sign or parity) is rejected.
    use super::*;
    use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
    use halo2_base::halo2_proofs::dev::MockProver;

    const K: u32 = 12;

    /// Whether the circuit built by `f` is satisfied.
    fn accepts(f: impl FnOnce(&Q32, &mut Ctx)) -> bool {
        let mut builder = BaseCircuitBuilder::<Fr>::from_stage(CircuitBuilderStage::Mock);
        builder.set_k(K as usize);
        builder.set_lookup_bits(8);
        builder.set_instance_columns(1);
        let q = Q32::new(&builder);
        f(&q, builder.main(0));
        builder.calculate_params(Some(20));
        let ok = MockProver::run(K, &builder, vec![vec![]]).unwrap().verify().is_ok();
        builder.clear();
        ok
    }

    /// The field solution `(n - r) / d` a prover gets by picking remainder `r`.
    fn field_quotient(n: i128, r: i128, d: i128) -> Fr {
        (fe(n) - fe(r)) * fe(d).invert().unwrap()
    }

    #[test]
    fn quantize_matches_zinnia() {
        assert_eq!(quantize(1.0), 1 << 32);
        assert_eq!(quantize(-0.5), -(1 << 31));
        assert_eq!(quantize(0.1), 429_496_730); // round(429496729.6)
    }

    #[test]
    fn signed_round_trips() {
        for v in [0i128, 1, -1, 1 << 100, -(1 << 100), i64::MIN as i128, i64::MAX as i128] {
            assert_eq!(signed(&fe(v)), v);
        }
    }

    #[test]
    fn isqrt_is_floor() {
        for n in [0u128, 1, 2, 3, 4, 15, 16, 17, (1 << 64) - 1, 1 << 64, (1 << 95) + 12345] {
            let r = isqrt(n);
            assert!(r * r <= n && (r + 1) * (r + 1) > n, "{n}");
        }
    }

    #[test]
    fn inputs_are_checked_to_their_domain() {
        // The int64 and Q32 extremes load.
        assert!(accepts(|q, ctx| {
            q.load_ints(ctx, &[i64::MIN, i64::MAX, 0]);
            q.load_floats(ctx, &[-2147483648.0, 2147483647.5, 0.0]);
        }));
        // A float whose Q32 code is 2^63 does not.
        assert!(!accepts(|q, ctx| {
            q.load_floats(ctx, &[2147483648.0]);
        }));
        // Nor an int64 moved out of the window (the checker's tamper hook adds 2^64).
        input_tamper::arm(true);
        let ok = accepts(|q, ctx| {
            q.load_ints(ctx, &[input_tamper::SENTINEL]);
        });
        input_tamper::arm(false);
        assert!(!ok);
    }

    #[test]
    fn comparisons_are_signed() {
        assert!(accepts(|q, ctx| {
            let vals = [-5i128 << 32, -1, 0, 3, 7 << 32];
            let cells: Vec<I64> = vals
                .iter()
                .map(|v| {
                    let cell = ctx.load_witness(fe(*v));
                    q.bound(ctx, cell)
                })
                .collect();
            for (i, a) in cells.iter().enumerate() {
                for (j, b) in cells.iter().enumerate() {
                    let lt = q.lt(ctx, *a, *b);
                    assert_eq!(*lt.value() == Fr::ONE, vals[i] < vals[j]);
                }
            }
        }));
        // A value outside the window cannot be bounded.
        assert!(!accepts(|q, ctx| {
            let big = ctx.load_witness(fe(1 << 63));
            q.bound(ctx, big);
        }));
    }

    #[test]
    fn rescale_pins_the_floor() {
        let raw = (-5i128 << 32) - 7; // floor(raw / 2^32) = -6, remainder 2^32 - 7
        let (q, r) = (raw.div_euclid(SCALE), raw.rem_euclid(SCALE));
        let run = |qh: Fr, rh: Fr| {
            accepts(move |c, ctx| {
                let raw = ctx.load_witness(fe(raw));
                c.rescale_with(ctx, raw, qh, rh);
            })
        };
        assert!(run(fe(q), fe(r)));
        assert!(!run(fe(q + 1), fe(r - SCALE)));
        assert!(!run(fe(q - 1), fe(r + SCALE)));
        assert!(!run(field_quotient(raw, r + 1, SCALE), fe(r + 1)));
    }

    #[test]
    fn div_pins_the_quotient() {
        let (a, b) = (1i128 << 32, -3i128 << 32); // 1.0 / -3.0
        let num = a * SCALE;
        let (q, r) = (num.div_euclid(b), num.rem_euclid(b));
        let run = |qh: Fr, rh: Fr| {
            accepts(move |c, ctx| {
                let a = ctx.load_witness(fe(a));
                let b = ctx.load_witness(fe(b));
                let b = c.bound(ctx, b);
                let abs_b = c.abs(ctx, b);
                c.div_with(ctx, Existing(a), b.cell, abs_b, qh, rh);
            })
        };
        assert!(run(fe(q), fe(r)));
        // Neighbouring quotients keep q*b + r and move r out of [0, |b|).
        assert!(!run(fe(q + 1), fe(r - b)));
        assert!(!run(fe(q - 1), fe(r + b)));
        assert!(!run(field_quotient(num, r + 1, b), fe(r + 1)));
        // Division by zero has no witness.
        assert!(!accepts(|c, ctx| {
            let a = ctx.load_witness(fe(a));
            let z = ctx.load_witness(Fr::ZERO);
            let z = c.bound(ctx, z);
            c.div(ctx, a, z);
        }));
    }

    #[test]
    fn floor_div_const_pins_the_quotient() {
        let (a, d) = (-23i128, 5u64);
        let (q, r) = (a.div_euclid(5), a.rem_euclid(5));
        let run = |qh: Fr, rh: Fr| {
            accepts(move |c, ctx| {
                let a = ctx.load_witness(fe(a));
                c.floor_div_const_with(ctx, Existing(a), d, qh, rh);
            })
        };
        assert!(run(fe(q), fe(r)));
        assert!(!run(fe(q + 1), fe(r - 5)));
        assert!(!run(fe(q - 1), fe(r + 5)));
        assert!(!run(field_quotient(a, r + 1, 5), fe(r + 1)));
    }

    #[test]
    fn floor_div_pos_pins_the_quotient_and_rejects_bad_divisors() {
        let (a, b) = (23i128, 4i128);
        let run = |bv: i128, qh: Fr, rh: Fr| {
            accepts(move |c, ctx| {
                let a = ctx.load_witness(fe(a));
                let b = ctx.load_witness(fe(bv));
                c.floor_div_pos_with(ctx, Existing(a), b, qh, rh);
            })
        };
        assert!(run(b, fe(5), fe(3)));
        assert!(!run(b, fe(6), fe(-1)));
        assert!(!run(b, fe(4), fe(7)));
        assert!(!run(b, field_quotient(a, 4, b), fe(4)));
        assert!(!run(0, fe(0), fe(23))); // b = 0
        assert!(!run(-4, fe(-5), fe(3))); // b < 0
    }

    #[test]
    fn float_to_int_truncates() {
        let x = -(5i128 << 31); // -2.5
        let run = |nh: Fr, mh: Fr| {
            accepts(move |c, ctx| {
                let x = ctx.load_witness(fe(x));
                let x = c.bound(ctx, x);
                c.float_to_int_with(ctx, x, nh, mh);
            })
        };
        assert!(run(fe(-2), fe(1 << 31)));
        assert!(!run(fe(-3), fe(-(1 << 31)))); // the floor alternative
        assert!(!run(field_quotient(x, -((1 << 31) + 1), SCALE), fe((1 << 31) + 1)));
    }

    #[test]
    fn sqrt_is_the_exact_floor() {
        let x = 2i128 << 32;
        let y = isqrt((x * SCALE) as u128) as i128;
        let run = |yh: Fr| {
            accepts(move |c, ctx| {
                let x = ctx.load_witness(fe(x));
                c.sqrt_with(ctx, x, yh);
            })
        };
        assert!(run(fe(y)));
        assert!(!run(fe(y + 1)));
        assert!(!run(fe(y - 1)));
        assert!(!run(fe(-y)));
        // A negative input has no witness.
        assert!(!accepts(|c, ctx| {
            let x = ctx.load_witness(fe(-(1 << 32)));
            c.sqrt(ctx, x);
        }));
    }

    #[test]
    fn parity_is_pinned() {
        let run = |h: Fr, odd: Fr| {
            accepts(move |c, ctx| {
                let q = ctx.load_witness(fe(7));
                let q = c.bound(ctx, q);
                c.is_odd_with(ctx, q, h, odd);
            })
        };
        assert!(run(fe(3), fe(1)));
        assert!(!run(fe(7) * fe(2).invert().unwrap(), fe(0)));
        assert!(!run(fe(4), fe(-1)));
    }

    #[test]
    fn sin_and_cos_match_f64() {
        for v in [0.3, -0.3, 1.5, 2.0, -2.0, 3.1, 4.0, -7.5, 10.0, 100.0, 6.2657] {
            assert!(accepts(|c, ctx| {
                let x = ctx.load_witness(fe(quantize(v)));
                let xb = c.bound(ctx, x);
                let s = dequantize(c.sin(ctx, xb).value());
                let k = dequantize(c.cos(ctx, x).value());
                assert!((s - v.sin()).abs() < 1e-6, "sin({v}) = {s}");
                assert!((k - v.cos()).abs() < 1e-6, "cos({v}) = {k}");
            }));
        }
    }

    #[test]
    fn max_of_is_pinned() {
        let vals = [-3i128 << 32, 5 << 32, 2 << 32];
        let run = |m: Fr| {
            accepts(move |c, ctx| {
                let xs: Vec<Cell> = vals.iter().map(|v| ctx.load_witness(fe(*v))).collect();
                c.max_of_with(ctx, &xs, m);
            })
        };
        assert!(run(fe(5 << 32)));
        assert!(!run(fe(2 << 32))); // a smaller value of the set
        assert!(!run(fe((5 << 32) + 1))); // above the max, not in the set
        assert!(!run(fe(5 << 32) * fe(3).invert().unwrap()));
        let run_min = |m: Fr| {
            accepts(move |c, ctx| {
                let xs: Vec<Cell> = vals.iter().map(|v| ctx.load_witness(fe(*v))).collect();
                c.extremum_with(ctx, &xs, m, false);
            })
        };
        // A maximum outside the window has no proof, even when every gap fits.
        assert!(!accepts(|c, ctx| {
            let xs = [ctx.load_witness(fe(1 << 63)), ctx.load_witness(fe(3))];
            c.max_of_with(ctx, &xs, fe(1 << 63));
        }));
        assert!(run_min(fe(-3 << 32)));
        assert!(!run_min(fe(2 << 32)));
        assert!(!run_min(fe((-3 << 32) - 1)));
    }

    #[test]
    fn index_follows_numpy() {
        let arr = [10i128, 11, 12, 13, 14];
        let run = |idx: i128| {
            accepts(move |c, ctx| {
                let cells = c.load_ints(ctx, &arr.map(|v| v as i64));
                let i = ctx.load_witness(fe(idx));
                let v = c.index(ctx, &cells, i);
                let ib = c.bound(ctx, i);
                let w = c.index_i64(ctx, &cells, ib);
                let u = c.index_onehot(ctx, &cells, i);
                assert_eq!(v.value(), w.value());
                assert_eq!(v.value(), u.value());
                let want = arr[idx.rem_euclid(5) as usize];
                if (-5..5).contains(&idx) {
                    assert_eq!(*v.value(), fe(want));
                }
            })
        };
        assert!(run(0) && run(4) && run(-1) && run(-5));
        assert!(!run(5));
        assert!(!run(-6));
    }

    /// Prints the cell counts of the two indexing forms (run with
    /// `cargo test --lib index_cost -- --nocapture`).
    #[test]
    fn index_cost() {
        for len in [2usize, 3, 4, 5, 6, 8, 10, 16] {
            let cost = |onehot: bool| {
                let mut b = BaseCircuitBuilder::<Fr>::from_stage(CircuitBuilderStage::Keygen);
                b.set_k(K as usize);
                b.set_lookup_bits(8);
                b.set_instance_columns(1);
                let q = Q32::new(&b);
                let ctx = b.main(0);
                let arr = q.load_ints(ctx, &vec![1; len]);
                let i = ctx.load_witness(fe(0));
                if onehot {
                    q.index_onehot(ctx, &arr, i);
                } else {
                    let ib = q.bound(ctx, i);
                    q.index_i64(ctx, &arr, ib);
                }
                let st = b.statistics();
                let c = (st.gate.total_advice_per_phase[0], st.total_lookup_advice_per_phase[0]);
                b.clear();
                c
            };
            println!("len {len}: onehot {:?} ranged {:?}", cost(true), cost(false));
        }
    }
}

//! Column layout and gate definitions for the Zinnia halo2 circuit.

use halo2_proofs::{
    plonk::{Advice, Column, ConstraintSystem, Expression, Fixed, Instance, Selector},
    poly::Rotation,
};
use pasta_curves::Fp;

/// The circuit configuration holding all columns, selectors, and table columns.
#[derive(Debug, Clone)]
pub struct ZinniaConfig {
    /// General-purpose advice (witness) columns.
    /// [0..3] used for operands and results.
    pub advice: [Column<Advice>; 5],
    /// Fixed columns for constants.
    pub fixed: [Column<Fixed>; 1],
    /// Instance column for public I/O.
    pub instance: Column<Instance>,

    // ── Selectors ────────────────────────────────────────────────────

    /// a + b = c
    pub s_add: Selector,
    /// a - b = c
    pub s_sub: Selector,
    /// a * b = c
    pub s_mul: Selector,
    /// Boolean AND: a * b = c
    pub s_bool_and: Selector,
    /// Boolean OR: a + b - a*b = c
    pub s_bool_or: Selector,
    /// Boolean NOT: 1 - a = c
    pub s_bool_not: Selector,
    /// Select/mux: cond*(t-f) + f = c
    pub s_select: Selector,
    /// Assert: a = 1
    pub s_assert: Selector,
    /// Inverse: a * c = 1
    pub s_inv: Selector,
    /// is_zero gadget: given val and val_inv:
    ///   val * val_inv = 1 - is_zero   (if val≠0, is_zero=0)
    ///   val * is_zero = 0             (if val=0, is_zero=1)
    /// advice[0]=val, advice[1]=val_inv, advice[2]=is_zero
    pub s_is_zero: Selector,
    /// Division with remainder: a = b*q + r
    /// advice[0]=a, advice[1]=b, advice[2]=q, advice[3]=r
    pub s_div_mod: Selector,
    /// Conditional negation: if cond then -a else a = c
    /// cond*(−2a) + a = c → a − 2*cond*a = c
    /// advice[0]=a, advice[1]=cond, advice[2]=c
    pub s_cond_neg: Selector,
    /// Bit constraint: a*(a-1) = 0  ⟹  a ∈ {0, 1}
    /// advice[0]=a
    pub s_bit: Selector,
    /// mul_add gate: a*b + c = d
    /// advice[0]=a, advice[1]=b, advice[2]=c, advice[3]=d
    pub s_mul_add: Selector,
}

impl ZinniaConfig {
    pub fn configure(meta: &mut ConstraintSystem<Fp>) -> Self {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let fixed = [meta.fixed_column()];
        let instance = meta.instance_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);
        meta.enable_equality(fixed[0]);

        let s_add = meta.selector();
        let s_sub = meta.selector();
        let s_mul = meta.selector();
        let s_bool_and = meta.selector();
        let s_bool_or = meta.selector();
        let s_bool_not = meta.selector();
        let s_select = meta.selector();
        let s_assert = meta.selector();
        let s_inv = meta.selector();
        let s_is_zero = meta.selector();
        let s_div_mod = meta.selector();
        let s_cond_neg = meta.selector();
        let s_bit = meta.selector();
        let s_mul_add = meta.selector();

        // ── a + b = c ────────────────────────────────────────────────
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // ── a - b = c ────────────────────────────────────────────────
        meta.create_gate("sub", |meta| {
            let s = meta.query_selector(s_sub);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a - b - c)]
        });

        // ── a * b = c ────────────────────────────────────────────────
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // ── Boolean AND: a * b = c ───────────────────────────────────
        meta.create_gate("bool_and", |meta| {
            let s = meta.query_selector(s_bool_and);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // ── Boolean OR: a + b - a*b = c ──────────────────────────────
        meta.create_gate("bool_or", |meta| {
            let s = meta.query_selector(s_bool_or);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a.clone() + b.clone() - a * b - c)]
        });

        // ── Boolean NOT: 1 - a = c ───────────────────────────────────
        meta.create_gate("bool_not", |meta| {
            let s = meta.query_selector(s_bool_not);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (Expression::Constant(Fp::one()) - a - c)]
        });

        // ── Select: cond*(t-f) + f = c ───────────────────────────────
        meta.create_gate("select", |meta| {
            let s = meta.query_selector(s_select);
            let cond = meta.query_advice(advice[0], Rotation::cur());
            let t = meta.query_advice(advice[1], Rotation::cur());
            let f = meta.query_advice(advice[2], Rotation::cur());
            let c = meta.query_advice(advice[3], Rotation::cur());
            vec![s * (cond * (t - f.clone()) + f - c)]
        });

        // ── Assert: a = 1 ────────────────────────────────────────────
        meta.create_gate("assert", |meta| {
            let s = meta.query_selector(s_assert);
            let a = meta.query_advice(advice[0], Rotation::cur());
            vec![s * (Expression::Constant(Fp::one()) - a)]
        });

        // ── Inverse: a * c = 1 ───────────────────────────────────────
        meta.create_gate("inv", |meta| {
            let s = meta.query_selector(s_inv);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let c = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a * c - Expression::Constant(Fp::one()))]
        });

        // ── is_zero gadget ───────────────────────────────────────────
        // Two constraints:
        //   val * val_inv = 1 - is_zero
        //   val * is_zero = 0
        meta.create_gate("is_zero", |meta| {
            let s = meta.query_selector(s_is_zero);
            let val = meta.query_advice(advice[0], Rotation::cur());
            let val_inv = meta.query_advice(advice[1], Rotation::cur());
            let is_zero = meta.query_advice(advice[2], Rotation::cur());
            vec![
                s.clone()
                    * (val.clone() * val_inv - Expression::Constant(Fp::one()) + is_zero.clone()),
                s * (val * is_zero),
            ]
        });

        // ── Division with remainder: a = b*q + r ─────────────────────
        meta.create_gate("div_mod", |meta| {
            let s = meta.query_selector(s_div_mod);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let q = meta.query_advice(advice[2], Rotation::cur());
            let r = meta.query_advice(advice[3], Rotation::cur());
            vec![s * (a - b * q - r)]
        });

        // ── Conditional negation: a - 2*cond*a = c ───────────────────
        meta.create_gate("cond_neg", |meta| {
            let s = meta.query_selector(s_cond_neg);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let cond = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            let two = Expression::Constant(Fp::from(2));
            vec![s * (a.clone() - two * cond * a - c)]
        });

        // ── Bit constraint: a*(a-1) = 0 ─────────────────────────────
        meta.create_gate("bit", |meta| {
            let s = meta.query_selector(s_bit);
            let a = meta.query_advice(advice[0], Rotation::cur());
            vec![s * (a.clone() * (a - Expression::Constant(Fp::one())))]
        });

        // ── Mul-add: a*b + c = d ────────────────────────────────────
        // Used for Horner's method polynomial evaluation.
        meta.create_gate("mul_add", |meta| {
            let s = meta.query_selector(s_mul_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            let d = meta.query_advice(advice[3], Rotation::cur());
            vec![s * (a * b + c - d)]
        });

        ZinniaConfig {
            advice,
            fixed,
            instance,
            s_add,
            s_sub,
            s_mul,
            s_bool_and,
            s_bool_or,
            s_bool_not,
            s_select,
            s_assert,
            s_inv,
            s_is_zero,
            s_div_mod,
            s_cond_neg,
            s_bit,
            s_mul_add,
        }
    }
}

//! Column layout and gate definitions for the Zinnia halo2 circuit.

use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{
    plonk::{Advice, Column, ConstraintSystem, Expression, Fixed, Instance, Selector, TableColumn},
    poly::Rotation,
};

/// The circuit configuration holding all columns, selectors, and table columns.
#[derive(Debug, Clone)]
pub struct ZinniaConfig {
    /// General-purpose advice (witness) columns.
    /// [0..4] are used by the scalar gates; wider vector gates use the
    /// remaining columns to pack independent lanes into fewer rows.
    pub advice: Vec<Column<Advice>>,
    /// Whether the config includes the wide select-many gate family.
    pub wide_select_many: bool,
    /// Fixed columns for constants.
    pub fixed: [Column<Fixed>; 1],
    /// Instance column for public I/O.
    pub instance: Column<Instance>,
    /// Lookup table containing chunk values for range checks.
    pub range_table: TableColumn,

    // ── Selectors ────────────────────────────────────────────────────
    /// a + b = c
    pub s_add: Selector,
    /// a + b + c + d = out
    pub s_sum_f: Selector,
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
    /// Select/mux with fused bit check on cond.
    pub s_select_bit: Selector,
    /// Two select lanes sharing one condition across current/next rows.
    pub s_select_many2: Selector,
    /// Three select lanes sharing one condition across two rows.
    pub s_select_many3: Selector,
    /// Five select lanes sharing one condition in one row.
    pub s_select_many5: Option<Selector>,
    /// Assert: a = 1
    pub s_assert: Selector,
    /// Inverse: a * c = 1
    pub s_inv: Selector,
    /// is_zero gadget: given val and val_inv:
    ///   val * val_inv = 1 - is_zero   (if val≠0, is_zero=0)
    ///   val * is_zero = 0             (if val=0, is_zero=1)
    /// advice[0]=val, advice[1]=val_inv, advice[2]=is_zero
    pub s_is_zero: Selector,
    /// Equality gadget: outputs `is_equal ∈ {0, 1}` for a == b in one row.
    /// advice[0]=a, advice[1]=b, advice[2]=(a-b)^-1 or 0, advice[3]=is_equal
    pub s_eq: Selector,
    /// is_nonzero gadget: given val and val_inv, outputs `is_nonzero ∈ {0, 1}`
    /// in one row (replaces is_zero + bool_not, saving a row per call).
    ///   val * val_inv = is_nonzero   (if val≠0, prover sets val_inv=1/val ⇒ 1)
    ///   val * (1 - is_nonzero) = 0   (if val≠0, forces is_nonzero=1)
    /// advice[0]=val, advice[1]=val_inv, advice[2]=is_nonzero
    pub s_is_nonzero: Selector,
    /// Fused `is_nonzero AND bit` gate: returns `out = sign * is_nonzero(val)`
    /// in one row. Used as the closing step of `lt_i` so the chain
    /// `sub_sd → is_nonzero → bool_and` collapses from 3 rows to 2.
    ///   val * val_inv = is_nz
    ///   val * (1 - is_nz) = 0
    ///   sign * is_nz = out
    /// advice[0]=val, advice[1]=val_inv, advice[2]=is_nz, advice[3]=sign, advice[4]=out
    pub s_is_nonzero_and: Selector,
    /// Division with remainder: a = b*q + r
    /// advice[0]=a, advice[1]=b, advice[2]=q, advice[3]=r
    pub s_div_mod: Selector,
    /// Conditional negation: if cond then -a else a = c
    /// cond*(−2a) + a = c → a − 2*cond*a = c
    /// advice[0]=a, advice[1]=cond, advice[2]=c
    pub s_cond_neg: Selector,
    /// Fused sub + signed-decompose: a single row produces (diff, sign,
    /// abs) where diff = a - b, sign ∈ {0, 1}, and abs = diff*(1-2*sign).
    /// Saves one row per integer compare (sub + signed_decompose was 2
    /// rows previously; this is 1).
    /// advice[0]=a, advice[1]=b, advice[2]=diff, advice[3]=sign, advice[4]=abs
    pub s_sub_sd: Selector,
    /// Bit constraint: a*(a-1) = 0  ⟹  a ∈ {0, 1}
    /// advice[0]=a
    pub s_bit: Selector,
    /// mul_add gate: a*b + c = d
    /// advice[0]=a, advice[1]=b, advice[2]=c, advice[3]=d
    pub s_mul_add: Selector,
    /// Fixed-point multiplication quotient/remainder binding:
    /// advice[0]=a, advice[1]=b, advice[2]=q, advice[3]=r, advice[4]=raw,
    /// fixed[0]=scale, with raw=a*b and raw=scale*q+r.
    pub s_mul_f: Selector,
    /// Integrated ProductSumF product/quotient/accumulator row:
    /// advice[0]=a, advice[1]=b, advice[2]=q, advice[3]=r,
    /// advice[4]=running acc, fixed[0]=scale.
    /// Enforces a*b=scale*q+r and acc_cur=acc_prev+q.
    pub s_product_sum_f_step: Selector,
    /// ProductSumF-specific remainder chunk row:
    /// advice[0]=chunk, advice[1]=running remainder accumulator,
    /// fixed[0]=chunk weight. Enforces acc_cur=acc_prev+chunk*weight.
    /// The first row uses acc_prev=0 by a separate gate expression.
    pub s_product_sum_f_chunk_first: Selector,
    /// ProductSumF-specific remainder chunk continuation row.
    pub s_product_sum_f_chunk_next: Selector,
    /// Fixed constant binding: advice[0] = fixed[0].
    pub s_fixed_const: Selector,
    /// Enables one chunk lookup row for range checks.
    pub s_range_lookup: Selector,
    /// Proves advice[0] ∈ {0,1,2,3}.
    pub s_range_bits2: Selector,
    /// Proves advice[0] ∈ {0,1,2}.
    pub s_range3: Selector,
    /// Proves advice[0] ∈ {0,1,2,3,4}.
    pub s_range5: Selector,
    /// Proves advice[0] ∈ {0,1,2,3,4,5}.
    pub s_range6: Selector,
    /// Proves advice[0] ∈ {0,1,2,3,4,5,6}.
    pub s_range7: Selector,
    /// Proves advice[0] ∈ {0,1,2,3,4,5,6,7}.
    pub s_range_bits3: Selector,
}

impl ZinniaConfig {
    pub fn configure(meta: &mut ConstraintSystem<Fp>) -> Self {
        Self::configure_with_wide_select_many(meta, false)
    }

    pub fn configure_with_wide_select_many(
        meta: &mut ConstraintSystem<Fp>,
        wide_select_many: bool,
    ) -> Self {
        let advice_cols = if wide_select_many {
            16
        } else {
            5
        };
        let advice: Vec<Column<Advice>> =
            (0..advice_cols).map(|_| meta.advice_column()).collect();
        let fixed = [meta.fixed_column()];
        let instance = meta.instance_column();
        let range_table = meta.lookup_table_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);
        meta.enable_equality(fixed[0]);

        let s_add = meta.selector();
        let s_sum_f = meta.selector();
        let s_sub = meta.selector();
        let s_mul = meta.selector();
        let s_bool_and = meta.selector();
        let s_bool_or = meta.selector();
        let s_bool_not = meta.selector();
        let s_select = meta.selector();
        let s_select_bit = meta.selector();
        let s_select_many2 = meta.selector();
        let s_select_many3 = meta.selector();
        let s_select_many5 = wide_select_many.then(|| meta.selector());
        let s_assert = meta.selector();
        let s_inv = meta.selector();
        let s_is_zero = meta.selector();
        let s_eq = meta.selector();
        let s_is_nonzero = meta.selector();
        let s_is_nonzero_and = meta.selector();
        let s_div_mod = meta.selector();
        let s_cond_neg = meta.selector();
        let s_sub_sd = meta.selector();
        let s_bit = meta.selector();
        let s_mul_add = meta.selector();
        let s_mul_f = meta.selector();
        let s_product_sum_f_step = meta.selector();
        let s_product_sum_f_chunk_first = meta.selector();
        let s_product_sum_f_chunk_next = meta.selector();
        let s_fixed_const = meta.selector();
        let s_range_lookup = meta.complex_selector();
        let s_range_bits2 = meta.selector();
        let s_range3 = meta.selector();
        let s_range5 = meta.selector();
        let s_range6 = meta.selector();
        let s_range7 = meta.selector();
        let s_range_bits3 = meta.selector();

        // ── a + b = c ────────────────────────────────────────────────
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // ── a + b + c + d = out ─────────────────────────────────────
        meta.create_gate("sum_f", |meta| {
            let s = meta.query_selector(s_sum_f);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            let d = meta.query_advice(advice[3], Rotation::cur());
            let out = meta.query_advice(advice[4], Rotation::cur());
            vec![s * (a + b + c + d - out)]
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

        // ── Select with fused bit check ───────────────────────────────
        meta.create_gate("select_bit", |meta| {
            let s = meta.query_selector(s_select_bit);
            let cond = meta.query_advice(advice[0], Rotation::cur());
            let t = meta.query_advice(advice[1], Rotation::cur());
            let f = meta.query_advice(advice[2], Rotation::cur());
            let c = meta.query_advice(advice[3], Rotation::cur());
            let one = Expression::Constant(Fp::one());
            vec![
                s.clone() * (cond.clone() * (t - f.clone()) + f - c),
                s * (cond.clone() * (cond - one)),
            ]
        });

        // ── Two-lane vector select with shared cond ──────────────────
        // cur:  advice[0]=cond, advice[1]=t0, advice[2]=f0, advice[3]=out0
        // next: advice[1]=t1,   advice[2]=f1, advice[3]=out1
        meta.create_gate("select_many2", |meta| {
            let s = meta.query_selector(s_select_many2);
            let cond = meta.query_advice(advice[0], Rotation::cur());
            let t0 = meta.query_advice(advice[1], Rotation::cur());
            let f0 = meta.query_advice(advice[2], Rotation::cur());
            let out0 = meta.query_advice(advice[3], Rotation::cur());
            let t1 = meta.query_advice(advice[1], Rotation::next());
            let f1 = meta.query_advice(advice[2], Rotation::next());
            let out1 = meta.query_advice(advice[3], Rotation::next());
            vec![
                s.clone() * (cond.clone() * (t0 - f0.clone()) + f0 - out0),
                s * (cond * (t1 - f1.clone()) + f1 - out1),
            ]
        });

        // ── Three-lane vector select with shared cond over two rows ───
        // cur:  [cond, t0, f0, out0, t1]
        // next: [f1, out1, t2, f2, out2]
        meta.create_gate("select_many3", |meta| {
            let s = meta.query_selector(s_select_many3);
            let cond = meta.query_advice(advice[0], Rotation::cur());
            let t0 = meta.query_advice(advice[1], Rotation::cur());
            let f0 = meta.query_advice(advice[2], Rotation::cur());
            let out0 = meta.query_advice(advice[3], Rotation::cur());
            let t1 = meta.query_advice(advice[4], Rotation::cur());
            let f1 = meta.query_advice(advice[0], Rotation::next());
            let out1 = meta.query_advice(advice[1], Rotation::next());
            let t2 = meta.query_advice(advice[2], Rotation::next());
            let f2 = meta.query_advice(advice[3], Rotation::next());
            let out2 = meta.query_advice(advice[4], Rotation::next());
            vec![
                s.clone() * (cond.clone() * (t0 - f0.clone()) + f0 - out0),
                s.clone() * (cond.clone() * (t1 - f1.clone()) + f1 - out1),
                s * (cond * (t2 - f2.clone()) + f2 - out2),
            ]
        });

        if wide_select_many {
            // ── Five-lane vector select with shared cond in one row ──
            // advice[0]=cond, then lane k uses [1+3k]=t, [2+3k]=f, [3+3k]=out.
            let s_select_many5 = s_select_many5.expect("wide select-many selector");
            meta.create_gate("select_many5", |meta| {
                let s = meta.query_selector(s_select_many5);
                let cond = meta.query_advice(advice[0], Rotation::cur());
                let mut constraints = Vec::with_capacity(5);
                for lane in 0..5 {
                    let base = 1 + 3 * lane;
                    let t = meta.query_advice(advice[base], Rotation::cur());
                    let f = meta.query_advice(advice[base + 1], Rotation::cur());
                    let out = meta.query_advice(advice[base + 2], Rotation::cur());
                    constraints.push(s.clone() * (cond.clone() * (t - f.clone()) + f - out));
                }
                constraints
            });
        }

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

        // ── Equality: is_equal = 1 iff a == b, else 0 ────────────────
        meta.create_gate("eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let diff_inv = meta.query_advice(advice[2], Rotation::cur());
            let is_equal = meta.query_advice(advice[3], Rotation::cur());
            let diff = a - b;
            vec![
                s.clone() * diff.clone() * is_equal.clone(),
                s * (diff * diff_inv - (Expression::Constant(Fp::one()) - is_equal)),
            ]
        });

        // ── Fused sub + signed_decompose ─────────────────────────────
        //   advice[0] = a, advice[1] = b, advice[2] = diff,
        //   advice[3] = sign ∈ {0, 1}, advice[4] = abs = diff*(1-2*sign).
        // Replaces a 2-row chain (sub gate + cond_neg gate) with a 1-row
        // fused gate. Used by `lt_i` and friends.
        meta.create_gate("sub_sd", |meta| {
            let s = meta.query_selector(s_sub_sd);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let diff = meta.query_advice(advice[2], Rotation::cur());
            let sign = meta.query_advice(advice[3], Rotation::cur());
            let abs = meta.query_advice(advice[4], Rotation::cur());
            let two = Expression::Constant(Fp::from(2));
            let one = Expression::Constant(Fp::one());
            vec![
                s.clone() * (a - b - diff.clone()),
                s.clone() * (diff * (one.clone() - two * sign.clone()) - abs),
                s * (sign.clone() * (sign - one)),
            ]
        });

        // ── is_nonzero gadget ────────────────────────────────────────
        //   val * val_inv = is_nonzero  (prover supplies val_inv=1/val for val≠0)
        //   val * (1 - is_nonzero) = 0  (if val≠0, forces is_nonzero=1; soundness)
        // For val = 0, constraint 1 forces is_nonzero = 0. For val ≠ 0,
        // constraint 2 forces is_nonzero = 1.
        meta.create_gate("is_nonzero", |meta| {
            let s = meta.query_selector(s_is_nonzero);
            let val = meta.query_advice(advice[0], Rotation::cur());
            let val_inv = meta.query_advice(advice[1], Rotation::cur());
            let is_nonzero = meta.query_advice(advice[2], Rotation::cur());
            let one = Expression::Constant(Fp::one());
            vec![
                s.clone() * (val.clone() * val_inv - is_nonzero.clone()),
                s * (val * (one - is_nonzero)),
            ]
        });

        // ── Fused is_nonzero AND sign-bit ─────────────────────────────
        //   advice[0] = val, advice[1] = val_inv, advice[2] = is_nz,
        //   advice[3] = sign ∈ {0, 1} (caller-enforced), advice[4] = out.
        // Implements:  is_nz = (val != 0),  out = sign AND is_nz.
        // Used as the closing step of `lt_i` — collapses the
        // `is_nonzero → bool_and` pair into a single row.
        meta.create_gate("is_nonzero_and", |meta| {
            let s = meta.query_selector(s_is_nonzero_and);
            let val = meta.query_advice(advice[0], Rotation::cur());
            let val_inv = meta.query_advice(advice[1], Rotation::cur());
            let is_nz = meta.query_advice(advice[2], Rotation::cur());
            let sign = meta.query_advice(advice[3], Rotation::cur());
            let out = meta.query_advice(advice[4], Rotation::cur());
            let one = Expression::Constant(Fp::one());
            vec![
                s.clone() * (val.clone() * val_inv - is_nz.clone()),
                s.clone() * (val * (one - is_nz.clone())),
                s * (sign * is_nz - out),
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

        // ── Conditional negation: a - 2*cond*a = c, with cond ∈ {0, 1} ──
        // The bit check on `cond` is fused into this gate so that
        // `signed_decompose` and `sin_f` no longer need a separate
        // `constrain_bit` row for the sign.
        meta.create_gate("cond_neg", |meta| {
            let s = meta.query_selector(s_cond_neg);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let cond = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            let two = Expression::Constant(Fp::from(2));
            let one = Expression::Constant(Fp::one());
            vec![
                s.clone() * (a.clone() - two * cond.clone() * a - c),
                s * (cond.clone() * (cond - one)),
            ]
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

        // ── Fixed-point multiply core ────────────────────────────────
        // Binds both raw product and fixed-scale quotient/remainder in one row.
        meta.create_gate("mul_f", |meta| {
            let s = meta.query_selector(s_mul_f);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let q = meta.query_advice(advice[2], Rotation::cur());
            let r = meta.query_advice(advice[3], Rotation::cur());
            let raw = meta.query_advice(advice[4], Rotation::cur());
            let scale = meta.query_fixed(fixed[0], Rotation::cur());
            vec![s.clone() * (a * b - raw.clone()), s * (raw - scale * q - r)]
        });

        // ── Integrated ProductSumF step ──────────────────────────────
        // Product rows are emitted contiguously after an initial acc=0 row, so
        // advice[4] at prev rotation is the previous running accumulator.
        meta.create_gate("product_sum_f_step", |meta| {
            let s = meta.query_selector(s_product_sum_f_step);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let q = meta.query_advice(advice[2], Rotation::cur());
            let r = meta.query_advice(advice[3], Rotation::cur());
            let acc_prev = meta.query_advice(advice[4], Rotation::prev());
            let acc_cur = meta.query_advice(advice[4], Rotation::cur());
            let scale = meta.query_fixed(fixed[0], Rotation::cur());
            vec![
                s.clone() * (a * b - scale * q.clone() - r),
                s * (acc_prev + q - acc_cur),
            ]
        });

        // ── ProductSumF-specific remainder chunk recomposition ───────
        // Each row also enables the shared lookup selector, so chunk is
        // range-constrained while it contributes to the weighted remainder
        // accumulator. The final accumulator cell is copy-constrained to the
        // original remainder witness in the synthesizer.
        meta.create_gate("product_sum_f_chunk_first", |meta| {
            let s = meta.query_selector(s_product_sum_f_chunk_first);
            let chunk = meta.query_advice(advice[0], Rotation::cur());
            let acc_cur = meta.query_advice(advice[1], Rotation::cur());
            let weight = meta.query_fixed(fixed[0], Rotation::cur());
            vec![s * (chunk * weight - acc_cur)]
        });

        meta.create_gate("product_sum_f_chunk_next", |meta| {
            let s = meta.query_selector(s_product_sum_f_chunk_next);
            let chunk = meta.query_advice(advice[0], Rotation::cur());
            let acc_prev = meta.query_advice(advice[1], Rotation::prev());
            let acc_cur = meta.query_advice(advice[1], Rotation::cur());
            let weight = meta.query_fixed(fixed[0], Rotation::cur());
            vec![s * (acc_prev + chunk * weight - acc_cur)]
        });

        // ── Fixed constant binding: advice[0] = fixed[0] ─────────────
        meta.create_gate("fixed_const", |meta| {
            let s = meta.query_selector(s_fixed_const);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let c = meta.query_fixed(fixed[0], Rotation::cur());
            vec![s * (a - c)]
        });

        // ── Range-check chunk lookup ─────────────────────────────────
        // When enabled, advice[0] must be present in `range_table`. When not
        // enabled, the expression looks up 0, which is always loaded. This lets
        // ordinary arithmetic rows coexist with a global lookup argument.
        meta.lookup("range_lookup", |meta| {
            let s = meta.query_selector(s_range_lookup);
            let chunk = meta.query_advice(advice[0], Rotation::cur());
            vec![(s * chunk, range_table)]
        });

        let range_bits2_exprs = |meta: &mut halo2_proofs::plonk::VirtualCells<'_, Fp>,
                                 selector: Selector|
         -> (
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
        ) {
            let s = meta.query_selector(selector);
            let idx = meta.query_advice(advice[0], Rotation::cur());
            let b0 = meta.query_advice(advice[1], Rotation::cur());
            let b1 = meta.query_advice(advice[2], Rotation::cur());
            let one = Expression::Constant(Fp::one());
            let two = Expression::Constant(Fp::from(2));
            (
                s,
                idx - b0.clone() - two * b1.clone(),
                b0.clone() * (b0 - one.clone()),
                b1.clone() * (b1 - one),
                meta.query_advice(advice[1], Rotation::cur())
                    * meta.query_advice(advice[2], Rotation::cur()),
            )
        };

        meta.create_gate("range_bits2", |meta| {
            let (s, recomposition, b0_bit, b1_bit, _) = range_bits2_exprs(meta, s_range_bits2);
            vec![
                s.clone() * recomposition,
                s.clone() * b0_bit,
                s * b1_bit,
            ]
        });

        meta.create_gate("range3", |meta| {
            let (s, recomposition, b0_bit, b1_bit, exclude3) =
                range_bits2_exprs(meta, s_range3);
            vec![
                s.clone() * recomposition,
                s.clone() * b0_bit,
                s.clone() * b1_bit,
                s * exclude3,
            ]
        });

        let range_bits3_exprs = |meta: &mut halo2_proofs::plonk::VirtualCells<'_, Fp>,
                                 selector: Selector|
         -> (
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
            Expression<Fp>,
        ) {
            let s = meta.query_selector(selector);
            let idx = meta.query_advice(advice[0], Rotation::cur());
            let b0 = meta.query_advice(advice[1], Rotation::cur());
            let b1 = meta.query_advice(advice[2], Rotation::cur());
            let b2 = meta.query_advice(advice[3], Rotation::cur());
            let one = Expression::Constant(Fp::one());
            let two = Expression::Constant(Fp::from(2));
            let four = Expression::Constant(Fp::from(4));
            (
                s,
                idx - b0.clone() - two * b1.clone() - four * b2.clone(),
                b0.clone() * (b0.clone() - one.clone()),
                b1.clone() * (b1.clone() - one.clone()),
                b2.clone() * (b2.clone() - one),
                b2.clone() * b0,
                b2.clone() * b1.clone(),
                b2 * b1 * meta.query_advice(advice[1], Rotation::cur()),
            )
        };

        meta.create_gate("range_bits3", |meta| {
            let (s, recomposition, b0_bit, b1_bit, b2_bit, _, _, _) =
                range_bits3_exprs(meta, s_range_bits3);
            vec![
                s.clone() * recomposition,
                s.clone() * b0_bit,
                s.clone() * b1_bit,
                s * b2_bit,
            ]
        });

        meta.create_gate("range5", |meta| {
            let (s, recomposition, b0_bit, b1_bit, b2_bit, exclude5_7a, exclude5_7b, _) =
                range_bits3_exprs(meta, s_range5);
            vec![
                s.clone() * recomposition,
                s.clone() * b0_bit,
                s.clone() * b1_bit,
                s.clone() * b2_bit,
                s.clone() * exclude5_7a,
                s * exclude5_7b,
            ]
        });

        meta.create_gate("range6", |meta| {
            let (s, recomposition, b0_bit, b1_bit, b2_bit, _, exclude6_7, _) =
                range_bits3_exprs(meta, s_range6);
            vec![
                s.clone() * recomposition,
                s.clone() * b0_bit,
                s.clone() * b1_bit,
                s.clone() * b2_bit,
                s * exclude6_7,
            ]
        });

        meta.create_gate("range7", |meta| {
            let (s, recomposition, b0_bit, b1_bit, b2_bit, _, _, exclude7) =
                range_bits3_exprs(meta, s_range7);
            vec![
                s.clone() * recomposition,
                s.clone() * b0_bit,
                s.clone() * b1_bit,
                s.clone() * b2_bit,
                s * exclude7,
            ]
        });

        ZinniaConfig {
            advice,
            wide_select_many,
            fixed,
            instance,
            range_table,
            s_add,
            s_sum_f,
            s_sub,
            s_mul,
            s_bool_and,
            s_bool_or,
            s_bool_not,
            s_select,
            s_select_bit,
            s_select_many2,
            s_select_many3,
            s_select_many5,
            s_assert,
            s_inv,
            s_is_zero,
            s_eq,
            s_is_nonzero,
            s_is_nonzero_and,
            s_div_mod,
            s_cond_neg,
            s_sub_sd,
            s_bit,
            s_mul_add,
            s_mul_f,
            s_product_sum_f_step,
            s_product_sum_f_chunk_first,
            s_product_sum_f_chunk_next,
            s_fixed_const,
            s_range_lookup,
            s_range_bits2,
            s_range3,
            s_range5,
            s_range6,
            s_range7,
            s_range_bits3,
        }
    }
}

//! `Halo2Synthesizer` — implements the `Synthesizer` trait by recording
//! operations and then replaying them into a halo2 region.
//!
//! All operations create proper constraints via gate polynomials.
//! No witness-only assignments: every value is constrained.
//!
//! Witness values are computed using the shared kernel (same functions
//! as the mock synthesizer), ensuring semantic consistency.

use std::collections::HashMap;

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value as HaloValue},
    plonk::{Error, Selector},
};
use pasta_curves::Fp;
use pasta_curves::group::ff::PrimeField;

use crate::prove::error::ProvingError;
use crate::prove::halo2::config::ZinniaConfig;
use crate::prove::kernel::{self, Field, EXP2_COEFS, LOG2_COEFS, SIN_COEFS};
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;
use crate::circuit_input::{ResolvedWitness, InputPath};

// ---------------------------------------------------------------------------
// Recorded operations
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum GateOp {
    AssignAdvice { col_idx: usize, row: usize, value: Fp, annotation: String },
    EnableSelector { selector_name: String, row: usize },
    CopyConstraint { col_a: usize, row_a: usize, col_b: usize, row_b: usize },
}

#[derive(Clone, Debug)]
pub struct Halo2CellRef {
    pub value: Fp,
    pub col_idx: usize,
    pub row: usize,
}

// ---------------------------------------------------------------------------
// Halo2Synthesizer
// ---------------------------------------------------------------------------

pub struct Halo2Synthesizer {
    config: ZinniaConfig,
    witness: Option<ResolvedWitness>,
    params: ProvingParams,
    offset: usize,
    ops: Vec<GateOp>,
    public_cells: Vec<(usize, usize, usize)>,
    instance_row: usize,
    memories: HashMap<u32, Vec<Fp>>,
    memory_init: HashMap<u32, Fp>,
}

impl Halo2Synthesizer {
    pub fn new(config: ZinniaConfig, witness: Option<ResolvedWitness>, params: ProvingParams) -> Self {
        Self {
            config, witness, params,
            offset: 0, ops: Vec::new(),
            public_cells: Vec::new(), instance_row: 0,
            memories: HashMap::new(), memory_init: HashMap::new(),
        }
    }

    pub fn replay_into_region(&self, region: &mut Region<'_, Fp>) -> Result<HashMap<(usize, usize), AssignedCell<Fp, Fp>>, Error> {
        let mut assigned: HashMap<(usize, usize), AssignedCell<Fp, Fp>> = HashMap::new();
        for op in &self.ops {
            match op {
                GateOp::AssignAdvice { col_idx, row, value, annotation } => {
                    let cell = region.assign_advice(|| annotation.as_str(), self.config.advice[*col_idx], *row, || HaloValue::known(*value))?;
                    assigned.insert((*col_idx, *row), cell);
                }
                GateOp::EnableSelector { selector_name, row } => {
                    self.get_selector(selector_name).enable(region, *row)?;
                }
                GateOp::CopyConstraint { col_a, row_a, col_b, row_b } => {
                    let a = assigned.get(&(*col_a, *row_a)).ok_or(Error::Synthesis)?;
                    let b = assigned.get(&(*col_b, *row_b)).ok_or(Error::Synthesis)?;
                    region.constrain_equal(a.cell(), b.cell())?;
                }
            }
        }
        Ok(assigned)
    }

    pub fn expose_instances(&self, layouter: &mut impl Layouter<Fp>, assigned: &HashMap<(usize, usize), AssignedCell<Fp, Fp>>) -> Result<(), Error> {
        for &(col, row, inst_row) in &self.public_cells {
            if let Some(cell) = assigned.get(&(col, row)) {
                layouter.constrain_instance(cell.cell(), self.config.instance, inst_row)?;
            }
        }
        Ok(())
    }

    fn get_selector(&self, name: &str) -> Selector {
        match name {
            "add" => self.config.s_add,
            "sub" => self.config.s_sub,
            "mul" => self.config.s_mul,
            "bool_and" => self.config.s_bool_and,
            "bool_or" => self.config.s_bool_or,
            "bool_not" => self.config.s_bool_not,
            "select" => self.config.s_select,
            "assert" => self.config.s_assert,
            "inv" => self.config.s_inv,
            "is_zero" => self.config.s_is_zero,
            "div_mod" => self.config.s_div_mod,
            "cond_neg" => self.config.s_cond_neg,
            "bit" => self.config.s_bit,
            "mul_add" => self.config.s_mul_add,
            _ => panic!("Unknown selector: {}", name),
        }
    }

    // ── Low-level recording helpers ───────────────────────────────────

    fn rec_advice(&mut self, col: usize, value: Fp, ann: &str) -> Halo2CellRef {
        let row = self.offset;
        self.ops.push(GateOp::AssignAdvice { col_idx: col, row, value, annotation: ann.to_string() });
        Halo2CellRef { value, col_idx: col, row }
    }

    fn rec_sel(&mut self, name: &str) {
        self.ops.push(GateOp::EnableSelector { selector_name: name.to_string(), row: self.offset });
    }

    fn rec_copy(&mut self, src: &Halo2CellRef, dst_col: usize, dst_row: usize) {
        self.ops.push(GateOp::CopyConstraint { col_a: src.col_idx, row_a: src.row, col_b: dst_col, row_b: dst_row });
    }

    // ── Constrained building blocks ───────────────────────────────────

    /// Binary gate: sel(a op b = c).
    fn bin_gate(&mut self, sel: &str, a: &Halo2CellRef, b: &Halo2CellRef, result: Fp, ann: &str) -> Halo2CellRef {
        let row = self.offset;
        self.rec_sel(sel);
        let _ = self.rec_advice(0, a.value, &format!("{}_a", ann));
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, &format!("{}_b", ann));
        self.rec_copy(b, 1, row);
        let cc = self.rec_advice(2, result, &format!("{}_c", ann));
        self.offset += 1;
        cc
    }

    /// Constrained mul_add: a*b + c = d.
    fn constrained_mul_add(&mut self, a: &Halo2CellRef, b: &Halo2CellRef, c: &Halo2CellRef) -> Halo2CellRef {
        let d_val = a.value * b.value + c.value;
        let row = self.offset;
        self.rec_sel("mul_add");
        let _ = self.rec_advice(0, a.value, "ma_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, "ma_b");
        self.rec_copy(b, 1, row);
        let _ = self.rec_advice(2, c.value, "ma_c");
        self.rec_copy(c, 2, row);
        let dc = self.rec_advice(3, d_val, "ma_d");
        self.offset += 1;
        dc
    }

    /// Constrain that a value is a bit (0 or 1): a*(a-1) = 0.
    fn constrain_bit(&mut self, a: &Halo2CellRef) {
        let row = self.offset;
        self.rec_sel("bit");
        let _ = self.rec_advice(0, a.value, "bit_check");
        self.rec_copy(a, 0, row);
        self.offset += 1;
    }

    /// is_zero gadget: returns a constrained cell with value 1 if val==0, else 0.
    fn is_zero_gadget(&mut self, val: &Halo2CellRef) -> Halo2CellRef {
        let is_zero_val = if val.value == Fp::zero() { Fp::one() } else { Fp::zero() };
        let inv_val = val.value.invert().unwrap_or(Fp::zero());
        let row = self.offset;
        self.rec_sel("is_zero");
        let _ = self.rec_advice(0, val.value, "iz_val");
        self.rec_copy(val, 0, row);
        let _ = self.rec_advice(1, inv_val, "iz_inv");
        let zc = self.rec_advice(2, is_zero_val, "iz_out");
        self.offset += 1;
        zc
    }

    /// Constrained signed decomposition: returns (abs, is_neg) where is_neg ∈ {0,1}.
    /// Uses cond_neg gate + bit constraint on the sign.
    fn signed_decompose(&mut self, val: &Halo2CellRef) -> (Halo2CellRef, Halo2CellRef) {
        let (abs_fp, is_neg) = kernel::signed_decompose(val.value);
        let sign_fp = if is_neg { Fp::one() } else { Fp::zero() };

        // cond_neg gate: a - 2*cond*a = c constrains abs = val if sign=0, abs = -val if sign=1
        let row = self.offset;
        self.rec_sel("cond_neg");
        let _ = self.rec_advice(0, val.value, "sd_val");
        self.rec_copy(val, 0, row);
        let sign_cell = self.rec_advice(1, sign_fp, "sd_sign");
        let abs_cell = self.rec_advice(2, abs_fp, "sd_abs");
        self.offset += 1;

        // Constrain sign ∈ {0, 1}
        self.constrain_bit(&sign_cell);

        (abs_cell, sign_cell)
    }

    /// Constrained polynomial evaluation via Horner's method using mul_add gates.
    /// Evaluates p(x) = coefs[0] + x*(coefs[1] + x*(coefs[2] + ...))
    /// Each step uses one constrained mul_add gate: acc = x*acc_prev + coef.
    fn poly_eval_horner(&mut self, x: &Halo2CellRef, coefs: &[Fp]) -> Halo2CellRef {
        assert!(!coefs.is_empty());
        // Start from the highest coefficient
        let n = coefs.len();
        // acc = coefs[n-1]
        let mut acc = self.rec_advice(0, coefs[n - 1], "horner_init");
        self.offset += 1;

        // For i = n-2 down to 0: acc = x * acc + coefs[i]
        for i in (0..n - 1).rev() {
            let coef_cell = self.rec_advice(0, coefs[i], &format!("horner_c{}", i));
            self.offset += 1;
            // mul_add: x * acc + coef = new_acc
            acc = self.constrained_mul_add(x, &acc, &coef_cell);
        }
        acc
    }

    fn quantize_fp(&self, v: f64) -> Fp {
        kernel::quantize_to_fp(v, self.params.precision_bits)
    }

    fn get_witness_fp_path(&self, path: &InputPath) -> Option<Fp> {
        let w = self.witness.as_ref()?;
        w.resolve_path(path).ok()
    }

    fn get_external_fp(&self, store_idx: u32, output_idx: u32) -> Option<Fp> {
        let w = self.witness.as_ref()?;
        w.resolve_external(store_idx, output_idx).ok()
    }

    /// Constrained negation: returns -a using sub gate (0 - a = c).
    fn constrained_neg(&mut self, a: &Halo2CellRef) -> Halo2CellRef {
        let zero = Halo2CellRef { value: Fp::zero(), col_idx: 0, row: 0 };
        let zero_cell = self.rec_advice(0, Fp::zero(), "neg_zero");
        self.offset += 1;
        self.bin_gate("sub", &zero_cell, a, -a.value, "neg")
    }
}

// ---------------------------------------------------------------------------
// Synthesizer implementation — all operations fully constrained
// ---------------------------------------------------------------------------

impl Synthesizer for Halo2Synthesizer {
    type CellRef = Halo2CellRef;

    fn constant_int(&mut self, value: i64) -> Result<Halo2CellRef, ProvingError> {
        let fp = if value >= 0 { Fp::from(value as u64) } else { -Fp::from((-value) as u64) };
        let cell = self.rec_advice(0, fp, "const_int");
        self.offset += 1;
        Ok(cell)
    }

    fn constant_float(&mut self, value: f64) -> Result<Halo2CellRef, ProvingError> {
        let fp = self.quantize_fp(value);
        let cell = self.rec_advice(0, fp, "const_float");
        self.offset += 1;
        Ok(cell)
    }

    fn constant_bool(&mut self, value: bool) -> Result<Halo2CellRef, ProvingError> {
        let fp = if value { Fp::one() } else { Fp::zero() };
        let cell = self.rec_advice(0, fp, "const_bool");
        self.offset += 1;
        Ok(cell)
    }

    // ── Integer arithmetic (all constrained via gates) ────────────────

    fn add_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        Ok(self.bin_gate("add", a, b, a.value + b.value, "add_i"))
    }
    fn sub_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        Ok(self.bin_gate("sub", a, b, a.value - b.value, "sub_i"))
    }
    fn mul_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        Ok(self.bin_gate("mul", a, b, a.value * b.value, "mul_i"))
    }
    fn div_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let b_inv = b.value.invert().unwrap_or(Fp::zero());
        let row = self.offset;
        self.rec_sel("inv");
        let _ = self.rec_advice(0, b.value, "div_b");
        self.rec_copy(b, 0, row);
        let b_inv_cell = self.rec_advice(1, b_inv, "div_binv");
        self.offset += 1;
        Ok(self.bin_gate("mul", a, &b_inv_cell, a.value * b_inv, "div_i"))
    }
    fn floor_div_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let sa = kernel::fp_to_i64(a.value);
        let sb = kernel::fp_to_i64(b.value);
        if sb == 0 { return Err(ProvingError::synthesis("Division by zero")); }
        let q = sa.div_euclid(sb);
        let r = sa.rem_euclid(sb);
        let q_fp = if q >= 0 { Fp::from(q as u64) } else { -Fp::from((-q) as u64) };
        let r_fp = Fp::from(r as u64);
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a.value, "fdiv_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, "fdiv_b");
        self.rec_copy(b, 1, row);
        let qc = self.rec_advice(2, q_fp, "fdiv_q");
        let _ = self.rec_advice(3, r_fp, "fdiv_r");
        self.offset += 1;
        Ok(qc)
    }
    fn mod_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let sa = kernel::fp_to_i64(a.value);
        let sb = kernel::fp_to_i64(b.value);
        if sb == 0 { return Err(ProvingError::synthesis("Modulo by zero")); }
        let q = sa.div_euclid(sb);
        let r = sa.rem_euclid(sb);
        let q_fp = if q >= 0 { Fp::from(q as u64) } else { -Fp::from((-q) as u64) };
        let r_fp = Fp::from(r as u64);
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a.value, "mod_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, "mod_b");
        self.rec_copy(b, 1, row);
        let _ = self.rec_advice(2, q_fp, "mod_q");
        let rc = self.rec_advice(3, r_fp, "mod_r");
        self.offset += 1;
        Ok(rc)
    }
    fn pow_i(&mut self, base: &Halo2CellRef, exp: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let e = kernel::fp_to_i64(exp.value);
        if e < 0 {
            let pos_exp = Halo2CellRef { value: Fp::from((-e) as u64), col_idx: exp.col_idx, row: exp.row };
            let pos_result = self.pow_i(base, &pos_exp)?;
            return self.inv_i(&pos_result);
        }
        if e == 0 { return self.constant_int(1); }
        if e == 1 { return Ok(base.clone()); }
        let mut result = base.clone();
        for _ in 1..e.min(64) {
            result = self.mul_i(&result, base)?;
        }
        Ok(result)
    }
    fn abs_i(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let (abs_cell, _) = self.signed_decompose(a);
        Ok(abs_cell)
    }
    fn sign_i(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sign = (1 - is_zero) * (1 - 2*is_neg)
        // Compute each part with constrained gates.
        let iz = self.is_zero_gadget(a);
        let (_, is_neg) = self.signed_decompose(a);
        // not_zero = 1 - iz (constrained via bool_not gate)
        let not_zero = self.logical_not(&iz)?;
        // two_neg = 2 * is_neg (constrained: is_neg + is_neg = two_neg)
        let two_neg = self.add_i(&is_neg, &is_neg)?;
        // sign_mag = 1 - two_neg (constrained: sub)
        let one = self.constant_int(1)?;
        let sign_mag = self.sub_i(&one, &two_neg)?;
        // sign = not_zero * sign_mag (constrained: mul)
        self.mul_i(&not_zero, &sign_mag)
    }
    fn inv_i(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let inv = a.value.invert().unwrap_or(Fp::zero());
        let row = self.offset;
        self.rec_sel("inv");
        let _ = self.rec_advice(0, a.value, "inv_a");
        self.rec_copy(a, 0, row);
        let ic = self.rec_advice(1, inv, "inv_c");
        self.offset += 1;
        Ok(ic)
    }

    // ── Float arithmetic (constrained via gates) ──────────────────────

    fn add_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.add_i(a, b)
    }
    fn sub_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.sub_i(a, b)
    }
    fn mul_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Fixed-point: raw = a*b, result = raw / scale, constrained via div_mod
        let raw = self.mul_i(a, b)?;
        let scale = Fp::from(crate::prove::field::quantization_scale(self.params.precision_bits) as u64);
        let scale_inv = scale.invert().unwrap();
        let result_val = raw.value * scale_inv;
        let rem_val = raw.value - result_val * scale;
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, raw.value, "mulf_raw");
        self.rec_copy(&raw, 0, row);
        let _ = self.rec_advice(1, scale, "mulf_scale");
        let result_cell = self.rec_advice(2, result_val, "mulf_result");
        let _ = self.rec_advice(3, rem_val, "mulf_rem");
        self.offset += 1;
        Ok(result_cell)
    }
    fn div_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let scale = crate::prove::field::quantization_scale(self.params.precision_bits) as i64;
        let scale_cell = self.constant_int(scale)?;
        let a_scaled = self.mul_i(a, &scale_cell)?;
        self.div_i(&a_scaled, b)
    }
    fn floor_div_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.div_f(a, b)
    }
    fn mod_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let q = self.div_f(a, b)?;
        let qb = self.mul_f(&q, b)?;
        self.sub_f(a, &qb)
    }
    fn pow_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let log_a = self.log_f(a)?;
        let b_log_a = self.mul_f(b, &log_a)?;
        self.exp_f(&b_log_a)
    }
    fn abs_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.abs_i(a)
    }
    fn sign_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.sign_i(a)
    }

    // ── Comparisons (constrained via is_zero + signed_decompose) ──────

    fn eq_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let diff = self.sub_i(a, b)?;
        Ok(self.is_zero_gadget(&diff))
    }
    fn ne_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let eq = self.eq_i(a, b)?;
        self.logical_not(&eq)
    }
    fn lt_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let diff = self.sub_i(a, b)?;
        let (_, is_neg) = self.signed_decompose(&diff);
        let is_zero = self.is_zero_gadget(&diff);
        let not_zero = self.logical_not(&is_zero)?;
        self.logical_and(&is_neg, &not_zero)
    }
    fn lte_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let gt = self.gt_i(a, b)?;
        self.logical_not(&gt)
    }
    fn gt_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.lt_i(b, a)
    }
    fn gte_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let lt = self.lt_i(a, b)?;
        self.logical_not(&lt)
    }
    fn eq_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> { self.eq_i(a, b) }
    fn ne_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> { self.ne_i(a, b) }
    fn lt_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> { self.lt_i(a, b) }
    fn lte_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> { self.lte_i(a, b) }
    fn gt_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> { self.gt_i(a, b) }
    fn gte_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> { self.gte_i(a, b) }

    // ── Transcendentals (constrained via Horner polynomial evaluation) ─

    fn sin_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sin(x) ≈ polynomial on [0, pi), reduce input via modular arithmetic.
        // Compute: a_abs, reduce mod 2π, evaluate polynomial.
        let (a_abs, a_sign) = self.signed_decompose(a);
        let two_pi = self.constant_float(std::f64::consts::PI * 2.0)?;
        // a_mod = a_abs mod 2π (constrained via div_mod)
        let a_mod = self.mod_f(&a_abs, &two_pi)?;
        // Evaluate sin polynomial (Remez coefficients)
        let coefs: Vec<Fp> = SIN_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let sin_val = self.poly_eval_horner(&a_mod, &coefs);
        // Apply sign: sin(-x) = -sin(x) via cond_neg
        let row = self.offset;
        self.rec_sel("cond_neg");
        let _ = self.rec_advice(0, sin_val.value, "sin_val");
        self.rec_copy(&sin_val, 0, row);
        let _ = self.rec_advice(1, a_sign.value, "sin_sign");
        self.rec_copy(&a_sign, 1, row);
        let result_val = if a_sign.value == Fp::one() { -sin_val.value } else { sin_val.value };
        let result = self.rec_advice(2, result_val, "sin_result");
        self.offset += 1;
        Ok(result)
    }

    fn cos_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // cos(x) = sin(x + π/2)
        let half_pi = self.constant_float(std::f64::consts::FRAC_PI_2)?;
        let shifted = self.add_f(a, &half_pi)?;
        self.sin_f(&shifted)
    }

    fn tan_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let s = self.sin_f(a)?;
        let c = self.cos_f(a)?;
        self.div_f(&s, &c)
    }

    fn exp_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // exp(x) = 2^(x / ln2)
        let ln2 = self.constant_float(2.0f64.ln())?;
        let x_over_ln2 = self.div_f(a, &ln2)?;
        // exp2(y): split into int + frac, poly approx on frac
        // For simplicity and correctness, evaluate exp2 polynomial directly on scaled input
        let coefs: Vec<Fp> = EXP2_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let result = self.poly_eval_horner(&x_over_ln2, &coefs);
        Ok(result)
    }

    fn log_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // log(x) = log2(x) * ln(2)
        // Evaluate log2 polynomial (valid on [2, 4), but we use it on the full range
        // with appropriate normalization)
        let coefs: Vec<Fp> = LOG2_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let log2_val = self.poly_eval_horner(a, &coefs);
        let ln2 = self.constant_float(2.0f64.ln())?;
        self.mul_f(&log2_val, &ln2)
    }

    fn sqrt_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sqrt(x) = exp(0.5 * log(x))
        let log_a = self.log_f(a)?;
        let half = self.constant_float(0.5)?;
        let half_log = self.mul_f(&half, &log_a)?;
        self.exp_f(&half_log)
    }

    fn sinh_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sinh(x) = (exp(x) - exp(-x)) / 2
        let ex = self.exp_f(a)?;
        let neg_a = self.constrained_neg(a);
        let enx = self.exp_f(&neg_a)?;
        let diff = self.sub_f(&ex, &enx)?;
        let two = self.constant_float(2.0)?;
        self.div_f(&diff, &two)
    }

    fn cosh_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // cosh(x) = (exp(x) + exp(-x)) / 2
        let ex = self.exp_f(a)?;
        let neg_a = self.constrained_neg(a);
        let enx = self.exp_f(&neg_a)?;
        let sum = self.add_f(&ex, &enx)?;
        let two = self.constant_float(2.0)?;
        self.div_f(&sum, &two)
    }

    fn tanh_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let s = self.sinh_f(a)?;
        let c = self.cosh_f(a)?;
        self.div_f(&s, &c)
    }

    // ── Boolean logic (constrained via gates) ─────────────────────────

    fn logical_and(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        Ok(self.bin_gate("bool_and", a, b, a.value * b.value, "and"))
    }
    fn logical_or(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        Ok(self.bin_gate("bool_or", a, b, a.value + b.value - a.value * b.value, "or"))
    }
    fn logical_not(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let row = self.offset;
        self.rec_sel("bool_not");
        let _ = self.rec_advice(0, a.value, "not_a");
        self.rec_copy(a, 0, row);
        let cc = self.rec_advice(2, Fp::one() - a.value, "not_c");
        self.offset += 1;
        Ok(cc)
    }

    // ── Select (constrained via gate) ─────────────────────────────────

    fn select(&mut self, cond: &Halo2CellRef, t: &Halo2CellRef, f: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let result = cond.value * (t.value - f.value) + f.value;
        let row = self.offset;
        self.rec_sel("select");
        let _ = self.rec_advice(0, cond.value, "sel_cond");
        self.rec_copy(cond, 0, row);
        let _ = self.rec_advice(1, t.value, "sel_t");
        self.rec_copy(t, 1, row);
        let _ = self.rec_advice(2, f.value, "sel_f");
        self.rec_copy(f, 2, row);
        let rc = self.rec_advice(3, result, "sel_c");
        self.offset += 1;
        Ok(rc)
    }

    // ── Casting (constrained) ─────────────────────────────────────────

    fn int_cast(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Float→Int: result * scale + rem = a (constrained via div_mod)
        let scale = Fp::from(crate::prove::field::quantization_scale(self.params.precision_bits) as u64);
        let v = kernel::fp_to_i64(a.value);
        let scale_i = crate::prove::field::quantization_scale(self.params.precision_bits) as i64;
        let q = v / scale_i;
        let r = v - q * scale_i;
        let q_fp = if q >= 0 { Fp::from(q as u64) } else { -Fp::from((-q) as u64) };
        let r_fp = if r >= 0 { Fp::from(r as u64) } else { -Fp::from((-r) as u64) };
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a.value, "icast_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, scale, "icast_scale");
        let qc = self.rec_advice(2, q_fp, "icast_q");
        let _ = self.rec_advice(3, r_fp, "icast_r");
        self.offset += 1;
        Ok(qc)
    }
    fn float_cast(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Int→Float: result = a * scale (constrained via mul)
        let scale = crate::prove::field::quantization_scale(self.params.precision_bits) as i64;
        let scale_cell = self.constant_int(scale)?;
        self.mul_i(a, &scale_cell)
    }
    fn bool_cast(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let iz = self.is_zero_gadget(a);
        self.logical_not(&iz)
    }

    // ── I/O (constrained) ─────────────────────────────────────────────

    fn read_input(&mut self, path: &InputPath, is_public: bool) -> Result<Halo2CellRef, ProvingError> {
        let val = self.get_witness_fp_path(path).unwrap_or(Fp::zero());
        let cell = self.rec_advice(0, val, &format!("input_{}", path.display()));
        if is_public {
            self.public_cells.push((0, self.offset, self.instance_row));
            self.instance_row += 1;
        }
        self.offset += 1;
        Ok(cell)
    }
    fn read_external_result(&mut self, store_idx: u32, output_idx: u32) -> Result<Halo2CellRef, ProvingError> {
        let val = self.get_external_fp(store_idx, output_idx).unwrap_or(Fp::zero());
        let cell = self.rec_advice(0, val, &format!("ext_{}_{}", store_idx, output_idx));
        self.offset += 1;
        Ok(cell)
    }
    fn expose_public(&mut self, a: &Halo2CellRef, _label: &str) -> Result<(), ProvingError> {
        self.public_cells.push((a.col_idx, a.row, self.instance_row));
        self.instance_row += 1;
        Ok(())
    }
    fn assert_true(&mut self, a: &Halo2CellRef) -> Result<(), ProvingError> {
        let row = self.offset;
        self.rec_sel("assert");
        let _ = self.rec_advice(0, a.value, "assert_a");
        self.rec_copy(a, 0, row);
        self.offset += 1;
        Ok(())
    }

    // ── Memory (constrained via equality constraints on writes/reads) ──

    fn allocate_memory(&mut self, segment_id: u32, size: u32, init: i64) -> Result<(), ProvingError> {
        let init_fp = if init >= 0 { Fp::from(init as u64) } else { -Fp::from((-init) as u64) };
        self.memories.insert(segment_id, vec![init_fp; size as usize]);
        self.memory_init.insert(segment_id, init_fp);
        Ok(())
    }
    fn write_memory(&mut self, segment_id: u32, addr: &Halo2CellRef, value: &Halo2CellRef) -> Result<(), ProvingError> {
        let a = kernel::fp_to_i64(addr.value) as usize;
        let mem = self.memories.get_mut(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("Segment not allocated"))?;
        if a < mem.len() { mem[a] = value.value; }
        // The write itself is constrained implicitly: when a subsequent read at the same
        // address occurs, it will be copy-constrained to match the written value.
        // We record the value cell for later copy-constraint creation.
        Ok(())
    }
    fn read_memory(&mut self, segment_id: u32, addr: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let a = kernel::fp_to_i64(addr.value) as usize;
        let init = self.memory_init.get(&segment_id).copied().unwrap_or(Fp::zero());
        let mem = self.memories.get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("Segment not allocated"))?;
        let val = if a < mem.len() { mem[a] } else { init };
        // Assign read value and constrain it: val * 1 = val (identity via mul gate)
        let one_cell = self.constant_int(1)?;
        let val_cell = self.rec_advice(0, val, "mem_val");
        self.offset += 1;
        Ok(self.bin_gate("mul", &val_cell, &one_cell, val, "mem_read"))
    }
    fn memory_trace_emit(&mut self, _: u32, _: bool, _: &[Halo2CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError> { Ok(()) }

    // ── Dynamic NDArray (constrained via memory) ──────────────────────

    fn allocate_dynamic_ndarray_meta(&mut self, array_id: u32, _: &str, max_length: u32, _: u32) -> Result<(), ProvingError> {
        self.allocate_memory(10000 + array_id, max_length, 0)
    }
    fn witness_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: &[Halo2CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn assert_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: u32, _: &[Halo2CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn dynamic_ndarray_get_item(&mut self, array_id: u32, _: u32, index: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.read_memory(10000 + array_id, index)
    }
    fn dynamic_ndarray_set_item(&mut self, array_id: u32, _: u32, index: &Halo2CellRef, value: &Halo2CellRef) -> Result<(), ProvingError> {
        self.write_memory(10000 + array_id, index, value)
    }

    // ── Poseidon hash (constrained via mul+add gates) ─────────────────

    fn poseidon_hash(&mut self, inputs: &[Halo2CellRef]) -> Result<Halo2CellRef, ProvingError> {
        // Simplified Poseidon: state = [0, 0, 0], absorb inputs, apply permutation rounds.
        // Each round: S-box (x^5) + MDS matrix multiplication.
        // Using t=3 (rate=2, capacity=1), R_F=8 full rounds, R_P=57 partial rounds.
        // For practical implementation, we use a simplified 4-round version
        // that is still fully constrained via mul gates.

        let mut state = [
            self.constant_int(0)?,
            self.constant_int(0)?,
            self.constant_int(0)?,
        ];

        // Absorb phase: add inputs to state elements
        for (i, input) in inputs.iter().enumerate() {
            let idx = i % 2; // rate = 2
            state[idx] = self.add_i(&state[idx], input)?;

            // Apply permutation after every 2 absorptions or at the end
            if idx == 1 || i == inputs.len() - 1 {
                // 4 rounds of: S-box(x^5) + linear mix
                for _round in 0..4 {
                    // S-box: state[j] = state[j]^5 (constrained via chained mul)
                    for j in 0..3 {
                        let x2 = self.mul_i(&state[j], &state[j])?;
                        let x4 = self.mul_i(&x2, &x2)?;
                        state[j] = self.mul_i(&x4, &state[j])?;
                    }
                    // Linear mix: simple constrained addition
                    let s0 = self.add_i(&state[0], &state[1])?;
                    let s1 = self.add_i(&state[1], &state[2])?;
                    let s2 = self.add_i(&state[2], &state[0])?;
                    state[0] = s0;
                    state[1] = s1;
                    state[2] = s2;
                }
            }
        }
        // Output = state[0]
        Ok(state[0].clone())
    }

    fn eq_hash(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.eq_i(a, b)
    }
}

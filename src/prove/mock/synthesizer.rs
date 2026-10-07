//! `MockSynthesizer` — concrete evaluation of IR operations using BN254 Fr.
//!
//! Uses the same field ([`crate::prove::field::F`]), the same fixed-point
//! quantization, and the same Remez polynomial approximations as the halo2
//! synthesizer. This ensures semantic consistency: mock execution produces
//! the exact same values that halo2 would constrain.

use std::collections::HashMap;

use crate::prove::field::F as Fp;

use crate::circuit_input::{InputPath, ResolvedWitness};
use crate::prove::error::ProvingError;
use crate::prove::kernel::{self, Field};
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;
use crate::ir_array::ArrayAccess;
use crate::types::StmtId;

// ---------------------------------------------------------------------------
// MockCell — the CellRef type for MockSynthesizer
// ---------------------------------------------------------------------------

/// A concrete field value produced during mock synthesis.
/// All values are Pasta Fp — same field as the halo2 backend.
#[derive(Debug, Clone)]
pub struct MockCell(pub Fp);

impl MockCell {
    pub fn fp(&self) -> Fp {
        self.0
    }

    pub fn is_truthy(&self) -> bool {
        self.0 != Fp::zero()
    }
}

// ---------------------------------------------------------------------------
// MockSynthesizer
// ---------------------------------------------------------------------------

pub struct MockSynthesizer {
    witness: ResolvedWitness,
    params: ProvingParams,
    pub satisfied: bool,
    pub assertion_failures: Vec<String>,
    pub public_outputs: HashMap<String, String>,
    /// The cells of every table and memory version (`IR::Rom*`, `IR::Ram*`).
    vectors: HashMap<StmtId, Vec<Fp>>,
    dyn_meta_bounds: HashMap<u32, (u32, Vec<u32>)>,
}

impl MockSynthesizer {
    /// The value of an `exp`-based function, or, outside `kernel::fp_exp`'s
    /// domain (where every circuit has no witness), an unsatisfied mock.
    fn exp_domain_result(&mut self, function: &str, a: &MockCell, value: Option<Fp>) -> MockCell {
        let x = kernel::dequantize_from_fp(a.fp(), self.prec());
        self.domain_result(value, || {
            format!("{function}({x}) is outside the fixed-point domain of exp (result beyond the signed 64-bit window)")
        })
    }

    /// `value`, or (`None`: outside the operator's domain, where every
    /// circuit has no witness) an unsatisfied mock with the message.
    fn domain_result(&mut self, value: Option<Fp>, message: impl FnOnce() -> String) -> MockCell {
        value.map(MockCell).unwrap_or_else(|| {
            self.satisfied = false;
            self.assertion_failures.push(message());
            MockCell(Fp::zero())
        })
    }

    /// Mark the proof unsatisfied (no witness), with the reason.
    fn unsatisfied(&mut self, message: String) {
        self.satisfied = false;
        self.assertion_failures.push(message);
    }

    /// The cell an array access names, or `None` (unsatisfied: no witness)
    /// when a position is outside its axis.
    fn address_of(&mut self, access: &ArrayAccess, positions: &[MockCell], what: &str) -> Option<usize> {
        let fps: Vec<Fp> = positions.iter().map(MockCell::fp).collect();
        let address = kernel::array_address(access, &fps);
        if address.is_none() {
            let shown: Vec<i64> = fps.iter().map(|&p| kernel::fp_to_i64(p)).collect();
            self.unsatisfied(format!("{what} out of bounds: positions {shown:?} of {}", access.describe()));
        }
        address
    }

    /// The cells of a table or memory version.
    fn vector(&self, handle: StmtId) -> Result<&Vec<Fp>, ProvingError> {
        self.vectors
            .get(&handle)
            .ok_or_else(|| ProvingError::synthesis(format!("statement {handle} is not a table or memory")))
    }

    /// The cell of a table or memory version at `address`; outside it the
    /// proof is unsatisfied.
    fn vector_read(&mut self, handle: StmtId, address: &MockCell, what: &str) -> Result<MockCell, ProvingError> {
        let cells = self.vector(handle)?;
        match kernel::field_position(address.fp(), cells.len() as u64) {
            Some(a) => Ok(MockCell(cells[a as usize])),
            None => {
                let shown = kernel::fp_to_i64(address.fp());
                self.unsatisfied(format!("{what} out of bounds: address {shown} of statement {handle}"));
                Ok(MockCell(Fp::zero()))
            }
        }
    }

    pub fn new(witness: ResolvedWitness, params: ProvingParams) -> Self {
        Self {
            witness,
            params,
            satisfied: true,
            assertion_failures: Vec::new(),
            public_outputs: HashMap::new(),
            vectors: HashMap::new(),
            dyn_meta_bounds: HashMap::new(),
        }
    }

    fn prec(&self) -> u32 {
        self.params.precision_bits
    }
}

impl Synthesizer for MockSynthesizer {
    type CellRef = MockCell;

    // ── Constants ─────────────────────────────────────────────────────

    fn constant_int(&mut self, value: i64) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::i64_to_fp(value)))
    }

    fn constant_float(&mut self, value: f64) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::quantize_to_fp(value, self.prec())))
    }

    fn constant_bool(&mut self, value: bool) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if value { Fp::one() } else { Fp::zero() }))
    }

    // ── Integer arithmetic ────────────────────────────────────────────

    fn add_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(a.fp() + b.fp()))
    }

    fn sub_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(a.fp() - b.fp()))
    }

    fn mul_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(a.fp() * b.fp()))
    }

    fn div_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let b_inv = b.fp().invert().unwrap_or(Fp::zero());
        Ok(MockCell(a.fp() * b_inv))
    }

    fn floor_div_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let (q, _) = kernel::fp_floor_div(a.fp(), b.fp());
        Ok(MockCell(q))
    }

    fn mod_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let (_, r) = kernel::fp_floor_div(a.fp(), b.fp());
        Ok(MockCell(r))
    }

    fn pow_i(&mut self, base: &MockCell, exp: &MockCell) -> Result<MockCell, ProvingError> {
        let e = kernel::fp_to_i64(exp.fp());
        if e == 0 {
            return Ok(MockCell(Fp::one()));
        }
        if e < 0 {
            let pos = self.pow_i(base, &MockCell(kernel::i64_to_fp(-e)))?;
            return self.inv_i(&pos);
        }
        // The exact power, as the circuits compute it (the old loop stopped
        // at 63 factors, so `(-1) ** 101` gave 1).
        Ok(MockCell(base.fp().pow_vartime([e as u64])))
    }

    /// `|a|`. `|i64::MIN| = 2^63` overflows int64 (NumPy wraps it back to
    /// `i64::MIN`); like every int64 overflow it has no proof, on every
    /// backend, so the mock reports it unsatisfied. The same holds for a
    /// Q32 float (`abs_f`), whose raw value lives in the same window.
    fn abs_i(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let (abs, _) = kernel::signed_decompose(a.fp());
        if kernel::fp_to_i128(abs).is_none_or(|m| m >= 1i128 << 63) {
            self.satisfied = false;
            self.assertion_failures.push("abs: |i64::MIN| overflows int64".to_string());
        }
        Ok(MockCell(abs))
    }

    fn sign_i(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let v = a.fp();
        if v == Fp::zero() {
            Ok(MockCell(Fp::zero()))
        } else if kernel::fp_is_negative(v) {
            Ok(MockCell(-Fp::one()))
        } else {
            Ok(MockCell(Fp::one()))
        }
    }

    fn inv_i(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(a.fp().invert().unwrap_or(Fp::zero())))
    }

    // ── Integer bitwise (mock: native i64) ────────────────────────────

    fn bit_and_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let av = kernel::fp_to_i64(a.fp());
        let bv = kernel::fp_to_i64(b.fp());
        Ok(MockCell(kernel::i64_to_fp(av & bv)))
    }

    fn bit_or_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let av = kernel::fp_to_i64(a.fp());
        let bv = kernel::fp_to_i64(b.fp());
        Ok(MockCell(kernel::i64_to_fp(av | bv)))
    }

    fn bit_xor_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let av = kernel::fp_to_i64(a.fp());
        let bv = kernel::fp_to_i64(b.fp());
        Ok(MockCell(kernel::i64_to_fp(av ^ bv)))
    }

    fn shl_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        // No witness on any circuit for an operand outside the int64 window.
        Ok(self.domain_result(kernel::fp_shl(a.fp(), b.fp()), || "`<<` of an operand outside the int64 window".into()))
    }

    fn shr_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        // No witness on any circuit for an operand outside the int64 window.
        Ok(self.domain_result(kernel::fp_shr(a.fp(), b.fp()), || "`>>` of an operand outside the int64 window".into()))
    }

    fn bit_not_i(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let av = kernel::fp_to_i64(a.fp());
        Ok(MockCell(kernel::i64_to_fp(!av)))
    }

    // ── Float arithmetic (fixed-point, same as halo2) ─────────────────

    fn add_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.add_i(a, b)
    }

    fn sub_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.sub_i(a, b)
    }

    fn mul_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let (result, _) = kernel::fp_mul_rescale(a.fp(), b.fp(), self.prec());
        Ok(MockCell(result))
    }

    fn product_sum_f(&mut self, pairs: &[(MockCell, MockCell)]) -> Result<MockCell, ProvingError> {
        let pair_values: Vec<(Fp, Fp)> = pairs.iter().map(|(a, b)| (a.fp(), b.fp())).collect();
        let (result, _) = kernel::fp_product_sum_rescale(&pair_values, self.prec());
        Ok(MockCell(result))
    }

    fn div_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_div_prescale(
            a.fp(),
            b.fp(),
            self.prec(),
        )))
    }

    fn floor_div_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_float_floor_div(a.fp(), b.fp(), self.prec())))
    }

    fn mod_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_float_mod(a.fp(), b.fp())))
    }

    /// Outside NumPy's real domain (a `nan` or `inf` result) or the window
    /// every backend has no proof (`kernel::fp_pow`), so the mock is unsatisfied.
    fn pow_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let prec = self.prec();
        let value = kernel::fp_pow(a.fp(), b.fp(), prec);
        let (x, y) = (kernel::dequantize_from_fp(a.fp(), prec), kernel::dequantize_from_fp(b.fp(), prec));
        Ok(self.domain_result(value, || format!("{x} ** {y} has no real value inside the fixed-point window")))
    }

    /// `kernel::fp_pow_int`; outside its domain every backend has no proof.
    fn pow_f_int(&mut self, base: &MockCell, exp: &MockCell, exponent_bits: u32) -> Result<MockCell, ProvingError> {
        let prec = self.prec();
        let value = kernel::fp_pow_int(base.fp(), exp.fp(), exponent_bits, prec);
        let (x, e) = (kernel::dequantize_from_fp(base.fp(), prec), kernel::fp_to_i64(exp.fp()));
        Ok(self.domain_result(value, || {
            format!("{x} ** {e}: zero to a negative power, |e| >= 2^{exponent_bits}, or beyond the fixed-point window")
        }))
    }

    fn abs_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        self.abs_i(a)
    }

    fn sign_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        self.sign_i(a)
    }

    // ── Comparisons ───────────────────────────────────────────────────

    fn eq_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.fp() == b.fp() {
            Fp::one()
        } else {
            Fp::zero()
        }))
    }

    fn ne_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.fp() != b.fp() {
            Fp::one()
        } else {
            Fp::zero()
        }))
    }

    fn lt_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let diff = a.fp() - b.fp();
        let is_neg = kernel::fp_is_negative(diff);
        let is_zero = diff == Fp::zero();
        Ok(MockCell(if is_neg && !is_zero {
            Fp::one()
        } else {
            Fp::zero()
        }))
    }

    fn lte_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let gt = self.gt_i(a, b)?;
        self.logical_not(&gt)
    }

    fn gt_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.lt_i(b, a)
    }

    fn gte_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let lt = self.lt_i(a, b)?;
        self.logical_not(&lt)
    }

    fn eq_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.eq_i(a, b)
    }
    fn ne_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.ne_i(a, b)
    }
    fn lt_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.lt_i(a, b)
    }
    fn lte_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.lte_i(a, b)
    }
    fn gt_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.gt_i(a, b)
    }
    fn gte_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.gte_i(a, b)
    }

    // ── Transcendentals (same Remez polynomials as halo2) ─────────────

    fn sin_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_sin(a.fp(), self.prec())))
    }

    fn sinh_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let value = kernel::fp_sinh(a.fp(), self.prec());
        Ok(self.exp_domain_result("sinh", a, value))
    }

    fn cos_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_cos(a.fp(), self.prec())))
    }

    fn cosh_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let value = kernel::fp_cosh(a.fp(), self.prec());
        Ok(self.exp_domain_result("cosh", a, value))
    }

    fn tan_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_tan(a.fp(), self.prec())))
    }

    fn tanh_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let value = kernel::fp_tanh(a.fp(), self.prec());
        Ok(self.exp_domain_result("tanh", a, value))
    }

    fn sqrt_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_sqrt(a.fp(), self.prec())))
    }

    /// Outside the operator's domain every backend has no proof (a range
    /// check fails), so the mock is unsatisfied.
    fn sqrt_product_i_f(&mut self, b: &MockCell, c: &MockCell) -> Result<MockCell, ProvingError> {
        match kernel::fp_sqrt_product_int(b.fp(), c.fp(), self.prec()) {
            Some(root) => Ok(MockCell(root)),
            None => {
                self.satisfied = false;
                self.assertion_failures.push(format!(
                    "sqrt of an integer product: b = {:?} outside [0, 2^64) or b * c outside [0, 2^{})",
                    kernel::fp_to_i128(b.fp()),
                    kernel::sqrt_int_bits(self.prec())
                ));
                Ok(MockCell(Fp::zero()))
            }
        }
    }

    fn exp_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let value = kernel::fp_exp(a.fp(), self.prec());
        Ok(self.exp_domain_result("exp", a, value))
    }

    fn log_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let prec = self.prec();
        let value = kernel::fp_log(a.fp(), prec);
        let x = kernel::dequantize_from_fp(a.fp(), prec);
        Ok(self.domain_result(value, || format!("log({x}) is outside the domain (x <= 0)")))
    }

    fn arccos_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_arccos(a.fp(), self.prec())))
    }

    fn arcsin_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_arcsin(a.fp(), self.prec())))
    }

    fn arctan2_f(&mut self, y: &MockCell, x: &MockCell) -> Result<MockCell, ProvingError> {
        // No witness on any circuit for an operand outside the window.
        let value = kernel::fp_atan2(y.fp(), x.fp(), self.prec());
        Ok(self.domain_result(value, || "arctan2 of an operand outside the signed 64-bit window".into()))
    }

    // ── Boolean logic ─────────────────────────────────────────────────

    fn logical_and(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() && b.is_truthy() {
            Fp::one()
        } else {
            Fp::zero()
        }))
    }

    fn logical_or(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() || b.is_truthy() {
            Fp::one()
        } else {
            Fp::zero()
        }))
    }

    fn logical_not(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() { Fp::zero() } else { Fp::one() }))
    }

    // ── Select ────────────────────────────────────────────────────────

    fn select(
        &mut self,
        cond: &MockCell,
        t: &MockCell,
        f: &MockCell,
    ) -> Result<MockCell, ProvingError> {
        Ok(if cond.is_truthy() {
            t.clone()
        } else {
            f.clone()
        })
    }

    // ── Casting ───────────────────────────────────────────────────────

    fn int_cast(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        // Float→Int: truncate toward zero (shared convention, see kernel).
        let (q, _r) = kernel::fp_int_cast(a.fp(), self.prec());
        Ok(MockCell(q))
    }

    fn float_cast(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let scale = kernel::scale_fp(self.prec());
        Ok(MockCell(a.fp() * scale))
    }

    fn bool_cast(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() { Fp::one() } else { Fp::zero() }))
    }

    // ── I/O ───────────────────────────────────────────────────────────

    fn read_input(&mut self, path: &InputPath, _is_public: bool) -> Result<MockCell, ProvingError> {
        let fp = self.witness.resolve_path(path)?;
        Ok(MockCell(fp))
    }

    fn read_external_result(
        &mut self,
        store_idx: u32,
        output_idx: u32,
    ) -> Result<MockCell, ProvingError> {
        let fp = self.witness.resolve_external(store_idx, output_idx)?;
        Ok(MockCell(fp))
    }

    fn expose_public(&mut self, a: &MockCell, label: &str) -> Result<(), ProvingError> {
        self.public_outputs
            .insert(label.to_string(), format!("{}", kernel::fp_to_i64(a.fp())));
        Ok(())
    }

    fn assert_true(&mut self, a: &MockCell) -> Result<(), ProvingError> {
        if !a.is_truthy() {
            self.satisfied = false;
            self.assertion_failures
                .push(format!("Assertion failed: value is zero"));
        }
        Ok(())
    }

    // ── Memory ────────────────────────────────────────────────────────

    // ── Arrays, tables, memories (`crate::ir_array`) ──────────────────

    fn array_write(
        &mut self,
        cells: &[MockCell],
        access: &ArrayAccess,
        positions: &[MockCell],
        value: &MockCell,
        predicate: &MockCell,
    ) -> Result<Vec<MockCell>, ProvingError> {
        let mut new = cells.to_vec();
        if let Some(address) = self.address_of(access, positions, "array write") {
            if predicate.is_truthy() {
                new[address] = value.clone();
            }
        }
        Ok(new)
    }

    fn array_read(&mut self, cells: &[MockCell], access: &ArrayAccess, positions: &[MockCell]) -> Result<MockCell, ProvingError> {
        Ok(match self.address_of(access, positions, "array read") {
            Some(address) => cells[address].clone(),
            None => MockCell(Fp::zero()),
        })
    }

    fn rom_init(&mut self, table: StmtId, cells: &[MockCell]) -> Result<(), ProvingError> {
        self.vectors.insert(table, cells.iter().map(MockCell::fp).collect());
        Ok(())
    }

    fn rom_read(&mut self, table: StmtId, address: &MockCell) -> Result<MockCell, ProvingError> {
        self.vector_read(table, address, "table read")
    }

    fn ram_init(&mut self, memory: StmtId, cells: &[MockCell]) -> Result<(), ProvingError> {
        self.vectors.insert(memory, cells.iter().map(MockCell::fp).collect());
        Ok(())
    }

    fn ram_write(&mut self, from: StmtId, to: StmtId, address: &MockCell, value: &MockCell) -> Result<(), ProvingError> {
        let mut cells = self.vector(from)?.clone();
        match kernel::field_position(address.fp(), cells.len() as u64) {
            Some(a) => cells[a as usize] = value.fp(),
            None => self.unsatisfied(format!("memory write out of bounds: version {from}")),
        }
        self.vectors.insert(to, cells);
        Ok(())
    }

    fn ram_read(&mut self, version: StmtId, address: &MockCell) -> Result<MockCell, ProvingError> {
        self.vector_read(version, address, "memory read")
    }

    // ── Dynamic NDArray ───────────────────────────────────────────────

    fn allocate_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        _: &str,
        max_length: u32,
        max_rank: u32,
        max_shape: &[u32],
    ) -> Result<(), ProvingError> {
        let axis_bounds = if max_shape.is_empty() {
            vec![max_length; max_rank as usize]
        } else {
            max_shape.to_vec()
        };
        self.dyn_meta_bounds
            .insert(array_id, (max_length, axis_bounds));
        Ok(())
    }
    fn witness_dynamic_ndarray_meta(
        &mut self,
        _: u32,
        _: u32,
        _: &str,
        _: &[MockCell],
    ) -> Result<(), ProvingError> {
        Ok(())
    }
    fn dyn_meta_axis_bounds(
        &self,
        array_id: u32,
        max_rank: u32,
        max_length: u32,
    ) -> (u32, Vec<u32>) {
        self.dyn_meta_bounds
            .get(&array_id)
            .cloned()
            .unwrap_or_else(|| (max_length, vec![max_length; max_rank as usize]))
    }
    // Mock overrides the trait-default lowering with a value-inspecting
    // diagnostic: it flips `satisfied` and records a human-readable failure
    // instead of emitting (unenforceable) constraints into the mock backend.
    fn assert_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        max_length: u32,
        args: &[MockCell],
    ) -> Result<(), ProvingError> {
        let rank = max_rank as usize;
        if args.len() != rank && args.len() != rank * 2 {
            self.satisfied = false;
            self.assertion_failures.push(format!(
                "DynamicNDArray metadata assertion failed for array {}: expected {} runtime shape axes with optional bounds, got {} args",
                array_id, max_rank, args.len()
            ));
            return Ok(());
        }
        let (total_bound, mut axis_bounds) = self
            .dyn_meta_bounds
            .get(&array_id)
            .cloned()
            .unwrap_or_else(|| (max_length, vec![max_length; max_rank as usize]));
        if args.len() == rank * 2 {
            axis_bounds = args[rank..]
                .iter()
                .map(|cell| kernel::fp_to_i64(cell.fp()).max(0) as u32)
                .collect();
        }
        let mut runtime_len: u128 = 1;
        for (axis, dim) in args[..rank].iter().enumerate() {
            let dim_value = kernel::fp_to_i64(dim.fp());
            let axis_bound = axis_bounds.get(axis).copied().unwrap_or(max_length);
            if dim_value < 0 || dim_value as u128 > axis_bound as u128 {
                self.satisfied = false;
                self.assertion_failures.push(format!(
                    "DynamicNDArray metadata assertion failed for array {}: axis {} expected 0..={}, got {}",
                    array_id, axis, axis_bound, dim_value
                ));
            }
            runtime_len = runtime_len.saturating_mul(dim_value.max(0) as u128);
        }
        if runtime_len > total_bound as u128 {
            self.satisfied = false;
            self.assertion_failures.push(format!(
                "DynamicNDArray metadata assertion failed for array {}: runtime length {} exceeds max_length {}",
                array_id, runtime_len, total_bound
            ));
        }
        Ok(())
    }

    // ── Hash (same Poseidon as halo2) ─────────────────────────────────

    fn poseidon_hash(&mut self, inputs: &[MockCell]) -> Result<MockCell, ProvingError> {
        let fps: Vec<Fp> = inputs.iter().map(|c| c.fp()).collect();
        Ok(MockCell(kernel::fp_poseidon(&fps)))
    }

    fn eq_hash(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.eq_i(a, b)
    }

    fn packed_key_index(&mut self, key: &MockCell, index_bits: u32) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::packed_key_parts(key.fp(), index_bits).1))
    }

    // ── Prover hints ──────────────────────────────────────────────────

    fn prover_hint(
        &mut self,
        kind: crate::ir_defs::HintKind,
        operands: &[MockCell],
        rank: &MockCell,
    ) -> Result<MockCell, ProvingError> {
        let values: Vec<Fp> = operands.iter().map(|c| c.fp()).collect();
        Ok(MockCell(kernel::hint_lane(kind, &values, rank.fp())))
    }

    /// Exactly, in field arithmetic (the circuits check it at a random
    /// point, which a false identity passes with negligible probability).
    fn identity_check(&mut self, identity: crate::ir_defs::Identity, operands: &[MockCell]) -> Result<(), ProvingError> {
        let values: Vec<Fp> = operands.iter().map(|c| c.fp()).collect();
        if !kernel::identity_holds(identity, &values) {
            self.satisfied = false;
            self.assertion_failures.push(format!("identity {} does not hold", identity.describe()));
        }
        Ok(())
    }

    /// The rescale of `ProductSumF` applied to the raw sum.
    fn rescale_raw_f(&mut self, raw: &MockCell) -> Result<MockCell, ProvingError> {
        let raw = kernel::fp_to_i128(raw.fp()).unwrap_or(0);
        Ok(MockCell(kernel::fp_rescale_raw_i128(raw, self.prec()).0))
    }
}

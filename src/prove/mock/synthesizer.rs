//! `MockSynthesizer` — concrete evaluation of IR operations using Pasta Fp.
//!
//! Uses the same field (Pasta Fp), the same fixed-point quantization, and
//! the same Remez polynomial approximations as the halo2 synthesizer.
//! This ensures semantic consistency: mock execution produces the exact
//! same values that halo2 would constrain.

use std::collections::HashMap;

use pasta_curves::Fp;

use crate::prove::error::ProvingError;
use crate::prove::kernel::{self, Field};
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;
use crate::circuit_input::{ResolvedWitness, InputPath};

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
    memories: HashMap<u32, Vec<Fp>>,
    memory_init: HashMap<u32, Fp>,
}

impl MockSynthesizer {
    pub fn new(witness: ResolvedWitness, params: ProvingParams) -> Self {
        Self {
            witness,
            params,
            satisfied: true,
            assertion_failures: Vec::new(),
            public_outputs: HashMap::new(),
            memories: HashMap::new(),
            memory_init: HashMap::new(),
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
        let mut result = base.fp();
        for _ in 1..e.min(64) {
            result = result * base.fp();
        }
        Ok(MockCell(result))
    }

    fn abs_i(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        let (abs, _) = kernel::signed_decompose(a.fp());
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

    fn div_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_div_prescale(a.fp(), b.fp(), self.prec())))
    }

    fn floor_div_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.div_f(a, b)
    }

    fn mod_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let q = kernel::fp_div_prescale(a.fp(), b.fp(), self.prec());
        let (qb, _) = kernel::fp_mul_rescale(q, b.fp(), self.prec());
        Ok(MockCell(a.fp() - qb))
    }

    fn pow_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let log_a = self.log_f(a)?;
        let (b_log_a, _) = kernel::fp_mul_rescale(b.fp(), log_a.fp(), self.prec());
        self.exp_f(&MockCell(b_log_a))
    }

    fn abs_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        self.abs_i(a)
    }

    fn sign_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        self.sign_i(a)
    }

    // ── Comparisons ───────────────────────────────────────────────────

    fn eq_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.fp() == b.fp() { Fp::one() } else { Fp::zero() }))
    }

    fn ne_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.fp() != b.fp() { Fp::one() } else { Fp::zero() }))
    }

    fn lt_i(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        let diff = a.fp() - b.fp();
        let is_neg = kernel::fp_is_negative(diff);
        let is_zero = diff == Fp::zero();
        Ok(MockCell(if is_neg && !is_zero { Fp::one() } else { Fp::zero() }))
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

    fn eq_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> { self.eq_i(a, b) }
    fn ne_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> { self.ne_i(a, b) }
    fn lt_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> { self.lt_i(a, b) }
    fn lte_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> { self.lte_i(a, b) }
    fn gt_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> { self.gt_i(a, b) }
    fn gte_f(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> { self.gte_i(a, b) }

    // ── Transcendentals (same Remez polynomials as halo2) ─────────────

    fn sin_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_sin(a.fp(), self.prec())))
    }

    fn sinh_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_sinh(a.fp(), self.prec())))
    }

    fn cos_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_cos(a.fp(), self.prec())))
    }

    fn cosh_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_cosh(a.fp(), self.prec())))
    }

    fn tan_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_tan(a.fp(), self.prec())))
    }

    fn tanh_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_tanh(a.fp(), self.prec())))
    }

    fn sqrt_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_sqrt(a.fp(), self.prec())))
    }

    fn exp_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_exp(a.fp(), self.prec())))
    }

    fn log_f(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(kernel::fp_log(a.fp(), self.prec())))
    }

    // ── Boolean logic ─────────────────────────────────────────────────

    fn logical_and(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() && b.is_truthy() { Fp::one() } else { Fp::zero() }))
    }

    fn logical_or(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() || b.is_truthy() { Fp::one() } else { Fp::zero() }))
    }

    fn logical_not(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(MockCell(if a.is_truthy() { Fp::zero() } else { Fp::one() }))
    }

    // ── Select ────────────────────────────────────────────────────────

    fn select(&mut self, cond: &MockCell, t: &MockCell, f: &MockCell) -> Result<MockCell, ProvingError> {
        Ok(if cond.is_truthy() { t.clone() } else { f.clone() })
    }

    // ── Casting ───────────────────────────────────────────────────────

    fn int_cast(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        // Float→Int: divide by scale (truncate)
        let scale_i = crate::prove::field::quantization_scale(self.prec()) as i64;
        let v = kernel::fp_to_i64(a.fp());
        let q = v / scale_i;
        Ok(MockCell(kernel::i64_to_fp(q)))
    }

    fn float_cast(&mut self, a: &MockCell) -> Result<MockCell, ProvingError> {
        // Int→Float: multiply by scale
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

    fn read_external_result(&mut self, store_idx: u32, output_idx: u32) -> Result<MockCell, ProvingError> {
        let fp = self.witness.resolve_external(store_idx, output_idx)?;
        Ok(MockCell(fp))
    }

    fn expose_public(&mut self, a: &MockCell, label: &str) -> Result<(), ProvingError> {
        self.public_outputs.insert(label.to_string(), format!("{}", kernel::fp_to_i64(a.fp())));
        Ok(())
    }

    fn assert_true(&mut self, a: &MockCell) -> Result<(), ProvingError> {
        if !a.is_truthy() {
            self.satisfied = false;
            self.assertion_failures.push(format!("Assertion failed: value is zero"));
        }
        Ok(())
    }

    // ── Memory ────────────────────────────────────────────────────────

    fn allocate_memory(&mut self, segment_id: u32, size: u32, init: i64) -> Result<(), ProvingError> {
        let init_fp = kernel::i64_to_fp(init);
        self.memories.insert(segment_id, vec![init_fp; size as usize]);
        self.memory_init.insert(segment_id, init_fp);
        Ok(())
    }

    fn write_memory(&mut self, segment_id: u32, addr: &MockCell, value: &MockCell) -> Result<(), ProvingError> {
        let a = kernel::fp_to_i64(addr.fp()) as usize;
        let mem = self.memories.get_mut(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("Segment not allocated"))?;
        if a < mem.len() { mem[a] = value.fp(); }
        Ok(())
    }

    fn read_memory(&mut self, segment_id: u32, addr: &MockCell) -> Result<MockCell, ProvingError> {
        let a = kernel::fp_to_i64(addr.fp()) as usize;
        let init = self.memory_init.get(&segment_id).copied().unwrap_or(Fp::zero());
        let mem = self.memories.get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("Segment not allocated"))?;
        let val = if a < mem.len() { mem[a] } else { init };
        Ok(MockCell(val))
    }

    fn memory_trace_emit(&mut self, _: u32, _: bool, _: &[MockCell]) -> Result<(), ProvingError> { Ok(()) }
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError> { Ok(()) }

    // ── Dynamic NDArray ───────────────────────────────────────────────

    fn allocate_dynamic_ndarray_meta(&mut self, array_id: u32, _: &str, max_length: u32, _: u32) -> Result<(), ProvingError> {
        self.allocate_memory(10000 + array_id, max_length, 0)
    }
    fn witness_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: &[MockCell]) -> Result<(), ProvingError> { Ok(()) }
    fn assert_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: u32, _: &[MockCell]) -> Result<(), ProvingError> { Ok(()) }
    fn dynamic_ndarray_get_item(&mut self, array_id: u32, _: u32, index: &MockCell) -> Result<MockCell, ProvingError> {
        self.read_memory(10000 + array_id, index)
    }
    fn dynamic_ndarray_set_item(&mut self, array_id: u32, _: u32, index: &MockCell, value: &MockCell) -> Result<(), ProvingError> {
        self.write_memory(10000 + array_id, index, value)
    }

    // ── Hash (same Poseidon as halo2) ─────────────────────────────────

    fn poseidon_hash(&mut self, inputs: &[MockCell]) -> Result<MockCell, ProvingError> {
        let fps: Vec<Fp> = inputs.iter().map(|c| c.fp()).collect();
        Ok(MockCell(kernel::fp_poseidon(&fps)))
    }

    fn eq_hash(&mut self, a: &MockCell, b: &MockCell) -> Result<MockCell, ProvingError> {
        self.eq_i(a, b)
    }
}

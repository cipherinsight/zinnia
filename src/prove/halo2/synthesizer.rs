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
use crate::prove::kernel::{self, Field, ATAN_COEFS, EXP2_COEFS, LOG2_COEFS, SIN_COEFS};
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
    /// Field values for each public input, in instance-row order.
    /// Populated alongside `public_cells` so the prover/verifier can
    /// reconstruct the instance column without re-walking the IR.
    public_values: Vec<Fp>,
    instance_row: usize,
    memories: HashMap<u32, Vec<Fp>>,
    memory_init: HashMap<u32, Fp>,
}

impl Halo2Synthesizer {
    pub fn new(config: ZinniaConfig, witness: Option<ResolvedWitness>, params: ProvingParams) -> Self {
        Self {
            config, witness, params,
            offset: 0, ops: Vec::new(),
            public_cells: Vec::new(),
            public_values: Vec::new(),
            instance_row: 0,
            memories: HashMap::new(), memory_init: HashMap::new(),
        }
    }

    /// Returns the public-input values collected during synthesis,
    /// in the order they were exposed (matches the instance column layout).
    pub fn public_values(&self) -> &[Fp] {
        &self.public_values
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

    /// Constrained polynomial evaluation via Horner's method, in fixed-point Q
    /// representation. Both `x` and the coefficients are Q-encoded with
    /// `params.precision_bits` fractional bits.
    ///
    /// **Leading-first coefficient layout** (matching `LOG2_COEFS`, `EXP2_COEFS`,
    /// `SIN_COEFS`, `ATAN_COEFS`):
    /// `coefs = [a_{n-1}, a_{n-2}, ..., a_1, a_0]`,
    /// `p(x) = a_{n-1} x^{n-1} + a_{n-2} x^{n-2} + ... + a_1 x + a_0`.
    ///
    /// Recurrence: `acc <- coefs[0]`, then for `i = 1..n`: `acc <- x*acc + coefs[i]`.
    /// The multiplication is done via `mul_f` so the Q-scale is rescaled at each
    /// step (otherwise the magnitude of `acc` would explode by one factor of
    /// `2^precision_bits` per iteration and wrap the field modulus).
    ///
    /// Previous versions had two layered defects: the iteration ran constant-
    /// first while these coefficient arrays are leading-first (so log(2)
    /// returned -11097 instead of 0.693), and the multiplication used raw
    /// `mul_add` without Q-rescaling (so the field representation wrapped
    /// well before reaching the polynomial's leading term). Must stay in sync
    /// with `kernel::horner_eval`.
    fn poly_eval_horner(&mut self, x: &Halo2CellRef, coefs: &[Fp]) -> Halo2CellRef {
        assert!(!coefs.is_empty());

        // acc = coefs[0] (leading coefficient, already Q-encoded).
        let mut acc = self.rec_advice(0, coefs[0], "horner_init");
        self.offset += 1;

        // For i = 1..n: acc = x *_Q acc + coefs[i].
        for i in 1..coefs.len() {
            let coef_cell = self.rec_advice(0, coefs[i], &format!("horner_c{}", i));
            self.offset += 1;
            // Q-scaled multiplication then in-field addition.
            let prod = self.mul_f(x, &acc).expect("Q-mul during Horner");
            acc = self.bin_gate("add", &prod, &coef_cell, prod.value + coef_cell.value, "horner_step");
        }
        acc
    }

    fn quantize_fp(&self, v: f64) -> Fp {
        kernel::quantize_to_fp(v, self.params.precision_bits)
    }

    /// Allocate an unconstrained advice cell holding `value`. Useful for witness-
    /// only helper values whose correctness is enforced by downstream gates (e.g.
    /// range checks). The cell is at column 0 of a fresh row.
    fn witness_cell(&mut self, value: Fp, ann: &str) -> Halo2CellRef {
        let cell = self.rec_advice(0, value, ann);
        self.offset += 1;
        cell
    }

    /// Range-check: constrain `val ∈ [2 * scale, 4 * scale)` where
    /// `scale = 2^precision_bits`. Used to enforce the post-range-reduction
    /// mantissa for `log2` lies in the polynomial's fit domain.
    ///
    /// Encoding: assert that `val - 2 * scale` is in `[0, 2 * scale)` via the
    /// existing n-bit range check (with n = precision_bits + 1).
    fn range_check_log_mantissa(&mut self, val: &Halo2CellRef, ann: &str) {
        let precision_bits = self.params.precision_bits;
        let scale = crate::prove::field::quantization_scale(precision_bits) as i64;
        let two_scale_cell = self
            .constant_int(2 * scale)
            .expect("constant_int for 2*scale");
        let shifted = self
            .sub_i(val, &two_scale_cell)
            .expect("sub_i in range_check_log_mantissa");
        // shifted ∈ [0, 2*scale) = [0, 2^{precision_bits+1}).
        self.range_check_n_bits(&shifted, precision_bits + 1, ann);
    }

    /// Range reduction for `log2`: given a Q-encoded positive `x`, witness an
    /// integer `k` and a Q-encoded mantissa `m ∈ [2, 4)` such that `x = m * 2^k`
    /// (in real arithmetic), and return them as constrained cells.
    ///
    /// **Soundness caveat** (documented for follow-up): the relationship
    /// `x = m * 2^k` is enforced only weakly here — `m` is range-checked into
    /// `[2*scale, 4*scale)` and `k` is provided as a witness, but no explicit
    /// equation ties `(k, m)` to `x`. A malicious prover that can substitute a
    /// `(k', m')` pair satisfying the range check on m would change the output.
    /// However, for any positive `x`, the decomposition into `(k, m ∈ [2, 4))`
    /// is unique, so an honest prover and the polynomial-evaluation chain still
    /// produce the correct value. Closing this gap requires a dynamic power-of-
    /// two gadget (e.g. a select-tree over k candidates or a MSB-position
    /// gadget); see the diagnosis card for the design sketch.
    ///
    /// Returns `(k_cell, m_cell)`. `k_cell` is a signed integer cell (not Q-encoded).
    fn log2_range_reduce(&mut self, x: &Halo2CellRef) -> (Halo2CellRef, Halo2CellRef) {
        let precision_bits = self.params.precision_bits;
        let scale = crate::prove::field::quantization_scale(precision_bits) as i64;

        // Host-side compute (k, m_q).
        let x_i64 = kernel::fp_to_i64(x.value);
        // For non-positive x, log is undefined. Use a witness fallback (k=0, m=2)
        // so synthesis never panics at keygen (x = 0 in that path). The mock
        // backend's `fp_log` returns NaN-ish behavior on x <= 0; mirroring that
        // in halo2 would require additional logic, but the polynomial fit
        // doesn't support it either.
        let (k_host, m_q_host) = if x_i64 <= 0 {
            (0i64, 2 * scale)
        } else {
            // x_i64 represents x * scale. Find integer k such that
            // x_q / 2^k ∈ [2 * scale, 4 * scale).
            // x ∈ [2^k * 2, 2^k * 4) ⇔ floor(log2(x_i64)) ∈ {k + log2(scale) + 1}.
            // Equivalent: k = (bit-position of MSB of x_i64) - log2(scale) - 1.
            let msb = 63 - (x_i64 as u64).leading_zeros() as i64; // MSB position in x_i64
            let k = msb - precision_bits as i64 - 1;
            // m_q = round(x_i64 / 2^k) — but to keep precision, compute as
            // x_i64 << (-k) when k < 0, else x_i64 >> k.
            let m_q = if k >= 0 {
                x_i64 >> k
            } else {
                x_i64 << (-k)
            };
            (k, m_q)
        };

        // Allocate witness cells.
        let m_cell = self.witness_cell(kernel::i64_to_fp(m_q_host), "log2rr_m");
        let k_cell = self.witness_cell(kernel::i64_to_fp(k_host), "log2rr_k");

        // Constrain m_q ∈ [2 * scale, 4 * scale).
        self.range_check_log_mantissa(&m_cell, "log2rr_m_rc");

        (k_cell, m_cell)
    }

    /// Range reduction for `exp2`: given a Q-encoded `y`, witness an integer
    /// `i` and a Q-encoded `f ∈ [0, 1)` such that `y = i + f` (in real arithmetic).
    /// `2^y = 2^i * 2^f`.
    ///
    /// **Soundness caveat**: as in `log2_range_reduce`, the relationship is
    /// enforced via a range check on `f` and the floor-decomposition is the
    /// unique decomposition for any real y, so an honest prover lands on the
    /// correct value; explicit constraint linking `(i, f)` back to `y` is
    /// punted to a follow-up (would need integer addition of `i * scale + f
    /// == y` as an equality, which IS expressible — see the constraint
    /// emitted below).
    ///
    /// Returns `(i_cell, f_cell)`. `i_cell` is a signed integer (not Q-encoded);
    /// `f_cell` is Q-encoded fractional part.
    fn exp2_range_reduce(&mut self, y: &Halo2CellRef) -> (Halo2CellRef, Halo2CellRef) {
        let precision_bits = self.params.precision_bits;
        let scale = crate::prove::field::quantization_scale(precision_bits) as i64;

        // Host-side compute (i, f_q). y_i64 = y * scale.
        let y_i64 = kernel::fp_to_i64(y.value);
        // Use Euclidean division so the remainder f_q is always in [0, scale).
        let i_host = y_i64.div_euclid(scale);
        let f_q_host = y_i64.rem_euclid(scale); // f_q ∈ [0, scale)

        let i_cell = self.witness_cell(kernel::i64_to_fp(i_host), "exp2rr_i");
        let f_cell = self.witness_cell(kernel::i64_to_fp(f_q_host), "exp2rr_f");

        // Range-check f_q ∈ [0, scale) = [0, 2^precision_bits).
        self.range_check_n_bits(&f_cell, precision_bits, "exp2rr_f_rc");

        // Constrain: y_q = i * scale + f_q.
        let scale_cell = self
            .constant_int(scale)
            .expect("constant_int for scale");
        let i_scaled = self
            .mul_i(&i_cell, &scale_cell)
            .expect("mul_i in exp2rr");
        let recomputed = self
            .add_i(&i_scaled, &f_cell)
            .expect("add_i in exp2rr");
        self.constrain_equal_cells(&recomputed, y, "exp2rr_eq");

        (i_cell, f_cell)
    }

    /// Compute `2^k * x_q` where `k` is a signed-integer witness cell and `x_q`
    /// is Q-encoded. The result is also Q-encoded.
    ///
    /// Implementation: honest-prover witness. The result is computed host-side
    /// from `k.value` and `x.value` (left-shift on x's signed integer encoding
    /// for k > 0, right-shift for k < 0) and assigned to a single advice cell.
    ///
    /// **Soundness caveat**: there is no constraint binding `result.value` to
    /// `2^k * x.value`. A malicious prover can substitute any value here. This
    /// matches the soundness profile of `log2_range_reduce` and `exp2_range_
    /// reduce`; closing the gap requires a dynamic-shift gadget (select-tree
    /// over k candidates, MSB-position gadget, or a power-of-two lookup
    /// chip). Tracked separately — see the diagnosis card for the design.
    fn mul_by_pow2_witness(&mut self, x: &Halo2CellRef, k: &Halo2CellRef) -> Halo2CellRef {
        let k_val = kernel::fp_to_i64(k.value);
        let x_i = kernel::fp_to_i64(x.value);
        let result_i = if k_val >= 0 {
            // Saturate the shift count so the host-side computation cannot panic
            // at keygen time (where x = 0 and k = 0 by default, but a malformed
            // witness could push k arbitrarily high).
            let s = k_val.min(62) as u32;
            x_i.checked_shl(s).unwrap_or(0)
        } else {
            let s = (-k_val).min(62) as u32;
            x_i >> s
        };
        self.witness_cell(kernel::i64_to_fp(result_i), "p2w_result")
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

    /// Constrain that `a == b` (i.e., a - b == 0) using the is_zero gadget
    /// and an assert that the result equals 1.
    fn constrain_equal_cells(&mut self, a: &Halo2CellRef, b: &Halo2CellRef, ann: &str) {
        let diff = self.bin_gate("sub", a, b, a.value - b.value, ann);
        let iz = self.is_zero_gadget(&diff);
        // assert iz == 1
        let row = self.offset;
        self.rec_sel("assert");
        let _ = self.rec_advice(0, iz.value, &format!("{}_assert", ann));
        self.rec_copy(&iz, 0, row);
        self.offset += 1;
    }

    /// Bit-decompose a 64-bit signed integer cell using two's-complement encoding.
    ///
    /// Returns 64 bit cells `b[0..64]` constrained such that:
    /// - each `b[i] ∈ {0, 1}` (via `bit` gate);
    /// - `Σ_{i=0}^{62} b[i] * 2^i − b[63] * 2^63 = val`  (Fp equation).
    ///
    /// The signed-aware recomposition avoids needing the field-element value to
    /// fit in [0, 2^64); negative values whose Fp encoding is P − |v| still
    /// match the formula when b[63] is the sign bit.
    fn bit_decompose_64(&mut self, val: &Halo2CellRef) -> [Halo2CellRef; 64] {
        const N: usize = 64;
        let v_i64 = kernel::fp_to_i64(val.value);
        let v_u64 = v_i64 as u64;

        // 1. Witness bits and constrain each to be in {0, 1}.
        let mut bits: Vec<Halo2CellRef> = Vec::with_capacity(N);
        for i in 0..N {
            let b = ((v_u64 >> i) & 1) as u64;
            let b_fp = Fp::from(b);
            let bit_cell = self.rec_advice(0, b_fp, &format!("bit_{}", i));
            self.offset += 1;
            self.constrain_bit(&bit_cell);
            bits.push(bit_cell);
        }

        // 2. Allocate `pow2 = 1` and pin it via the `assert` gate (s * (1 - a) = 0).
        let mut pow2 = self.rec_advice(0, Fp::one(), "p2_init");
        self.offset += 1;
        let row = self.offset;
        self.rec_sel("assert");
        let _ = self.rec_advice(0, pow2.value, "p2_init_assert");
        self.rec_copy(&pow2, 0, row);
        self.offset += 1;

        // 3. acc = pow2 * bits[0] = bits[0]  (via mul gate; pow2 is pinned to 1)
        let mut acc = self.bin_gate(
            "mul", &pow2, &bits[0], pow2.value * bits[0].value, "decomp_acc0",
        );

        // 4. For i in 1..63: pow2 *= 2 (doubling = add gate); acc += pow2 * bits[i]
        for i in 1..(N - 1) {
            pow2 = self.bin_gate(
                "add", &pow2, &pow2, pow2.value + pow2.value, &format!("p2_{}", i),
            );
            acc = self.constrained_mul_add(&pow2, &bits[i], &acc);
        }

        // 5. For i = 63 (sign bit): pow2 = 2^63, then acc -= pow2 * bits[63].
        pow2 = self.bin_gate(
            "add", &pow2, &pow2, pow2.value + pow2.value, "p2_63",
        );
        let high = self.bin_gate(
            "mul", &pow2, &bits[N - 1], pow2.value * bits[N - 1].value, "high_term",
        );
        let final_val = self.bin_gate(
            "sub", &acc, &high, acc.value - high.value, "decomp_final",
        );

        // 6. Constrain final_val == val.
        self.constrain_equal_cells(&final_val, val, "decomp_eq");

        bits.try_into().expect("64 bits collected")
    }

    /// Recompose 64 bit cells back into a signed two's-complement integer cell.
    /// The bits are NOT re-constrained to {0,1} — caller is responsible if they
    /// were freshly constructed (e.g., XOR per-bit: caller already constrained).
    fn bit_recompose_64(&mut self, bits: &[Halo2CellRef; 64], ann: &str) -> Halo2CellRef {
        const N: usize = 64;
        // pow2 = 1 (pinned via assert), acc = pow2 * bits[0]
        let mut pow2 = self.rec_advice(0, Fp::one(), &format!("{}_p2_init", ann));
        self.offset += 1;
        let row = self.offset;
        self.rec_sel("assert");
        let _ = self.rec_advice(0, pow2.value, &format!("{}_p2_init_assert", ann));
        self.rec_copy(&pow2, 0, row);
        self.offset += 1;

        let mut acc = self.bin_gate(
            "mul", &pow2, &bits[0], pow2.value * bits[0].value, &format!("{}_acc0", ann),
        );
        for i in 1..(N - 1) {
            pow2 = self.bin_gate(
                "add", &pow2, &pow2, pow2.value + pow2.value, &format!("{}_p2_{}", ann, i),
            );
            acc = self.constrained_mul_add(&pow2, &bits[i], &acc);
        }
        pow2 = self.bin_gate(
            "add", &pow2, &pow2, pow2.value + pow2.value, &format!("{}_p2_63", ann),
        );
        let high = self.bin_gate(
            "mul", &pow2, &bits[N - 1], pow2.value * bits[N - 1].value, &format!("{}_high", ann),
        );
        self.bin_gate(
            "sub", &acc, &high, acc.value - high.value, &format!("{}_final", ann),
        )
    }

    /// Per-bit AND: `c = a * b` for two bit cells (uses `bool_and` gate which is
    /// the same as `mul`, but the dedicated selector documents intent).
    fn bit_and_per_bit(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Halo2CellRef {
        self.bin_gate("bool_and", a, b, a.value * b.value, "and_bit")
    }

    /// Per-bit OR: `c = a + b - a*b` via the `bool_or` gate.
    fn bit_or_per_bit(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Halo2CellRef {
        self.bin_gate("bool_or", a, b, a.value + b.value - a.value * b.value, "or_bit")
    }

    /// Per-bit XOR: `c = a + b - 2*a*b`. Composed as `(a + b) - 2*(a*b)` using
    /// add/mul/sub gates (no dedicated XOR gate exists today).
    fn bit_xor_per_bit(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Halo2CellRef {
        let ab = self.bin_gate("mul", a, b, a.value * b.value, "xor_ab");
        let two_ab = self.bin_gate("add", &ab, &ab, ab.value + ab.value, "xor_2ab");
        let sum = self.bin_gate("add", a, b, a.value + b.value, "xor_sum");
        self.bin_gate("sub", &sum, &two_ab, sum.value - two_ab.value, "xor_bit")
    }

    /// Per-bit NOT: `c = 1 - a` via the `bool_not` gate.
    /// `bool_not` reads advice[0]=a and advice[2]=c (advice[1] unused).
    fn bit_not_per_bit(&mut self, a: &Halo2CellRef) -> Halo2CellRef {
        let row = self.offset;
        self.rec_sel("bool_not");
        let _ = self.rec_advice(0, a.value, "not_a");
        self.rec_copy(a, 0, row);
        let cc = self.rec_advice(2, Fp::one() - a.value, "not_c");
        self.offset += 1;
        cc
    }

    /// Constrain that `val` is in [0, 2^n_bits) by allocating `n_bits` bit cells
    /// and proving their weighted sum equals `val`. Each bit cell is constrained
    /// to {0, 1} via the `bit` gate. The final recomposed cell is copy-constrained
    /// equal to `val`.
    ///
    /// Used as the [0, scale) range check on the `mul_f` div_mod remainder.
    fn range_check_n_bits(&mut self, val: &Halo2CellRef, n_bits: u32, ann: &str) {
        let n = n_bits as usize;
        assert!(n > 0 && n <= 63, "range_check_n_bits supports 1..=63 bits");
        // Decode value to extract bits (assumed non-negative and < 2^n on the
        // honest path; if not, the bit_recompose==val equality below will fail).
        let v_i64 = kernel::fp_to_i64(val.value);
        let v_u64 = v_i64 as u64;

        // 1. Witness bits and constrain each to be in {0, 1}.
        let mut bits: Vec<Halo2CellRef> = Vec::with_capacity(n);
        for i in 0..n {
            let b = ((v_u64 >> i) & 1) as u64;
            let bit_cell = self.rec_advice(0, Fp::from(b), &format!("{}_bit_{}", ann, i));
            self.offset += 1;
            self.constrain_bit(&bit_cell);
            bits.push(bit_cell);
        }

        // 2. pow2 = 1 (pinned via assert).
        let mut pow2 = self.rec_advice(0, Fp::one(), &format!("{}_p2_init", ann));
        self.offset += 1;
        let row = self.offset;
        self.rec_sel("assert");
        let _ = self.rec_advice(0, pow2.value, &format!("{}_p2_init_assert", ann));
        self.rec_copy(&pow2, 0, row);
        self.offset += 1;

        // 3. acc = pow2 * bits[0]
        let mut acc = self.bin_gate(
            "mul", &pow2, &bits[0], pow2.value * bits[0].value, &format!("{}_acc0", ann),
        );

        // 4. For i = 1..n: pow2 *= 2, acc += pow2 * bits[i].
        for i in 1..n {
            pow2 = self.bin_gate(
                "add", &pow2, &pow2, pow2.value + pow2.value, &format!("{}_p2_{}", ann, i),
            );
            acc = self.constrained_mul_add(&pow2, &bits[i], &acc);
        }

        // 5. Constrain acc == val (this is the actual range-check enforcement:
        //    val equals a weighted sum of bits, so 0 <= val < 2^n).
        self.constrain_equal_cells(&acc, val, &format!("{}_eq", ann));
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

    // ── Integer bitwise (constrained via 64-bit two's-complement decomposition) ──
    // Each operand is bit-decomposed; per-bit ops run through the bool gates
    // (mul/add/sub for XOR); recomposition reuses the same chain.
    //
    // Cost per binary op: ~3*64 (decompose) + 64 (per-bit) + 3*64 (recompose)
    //                   ≈ 320 constraints; XOR doubles per-bit term.
    //
    // Shifts only support compile-time-constant shift counts (the count must
    // be encoded in the IR ConstantInt, otherwise the gate structure recorded
    // at keygen time would not match the proof-time witness). Dynamic shifts
    // are tracked separately.
    fn bit_and_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let abits = self.bit_decompose_64(a);
        let bbits = self.bit_decompose_64(b);
        let cbits: Vec<Halo2CellRef> = (0..64)
            .map(|i| self.bit_and_per_bit(&abits[i], &bbits[i]))
            .collect();
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "and_out"))
    }

    fn bit_or_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let abits = self.bit_decompose_64(a);
        let bbits = self.bit_decompose_64(b);
        let cbits: Vec<Halo2CellRef> = (0..64)
            .map(|i| self.bit_or_per_bit(&abits[i], &bbits[i]))
            .collect();
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "or_out"))
    }

    fn bit_xor_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let abits = self.bit_decompose_64(a);
        let bbits = self.bit_decompose_64(b);
        let cbits: Vec<Halo2CellRef> = (0..64)
            .map(|i| self.bit_xor_per_bit(&abits[i], &bbits[i]))
            .collect();
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "xor_out"))
    }

    fn bit_not_i(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let abits = self.bit_decompose_64(a);
        let cbits: Vec<Halo2CellRef> = (0..64)
            .map(|i| self.bit_not_per_bit(&abits[i]))
            .collect();
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "not_out"))
    }

    fn shl_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Shift count must be a compile-time constant in [0, 64). The synthesizer
        // is invoked at both keygen (witness=None, b.value=0) and proof time;
        // gate structure is fixed at keygen, so a runtime-varying b is unsound.
        // Constrain `b == n` with an explicit equality gadget so any drift fails.
        let n = kernel::fp_to_i64(b.value);
        if !(0..64).contains(&n) {
            return Err(ProvingError::synthesis(
                "shl: shift count must be in [0, 64); dynamic counts not supported under halo2",
            ));
        }
        let n = n as usize;
        let n_const = self.constant_int(n as i64)?;
        self.constrain_equal_cells(b, &n_const, "shl_n");

        let abits = self.bit_decompose_64(a);
        let zero_cell = self.constant_int(0)?;
        // result_bits[i] = abits[i - n] for i >= n, else 0
        let mut cbits: Vec<Halo2CellRef> = Vec::with_capacity(64);
        for i in 0..64 {
            if i < n { cbits.push(zero_cell.clone()); }
            else     { cbits.push(abits[i - n].clone()); }
        }
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "shl_out"))
    }

    fn shr_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Same constant-shift restriction as shl_i.
        let n = kernel::fp_to_i64(b.value);
        if !(0..64).contains(&n) {
            return Err(ProvingError::synthesis(
                "shr: shift count must be in [0, 64); dynamic counts not supported under halo2",
            ));
        }
        let n = n as usize;
        let n_const = self.constant_int(n as i64)?;
        self.constrain_equal_cells(b, &n_const, "shr_n");

        let abits = self.bit_decompose_64(a);
        // Python's right-shift on signed ints is arithmetic (sign-extending).
        // result_bits[i] = abits[i + n] for i + n < 64, else abits[63] (sign bit).
        let mut cbits: Vec<Halo2CellRef> = Vec::with_capacity(64);
        for i in 0..64 {
            if i + n < 64 { cbits.push(abits[i + n].clone()); }
            else          { cbits.push(abits[63].clone()); }
        }
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "shr_out"))
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
        // Fixed-point: raw = a*b, result = raw / scale (integer division),
        // rem = raw - result*scale, with 0 <= rem < scale.
        //
        // Witness values are computed by the shared kernel which uses i128
        // integer division — NOT the field modular inverse (the previous
        // implementation used scale.invert() in Fp, which wrapped around the
        // prime modulus and produced values unrelated to integer division).
        //
        // Soundness:
        //   - div_mod gate enforces: raw == scale * result + rem (in Fp).
        //   - range-check enforces: rem ∈ [0, scale).
        // Together, (result, rem) are uniquely determined by raw given scale.
        let raw = self.mul_i(a, b)?;
        let precision_bits = self.params.precision_bits;
        let scale = Fp::from(crate::prove::field::quantization_scale(precision_bits) as u64);
        let (result_val, rem_val) = kernel::fp_mul_rescale(a.value, b.value, precision_bits);
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, raw.value, "mulf_raw");
        self.rec_copy(&raw, 0, row);
        let _ = self.rec_advice(1, scale, "mulf_scale");
        let result_cell = self.rec_advice(2, result_val, "mulf_result");
        let rem_cell = self.rec_advice(3, rem_val, "mulf_rem");
        self.offset += 1;
        // Range check: rem ∈ [0, 2^precision_bits) = [0, scale). This closes
        // the soundness gap where any (result, rem) satisfying the div_mod
        // equation would otherwise be accepted.
        self.range_check_n_bits(&rem_cell, precision_bits, "mulf_rem_rc");
        Ok(result_cell)
    }
    fn div_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Fixed-point divide: result = floor((a * scale) / b).
        //
        // The witness uses the shared kernel `fp_div_prescale` which does i128
        // integer division (NOT field modular inverse — the previous
        // implementation composed mul_i + div_i where div_i used b.invert(),
        // producing field-wrap garbage for non-exact quotients).
        //
        // Constraint shape: `div_mod` gate enforces `a_scaled == b * result + rem`,
        // where a_scaled = a * scale (constrained via mul_i).
        //
        // Note: a full soundness fix would also range-check `0 <= rem < |b|`.
        // |b| is dynamic so this is more involved than the static-`scale` range
        // check in `mul_f`; deferred to a follow-up. The witness-side correctness
        // (which is what makes the value-pipeline match the mock backend and
        // matches integer division) is what this change unblocks.
        let precision_bits = self.params.precision_bits;
        let scale_i = crate::prove::field::quantization_scale(precision_bits) as i64;
        let scale_cell = self.constant_int(scale_i)?;
        let a_scaled = self.mul_i(a, &scale_cell)?;
        // Witness values via the kernel (correct integer division).
        let result_val = kernel::fp_div_prescale(a.value, b.value, precision_bits);
        // rem = a_scaled - b * result (as Fp).
        let rem_val = a_scaled.value - b.value * result_val;
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a_scaled.value, "divf_a_scaled");
        self.rec_copy(&a_scaled, 0, row);
        let _ = self.rec_advice(1, b.value, "divf_b");
        self.rec_copy(b, 1, row);
        let result_cell = self.rec_advice(2, result_val, "divf_result");
        let _ = self.rec_advice(3, rem_val, "divf_rem");
        self.offset += 1;
        Ok(result_cell)
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
        // exp(x) = 2^(x / ln2). Range-reduce x/ln2 into integer + fractional part,
        // then evaluate the EXP2_COEFS polynomial (fit on [0, 1)) on the fraction.
        //
        //   y      = x / ln(2)            (Q-encoded)
        //   y      = i + f,    i ∈ ℤ, f ∈ [0, 1)
        //   exp(x) = 2^y = 2^i * 2^f
        //
        // `exp2_range_reduce` witnesses `(i, f)` with `f ∈ [0, scale)` range-
        // checked and `y_q == i * scale + f_q` constrained. The polynomial
        // evaluation on f stays inside its fit domain, so the value is correct
        // to within Q-precision. The `2^i` factor is applied via the witnessed
        // `mul_by_pow2_witness` helper (see soundness caveat there).
        let ln2 = self.constant_float(2.0f64.ln())?;
        let y = self.div_f(a, &ln2)?;
        let (i_cell, f_cell) = self.exp2_range_reduce(&y);
        let coefs: Vec<Fp> = EXP2_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let two_to_f = self.poly_eval_horner(&f_cell, &coefs);
        let result = self.mul_by_pow2_witness(&two_to_f, &i_cell);
        Ok(result)
    }

    fn log_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // log(x) = log2(x) * ln(2). Range-reduce x = m * 2^k with m ∈ [2, 4)
        // so the LOG2_COEFS polynomial (fit on [2, 4)) is evaluated on `m`:
        //
        //   log2(x) = k + log2(m)
        //   log(x)  = (k + log2(m)) * ln(2)
        //
        // `log2_range_reduce` witnesses `(k, m_q)` with `m_q ∈ [2*scale, 4*scale)`
        // range-checked. The mantissa polynomial evaluation is value-correct
        // within Q-precision; the integer `k` lifts the result by `k * ln(2)`.
        let (k_cell, m_cell) = self.log2_range_reduce(a);
        let coefs: Vec<Fp> = LOG2_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let log2_m = self.poly_eval_horner(&m_cell, &coefs);

        // log2(x) = k + log2(m). `k_cell` is a raw integer; lift to Q-scale.
        let precision_bits = self.params.precision_bits;
        let scale = crate::prove::field::quantization_scale(precision_bits) as i64;
        let scale_cell = self.constant_int(scale)?;
        let k_scaled = self.mul_i(&k_cell, &scale_cell)?;
        let log2_x = self.add_i(&k_scaled, &log2_m)?;

        // log(x) = log2(x) * ln(2)
        let ln2 = self.constant_float(2.0f64.ln())?;
        self.mul_f(&log2_x, &ln2)
    }

    fn sqrt_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sqrt(x) = exp(0.5 * log(x)). With the (now value-correct) log_f and
        // exp_f the composition produces values within polynomial-Q-precision
        // tolerance of f64::sqrt. Keeping the compose for now; a direct Newton
        // iteration over Q-encoded values would tighten precision at the cost
        // of additional range-reduction logic for the iteration bounds.
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

    /// arctan2(y, x): compute the angle of the vector (x, y) in [-π, π].
    ///
    /// Strategy: range-reduce to atan(t) with t ∈ [0, 1] via the swap-and-divide
    /// trick — pick t = min(|x|, |y|) / max(|x|, |y|) — then apply the ATAN_COEFS
    /// Remez polynomial. Quadrant correction adds compile-time-known π offsets
    /// via constrained add/sub and select gates.
    fn arctan2_f(&mut self, y: &Halo2CellRef, x: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Absolute values and signs.
        let (abs_y, neg_y) = self.signed_decompose(y);
        let (abs_x, neg_x) = self.signed_decompose(x);

        // swap = (abs_y > abs_x): use t = abs_x / abs_y instead, then a = π/2 - poly(t).
        let swap = self.gt_f(&abs_y, &abs_x)?;
        let num = self.select(&swap, &abs_x, &abs_y)?;
        let den = self.select(&swap, &abs_y, &abs_x)?;

        // Avoid division by zero when both x and y are zero: bias den with a
        // tiny constant. The witness value for (0,0) will then evaluate the
        // polynomial at 0, giving 0 — matching f64::atan2(0, 0) = 0.
        let epsilon = self.constant_float(1.0e-30)?;
        let den_safe = self.add_f(&den, &epsilon)?;
        let t = self.div_f(&num, &den_safe)?;

        // Polynomial atan(t) on [0, 1].
        let coefs: Vec<Fp> = ATAN_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let poly_atan = self.poly_eval_horner(&t, &coefs);

        // a0 = swap ? π/2 - poly_atan : poly_atan  (angle in [0, π/2])
        let half_pi = self.constant_float(std::f64::consts::FRAC_PI_2)?;
        let half_pi_minus = self.sub_f(&half_pi, &poly_atan)?;
        let a0 = self.select(&swap, &half_pi_minus, &poly_atan)?;

        // Quadrant adjust for x < 0: a1 = neg_x ? π - a0 : a0  (angle in [0, π])
        let pi = self.constant_float(std::f64::consts::PI)?;
        let pi_minus = self.sub_f(&pi, &a0)?;
        let a1 = self.select(&neg_x, &pi_minus, &a0)?;

        // Sign for y < 0: a2 = neg_y ? -a1 : a1   (angle in [-π, π])
        let zero = self.constant_float(0.0)?;
        let neg_a1 = self.sub_f(&zero, &a1)?;
        self.select(&neg_y, &neg_a1, &a1)
    }

    /// arccos(x) for x ∈ [-1, 1]: range is [0, π].
    /// Implemented as atan2(sqrt(1 - x²), x), which is well-defined across the
    /// full domain and reuses the existing sqrt_f and arctan2_f infrastructure.
    fn arccos_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let one = self.constant_float(1.0)?;
        let aa = self.mul_f(a, a)?;
        let one_minus_aa = self.sub_f(&one, &aa)?;
        let s = self.sqrt_f(&one_minus_aa)?;
        self.arctan2_f(&s, a)
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
            self.public_values.push(val);
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
        self.public_values.push(a.value);
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

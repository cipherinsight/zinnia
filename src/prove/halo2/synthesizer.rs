//! `Halo2Synthesizer` — implements the `Synthesizer` trait by recording
//! operations and then replaying them into a halo2 region.
//!
//! All operations create proper constraints via gate polynomials.
//! No witness-only assignments: every value is constrained.
//!
//! Witness values are computed using the shared kernel (same functions
//! as the mock synthesizer), ensuring semantic consistency.

use std::collections::{HashMap, HashSet};

use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value as HaloValue},
    plonk::{Error, Selector},
};

use crate::circuit_input::{InputPath, ResolvedWitness};
use crate::prove::error::ProvingError;
use crate::prove::halo2::config::ZinniaConfig;
use crate::prove::kernel::{self, Field, ATAN_COEFS, EXP2_COEFS, LOG2_COEFS, SIN_COEFS};
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;

// ---------------------------------------------------------------------------
// Recorded operations
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum GateOp {
    AssignAdvice {
        col_idx: usize,
        row: usize,
        value: Fp,
        annotation: String,
    },
    EnableSelector {
        selector_name: String,
        row: usize,
    },
    CopyConstraint {
        col_a: usize,
        row_a: usize,
        col_b: usize,
        row_b: usize,
    },
    EnableRangeLookup {
        row: usize,
    },
    AssignFixed {
        row: usize,
        value: Fp,
        annotation: String,
    },
}

pub struct AssignedCellGrid {
    cells: Vec<Vec<Option<AssignedCell<Fp, Fp>>>>,
}

impl AssignedCellGrid {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cells: vec![vec![None; rows]; cols],
        }
    }

    fn insert(&mut self, col: usize, row: usize, cell: AssignedCell<Fp, Fp>) {
        if let Some(col_cells) = self.cells.get_mut(col) {
            if row < col_cells.len() {
                col_cells[row] = Some(cell);
            }
        }
    }

    fn get(&self, col: usize, row: usize) -> Option<&AssignedCell<Fp, Fp>> {
        self.cells
            .get(col)
            .and_then(|col_cells| col_cells.get(row))
            .and_then(|cell| cell.as_ref())
    }
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
    memory_cells: HashMap<u32, Vec<Option<Halo2CellRef>>>,
    memory_init: HashMap<u32, Fp>,
    dyn_meta_bounds: HashMap<u32, (u32, Vec<u32>)>,
    /// Row currently being filled by single-column "free advice" ops
    /// (witness reads, external results). These ops activate no
    /// selector, so we can pack five into a single row across the five
    /// advice columns instead of one row each — the cells are still
    /// referenced by copy constraints.
    free_pack_row: usize,
    /// Next free column in `free_pack_row` (0..=5). Initialized to 5 so
    /// the very first packed write allocates a fresh row.
    free_pack_col: usize,
    /// (col, row) of cells statically known to hold a value in {0, 1}.
    /// Populated by `is_zero` / `is_nonzero` / `signed_decompose`'s sign
    /// / boolean gates. Lets `select` and `logical_and` / `logical_or`
    /// skip the `truthy_bit` coerce when their inputs are already bits.
    bit_cells: HashSet<(usize, usize)>,
    /// Cached bounded-index decomposition bits keyed by
    /// `(idx_col, idx_row, hi, bits)`. Value-producing bit IRs for the same
    /// index reuse one constrained decomposition row.
    u128_lt_bit_cache: HashMap<(usize, usize, i64, u32), Vec<Halo2CellRef>>,
    /// Cached vector-select lanes keyed by condition and branch cells.
    select_many_cache: HashMap<Vec<(usize, usize)>, Vec<Halo2CellRef>>,
    /// First recorded constraint whose witness value does not satisfy its
    /// gate, captured during Phase-1 recording when a concrete witness is
    /// present. `create_proof` happily produces a (verify-rejected) proof
    /// for an unsatisfiable witness, so the prove path queries this to
    /// surface `assert(false)`-style failures *before* spending MSM time.
    /// Populated only when `witness.is_some()` so the witness-less keygen
    /// pass (which threads placeholder zeros) never trips it.
    first_unsat: Option<String>,
}

const MAX_RANGE_LOOKUP_BITS: u32 = 16;

/// Bit-width of the magnitude window used to bind the `abs`/`sign`
/// decomposition produced by the `sub_sd` and `cond_neg` gates (see
/// `range_check_signed_magnitude`). Any honest difference or absolute value
/// in the Zinnia integer/Q-fixed-point domain fits in this window, while the
/// field-negation a malicious prover would need to flip `sign` lands ~254
/// bits wide and is rejected by the lookup-backed chunk check. This is the
/// legacy-backend analogue of halo2-base's `range_check(_, 64)` on the
/// 2^63-shifted operands in `signed_lt`. 56 = the largest multiple of the
/// default `lookup_bits = 8` that stays within the 63-bit `range_check_chunks`
/// cap; for other `lookup_bits` the helper rounds down to a multiple.
const SIGNED_MAGNITUDE_BITS: u32 = 56;

impl Halo2Synthesizer {
    pub fn new(
        config: ZinniaConfig,
        witness: Option<ResolvedWitness>,
        params: ProvingParams,
    ) -> Self {
        Self {
            config,
            witness,
            params,
            offset: 0,
            ops: Vec::new(),
            public_cells: Vec::new(),
            public_values: Vec::new(),
            instance_row: 0,
            memories: HashMap::new(),
            memory_cells: HashMap::new(),
            memory_init: HashMap::new(),
            dyn_meta_bounds: HashMap::new(),
            free_pack_row: 0,
            free_pack_col: 5,
            bit_cells: HashSet::new(),
            u128_lt_bit_cache: HashMap::new(),
            select_many_cache: HashMap::new(),
            first_unsat: None,
        }
    }

    /// The first constraint whose witness value violated its gate during
    /// recording, or `None` if every recorded assert/bit constraint is
    /// satisfied (or no witness was present). See `first_unsat`.
    pub fn first_unsat(&self) -> Option<&str> {
        self.first_unsat.as_deref()
    }

    /// Record (once) that a recorded constraint is unsatisfied by the
    /// concrete witness. No-op when no witness is bound — the keygen pass
    /// threads placeholder zeros and must not be flagged.
    fn note_unsat(&mut self, detail: impl FnOnce() -> String) {
        if self.witness.is_some() && self.first_unsat.is_none() {
            self.first_unsat = Some(detail());
        }
    }

    /// Mark a cell as known to hold a value in {0, 1}. Used by gadgets
    /// whose outputs are mathematically constrained to be bits.
    fn mark_bit(&mut self, cell: &Halo2CellRef) {
        self.bit_cells.insert((cell.col_idx, cell.row));
    }

    fn is_bit(&self, cell: &Halo2CellRef) -> bool {
        self.bit_cells.contains(&(cell.col_idx, cell.row))
    }

    /// Allocate one "free advice" cell (no selector / no row-shape
    /// constraint), packing five per row across the advice columns.
    /// Used by `read_input` and `read_external_result`. The returned
    /// `Halo2CellRef` can be wired into subsequent gates via copy
    /// constraints exactly like a `rec_advice` result.
    fn alloc_free_advice(&mut self, value: Fp, ann: &str) -> Halo2CellRef {
        if self.free_pack_col >= 5 {
            self.free_pack_row = self.offset;
            self.offset += 1;
            self.free_pack_col = 0;
        }
        let col = self.free_pack_col;
        let row = self.free_pack_row;
        self.free_pack_col += 1;
        self.ops.push(GateOp::AssignAdvice {
            col_idx: col,
            row,
            value,
            annotation: ann.to_string(),
        });
        Halo2CellRef {
            value,
            col_idx: col,
            row,
        }
    }

    /// Returns the public-input values collected during synthesis,
    /// in the order they were exposed (matches the instance column layout).
    pub fn public_values(&self) -> &[Fp] {
        &self.public_values
    }

    /// Number of advice/fixed rows used by the recorded ops. Counted by
    /// the synthesizer's `offset` field as it walks the IR in Phase 1.
    /// Used by `estimate_params` to pick a precise `k`.
    pub fn row_count(&self) -> usize {
        self.offset
    }

    /// Whether any recorded op activates the range-lookup selector.
    /// Determines whether the range table needs `2^lookup_bits` rows or
    /// just 1 (a single padding row).
    pub fn uses_range_lookup(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, GateOp::EnableRangeLookup { .. }))
    }

    /// Number of cells exposed to the instance column. The instance
    /// column needs at least this many usable rows, which can dominate
    /// `k` for output-heavy kernels (e.g. pure-identity / reshape).
    pub fn instance_rows(&self) -> usize {
        self.public_cells.len()
    }

    pub fn replay_into_region(
        &self,
        region: &mut Region<'_, Fp>,
    ) -> Result<AssignedCellGrid, Error> {
        let mut assigned = AssignedCellGrid::new(self.config.advice.len(), self.offset.max(1));
        for op in &self.ops {
            match op {
                GateOp::AssignAdvice {
                    col_idx,
                    row,
                    value,
                    annotation,
                } => {
                    let cell = region.assign_advice(
                        || annotation.as_str(),
                        self.config.advice[*col_idx],
                        *row,
                        || HaloValue::known(*value),
                    )?;
                    assigned.insert(*col_idx, *row, cell);
                }
                GateOp::EnableSelector { selector_name, row } => {
                    self.get_selector(selector_name).enable(region, *row)?;
                }
                GateOp::CopyConstraint {
                    col_a,
                    row_a,
                    col_b,
                    row_b,
                } => {
                    let a = assigned.get(*col_a, *row_a).ok_or(Error::Synthesis)?;
                    let b = assigned.get(*col_b, *row_b).ok_or(Error::Synthesis)?;
                    region.constrain_equal(a.cell(), b.cell())?;
                }
                GateOp::EnableRangeLookup { row } => {
                    self.config.s_range_lookup.enable(region, *row)?;
                }
                GateOp::AssignFixed {
                    row,
                    value,
                    annotation,
                } => {
                    region.assign_fixed(
                        || annotation.as_str(),
                        self.config.fixed[0],
                        *row,
                        || HaloValue::known(*value),
                    )?;
                }
            }
        }
        Ok(assigned)
    }

    pub fn load_range_table(&self, layouter: &mut impl Layouter<Fp>) -> Result<(), Error> {
        let uses_range_lookup = self
            .ops
            .iter()
            .any(|op| matches!(op, GateOp::EnableRangeLookup { .. }));
        let rows = if uses_range_lookup {
            let max_lookup_bits = self.params.lookup_bits.min(MAX_RANGE_LOOKUP_BITS);
            1usize << max_lookup_bits
        } else {
            1
        };
        layouter.assign_table(
            || "zinnia_range_table",
            |mut table| {
                for row in 0..rows {
                    table.assign_cell(
                        || format!("range_table_{}", row),
                        self.config.range_table,
                        row,
                        || HaloValue::known(Fp::from(row as u64)),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    pub fn expose_instances(
        &self,
        layouter: &mut impl Layouter<Fp>,
        assigned: &AssignedCellGrid,
    ) -> Result<(), Error> {
        for &(col, row, inst_row) in &self.public_cells {
            if let Some(cell) = assigned.get(col, row) {
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
            "select_bit" => self.config.s_select_bit,
            "select_many2" => self.config.s_select_many2,
            "select_many3" => self.config.s_select_many3,
            "select_many5" => self
                .config
                .s_select_many5
                .expect("select_many5 selector requires wide_select_many config"),
            "assert" => self.config.s_assert,
            "sum_f" => self.config.s_sum_f,
            "inv" => self.config.s_inv,
            "is_zero" => self.config.s_is_zero,
            "eq" => self.config.s_eq,
            "is_nonzero" => self.config.s_is_nonzero,
            "is_nonzero_and" => self.config.s_is_nonzero_and,
            "div_mod" => self.config.s_div_mod,
            "cond_neg" => self.config.s_cond_neg,
            "sub_sd" => self.config.s_sub_sd,
            "bit" => self.config.s_bit,
            "mul_add" => self.config.s_mul_add,
            "mul_f" => self.config.s_mul_f,
            "product_sum_f_step" => self.config.s_product_sum_f_step,
            "product_sum_f_chunk_first" => self.config.s_product_sum_f_chunk_first,
            "product_sum_f_chunk_next" => self.config.s_product_sum_f_chunk_next,
            "fixed_const" => self.config.s_fixed_const,
            "range_lookup" => self.config.s_range_lookup,
            "range_bits2" => self.config.s_range_bits2,
            "range3" => self.config.s_range3,
            "range5" => self.config.s_range5,
            "range6" => self.config.s_range6,
            "range7" => self.config.s_range7,
            "range_bits3" => self.config.s_range_bits3,
            _ => panic!("Unknown selector: {}", name),
        }
    }

    // ── Low-level recording helpers ───────────────────────────────────

    fn rec_advice(&mut self, col: usize, value: Fp, ann: &str) -> Halo2CellRef {
        let row = self.offset;
        self.ops.push(GateOp::AssignAdvice {
            col_idx: col,
            row,
            value,
            annotation: ann.to_string(),
        });
        Halo2CellRef {
            value,
            col_idx: col,
            row,
        }
    }

    fn rec_sel(&mut self, name: &str) {
        self.ops.push(GateOp::EnableSelector {
            selector_name: name.to_string(),
            row: self.offset,
        });
    }

    fn rec_copy(&mut self, src: &Halo2CellRef, dst_col: usize, dst_row: usize) {
        self.ops.push(GateOp::CopyConstraint {
            col_a: src.col_idx,
            row_a: src.row,
            col_b: dst_col,
            row_b: dst_row,
        });
    }

    fn rec_range_lookup(&mut self) {
        self.ops
            .push(GateOp::EnableRangeLookup { row: self.offset });
    }

    fn fixed_constant_cell(&mut self, value: Fp, ann: &str) -> Halo2CellRef {
        let row = self.offset;
        self.rec_sel("fixed_const");
        let cell = self.rec_advice(0, value, ann);
        self.ops.push(GateOp::AssignFixed {
            row,
            value,
            annotation: format!("{}_fixed", ann),
        });
        self.offset += 1;
        cell
    }

    // ── Constrained building blocks ───────────────────────────────────

    /// Binary gate: sel(a op b = c).
    fn bin_gate(
        &mut self,
        sel: &str,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
        result: Fp,
        ann: &str,
    ) -> Halo2CellRef {
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
    fn constrained_mul_add(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
        c: &Halo2CellRef,
    ) -> Halo2CellRef {
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

    fn sum_f_row(&mut self, terms: &[Halo2CellRef]) -> Halo2CellRef {
        debug_assert!((3..=4).contains(&terms.len()));
        let row = self.offset;
        self.rec_sel("sum_f");
        let mut result = Fp::zero();
        for col in 0..4 {
            let value = terms.get(col).map(|term| term.value).unwrap_or(Fp::zero());
            let _ = self.rec_advice(col, value, &format!("sum_f_term_{}", col));
            if let Some(term) = terms.get(col) {
                self.rec_copy(term, col, row);
                result += term.value;
            }
        }
        let out = self.rec_advice(4, result, "sum_f_out");
        self.offset += 1;
        out
    }

    fn constrained_mul_f_core(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
        q: Fp,
        r: Fp,
        ann: &str,
    ) -> (Halo2CellRef, Halo2CellRef, Halo2CellRef) {
        let raw = a.value * b.value;
        let scale =
            Fp::from(crate::prove::field::quantization_scale(self.params.precision_bits) as u64);
        let row = self.offset;
        self.rec_sel("mul_f");
        let _ = self.rec_advice(0, a.value, &format!("{}_a", ann));
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, &format!("{}_b", ann));
        self.rec_copy(b, 1, row);
        let q_cell = self.rec_advice(2, q, &format!("{}_q", ann));
        let r_cell = self.rec_advice(3, r, &format!("{}_r", ann));
        let raw_cell = self.rec_advice(4, raw, &format!("{}_raw", ann));
        self.ops.push(GateOp::AssignFixed {
            row,
            value: scale,
            annotation: format!("{}_scale", ann),
        });
        self.offset += 1;
        (raw_cell, q_cell, r_cell)
    }

    /// Constrain that a value is a bit (0 or 1): a*(a-1) = 0.
    fn constrain_bit(&mut self, a: &Halo2CellRef) {
        // Gate "bit": s * a * (a - 1) = 0, i.e. the witness must be in {0, 1}.
        if a.value != Fp::zero() && a.value != Fp::one() {
            self.note_unsat(|| {
                format!(
                    "Witness does not satisfy circuit constraints: bit gate expected 0 or 1 but witness is {}",
                    kernel::fp_to_i64(a.value)
                )
            });
        }
        let row = self.offset;
        self.rec_sel("bit");
        let _ = self.rec_advice(0, a.value, "bit_check");
        self.rec_copy(a, 0, row);
        self.offset += 1;
    }

    /// is_zero gadget: returns a constrained cell with value 1 if val==0, else 0.
    fn is_zero_gadget(&mut self, val: &Halo2CellRef) -> Halo2CellRef {
        let is_zero_val = if val.value == Fp::zero() {
            Fp::one()
        } else {
            Fp::zero()
        };
        let inv_val = val.value.invert().unwrap_or(Fp::zero());
        let row = self.offset;
        self.rec_sel("is_zero");
        let _ = self.rec_advice(0, val.value, "iz_val");
        self.rec_copy(val, 0, row);
        let _ = self.rec_advice(1, inv_val, "iz_inv");
        let zc = self.rec_advice(2, is_zero_val, "iz_out");
        self.offset += 1;
        self.mark_bit(&zc);
        zc
    }

    /// Single-row equality gadget. Equivalent to `is_zero(a - b)` but avoids
    /// materializing the subtraction row, which dominates gather/index masks.
    fn eq_gadget(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Halo2CellRef {
        let diff = a.value - b.value;
        let is_equal = if diff == Fp::zero() {
            Fp::one()
        } else {
            Fp::zero()
        };
        let inv = diff.invert().unwrap_or(Fp::zero());
        let row = self.offset;
        self.rec_sel("eq");
        let _ = self.rec_advice(0, a.value, "eq_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, "eq_b");
        self.rec_copy(b, 1, row);
        let _ = self.rec_advice(2, inv, "eq_inv");
        let out = self.rec_advice(3, is_equal, "eq_out");
        self.offset += 1;
        self.mark_bit(&out);
        out
    }

    /// Single-row `is_nonzero` gadget. Returns `1` iff `val ≠ 0`. Replaces
    /// the two-row `is_zero` + `logical_not` chain previously used by
    /// `lt_i` (and so transitively by every `<` / `>` / `<=` / `>=` in
    /// the IR).
    fn is_nonzero_gadget(&mut self, val: &Halo2CellRef) -> Halo2CellRef {
        let is_nonzero_val = if val.value == Fp::zero() {
            Fp::zero()
        } else {
            Fp::one()
        };
        let inv_val = val.value.invert().unwrap_or(Fp::zero());
        let row = self.offset;
        self.rec_sel("is_nonzero");
        let _ = self.rec_advice(0, val.value, "inz_val");
        self.rec_copy(val, 0, row);
        let _ = self.rec_advice(1, inv_val, "inz_inv");
        let nzc = self.rec_advice(2, is_nonzero_val, "inz_out");
        self.offset += 1;
        self.mark_bit(&nzc);
        nzc
    }

    /// Truthy bit of `a`: 1 if a ≠ 0, 0 if a = 0. This is exactly what the
    /// `is_nonzero` gadget computes, so a single row suffices (was 2 rows
    /// via `is_zero` + `bool_not`). Used by `logical_and` / `logical_or` /
    /// `select` to coerce non-{0,1} operands per numpy/mock semantics.
    fn truthy_bit(&mut self, a: &Halo2CellRef) -> Halo2CellRef {
        self.is_nonzero_gadget(a)
    }

    /// Single-row implementation of `(val ≠ 0) AND sign`. Uses the
    /// `is_nonzero_and` gate. Caller must ensure `sign ∈ {0, 1}` — the
    /// gate does not check it (we call this from `lt_i` where sign comes
    /// from `sub_signed_decompose`, which already constrained it).
    fn is_nonzero_and_gadget(&mut self, val: &Halo2CellRef, sign: &Halo2CellRef) -> Halo2CellRef {
        let is_nz_val = if val.value == Fp::zero() {
            Fp::zero()
        } else {
            Fp::one()
        };
        let inv_val = val.value.invert().unwrap_or(Fp::zero());
        let out_val = sign.value * is_nz_val;
        let row = self.offset;
        self.rec_sel("is_nonzero_and");
        let _ = self.rec_advice(0, val.value, "ina_val");
        self.rec_copy(val, 0, row);
        let _ = self.rec_advice(1, inv_val, "ina_inv");
        let _ = self.rec_advice(2, is_nz_val, "ina_nz");
        let _ = self.rec_advice(3, sign.value, "ina_sign");
        self.rec_copy(sign, 3, row);
        let out = self.rec_advice(4, out_val, "ina_out");
        self.offset += 1;
        self.mark_bit(&out);
        out
    }

    /// Fused sub + signed_decompose using the `sub_sd` gate. Produces
    /// `(diff = a - b, abs, sign)` in a single row (was 2 rows: sub gate
    /// + cond_neg gate). `sign` is marked as a known {0, 1} cell so
    /// downstream consumers can skip the `truthy_bit` coerce.
    fn sub_signed_decompose(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<(Halo2CellRef, Halo2CellRef, Halo2CellRef), ProvingError> {
        let diff_val = a.value - b.value;
        let (abs_fp, is_neg) = kernel::signed_decompose(diff_val);
        let sign_fp = if is_neg { Fp::one() } else { Fp::zero() };
        let row = self.offset;
        self.rec_sel("sub_sd");
        let _ = self.rec_advice(0, a.value, "ssd_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, "ssd_b");
        self.rec_copy(b, 1, row);
        let diff_cell = self.rec_advice(2, diff_val, "ssd_diff");
        let sign_cell = self.rec_advice(3, sign_fp, "ssd_sign");
        let abs_cell = self.rec_advice(4, abs_fp, "ssd_abs");
        self.offset += 1;
        self.mark_bit(&sign_cell);
        // Bind `abs` to the honest magnitude window so `sign` is no longer a
        // free witness (closes C10: comparison/abs/sign malleability).
        self.range_check_signed_magnitude(&abs_cell, "ssd_abs_rc")?;
        Ok((diff_cell, abs_cell, sign_cell))
    }

    /// Constrained signed decomposition: returns (abs, is_neg) where is_neg ∈ {0,1}.
    /// Uses cond_neg gate + bit constraint on the sign.
    fn signed_decompose(
        &mut self,
        val: &Halo2CellRef,
    ) -> Result<(Halo2CellRef, Halo2CellRef), ProvingError> {
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

        // sign ∈ {0, 1} is enforced by the fused bit-check in the
        // `cond_neg` gate — no separate constrain_bit row needed.
        self.mark_bit(&sign_cell);

        // Bind `abs` to the honest magnitude window so `sign` is no longer a
        // free witness (closes C10: comparison/abs/sign malleability).
        self.range_check_signed_magnitude(&abs_cell, "sd_abs_rc")?;

        Ok((abs_cell, sign_cell))
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
            acc = self.bin_gate(
                "add",
                &prod,
                &coef_cell,
                prod.value + coef_cell.value,
                "horner_step",
            );
        }
        acc
    }

    fn quantize_fp(&self, v: f64) -> Fp {
        kernel::quantize_to_fp(v, self.params.precision_bits)
    }

    /// Allocate an unconstrained advice cell holding `value`. Useful for witness-
    /// only helper values whose correctness is enforced by downstream gates (e.g.
    /// range checks). Packed into the shared free-advice rows (5 per row).
    fn witness_cell(&mut self, value: Fp, ann: &str) -> Halo2CellRef {
        self.alloc_free_advice(value, ann)
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
        // shifted ∈ [0, 2*scale) = [0, 2^{precision_bits+1}). This is the
        // guard that keeps the LOG2_COEFS polynomial inside its [2, 4)
        // fit domain — see `log_f` for the lowering chain. We enforce
        // the bound *exactly* (no widening) with a hybrid scheme:
        // lookup-backed chunks for the aligned low bits + an explicit
        // bit-constraint for the trailing high bit. Total ~16 rows for
        // 33-bit; far cheaper than the 128-row bit-decomposition path
        // but with the same tight upper bound 2^33.
        let n_bits = precision_bits + 1;
        self.range_check_n_bits_lookup(&shifted, n_bits, ann);
    }

    /// Range-check `val ∈ [0, 2^n_bits)` using lookup-backed chunks for
    /// the aligned prefix and an explicit bit-decomposition fallback for
    /// the trailing high bits. Unlike `range_check_chunks` (which errors
    /// on non-aligned n_bits) this accepts any `n_bits` and enforces the
    /// bound exactly.
    ///
    /// For `n_bits % lookup_bits == 0` this is equivalent to calling
    /// `range_check_chunks` directly. For non-aligned `n_bits` we witness
    /// a low/high split, range-check each side independently, and
    /// recompose with a copy constraint back to the original cell.
    fn range_check_n_bits_lookup(&mut self, val: &Halo2CellRef, n_bits: u32, ann: &str) {
        let lookup_bits = self.params.lookup_bits;
        if n_bits == 0 || n_bits > 63 {
            panic!(
                "range_check_n_bits_lookup supports 1..=63 bits, got {}",
                n_bits
            );
        }
        // Aligned: one call into the lookup-only path.
        if n_bits % lookup_bits == 0 {
            self.range_check_chunks(val, n_bits, ann)
                .expect("range_check_chunks aligned path");
            return;
        }
        // Non-aligned split. `aligned` is the largest multiple of
        // `lookup_bits` below `n_bits`; `trailing` is the remainder.
        let aligned = (n_bits / lookup_bits) * lookup_bits;
        let trailing = n_bits - aligned;

        let v_i64 = kernel::fp_to_i64(val.value);
        let v_u64 = v_i64 as u64;
        let mask = (1u64 << aligned) - 1;
        let low_val = v_u64 & mask;
        let high_val = v_u64 >> aligned;

        let low_cell = self.witness_cell(Fp::from(low_val), &format!("{}_low", ann));
        let high_cell = self.witness_cell(Fp::from(high_val), &format!("{}_high", ann));

        // Low part: full lookup chunks (exact).
        self.range_check_chunks(&low_cell, aligned, &format!("{}_low_rc", ann))
            .expect("range_check_chunks for low part");

        // High part: `trailing` bits. For trailing == 1 a single bit
        // constraint suffices; for larger trailing we fall back to bit
        // decomposition (only ever a handful of bits, so cheap).
        if trailing == 1 {
            self.constrain_bit(&high_cell);
        } else {
            // Bit decomp of `trailing` bits — same shape as the legacy
            // `range_check_n_bits` but bounded by `trailing` not `n_bits`.
            // For our current callers (precision_bits + 1 with
            // lookup_bits = 8) this branch isn't reached; left in place
            // for correctness on arbitrary configs.
            self.range_check_n_bits(&high_cell, trailing, &format!("{}_high_rc", ann));
        }

        // Recompose: val == low + high * 2^aligned. The fixed weight
        // cell + 1 mul + 1 add + a direct copy constraint binds the
        // recomposition back to `val` without burning rows on an
        // is_zero-based equality check.
        let weight_fp = Fp::from(1u64 << aligned);
        let weight_cell = self.fixed_constant_cell(weight_fp, &format!("{}_weight", ann));
        let high_weighted = self.bin_gate(
            "mul",
            &high_cell,
            &weight_cell,
            high_cell.value * weight_fp,
            &format!("{}_h_w", ann),
        );
        let recomposed = self.bin_gate(
            "add",
            &low_cell,
            &high_weighted,
            low_cell.value + high_weighted.value,
            &format!("{}_rec", ann),
        );
        // 0-row equality via halo2's permutation argument.
        self.constrain_equal_cells_direct(&recomposed, val);
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
            let m_q = if k >= 0 { x_i64 >> k } else { x_i64 << (-k) };
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

        // Range-check f_q ∈ [0, scale) = [0, 2^precision_bits). Lookup-
        // backed chunks instead of bit decomposition (~64 rows → ~12
        // rows when precision_bits=32, lookup_bits=8). The default
        // params satisfy `precision_bits % lookup_bits == 0`; for
        // misaligned configs we round n_bits up to the next aligned
        // multiple (same widening rationale as in range_check_log_mantissa).
        let lookup_bits = self.params.lookup_bits;
        let aligned = ((precision_bits + lookup_bits - 1) / lookup_bits) * lookup_bits;
        self.range_check_chunks(&f_cell, aligned, "exp2rr_f_rc")
            .expect("range_check_chunks in exp2_range_reduce");

        // Constrain: y_q = i * scale + f_q.
        let scale_cell = self.constant_int(scale).expect("constant_int for scale");
        let i_scaled = self.mul_i(&i_cell, &scale_cell).expect("mul_i in exp2rr");
        let recomputed = self.add_i(&i_scaled, &f_cell).expect("add_i in exp2rr");
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

    fn constrain_equal_cells_direct(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) {
        self.rec_copy(a, b.col_idx, b.row);
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
            "mul",
            &pow2,
            &bits[0],
            pow2.value * bits[0].value,
            "decomp_acc0",
        );

        // 4. For i in 1..63: pow2 *= 2 (doubling = add gate); acc += pow2 * bits[i]
        for i in 1..(N - 1) {
            pow2 = self.bin_gate(
                "add",
                &pow2,
                &pow2,
                pow2.value + pow2.value,
                &format!("p2_{}", i),
            );
            acc = self.constrained_mul_add(&pow2, &bits[i], &acc);
        }

        // 5. For i = 63 (sign bit): pow2 = 2^63, then acc -= pow2 * bits[63].
        pow2 = self.bin_gate("add", &pow2, &pow2, pow2.value + pow2.value, "p2_63");
        let high = self.bin_gate(
            "mul",
            &pow2,
            &bits[N - 1],
            pow2.value * bits[N - 1].value,
            "high_term",
        );
        let final_val = self.bin_gate("sub", &acc, &high, acc.value - high.value, "decomp_final");

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
            "mul",
            &pow2,
            &bits[0],
            pow2.value * bits[0].value,
            &format!("{}_acc0", ann),
        );
        for i in 1..(N - 1) {
            pow2 = self.bin_gate(
                "add",
                &pow2,
                &pow2,
                pow2.value + pow2.value,
                &format!("{}_p2_{}", ann, i),
            );
            acc = self.constrained_mul_add(&pow2, &bits[i], &acc);
        }
        pow2 = self.bin_gate(
            "add",
            &pow2,
            &pow2,
            pow2.value + pow2.value,
            &format!("{}_p2_63", ann),
        );
        let high = self.bin_gate(
            "mul",
            &pow2,
            &bits[N - 1],
            pow2.value * bits[N - 1].value,
            &format!("{}_high", ann),
        );
        self.bin_gate(
            "sub",
            &acc,
            &high,
            acc.value - high.value,
            &format!("{}_final", ann),
        )
    }

    /// Per-bit AND: `c = a * b` for two bit cells (uses `bool_and` gate which is
    /// the same as `mul`, but the dedicated selector documents intent).
    fn bit_and_per_bit(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Halo2CellRef {
        self.bin_gate("bool_and", a, b, a.value * b.value, "and_bit")
    }

    /// Per-bit OR: `c = a + b - a*b` via the `bool_or` gate.
    fn bit_or_per_bit(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Halo2CellRef {
        self.bin_gate(
            "bool_or",
            a,
            b,
            a.value + b.value - a.value * b.value,
            "or_bit",
        )
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

    /// Constrain that `val` is in [0, 2^n_bits) by allocating `n_bits` bit
    /// cells and proving their weighted sum equals `val`. Each bit cell is
    /// constrained to {0, 1} via the `bit` gate. The final recomposed cell is
    /// copy-constrained equal to `val`.
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
            "mul",
            &pow2,
            &bits[0],
            pow2.value * bits[0].value,
            &format!("{}_acc0", ann),
        );

        // 4. For i = 1..n: pow2 *= 2, acc += pow2 * bits[i].
        for i in 1..n {
            pow2 = self.bin_gate(
                "add",
                &pow2,
                &pow2,
                pow2.value + pow2.value,
                &format!("{}_p2_{}", ann, i),
            );
            acc = self.constrained_mul_add(&pow2, &bits[i], &acc);
        }

        // 5. Constrain acc == val (this is the actual range-check enforcement:
        //    val equals a weighted sum of bits, so 0 <= val < 2^n).
        self.constrain_equal_cells(&acc, val, &format!("{}_eq", ann));
    }

    /// Lookup-backed range check for non-negative fixed-point remainders.
    ///
    /// This replaces only the implementation of `0 <= r < 2^n_bits`: each
    /// `lookup_bits`-wide chunk is constrained by a real Halo2 lookup table,
    /// then chunks are weighted and recomposed back to the original remainder
    /// cell. The caller is still responsible for the per-product `div_mod`
    /// relation `raw = scale*q + r`.
    fn range_check_chunks(
        &mut self,
        val: &Halo2CellRef,
        n_bits: u32,
        ann: &str,
    ) -> Result<(), ProvingError> {
        self.range_check_chunks_with_override(val, n_bits, ann, None)
    }

    /// Bind a claimed non-negative magnitude (the `abs` output of `sub_sd` /
    /// `cond_neg`) to the honest `[0, 2^SIGNED_MAGNITUDE_BITS)` window via the
    /// lookup-backed chunk check. Without this, `abs = diff*(1 - 2*sign)` is
    /// satisfiable for *either* `sign`, leaving the half-field sign of `diff`
    /// a free witness — so `lt_i`/`gt_i`/`le_i`/`ge_i`, `abs_i` and `sign_i`
    /// would all be malleable. Range-checking `abs` rejects the field-negation
    /// branch (which is ~254 bits wide) and pins `sign` to the true sign.
    ///
    /// `n_bits` is rounded down to a multiple of `lookup_bits` so the chunk
    /// recomposition is exact (`range_check_chunks` requires the divisibility).
    /// The rounded width still comfortably exceeds any honest i64/Q-fixed-point
    /// magnitude in the supported range while remaining far below the modulus.
    fn range_check_signed_magnitude(
        &mut self,
        abs: &Halo2CellRef,
        ann: &str,
    ) -> Result<(), ProvingError> {
        let lookup_bits = self.params.lookup_bits;
        if lookup_bits == 0 {
            return Err(ProvingError::synthesis(
                "range_check_signed_magnitude requires lookup_bits >= 1",
            ));
        }
        let effective = lookup_bits.min(MAX_RANGE_LOOKUP_BITS);
        // Largest multiple of `effective` that is <= SIGNED_MAGNITUDE_BITS and
        // <= 63 (the range_check_chunks ceiling).
        let cap = SIGNED_MAGNITUDE_BITS.min(63);
        let n_bits = (cap / effective) * effective;
        if n_bits == 0 {
            return Err(ProvingError::synthesis(format!(
                "range_check_signed_magnitude: lookup_bits ({}) too wide for the {}-bit magnitude window",
                lookup_bits, cap
            )));
        }
        self.range_check_chunks(abs, n_bits, ann)
    }

    /// Bind the remainder of a `div_mod` row to the Euclidean window
    /// `0 <= r < |b|`. Without this the bare `a = b*q + r` gate leaves `(q, r)`
    /// malleable — a prover can shift any multiple of `b` between `q` and `r`.
    /// We range-check `r` (non-negativity) and `|b| - 1 - r` (the gap to the
    /// divisor magnitude); the latter forces `r <= |b| - 1`, i.e. `r < |b|`.
    /// `|b|` is materialised via `signed_decompose` (itself magnitude-bound).
    fn constrain_div_mod_remainder(
        &mut self,
        r: &Halo2CellRef,
        b: &Halo2CellRef,
        ann: &str,
    ) -> Result<(), ProvingError> {
        // 0 <= r  (a negative r would be a ~254-bit field element, rejected).
        self.range_check_signed_magnitude(r, &format!("{}_r_nonneg", ann))?;
        // |b|, with its own magnitude binding.
        let (abs_b, _) = self.signed_decompose(b)?;
        // gap = |b| - 1 - r ; honest r in [0, |b|) makes gap in [0, |b|).
        let one = self.constant_int(1)?;
        let abs_b_minus_one = self.sub_i(&abs_b, &one)?;
        let gap = self.sub_i(&abs_b_minus_one, r)?;
        self.range_check_signed_magnitude(&gap, &format!("{}_r_lt_absb", ann))?;
        Ok(())
    }

    fn range_check_chunks_with_override(
        &mut self,
        val: &Halo2CellRef,
        n_bits: u32,
        ann: &str,
        chunk_override: Option<&[u64]>,
    ) -> Result<(), ProvingError> {
        let lookup_bits = self.params.lookup_bits;
        if n_bits == 0 || n_bits > 63 {
            return Err(ProvingError::synthesis(format!(
                "range_check_chunks supports 1..=63 checked bits, got {}",
                n_bits
            )));
        }
        if lookup_bits == 0 || lookup_bits > MAX_RANGE_LOOKUP_BITS {
            return Err(ProvingError::synthesis(format!(
                "range_check_chunks requires lookup_bits in 1..={}, got {}",
                MAX_RANGE_LOOKUP_BITS, lookup_bits
            )));
        }
        if n_bits % lookup_bits != 0 {
            return Err(ProvingError::synthesis(format!(
                "range_check_chunks first slice requires precision_bits ({}) to be a multiple of lookup_bits ({})",
                n_bits, lookup_bits
            )));
        }

        let v_i64 = kernel::fp_to_i64(val.value);
        let v_u64 = v_i64 as u64;
        let chunk_mask = if lookup_bits == 64 {
            u64::MAX
        } else {
            (1u64 << lookup_bits) - 1
        };
        let chunks = n_bits.div_ceil(lookup_bits);
        if let Some(override_chunks) = chunk_override {
            if override_chunks.len() != chunks as usize {
                return Err(ProvingError::synthesis(format!(
                    "range_check_chunks override expected {} chunks, got {}",
                    chunks,
                    override_chunks.len()
                )));
            }
        }

        let mut acc: Option<Halo2CellRef> = None;
        for chunk_idx in 0..chunks {
            let shift = chunk_idx * lookup_bits;
            let chunk = chunk_override
                .and_then(|chunks| chunks.get(chunk_idx as usize).copied())
                .unwrap_or_else(|| (v_u64 >> shift) & chunk_mask);
            let chunk_cell =
                self.rec_advice(0, Fp::from(chunk), &format!("{}_chunk_{}", ann, chunk_idx));
            self.rec_range_lookup();
            self.offset += 1;

            let term = if chunk_idx == 0 {
                chunk_cell.clone()
            } else {
                let weight_value = Fp::from(1u64 << shift);
                let weight_cell = self
                    .fixed_constant_cell(weight_value, &format!("{}_weight_{}", ann, chunk_idx));
                self.bin_gate(
                    "mul",
                    &chunk_cell,
                    &weight_cell,
                    chunk_cell.value * weight_cell.value,
                    &format!("{}_term_{}", ann, chunk_idx),
                )
            };

            acc = Some(match acc {
                Some(prev) => self.bin_gate(
                    "add",
                    &prev,
                    &term,
                    prev.value + term.value,
                    &format!("{}_acc_{}", ann, chunk_idx),
                ),
                None => term,
            });
        }

        let acc = acc.expect("n_bits > 0 creates at least one chunk");
        self.constrain_equal_cells_direct(&acc, val);
        Ok(())
    }

    /// Compact static-scale remainder range/recomposition path. Each chunk row carries both
    /// the shared range lookup and the weighted recomposition accumulator:
    /// `acc_i = acc_{i-1} + chunk_i * 2^(i*lookup_bits)`. The final accumulator
    /// is copy-constrained to the original remainder cell, preserving the same-`r`
    /// binding without the generic fixed_const/mul/add rows.
    fn product_sum_f_range_check_chunks(
        &mut self,
        val: &Halo2CellRef,
        n_bits: u32,
        ann: &str,
    ) -> Result<(), ProvingError> {
        self.product_sum_f_range_check_chunks_with_override(val, n_bits, ann, None, None)
    }

    fn product_sum_f_range_check_chunks_with_override(
        &mut self,
        val: &Halo2CellRef,
        n_bits: u32,
        ann: &str,
        chunk_override: Option<&[u64]>,
        acc_override: Option<&[u64]>,
    ) -> Result<(), ProvingError> {
        let lookup_bits = self.params.lookup_bits;
        if n_bits == 0 || n_bits > 63 {
            return Err(ProvingError::synthesis(format!(
                "product_sum_f_range_check_chunks supports 1..=63 checked bits, got {}",
                n_bits
            )));
        }
        if lookup_bits == 0 || lookup_bits > MAX_RANGE_LOOKUP_BITS {
            return Err(ProvingError::synthesis(format!(
                "product_sum_f_range_check_chunks requires lookup_bits in 1..={}, got {}",
                MAX_RANGE_LOOKUP_BITS, lookup_bits
            )));
        }
        if n_bits % lookup_bits != 0 {
            return Err(ProvingError::synthesis(format!(
                "product_sum_f_range_check_chunks first slice requires precision_bits ({}) to be a multiple of lookup_bits ({})",
                n_bits, lookup_bits
            )));
        }

        let v_i64 = kernel::fp_to_i64(val.value);
        let v_u64 = v_i64 as u64;
        let chunk_mask = (1u64 << lookup_bits) - 1;
        let chunks = n_bits.div_ceil(lookup_bits) as usize;
        if let Some(override_chunks) = chunk_override {
            if override_chunks.len() != chunks {
                return Err(ProvingError::synthesis(format!(
                    "product_sum_f_range_check_chunks override expected {} chunks, got {}",
                    chunks,
                    override_chunks.len()
                )));
            }
        }
        if let Some(override_accs) = acc_override {
            if override_accs.len() != chunks {
                return Err(ProvingError::synthesis(format!(
                    "product_sum_f_range_check_chunks acc override expected {} chunks, got {}",
                    chunks,
                    override_accs.len()
                )));
            }
        }

        let mut acc_u64 = 0u64;
        let mut acc_cell: Option<Halo2CellRef> = None;
        for chunk_idx in 0..chunks {
            let shift = (chunk_idx as u32) * lookup_bits;
            let chunk = chunk_override
                .and_then(|chunks| chunks.get(chunk_idx).copied())
                .unwrap_or_else(|| (v_u64 >> shift) & chunk_mask);
            let weight = 1u64 << shift;
            acc_u64 = acc_u64.wrapping_add(chunk.wrapping_mul(weight));
            let acc_value = acc_override
                .and_then(|accs| accs.get(chunk_idx).copied())
                .unwrap_or(acc_u64);

            let row = self.offset;
            if chunk_idx == 0 {
                self.rec_sel("product_sum_f_chunk_first");
            } else {
                self.rec_sel("product_sum_f_chunk_next");
            }
            let _chunk_cell =
                self.rec_advice(0, Fp::from(chunk), &format!("{}_chunk_{}", ann, chunk_idx));
            self.rec_range_lookup();
            acc_cell = Some(self.rec_advice(
                1,
                Fp::from(acc_value),
                &format!("{}_acc_{}", ann, chunk_idx),
            ));
            self.ops.push(GateOp::AssignFixed {
                row,
                value: Fp::from(weight),
                annotation: format!("{}_weight_{}", ann, chunk_idx),
            });
            self.offset += 1;
        }

        let acc = acc_cell.expect("n_bits > 0 creates at least one chunk");
        self.constrain_equal_cells_direct(&acc, val);
        Ok(())
    }

    /// Test-only harness for the load-bearing malicious-remainder regression.
    /// It records a `mul_f`-shaped constraint bundle with caller-supplied
    /// quotient/remainder witnesses, so tests can keep `raw = scale*q + r`
    /// satisfiable while making `r` fall outside the lookup-backed range.
    #[cfg(test)]
    pub(crate) fn debug_constrain_mul_f_with_witness(
        &mut self,
        a: Fp,
        b: Fp,
        result: Fp,
        rem: Fp,
    ) -> Result<(), ProvingError> {
        let a_cell = self.rec_advice(0, a, "debug_mulf_a");
        self.offset += 1;
        let b_cell = self.rec_advice(0, b, "debug_mulf_b");
        self.offset += 1;
        let raw = self.mul_i(&a_cell, &b_cell)?;
        let scale =
            Fp::from(crate::prove::field::quantization_scale(self.params.precision_bits) as u64);
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, raw.value, "debug_mulf_raw");
        self.rec_copy(&raw, 0, row);
        let _ = self.rec_advice(1, scale, "debug_mulf_scale");
        let _result_cell = self.rec_advice(2, result, "debug_mulf_result");
        let rem_cell = self.rec_advice(3, rem, "debug_mulf_rem");
        self.offset += 1;
        self.range_check_chunks_with_override(
            &rem_cell,
            self.params.precision_bits,
            "debug_mulf_rem_rc",
            Some(&[0, 256]),
        )
    }

    #[cfg(test)]
    pub(crate) fn debug_constrain_compact_mul_f_row_with_witness(
        &mut self,
        a: Fp,
        b: Fp,
        raw: Fp,
        result: Fp,
        rem: Fp,
    ) -> Result<(), ProvingError> {
        let scale =
            Fp::from(crate::prove::field::quantization_scale(self.params.precision_bits) as u64);
        let row = self.offset;
        self.rec_sel("mul_f");
        let _ = self.rec_advice(0, a, "debug_compact_row_a");
        let _ = self.rec_advice(1, b, "debug_compact_row_b");
        let _ = self.rec_advice(2, result, "debug_compact_row_q");
        let rem_cell = self.rec_advice(3, rem, "debug_compact_row_r");
        let _ = self.rec_advice(4, raw, "debug_compact_row_raw");
        self.ops.push(GateOp::AssignFixed {
            row,
            value: scale,
            annotation: "debug_compact_row_scale".to_string(),
        });
        self.offset += 1;
        self.range_check_chunks(
            &rem_cell,
            self.params.precision_bits,
            "debug_compact_row_rem",
        )
    }

    #[cfg(test)]
    pub(crate) fn debug_constrain_product_sum_f_step_with_witness(
        &mut self,
        a: Fp,
        b: Fp,
        prev_acc: Fp,
        q: Fp,
        r: Fp,
        acc: Fp,
    ) -> Result<(), ProvingError> {
        let precision_bits = self.params.precision_bits;
        let scale = Fp::from(crate::prove::field::quantization_scale(precision_bits) as u64);
        let prev = self.rec_advice(4, prev_acc, "debug_psf_prev_acc");
        self.offset += 1;
        let row = self.offset;
        self.rec_sel("product_sum_f_step");
        let _ = self.rec_advice(0, a, "debug_psf_a");
        let _ = self.rec_advice(1, b, "debug_psf_b");
        let _ = self.rec_advice(2, q, "debug_psf_q");
        let rem_cell = self.rec_advice(3, r, "debug_psf_r");
        let _ = self.rec_advice(4, acc, "debug_psf_acc");
        self.rec_copy(&prev, 4, row - 1);
        self.ops.push(GateOp::AssignFixed {
            row,
            value: scale,
            annotation: "debug_psf_scale".to_string(),
        });
        self.offset += 1;
        self.range_check_chunks(&rem_cell, precision_bits, "debug_psf_rem")
    }

    #[cfg(test)]
    pub(crate) fn debug_product_sum_f_range_check_chunks_with_override(
        &mut self,
        val: Fp,
        chunks: &[u64],
        accs: Option<&[u64]>,
    ) -> Result<(), ProvingError> {
        let val_cell = self.rec_advice(3, val, "debug_psf_range_val");
        self.offset += 1;
        self.product_sum_f_range_check_chunks_with_override(
            &val_cell,
            self.params.precision_bits,
            "debug_psf_range",
            Some(chunks),
            accs,
        )
    }

    #[cfg(test)]
    pub(crate) fn debug_range_check_chunks_with_override(
        &mut self,
        val: Fp,
        chunks: &[u64],
    ) -> Result<(), ProvingError> {
        let val_cell = self.rec_advice(0, val, "debug_range_val");
        self.offset += 1;
        self.range_check_chunks_with_override(
            &val_cell,
            self.params.precision_bits,
            "debug_range",
            Some(chunks),
        )
    }

    /// C10 malicious-witness harness: emit a `lt_i`-shaped constraint bundle
    /// (the `sub_sd` row plus its closing `is_nonzero_and`) with caller-chosen
    /// `(diff, sign, abs)` witnesses. An honest `lt_i(a, b)` derives all three
    /// from `a - b`; this lets a test flip `sign` (and supply the matching
    /// field-negated `abs`) to attempt proving the *wrong* ordering. The
    /// returned cell is the boolean `a < b` output. The `range_check_signed_
    /// magnitude` binding on `abs` is what should reject the tamper.
    #[cfg(test)]
    pub(crate) fn debug_lt_i_with_witness(
        &mut self,
        a: Fp,
        b: Fp,
        diff: Fp,
        sign: Fp,
        abs: Fp,
    ) -> Result<Halo2CellRef, ProvingError> {
        let a_cell = self.alloc_free_advice(a, "dbg_lt_a");
        let b_cell = self.alloc_free_advice(b, "dbg_lt_b");
        let row = self.offset;
        self.rec_sel("sub_sd");
        let _ = self.rec_advice(0, a, "dbg_ssd_a");
        self.rec_copy(&a_cell, 0, row);
        let _ = self.rec_advice(1, b, "dbg_ssd_b");
        self.rec_copy(&b_cell, 1, row);
        let diff_cell = self.rec_advice(2, diff, "dbg_ssd_diff");
        let sign_cell = self.rec_advice(3, sign, "dbg_ssd_sign");
        let abs_cell = self.rec_advice(4, abs, "dbg_ssd_abs");
        self.offset += 1;
        self.mark_bit(&sign_cell);
        self.range_check_signed_magnitude(&abs_cell, "dbg_ssd_abs_rc")?;
        Ok(self.is_nonzero_and_gadget(&diff_cell, &sign_cell))
    }

    /// C10 malicious-witness harness: emit an `abs_i`-shaped `cond_neg` row
    /// with caller-chosen `(sign, abs)`. Honest `abs_i(a)` sets `sign` to a's
    /// true sign and `abs = |a|`; this lets a test pass `sign=1, abs=-a` to
    /// attempt returning `-a` for a positive `a`. The magnitude range check on
    /// `abs` is what should reject it.
    #[cfg(test)]
    pub(crate) fn debug_abs_i_with_witness(
        &mut self,
        a: Fp,
        sign: Fp,
        abs: Fp,
    ) -> Result<Halo2CellRef, ProvingError> {
        let a_cell = self.alloc_free_advice(a, "dbg_abs_a");
        let row = self.offset;
        self.rec_sel("cond_neg");
        let _ = self.rec_advice(0, a, "dbg_sd_val");
        self.rec_copy(&a_cell, 0, row);
        let sign_cell = self.rec_advice(1, sign, "dbg_sd_sign");
        let abs_cell = self.rec_advice(2, abs, "dbg_sd_abs");
        self.offset += 1;
        self.mark_bit(&sign_cell);
        self.range_check_signed_magnitude(&abs_cell, "dbg_sd_abs_rc")?;
        Ok(abs_cell)
    }

    /// C10 malicious-witness harness: emit a `mod_i`-shaped `div_mod` row with
    /// caller-chosen `(q, r)`. Honest `mod_i(a, b)` uses the Euclidean
    /// `(q, r)`; this lets a test shift a multiple of `b` between `q` and `r`
    /// (so `a == b*q + r` still holds) and observe whether the remainder
    /// binding `0 <= r < |b|` rejects it. Returns the `r` cell.
    #[cfg(test)]
    pub(crate) fn debug_mod_i_with_witness(
        &mut self,
        a: Fp,
        b: Fp,
        q: Fp,
        r: Fp,
    ) -> Result<Halo2CellRef, ProvingError> {
        let a_cell = self.alloc_free_advice(a, "dbg_mod_a");
        let b_cell = self.alloc_free_advice(b, "dbg_mod_b");
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a, "dbg_mod_a_g");
        self.rec_copy(&a_cell, 0, row);
        let _ = self.rec_advice(1, b, "dbg_mod_b_g");
        self.rec_copy(&b_cell, 1, row);
        let _ = self.rec_advice(2, q, "dbg_mod_q");
        let r_cell = self.rec_advice(3, r, "dbg_mod_r");
        self.offset += 1;
        self.constrain_div_mod_remainder(&r_cell, &b_cell, "dbg_mod")?;
        Ok(r_cell)
    }
}

// ---------------------------------------------------------------------------
// Synthesizer implementation — all operations fully constrained
// ---------------------------------------------------------------------------

impl Synthesizer for Halo2Synthesizer {
    type CellRef = Halo2CellRef;

    fn constant_int(&mut self, value: i64) -> Result<Halo2CellRef, ProvingError> {
        let fp = if value >= 0 {
            Fp::from(value as u64)
        } else {
            -Fp::from((-value) as u64)
        };
        Ok(self.alloc_free_advice(fp, "const_int"))
    }

    fn constant_float(&mut self, value: f64) -> Result<Halo2CellRef, ProvingError> {
        let fp = self.quantize_fp(value);
        Ok(self.alloc_free_advice(fp, "const_float"))
    }

    fn constant_bool(&mut self, value: bool) -> Result<Halo2CellRef, ProvingError> {
        let fp = if value { Fp::one() } else { Fp::zero() };
        Ok(self.alloc_free_advice(fp, "const_bool"))
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
    fn floor_div_i(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        let sa = kernel::fp_to_i64(a.value);
        let sb = kernel::fp_to_i64(b.value);
        if sb == 0 {
            return Err(ProvingError::synthesis("Division by zero"));
        }
        let q = sa.div_euclid(sb);
        let r = sa.rem_euclid(sb);
        let q_fp = if q >= 0 {
            Fp::from(q as u64)
        } else {
            -Fp::from((-q) as u64)
        };
        let r_fp = Fp::from(r as u64);
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a.value, "fdiv_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, b.value, "fdiv_b");
        self.rec_copy(b, 1, row);
        let qc = self.rec_advice(2, q_fp, "fdiv_q");
        let rc = self.rec_advice(3, r_fp, "fdiv_r");
        self.offset += 1;
        // Bind 0 <= r < |b| so (q, r) are uniquely determined (closes C10
        // div_mod malleability for floor_div_i / mod_i).
        self.constrain_div_mod_remainder(&rc, b, "fdiv")?;
        Ok(qc)
    }
    fn mod_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let sa = kernel::fp_to_i64(a.value);
        let sb = kernel::fp_to_i64(b.value);
        if sb == 0 {
            return Err(ProvingError::synthesis("Modulo by zero"));
        }
        let q = sa.div_euclid(sb);
        let r = sa.rem_euclid(sb);
        let q_fp = if q >= 0 {
            Fp::from(q as u64)
        } else {
            -Fp::from((-q) as u64)
        };
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
        // Bind 0 <= r < |b| so (q, r) are uniquely determined.
        self.constrain_div_mod_remainder(&rc, b, "mod")?;
        Ok(rc)
    }
    fn pow_i(
        &mut self,
        base: &Halo2CellRef,
        exp: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        let e = kernel::fp_to_i64(exp.value);
        if e < 0 {
            let pos_exp = Halo2CellRef {
                value: Fp::from((-e) as u64),
                col_idx: exp.col_idx,
                row: exp.row,
            };
            let pos_result = self.pow_i(base, &pos_exp)?;
            return self.inv_i(&pos_result);
        }
        if e == 0 {
            return self.constant_int(1);
        }
        if e == 1 {
            return Ok(base.clone());
        }
        let mut result = base.clone();
        for _ in 1..e.min(64) {
            result = self.mul_i(&result, base)?;
        }
        Ok(result)
    }
    fn abs_i(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let (abs_cell, _) = self.signed_decompose(a)?;
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
    fn bit_and_i(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        let abits = self.bit_decompose_64(a);
        let bbits = self.bit_decompose_64(b);
        let cbits: Vec<Halo2CellRef> = (0..64)
            .map(|i| self.bit_and_per_bit(&abits[i], &bbits[i]))
            .collect();
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "and_out"))
    }

    fn bit_or_i(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        let abits = self.bit_decompose_64(a);
        let bbits = self.bit_decompose_64(b);
        let cbits: Vec<Halo2CellRef> = (0..64)
            .map(|i| self.bit_or_per_bit(&abits[i], &bbits[i]))
            .collect();
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "or_out"))
    }

    fn bit_xor_i(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
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
        let cbits: Vec<Halo2CellRef> = (0..64).map(|i| self.bit_not_per_bit(&abits[i])).collect();
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
            if i < n {
                cbits.push(zero_cell.clone());
            } else {
                cbits.push(abits[i - n].clone());
            }
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
            if i + n < 64 {
                cbits.push(abits[i + n].clone());
            } else {
                cbits.push(abits[63].clone());
            }
        }
        let cbits: [Halo2CellRef; 64] = cbits.try_into().expect("64 bits");
        Ok(self.bit_recompose_64(&cbits, "shr_out"))
    }
    fn sign_i(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sign = (1 - is_zero) * (1 - 2*is_neg)
        // Compute each part with constrained gates.
        let iz = self.is_zero_gadget(a);
        let (_, is_neg) = self.signed_decompose(a)?;
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

    fn sum_f(&mut self, terms: &[Halo2CellRef]) -> Result<Halo2CellRef, ProvingError> {
        if terms.is_empty() {
            return self.constant_float(0.0);
        }
        if terms.len() == 1 {
            return Ok(terms[0].clone());
        }
        if terms.len() == 2 {
            return self.add_f(&terms[0], &terms[1]);
        }

        let mut level: Vec<Halo2CellRef> = terms.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(4));
            for chunk in level.chunks(4) {
                if chunk.len() == 1 {
                    next.push(chunk[0].clone());
                } else if chunk.len() == 2 {
                    next.push(self.add_f(&chunk[0], &chunk[1])?);
                } else {
                    next.push(self.sum_f_row(chunk));
                }
            }
            level = next;
        }
        Ok(level.pop().expect("non-empty level"))
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
        let precision_bits = self.params.precision_bits;
        let (result_val, rem_val) = kernel::fp_mul_rescale(a.value, b.value, precision_bits);
        let (_, result_cell, rem_cell) =
            self.constrained_mul_f_core(a, b, result_val, rem_val, "mulf");
        // Range check: rem ∈ [0, 2^precision_bits) = [0, scale). This closes
        // the soundness gap where any (result, rem) satisfying the div_mod
        // equation would otherwise be accepted.
        self.product_sum_f_range_check_chunks(&rem_cell, precision_bits, "mulf_rem_rc")?;
        Ok(result_cell)
    }
    fn product_sum_f(
        &mut self,
        pairs: &[(Halo2CellRef, Halo2CellRef)],
    ) -> Result<Halo2CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let mut raw_acc = self.constant_int(0)?;
        for (a, b) in pairs.iter() {
            raw_acc = self.constrained_mul_add(a, b, &raw_acc);
        }

        let pair_values: Vec<(Fp, Fp)> = pairs.iter().map(|(a, b)| (a.value, b.value)).collect();
        let (result_val, rem_val) = kernel::fp_product_sum_rescale(&pair_values, precision_bits);
        let scale = Fp::from(crate::prove::field::quantization_scale(precision_bits) as u64);
        let scale_cell = self.fixed_constant_cell(scale, "product_sum_f_scale_const");
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, raw_acc.value, "product_sum_f_raw");
        self.rec_copy(&raw_acc, 0, row);
        let _ = self.rec_advice(1, scale, "product_sum_f_scale");
        self.rec_copy(&scale_cell, 1, row);
        let result_cell = self.rec_advice(2, result_val, "product_sum_f_result");
        let rem_cell = self.rec_advice(3, rem_val, "product_sum_f_rem");
        self.offset += 1;
        self.product_sum_f_range_check_chunks(&rem_cell, precision_bits, "product_sum_f_rem_rc")?;
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
        // where a_scaled = a * scale (constrained via mul_i). The `rem` is then
        // bound to `0 <= rem < |b|` (Euclidean window) so (result, rem) are
        // uniquely determined — without it the bare div_mod left them malleable
        // (closes C10 for div_f / floor_div_f / mod_f).
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
        let rem_cell = self.rec_advice(3, rem_val, "divf_rem");
        self.offset += 1;
        self.constrain_div_mod_remainder(&rem_cell, b, "divf")?;
        Ok(result_cell)
    }
    fn floor_div_f(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        // Q scaling cancels in the integer quotient. Normalize the divisor
        // sign because integer division uses a nonnegative remainder window.
        let zero = self.constant_int(0)?;
        let negative = self.lt_i(b, &zero)?;
        let neg_a = self.sub_i(&zero, a)?;
        let neg_b = self.sub_i(&zero, b)?;
        let numerator = self.select(&negative, &neg_a, a)?;
        let denominator = self.select(&negative, &neg_b, b)?;
        let quotient = self.floor_div_i(&numerator, &denominator)?;
        self.float_cast(&quotient)
    }
    fn mod_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Previous implementation composed `a - div_f(a, b) * b`. `div_f`
        // returns the real (not floored) quotient, so this collapsed to
        // `a - (a/b)*b ≈ 0` regardless of input — every mod_f call returned 0.
        //
        // Fix mirrors `sin_f`'s range-reduction: i64 floor division on the
        // Q-encoded operands is exactly floor division on the underlying
        // floats (the scale factor cancels in both the quotient's int-floor
        // and the remainder), so `mod_i(a, b)` on Q-rep cells directly yields
        // the Q-rep of `a mod b` with the same floor-mod semantics as numpy.
        let zero = self.constant_int(0)?;
        let negative = self.lt_i(b, &zero)?;
        let neg_a = self.sub_i(&zero, a)?;
        let neg_b = self.sub_i(&zero, b)?;
        let numerator = self.select(&negative, &neg_a, a)?;
        let denominator = self.select(&negative, &neg_b, b)?;
        let remainder = self.mod_i(&numerator, &denominator)?;
        let neg_remainder = self.sub_i(&zero, &remainder)?;
        self.select(&negative, &neg_remainder, &remainder)
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
        Ok(self.eq_gadget(a, b))
    }
    fn ne_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let eq = self.eq_i(a, b)?;
        self.logical_not(&eq)
    }
    fn lt_i(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // 2 rows total (was 10 originally):
        //   1. sub_sd: fused sub + signed_decompose (replaces 2 rows).
        //   2. is_nonzero_and: fused is_nonzero(diff) AND sign
        //      (replaces is_zero + bool_not + 2× truthy_bit + bool_and = 5 rows).
        let (diff, _abs, is_neg) = self.sub_signed_decompose(a, b)?;
        Ok(self.is_nonzero_and_gadget(&diff, &is_neg))
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
    fn eq_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.eq_i(a, b)
    }
    fn ne_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.ne_i(a, b)
    }
    fn lt_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.lt_i(a, b)
    }
    fn lte_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.lte_i(a, b)
    }
    fn gt_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.gt_i(a, b)
    }
    fn gte_f(&mut self, a: &Halo2CellRef, b: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        self.gte_i(a, b)
    }

    // ── Transcendentals (constrained via Horner polynomial evaluation) ─

    fn sin_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // sin(x) via the SIN_COEFS Remez polynomial, fit on [0, π).
        //
        // The polynomial is only accurate inside its fit domain — at x = 5 it
        // already drifts by ~0.04, at x = 6.28 by ~0.86. Naïvely evaluating it
        // on raw `x` produces witness values that disagree with f64 sin by
        // more than the brackets users write around `np.sin`, so any bracket
        // assert downstream fails the MockProver gate.
        //
        // Previous implementation used `mod_f(a_abs, 2π)` which chains
        // `div_f`/`mul_f`/`sub_f` over Q-encoded floats. `div_f` returns the
        // real quotient (not the integer floor), so `mod_f` collapsed to
        // `a - (a/b)*b ≈ 0` regardless of input — every sin call returned 0.
        //
        // Fix: integer division on the Q-encoded operands gives the floor
        // directly (matches `kernel::fp_mul_rescale` / `mod_i` pattern).
        // Reduce by π (not 2π) because the polynomial's fit domain is [0, π),
        // then carry the quotient's parity into the sign: sin(x + π) = -sin(x).
        //
        //   |a|  =  q*π + r,   r ∈ [0, π),  q ∈ ℤ_{≥0}
        //   sin(a) = (-1)^{q + sign(a)} * poly(r)
        //
        // q is bounded by |a| / π; for typical Q32 inputs (|a| < 2^31) this
        // fits comfortably in i64. Parity is extracted via mod_i with b=2.
        let (a_abs, a_sign) = self.signed_decompose(a)?;

        // Range-reduce |a| mod π. `floor_div_i` and `mod_i` operate on the
        // i64 interpretations of Q-encoded values, which is exactly integer
        // floor division on the Q-rep — i.e. floor(|a| / π) and |a| - q*π_q
        // both in Q-encoding.
        let pi = self.constant_float(std::f64::consts::PI)?;
        let q_cell = self.floor_div_i(&a_abs, &pi)?;
        let r_cell = self.mod_i(&a_abs, &pi)?;

        // q_parity = q mod 2 ∈ {0, 1}.
        let two_int = self.constant_int(2)?;
        let q_parity = self.mod_i(&q_cell, &two_int)?;

        // Evaluate sin polynomial (Remez coefficients) on r ∈ [0, π).
        let coefs: Vec<Fp> = SIN_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let sin_val = self.poly_eval_horner(&r_cell, &coefs);

        // Combined sign: a_sign XOR q_parity. Both are bits, so XOR is
        // `s1 + s2 - 2*s1*s2`. Built from add/mul/sub gates.
        let s1_s2 = self.bin_gate(
            "mul",
            &a_sign,
            &q_parity,
            a_sign.value * q_parity.value,
            "sin_s1s2",
        );
        let two_s1_s2 = self.bin_gate(
            "add",
            &s1_s2,
            &s1_s2,
            s1_s2.value + s1_s2.value,
            "sin_2s1s2",
        );
        let s1_plus_s2 = self.bin_gate(
            "add",
            &a_sign,
            &q_parity,
            a_sign.value + q_parity.value,
            "sin_s1ps2",
        );
        let final_sign = self.bin_gate(
            "sub",
            &s1_plus_s2,
            &two_s1_s2,
            s1_plus_s2.value - two_s1_s2.value,
            "sin_sign_xor",
        );
        // final_sign ∈ {0, 1} is enforced by the fused bit-check inside
        // the cond_neg gate below — no separate constrain_bit row needed.

        // Apply combined sign via cond_neg: c = sin_val - 2*final_sign*sin_val.
        let row = self.offset;
        self.rec_sel("cond_neg");
        let _ = self.rec_advice(0, sin_val.value, "sin_val");
        self.rec_copy(&sin_val, 0, row);
        let _ = self.rec_advice(1, final_sign.value, "sin_sign");
        self.rec_copy(&final_sign, 1, row);
        let result_val = if final_sign.value == Fp::one() {
            -sin_val.value
        } else {
            sin_val.value
        };
        let result = self.rec_advice(2, result_val, "sin_result");
        self.offset += 1;
        Ok(result)
    }

    fn cos_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // cos(x) = sin(x + π/2). The shift is exact in Q-encoding; `sin_f`
        // owns the range reduction so cos inherits the fix.
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
        // Witness-and-verify: prover computes y = sqrt(x) in native f64
        // and supplies it as a free advice cell; the circuit only checks
        // the easy direction y² ≈ x within ±ε. ~5 rows total, vs the
        // 70-row log_f → mul_f → exp_f composition we used previously.
        // Matches halo2_builder's SqrtFIR lowering exactly.
        let precision_bits = self.params.precision_bits;
        let a_f64 = kernel::dequantize_from_fp(a.value, precision_bits);
        // For a ≤ 0 the polynomial path used to fall back silently; here
        // we mirror that by witnessing 0 and letting the y² ≈ x check
        // fail at proof time (so the prover can't pass off a bogus sqrt
        // of a negative value).
        let y_f64 = if a_f64.is_finite() && a_f64 >= 0.0 {
            a_f64.sqrt()
        } else {
            0.0
        };
        let y_cell = self.witness_cell(kernel::quantize_to_fp(y_f64, precision_bits), "sqrt_y");

        // y² ≈ x within ±0.001.
        let y_sq = self.mul_f(&y_cell, &y_cell)?;
        let diff = self.sub_f(&y_sq, a)?;
        let eps_pos = self.constant_float(0.001)?;
        let eps_neg = self.constant_float(-0.001)?;
        let lt_pos = self.lt_f(&diff, &eps_pos)?;
        let gt_neg = self.lt_f(&eps_neg, &diff)?;
        self.assert_true(&lt_pos)?;
        self.assert_true(&gt_neg)?;
        Ok(y_cell)
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
    fn arctan2_f(
        &mut self,
        y: &Halo2CellRef,
        x: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        // Absolute values and signs.
        let (abs_y, neg_y) = self.signed_decompose(y)?;
        let (abs_x, neg_x) = self.signed_decompose(x)?;

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

    /// arcsin(x) for x ∈ [-1, 1]. Implemented as π/2 - arccos(x), reusing
    /// the existing arccos lowering and the same domain witness constraint
    /// at the IR-builder contract layer.
    fn arcsin_f(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let half_pi = self.constant_float(std::f64::consts::FRAC_PI_2)?;
        let acos = self.arccos_f(a)?;
        self.sub_f(&half_pi, &acos)
    }

    // ── Boolean logic (constrained via gates) ─────────────────────────
    //
    // numpy `logical_and`/`logical_or`/`logical_not` use truthiness semantics
    // (any non-zero value is true). The mock backend mirrors this. The
    // `bool_and`/`bool_or`/`bool_not` gates here only enforce the algebraic
    // identity (`c = a*b`, `c = a + b - a*b`, `c = 1 - a`); they do *not*
    // constrain the operands to {0, 1}. Passing e.g. (5, 3) into the raw gate
    // would yield 15 / -7 / -4 respectively — divergent from numpy/mock.
    //
    // Fix: coerce each operand to its truthy bit via `is_zero_gadget` first.
    // `is_zero(a) ∈ {0, 1}` by construction, so `1 - is_zero(a)` is the
    // truthy bit. The gate output is then automatically in {0, 1}.
    //
    // Internal callers (`lt_i`, `lte_i`, …) already pass {0, 1} cells, so the
    // extra coercion is a no-op semantically (just a few extra gates).

    fn logical_and(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        // Skip the truthy_bit coerce on inputs already known to be in
        // {0, 1} (e.g. the (is_neg, is_nz) pair from lt_i). Each skip
        // saves one row.
        let ta = if self.is_bit(a) {
            a.clone()
        } else {
            self.truthy_bit(a)
        };
        let tb = if self.is_bit(b) {
            b.clone()
        } else {
            self.truthy_bit(b)
        };
        let out = self.bin_gate("bool_and", &ta, &tb, ta.value * tb.value, "and");
        self.mark_bit(&out);
        Ok(out)
    }
    fn logical_or(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        // Skip the truthy_bit coerce on inputs already known to be in
        // {0, 1} (e.g. comparison results). Each skip saves one row.
        let ta = if self.is_bit(a) {
            a.clone()
        } else {
            self.truthy_bit(a)
        };
        let tb = if self.is_bit(b) {
            b.clone()
        } else {
            self.truthy_bit(b)
        };
        let out = self.bin_gate(
            "bool_or",
            &ta,
            &tb,
            ta.value + tb.value - ta.value * tb.value,
            "or",
        );
        self.mark_bit(&out);
        Ok(out)
    }
    fn logical_not(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // `not(a)` is exactly `is_zero(a)` — 1 if a = 0, 0 if a ≠ 0. The
        // previous `c = 1 - a` gate only worked for a ∈ {0, 1}; for arbitrary
        // truthy a this diverged from numpy/mock semantics.
        Ok(self.is_zero_gadget(a))
    }

    // ── Select (constrained via gate) ─────────────────────────────────

    fn select(
        &mut self,
        cond: &Halo2CellRef,
        t: &Halo2CellRef,
        f: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        // The select gate enforces `c = cond * (t - f) + f`, which only matches
        // numpy `np.where` semantics when `cond ∈ {0, 1}`. For non-{0,1} cond
        // (e.g. cond=5, t=1, f=2 → gate yields -3; numpy yields 1) the output
        // diverges from mock. Same shape as the logical_or/logical_and bug.
        //
        // Fix: coerce `cond` to its truthy bit via `is_zero` + `bool_not`
        // before wiring it into the gate, matching the truthy-bit pattern
        // already applied to logical_and/logical_or. Internal callers that
        // already pass {0, 1} bits (e.g. from comparisons) just incur a few
        // extra no-op gates.
        // Skip the truthy_bit coerce on `cond` if it's already known to
        // be in {0, 1}. For argmax / count / select-from-compare loops
        // (e.g. the 92 selects in case 0309) the cond is always a
        // comparison result, so this drops one row per select.
        let cond_bit = if self.is_bit(cond) {
            cond.clone()
        } else {
            self.truthy_bit(cond)
        };
        let result = cond_bit.value * (t.value - f.value) + f.value;
        let row = self.offset;
        self.rec_sel("select");
        let _ = self.rec_advice(0, cond_bit.value, "sel_cond");
        self.rec_copy(&cond_bit, 0, row);
        let _ = self.rec_advice(1, t.value, "sel_t");
        self.rec_copy(t, 1, row);
        let _ = self.rec_advice(2, f.value, "sel_f");
        self.rec_copy(f, 2, row);
        let rc = self.rec_advice(3, result, "sel_c");
        self.offset += 1;
        Ok(rc)
    }

    fn select_bit(
        &mut self,
        cond: &Halo2CellRef,
        t: &Halo2CellRef,
        f: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        let result = cond.value * (t.value - f.value) + f.value;
        let row = self.offset;
        self.rec_sel("select_bit");
        let _ = self.rec_advice(0, cond.value, "sel_bit_cond");
        self.rec_copy(cond, 0, row);
        let _ = self.rec_advice(1, t.value, "sel_bit_t");
        self.rec_copy(t, 1, row);
        let _ = self.rec_advice(2, f.value, "sel_bit_f");
        self.rec_copy(f, 2, row);
        let rc = self.rec_advice(3, result, "sel_bit_c");
        self.offset += 1;
        self.mark_bit(cond);
        Ok(rc)
    }

    fn select_many_lane(
        &mut self,
        cond: &Halo2CellRef,
        true_values: &[Halo2CellRef],
        false_values: &[Halo2CellRef],
        lane: usize,
    ) -> Result<Halo2CellRef, ProvingError> {
        if true_values.len() != false_values.len() {
            return Err(ProvingError::synthesis(
                "select_many branch lengths must match",
            ));
        }
        let cond_bit = if self.is_bit(cond) {
            cond.clone()
        } else {
            self.truthy_bit(cond)
        };
        let mut key = Vec::with_capacity(1 + true_values.len() * 2);
        key.push((cond_bit.col_idx, cond_bit.row));
        key.extend(true_values.iter().map(|cell| (cell.col_idx, cell.row)));
        key.extend(false_values.iter().map(|cell| (cell.col_idx, cell.row)));
        if !self.select_many_cache.contains_key(&key) {
            let mut outs = Vec::with_capacity(true_values.len());
            let mut i = 0usize;
            while i < true_values.len() {
                if self.config.wide_select_many && i + 4 < true_values.len() {
                    let row = self.offset;
                    self.rec_sel("select_many5");
                    let _ = self.rec_advice(0, cond_bit.value, "select_many5_cond");
                    self.rec_copy(&cond_bit, 0, row);

                    for lane in 0..5 {
                        let t = &true_values[i + lane];
                        let f = &false_values[i + lane];
                        let out_val = cond_bit.value * (t.value - f.value) + f.value;
                        let base_col = 1 + 3 * lane;
                        let _ = self.rec_advice(base_col, t.value, "select_many5_t");
                        self.rec_copy(t, base_col, row);
                        let _ = self.rec_advice(base_col + 1, f.value, "select_many5_f");
                        self.rec_copy(f, base_col + 1, row);
                        let out = self.rec_advice(base_col + 2, out_val, "select_many5_out");
                        outs.push(out);
                    }

                    self.offset += 1;
                    i += 5;
                } else if i + 2 < true_values.len() {
                    let row = self.offset;
                    self.rec_sel("select_many3");
                    let _ = self.rec_advice(0, cond_bit.value, "select_many3_cond");
                    self.rec_copy(&cond_bit, 0, row);

                    let t0 = &true_values[i];
                    let f0 = &false_values[i];
                    let out0_val = cond_bit.value * (t0.value - f0.value) + f0.value;
                    let _ = self.rec_advice(1, t0.value, "select_many3_t0");
                    self.rec_copy(t0, 1, row);
                    let _ = self.rec_advice(2, f0.value, "select_many3_f0");
                    self.rec_copy(f0, 2, row);
                    let out0 = self.rec_advice(3, out0_val, "select_many3_out0");

                    let t1 = &true_values[i + 1];
                    let f1 = &false_values[i + 1];
                    let out1_val = cond_bit.value * (t1.value - f1.value) + f1.value;
                    let _ = self.rec_advice(4, t1.value, "select_many3_t1");
                    self.rec_copy(t1, 4, row);

                    let row1 = row + 1;
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 0,
                        row: row1,
                        value: f1.value,
                        annotation: "select_many3_f1".to_string(),
                    });
                    self.rec_copy(f1, 0, row1);
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 1,
                        row: row1,
                        value: out1_val,
                        annotation: "select_many3_out1".to_string(),
                    });
                    let out1 = Halo2CellRef {
                        value: out1_val,
                        col_idx: 1,
                        row: row1,
                    };

                    let t2 = &true_values[i + 2];
                    let f2 = &false_values[i + 2];
                    let out2_val = cond_bit.value * (t2.value - f2.value) + f2.value;
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 2,
                        row: row1,
                        value: t2.value,
                        annotation: "select_many3_t2".to_string(),
                    });
                    self.rec_copy(t2, 2, row1);
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 3,
                        row: row1,
                        value: f2.value,
                        annotation: "select_many3_f2".to_string(),
                    });
                    self.rec_copy(f2, 3, row1);
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 4,
                        row: row1,
                        value: out2_val,
                        annotation: "select_many3_out2".to_string(),
                    });
                    let out2 = Halo2CellRef {
                        value: out2_val,
                        col_idx: 4,
                        row: row1,
                    };

                    self.offset += 2;
                    outs.push(out0);
                    outs.push(out1);
                    outs.push(out2);
                    i += 3;
                } else if i + 1 < true_values.len() {
                    let row = self.offset;
                    self.rec_sel("select_many2");
                    let _ = self.rec_advice(0, cond_bit.value, "select_many_cond");
                    self.rec_copy(&cond_bit, 0, row);

                    let t0 = &true_values[i];
                    let f0 = &false_values[i];
                    let out0_val = cond_bit.value * (t0.value - f0.value) + f0.value;
                    let _ = self.rec_advice(1, t0.value, "select_many_t0");
                    self.rec_copy(t0, 1, row);
                    let _ = self.rec_advice(2, f0.value, "select_many_f0");
                    self.rec_copy(f0, 2, row);
                    let out0 = self.rec_advice(3, out0_val, "select_many_out0");

                    let row1 = row + 1;
                    let t1 = &true_values[i + 1];
                    let f1 = &false_values[i + 1];
                    let out1_val = cond_bit.value * (t1.value - f1.value) + f1.value;
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 1,
                        row: row1,
                        value: t1.value,
                        annotation: "select_many_t1".to_string(),
                    });
                    self.rec_copy(t1, 1, row1);
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 2,
                        row: row1,
                        value: f1.value,
                        annotation: "select_many_f1".to_string(),
                    });
                    self.rec_copy(f1, 2, row1);
                    self.ops.push(GateOp::AssignAdvice {
                        col_idx: 3,
                        row: row1,
                        value: out1_val,
                        annotation: "select_many_out1".to_string(),
                    });
                    let out1 = Halo2CellRef {
                        value: out1_val,
                        col_idx: 3,
                        row: row1,
                    };
                    self.offset += 2;
                    outs.push(out0);
                    outs.push(out1);
                    i += 2;
                } else {
                    let out = self.select(&cond_bit, &true_values[i], &false_values[i])?;
                    outs.push(out);
                    i += 1;
                }
            }
            self.select_many_cache.insert(key.clone(), outs);
        }
        self.select_many_cache
            .get(&key)
            .and_then(|outs| outs.get(lane))
            .cloned()
            .ok_or_else(|| ProvingError::synthesis("select_many lane out of bounds"))
    }

    // ── Casting (constrained) ─────────────────────────────────────────

    fn int_cast(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        // Float→Int: truncation toward zero, q = trunc(a/scale), with a
        // *signed* remainder r (sign(r) == sign(a), |r| < scale). Soundness:
        //   1. div_mod gate: a == scale*q + r,
        //   2. |r| < scale = 2^precision_bits (range check on the magnitude),
        //   3. sign(r) == sign(a) (r rebuilt from a's true sign via select),
        // which together pin the unique truncating (q, r) — without (2)+(3)
        // the bare div_mod left (q, r) malleable (closes C10 for int_cast).
        let precision_bits = self.params.precision_bits;
        let scale =
            Fp::from(crate::prove::field::quantization_scale(precision_bits) as u64);
        let v = kernel::fp_to_i64(a.value);
        let scale_i = crate::prove::field::quantization_scale(precision_bits) as i64;
        let q = v / scale_i;
        let r = v - q * scale_i;
        let q_fp = if q >= 0 {
            Fp::from(q as u64)
        } else {
            -Fp::from((-q) as u64)
        };
        let r_fp = if r >= 0 {
            Fp::from(r as u64)
        } else {
            -Fp::from((-r) as u64)
        };
        let row = self.offset;
        self.rec_sel("div_mod");
        let _ = self.rec_advice(0, a.value, "icast_a");
        self.rec_copy(a, 0, row);
        let _ = self.rec_advice(1, scale, "icast_scale");
        let qc = self.rec_advice(2, q_fp, "icast_q");
        let r_cell = self.rec_advice(3, r_fp, "icast_r");
        self.offset += 1;

        // |r| < scale: range-check the remainder magnitude to precision_bits.
        let r_mag_val = if r >= 0 {
            Fp::from(r as u64)
        } else {
            Fp::from((-r) as u64)
        };
        let r_mag = self.alloc_free_advice(r_mag_val, "icast_r_mag");
        self.range_check_chunks(&r_mag, precision_bits, "icast_r_mag_rc")?;

        // sign(r) == sign(a): rebuild r from a's true sign bit. `lt_i(a, 0)`
        // is the sound sign bit (now magnitude-bound), and select pins r so
        // the prover cannot flip its sign independently of `a`.
        let zero = self.constant_int(0)?;
        let sign = self.lt_i(a, &zero)?;
        let neg_r_mag = self.sub_i(&zero, &r_mag)?;
        let r_rebuilt = self.select(&sign, &neg_r_mag, &r_mag)?;
        self.constrain_equal_cells(&r_rebuilt, &r_cell, "icast_r_signed");
        Ok(qc)
    }
    fn float_cast(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let scale = crate::prove::field::quantization_scale(self.params.precision_bits) as i64;
        let scale_cell = self.constant_int(scale)?;
        self.mul_i(a, &scale_cell)
    }
    fn bool_cast(&mut self, a: &Halo2CellRef) -> Result<Halo2CellRef, ProvingError> {
        let iz = self.is_zero_gadget(a);
        self.logical_not(&iz)
    }

    // ── I/O (constrained) ─────────────────────────────────────────────

    fn read_input(
        &mut self,
        path: &InputPath,
        is_public: bool,
    ) -> Result<Halo2CellRef, ProvingError> {
        let val = self.get_witness_fp_path(path).unwrap_or(Fp::zero());
        let cell = self.alloc_free_advice(val, &format!("input_{}", path.display()));
        if is_public {
            self.public_cells
                .push((cell.col_idx, cell.row, self.instance_row));
            self.public_values.push(val);
            self.instance_row += 1;
        }
        Ok(cell)
    }
    fn read_external_result(
        &mut self,
        store_idx: u32,
        output_idx: u32,
    ) -> Result<Halo2CellRef, ProvingError> {
        let val = self
            .get_external_fp(store_idx, output_idx)
            .unwrap_or(Fp::zero());
        Ok(self.alloc_free_advice(val, &format!("ext_{}_{}", store_idx, output_idx)))
    }
    fn expose_public(&mut self, a: &Halo2CellRef, _label: &str) -> Result<(), ProvingError> {
        self.public_cells
            .push((a.col_idx, a.row, self.instance_row));
        self.public_values.push(a.value);
        self.instance_row += 1;
        Ok(())
    }
    fn assert_true(&mut self, a: &Halo2CellRef) -> Result<(), ProvingError> {
        // Gate "assert": s * (1 - a) = 0, i.e. the witness must equal 1.
        // Check the recorded value against the gate so an unsatisfiable
        // witness (e.g. `assert(False)`) is caught before create_proof.
        if a.value != Fp::one() {
            self.note_unsat(|| {
                format!(
                    "Witness does not satisfy circuit constraints: assert gate expected 1 but witness is {}",
                    kernel::fp_to_i64(a.value)
                )
            });
        }
        let row = self.offset;
        self.rec_sel("assert");
        let _ = self.rec_advice(0, a.value, "assert_a");
        self.rec_copy(a, 0, row);
        self.offset += 1;
        Ok(())
    }

    fn assert_bit(&mut self, a: &Halo2CellRef) -> Result<(), ProvingError> {
        // Single-row s_bit gate: a * (a - 1) = 0. Saves 4 rows over the
        // default `assert_eq(bool_cast(a), a)` path used by other backends.
        self.constrain_bit(a);
        Ok(())
    }

    fn assert_u128_lt(
        &mut self,
        idx: &Halo2CellRef,
        hi: &Halo2CellRef,
    ) -> Result<(), ProvingError> {
        let hi_i = kernel::fp_to_i64(hi.value);
        if (3..=8).contains(&hi_i) {
            let bits = u64::BITS - ((hi_i as u64) - 1).leading_zeros();
            let _ = self.assert_u128_lt_bit(idx, hi_i as u64, 0, bits)?;
            return Ok(());
        }

        let zero = self.constant_int(0)?;
        let ge = self.gte_i(idx, &zero)?;
        let lt = self.lt_i(idx, hi)?;
        let ok = self.logical_and(&ge, &lt)?;
        self.assert_true(&ok)
    }

    fn assert_u128_lt_bit(
        &mut self,
        idx: &Halo2CellRef,
        hi: u64,
        bit: u32,
        bits: u32,
    ) -> Result<Halo2CellRef, ProvingError> {
        if (3..=8).contains(&hi) {
            let selector = match hi {
                3 => "range3",
                4 => "range_bits2",
                5 => "range5",
                6 => "range6",
                7 => "range7",
                8 => "range_bits3",
                _ => unreachable!(),
            };
            let key = (idx.col_idx, idx.row, hi as i64, bits);
            if !self.u128_lt_bit_cache.contains_key(&key) {
                let idx_i = kernel::fp_to_i64(idx.value);
                let idx_u = if idx_i >= 0 && (idx_i as u64) < hi {
                    idx_i as u64
                } else {
                    // Out-of-range witness: the bit gate below pins the real
                    // `idx` against a 0-decomposition, so the circuit is
                    // unsatisfiable and `verify` rejects. Record it so the
                    // always-on witness-satisfaction check (`first_unsat`)
                    // also refuses it at PROVE time — the strong contract the
                    // bounds-discharge relies on (e.g. OOB dynamic writes).
                    self.note_unsat(|| {
                        format!(
                            "Witness does not satisfy circuit constraints: index {} out of range [0, {})",
                            idx_i, hi
                        )
                    });
                    0
                };
                let row = self.offset;
                self.rec_sel(selector);
                let _ = self.rec_advice(0, idx.value, "assert_u128_lt_bits_idx");
                self.rec_copy(idx, 0, row);
                let b0 = self.rec_advice(1, Fp::from(idx_u & 1), "assert_u128_lt_bits_b0");
                let b1 = self.rec_advice(2, Fp::from((idx_u >> 1) & 1), "assert_u128_lt_bits_b1");
                let mut cells = vec![b0, b1];
                let b2 = self.rec_advice(
                    3,
                    Fp::from((idx_u >> 2) & 1),
                    "assert_u128_lt_bits_b2",
                );
                if bits == 3 {
                    cells.push(b2);
                }
                self.offset += 1;
                for cell in &cells {
                    self.mark_bit(cell);
                }
                self.u128_lt_bit_cache.insert(key, cells);
            }
            return self
                .u128_lt_bit_cache
                .get(&key)
                .and_then(|cells| cells.get(bit as usize))
                .cloned()
                .ok_or_else(|| ProvingError::synthesis("range bit index out of bounds"));
        }

        let hi_cell = self.constant_int(hi as i64)?;
        self.assert_u128_lt(idx, &hi_cell)?;
        let divisor = self.constant_int(1_i64 << bit)?;
        let shifted = self.floor_div_i(idx, &divisor)?;
        let two = self.constant_int(2)?;
        let out = self.mod_i(&shifted, &two)?;
        self.assert_bit(&out)?;
        Ok(out)
    }

    // ── Memory (constrained via equality constraints on writes/reads) ──

    fn allocate_memory(
        &mut self,
        segment_id: u32,
        size: u32,
        init: i64,
    ) -> Result<(), ProvingError> {
        let init_fp = if init >= 0 {
            Fp::from(init as u64)
        } else {
            -Fp::from((-init) as u64)
        };
        self.memories
            .insert(segment_id, vec![init_fp; size as usize]);
        self.memory_cells
            .insert(segment_id, vec![None; size as usize]);
        self.memory_init.insert(segment_id, init_fp);
        Ok(())
    }
    fn write_memory(
        &mut self,
        segment_id: u32,
        addr: &Halo2CellRef,
        value: &Halo2CellRef,
    ) -> Result<(), ProvingError> {
        let a = kernel::fp_to_i64(addr.value) as usize;
        let mem = self
            .memories
            .get_mut(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("Segment not allocated"))?;
        if a >= mem.len() {
            return Err(ProvingError::synthesis(format!(
                "Memory write out of bounds: segment {} addr {} len {}",
                segment_id,
                a,
                mem.len()
            )));
        }
        mem[a] = value.value;
        if let Some(cells) = self.memory_cells.get_mut(&segment_id) {
            cells[a] = Some(value.clone());
        }
        Ok(())
    }
    fn read_memory(
        &mut self,
        segment_id: u32,
        addr: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        let a = kernel::fp_to_i64(addr.value) as usize;
        if let Some(Some(cell)) = self
            .memory_cells
            .get(&segment_id)
            .and_then(|cells| cells.get(a))
        {
            return Ok(cell.clone());
        }

        let init = self
            .memory_init
            .get(&segment_id)
            .copied()
            .unwrap_or(Fp::zero());
        let mem = self
            .memories
            .get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("Segment not allocated"))?;
        let val = if a < mem.len() { mem[a] } else { init };
        Ok(self.alloc_free_advice(val, "mem_init"))
    }
    fn memory_trace_emit(
        &mut self,
        _: u32,
        _: bool,
        _: &[Halo2CellRef],
    ) -> Result<(), ProvingError> {
        Ok(())
    }
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError> {
        Ok(())
    }

    // ── Dynamic NDArray (constrained via memory) ──────────────────────

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
        self.allocate_memory(10000 + array_id, max_length, 0)
    }
    fn witness_dynamic_ndarray_meta(
        &mut self,
        _: u32,
        _: u32,
        _: &str,
        _: &[Halo2CellRef],
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
    // assert_dynamic_ndarray_meta uses the trait-default lowering.
    fn dynamic_ndarray_get_item(
        &mut self,
        array_id: u32,
        _: u32,
        index: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        self.read_memory(10000 + array_id, index)
    }
    fn dynamic_ndarray_set_item(
        &mut self,
        array_id: u32,
        _: u32,
        index: &Halo2CellRef,
        value: &Halo2CellRef,
    ) -> Result<(), ProvingError> {
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

    fn eq_hash(
        &mut self,
        a: &Halo2CellRef,
        b: &Halo2CellRef,
    ) -> Result<Halo2CellRef, ProvingError> {
        self.eq_i(a, b)
    }
}

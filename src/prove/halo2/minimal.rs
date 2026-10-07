//! Minimal Halo2 circuit variant for kernels that only read inputs,
//! materialise constants, and expose public outputs — no arithmetic, no
//! comparisons, no transcendentals, no assertions, no zkram I/O.
//!
//! The full [`crate::prove::halo2::ZinniaCircuit`] configures **8 columns**
//! (5 advice + 1 fixed + 1 instance + 1 lookup-table) plus ~20 selectors
//! and ~20 gate definitions. For "pure I/O" kernels — `np.array(a)`,
//! `np.reshape(a, ...)`, `a.shape`, etc. — that's strictly wasted MSM
//! cost: the lookup table holds no useful rows, the fixed column has no
//! useful rows, and 4 of the 5 advice columns sit empty.
//!
//! This module supplies a stripped-down replacement with **2 columns**
//! (1 advice + 1 instance), no selectors, no custom gates — just witness
//! assignment and instance copy constraints. For identity-style kernels
//! this drops the per-proof polynomial commitment count from 8 to 2.
//!
//! Eligibility is determined deterministically from the IR via
//! [`is_minimal_eligible`]. Both prove and verify must use the same
//! predicate so the verifier can re-keygen the matching circuit shape
//! from `vk_bytes`.
//!
//! The `expose_public` path is shared symmetrically between
//! `ExposePublicF` and `ExposePublicI` — any improvement here applies
//! to both variants identically.

use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value as HaloValue},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
};

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::kernel;
use crate::prove::types::ProvingParams;

/// IR ops the minimal circuit can synthesise.
fn op_is_minimal(op: &IR) -> bool {
    matches!(
        op,
        IR::ReadInteger { .. }
            | IR::ReadFloat { .. }
            | IR::ConstantInt { .. }
            | IR::ConstantFloat { .. }
            | IR::ConstantBool { .. }
            | IR::ExposePublicF
            | IR::ExposePublicI
    )
}

/// True iff every IR statement is in the minimal subset *and* the
/// 1-column layout doesn't force a worse k than the full 5-column
/// packed layout would.
///
/// The 1-column minimal layout consumes 1 advice row per produced
/// value (no packing). The full layout packs 5 produced values per
/// row across 5 advice columns. So 1-column is k-compatible iff the
/// instance column already dominates: `advice_count ≤ instance_count`.
/// Beyond that we'd over-size k and pay back the column savings.
///
/// Must be a pure function of the IR — prove and verify dispatch
/// must agree.
pub fn is_minimal_eligible(ir: &IRGraph) -> bool {
    if !ir.stmts.iter().all(|s| op_is_minimal(&s.ir)) {
        return false;
    }
    let (advice, instance) = estimate_rows(ir);
    advice <= instance
}

#[derive(Debug, Clone)]
pub struct MinimalConfig {
    pub advice: Column<Advice>,
    pub instance: Column<Instance>,
}

impl MinimalConfig {
    pub fn configure(meta: &mut ConstraintSystem<Fp>) -> Self {
        let advice = meta.advice_column();
        let instance = meta.instance_column();
        meta.enable_equality(advice);
        meta.enable_equality(instance);
        Self { advice, instance }
    }
}

/// Slim Circuit struct that holds *only* the data `synthesize` needs:
/// the per-row advice values and the (advice_row, instance_row) constraint
/// pairs. We pre-resolve the IR + witness once before calling
/// `create_proof`, so the synthesize closure never has to walk the IR or
/// touch the witness — it just iterates two pre-built Vec`s. Trimming
/// MinimalCircuit's old IRGraph/ResolvedWitness fields here saves ~250 ms
/// per prove on 0387 by shrinking the working set (and thus L1/L2 cache
/// pressure) inside halo2's `create_proof`.
#[derive(Clone)]
pub struct MinimalCircuit {
    /// Per-advice-row witness value. `Some(_)` at prove time, `None` for
    /// `without_witnesses` (keygen-only synthesis).
    pub row_values: Vec<Fp>,
    /// `exposes[i] = advice_row` constrained to `instance_row = i`.
    pub exposes: Vec<u32>,
    /// Whether this is a witness-bearing circuit (set false for
    /// `without_witnesses`, used by halo2 keygen).
    pub witnesses_present: bool,
}

impl MinimalCircuit {
    /// Build a witness-absent MinimalCircuit (for keygen on the verify
    /// side, where we only have the IR). row_values are all zero;
    /// `exposes` carries the (advice_row, instance_row) shape so halo2's
    /// keygen can derive the same VK as the prover did.
    pub fn shape_only(ir: &IRGraph) -> Self {
        let n_stmts = ir.stmts.len();
        let mut stmt_to_row: Vec<u32> = vec![u32::MAX; n_stmts];
        let mut next_row: u32 = 0;
        let mut exposes: Vec<u32> = Vec::new();
        for stmt in &ir.stmts {
            match &stmt.ir {
                IR::ReadInteger { .. }
                | IR::ReadFloat { .. }
                | IR::ConstantInt { .. }
                | IR::ConstantFloat { .. }
                | IR::ConstantBool { .. } => {
                    stmt_to_row[stmt.stmt_id as usize] = next_row;
                    next_row += 1;
                }
                IR::ExposePublicF | IR::ExposePublicI => {
                    let src = stmt.arguments[0] as usize;
                    let row = stmt_to_row[src];
                    if row != u32::MAX {
                        exposes.push(row);
                    }
                }
                _ => {}
            }
        }
        MinimalCircuit {
            row_values: vec![Fp::zero(); next_row as usize],
            exposes,
            witnesses_present: false,
        }
    }

    /// Resolve the IR + witness into the prove-ready (row_values, exposes,
    /// public_inputs) triple. Walks the IR once, witness-resolving every
    /// read/constant and building the per-instance source-advice map.
    /// This is the *only* IR walk per prove call — the slim circuit then
    /// passes through halo2's `create_proof` without dragging the heavy
    /// IRGraph + ResolvedWitness data around.
    pub fn resolve(
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> (Vec<Fp>, Vec<u32>, Vec<Fp>) {
        let n_stmts = ir.stmts.len();
        let mut stmt_to_row: Vec<u32> = vec![u32::MAX; n_stmts];
        let mut row_values: Vec<Fp> = Vec::with_capacity(n_stmts);
        let mut exposes: Vec<u32> = Vec::new();
        let mut publics: Vec<Fp> = Vec::new();
        for stmt in &ir.stmts {
            match &stmt.ir {
                IR::ReadInteger { path, .. } | IR::ReadFloat { path, .. } => {
                    let v = witness.resolve_path(path).unwrap_or(Fp::zero());
                    stmt_to_row[stmt.stmt_id as usize] = row_values.len() as u32;
                    row_values.push(v);
                }
                IR::ConstantInt { value } => {
                    let v = if *value >= 0 {
                        Fp::from(*value as u64)
                    } else {
                        -Fp::from((-*value) as u64)
                    };
                    stmt_to_row[stmt.stmt_id as usize] = row_values.len() as u32;
                    row_values.push(v);
                }
                IR::ConstantFloat { value } => {
                    let v = kernel::quantize_to_fp(*value, params.precision_bits);
                    stmt_to_row[stmt.stmt_id as usize] = row_values.len() as u32;
                    row_values.push(v);
                }
                IR::ConstantBool { value } => {
                    let v = if *value { Fp::one() } else { Fp::zero() };
                    stmt_to_row[stmt.stmt_id as usize] = row_values.len() as u32;
                    row_values.push(v);
                }
                IR::ExposePublicF | IR::ExposePublicI => {
                    let src = stmt.arguments[0] as usize;
                    let row = stmt_to_row[src];
                    if row != u32::MAX {
                        exposes.push(row);
                        publics.push(row_values[row as usize]);
                    }
                }
                _ => {}
            }
        }
        (row_values, exposes, publics)
    }

}

impl Circuit<Fp> for MinimalCircuit {
    type Config = MinimalConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        // For keygen-only synthesis halo2 just needs the shape — row_count
        // and which advice rows the instance column points at — so we
        // clone `exposes` but zero out the witness values.
        Self {
            row_values: vec![Fp::zero(); self.row_values.len()],
            exposes: self.exposes.clone(),
            witnesses_present: false,
        }
    }

    fn configure_with_params(meta: &mut ConstraintSystem<Fp>, _params: ()) -> MinimalConfig {
        MinimalConfig::configure(meta)
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> MinimalConfig {
        MinimalConfig::configure(meta)
    }

    fn params(&self) -> Self::Params {}

    fn synthesize(
        &self,
        config: MinimalConfig,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // The IR walk has already been done by `MinimalCircuit::resolve`
        // (called once per prove). At synth time we only iterate two
        // pre-built Vec`s — no IR, no witness, no HashMap.
        let assigned = layouter.assign_region(
            || "minimal_main",
            |mut region| {
                let mut cells: Vec<AssignedCell<Fp, Fp>> =
                    Vec::with_capacity(self.row_values.len());
                for (row, value) in self.row_values.iter().enumerate() {
                    let cell = region.assign_advice(
                        || "",
                        config.advice,
                        row,
                        || HaloValue::known(*value),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;

        for (instance_row, advice_row) in self.exposes.iter().enumerate() {
            layouter.constrain_instance(
                assigned[*advice_row as usize].cell(),
                config.instance,
                instance_row,
            )?;
        }

        Ok(())
    }
}

/// Compute (`main_rows`, `instance_rows`) for k estimation, mirroring
/// what `synthesize` would actually consume. Used by the estimator so
/// minimal kernels pick the right k for the minimal layout (which is
/// strictly tighter than the full layout because there's no range table
/// or fixed column reserving rows).
pub fn estimate_rows(ir: &IRGraph) -> (usize, usize) {
    let mut advice_rows = 0;
    let mut instance_rows = 0;
    for stmt in &ir.stmts {
        match &stmt.ir {
            IR::ReadInteger { .. }
            | IR::ReadFloat { .. }
            | IR::ConstantInt { .. }
            | IR::ConstantFloat { .. }
            | IR::ConstantBool { .. } => advice_rows += 1,
            IR::ExposePublicF | IR::ExposePublicI => instance_rows += 1,
            _ => {}
        }
    }
    (advice_rows, instance_rows)
}

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::error::ProvingError;
use crate::prove::traits::Synthesizer;

/// Walk the IRGraph in statement order and dispatch each IR instruction
/// to the corresponding `Synthesizer` method.
///
/// The interpreter is backend-agnostic: it knows nothing about halo2 or
/// field types. It simply maps IR variants to trait method calls.
///
/// String operations and `Print` are no-ops (they have no circuit
/// representation). External function calls are rejected — they must be
/// resolved during preprocessing before the IR reaches the prover.
pub fn interpret_ir<S: Synthesizer>(
    graph: &IRGraph,
    synth: &mut S,
) -> Result<(), ProvingError> {
    let n = graph.stmts.len();
    let mut cells: Vec<Option<S::CellRef>> = Vec::with_capacity(n);
    cells.resize_with(n, || None);

    for stmt in &graph.stmts {
        let id = stmt.stmt_id as usize;

        // Collect argument cell references.
        let args: Vec<S::CellRef> = stmt
            .arguments
            .iter()
            .map(|&arg_id| {
                cells[arg_id as usize]
                    .clone()
                    .ok_or_else(|| {
                        ProvingError::synthesis(format!(
                            "Statement {} references undefined cell {}",
                            id, arg_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let result: Option<S::CellRef> = match &stmt.ir {
            // ── Constants ─────────────────────────────────────────────
            IR::ConstantInt { value } => Some(synth.constant_int(*value)?),
            IR::ConstantFloat { value } => Some(synth.constant_float(*value)?),
            IR::ConstantBool { value } => Some(synth.constant_bool(*value)?),
            IR::ConstantStr { .. } => None,

            // ── Integer arithmetic ────────────────────────────────────
            IR::AddI => Some(synth.add_i(&args[0], &args[1])?),
            IR::SubI => Some(synth.sub_i(&args[0], &args[1])?),
            IR::MulI => Some(synth.mul_i(&args[0], &args[1])?),
            IR::DivI => Some(synth.div_i(&args[0], &args[1])?),
            IR::FloorDivI => Some(synth.floor_div_i(&args[0], &args[1])?),
            IR::ModI => Some(synth.mod_i(&args[0], &args[1])?),
            IR::PowI => Some(synth.pow_i(&args[0], &args[1])?),
            IR::AbsI => Some(synth.abs_i(&args[0])?),
            IR::SignI => Some(synth.sign_i(&args[0])?),
            IR::InvI => Some(synth.inv_i(&args[0])?),

            // ── Float arithmetic ──────────────────────────────────────
            IR::AddF => Some(synth.add_f(&args[0], &args[1])?),
            IR::SubF => Some(synth.sub_f(&args[0], &args[1])?),
            IR::MulF => Some(synth.mul_f(&args[0], &args[1])?),
            IR::DivF => Some(synth.div_f(&args[0], &args[1])?),
            IR::FloorDivF => Some(synth.floor_div_f(&args[0], &args[1])?),
            IR::ModF => Some(synth.mod_f(&args[0], &args[1])?),
            IR::PowF => Some(synth.pow_f(&args[0], &args[1])?),
            IR::AbsF => Some(synth.abs_f(&args[0])?),
            IR::SignF => Some(synth.sign_f(&args[0])?),

            // ── Integer comparisons ───────────────────────────────────
            IR::EqI => Some(synth.eq_i(&args[0], &args[1])?),
            IR::NeI => Some(synth.ne_i(&args[0], &args[1])?),
            IR::LtI => Some(synth.lt_i(&args[0], &args[1])?),
            IR::LteI => Some(synth.lte_i(&args[0], &args[1])?),
            IR::GtI => Some(synth.gt_i(&args[0], &args[1])?),
            IR::GteI => Some(synth.gte_i(&args[0], &args[1])?),

            // ── Float comparisons ─────────────────────────────────────
            IR::EqF => Some(synth.eq_f(&args[0], &args[1])?),
            IR::NeF => Some(synth.ne_f(&args[0], &args[1])?),
            IR::LtF => Some(synth.lt_f(&args[0], &args[1])?),
            IR::LteF => Some(synth.lte_f(&args[0], &args[1])?),
            IR::GtF => Some(synth.gt_f(&args[0], &args[1])?),
            IR::GteF => Some(synth.gte_f(&args[0], &args[1])?),

            // ── Transcendentals ───────────────────────────────────────
            IR::SinF => Some(synth.sin_f(&args[0])?),
            IR::SinHF => Some(synth.sinh_f(&args[0])?),
            IR::CosF => Some(synth.cos_f(&args[0])?),
            IR::CosHF => Some(synth.cosh_f(&args[0])?),
            IR::TanF => Some(synth.tan_f(&args[0])?),
            IR::TanHF => Some(synth.tanh_f(&args[0])?),
            IR::SqrtF => Some(synth.sqrt_f(&args[0])?),
            IR::ExpF => Some(synth.exp_f(&args[0])?),
            IR::LogF => Some(synth.log_f(&args[0])?),

            // ── Boolean logic ─────────────────────────────────────────
            IR::LogicalAnd => Some(synth.logical_and(&args[0], &args[1])?),
            IR::LogicalOr => Some(synth.logical_or(&args[0], &args[1])?),
            IR::LogicalNot => Some(synth.logical_not(&args[0])?),

            // ── Selection (mux) ───────────────────────────────────────
            IR::SelectI | IR::SelectF | IR::SelectB => {
                Some(synth.select(&args[0], &args[1], &args[2])?)
            }

            // ── Casting ───────────────────────────────────────────────
            IR::IntCast => Some(synth.int_cast(&args[0])?),
            IR::FloatCast => Some(synth.float_cast(&args[0])?),
            IR::BoolCast => Some(synth.bool_cast(&args[0])?),

            // ── String operations (debug only, no circuit) ────────────
            IR::AddStr | IR::StrI | IR::StrF => None,

            // ── I/O ───────────────────────────────────────────────────
            IR::ReadInteger { path, is_public } => {
                Some(synth.read_input(path, *is_public)?)
            }
            IR::ReadFloat { path, is_public } => {
                Some(synth.read_input(path, *is_public)?)
            }
            IR::ReadHash { path, is_public } => {
                Some(synth.read_input(path, *is_public)?)
            }
            IR::ReadExternalResult { store_idx, output_idx, .. } => {
                Some(synth.read_external_result(*store_idx, *output_idx)?)
            }
            IR::Print => None,

            // ── Assertions & public exposure ──────────────────────────
            IR::Assert => {
                synth.assert_true(&args[0])?;
                None
            }
            IR::ExposePublicI => {
                let label = format!("public_i_{}", id);
                synth.expose_public(&args[0], &label)?;
                None
            }
            IR::ExposePublicF => {
                let label = format!("public_f_{}", id);
                synth.expose_public(&args[0], &label)?;
                None
            }

            // ── Memory operations ─────────────────────────────────────
            IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            } => {
                synth.allocate_memory(*segment_id, *size, *init_value)?;
                None
            }
            IR::WriteMemory { segment_id } => {
                synth.write_memory(*segment_id, &args[0], &args[1])?;
                None
            }
            IR::ReadMemory { segment_id } => {
                Some(synth.read_memory(*segment_id, &args[0])?)
            }
            IR::MemoryTraceEmit {
                segment_id,
                is_write,
            } => {
                synth.memory_trace_emit(*segment_id, *is_write, &args)?;
                None
            }
            IR::MemoryTraceSeal => {
                synth.memory_trace_seal()?;
                None
            }

            // ── Dynamic NDArray ───────────────────────────────────────
            IR::AllocateDynamicNDArrayMeta {
                array_id,
                dtype_name,
                max_length,
                max_rank,
            } => {
                synth.allocate_dynamic_ndarray_meta(
                    *array_id,
                    dtype_name,
                    *max_length,
                    *max_rank,
                )?;
                None
            }
            IR::WitnessDynamicNDArrayMeta { array_id, max_rank } => {
                synth.witness_dynamic_ndarray_meta(*array_id, *max_rank, &args)?;
                None
            }
            IR::AssertDynamicNDArrayMeta {
                array_id,
                max_rank,
                max_length,
            } => {
                synth.assert_dynamic_ndarray_meta(*array_id, *max_rank, *max_length, &args)?;
                None
            }
            IR::DynamicNDArrayGetItem {
                array_id,
                segment_id,
            } => Some(synth.dynamic_ndarray_get_item(*array_id, *segment_id, &args[0])?),
            IR::DynamicNDArraySetItem {
                array_id,
                segment_id,
            } => {
                synth.dynamic_ndarray_set_item(*array_id, *segment_id, &args[0], &args[1])?;
                None
            }

            // ── External function calls ──────────────────────────────
            // These are resolved during preprocessing. The prover skips
            // them — the external-computed values are already in the witness.
            IR::InvokeExternal { .. }
            | IR::ExportExternalI { .. }
            | IR::ExportExternalF { .. } => None,

            // ── Hash ──────────────────────────────────────────────────
            IR::PoseidonHash => Some(synth.poseidon_hash(&args)?),
            IR::EqHash => Some(synth.eq_hash(&args[0], &args[1])?),
        };

        if id < cells.len() {
            cells[id] = result;
        }
    }

    Ok(())
}


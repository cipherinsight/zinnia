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
/// The statements [`pure_value_op`] evaluates, as one pattern (used by
/// `is_pure_value_op` and by `interpret_ir`'s dispatch).
macro_rules! pure_value_ops {
    () => {
        IR::ConstantInt { .. }
            | IR::ConstantFloat { .. }
            | IR::ConstantBool { .. }
            | IR::AddI
            | IR::SubI
            | IR::MulI
            | IR::FloorDivI
            | IR::ModI
            | IR::AddF
            | IR::SubF
            | IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LteI
            | IR::GtI
            | IR::GteI
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::LogicalNot
            | IR::SelectI
            | IR::SelectF
            | IR::SelectB
    };
}

/// Whether [`pure_value_op`] evaluates `ir`: a statement whose value is a
/// function of its operands alone (no input, memory, hint or check).
pub fn is_pure_value_op(ir: &IR) -> bool {
    matches!(ir, pure_value_ops!())
}

/// The value of a statement `is_pure_value_op` accepts, on `synth`. This is
/// the dispatch `interpret_ir` uses for these statements, shared with
/// compile-time evaluation (`optim::select_merge`), so both read the same
/// semantics.
pub fn pure_value_op<S: Synthesizer>(synth: &mut S, ir: &IR, args: &[S::CellRef]) -> Result<S::CellRef, ProvingError> {
    match ir {
        IR::ConstantInt { value } => synth.constant_int(*value),
        IR::ConstantFloat { value } => synth.constant_float(*value),
        IR::ConstantBool { value } => synth.constant_bool(*value),
        IR::AddI => synth.add_i(&args[0], &args[1]),
        IR::SubI => synth.sub_i(&args[0], &args[1]),
        IR::MulI => synth.mul_i(&args[0], &args[1]),
        IR::FloorDivI => synth.floor_div_i(&args[0], &args[1]),
        IR::ModI => synth.mod_i(&args[0], &args[1]),
        IR::AddF => synth.add_f(&args[0], &args[1]),
        IR::SubF => synth.sub_f(&args[0], &args[1]),
        IR::EqI => synth.eq_i(&args[0], &args[1]),
        IR::NeI => synth.ne_i(&args[0], &args[1]),
        IR::LtI => synth.lt_i(&args[0], &args[1]),
        IR::LteI => synth.lte_i(&args[0], &args[1]),
        IR::GtI => synth.gt_i(&args[0], &args[1]),
        IR::GteI => synth.gte_i(&args[0], &args[1]),
        IR::LogicalAnd => synth.logical_and(&args[0], &args[1]),
        IR::LogicalOr => synth.logical_or(&args[0], &args[1]),
        IR::LogicalNot => synth.logical_not(&args[0]),
        IR::SelectI | IR::SelectF | IR::SelectB => synth.select(&args[0], &args[1], &args[2]),
        other => Err(ProvingError::synthesis(format!("not a pure value statement: {other:?}"))),
    }
}

pub fn interpret_ir<S: Synthesizer>(graph: &IRGraph, synth: &mut S) -> Result<(), ProvingError> {
    let n = graph.stmts.len();
    let mut cells: Vec<Option<S::CellRef>> = Vec::with_capacity(n);
    cells.resize_with(n, || None);

    for stmt in &graph.stmts {
        let id = stmt.stmt_id as usize;
        synth.on_stmt_begin(stmt.stmt_id);

        // Collect argument cell references.
        let args: Vec<S::CellRef> = stmt
            .arguments
            .iter()
            .map(|&arg_id| {
                cells[arg_id as usize].clone().ok_or_else(|| {
                    ProvingError::synthesis(format!(
                        "Statement {} references undefined cell {}",
                        id, arg_id
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let result: Option<S::CellRef> = match &stmt.ir {
            // Constants, total integer arithmetic, integer comparisons,
            // Boolean logic and selection: see `pure_value_op`.
            ir @ (pure_value_ops!()) => Some(pure_value_op(synth, ir, &args)?),

            // ── Constants ─────────────────────────────────────────────
            IR::ConstantStr { .. } => None,

            // ── Integer arithmetic ────────────────────────────────────
            IR::DivI => Some(synth.div_i(&args[0], &args[1])?),
            IR::FloorDivIPinned { divisor_bits } => {
                Some(synth.floor_div_i_pinned(&args[0], &args[1], *divisor_bits)?)
            }
            IR::PowI => Some(synth.pow_i(&args[0], &args[1])?),
            IR::AbsI => Some(synth.abs_i(&args[0])?),
            IR::SignI => Some(synth.sign_i(&args[0])?),
            IR::InvI => Some(synth.inv_i(&args[0])?),

            // ── Integer bitwise ───────────────────────────────────────
            IR::BitAndI => Some(synth.bit_and_i(&args[0], &args[1])?),
            IR::BitOrI => Some(synth.bit_or_i(&args[0], &args[1])?),
            IR::BitXorI => Some(synth.bit_xor_i(&args[0], &args[1])?),
            IR::ShlI => Some(synth.shl_i(&args[0], &args[1])?),
            IR::ShrI => Some(synth.shr_i(&args[0], &args[1])?),
            IR::BitNotI => Some(synth.bit_not_i(&args[0])?),

            // ── Float arithmetic ──────────────────────────────────────
            IR::SumF { len } => {
                if args.len() != *len as usize {
                    return Err(ProvingError::synthesis(format!(
                        "SumF expected {} operands, got {}",
                        len,
                        args.len()
                    )));
                }
                Some(synth.sum_f(&args)?)
            }
            IR::MulF => Some(synth.mul_f(&args[0], &args[1])?),
            IR::ProductSumF { len } => {
                let expected = (*len as usize).saturating_mul(2);
                if args.len() != expected {
                    return Err(ProvingError::synthesis(format!(
                        "ProductSumF expected {} operands for {} pairs, got {}",
                        expected,
                        len,
                        args.len()
                    )));
                }
                let pairs: Vec<(S::CellRef, S::CellRef)> = args
                    .chunks_exact(2)
                    .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
                    .collect();
                Some(synth.product_sum_f(&pairs)?)
            }
            IR::DivF => Some(synth.div_f(&args[0], &args[1])?),
            IR::FloorDivF => Some(synth.floor_div_f(&args[0], &args[1])?),
            IR::ModF => Some(synth.mod_f(&args[0], &args[1])?),
            IR::PowF => Some(synth.pow_f(&args[0], &args[1])?),
            IR::PowFI => Some(synth.pow_f_int(&args[0], &args[1])?),
            IR::AbsF => Some(synth.abs_f(&args[0])?),
            IR::SignF => Some(synth.sign_f(&args[0])?),

            // ── Integer comparisons ───────────────────────────────────
            IR::LtIBounded => Some(synth.lt_i_bounded(&args[0], &args[1])?),
            IR::GtIBounded => Some(synth.gt_i_bounded(&args[0], &args[1])?),
            IR::IndexInBox { rank } => {
                // args laid out flat as [idx_0, lo_0, hi_0, idx_1, lo_1, hi_1, ...].
                let mut triples: Vec<(_, _, _)> = Vec::with_capacity(*rank);
                for k in 0..*rank {
                    triples.push((
                        args[k * 3].clone(),
                        args[k * 3 + 1].clone(),
                        args[k * 3 + 2].clone(),
                    ));
                }
                Some(synth.index_in_box(&triples)?)
            }

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
            IR::SqrtProductIF => Some(synth.sqrt_product_i_f(&args[0], &args[1])?),
            IR::ExpF => Some(synth.exp_f(&args[0])?),
            IR::LogF => Some(synth.log_f(&args[0])?),
            IR::ArcSinF => Some(synth.arcsin_f(&args[0])?),
            IR::ArcCosF => Some(synth.arccos_f(&args[0])?),
            IR::ArcTan2F => Some(synth.arctan2_f(&args[0], &args[1])?),

            // ── Boolean logic ─────────────────────────────────────────

            // ── Selection (mux) ───────────────────────────────────────
            IR::SelectBitI | IR::SelectBitF => {
                Some(synth.select_bit(&args[0], &args[1], &args[2])?)
            }
            IR::SelectManyI { len, lane } | IR::SelectManyF { len, lane } => {
                let len = *len as usize;
                let lane = *lane as usize;
                let true_values = args[1..1 + len].to_vec();
                let false_values = args[1 + len..1 + len * 2].to_vec();
                Some(synth.select_many_lane(
                    &args[0],
                    &true_values,
                    &false_values,
                    lane,
                )?)
            }
            // ── Casting ───────────────────────────────────────────────
            IR::IntCast => Some(synth.int_cast(&args[0])?),
            IR::FloatCast => Some(synth.float_cast(&args[0])?),
            IR::BoolCast => Some(synth.bool_cast(&args[0])?),

            // ── String operations (debug only, no circuit) ────────────
            IR::AddStr | IR::StrI | IR::StrF => None,

            // ── I/O ───────────────────────────────────────────────────
            IR::ReadInteger { path, is_public } => Some(synth.read_input(path, *is_public)?),
            IR::ReadFloat { path, is_public } => Some(synth.read_input(path, *is_public)?),
            IR::ReadHash { path, is_public } => Some(synth.read_input(path, *is_public)?),
            IR::ReadExternalResult {
                store_idx,
                output_idx,
                ..
            } => Some(synth.read_external_result(*store_idx, *output_idx)?),
            IR::Print => None,

            // ── Assertions & public exposure ──────────────────────────
            IR::Assert => {
                synth.assert_true(&args[0])?;
                None
            }
            IR::AssertBit => {
                synth.assert_bit(&args[0])?;
                None
            }
            IR::AssertNonNegI => {
                synth.assert_non_neg_i(&args[0])?;
                None
            }
            IR::AssertU128Lt => {
                synth.assert_u128_lt(&args[0], &args[1])?;
                None
            }
            IR::AssertU128LtBit { hi, bit, bits } => {
                Some(synth.assert_u128_lt_bit(&args[0], *hi, *bit, *bits)?)
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

            // ── Structural-predicate precondition (inert at prove time) ──
            // The IR carries the precondition payload for the SMT layer;
            // the prover treats it as a no-op (no stack args, no result).
            IR::StructuralPredicate { .. } => None,

            // ── Scalar precondition (inert at prove time) ────────────
            // Proof-time enforcement lives in the separate IR emitted
            // by `witness::lower_precondition_to_ir`. This atom itself
            // carries only the compile-time fact for the SMT layer.
            IR::ScalarPrecondition { .. } => None,

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
            IR::ReadMemory { segment_id } => Some(synth.read_memory(*segment_id, &args[0])?),
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
                max_shape,
            } => {
                synth.allocate_dynamic_ndarray_meta(
                    *array_id,
                    dtype_name,
                    *max_length,
                    *max_rank,
                    max_shape,
                )?;
                None
            }
            IR::WitnessDynamicNDArrayMeta {
                array_id,
                max_rank,
                param_name,
            } => {
                synth.witness_dynamic_ndarray_meta(*array_id, *max_rank, param_name, &args)?;
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
            IR::InvokeExternal { .. } | IR::ExportExternalI { .. } | IR::ExportExternalF { .. } => {
                None
            }

            // ── Hash ──────────────────────────────────────────────────
            IR::PoseidonHash => Some(synth.poseidon_hash(&args)?),
            IR::EqHash => Some(synth.eq_hash(&args[0], &args[1])?),

            // ── Prover hints ──────────────────────────────────────────
            IR::Hint { kind, len } => {
                let len = *len as usize;
                if args.len() != len + 1 {
                    return Err(ProvingError::synthesis(format!(
                        "Hint expected {} operands and a rank, got {} arguments",
                        len,
                        args.len()
                    )));
                }
                Some(synth.prover_hint(*kind, &args[..len], &args[len])?)
            }
            IR::KeyLess { index_bits } => {
                Some(synth.key_less(&args[0], &args[1], &args[2], &args[3], *index_bits)?)
            }
            IR::PackKey { index_bits } => Some(synth.pack_key(&args[0], &args[1], *index_bits)?),
            IR::PackedKeyLess { index_bits } => {
                Some(synth.packed_key_less(&args[0], &args[1], *index_bits)?)
            }
            IR::PackedKeyIndex { index_bits } => Some(synth.packed_key_index(&args[0], *index_bits)?),
            IR::ArrayOp(_) => {
                return Err(ProvingError::synthesis(
                    "an ArrayOp placeholder reached the prover; lower::ArrayLowering must lower it",
                ));
            }
        };

        if id < cells.len() {
            if let Some(cell) = &result {
                synth.on_stmt_result(stmt.stmt_id, stmt.value_id, cell);
            }
            cells[id] = result;
        }
    }

    Ok(())
}

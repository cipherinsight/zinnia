//! Preprocessing stage: resolves external function calls before proving.
//!
//! The preprocessor walks the IR graph, evaluates all instructions using
//! Fp field arithmetic (same as the mock/halo2 synthesizers), and invokes
//! external functions via a callback. External return values are converted
//! back to `Value` and merged into the witness.
//!
//! # Architecture
//!
//! ```text
//! IR graph ──► Preprocessor ──► ResolvedWitness
//!                    │
//!             ExternalCallback trait
//!                    │
//!          ┌─────────┴─────────┐
//!          │                   │
//!   PyExternalCallback    FnExternalCallback
//!   (Python via PyO3)     (test closures)
//! ```

pub mod callback;
pub mod py_callback;

use std::collections::HashMap;

use crate::prove::field::F as Fp;

use std::collections::HashMap as StdHashMap;

use crate::circuit_input::{CircuitInputs, ResolvedWitness};
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::error::ProvingError;
use crate::prove::kernel::{self, Field};
use crate::prove::types::{ProvingParams, Value};

use self::callback::ExternalCallback;

/// Execute preprocessing on the IR graph: resolve external calls, build resolved witness.
pub fn run_preprocess(
    ir: &IRGraph,
    witness: &CircuitInputs,
    params: &ProvingParams,
    callback: &dyn ExternalCallback,
) -> Result<ResolvedWitness, ProvingError> {
    let mut resolved = ResolvedWitness::new(witness.clone(), params.precision_bits);

    if ir.stmts.is_empty() {
        return Ok(resolved);
    }

    let mut preprocessor = Preprocessor::new(witness, params.precision_bits, callback);
    preprocessor.execute(ir)?;

    // Transfer external results to the resolved witness
    resolved.external_results = preprocessor.external_results;
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// Preprocessor
// ---------------------------------------------------------------------------

struct Preprocessor<'a> {
    values: Vec<Option<Fp>>,
    witness: CircuitInputs,
    precision_bits: u32,
    external_arg_store: HashMap<u32, Vec<ExternArg>>,
    external_results: StdHashMap<(u32, u32), Fp>,
    callback: &'a dyn ExternalCallback,
    memories: HashMap<u32, Vec<Fp>>,
    memory_init: HashMap<u32, Fp>,
}

#[derive(Debug, Clone)]
struct ExternArg {
    value: Fp,
    is_float: bool,
}

impl<'a> Preprocessor<'a> {
    fn new(
        witness: &CircuitInputs,
        precision_bits: u32,
        callback: &'a dyn ExternalCallback,
    ) -> Self {
        Self {
            values: Vec::new(),
            witness: witness.clone(),
            precision_bits,
            external_arg_store: HashMap::new(),
            external_results: StdHashMap::new(),
            callback,
            memories: HashMap::new(),
            memory_init: HashMap::new(),
        }
    }

    fn execute(&mut self, graph: &IRGraph) -> Result<(), ProvingError> {
        self.values.resize(graph.stmts.len(), None);
        for stmt in &graph.stmts {
            let id = stmt.stmt_id as usize;
            // Collect operand values positionally. A missing value here means
            // a CONSUMED operand has no produced value — error loudly rather
            // than letting `filter_map` compact the list and silently shift
            // every later operand left (mirrors `interpreter.rs` strictness).
            let args: Vec<Fp> = stmt
                .arguments
                .iter()
                .map(|&arg_id| {
                    self.values[arg_id as usize].ok_or_else(|| {
                        ProvingError::synthesis(format!(
                            "Statement {} references operand {} with no produced value \
                             (preprocessor cannot evaluate its producer)",
                            id, arg_id
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let result = self.eval_ir(&stmt.ir, &args, id)?;
            if id < self.values.len() {
                self.values[id] = result;
            }
        }
        Ok(())
    }

    fn eval_ir(
        &mut self,
        ir: &IR,
        args: &[Fp],
        stmt_id: usize,
    ) -> Result<Option<Fp>, ProvingError> {
        let prec = self.precision_bits;
        match ir {
            // Constants
            IR::ConstantInt { value } => Ok(Some(kernel::i64_to_fp(*value))),
            IR::ConstantFloat { value } => Ok(Some(kernel::quantize_to_fp(*value, prec))),
            IR::ConstantBool { value } => Ok(Some(if *value { Fp::one() } else { Fp::zero() })),
            IR::ConstantStr { .. } => Ok(None),

            // Inputs
            IR::ReadInteger { path, .. }
            | IR::ReadFloat { path, .. }
            | IR::ReadHash { path, .. } => {
                // The preprocessor only runs on the prove path with a real
                // witness present (keygen never calls it). A missing path here
                // means a misspelled/absent input — propagate the
                // `witness_missing` error instead of silently substituting zero
                // (which would yield a consistent all-zeros run that proves and
                // verifies, masking the user's typo).
                Ok(Some(self.witness.resolve(path, self.precision_bits)?))
            }
            IR::ReadExternalResult {
                store_idx,
                output_idx,
                ..
            } => {
                // Reading an external result that was never produced (e.g. an
                // `output_idx` the call never returned) must error, not silently
                // yield zero — mirrors `ResolvedWitness::resolve_external` and
                // the mock/halo2 synthesizers' `witness_missing` behaviour.
                self.external_results
                    .get(&(*store_idx, *output_idx))
                    .copied()
                    .map(Some)
                    .ok_or_else(|| {
                        ProvingError::witness_missing(format!(
                            "__ext.{}.{}",
                            store_idx, output_idx
                        ))
                    })
            }

            // Integer arithmetic
            IR::AddI => Ok(Some(args[0] + args[1])),
            IR::SubI => Ok(Some(args[0] - args[1])),
            IR::MulI => Ok(Some(args[0] * args[1])),
            IR::DivI => Ok(Some(args[0] * args[1].invert().unwrap_or(Fp::zero()))),
            IR::FloorDivI | IR::FloorDivIPinned { .. } => {
                let (q, _) = kernel::fp_floor_div(args[0], args[1]);
                Ok(Some(q))
            }
            IR::ModI => {
                let (_, r) = kernel::fp_floor_div(args[0], args[1]);
                Ok(Some(r))
            }
            IR::PowI => {
                let e = kernel::fp_to_i64(args[1]);
                let mut r = Fp::one();
                for _ in 0..e.abs().min(64) {
                    r = r * args[0];
                }
                if e < 0 {
                    r = r.invert().unwrap_or(Fp::zero());
                }
                Ok(Some(r))
            }
            IR::AbsI => {
                let (abs, _) = kernel::signed_decompose(args[0]);
                Ok(Some(abs))
            }
            IR::SignI => {
                let v = args[0];
                Ok(Some(if v == Fp::zero() {
                    Fp::zero()
                } else if kernel::fp_is_negative(v) {
                    -Fp::one()
                } else {
                    Fp::one()
                }))
            }
            IR::InvI => Ok(Some(args[0].invert().unwrap_or(Fp::zero()))),

            // Integer bitwise (preprocess: native i64)
            IR::BitAndI => {
                let av = kernel::fp_to_i64(args[0]);
                let bv = kernel::fp_to_i64(args[1]);
                Ok(Some(kernel::i64_to_fp(av & bv)))
            }
            IR::BitOrI => {
                let av = kernel::fp_to_i64(args[0]);
                let bv = kernel::fp_to_i64(args[1]);
                Ok(Some(kernel::i64_to_fp(av | bv)))
            }
            IR::BitXorI => {
                let av = kernel::fp_to_i64(args[0]);
                let bv = kernel::fp_to_i64(args[1]);
                Ok(Some(kernel::i64_to_fp(av ^ bv)))
            }
            IR::ShlI => {
                let av = kernel::fp_to_i64(args[0]);
                let bv = kernel::fp_to_i64(args[1]).max(0).min(63) as u32;
                Ok(Some(kernel::i64_to_fp(av.wrapping_shl(bv))))
            }
            IR::ShrI => {
                let av = kernel::fp_to_i64(args[0]);
                let bv = kernel::fp_to_i64(args[1]).max(0).min(63) as u32;
                Ok(Some(kernel::i64_to_fp(av.wrapping_shr(bv))))
            }
            IR::BitNotI => {
                let av = kernel::fp_to_i64(args[0]);
                Ok(Some(kernel::i64_to_fp(!av)))
            }

            // Float arithmetic
            IR::AddF => Ok(Some(args[0] + args[1])),
            IR::SumF { len } => {
                if args.len() != *len as usize {
                    return Err(ProvingError::synthesis(format!(
                        "SumF expected {} operands, got {}",
                        len,
                        args.len()
                    )));
                }
                Ok(Some(
                    args.iter().copied().fold(Fp::zero(), |acc, v| acc + v),
                ))
            }
            IR::SubF => Ok(Some(args[0] - args[1])),
            IR::MulF => {
                let (r, _) = kernel::fp_mul_rescale(args[0], args[1], prec);
                Ok(Some(r))
            }
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
                let pairs: Vec<(Fp, Fp)> = args
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect();
                let (result, _) = kernel::fp_product_sum_rescale(&pairs, prec);
                Ok(Some(result))
            }
            IR::DivF => Ok(Some(kernel::fp_div_prescale(args[0], args[1], prec))),
            IR::FloorDivF => Ok(Some(kernel::fp_float_floor_div(args[0], args[1], prec))),
            IR::ModF => Ok(Some(kernel::fp_float_mod(args[0], args[1]))),
            IR::PowF => {
                let log_a = kernel::fp_log(args[0], prec);
                let (bl, _) = kernel::fp_mul_rescale(args[1], log_a, prec);
                Ok(Some(kernel::fp_exp(bl, prec)))
            }
            IR::PowFI => {
                // Float base ^ Integer exponent: square-and-multiply over the
                // unscaled integer. args[0] is Q-scaled float, args[1] is raw
                // integer field element.
                let e = kernel::fp_to_i64(args[1]);
                let mut result = kernel::quantize_to_fp(1.0, prec);
                let mut base = args[0];
                let mut e_rem = e.unsigned_abs();
                let mut iters = 0;
                while e_rem != 0 && iters < 64 {
                    if (e_rem & 1) == 1 {
                        let (rb, _) = kernel::fp_mul_rescale(result, base, prec);
                        result = rb;
                    }
                    e_rem >>= 1;
                    if e_rem != 0 {
                        let (bb, _) = kernel::fp_mul_rescale(base, base, prec);
                        base = bb;
                    }
                    iters += 1;
                }
                if e < 0 {
                    // Negative exponent: invert in Q-scaled float (== Q^2 / result).
                    let one_q = kernel::quantize_to_fp(1.0, prec);
                    result = kernel::fp_div_prescale(one_q, result, prec);
                }
                Ok(Some(result))
            }
            IR::AbsF => {
                let (abs, _) = kernel::signed_decompose(args[0]);
                Ok(Some(abs))
            }
            IR::SignF => self.eval_ir(&IR::SignI, args, stmt_id),

            // Comparisons
            IR::EqI | IR::EqF => Ok(Some(if args[0] == args[1] {
                Fp::one()
            } else {
                Fp::zero()
            })),
            IR::NeI | IR::NeF => Ok(Some(if args[0] != args[1] {
                Fp::one()
            } else {
                Fp::zero()
            })),
            IR::LtI | IR::LtIBounded | IR::LtF => {
                let d = args[0] - args[1];
                Ok(Some(if kernel::fp_is_negative(d) && d != Fp::zero() {
                    Fp::one()
                } else {
                    Fp::zero()
                }))
            }
            IR::LteI | IR::LteF => {
                let d = args[0] - args[1];
                Ok(Some(if kernel::fp_is_negative(d) || d == Fp::zero() {
                    Fp::one()
                } else {
                    Fp::zero()
                }))
            }
            IR::GtI | IR::GtIBounded | IR::GtF => {
                self.eval_ir(&IR::LtI, &[args[1], args[0]], stmt_id)
            }
            IR::GteI | IR::GteF => self.eval_ir(&IR::LteI, &[args[1], args[0]], stmt_id),
            IR::IndexInBox { rank } => {
                // args laid out flat as [idx_0, lo_0, hi_0, idx_1, lo_1, hi_1, ...].
                let mut all_in = true;
                for k in 0..*rank {
                    let idx = args[k * 3];
                    let lo = args[k * 3 + 1];
                    let hi = args[k * 3 + 2];
                    let ge_lo = {
                        let d = idx - lo;
                        !kernel::fp_is_negative(d) || d == Fp::zero()
                    };
                    let lt_hi = {
                        let d = idx - hi;
                        kernel::fp_is_negative(d) && d != Fp::zero()
                    };
                    if !(ge_lo && lt_hi) {
                        all_in = false;
                        break;
                    }
                }
                Ok(Some(if all_in { Fp::one() } else { Fp::zero() }))
            }

            // Transcendentals
            IR::SinF => Ok(Some(kernel::fp_sin(args[0], prec))),
            IR::SinHF => Ok(Some(kernel::fp_sinh(args[0], prec))),
            IR::CosF => Ok(Some(kernel::fp_cos(args[0], prec))),
            IR::CosHF => Ok(Some(kernel::fp_cosh(args[0], prec))),
            IR::TanF => Ok(Some(kernel::fp_tan(args[0], prec))),
            IR::TanHF => Ok(Some(kernel::fp_tanh(args[0], prec))),
            IR::SqrtF => Ok(Some(kernel::fp_sqrt(args[0], prec))),
            IR::ExpF => Ok(Some(kernel::fp_exp(args[0], prec))),
            IR::LogF => Ok(Some(kernel::fp_log(args[0], prec))),
            IR::ArcSinF => Ok(Some(kernel::fp_arcsin(args[0], prec))),
            IR::ArcCosF => Ok(Some(kernel::fp_arccos(args[0], prec))),
            IR::ArcTan2F => Ok(Some(kernel::fp_atan2(args[0], args[1], prec))),

            // Logic
            IR::LogicalAnd => Ok(Some(if args[0] != Fp::zero() && args[1] != Fp::zero() {
                Fp::one()
            } else {
                Fp::zero()
            })),
            IR::LogicalOr => Ok(Some(if args[0] != Fp::zero() || args[1] != Fp::zero() {
                Fp::one()
            } else {
                Fp::zero()
            })),
            IR::LogicalNot => Ok(Some(if args[0] == Fp::zero() {
                Fp::one()
            } else {
                Fp::zero()
            })),

            // Prover hints (the same function as every backend)
            IR::Hint { kind, len } => {
                let len = *len as usize;
                if args.len() != len + 1 {
                    return Err(ProvingError::synthesis(format!(
                        "Hint expected {} operands and a rank, got {} arguments",
                        len,
                        args.len()
                    )));
                }
                Ok(Some(kernel::hint_lane(*kind, &args[..len], args[len])))
            }
            IR::Sorted { .. } => Err(ProvingError::synthesis(
                "a Sorted placeholder reached preprocessing; optim::sort_lowering must lower it",
            )),

            // Select
            IR::SelectI | IR::SelectF | IR::SelectB | IR::SelectBitI | IR::SelectBitF => Ok(Some(if args[0] != Fp::zero() {
                args[1]
            } else {
                args[2]
            })),
            IR::SelectManyI { len, lane } | IR::SelectManyF { len, lane } => {
                let len = *len as usize;
                let lane = *lane as usize;
                Ok(Some(if args[0] != Fp::zero() {
                    args[1 + lane]
                } else {
                    args[1 + len + lane]
                }))
            }
            // Casting
            IR::IntCast => {
                let (q, _r) = kernel::fp_int_cast(args[0], prec);
                Ok(Some(q))
            }
            IR::FloatCast => Ok(Some(args[0] * kernel::scale_fp(prec))),
            IR::BoolCast => Ok(Some(if args[0] != Fp::zero() {
                Fp::one()
            } else {
                Fp::zero()
            })),
            IR::AssertU128LtBit { hi, bit, .. } => {
                let idx = kernel::fp_to_i64(args[0]);
                if idx < 0 || idx as u64 >= *hi {
                    Ok(Some(Fp::zero()))
                } else {
                    Ok(Some(Fp::from(((idx as u64 >> bit) & 1) as u64)))
                }
            }

            // No-ops
            IR::AddStr | IR::StrI | IR::StrF | IR::Print => Ok(None),
            IR::Assert
            | IR::AssertBit
            | IR::AssertNonNegI
            | IR::AssertU128Lt
            | IR::ExposePublicI
            | IR::ExposePublicF => Ok(None),
            IR::StructuralPredicate { .. } => Ok(None),
            IR::ScalarPrecondition { .. } => Ok(None),
            IR::MemoryTraceEmit { .. } | IR::MemoryTraceSeal => Ok(None),
            IR::WitnessDynamicNDArrayMeta { .. } | IR::AssertDynamicNDArrayMeta { .. } => Ok(None),

            // Memory
            IR::AllocateMemory {
                segment_id,
                size,
                init_value,
            } => {
                let init = kernel::i64_to_fp(*init_value);
                self.memories
                    .insert(*segment_id, vec![init; *size as usize]);
                self.memory_init.insert(*segment_id, init);
                Ok(None)
            }
            IR::WriteMemory { segment_id } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                // Match `mock/synthesizer.rs::write_memory`: a write to an
                // unallocated segment or out-of-bounds address is an error, not
                // a silent no-op.
                let mem = self.memories.get_mut(segment_id).ok_or_else(|| {
                    ProvingError::synthesis(format!(
                        "Memory write to unallocated segment {}",
                        segment_id
                    ))
                })?;
                if addr >= mem.len() {
                    return Err(ProvingError::synthesis(format!(
                        "Memory write out of bounds: segment {} addr {} len {}",
                        segment_id,
                        addr,
                        mem.len()
                    )));
                }
                mem[addr] = args[1];
                Ok(None)
            }
            IR::ReadMemory { segment_id } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                let init = self
                    .memory_init
                    .get(segment_id)
                    .copied()
                    .unwrap_or(Fp::zero());
                Ok(Some(
                    self.memories
                        .get(segment_id)
                        .and_then(|m| m.get(addr).copied())
                        .unwrap_or(init),
                ))
            }

            // Dynamic NDArray
            IR::AllocateDynamicNDArrayMeta {
                array_id,
                max_length,
                ..
            } => {
                self.memories
                    .insert(10000 + array_id, vec![Fp::zero(); *max_length as usize]);
                self.memory_init.insert(10000 + array_id, Fp::zero());
                Ok(None)
            }
            IR::DynamicNDArrayGetItem { array_id, .. } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                let seg = 10000 + array_id;
                Ok(Some(
                    self.memories
                        .get(&seg)
                        .and_then(|m| m.get(addr).copied())
                        .unwrap_or(Fp::zero()),
                ))
            }
            IR::DynamicNDArraySetItem { array_id, .. } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                let seg = 10000 + array_id;
                if let Some(mem) = self.memories.get_mut(&seg) {
                    if addr < mem.len() {
                        mem[addr] = args[1];
                    }
                }
                Ok(None)
            }

            // External function calls
            IR::ExportExternalI { for_which, .. } => {
                self.external_arg_store
                    .entry(*for_which)
                    .or_default()
                    .push(ExternArg {
                        value: args[0],
                        is_float: false,
                    });
                Ok(None)
            }
            IR::ExportExternalF { for_which, .. } => {
                self.external_arg_store
                    .entry(*for_which)
                    .or_default()
                    .push(ExternArg {
                        value: args[0],
                        is_float: true,
                    });
                Ok(None)
            }
            IR::InvokeExternal {
                store_idx,
                func_name,
                ..
            } => self.invoke_external(*store_idx, func_name, stmt_id),

            // Hash
            IR::PoseidonHash => Ok(Some(kernel::fp_poseidon(&args))),
            IR::EqHash => Ok(Some(if args[0] == args[1] {
                Fp::one()
            } else {
                Fp::zero()
            })),
        }
    }

    fn invoke_external(
        &mut self,
        store_idx: u32,
        func_name: &str,
        _stmt_id: usize,
    ) -> Result<Option<Fp>, ProvingError> {
        let ext_args = self
            .external_arg_store
            .remove(&store_idx)
            .unwrap_or_default();

        // Convert Fp → Value for the callback (human-readable)
        let call_args: Vec<Value> = ext_args
            .iter()
            .map(|arg| kernel::fp_to_value(arg.value, arg.is_float, self.precision_bits))
            .collect();

        // Call external function
        let result = self.callback.call(func_name, call_args)?;

        // Store external results keyed by (store_idx, output_idx). A list return
        // fans out to one entry per element so reads of higher output indices
        // resolve to the correct value instead of silently returning zero. A
        // scalar return is stored at output index 0.
        match &result {
            Value::List(items) => {
                if items.is_empty() {
                    return Err(ProvingError::synthesis(format!(
                        "External function '{}' returned an empty list",
                        func_name
                    )));
                }
                let mut first_fp = Fp::zero();
                for (k, item) in items.iter().enumerate() {
                    let fp = kernel::value_to_fp(item, self.precision_bits)?;
                    self.external_results.insert((store_idx, k as u32), fp);
                    if k == 0 {
                        first_fp = fp;
                    }
                }
                Ok(Some(first_fp))
            }
            _ => {
                let result_fp = kernel::value_to_fp(&result, self.precision_bits)?;
                self.external_results.insert((store_idx, 0), result_fp);
                Ok(Some(result_fp))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRGraph, IRStatement};
    use crate::prove::preprocess::callback::FnExternalCallback;

    fn make_graph(stmts: Vec<(IR, Vec<u32>)>) -> IRGraph {
        let ir_stmts: Vec<IRStatement> = stmts
            .into_iter()
            .enumerate()
            .map(|(i, (ir, args))| {
                IRStatement::new(
                    i as u32,
                    crate::types::ValueId::next(),
                    ir,
                    args,
                    vec![],
                    None,
                )
            })
            .collect();
        IRGraph::new(ir_stmts)
    }

    #[test]
    fn test_empty_preprocess() {
        let graph = IRGraph::new(vec![]);
        let witness = CircuitInputs::new();
        let params = ProvingParams::default();
        let cb = callback::NoExternalCallback;
        let result = run_preprocess(&graph, &witness, &params, &cb).unwrap();
        assert!(result.external_results.is_empty());
    }

    #[test]
    fn test_preprocess_arithmetic() {
        let graph = make_graph(vec![
            (IR::ConstantInt { value: 3 }, vec![]),
            (IR::ConstantInt { value: 5 }, vec![]),
            (IR::AddI, vec![0, 1]),
        ]);
        let cb = callback::NoExternalCallback;
        let result = run_preprocess(
            &graph,
            &CircuitInputs::new(),
            &ProvingParams::default(),
            &cb,
        )
        .unwrap();
        assert!(result.external_results.is_empty());
    }

    #[test]
    fn test_preprocess_errors_on_missing_operand() {
        // stmt1 (Print) produces no value; stmt2 consumes it as an operand.
        // The preprocessor must error loudly rather than `filter_map`-compact
        // the operand list (which would silently shift later operands left and
        // here would evaluate AddI with a single operand, panicking or — worse —
        // producing a wrong-but-valid value).
        let graph = make_graph(vec![
            (IR::ConstantInt { value: 7 }, vec![]),
            (IR::Print, vec![0]),
            (IR::AddI, vec![1, 0]),
        ]);
        let cb = callback::NoExternalCallback;
        let err = run_preprocess(
            &graph,
            &CircuitInputs::new(),
            &ProvingParams::default(),
            &cb,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no produced value") && msg.contains("operand"),
            "expected a missing-operand error, got: {msg}"
        );
    }

    #[test]
    fn test_preprocess_errors_on_missing_witness_path() {
        // A ReadInteger whose path is absent from the witness must surface a
        // `witness_missing` error instead of silently substituting zero.
        let path = crate::circuit_input::InputPath {
            param: "nonexistent_param".to_string(),
            segments: vec![],
        };
        let graph = make_graph(vec![(
            IR::ReadInteger {
                path,
                is_public: false,
            },
            vec![],
        )]);
        let cb = callback::NoExternalCallback;
        let err = run_preprocess(
            &graph,
            &CircuitInputs::new(),
            &ProvingParams::default(),
            &cb,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("nonexistent_param")
                || msg.to_lowercase().contains("witness")
                || msg.to_lowercase().contains("missing"),
            "expected a witness-missing error, got: {msg}"
        );
    }

    #[test]
    fn test_preprocess_with_external_call() {
        let graph = make_graph(vec![
            (IR::ConstantInt { value: 10 }, vec![]),
            (
                IR::ExportExternalI {
                    for_which: 0,
                    key: crate::ir_defs::ExternalKey::Int(0),
                    indices: vec![0],
                },
                vec![0],
            ),
            (
                IR::InvokeExternal {
                    store_idx: 0,
                    func_name: "double".to_string(),
                    args: vec![],
                    kwargs: std::collections::HashMap::new(),
                },
                vec![],
            ),
        ]);

        let cb = FnExternalCallback {
            func: |name, args| {
                assert_eq!(name, "double");
                match &args[0] {
                    Value::Integer(v) => Ok(Value::Integer(v * 2)),
                    _ => Err(ProvingError::other("Expected integer")),
                }
            },
        };

        let result = run_preprocess(
            &graph,
            &CircuitInputs::new(),
            &ProvingParams::default(),
            &cb,
        )
        .unwrap();
        assert_eq!(result.external_results.len(), 1);
        let fp = result.external_results.get(&(0, 0)).unwrap();
        assert_eq!(crate::prove::kernel::fp_to_i64(*fp), 20);
    }
}

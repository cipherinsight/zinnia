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

use pasta_curves::Fp;

use std::collections::HashMap as StdHashMap;

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::error::ProvingError;
use crate::prove::kernel::{self, Field};
use crate::prove::types::{ProvingParams, Value};
use crate::circuit_input::{ResolvedWitness, CircuitInputs};

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
    fn new(witness: &CircuitInputs, precision_bits: u32, callback: &'a dyn ExternalCallback) -> Self {
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
            let args: Vec<Fp> = stmt.arguments.iter()
                .filter_map(|&arg_id| self.values[arg_id as usize])
                .collect();
            let result = self.eval_ir(&stmt.ir, &args, id)?;
            if id < self.values.len() {
                self.values[id] = result;
            }
        }
        Ok(())
    }

    fn eval_ir(&mut self, ir: &IR, args: &[Fp], stmt_id: usize) -> Result<Option<Fp>, ProvingError> {
        let prec = self.precision_bits;
        match ir {
            // Constants
            IR::ConstantInt { value } => Ok(Some(kernel::i64_to_fp(*value))),
            IR::ConstantFloat { value } => Ok(Some(kernel::quantize_to_fp(*value, prec))),
            IR::ConstantBool { value } => Ok(Some(if *value { Fp::one() } else { Fp::zero() })),
            IR::ConstantStr { .. } => Ok(None),

            // Inputs
            IR::ReadInteger { path, .. } | IR::ReadFloat { path, .. } | IR::ReadHash { path, .. } => {
                Ok(Some(self.witness.resolve(path, self.precision_bits).unwrap_or(Fp::zero())))
            }
            IR::ReadExternalResult { store_idx, output_idx, .. } => {
                Ok(Some(self.external_results.get(&(*store_idx, *output_idx)).copied().unwrap_or(Fp::zero())))
            }

            // Integer arithmetic
            IR::AddI => Ok(Some(args[0] + args[1])),
            IR::SubI => Ok(Some(args[0] - args[1])),
            IR::MulI => Ok(Some(args[0] * args[1])),
            IR::DivI => Ok(Some(args[0] * args[1].invert().unwrap_or(Fp::zero()))),
            IR::FloorDivI => { let (q, _) = kernel::fp_floor_div(args[0], args[1]); Ok(Some(q)) }
            IR::ModI => { let (_, r) = kernel::fp_floor_div(args[0], args[1]); Ok(Some(r)) }
            IR::PowI => {
                let e = kernel::fp_to_i64(args[1]);
                let mut r = Fp::one();
                for _ in 0..e.abs().min(64) { r = r * args[0]; }
                if e < 0 { r = r.invert().unwrap_or(Fp::zero()); }
                Ok(Some(r))
            }
            IR::AbsI => { let (abs, _) = kernel::signed_decompose(args[0]); Ok(Some(abs)) }
            IR::SignI => {
                let v = args[0];
                Ok(Some(if v == Fp::zero() { Fp::zero() } else if kernel::fp_is_negative(v) { -Fp::one() } else { Fp::one() }))
            }
            IR::InvI => Ok(Some(args[0].invert().unwrap_or(Fp::zero()))),

            // Float arithmetic
            IR::AddF => Ok(Some(args[0] + args[1])),
            IR::SubF => Ok(Some(args[0] - args[1])),
            IR::MulF => { let (r, _) = kernel::fp_mul_rescale(args[0], args[1], prec); Ok(Some(r)) }
            IR::DivF => Ok(Some(kernel::fp_div_prescale(args[0], args[1], prec))),
            IR::FloorDivF => Ok(Some(kernel::fp_div_prescale(args[0], args[1], prec))),
            IR::ModF => {
                let q = kernel::fp_div_prescale(args[0], args[1], prec);
                let (qb, _) = kernel::fp_mul_rescale(q, args[1], prec);
                Ok(Some(args[0] - qb))
            }
            IR::PowF => {
                let log_a = kernel::fp_log(args[0], prec);
                let (bl, _) = kernel::fp_mul_rescale(args[1], log_a, prec);
                Ok(Some(kernel::fp_exp(bl, prec)))
            }
            IR::AbsF => { let (abs, _) = kernel::signed_decompose(args[0]); Ok(Some(abs)) }
            IR::SignF => self.eval_ir(&IR::SignI, args, stmt_id),

            // Comparisons
            IR::EqI | IR::EqF => Ok(Some(if args[0] == args[1] { Fp::one() } else { Fp::zero() })),
            IR::NeI | IR::NeF => Ok(Some(if args[0] != args[1] { Fp::one() } else { Fp::zero() })),
            IR::LtI | IR::LtF => {
                let d = args[0] - args[1];
                Ok(Some(if kernel::fp_is_negative(d) && d != Fp::zero() { Fp::one() } else { Fp::zero() }))
            }
            IR::LteI | IR::LteF => {
                let d = args[0] - args[1];
                Ok(Some(if kernel::fp_is_negative(d) || d == Fp::zero() { Fp::one() } else { Fp::zero() }))
            }
            IR::GtI | IR::GtF => self.eval_ir(&IR::LtI, &[args[1], args[0]], stmt_id),
            IR::GteI | IR::GteF => self.eval_ir(&IR::LteI, &[args[1], args[0]], stmt_id),

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

            // Logic
            IR::LogicalAnd => Ok(Some(if args[0] != Fp::zero() && args[1] != Fp::zero() { Fp::one() } else { Fp::zero() })),
            IR::LogicalOr => Ok(Some(if args[0] != Fp::zero() || args[1] != Fp::zero() { Fp::one() } else { Fp::zero() })),
            IR::LogicalNot => Ok(Some(if args[0] == Fp::zero() { Fp::one() } else { Fp::zero() })),

            // Select
            IR::SelectI | IR::SelectF | IR::SelectB => {
                Ok(Some(if args[0] != Fp::zero() { args[1] } else { args[2] }))
            }

            // Casting
            IR::IntCast => {
                let s = crate::prove::field::quantization_scale(prec) as i64;
                Ok(Some(kernel::i64_to_fp(kernel::fp_to_i64(args[0]) / s)))
            }
            IR::FloatCast => Ok(Some(args[0] * kernel::scale_fp(prec))),
            IR::BoolCast => Ok(Some(if args[0] != Fp::zero() { Fp::one() } else { Fp::zero() })),

            // No-ops
            IR::AddStr | IR::StrI | IR::StrF | IR::Print => Ok(None),
            IR::Assert | IR::ExposePublicI | IR::ExposePublicF => Ok(None),
            IR::MemoryTraceEmit { .. } | IR::MemoryTraceSeal => Ok(None),
            IR::WitnessDynamicNDArrayMeta { .. } | IR::AssertDynamicNDArrayMeta { .. } => Ok(None),

            // Memory
            IR::AllocateMemory { segment_id, size, init_value } => {
                let init = kernel::i64_to_fp(*init_value);
                self.memories.insert(*segment_id, vec![init; *size as usize]);
                self.memory_init.insert(*segment_id, init);
                Ok(None)
            }
            IR::WriteMemory { segment_id } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                if let Some(mem) = self.memories.get_mut(segment_id) {
                    if addr < mem.len() { mem[addr] = args[1]; }
                }
                Ok(None)
            }
            IR::ReadMemory { segment_id } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                let init = self.memory_init.get(segment_id).copied().unwrap_or(Fp::zero());
                Ok(Some(self.memories.get(segment_id).and_then(|m| m.get(addr).copied()).unwrap_or(init)))
            }

            // Dynamic NDArray
            IR::AllocateDynamicNDArrayMeta { array_id, max_length, .. } => {
                self.memories.insert(10000 + array_id, vec![Fp::zero(); *max_length as usize]);
                self.memory_init.insert(10000 + array_id, Fp::zero());
                Ok(None)
            }
            IR::DynamicNDArrayGetItem { array_id, .. } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                let seg = 10000 + array_id;
                Ok(Some(self.memories.get(&seg).and_then(|m| m.get(addr).copied()).unwrap_or(Fp::zero())))
            }
            IR::DynamicNDArraySetItem { array_id, .. } => {
                let addr = kernel::fp_to_i64(args[0]) as usize;
                let seg = 10000 + array_id;
                if let Some(mem) = self.memories.get_mut(&seg) {
                    if addr < mem.len() { mem[addr] = args[1]; }
                }
                Ok(None)
            }

            // External function calls
            IR::ExportExternalI { for_which, .. } => {
                self.external_arg_store.entry(*for_which).or_default().push(ExternArg { value: args[0], is_float: false });
                Ok(None)
            }
            IR::ExportExternalF { for_which, .. } => {
                self.external_arg_store.entry(*for_which).or_default().push(ExternArg { value: args[0], is_float: true });
                Ok(None)
            }
            IR::InvokeExternal { store_idx, func_name, .. } => {
                self.invoke_external(*store_idx, func_name, stmt_id)
            }

            // Hash
            IR::PoseidonHash => Ok(Some(kernel::fp_poseidon(&args))),
            IR::EqHash => Ok(Some(if args[0] == args[1] { Fp::one() } else { Fp::zero() })),
        }
    }

    fn invoke_external(&mut self, store_idx: u32, func_name: &str, _stmt_id: usize) -> Result<Option<Fp>, ProvingError> {
        let ext_args = self.external_arg_store.remove(&store_idx).unwrap_or_default();

        // Convert Fp → Value for the callback (human-readable)
        let call_args: Vec<Value> = ext_args.iter()
            .map(|arg| kernel::fp_to_value(arg.value, arg.is_float, self.precision_bits))
            .collect();

        // Call external function
        let result = self.callback.call(func_name, call_args)?;

        // Convert result Value → Fp
        let result_fp = kernel::value_to_fp(&result, self.precision_bits)?;

        // Store as external result
        self.external_results.insert((store_idx, 0), result_fp);

        Ok(Some(result_fp))
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
            .map(|(i, (ir, args))| IRStatement::new(i as u32, ir, args, None))
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
        let result = run_preprocess(&graph, &CircuitInputs::new(), &ProvingParams::default(), &cb).unwrap();
        assert!(result.external_results.is_empty());
    }

    #[test]
    fn test_preprocess_with_external_call() {
        let graph = make_graph(vec![
            (IR::ConstantInt { value: 10 }, vec![]),
            (IR::ExportExternalI { for_which: 0, key: crate::ir_defs::ExternalKey::Int(0), indices: vec![0] }, vec![0]),
            (IR::InvokeExternal {
                store_idx: 0,
                func_name: "double".to_string(),
                args: vec![],
                kwargs: std::collections::HashMap::new(),
            }, vec![]),
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

        let result = run_preprocess(&graph, &CircuitInputs::new(), &ProvingParams::default(), &cb).unwrap();
        assert_eq!(result.external_results.len(), 1);
        let fp = result.external_results.get(&(0, 0)).unwrap();
        assert_eq!(crate::prove::kernel::fp_to_i64(*fp), 20);
    }
}

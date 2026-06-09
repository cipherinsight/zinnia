//! Tests for the halo2 proving backend.
//! Each test builds a small IR circuit and verifies constraints pass the halo2 MockProver.

use crate::circuit_input::{
    CircuitInputs, InputNode, InputParam, InputPath, PathSegment, ResolvedWitness,
};
use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::prove::field::quantization_scale;
use crate::prove::halo2::config::ZinniaConfig;
use crate::prove::halo2::mock_prove;
use crate::prove::halo2::synthesizer::Halo2Synthesizer;
use crate::prove::kernel;
use crate::prove::types::ProvingParams;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};

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

fn empty_resolved(params: &ProvingParams) -> ResolvedWitness {
    ResolvedWitness::new(CircuitInputs::new(), params.precision_bits)
}

#[derive(Clone)]
struct MaliciousMulFRemainderCircuit {
    params: ProvingParams,
}

#[derive(Clone)]
struct MaliciousRangeRecomposeCircuit {
    params: ProvingParams,
}

#[derive(Clone, Copy)]
enum ProductSumFRangeMalice {
    OutOfRangeChunk,
    WeightedAccumulator,
    Recomposition,
}

#[derive(Clone)]
struct MaliciousProductSumFRangeCircuit {
    params: ProvingParams,
    malice: ProductSumFRangeMalice,
}

#[derive(Clone, Copy)]
enum CompactMulFMalice {
    RawProduct,
    DivMod,
}

#[derive(Clone, Copy)]
enum ProductSumFStepMalice {
    ProductDivMod,
    Accumulator,
}

#[derive(Clone)]
struct MaliciousCompactMulFCircuit {
    params: ProvingParams,
    malice: CompactMulFMalice,
}

#[derive(Clone)]
struct MaliciousProductSumFStepCircuit {
    params: ProvingParams,
    malice: ProductSumFStepMalice,
}

impl Circuit<Fp> for MaliciousMulFRemainderCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        ZinniaConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let mut synth = Halo2Synthesizer::new(config, None, self.params.clone());
        let scale = quantization_scale(self.params.precision_bits);
        let a = Fp::one();
        let b = Fp::from(scale as u64);
        let malicious_q = Fp::zero();
        let malicious_r = Fp::from(scale as u64);
        synth
            .debug_constrain_mul_f_with_witness(a, b, malicious_q, malicious_r)
            .map_err(|_| Error::Synthesis)?;
        synth.load_range_table(&mut layouter)?;
        let assigned = layouter.assign_region(
            || "malicious_mulf_remainder",
            |mut region| synth.replay_into_region(&mut region),
        )?;
        synth.expose_instances(&mut layouter, &assigned)
    }
}

impl Circuit<Fp> for MaliciousRangeRecomposeCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        ZinniaConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let mut synth = Halo2Synthesizer::new(config, None, self.params.clone());
        synth
            .debug_range_check_chunks_with_override(Fp::from(2u64), &[1, 0])
            .map_err(|_| Error::Synthesis)?;
        synth.load_range_table(&mut layouter)?;
        let assigned = layouter.assign_region(
            || "malicious_range_recompose",
            |mut region| synth.replay_into_region(&mut region),
        )?;
        synth.expose_instances(&mut layouter, &assigned)
    }
}

impl Circuit<Fp> for MaliciousProductSumFRangeCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        ZinniaConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let mut synth = Halo2Synthesizer::new(config, None, self.params.clone());
        match self.malice {
            ProductSumFRangeMalice::OutOfRangeChunk => {
                // For p16/l8, chunks [0, 256] recompose to r=65536, so the
                // final copy equality is satisfiable and only the lookup can
                // reject the out-of-table chunk.
                let scale = quantization_scale(self.params.precision_bits);
                synth
                    .debug_product_sum_f_range_check_chunks_with_override(
                        Fp::from(scale as u64),
                        &[0, 256],
                        None,
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
            ProductSumFRangeMalice::WeightedAccumulator => {
                // Chunks are in-table and final acc is copy-equal to r=2, but
                // the second accumulator row lies about the weighted update:
                // previous acc 1 + chunk 0 * 256 != current acc 2.
                synth
                    .debug_product_sum_f_range_check_chunks_with_override(
                        Fp::from(2u64),
                        &[1, 0],
                        Some(&[1, 2]),
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
            ProductSumFRangeMalice::Recomposition => {
                // Both chunks are in the lookup table, but [1, 0] recomposes
                // to 1 and is copy-bound to checked value 2.
                synth
                    .debug_product_sum_f_range_check_chunks_with_override(
                        Fp::from(2u64),
                        &[1, 0],
                        None,
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
        }
        synth.load_range_table(&mut layouter)?;
        let assigned = layouter.assign_region(
            || "malicious_product_sum_f_range",
            |mut region| synth.replay_into_region(&mut region),
        )?;
        synth.expose_instances(&mut layouter, &assigned)
    }
}

impl Circuit<Fp> for MaliciousCompactMulFCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        ZinniaConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let mut synth = Halo2Synthesizer::new(config, None, self.params.clone());
        let scale = quantization_scale(self.params.precision_bits);
        match self.malice {
            CompactMulFMalice::RawProduct => {
                synth
                    .debug_constrain_compact_mul_f_row_with_witness(
                        Fp::from(3),
                        Fp::from(5),
                        Fp::from(16),
                        Fp::zero(),
                        Fp::from(16),
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
            CompactMulFMalice::DivMod => {
                synth
                    .debug_constrain_compact_mul_f_row_with_witness(
                        Fp::one(),
                        Fp::from(scale as u64),
                        Fp::from(scale as u64),
                        Fp::zero(),
                        Fp::zero(),
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
        }
        synth.load_range_table(&mut layouter)?;
        let assigned = layouter.assign_region(
            || "malicious_compact_mulf",
            |mut region| synth.replay_into_region(&mut region),
        )?;
        synth.expose_instances(&mut layouter, &assigned)
    }
}

impl Circuit<Fp> for MaliciousProductSumFStepCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        ZinniaConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let mut synth = Halo2Synthesizer::new(config, None, self.params.clone());
        match self.malice {
            ProductSumFStepMalice::ProductDivMod => {
                synth
                    .debug_constrain_product_sum_f_step_with_witness(
                        Fp::from(3),
                        Fp::from(5),
                        Fp::zero(),
                        Fp::zero(),
                        Fp::from(16),
                        Fp::zero(),
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
            ProductSumFStepMalice::Accumulator => {
                let scale = quantization_scale(self.params.precision_bits);
                synth
                    .debug_constrain_product_sum_f_step_with_witness(
                        Fp::from(scale as u64),
                        Fp::one(),
                        Fp::zero(),
                        Fp::one(),
                        Fp::zero(),
                        Fp::zero(),
                    )
                    .map_err(|_| Error::Synthesis)?;
            }
        }
        synth.load_range_table(&mut layouter)?;
        let assigned = layouter.assign_region(
            || "malicious_product_sum_f_step",
            |mut region| synth.replay_into_region(&mut region),
        )?;
        synth.expose_instances(&mut layouter, &assigned)
    }
}

/// ProductSumF operands are adjacent float pairs:
/// `[lhs0, rhs0, lhs1, rhs1, ...]`. The `len` payload is the number of
/// pairs. Raw products are accumulated first, then the total is rescaled once,
/// matching halo2-graph `FixedPointChip::inner_product` and NumPy-style real
/// arithmetic more closely than per-product rounding.
fn product_sum_expected(values: &[(f64, f64)], precision_bits: u32) -> f64 {
    let pairs: Vec<_> = values
        .iter()
        .map(|(lhs, rhs)| {
            (
                kernel::quantize_to_fp(*lhs, precision_bits),
                kernel::quantize_to_fp(*rhs, precision_bits),
            )
        })
        .collect();
    let (result, _) = kernel::fp_product_sum_rescale(&pairs, precision_bits);
    kernel::dequantize_from_fp(result, precision_bits)
}

#[test]
fn test_constant_add() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_mul() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::MulI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_product_sum_f_exactish() {
    let params = ProvingParams {
        k: 10,
        precision_bits: 16,
        ..Default::default()
    };
    let pairs = [(1.5, 2.0), (0.25, 4.0)];
    let expected = product_sum_expected(&pairs, params.precision_bits);
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: pairs[0].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[0].1 }, vec![]),
        (IR::ConstantFloat { value: pairs[1].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[1].1 }, vec![]),
        (IR::ProductSumF { len: 2 }, vec![0, 1, 2, 3]),
        (IR::ConstantFloat { value: expected }, vec![]),
        (IR::EqF, vec![4, 5]),
        (IR::Assert, vec![6]),
    ]);
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_product_sum_f_non_exact_rescale_remainder() {
    let params = ProvingParams {
        k: 12,
        precision_bits: 16,
        ..Default::default()
    };
    let pairs = [(0.1, 0.2), (1.0 / 3.0, 0.7), (-0.45, 0.125)];
    let expected = product_sum_expected(&pairs, params.precision_bits);
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: pairs[0].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[0].1 }, vec![]),
        (IR::ConstantFloat { value: pairs[1].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[1].1 }, vec![]),
        (IR::ConstantFloat { value: pairs[2].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[2].1 }, vec![]),
        (IR::ProductSumF { len: 3 }, vec![0, 1, 2, 3, 4, 5]),
        (IR::ConstantFloat { value: expected }, vec![]),
        (IR::EqF, vec![6, 7]),
        (IR::Assert, vec![8]),
    ]);
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_sub() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::SubI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_boolean_logic() {
    let ir = make_graph(vec![
        (IR::ConstantBool { value: true }, vec![]),
        (IR::ConstantBool { value: false }, vec![]),
        (IR::LogicalAnd, vec![0, 1]),
        (IR::LogicalNot, vec![1]),
        (IR::LogicalOr, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_select() {
    let ir = make_graph(vec![
        (IR::ConstantBool { value: true }, vec![]),
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::ConstantInt { value: 20 }, vec![]),
        (IR::SelectI, vec![0, 1, 2]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_with_input() {
    let ir = make_graph(vec![
        (
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![],
        ),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    let witness = CircuitInputs {
        params: vec![InputParam {
            name: "x".to_string(),
            is_public: false,
            dtype: serde_json::json!("Integer"),
            value: InputNode::Int(42),
        }],
    };
    let params = ProvingParams {
        k: 8,
        ..Default::default()
    };
    let resolved = ResolvedWitness::new(witness, params.precision_bits);
    mock_prove(&ir, &resolved, &params, vec![]).unwrap();
}

#[test]
fn test_div() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 12 }, vec![]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::DivI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_inv() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::InvI, vec![0]),
        (IR::MulI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_equality() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_inequality() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::NeI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_floor_div() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::FloorDivI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_mod() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ModI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_pow() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::PowI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 8,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_assert_with_eq() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_memory() {
    let ir = make_graph(vec![
        (
            IR::AllocateMemory {
                segment_id: 0,
                size: 4,
                init_value: 0,
            },
            vec![],
        ),
        (IR::ConstantInt { value: 0 }, vec![]),
        (IR::ConstantInt { value: 42 }, vec![]),
        (IR::WriteMemory { segment_id: 0 }, vec![1, 2]),
        (IR::ReadMemory { segment_id: 0 }, vec![1]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_lt() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::LtI, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 7,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_sign() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::SignI, vec![0]),
    ]);
    let params = ProvingParams {
        k: 8,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_abs() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: -7 }, vec![]),
        (IR::AbsI, vec![0]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_exp_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.0 }, vec![]),
        (IR::ExpF, vec![0]),
    ]);
    let params = ProvingParams {
        k: 12,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_sin_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 0.5 }, vec![]),
        (IR::SinF, vec![0]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_poseidon_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 1 }, vec![]),
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::PoseidonHash, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 12,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bool_cast() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::BoolCast, vec![0]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_float_mul_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.5 }, vec![]),
        (IR::ConstantFloat { value: 2.0 }, vec![]),
        (IR::MulF, vec![0, 1]),
    ]);
    // MulF now uses a lookup table for fixed-point remainder range checks;
    // the lookup table itself needs 2^lookup_bits rows.
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_mulf_rejects_out_of_range_remainder_even_when_div_mod_holds() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousMulFRemainderCircuit {
        params: params.clone(),
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    let err = prover.verify().expect_err(
        "malicious remainder r=scale should fail lookup-backed range check even though div_mod holds",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Lookup"),
        "expected lookup failure for out-of-range remainder, got {msg}",
    );
}

#[test]
fn test_range_check_rejects_chunks_not_equal_to_checked_cell() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousRangeRecomposeCircuit {
        params: params.clone(),
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    let err = prover
        .verify()
        .expect_err("range chunks must recompose to the exact checked cell through copy equality");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Permutation"),
        "expected permutation failure for recomposition mismatch, got {msg}",
    );
}

#[test]
fn test_product_sum_f_range_rejects_out_of_range_chunk() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousProductSumFRangeCircuit {
        params: params.clone(),
        malice: ProductSumFRangeMalice::OutOfRangeChunk,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    let err = prover.verify().expect_err(
        "ProductSumF chunk row must reject an out-of-table chunk even when recomposition matches r",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Lookup"),
        "expected ProductSumF chunk lookup failure, got {msg}",
    );
}

#[test]
fn test_product_sum_f_range_rejects_weighted_accumulator_mismatch() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousProductSumFRangeCircuit {
        params: params.clone(),
        malice: ProductSumFRangeMalice::WeightedAccumulator,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    let err = prover
        .verify()
        .expect_err("ProductSumF chunk row must reject acc_cur != acc_prev + chunk * weight");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Constraint"),
        "expected ProductSumF weighted accumulator gate failure, got {msg}",
    );
}

#[test]
fn test_product_sum_f_range_rejects_recomposition_mismatch() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousProductSumFRangeCircuit {
        params: params.clone(),
        malice: ProductSumFRangeMalice::Recomposition,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    let err = prover
        .verify()
        .expect_err("ProductSumF chunk rows must copy-bind recomposition to the exact remainder");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Permutation"),
        "expected ProductSumF chunk recomposition permutation failure, got {msg}",
    );
}

#[test]
fn test_compact_mulf_rejects_raw_product_mismatch() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousCompactMulFCircuit {
        params: params.clone(),
        malice: CompactMulFMalice::RawProduct,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    prover
        .verify()
        .expect_err("compact mul_f must reject raw != a*b");
}

#[test]
fn test_compact_mulf_rejects_div_mod_mismatch() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousCompactMulFCircuit {
        params: params.clone(),
        malice: CompactMulFMalice::DivMod,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    prover
        .verify()
        .expect_err("compact mul_f must reject raw != scale*q+r");
}

#[test]
fn test_product_sum_f_step_rejects_product_div_mod_mismatch() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousProductSumFStepCircuit {
        params: params.clone(),
        malice: ProductSumFStepMalice::ProductDivMod,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    prover
        .verify()
        .expect_err("ProductSumF step must reject a*b != scale*q+r");
}

#[test]
fn test_product_sum_f_step_rejects_accumulator_mismatch() {
    let params = ProvingParams {
        k: 9,
        precision_bits: 16,
        lookup_bits: 8,
    };
    let circuit = MaliciousProductSumFStepCircuit {
        params: params.clone(),
        malice: ProductSumFStepMalice::Accumulator,
    };
    let prover = MockProver::run(params.k, &circuit, vec![vec![]]).unwrap();
    prover
        .verify()
        .expect_err("ProductSumF step must reject acc_cur != acc_prev+q");
}

#[test]
fn test_mulf_rejects_unsupported_lookup_chunk_shape() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 0.1 }, vec![]),
        (IR::ConstantFloat { value: 0.2 }, vec![]),
        (IR::MulF, vec![0, 1]),
    ]);
    let params = ProvingParams {
        k: 9,
        precision_bits: 10,
        lookup_bits: 8,
    };
    let err = mock_prove(&ir, &empty_resolved(&params), &params, vec![])
        .expect_err("non-multiple precision_bits/lookup_bits must be rejected explicitly");
    assert!(
        err.contains("MockProver::run failed"),
        "expected synthesis rejection for unsupported chunk shape, got {err}",
    );
}

#[test]
fn halo2_metadata_assertion_rejects_corrupted_runtime_shape() {
    let ir = make_graph(vec![
        (
            IR::AllocateDynamicNDArrayMeta {
                array_id: 0,
                dtype_name: "Float".to_string(),
                max_length: 12,
                max_rank: 2,
                max_shape: vec![4, 3],
            },
            vec![],
        ),
        (
            IR::ReadInteger {
                path: InputPath::new("x", vec![PathSegment::RuntimeShape(0)]),
                is_public: true,
            },
            vec![],
        ),
        (
            IR::ReadInteger {
                path: InputPath::new("x", vec![PathSegment::RuntimeShape(1)]),
                is_public: true,
            },
            vec![],
        ),
        (
            IR::AssertDynamicNDArrayMeta {
                array_id: 0,
                max_rank: 2,
                max_length: 12,
            },
            vec![1, 2],
        ),
    ]);
    let witness = CircuitInputs {
        params: vec![InputParam {
            name: "x".to_string(),
            is_public: false,
            dtype: serde_json::json!({
                "DynamicNDArray": {
                    "dtype": "Float",
                    "max_length": 12,
                    "max_rank": 2,
                    "max_shape": [4, 3],
                }
            }),
            value: InputNode::Array {
                shape: vec![4, 3],
                elements: vec![InputNode::Float(0.0); 12],
                runtime_shape: Some(vec![5, 2]),
            },
        }],
    };
    let params = ProvingParams {
        k: 8,
        ..Default::default()
    };
    let resolved = ResolvedWitness::new(witness, params.precision_bits);
    let err = mock_prove(&ir, &resolved, &params, vec![])
        .expect_err("Halo2 metadata assertion must reject axis-bound corruption");
    assert!(
        err.contains("MockProver verification failed") || err.contains("ConstraintNotSatisfied"),
        "expected Halo2 MockProver constraint failure for corrupted metadata, got {err}",
    );
}

// ── Bitwise ops (constrained via 64-bit two's-complement decomposition) ──

#[test]
fn test_bit_and_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 0b1100 }, vec![]),
        (IR::ConstantInt { value: 0b1010 }, vec![]),
        (IR::BitAndI, vec![0, 1]),
        (IR::ConstantInt { value: 0b1000 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bit_or_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 0b1100 }, vec![]),
        (IR::ConstantInt { value: 0b1010 }, vec![]),
        (IR::BitOrI, vec![0, 1]),
        (IR::ConstantInt { value: 0b1110 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bit_xor_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 0b1100 }, vec![]),
        (IR::ConstantInt { value: 0b1010 }, vec![]),
        (IR::BitXorI, vec![0, 1]),
        (IR::ConstantInt { value: 0b0110 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bit_not_const() {
    // ~5 = -6 under two's-complement i64 semantics.
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::BitNotI, vec![0]),
        (IR::ConstantInt { value: -6 }, vec![]),
        (IR::EqI, vec![1, 2]),
        (IR::Assert, vec![3]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_shl_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 1 }, vec![]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::ShlI, vec![0, 1]),
        (IR::ConstantInt { value: 16 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_shr_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 16 }, vec![]),
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::ShrI, vec![0, 1]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

// ── Transcendental value correctness ───────────────────────────────────
//
// These tests are the load-bearing regression guard for the layered defects
// fixed in this card:
//   1. `poly_eval_horner` constant-first iteration on leading-first coefs.
//   2. No range reduction (polynomial evaluated outside its fit domain).
//   3. Q32 wrap on deep Horner accumulator.
//
// Each test brackets the halo2-computed value against the numpy/f64 reference
// using the (lo, hi) gate pattern from the diagnosis card. A bracket fail
// means a value-level regression that MockProver will surface on the assert
// gate; running these under `cargo test` exercises every layer end-to-end.

/// Build an IR of the form:
///   ConstantFloat(x) → OP → ConstantFloat(lo), ConstantFloat(hi)
///                              ↓                ↓
///                              GtF              LtF
///                                  ↘            ↙
///                                   LogicalAnd
///                                       ↓
///                                     Assert
///
/// Asserts that `OP(x) ∈ (lo, hi)`. Fails MockProver verification if not.
fn assert_op_in_bracket(op: IR, x: f64, lo: f64, hi: f64, k: u32) {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: x }, vec![]),
        (op, vec![0]),
        (IR::ConstantFloat { value: lo }, vec![]),
        (IR::ConstantFloat { value: hi }, vec![]),
        (IR::GtF, vec![1, 2]),
        (IR::LtF, vec![1, 3]),
        (IR::LogicalAnd, vec![4, 5]),
        (IR::Assert, vec![6]),
    ]);
    let params = ProvingParams {
        k,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn halo2_log_value_correctness() {
    // For inputs where the mantissa `m = x * 2^{-k}` lands at exactly 2.0
    // (i.e., x is itself a power of two), the polynomial evaluates at the
    // left endpoint of its fit domain and is accurate to ~1e-6. For
    // arbitrary x (e.g. 6.77 → m ≈ 3.385), Q32 truncation accumulates
    // across the 14-step Horner chain to ~3e-3.
    //
    // Bracket widths reflect these observed bounds:
    //   log(2.0)  ≈ 0.6931 → (0.692, 0.694)   tight, m=2 exactly
    //   log(8.0)  ≈ 2.0794 → (2.078, 2.080)   tight, m=2 exactly
    //   log(6.77) ≈ 1.9125 → (1.907, 1.913)   loose, m=3.385
    assert_op_in_bracket(IR::LogF, 2.0, 0.692, 0.694, 14);
    assert_op_in_bracket(IR::LogF, 8.0, 2.078, 2.080, 14);
    assert_op_in_bracket(IR::LogF, 6.77, 1.907, 1.913, 14);
}

#[test]
fn halo2_exp_value_correctness() {
    // exp2 polynomial on [0, 1) preserves precision well in Q32 (coefs do
    // not have a tiny leading term — the smallest is 3.6e-11 in scaled
    // arithmetic but the accumulator magnitudes stay close to 1). Errors
    // are dominated by the ln(2) Q-rescaling and 2^i lift, ~1e-6 overall.
    //   exp(0.5) ≈ 1.6487 → (1.6485, 1.6489)
    //   exp(1.5) ≈ 4.4817 → (4.4815, 4.4819)
    assert_op_in_bracket(IR::ExpF, 0.5, 1.6485, 1.6489, 14);
    assert_op_in_bracket(IR::ExpF, 1.5, 4.4815, 4.4819, 14);
}

#[test]
fn halo2_sqrt_value_correctness() {
    // sqrt is composed as exp(0.5 * log(x)). For x a power of two, log(x)
    // is computed at m=2 exactly so the chain is precise; sqrt error is
    // dominated by exp's Q-rescaling, ~1e-5.
    //   sqrt(2.0) ≈ 1.4142 → (1.4140, 1.4144)
    //   sqrt(8.0) ≈ 2.8284 → (2.8283, 2.8286)
    assert_op_in_bracket(IR::SqrtF, 2.0, 1.4140, 1.4144, 14);
    assert_op_in_bracket(IR::SqrtF, 8.0, 2.8283, 2.8286, 14);
}

#[test]
fn halo2_sin_value_correctness() {
    // The previous `sin_f` evaluated SIN_COEFS (fit on [0, π)) without proper
    // range reduction. `mod_f(a_abs, 2π)` collapsed to ≈ 0 because `div_f`
    // returns the real quotient (not floor), so sin always returned ≈ 0 and
    // any bracket assert downstream failed MockProver. The fix range-reduces
    // |a| mod π and carries the quotient's parity into the sign.
    //
    // Brackets reflect Q32 polynomial precision (~1e-4 inside [0, π), wider
    // for values that mod-reduce near the endpoints).
    //   sin(0.1)  ≈ 0.0998  → (0.098, 0.101)
    //   sin(0.5)  ≈ 0.4794  → (0.477, 0.482)
    //   sin(1.0)  ≈ 0.8415  → (0.840, 0.843)
    //   sin(1.5)  ≈ 0.9975  → (0.996, 0.999)
    //   sin(3.0)  ≈ 0.1411  → (0.139, 0.143)
    //   sin(-0.5) ≈ -0.4794 → (-0.482, -0.477)  (sign branch)
    //   sin(6.0)  ≈ -0.2794 → (-0.282, -0.277)  (post-π reduction + parity flip)
    assert_op_in_bracket(IR::SinF, 0.1, 0.098, 0.101, 14);
    assert_op_in_bracket(IR::SinF, 0.5, 0.477, 0.482, 14);
    assert_op_in_bracket(IR::SinF, 1.0, 0.840, 0.843, 14);
    assert_op_in_bracket(IR::SinF, 1.5, 0.996, 0.999, 14);
    assert_op_in_bracket(IR::SinF, 3.0, 0.139, 0.143, 14);
    assert_op_in_bracket(IR::SinF, -0.5, -0.482, -0.477, 14);
    assert_op_in_bracket(IR::SinF, 6.0, -0.282, -0.277, 14);
}

#[test]
fn halo2_cos_value_correctness() {
    // cos(x) = sin(x + π/2). Inherits the range-reduction fix via sin.
    //   cos(0.0)  ≈ 1.0    → (0.998, 1.002)
    //   cos(0.1)  ≈ 0.9950 → (0.993, 0.997)
    //   cos(0.5)  ≈ 0.8776 → (0.875, 0.880)
    //   cos(1.0)  ≈ 0.5403 → (0.538, 0.543)
    //   cos(-0.5) ≈ 0.8776 → (0.875, 0.880)  (even function)
    //   cos(3.0)  ≈ -0.9900 → (-0.993, -0.987)
    assert_op_in_bracket(IR::CosF, 0.0, 0.998, 1.002, 14);
    assert_op_in_bracket(IR::CosF, 0.1, 0.993, 0.997, 14);
    assert_op_in_bracket(IR::CosF, 0.5, 0.875, 0.880, 14);
    assert_op_in_bracket(IR::CosF, 1.0, 0.538, 0.543, 14);
    assert_op_in_bracket(IR::CosF, -0.5, 0.875, 0.880, 14);
    assert_op_in_bracket(IR::CosF, 3.0, -0.993, -0.987, 14);
}

#[test]
fn halo2_arcsin_value_correctness() {
    assert_op_in_bracket(IR::ArcSinF, 0.0, -2.0, 2.0, 14);
    assert_op_in_bracket(IR::ArcSinF, 0.5, -2.0, 2.0, 14);
    assert_op_in_bracket(IR::ArcSinF, -0.5, -2.0, 2.0, 14);
}

/// Build an IR `LogicalAnd`/`LogicalOr` of two integer constants and assert
/// the boolean result equals `expected` ∈ {0, 1}. Used to verify the halo2
/// truthy-coercion fix: non-{0,1} operands like (5, 3) must reduce to 1
/// (numpy/mock semantics), not 15 / -7 / etc.
fn assert_logical_op_equals(op: IR, a: i64, b: i64, expected: i64, k: u32) {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: a }, vec![]),
        (IR::ConstantInt { value: b }, vec![]),
        (op, vec![0, 1]),
        (IR::ConstantInt { value: expected }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn halo2_logical_or_truthiness() {
    // numpy: np.logical_or(a, b) = (a != 0) | (b != 0). The previous halo2
    // lowering wired the raw values into `c = a + b - a*b`, producing 15 / -7
    // for (5, 3) etc. The fix coerces each operand to its truthy bit first.
    assert_logical_op_equals(IR::LogicalOr, 5, 0, 1, 8);
    assert_logical_op_equals(IR::LogicalOr, 0, 3, 1, 8);
    assert_logical_op_equals(IR::LogicalOr, 5, 3, 1, 8);
    assert_logical_op_equals(IR::LogicalOr, 0, 0, 0, 8);
    assert_logical_op_equals(IR::LogicalOr, -7, 0, 1, 8);
    assert_logical_op_equals(IR::LogicalOr, -1, -1, 1, 8);
}

#[test]
fn halo2_logical_and_truthiness() {
    // numpy: np.logical_and(a, b) = (a != 0) & (b != 0).
    assert_logical_op_equals(IR::LogicalAnd, 5, 0, 0, 8);
    assert_logical_op_equals(IR::LogicalAnd, 0, 3, 0, 8);
    assert_logical_op_equals(IR::LogicalAnd, 5, 3, 1, 8);
    assert_logical_op_equals(IR::LogicalAnd, 0, 0, 0, 8);
    assert_logical_op_equals(IR::LogicalAnd, -7, 4, 1, 8);
    assert_logical_op_equals(IR::LogicalAnd, -1, -1, 1, 8);
}

/// Build an IR `(a OP b)` of two float constants and bracket-assert the
/// result. Mirrors `assert_op_in_bracket` but for binary float ops.
fn assert_binop_f_in_bracket(op: IR, a: f64, b: f64, lo: f64, hi: f64, k: u32) {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: a }, vec![]),
        (IR::ConstantFloat { value: b }, vec![]),
        (op, vec![0, 1]),
        (IR::ConstantFloat { value: lo }, vec![]),
        (IR::ConstantFloat { value: hi }, vec![]),
        (IR::GtF, vec![2, 3]),
        (IR::LtF, vec![2, 4]),
        (IR::LogicalAnd, vec![5, 6]),
        (IR::Assert, vec![7]),
    ]);
    let params = ProvingParams {
        k,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn halo2_mod_f_value_correctness() {
    // The previous `mod_f` was `a - div_f(a, b) * b`. `div_f` returns the
    // real (not floored) quotient, so this collapsed to `a - a ≈ 0` for any
    // input — every mod_f call returned ~0 regardless of operands.
    //
    // Fix: i64 floor division on Q-encoded operands is exactly floor
    // division on the underlying floats; `mod_i(a, b)` on Q-rep cells
    // yields the Q-rep of `a mod b` with numpy floor-mod semantics.
    //
    //   mod(7.5, 2.0)   = 1.5    → (1.498, 1.502)
    //   mod(-1.5, 0.5)  = 0.0    → (-0.002, 0.002)   (exact multiple)
    //   mod(5.25, 1.5)  = 0.75   → (0.748, 0.752)
    //   mod(-3.5, 2.0)  = 0.5    → (0.498, 0.502)   (floor-mod, not truncation)
    //   mod(10.0, 3.0)  = 1.0    → (0.998, 1.002)
    assert_binop_f_in_bracket(IR::ModF, 7.5, 2.0, 1.498, 1.502, 14);
    assert_binop_f_in_bracket(IR::ModF, -1.5, 0.5, -0.002, 0.002, 14);
    assert_binop_f_in_bracket(IR::ModF, 5.25, 1.5, 0.748, 0.752, 14);
    assert_binop_f_in_bracket(IR::ModF, -3.5, 2.0, 0.498, 0.502, 14);
    assert_binop_f_in_bracket(IR::ModF, 10.0, 3.0, 0.998, 1.002, 14);
}

/// Build an IR `select(cond, t, f)` over integer constants and assert the
/// result equals `expected`. Mirrors `assert_logical_op_equals` but with a
/// ternary `SelectI`. Used to verify the truthy-bit coercion of `cond`.
fn assert_select_i_equals(cond: i64, t: i64, f: i64, expected: i64, k: u32) {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: cond }, vec![]),
        (IR::ConstantInt { value: t }, vec![]),
        (IR::ConstantInt { value: f }, vec![]),
        (IR::SelectI, vec![0, 1, 2]),
        (IR::ConstantInt { value: expected }, vec![]),
        (IR::EqI, vec![3, 4]),
        (IR::Assert, vec![5]),
    ]);
    let params = ProvingParams {
        k,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn halo2_select_non_boolean_cond_truthy() {
    // numpy `np.where(cond, t, f)` treats any non-zero cond as truthy (returns
    // t), zero cond as falsy (returns f). The previous halo2 lowering wired
    // raw cond into `c = cond*(t-f) + f`, so e.g. select(5, 100, 200) yielded
    // 5*(100-200) + 200 = -300 — divergent from numpy/mock. The fix coerces
    // cond through `truthy_bit` first.
    //
    //   select(5, 100, 200)  → 100  (non-zero truthy)
    //   select(0, 100, 200)  → 200  (zero falsy)
    //   select(-7, 100, 200) → 100  (negative is truthy in numpy)
    //   select(1, 100, 200)  → 100  (already a bit, no-op coercion)
    //   select(42, -1, -2)   → -1   (truthy with negative branches)
    assert_select_i_equals(5, 100, 200, 100, 8);
    assert_select_i_equals(0, 100, 200, 200, 8);
    assert_select_i_equals(-7, 100, 200, 100, 8);
    assert_select_i_equals(1, 100, 200, 100, 8);
    assert_select_i_equals(42, -1, -2, -1, 8);
}

#[test]
fn test_bitwise_with_negative() {
    // -1 & 0xF = 0xF (the low 4 bits of -1 are all set).
    let ir = make_graph(vec![
        (IR::ConstantInt { value: -1 }, vec![]),
        (IR::ConstantInt { value: 0xF }, vec![]),
        (IR::BitAndI, vec![0, 1]),
        (IR::ConstantInt { value: 0xF }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams {
        k: 14,
        ..Default::default()
    };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

// ── End-to-end prove() / verify() coverage ─────────────────────────────
//
// These tests exercise `Halo2ProverBackend` (not just `mock_prove`)
// to guard the gaps documented in
// `kanban/cards/compiler/halo2-prove-and-verify-no-op-gaps/`:
//
//   1. prove() must reject witnesses that don't satisfy the circuit.
//   2. prove() → verify() must round-trip a valid proof to Ok.
//   3. verify() must reject a tampered proof.

use crate::prove::error::ProvingError;
use crate::prove::halo2::artifact_format::{
    compute_ir_hash, decode as decode_payload, HALO2_IPA_REKEYGEN_V1,
};
use crate::prove::halo2::vk_cache::{
    cache_clear_for_test, halo2_vk_cache_telemetry, ipa_params_cache_clear_for_test,
    ipa_params_cache_size_for_test, ipa_params_get_or_create, proving_key_cache_clear_for_test,
    proving_key_cache_size_for_test,
};
use crate::prove::halo2::Halo2ProverBackend;
use crate::prove::traits::ProverBackend;
use crate::prove::types::ProofArtifact;

#[test]
fn halo2_prove_rejects_unsat_witness() {
    // assert(false) — a constant-false predicate flowing into Assert.
    // Without the MockProver validation step inside prove(), this would
    // produce a valid-looking proof artifact for an unsatisfiable
    // circuit.
    let ir = make_graph(vec![
        (IR::ConstantBool { value: false }, vec![]),
        (IR::Assert, vec![0]),
    ]);
    let params = ProvingParams {
        k: 5,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;
    let result = backend.prove(&ir, &empty_resolved(&params), &params);
    match result {
        Err(ProvingError::ProvingFailed { detail }) => {
            assert!(
                detail.contains("Witness does not satisfy")
                    || detail.contains("ConstraintNotSatisfied"),
                "expected constraint-not-satisfied error, got: {}",
                detail
            );
        }
        Err(other) => panic!("expected ProvingFailed, got: {:?}", other),
        Ok(_) => panic!("expected prove() to reject the unsatisfiable witness"),
    }
}

#[test]
fn halo2_prove_then_verify_in_process_roundtrip() {
    // A trivially satisfiable circuit: assert(3 == 3).
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    let artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed on a satisfiable circuit");
    assert_eq!(artifact.backend, "halo2-ipa");
    assert_eq!(artifact.format, HALO2_IPA_REKEYGEN_V1);
    assert!(!artifact.proof_bytes.is_empty());
    assert!(
        !artifact.vk_bytes.is_empty(),
        "vk_bytes should carry the re-keygen payload"
    );
    assert_eq!(
        artifact.ir_hash.len(),
        64,
        "ir_hash should be BLAKE2b-256 hex"
    );

    let payload = decode_payload(&artifact.vk_bytes).expect("v1 payload should decode");
    assert_eq!(payload.precision_bits, params.precision_bits);
    assert_eq!(payload.lookup_bits, params.lookup_bits);
    assert_eq!(
        artifact.ir_hash,
        compute_ir_hash(&payload.ir_stmts).expect("payload IR should hash")
    );

    let result = backend
        .verify(&artifact)
        .expect("verify() should not error on a well-formed artifact");
    assert!(
        result.valid,
        "expected verify() to accept a freshly-produced proof, got: {:?}",
        result.error
    );
}

#[test]
fn halo2_verify_reports_ir_hash_mismatch_before_proof_verify() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    let mut artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed");
    artifact.ir_hash = "00".repeat(32);

    match backend.verify(&artifact) {
        Err(ProvingError::IrHashMismatch { expected, computed }) => {
            assert_eq!(expected, "00".repeat(32));
            assert_ne!(expected, computed);
        }
        other => panic!("expected IrHashMismatch, got: {:?}", other),
    }
}

#[test]
fn halo2_verify_uses_warmed_vk_cache() {
    cache_clear_for_test();

    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    let before = halo2_vk_cache_telemetry().snapshot();
    let artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed");
    let after_prove = halo2_vk_cache_telemetry().snapshot();
    assert_eq!(
        after_prove.1, before.1,
        "prove-side cache warming must not count as a verify-side miss"
    );

    let result = backend.verify(&artifact).expect("verify should not error");
    assert!(result.valid);

    let after_verify = halo2_vk_cache_telemetry().snapshot();
    assert!(
        after_verify.0 > after_prove.0,
        "verify should hit the warmed ir_hash -> VK cache"
    );
    assert_eq!(
        after_verify.1, after_prove.1,
        "cache hit should not add a miss"
    );
}

#[test]
fn halo2_prove_reuses_cached_ipa_params_by_k() {
    cache_clear_for_test();
    ipa_params_cache_clear_for_test();
    proving_key_cache_clear_for_test();

    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    assert_eq!(ipa_params_cache_size_for_test(), 0);
    let artifact1 = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("first prove() should succeed");
    assert_eq!(
        ipa_params_cache_size_for_test(),
        1,
        "first prove should populate the k-keyed IPA params cache"
    );

    let artifact2 = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("second prove() should succeed with cached params");
    assert_eq!(
        ipa_params_cache_size_for_test(),
        1,
        "same k should reuse the cached IPA params entry"
    );

    assert!(backend.verify(&artifact1).expect("verify artifact1").valid);
    assert!(backend.verify(&artifact2).expect("verify artifact2").valid);
}

#[test]
fn halo2_ipa_params_persistent_cache_roundtrips_by_k() {
    ipa_params_cache_clear_for_test();

    let dir = std::env::temp_dir().join(format!(
        "zinnia-ipa-params-cache-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp params dir should be writable");

    let old = std::env::var_os("ZINNIA_HALO2_IPA_PARAMS_DIR");
    std::env::set_var("ZINNIA_HALO2_IPA_PARAMS_DIR", &dir);

    let params = ipa_params_get_or_create(6);
    assert_eq!(params.k(), 6);
    let path = dir.join("halo2-ipa-eqaffine-k6.params");
    assert!(
        path.exists(),
        "first cache miss should persist IPA params to disk"
    );

    ipa_params_cache_clear_for_test();
    assert_eq!(ipa_params_cache_size_for_test(), 0);

    let loaded = ipa_params_get_or_create(6);
    assert_eq!(loaded.k(), 6);
    assert_eq!(
        ipa_params_cache_size_for_test(),
        1,
        "disk-loaded params should populate the in-process cache"
    );

    if let Some(old) = old {
        std::env::set_var("ZINNIA_HALO2_IPA_PARAMS_DIR", old);
    } else {
        std::env::remove_var("ZINNIA_HALO2_IPA_PARAMS_DIR");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn halo2_prove_reuses_cached_proving_key_for_exact_shape() {
    cache_clear_for_test();
    ipa_params_cache_clear_for_test();
    proving_key_cache_clear_for_test();

    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    assert_eq!(proving_key_cache_size_for_test(), 0);
    let artifact1 = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("first prove() should succeed and populate the PK cache");
    assert_eq!(
        proving_key_cache_size_for_test(),
        1,
        "first exact shape should populate one proving-key cache entry"
    );

    let artifact2 = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("second prove() should reuse the cached PK");
    assert_eq!(
        proving_key_cache_size_for_test(),
        1,
        "repeated exact shape should reuse the cached proving key"
    );

    assert_eq!(
        artifact1.vk_bytes, artifact2.vk_bytes,
        "PK cache reuse must not change artifact payload compatibility"
    );
    assert_eq!(artifact1.ir_hash, artifact2.ir_hash);
    assert!(backend.verify(&artifact1).expect("verify artifact1").valid);
    assert!(backend.verify(&artifact2).expect("verify artifact2").valid);
}

#[test]
fn halo2_proving_key_cache_does_not_collide_for_different_shapes() {
    cache_clear_for_test();
    ipa_params_cache_clear_for_test();
    proving_key_cache_clear_for_test();

    let ir1 = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let ir2 = make_graph(vec![
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
        (IR::ConstantInt { value: 0 }, vec![]),
        (IR::AddI, vec![0, 4]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    let artifact1 = backend
        .prove(&ir1, &empty_resolved(&params), &params)
        .expect("first shape should prove");
    assert_eq!(proving_key_cache_size_for_test(), 1);

    let artifact2 = backend
        .prove(&ir2, &empty_resolved(&params), &params)
        .expect("different shape with same k should prove");
    assert_eq!(
        proving_key_cache_size_for_test(),
        2,
        "different canonical IR hash at same k must not reuse one PK entry"
    );
    assert_ne!(
        artifact1.ir_hash, artifact2.ir_hash,
        "test setup should exercise distinct proof-shape hashes"
    );
    assert!(backend.verify(&artifact1).expect("verify artifact1").valid);
    assert!(backend.verify(&artifact2).expect("verify artifact2").valid);
}

#[test]
fn halo2_cross_process_verify_roundtrip_parent() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;
    let artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed");

    let path = std::env::temp_dir().join(format!(
        "zinnia-halo2-artifact-{}-{}.json",
        std::process::id(),
        artifact.ir_hash
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&artifact).expect("artifact JSON should encode"),
    )
    .expect("artifact temp file should be writable");

    let output = std::process::Command::new(std::env::current_exe().expect("current test exe"))
        .arg("halo2_cross_process_verify_roundtrip_child")
        .arg("--exact")
        .arg("--nocapture")
        .env("ZINNIA_HALO2_VERIFY_ARTIFACT", &path)
        .output()
        .expect("child verifier process should run");
    let _ = std::fs::remove_file(&path);

    assert!(
        output.status.success(),
        "child verifier failed\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn halo2_cross_process_verify_roundtrip_child() {
    let Some(path) = std::env::var_os("ZINNIA_HALO2_VERIFY_ARTIFACT") else {
        return;
    };

    cache_clear_for_test();
    let bytes = std::fs::read(path).expect("artifact file should be readable");
    let artifact: ProofArtifact =
        serde_json::from_slice(&bytes).expect("artifact JSON should decode");
    let result = Halo2ProverBackend
        .verify(&artifact)
        .expect("fresh-process verify should not error");
    assert!(
        result.valid,
        "fresh-process verify should accept artifact, got: {:?}",
        result.error
    );
}

#[test]
fn halo2_verify_rejects_tampered_proof() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams {
        k: 6,
        ..Default::default()
    };
    let backend = Halo2ProverBackend;

    let mut artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed");

    // Flip a byte in the middle of the transcript. We pick a byte well
    // past the start of the hex string to avoid touching length-coded
    // prefixes (if any) — the goal is to corrupt a commitment in the
    // body of the proof so verify_proof rejects it.
    let mut bytes = hex_decode_for_test(&artifact.proof_bytes);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    artifact.proof_bytes = hex_encode_for_test(&bytes);

    let result = backend
        .verify(&artifact)
        .expect("verify() should return Ok(VerifyResult), not error");
    assert!(
        !result.valid,
        "expected tampered proof to be rejected, got valid=true"
    );
}

// Local hex helpers for tests (the ones in mod.rs are private to the module).
fn hex_decode_for_test(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("test hex"))
        .collect()
}

fn hex_encode_for_test(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

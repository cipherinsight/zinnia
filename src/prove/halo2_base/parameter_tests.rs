use super::*;
use crate::circuit_input::InputPath;
use crate::ir::IRStatement;
use crate::types::ValueId;
use halo2_base::halo2_proofs::dev::MockProver;

fn graph(ops: Vec<(IR, Vec<u32>)>) -> IRGraph {
    let mut stmts: Vec<IRStatement> = Vec::new();
    for (ir, args) in ops {
        let arg_values = args.iter().map(|a| stmts[*a as usize].value_id).collect();
        stmts.push(IRStatement::new(
            stmts.len() as u32,
            ValueId::next(),
            ir,
            args,
            arg_values,
            None,
        ));
    }
    IRGraph::new(stmts)
}

fn read(name: &str) -> IR {
    IR::ReadInteger {
        path: InputPath::new(name, vec![]),
        is_public: false,
    }
}

#[test]
fn rejected_public_layout_allocates_nothing() {
    for tail in [
        vec![],
        vec![(IR::ConstantInt { value: 1 }, vec![])],
        vec![(IR::ExposePublicI, vec![])],
        vec![
            (IR::ExposePublicI, vec![0]),
            (IR::ConstantInt { value: 1 }, vec![]),
        ],
    ] {
        let mut ops = vec![(read("x"), vec![]), (read("y"), vec![])];
        ops.extend(tail);
        let ir = graph(ops);
        let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Keygen);
        builder.set_instance_columns(1);
        assert!(try_build_public_layout_only(&mut builder, &ir, None)
            .unwrap()
            .is_none());
        assert_eq!(
            builder
                .statistics()
                .gate
                .total_advice_per_phase
                .iter()
                .sum::<usize>(),
            0
        );
        assert!(builder.assigned_instances[0].is_empty());
        builder.clear();
    }
}

#[test]
fn public_layout_stats_match_full_synthesis_and_bind_outputs() {
    for arithmetic in [false, true] {
        let mut ops = vec![(read("x"), vec![]), (read("y"), vec![])];
        let output = if arithmetic {
            ops.push((IR::AddI, vec![0, 1]));
            2
        } else {
            0
        };
        ops.push((IR::ExposePublicI, vec![output]));
        let ir = graph(ops);
        let params = ProvingParams {
            k: 9,
            lookup_bits: 8,
            ..ProvingParams::default()
        };
        let stats = cell_stats(&ir, params.lookup_bits, params.precision_bits).unwrap();
        let (mut circuit, publics) =
            build_circuit(&ir, None, &params, CircuitBuilderStage::Mock, None).unwrap();
        let full = circuit.builder.statistics();
        assert_eq!(
            stats.advice,
            full.gate.total_advice_per_phase.iter().sum::<usize>()
        );
        assert_eq!(
            stats.lookup,
            full.total_lookup_advice_per_phase.iter().sum::<usize>()
        );
        assert_eq!(stats.instances, publics.len());
        assert!(MockProver::run(params.k, &circuit, vec![publics.clone()])
            .unwrap()
            .verify()
            .is_ok());
        let mut wrong = publics;
        wrong[0] += Fp::one();
        assert!(MockProver::run(params.k, &circuit, vec![wrong])
            .unwrap()
            .verify()
            .is_err());
        circuit.builder.clear();
    }
}

#[test]
fn selector_minimizes_cells_then_proxy_over_all_widths() {
    // A full signed word prefers eight-bit limbs; a short unsigned range
    // prefers a narrow table even at k values that also fit wider tables.
    for (lower, upper, expected_width) in [(i64::MIN, i64::MAX, 8), (0, 7, 3)] {
        let ir = if lower == 0 {
            graph(vec![
                (read("x"), vec![]),
                (IR::ConstantInt { value: upper + 1 }, vec![]),
                (IR::AssertU128Lt, vec![0, 1]),
                (IR::ExposePublicI, vec![0]),
            ])
        } else {
            graph(vec![
                (read("x"), vec![]),
                (IR::ConstantInt { value: lower }, vec![]),
                (IR::GteI, vec![0, 1]),
                (IR::Assert, vec![2]),
                (IR::ConstantInt { value: upper }, vec![]),
                (IR::LteI, vec![0, 4]),
                (IR::Assert, vec![5]),
                (IR::ExposePublicI, vec![0]),
            ])
        };
        let selected = Halo2BaseProverBackend.estimate_params(&ir).unwrap();
        let mut candidates = Vec::new();
        for width in 1..=ProvingParams::default().lookup_bits {
            let stats = cell_stats(&ir, width, selected.precision_bits).unwrap();
            for k in ESTIMATE_K_MIN..=ESTIMATE_K_MAX {
                if width < k {
                    if let Some(polys) = committed_polys(&stats, k, width) {
                        candidates.push((
                            stats.advice + stats.lookup,
                            (polys as u128) << k,
                            k,
                            width,
                        ));
                    }
                }
            }
        }
        let (_, _, k, width) = candidates.into_iter().min().unwrap();
        assert_eq!((selected.k, selected.lookup_bits), (k, width));
        assert_eq!(width, expected_width);
        let (mut circuit, publics) =
            build_circuit(&ir, None, &selected, CircuitBuilderStage::Mock, None).unwrap();
        assert!(MockProver::run(k, &circuit, vec![publics])
            .unwrap()
            .verify()
            .is_ok());
        assert!(MockProver::run(k, &circuit, vec![vec![Fp::one()]])
            .unwrap()
            .verify()
            .is_err());
        circuit.builder.clear();
    }
}

#[test]
fn selector_breaks_equal_cost_ties_deterministically() {
    let ir = graph(vec![(read("x"), vec![]), (IR::ExposePublicI, vec![0])]);
    let selected = Halo2BaseProverBackend.estimate_params(&ir).unwrap();
    // No range lookups: every width has the same cells and polynomial cost.
    assert_eq!((selected.k, selected.lookup_bits), (5, 1));
}

#[test]
fn parameter_feasibility_keeps_instance_table_and_ram_limits() {
    let base = CellStats {
        advice: 1,
        ..CellStats::default()
    };
    assert!(committed_polys(&base, 4, 1).is_none());
    assert!(committed_polys(&base, 9, 8).is_some());
    for stats in [
        CellStats {
            instances: 493,
            ..base.clone()
        },
        CellStats {
            lookup: 1,
            ..base.clone()
        },
        CellStats {
            table_rows: 493,
            ..base.clone()
        },
        CellStats {
            ram_rows: 493,
            ..base.clone()
        },
    ] {
        assert!(committed_polys(&stats, 9, 9).is_none());
    }
}

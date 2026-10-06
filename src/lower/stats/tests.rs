//! The exact-sums corrcoef: the lowered graph's value on the mock is the
//! formula's (`semantics::exact_integer_r`, which the constant fold uses
//! too), and inputs outside its domain have no proof.

use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::lower::sort::test_support::mock_run;
use crate::lower::{ArrayOp, Construction, OpKind, Target};
use crate::types::Value;

use super::semantics::exact_integer_r;
use super::ExactIntegerSums;

/// `r` of `x{j}`, `y{j}` (n each), exposed, lowered by the exact sums.
fn corrcoef_graph(n: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let mut shared: Vec<Value> =
        (0..n).map(|j| b.ir_read_integer(InputPath::new(format!("x{j}"), vec![]), false)).collect();
    shared.extend((0..n).map(|j| b.ir_read_integer(InputPath::new(format!("y{j}"), vec![]), false)));
    let op = ArrayOp {
        kind: OpKind::Corrcoef { n: n as u32, integer: true, guarded: false },
        shared: 2 * n as u32,
        output: 0,
        position: Some(0),
    };
    let shared: Arc<[Value]> = shared.into();
    let r = b.ir_array_op(op, &shared, &[]);
    b.ir_expose_public_f(&r);
    crate::lower::emit::lower(&b.export_ir_graph(), Target::Halo2Base, &[&ExactIntegerSums as &dyn Construction])
        .expect("lowered")
}

fn inputs(x: &[i64], y: &[i64]) -> Vec<(String, i64)> {
    let named = |p: &str, v: &[i64]| v.iter().enumerate().map(|(j, e)| (format!("{p}{j}"), *e)).collect::<Vec<_>>();
    [named("x", x), named("y", y)].concat()
}

#[test]
fn exact_sums_prove_the_formula() {
    let graph = corrcoef_graph(4);
    for (x, y) in [
        ([14, 2, 17, 10], [153, 486, 588, 232]),
        ([1, 2, 3, 4], [8, 6, 4, 2]),
        ([-5, 0, 7, -3], [100, -100, 3, 0]),
        ([11585, -11585, 11585, -11585], [11585, 11585, -11585, -11585]),
    ] {
        let (ok, outputs) = mock_run(&graph, &inputs(&x, &y));
        let expected = exact_integer_r(&x, &y, 32).expect("in the domain");
        assert!(ok, "{x:?} {y:?}");
        assert_eq!(outputs, vec![expected as i64], "{x:?} {y:?}");
    }
}

#[test]
fn zero_variance_and_overflow_have_no_proof() {
    let graph = corrcoef_graph(4);
    for (x, y) in [
        ([3, 3, 3, 3], [1, 2, 3, 4]),
        ([0, 0, 0, 1 << 20], [0, 0, 0, 1 << 13]),
        ([1 << 31, -(1 << 31), 1 << 31, -(1 << 31)], [1, 2, 3, 4]),
    ] {
        assert_eq!(exact_integer_r(&x, &y, 32), None);
        assert!(!mock_run(&graph, &inputs(&x, &y)).0, "{x:?} {y:?}");
    }
}

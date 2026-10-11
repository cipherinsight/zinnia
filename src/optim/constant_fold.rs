use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct ConstantFold;

/// Raw zero and its rescale are zero at every precision. Run this during
/// construction too, so IRGen's existing output layout can elide the constant.
/// Refuse partial producer cones: erasing a zero product must not erase a
/// division/window check hidden in either factor.
pub(crate) fn exact_zero_product_sum(stmts: &[crate::ir::IRStatement], args: &[StmtId]) -> bool {
    if !args.chunks_exact(2).all(|pair| {
        pair.iter().any(|&id| {
            let s = &stmts[id as usize];
            matches!(s.ir, IR::ConstantFloat { value } if value == 0.0)
                || (matches!(s.ir, IR::SubF)
                    && s.arguments.len() == 2
                    && s.arguments[0] == s.arguments[1])
        })
    }) {
        return false;
    }
    let mut pending = args.to_vec();
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let s = &stmts[id as usize];
        if !super::constant_selector_queries::total_for_erasure(&s.ir) {
            return false;
        }
        pending.extend(s.arguments.iter().copied());
    }
    true
}

impl IRPass for ConstantFold {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut constant_int_cache: HashMap<i64, Value> = HashMap::new();
        let mut constant_float_cache: HashMap<u64, Value> = HashMap::new(); // f64 bits as key
        let constant_true = builder.ir_constant_bool(true);
        let constant_false = builder.ir_constant_bool(false);

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    let value = value_lookup[&arg].clone();
                    // Replace known constants with cached constant IRs
                    match &value {
                        Value::Boolean(sv) => match sv.static_val {
                            Some(true) => constant_true.clone(),
                            Some(false) => constant_false.clone(),
                            None => value,
                        },
                        Value::Integer(sv) => {
                            if let Some(v) = sv.static_val {
                                constant_int_cache
                                    .entry(v)
                                    .or_insert_with(|| builder.ir_constant_int(v))
                                    .clone()
                            } else {
                                value
                            }
                        }
                        Value::Float(sv) => {
                            if let Some(v) = sv.static_val {
                                let bits = v.to_bits();
                                constant_float_cache
                                    .entry(bits)
                                    .or_insert_with(|| builder.ir_constant_float(v))
                                    .clone()
                            } else {
                                value
                            }
                        }
                        _ => value,
                    }
                })
                .collect();

            // Array reads and writes through the builder's folds
            // (`builder/arrays.rs`): a read whose cell the operands decide
            // becomes that cell.
            let new_val = match &stmt.ir {
                IR::ArrayRead(access) => {
                    builder.ir_array_read(&ir_args[0], access.clone(), &ir_args[1..])
                }
                IR::ArrayWrite(access) => {
                    let rank = access.rank();
                    builder.ir_array_write(
                        &ir_args[0],
                        access.clone(),
                        &ir_args[1..1 + rank],
                        &ir_args[1 + rank],
                        &ir_args[2 + rank],
                    )
                }
                _ => builder.create_ir(&stmt.ir, &ir_args),
            };
            value_lookup.insert(stmt.stmt_id, new_val);
        }

        let mut new_graph = builder.export_ir_graph();
        // P0 seam: signal mutation to the resolver. No-op today via
        // StaticOnlyResolver; P1 uses this for cache invalidation.
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IRStatement;
    use crate::types::ValueId;

    fn fold(nodes: Vec<(IR, Vec<StmtId>)>) -> IRGraph {
        let graph = IRGraph::new(
            nodes
                .into_iter()
                .enumerate()
                .map(|(id, (ir, args))| {
                    IRStatement::new(id as StmtId, ValueId::next(), ir, args, vec![], None)
                })
                .collect(),
        );
        super::super::DeadCodeElimination.exec(ConstantFold.exec(graph))
    }

    #[test]
    fn zero_product_sum_folds_but_keeps_partial_producers() {
        for len in [1, 3, 7] {
            for value in [-3.5, 0.0, 2.25] {
                let out = fold(vec![
                    (IR::ConstantFloat { value: 0.0 }, vec![]),
                    (IR::ConstantFloat { value }, vec![]),
                    (IR::ProductSumF { len }, [0, 1].repeat(len as usize)),
                    (IR::ExposePublicF, vec![2]),
                ]);
                assert!(!out
                    .stmts
                    .iter()
                    .any(|s| matches!(s.ir, IR::ProductSumF { .. })));
                let public = out.stmts.last().unwrap();
                assert!(matches!(out.stmts[public.arguments[0] as usize].ir,
                    IR::ConstantFloat { value } if value == 0.0));
            }
        }
        let out = fold(vec![
            (
                IR::ReadFloat {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (IR::ConstantFloat { value: 0.0 }, vec![]),
            (IR::ProductSumF { len: 1 }, vec![0, 1]),
            (IR::ExposePublicF, vec![2]),
        ]);
        assert!(!out
            .stmts
            .iter()
            .any(|s| matches!(s.ir, IR::ProductSumF { .. })));
        let out = fold(vec![
            (
                IR::ReadFloat {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (IR::ConstantFloat { value: 0.0 }, vec![]),
            (IR::DivF, vec![0, 0]),
            (IR::ProductSumF { len: 1 }, vec![1, 2]),
            (IR::ExposePublicF, vec![3]),
        ]);
        assert!(out.stmts.iter().any(|s| matches!(s.ir, IR::DivF)));
        assert!(out
            .stmts
            .iter()
            .any(|s| matches!(s.ir, IR::ProductSumF { .. })));
    }

    #[test]
    fn nonzero_constant_product_sum_keeps_precision_dependent_rescale() {
        let out = fold(vec![
            (IR::ConstantFloat { value: 0.1 }, vec![]),
            (IR::ProductSumF { len: 1 }, vec![0, 0]),
            (IR::ExposePublicF, vec![1]),
        ]);
        assert!(out
            .stmts
            .iter()
            .any(|s| matches!(s.ir, IR::ProductSumF { .. })));
    }
}

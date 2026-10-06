//! Reads of a constant affine table at a runtime index, in closed form.
//!
//! A segment whose every slot holds a compile-time constant written once
//! (`ys = np.arange(1, n + 1) / n`) and whose codes are an affine
//! progression (`search::AffineForm`: `Y_k = floor((A k + B) / C)`, checked
//! against every entry) is read at a runtime index `x` by evaluating the
//! form at `x`: one multiplication and one constant division, instead of a
//! selection over the `n` slots (halo2-base: about 8 to 17 cells per slot;
//! Noir: the table built once and a native read). The pass rewrites a read
//! when the program bounds its index (`AssertU128Lt(x, k)` with `k <= n`,
//! the bounds check every runtime index gets) and the closed form is the
//! cheaper on the target.
//!
//! Soundness: for `x` in `[0, n)` the form equals the slot's constant
//! exactly (checked for every slot at compile time), and the program's
//! own assertion keeps `x` there; the division is exact integer arithmetic
//! on the window (the form is bounded at both ends of the table, see
//! `AffineForm::with`). No hint.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::Target;
use crate::optim::IRPass;
use crate::types::{ScalarValue, StmtId, Value};

use super::search::AffineForm;

/// The pass for one compile target.
pub struct ConstantTableReads {
    pub target: Target,
}

/// A constant segment: its slots' codes and whether they are floats.
struct Table {
    codes: Vec<i64>,
    float: bool,
    form: AffineForm,
}

impl IRPass for ConstantTableReads {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let tables = affine_tables(&graph);
        let rewrites = closed_form_reads(&graph, &tables, self.target);
        if rewrites.is_empty() {
            return graph;
        }
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        for stmt in graph.get_topological_order(false) {
            let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
            let out = match rewrites.get(&stmt.stmt_id) {
                Some(&(segment, _)) => {
                    let table = &tables[&segment];
                    let code = table.form.emit_at(&mut b, &args[0], 0);
                    if table.float {
                        Value::Float(ScalarValue::new(None, code.stmt_id()))
                    } else {
                        code
                    }
                }
                None => b.create_ir(&stmt.ir, &args),
            };
            lookup.insert(stmt.stmt_id, out);
        }
        // The reads the rewrite left unused, and the segments no longer
        // read, go with the dead statements.
        let live = crate::optim::DeadCodeElimination.exec(b.export_ir_graph());
        let pruned = crate::optim::UnreadMemoryEliminator.exec(live);
        crate::optim::DeadCodeElimination.exec(pruned)
    }
}

/// Segments whose every slot is one constant written once at a constant
/// address, never written at a runtime address, with affine codes.
fn affine_tables(graph: &IRGraph) -> HashMap<u32, Table> {
    let precision = crate::prove::types::ProvingParams::default().precision_bits;
    let constant = |id: StmtId| match graph.stmts[id as usize].ir {
        IR::ConstantInt { value } => Some((value, false)),
        IR::ConstantFloat { value } => {
            i64::try_from(crate::prove::field::quantize(value, precision)).ok().map(|c| (c, true))
        }
        _ => None,
    };
    let mut slots: HashMap<u32, Vec<Option<(i64, bool)>>> = HashMap::new();
    let mut spoiled: HashSet<u32> = HashSet::new();
    for stmt in &graph.stmts {
        match stmt.ir {
            IR::AllocateMemory { segment_id, size, .. } => {
                slots.insert(segment_id, vec![None; size as usize]);
            }
            IR::WriteMemory { segment_id } => {
                let address = constant(stmt.arguments[0]).filter(|&(_, float)| !float);
                let value = constant(stmt.arguments[1]);
                let slot = slots.get_mut(&segment_id).zip(address).and_then(|(s, (a, _))| {
                    usize::try_from(a).ok().and_then(|a| s.get_mut(a))
                });
                match (slot, value) {
                    (Some(slot @ None), Some(value)) => *slot = Some(value),
                    _ => {
                        spoiled.insert(segment_id);
                    }
                }
            }
            IR::DynamicNDArraySetItem { segment_id, .. } | IR::MemoryTraceEmit { segment_id, .. } => {
                spoiled.insert(segment_id);
            }
            _ => {}
        }
    }
    slots
        .into_iter()
        .filter(|(segment, _)| !spoiled.contains(segment))
        .filter_map(|(segment, slots)| {
            let slots: Vec<(i64, bool)> = slots.into_iter().collect::<Option<_>>()?;
            let float = slots.iter().any(|&(_, f)| f);
            if slots.iter().any(|&(_, f)| f != float) {
                return None;
            }
            let codes: Vec<i64> = slots.iter().map(|&(c, _)| c).collect();
            let form = AffineForm::find(&codes)?;
            Some((segment, Table { codes, float, form }))
        })
        .collect()
}

/// The runtime-index reads of the tables to rewrite: (segment, length) by
/// read statement.
fn closed_form_reads(graph: &IRGraph, tables: &HashMap<u32, Table>, target: Target) -> HashMap<StmtId, (u32, usize)> {
    if tables.is_empty() {
        return HashMap::new();
    }
    // Indices the program bounds: `AssertU128Lt(x, k)` with a constant `k`.
    let mut bounded: HashMap<StmtId, i64> = HashMap::new();
    for stmt in &graph.stmts {
        if let (IR::AssertU128Lt, [x, k]) = (&stmt.ir, &stmt.arguments[..]) {
            if let IR::ConstantInt { value } = graph.stmts[*k as usize].ir {
                let bound = bounded.entry(*x).or_insert(value);
                *bound = (*bound).min(value);
            }
        }
    }
    let mut reads: HashMap<u32, Vec<StmtId>> = HashMap::new();
    for stmt in &graph.stmts {
        let IR::ReadMemory { segment_id } = stmt.ir else { continue };
        let Some(table) = tables.get(&segment_id) else { continue };
        let index = stmt.arguments[0];
        let constant_index = matches!(graph.stmts[index as usize].ir, IR::ConstantInt { .. });
        if !constant_index && bounded.get(&index).is_some_and(|&k| k <= table.codes.len() as i64) {
            reads.entry(segment_id).or_default().push(stmt.stmt_id);
        }
    }
    reads
        .into_iter()
        .filter(|(segment, stmts)| closed_form_is_cheaper(tables[segment].codes.len(), stmts.len(), target))
        .flat_map(|(segment, stmts)| {
            let n = tables[&segment].codes.len();
            stmts.into_iter().map(move |s| (s, (segment, n)))
        })
        .collect()
}

/// `reads` reads of an `n`-slot table: a selection over the slots
/// (halo2-base about 8 cells per slot and read with the multiplexer read,
/// Noir 3 gates per slot once and 7 per read) against a multiplication and
/// a constant division per read (halo2-base about 40 cells, Noir about 15
/// gates).
fn closed_form_is_cheaper(n: usize, reads: usize, target: Target) -> bool {
    let (n, reads) = (n as u64, reads as u64);
    match target {
        Target::Halo2Base => reads * 40 < reads * 8 * n,
        Target::Noir => reads * 15 < 3 * n + 7 * reads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::InputPath;
    use crate::lower::sort::test_support::mock_run;

    /// A table written to a segment, read at the input `i` (bounded), the
    /// read exposed; rewritten when `rewrite`.
    fn table_read(table: &[f64], float: bool, rewrite: bool) -> IRGraph {
        let mut b = IRBuilder::new();
        let values: Vec<Value> =
            table.iter().map(|&v| if float { b.ir_constant_float(v) } else { b.ir_constant_int(v as i64) }).collect();
        let dtype = if float { crate::types::NumberType::Float } else { crate::types::NumberType::Integer };
        let segment = crate::helpers::segment::alloc_and_write_values(&mut b, &values, dtype);
        let i = b.ir_read_integer(InputPath::new("i", vec![]), false);
        let n = b.ir_constant_int(table.len() as i64);
        b.ir_assert_u128_lt(&i, &n);
        let read = crate::helpers::segment::read_typed(&mut b, segment, &i, dtype);
        if float {
            b.ir_expose_public_f(&read);
        } else {
            b.ir_expose_public_i(&read);
        }
        let graph = b.export_ir_graph();
        if rewrite {
            ConstantTableReads { target: Target::Halo2Base }.exec(graph)
        } else {
            graph
        }
    }

    #[test]
    fn affine_tables_read_in_closed_form_equal_their_entries() {
        let ecdf: Vec<f64> = (1..=50).map(|k| k as f64 / 50.0).collect();
        let ints: Vec<f64> = (0..12).map(|k| (7 * k - 30) as f64).collect();
        for (table, float) in [(ecdf, true), (ints, false)] {
            let graph = table_read(&table, float, true);
            assert!(!graph.stmts.iter().any(|s| matches!(s.ir, IR::ReadMemory { .. })), "read rewritten");
            let reference = table_read(&table, float, false);
            for i in 0..table.len() as i64 {
                let inputs = vec![("i".to_string(), i)];
                let (ok, got) = mock_run(&graph, &inputs);
                assert!(ok);
                assert_eq!((ok, got), mock_run(&reference, &inputs), "entry {i}");
            }
            for i in [-1, table.len() as i64, 1 << 40] {
                assert!(!mock_run(&graph, &[("i".to_string(), i)]).0, "index {i} out of range accepted");
            }
        }
    }

    #[test]
    fn other_tables_keep_their_reads() {
        let squares: Vec<f64> = (0..20).map(|k| (k * k) as f64).collect();
        let graph = table_read(&squares, false, true);
        assert!(graph.stmts.iter().any(|s| matches!(s.ir, IR::ReadMemory { .. })));
    }
}

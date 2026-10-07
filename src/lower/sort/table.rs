//! Reads of a constant affine table at a runtime index, in closed form.
//!
//! A read of known cells (`ArrayRead` of an `ArrayInit`, `crate::ir_array`)
//! whose cells along its axis are compile-time constants
//! (`ys = np.arange(1, n + 1) / n`) with codes in an affine progression
//! (`search::AffineForm`: `Y_k = floor((A k + B) / C)`, checked against every
//! entry) is evaluated at its runtime position `x` in closed form: one
//! multiplication and one constant division, instead of a selection over the
//! `n` cells (halo2-base: about 8 to 17 cells per cell; Noir: the table built
//! once and a native read). The pass rewrites a read when the program bounds
//! its position (`AssertU128Lt(x, k)` with `k <= n`, the bounds check every
//! runtime index gets) and the closed form is the cheaper on the target.
//!
//! Soundness: for `x` in `[0, n)` the form equals the cell's constant
//! exactly (checked for every cell at compile time), and the program's own
//! assertion keeps `x` there; the division is exact integer arithmetic on
//! the window (the form is bounded at both ends of the table, see
//! `AffineForm::with`). No hint.

use std::collections::{BTreeMap, HashMap};

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

/// A line of constant cells: their codes and whether they are floats.
struct Table {
    codes: Vec<i64>,
    float: bool,
    form: AffineForm,
}

impl IRPass for ConstantTableReads {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let rewrites = closed_form_reads(&graph, self.target);
        if rewrites.is_empty() {
            return graph;
        }
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        for stmt in graph.get_topological_order(false) {
            let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
            let out = match rewrites.get(&stmt.stmt_id) {
                Some(table) => {
                    let code = table.form.emit_at(&mut b, &args[1], 0);
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
        // The cells no longer read go with the dead statements.
        crate::optim::DeadCodeElimination.exec(b.export_ir_graph())
    }
}

/// The line of constant cells a read of known cells at one runtime position
/// reads, with affine codes; `None` for any other statement.
fn affine_line(graph: &IRGraph, stmt: StmtId) -> Option<Table> {
    let precision = crate::prove::types::ProvingParams::default().precision_bits;
    let constant = |id: StmtId| match graph.stmts[id as usize].ir {
        IR::ConstantInt { value } => Some((value, false)),
        IR::ConstantFloat { value } => {
            i64::try_from(crate::prove::field::quantize(value, precision)).ok().map(|c| (c, true))
        }
        _ => None,
    };
    let read = crate::lower::arrays::element_read(&graph.stmts, stmt)?;
    read.index?;
    let cells: Vec<(i64, bool)> = read.line.iter().map(|&c| constant(c)).collect::<Option<_>>()?;
    let float = cells.first()?.1;
    if cells.iter().any(|&(_, f)| f != float) {
        return None;
    }
    let codes: Vec<i64> = cells.iter().map(|&(c, _)| c).collect();
    let form = AffineForm::find(&codes)?;
    Some(Table { codes, float, form })
}

/// The reads to rewrite, with their tables.
fn closed_form_reads(graph: &IRGraph, target: Target) -> HashMap<StmtId, Table> {
    // Positions the program bounds: `AssertU128Lt(x, k)` with a constant `k`.
    let mut bounded: HashMap<StmtId, i64> = HashMap::new();
    for stmt in &graph.stmts {
        if let (IR::AssertU128Lt, [x, k]) = (&stmt.ir, &stmt.arguments[..]) {
            if let IR::ConstantInt { value } = graph.stmts[*k as usize].ir {
                let bound = bounded.entry(*x).or_insert(value);
                *bound = (*bound).min(value);
            }
        }
    }
    // The reads of each line (its array and access), in statement order.
    let mut reads: BTreeMap<(StmtId, String), Vec<(StmtId, Table)>> = BTreeMap::new();
    for stmt in &graph.stmts {
        let IR::ArrayRead(access) = &stmt.ir else { continue };
        let Some(table) = affine_line(graph, stmt.stmt_id) else { continue };
        let position = stmt.arguments[1];
        if bounded.get(&position).is_some_and(|&k| k <= table.codes.len() as i64) {
            reads.entry((stmt.arguments[0], access.describe())).or_default().push((stmt.stmt_id, table));
        }
    }
    reads
        .into_values()
        .filter(|line| closed_form_is_cheaper(line[0].1.codes.len(), line.len(), target))
        .flatten()
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

    /// A table in an array storage, read at the input `i` (bounded), the
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
            assert!(!graph.stmts.iter().any(|s| matches!(s.ir, IR::ArrayRead(_))), "read rewritten");
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
        assert!(graph.stmts.iter().any(|s| matches!(s.ir, IR::ArrayRead(_))));
    }
}

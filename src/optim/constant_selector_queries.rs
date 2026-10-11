//! Equality queries on finite constant/Boolean selectors reuse their routing.
//! All leaves and offsets are proven structurally, with checked i64 arithmetic.
//! Reuses the original branch predicates, preserving first-winner ties.
//! Every explicit acceptance-domain assertion remains in the graph.

use super::IRPass;
use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::types::{StmtId, ValueId};
use std::collections::{HashMap, HashSet};

pub struct ConstantSelectorQueries;

/// Reachability from observable operations, matching DCE's rule that array
/// facts do not keep computations alive. This is structural, not evaluation.
fn computed_live(stmts: &[IRStatement]) -> Vec<bool> {
    let mut live = vec![false; stmts.len()];
    let mut pending: Vec<_> = stmts
        .iter()
        .filter(|s| s.ir.is_fixed())
        .map(|s| s.stmt_id)
        .collect();
    while let Some(id) = pending.pop() {
        if live[id as usize] {
            continue;
        }
        live[id as usize] = true;
        pending.extend(stmts[id as usize].arguments.iter().copied());
    }
    live
}

/// Only these operations are known total over their operands' field values.
/// Conservatively retain every other live producer, including checked
/// arithmetic hidden underneath a BoolCast/equality or ordering predicate.
pub(super) fn total_for_erasure(ir: &IR) -> bool {
    matches!(
        ir,
        IR::ConstantInt { .. }
            | IR::ConstantFloat { .. }
            | IR::ConstantBool { .. }
            | IR::ConstantStr { .. }
            | IR::ReadInteger { .. }
            | IR::ReadFloat { .. }
            | IR::AddI
            | IR::SubI
            | IR::MulI
            | IR::AddF
            | IR::SubF
            | IR::SumF { .. }
            | IR::EqI
            | IR::NeI
            | IR::EqF
            | IR::NeF
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::LogicalNot
            | IR::BoolCast
            | IR::SelectI
            | IR::SelectF
            | IR::SelectB
    )
}

#[derive(Clone, Copy)]
enum Kind {
    Constant(i64),
    Bit,
    Select(StmtId, StmtId, StmtId),
    Offset(StmtId, i64),
}
#[derive(Clone, Copy)]
struct Shape {
    kind: Kind,
    lo: i64,
    hi: i64,
    depth: usize,
}

fn boolean(ir: &IR) -> bool {
    matches!(
        ir,
        IR::EqI
            | IR::NeI
            | IR::EqF
            | IR::NeF
            | IR::LtI
            | IR::GtI
            | IR::LteI
            | IR::GteI
            | IR::LtF
            | IR::GtF
            | IR::LteF
            | IR::GteF
            | IR::LtIBounded
            | IR::GtIBounded
            | IR::LogicalNot
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::BoolCast
            | IR::SelectB
    )
}

fn shape(stmt: &IRStatement, shapes: &[Option<Shape>]) -> Option<Shape> {
    let s = match (&stmt.ir, stmt.arguments.as_slice()) {
        (IR::ConstantInt { value }, _) => Shape {
            kind: Kind::Constant(*value),
            lo: *value,
            hi: *value,
            depth: 0,
        },
        (IR::ConstantBool { value }, _) => {
            let v = i64::from(*value);
            Shape {
                kind: Kind::Constant(v),
                lo: v,
                hi: v,
                depth: 0,
            }
        }
        (IR::SelectI | IR::SelectBitI, &[c, t, f]) => {
            let (cond, t_shape, f_shape) = (
                shapes[c as usize]?,
                shapes[t as usize]?,
                shapes[f as usize]?,
            );
            if cond.lo < 0 || cond.hi > 1 {
                return None;
            }
            Shape {
                kind: Kind::Select(c, t, f),
                lo: t_shape.lo.min(f_shape.lo),
                hi: t_shape.hi.max(f_shape.hi),
                depth: 1 + t_shape.depth.max(f_shape.depth),
            }
        }
        (IR::AddI | IR::SubI, &[a, b]) => {
            let (sa, sb) = (shapes[a as usize]?, shapes[b as usize]?);
            let (root, base, offset) = match (&stmt.ir, sa.kind, sb.kind) {
                (IR::AddI, _, Kind::Constant(k)) => (a, sa, k),
                (IR::AddI, Kind::Constant(k), _) => (b, sb, k),
                (IR::SubI, _, Kind::Constant(k)) => (a, sa, k.checked_neg()?),
                _ => return None,
            };
            Shape {
                kind: Kind::Offset(root, offset),
                lo: base.lo.checked_add(offset)?,
                hi: base.hi.checked_add(offset)?,
                depth: base.depth + 1,
            }
        }
        (ir, _) if boolean(ir) => Shape {
            kind: Kind::Bit,
            lo: 0,
            hi: 1,
            depth: 0,
        },
        _ => return None,
    };
    (s.depth <= 64).then_some(s)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bool {
    Constant(bool),
    Wire(StmtId),
}
struct Emit {
    stmts: Vec<IRStatement>,
    remap: Vec<StmtId>,
    cache: HashMap<(StmtId, i64), Bool>,
}
impl Emit {
    fn node(&mut self, ir: IR, args: Vec<StmtId>) -> StmtId {
        let id = self.stmts.len() as StmtId;
        self.stmts.push(IRStatement::new(
            id,
            ValueId::next(),
            ir,
            args,
            vec![],
            None,
        ));
        id
    }
    fn wire(&mut self, b: Bool) -> StmtId {
        match b {
            Bool::Wire(id) => id,
            Bool::Constant(value) => self.node(IR::ConstantBool { value }, vec![]),
        }
    }
    fn not(&mut self, b: Bool) -> Bool {
        match b {
            Bool::Constant(v) => Bool::Constant(!v),
            Bool::Wire(id) => Bool::Wire(self.node(IR::LogicalNot, vec![id])),
        }
    }
    fn and(&mut self, a: Bool, b: Bool) -> Bool {
        match (a, b) {
            (Bool::Constant(false), _) | (_, Bool::Constant(false)) => Bool::Constant(false),
            (Bool::Constant(true), x) | (x, Bool::Constant(true)) => x,
            _ if a == b => a,
            (Bool::Wire(a), Bool::Wire(b)) => Bool::Wire(self.node(IR::LogicalAnd, vec![a, b])),
        }
    }
    fn select(&mut self, c: Bool, t: Bool, f: Bool) -> Bool {
        if t == f {
            return t;
        }
        match (t, f) {
            (Bool::Constant(true), Bool::Constant(false)) => c,
            (Bool::Constant(false), Bool::Constant(true)) => self.not(c),
            (_, Bool::Constant(false)) => self.and(c, t),
            (Bool::Constant(false), _) => {
                let nc = self.not(c);
                self.and(nc, f)
            }
            _ => {
                let args = vec![self.wire(c), self.wire(t), self.wire(f)];
                Bool::Wire(self.node(IR::SelectB, args))
            }
        }
    }
    fn query(&mut self, id: StmtId, k: i64, shapes: &[Option<Shape>]) -> Bool {
        if let Some(&b) = self.cache.get(&(id, k)) {
            return b;
        }
        let shape = shapes[id as usize].unwrap();
        let out = if k < shape.lo || k > shape.hi {
            Bool::Constant(false)
        } else {
            match shape.kind {
                Kind::Constant(v) => Bool::Constant(v == k),
                Kind::Bit => {
                    let bit = Bool::Wire(self.remap[id as usize]);
                    if k == 1 {
                        bit
                    } else {
                        self.not(bit)
                    }
                }
                Kind::Offset(root, offset) => match k.checked_sub(offset) {
                    Some(k) => self.query(root, k, shapes),
                    None => Bool::Constant(false),
                },
                Kind::Select(c, t, f) => {
                    let t = self.query(t, k, shapes);
                    let f = self.query(f, k, shapes);
                    self.select(Bool::Wire(self.remap[c as usize]), t, f)
                }
            }
        };
        self.cache.insert((id, k), out);
        out
    }
}

impl IRPass for ConstantSelectorQueries {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let mut shapes = Vec::with_capacity(graph.stmts.len());
        let mut emit = Emit {
            stmts: vec![],
            remap: vec![],
            cache: HashMap::new(),
        };
        let mut vids = HashMap::new();
        for stmt in &graph.stmts {
            shapes.push(shape(stmt, &shapes));
            let query = if matches!(stmt.ir, IR::EqI | IR::NeI) {
                match stmt.arguments.as_slice() {
                    &[a, b] => match (shapes[a as usize], shapes[b as usize]) {
                        (
                            Some(_),
                            Some(Shape {
                                kind: Kind::Constant(k),
                                ..
                            }),
                        ) => Some((a, k)),
                        (
                            Some(Shape {
                                kind: Kind::Constant(k),
                                ..
                            }),
                            Some(_),
                        ) => Some((b, k)),
                        _ => None,
                    },
                    _ => None,
                }
            } else {
                None
            };
            let id = if let Some((root, k)) = query {
                let mut value = emit.query(root, k, &shapes);
                if matches!(stmt.ir, IR::NeI) {
                    value = emit.not(value);
                }
                let id = emit.wire(value);
                vids.insert(stmt.value_id, emit.stmts[id as usize].value_id);
                id
            } else {
                let id = emit.stmts.len() as StmtId;
                let args = stmt
                    .arguments
                    .iter()
                    .map(|a| emit.remap[*a as usize])
                    .collect();
                emit.stmts.push(IRStatement::new(
                    id,
                    stmt.value_id,
                    stmt.ir.clone(),
                    args,
                    vec![],
                    stmt.dbg.clone(),
                ));
                id
            };
            emit.remap.push(id);
        }
        // A constant query can make its selector's original predicate dead.
        // That predicate (or a checked arithmetic ancestor) may have an
        // implicit domain restriction, so dropping it could admit inputs
        // the original graph rejected. Refuse the pass if any such live
        // producer would disappear; never promote it to a fixed assertion.
        let original_live = computed_live(&graph.stmts);
        let candidate_live = computed_live(&emit.stmts);
        let surviving: HashSet<_> = emit
            .stmts
            .iter()
            .zip(candidate_live)
            .filter_map(|(s, live)| live.then_some(s.value_id))
            .collect();
        if graph
            .stmts
            .iter()
            .zip(original_live)
            .any(|(s, live)| live && !total_for_erasure(&s.ir) && !surviving.contains(&s.value_id))
        {
            return graph;
        }
        let mut out = IRGraph::new(emit.stmts);
        let mut provenance = graph.provenance;
        provenance.remap_values(&vids);
        out.provenance = provenance;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph(nodes: Vec<(IR, Vec<StmtId>)>) -> IRGraph {
        IRGraph::new(
            nodes
                .into_iter()
                .enumerate()
                .map(|(id, (ir, args))| {
                    IRStatement::new(id as StmtId, ValueId::next(), ir, args, vec![], None)
                })
                .collect(),
        )
    }
    #[test]
    fn selector_queries_reuse_boolean_routing_and_preserve_outputs() {
        let g = graph(vec![
            (
                IR::ReadInteger {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (IR::ConstantInt { value: 0 }, vec![]),
            (IR::GtI, vec![0, 1]), // bit
            (IR::ConstantInt { value: 2 }, vec![]),
            (IR::ConstantInt { value: 7 }, vec![]),
            (IR::SelectI, vec![2, 3, 4]),
            (IR::ConstantInt { value: 1 }, vec![]),
            (IR::AddI, vec![5, 6]), // 3 or 8
            (IR::ConstantInt { value: 3 }, vec![]),
            (IR::EqI, vec![7, 8]), // equals original bit
            (IR::ExposePublicI, vec![9]),
            (IR::ExposePublicI, vec![9]),
            (IR::Assert, vec![2]),
        ]);
        let out = super::super::DeadCodeElimination.exec(ConstantSelectorQueries.exec(g));
        assert!(!out
            .stmts
            .iter()
            .any(|s| matches!(s.ir, IR::EqI | IR::SelectI | IR::AddI)));
        let bit = out
            .stmts
            .iter()
            .find(|s| matches!(s.ir, IR::GtI))
            .unwrap()
            .stmt_id;
        let outputs: Vec<_> = out
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::ExposePublicI))
            .collect();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|s| s.arguments == [bit]));
        assert_eq!(
            out.stmts
                .iter()
                .filter(|s| matches!(s.ir, IR::Assert))
                .count(),
            1
        );
    }
    #[test]
    fn selector_queries_preserve_live_partial_predicate_domains() {
        for checked_arithmetic in [false, true] {
            let read = |name| IR::ReadFloat {
                path: crate::circuit_input::InputPath::new(name, vec![]),
                is_public: false,
            };
            let mut nodes = vec![(read("x"), vec![]), (read("y"), vec![])];
            if checked_arithmetic {
                // Even a total Boolean predicate must retain the domain of
                // checked arithmetic beneath it.
                nodes.push((IR::DivF, vec![0, 1]));
                nodes.push((IR::BoolCast, vec![2]));
            } else {
                nodes.push((IR::AddF, vec![0, 0]));
                nodes.push((IR::AddF, vec![1, 1]));
                nodes.push((IR::LtF, vec![2, 3]));
            }
            let predicate = nodes.len() as StmtId - 1;
            let start = nodes.len() as StmtId;
            nodes.extend([
                (IR::ConstantInt { value: 2 }, vec![]),
                (IR::ConstantInt { value: 7 }, vec![]),
                (IR::SelectI, vec![predicate, start, start + 1]),
                (IR::ConstantInt { value: 99 }, vec![]),
                (IR::EqI, vec![start + 2, start + 3]),
                (IR::ExposePublicI, vec![start + 4]),
            ]);
            let out =
                super::super::DeadCodeElimination.exec(ConstantSelectorQueries.exec(graph(nodes)));
            assert!(out.stmts.iter().any(|s| matches!(s.ir, IR::EqI)));
            assert!(out.stmts.iter().any(|s| if checked_arithmetic {
                matches!(s.ir, IR::DivF)
            } else {
                matches!(s.ir, IR::LtF)
            }));
            assert!(!out.stmts.iter().any(|s| matches!(s.ir, IR::Assert)));
        }
    }

    #[test]
    fn selector_queries_ignore_already_dead_partial_producers() {
        let g = graph(vec![
            (
                IR::ReadInteger {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (IR::ConstantInt { value: 0 }, vec![]),
            (IR::GtI, vec![0, 1]), // Already dead; does not block rewriting.
            (IR::BoolCast, vec![0]),
            (IR::ConstantInt { value: 2 }, vec![]),
            (IR::ConstantInt { value: 7 }, vec![]),
            (IR::SelectI, vec![3, 4, 5]),
            (IR::ConstantInt { value: 99 }, vec![]),
            (IR::EqI, vec![6, 7]),
            (IR::ExposePublicI, vec![8]),
        ]);
        let out = super::super::DeadCodeElimination.exec(ConstantSelectorQueries.exec(g));
        assert!(!out
            .stmts
            .iter()
            .any(|s| matches!(s.ir, IR::EqI | IR::SelectI | IR::GtI)));
        let output = out
            .stmts
            .iter()
            .find(|s| matches!(s.ir, IR::ExposePublicI))
            .unwrap();
        assert!(matches!(
            out.stmts[output.arguments[0] as usize].ir,
            IR::ConstantBool { value: false }
        ));
    }

    #[test]
    fn selector_queries_refuse_unknown_or_overflowing_arithmetic() {
        for unknown in [false, true] {
            let leaf = if unknown {
                IR::ReadInteger {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                }
            } else {
                IR::ConstantInt { value: i64::MAX }
            };
            let g = graph(vec![
                (leaf, vec![]),
                (IR::ConstantInt { value: 1 }, vec![]),
                (IR::AddI, vec![0, 1]),
                (IR::EqI, vec![2, 1]),
                (IR::ExposePublicI, vec![3]),
            ]);
            let out = ConstantSelectorQueries.exec(g);
            assert!(out.stmts.iter().any(|s| matches!(s.ir, IR::EqI)));
        }
    }
}

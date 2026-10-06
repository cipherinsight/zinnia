//! Lowering of sorts (`IR::Sorted` placeholders) by a cost rule over their
//! live outputs.
//!
//! `np.sort`, `np.argsort`, `np.median` and `np.percentile` emit one
//! `IR::Sorted` placeholder per output they use: the value or the stable
//! argsort index of a constant rank, or of a runtime rank (a read of a
//! sorted array at a runtime index). Placeholders over the same element
//! statements form one *group* (one sort). This pass runs after dead-code
//! elimination, so a group's remaining members are exactly the sort's live
//! outputs (`np.argsort(a)[::-1][:7]` keeps 7 of 100). It replaces every
//! member with real IR, choosing per group the cheapest plan for the
//! compile target (`cost::choose_plan`):
//!
//! * a sorting network (Batcher's merge exchange, or the adjacent-swap
//!   network for small stable argsorts), no hint;
//! * hinted outputs pinned in runs of ranks (`emit::pinned_run`; a run over
//!   every rank is the full permutation check), and per-output pins for
//!   isolated values (`emit::order_statistic`, a count) and runtime ranks.
//!
//! Every plan yields NumPy's stable sort exactly; each hint's soundness
//! argument is next to its pins in `emit`. The lowered IR contains no
//! placeholder, and no backend accepts one.

mod cost;
mod emit;
mod packed;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

pub use cost::SortTarget;
pub(crate) use emit::masked_order_statistic;

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{NumberType, StmtId, Value};

use self::cost::{Demand, Network, Pin, Plan, Run};
use self::emit::{ElementMemory, Elements, OutputMemory};
use super::IRPass;

/// The pass: every sort lowered by the cost rule of `target`.
pub struct SortLowering {
    pub target: SortTarget,
}

/// One sort: its element statements, live outputs and chosen plan.
struct Group {
    demand: Demand,
    plan: Plan,
}

impl IRPass for SortLowering {
    /// Lowers the sorts, then simplifies what the lowering emitted as the
    /// earlier passes simplified the rest (`select(c, 1, 0)` is `c`, ...),
    /// deduplicates it, and drops what no output needs (a network's
    /// comparators outside the cone of the live outputs). A graph without
    /// sorts is returned unchanged.
    fn exec(&self, graph: IRGraph) -> IRGraph {
        if !graph.stmts.iter().any(|s| sort_lane(&s.ir).is_some()) {
            return graph;
        }
        let costs = self.target.costs();
        let lowered = lower_sorts(graph, |demand| cost::choose_plan(demand, &costs).0);
        let simplified = super::PatternMatchOptim.exec(lowered);
        let deduplicated = super::DuplicateCodeElimination.exec(simplified);
        super::DeadCodeElimination.exec(deduplicated)
    }
}

/// Replace every `IR::Sorted` of `graph` by the IR of the plan `choose`
/// picks for its sort's live outputs.
fn lower_sorts(graph: IRGraph, choose: impl Fn(&Demand) -> Plan) -> IRGraph {
    if !graph.stmts.iter().any(|s| sort_lane(&s.ir).is_some()) {
        return graph;
    }
    let groups: HashMap<Vec<StmtId>, Group> = demands(&graph)
        .into_iter()
        .map(|(elements, demand)| {
            let plan = choose(&demand);
            (elements, Group { demand, plan })
        })
        .collect();

    let mut b = IRBuilder::new();
    b.reserve_segment_ids_below(next_free_segment_id(&graph));
    let mut lookup: HashMap<StmtId, Value> = HashMap::new();
    let mut lowered: HashMap<Vec<StmtId>, LoweredSort> = HashMap::new();
    for stmt in graph.get_topological_order(false) {
        let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
        let out = match stmt.ir {
            _ if sort_lane(&stmt.ir).is_some() => {
                let (len, rank, index) = sort_lane(&stmt.ir).expect("a sort lane");
                let len = len as usize;
                let key = stmt.arguments[..len].to_vec();
                let sort = lowered
                    .entry(key.clone())
                    .or_insert_with(|| LoweredSort::new(&mut b, &groups[&key], &args[..len]));
                match rank {
                    Some(r) => sort.output(r as usize, index),
                    None => sort.output_at_runtime_rank(&mut b, &args[len], index),
                }
            }
            _ => b.create_ir(&stmt.ir, &args),
        };
        lookup.insert(stmt.stmt_id, out);
    }
    b.export_ir_graph()
}

/// The live outputs of each sort in `graph`, keyed by its element
/// statements.
fn demands(graph: &IRGraph) -> HashMap<Vec<StmtId>, Demand> {
    let mut demands: HashMap<Vec<StmtId>, Demand> = HashMap::new();
    for stmt in &graph.stmts {
        let Some((len, rank, index)) = sort_lane(&stmt.ir) else {
            continue;
        };
        let len = len as usize;
        let demand = demands
            .entry(stmt.arguments[..len].to_vec())
            .or_insert_with(|| Demand { n: len, ..Demand::default() });
        match (rank, index) {
            (Some(r), false) => {
                demand.values.insert(r as usize);
            }
            (Some(r), true) => {
                demand.indices.insert(r as usize);
            }
            (None, false) => demand.runtime_values += 1,
            (None, true) => demand.runtime_indices += 1,
        }
    }
    demands
}

/// A sort placeholder's element count, constant rank and index flag.
fn sort_lane(ir: &IR) -> Option<(u32, Option<u32>, bool)> {
    match ir {
        IR::ArrayOp(op) if op.kind == crate::lower::OpKind::Sort => {
            Some((op.shared, op.position, op.output == crate::lower::sort::INDICES))
        }
        _ => None,
    }
}

/// One above the largest memory segment id `graph` uses.
fn next_free_segment_id(graph: &IRGraph) -> u32 {
    graph
        .stmts
        .iter()
        .filter_map(|s| match s.ir {
            IR::AllocateMemory { segment_id, .. }
            | IR::WriteMemory { segment_id }
            | IR::ReadMemory { segment_id }
            | IR::MemoryTraceEmit { segment_id, .. }
            | IR::DynamicNDArrayGetItem { segment_id, .. }
            | IR::DynamicNDArraySetItem { segment_id, .. } => Some(segment_id + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// A sort's lowered outputs, built when its first live output is reached
/// (every member's elements precede it); runtime-rank outputs are lowered
/// where they occur, since their ranks may be computed later.
struct LoweredSort {
    elements: Vec<Value>,
    dtype: NumberType,
    /// Values and indices by constant rank.
    values: HashMap<usize, Value>,
    indices: HashMap<usize, Value>,
    /// Whether every rank's outputs exist (a network or the full run), so
    /// runtime ranks read them instead of being pinned on their own.
    complete: bool,
    element_memory: Option<ElementMemory>,
    value_memory: Option<OutputMemory>,
    index_memory: Option<OutputMemory>,
}

impl LoweredSort {
    fn new(b: &mut IRBuilder, group: &Group, elements: &[Value]) -> LoweredSort {
        let dtype = if elements.iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        };
        let mut sort = LoweredSort {
            elements: elements.to_vec(),
            dtype,
            values: HashMap::new(),
            indices: HashMap::new(),
            complete: false,
            element_memory: None,
            value_memory: None,
            index_memory: None,
        };
        let n = elements.len();
        match &group.plan {
            Plan::Network { network: Network::PackedMergeExchange } => {
                let indices = packed::argsort_by_packed_keys(b, &sort.as_elements());
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::Network { network } => {
                let (values, indices) =
                    emit::sort_by_network(b, &sort.as_elements(), *network);
                sort.values = values.into_iter().enumerate().collect();
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::Hinted { runs } => {
                for run in runs {
                    match run.pin {
                        Pin::Keys => sort.pin_keys(b, run),
                        Pin::Counts => sort.pin_by_counts(b, &group.demand, run),
                        Pin::Extreme => {
                            let largest = run.first + 1 == n && n > 1;
                            let value = emit::extreme_value(b, &sort.as_elements(), largest);
                            sort.values.insert(run.first, value);
                        }
                    }
                }
                sort.complete = matches!(runs.as_slice(), [run] if run.pin == Pin::Keys && run.first == 0 && run.last + 1 == n);
            }
        }
        sort
    }

    /// The run's values and indices as pinned keys.
    fn pin_keys(&mut self, b: &mut IRBuilder, run: &Run) {
        let memory = self.element_memory(b);
        let keys = emit::pinned_run(b, &self.as_elements(), &memory, run.first, run.last);
        for (k, key) in keys.into_iter().enumerate() {
            self.values.insert(run.first + k, key.value);
            self.indices.insert(run.first + k, key.index);
        }
    }

    /// The run's live values, each pinned by its own counts.
    fn pin_by_counts(&mut self, b: &mut IRBuilder, demand: &Demand, run: &Run) {
        for &r in demand.values.range(run.first..=run.last) {
            let rank = b.ir_constant_int(r as i64);
            let value = emit::order_statistic(b, &self.as_elements(), &rank);
            self.values.insert(r, value);
        }
    }

    fn as_elements(&self) -> Elements<'_> {
        Elements { values: &self.elements, dtype: self.dtype }
    }

    /// The elements' segment for reads at hinted indices, built once.
    fn element_memory(&mut self, b: &mut IRBuilder) -> ElementMemory {
        if let Some(memory) = self.element_memory {
            return memory;
        }
        let memory = ElementMemory::new(b, &self.as_elements());
        self.element_memory = Some(memory);
        memory
    }

    fn output(&self, rank: usize, index: bool) -> Value {
        let outputs = if index { &self.indices } else { &self.values };
        outputs
            .get(&rank)
            .cloned()
            .unwrap_or_else(|| panic!("sort lowering: rank {rank} was not lowered"))
    }

    fn output_at_runtime_rank(&mut self, b: &mut IRBuilder, rank: &Value, index: bool) -> Value {
        if self.complete {
            let n = self.elements.len();
            let memory_slot = if index { &mut self.index_memory } else { &mut self.value_memory };
            if memory_slot.is_none() {
                let source = if index { &self.indices } else { &self.values };
                let outputs: Vec<Value> = (0..n).map(|r| source[&r].clone()).collect();
                let dtype = if index { NumberType::Integer } else { self.dtype };
                *memory_slot = Some(OutputMemory::new(b, &outputs, dtype));
            }
            return memory_slot.as_ref().expect("just built").read(b, rank);
        }
        if index {
            let memory = self.element_memory(b);
            emit::argsort_at_rank(b, &self.as_elements(), &memory, rank)
        } else {
            emit::order_statistic(b, &self.as_elements(), rank)
        }
    }
}

//! The sort family: `np.sort`, `np.argsort`, `np.median`, `np.percentile`
//! (`OpKind::Sort`), and `np.searchsorted` over a constant sorted table
//! (`OpKind::SearchSorted`, see `search`).
//!
//! Each sort emits one placeholder per output it uses: the value or the
//! stable argsort index of a constant rank, or of a runtime rank (a read of
//! a sorted array at a runtime index). After dead-code elimination a group's
//! lanes are exactly the sort's live outputs (`np.argsort(a)[::-1][:7]`
//! keeps 7 of 100). Three constructions offer plans that compute every live
//! lane (`cost` prices them per target; the selection takes the cheapest):
//!
//! * [`SortingNetwork`]: a sorting network (Batcher's merge exchange on
//!   values, on `(value, index)` keys, or on packed keys (`packed`) for an
//!   argsort, or the adjacent-swap network for small or one-sided demands),
//!   no hint;
//! * [`PermutationCheck`]: every rank's key hinted and pinned by one run
//!   over every rank (`emit::pinned_run`);
//! * [`PinnedSelection`]: hinted outputs pinned in runs of ranks, per-output
//!   pins for isolated values (`emit::order_statistic`, a count; the
//!   extremes by bounds) and runtime ranks.
//!
//! Every plan yields NumPy's stable sort exactly; each hint's soundness
//! argument is next to its pins in `emit`.

mod cost;
mod counts;
mod emit;
mod packed;
pub(crate) mod search;
mod semantics;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

pub(crate) use emit::masked_order_statistic;
pub use search::SearchSortedSemantics;
pub use counts::SortCountQueries;
pub use semantics::{SortSemantics, COUNT_AT_MOST, COUNT_BELOW, INDICES, VALUES};

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{NumberType, StmtId, Value};

use self::cost::{Demand, Network, Pin, Plan, Run, UnitCosts};
use self::emit::{ElementMemory, Elements, OutputMemory};
use super::{Candidate, Construction, Context, EmitCx, Emitter, Group, Lane, OpKind, Requirement};

/// The sort family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    let mut all: Vec<&'static dyn Construction> = vec![&SortingNetwork, &PermutationCheck, &PinnedSelection];
    all.extend(search::constructions());
    all
}

/// A sorting network over all keys (`emit::sort_by_network`), pruned to the
/// live outputs by dead-code elimination; runtime-rank outputs read the
/// network's outputs from a segment. One candidate per network that
/// computes the live outputs (`cost::network_plans`). No hint.
pub struct SortingNetwork;

/// One run of pinned keys over every rank: the permutation check
/// (`emit::pinned_run`); runtime-rank outputs read its outputs from a
/// segment (`cost::full_permutation`).
pub struct PermutationCheck;

/// The live constant ranks in pinned runs, counts or extreme pins, and the
/// runtime-rank outputs pinned one by one (`cost::selection_plans`).
pub struct PinnedSelection;

impl Construction for SortingNetwork {
    fn name(&self) -> &'static str {
        "sort: network"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        plans(group, cx, cost::network_plans)
    }
}

impl Construction for PermutationCheck {
    fn name(&self) -> &'static str {
        "sort: permutation check"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        plans(group, cx, |demand, costs| cost::full_permutation(demand, costs).into_iter().collect())
    }
}

impl Construction for PinnedSelection {
    fn name(&self) -> &'static str {
        "sort: pinned selection"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        plans(group, cx, cost::selection_plans)
    }
}

/// The candidates of the plans `offer` prices for `group`'s live lanes, each
/// producing every lane.
fn plans(group: &Group, cx: &Context, offer: impl Fn(&Demand, &UnitCosts) -> Vec<(Plan, u64)>) -> Vec<Candidate> {
    if group.kind != OpKind::Sort {
        return Vec::new();
    }
    let demand = demand(group);
    offer(&demand, &cost::unit_costs(cx.target))
        .into_iter()
        .map(|(plan, cost)| plan_candidate(group, demand.clone(), plan, cost))
        .collect()
}

/// A candidate that lowers every lane of `group` by `plan`.
fn plan_candidate(group: &Group, demand: Demand, plan: Plan, cost: u64) -> Candidate {
    Candidate {
        pre: Vec::new(),
        post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
        cost,
        emitter: Box::new(SortEmitter { elements: group.shared.clone(), demand, plan, sort: None }),
    }
}

/// The live outputs of a sort.
fn demand(group: &Group) -> Demand {
    let mut demand = Demand { n: group.shared.len(), ..Demand::default() };
    for lane in group.lanes.iter().filter(|lane| lane.output == VALUES || lane.output == INDICES) {
        match (lane.position, lane.output == INDICES) {
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
    demand
}

/// Emits one sort by its plan: the constant-rank outputs when it starts,
/// runtime-rank outputs where they occur (their ranks may be computed
/// later).
struct SortEmitter {
    elements: Vec<StmtId>,
    demand: Demand,
    plan: Plan,
    sort: Option<LoweredSort>,
}

impl Emitter for SortEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let elements = cx.values(&self.elements);
        self.sort = Some(LoweredSort::new(b, &self.demand, &self.plan, &elements));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let sort = self.sort.as_mut().expect("started");
        if lane.output == COUNT_AT_MOST || lane.output == COUNT_BELOW {
            let e = cx.value(lane.own[0]);
            return counts::count_query(b, &sort.as_elements(), &e, lane.output == COUNT_AT_MOST);
        }
        let index = lane.output == INDICES;
        match lane.position {
            Some(r) => sort.output(r as usize, index),
            None => sort.output_at_runtime_rank(b, &cx.value(lane.own[0]), index),
        }
    }
}

/// A sort's lowered outputs.
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
    fn new(b: &mut IRBuilder, demand: &Demand, plan: &Plan, elements: &[Value]) -> LoweredSort {
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
        match plan {
            Plan::Network { network: Network::PackedMergeExchange } => {
                let indices = packed::argsort_by_packed_keys(b, &sort.as_elements());
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::Network { network } => {
                let (values, indices) = emit::sort_by_network(b, &sort.as_elements(), *network);
                sort.values = values.into_iter().enumerate().collect();
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::Hinted { runs } => {
                for run in runs {
                    match run.pin {
                        Pin::Keys => sort.pin_keys(b, run),
                        Pin::Counts => sort.pin_by_counts(b, demand, run),
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

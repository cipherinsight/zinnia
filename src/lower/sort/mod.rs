//! The sort family: `np.sort`, `np.argsort`, `np.median`, `np.percentile`
//! (`OpKind::Sort`), and `np.searchsorted` over a constant sorted table
//! (`OpKind::SearchSorted`, see `search`).
//!
//! Each sort emits one placeholder per output it uses: the value or the
//! stable argsort index of a constant rank, or of a runtime rank (a read of
//! a sorted array at a runtime index). After dead-code elimination a group's
//! lanes are exactly the sort's live outputs (`np.argsort(a)[::-1][:7]`
//! keeps 7 of 100). Five constructions offer plans that compute every live
//! lane (`cost` prices them per target; the selection takes the cheapest):
//!
//! * [`SortingNetwork`]: a sorting network (Batcher's merge exchange on
//!   values, on `(value, index)` keys, or on packed keys (`packed`) for an
//!   argsort, or the adjacent-swap network for small or one-sided demands),
//!   no hint;
//! * [`RankSelection`]: one-hot rank selectors from the pairwise
//!   comparisons, every output an inner product (`emit::rank_selectors`);
//! * [`PermutationCheck`]: every rank's key hinted and pinned by one run
//!   over every rank (`emit::pinned_run`);
//! * [`PinnedSelection`]: hinted outputs pinned in runs of ranks, per-output
//!   pins for isolated values (`emit::order_statistic`, a count; the
//!   extremes by bounds) and runtime ranks;
//! * [`RoutedPermutation`]: packed keys routed into ascending order by a
//!   Waksman network whose switch bits the prover supplies, checked
//!   increasing (`routed`; halo2-base only);
//! * [`GrandProductPermutation`]: hinted sorted keys and carried entries, a
//!   permutation of the input by a grand product at verifier challenges,
//!   checked increasing (`grand_product`; halo2-base only, needs the
//!   program-wide challenge).
//!
//! Every plan yields NumPy's stable sort exactly; each hint's soundness
//! argument is next to its pins in `emit` (the routing bits' in `routed`).

mod argmax;
pub mod arg_extreme;
mod cost;
mod counts;
mod emit;
mod grand_product;
mod packed;
mod payload;
mod routed;
pub(crate) mod search;
mod table;
mod semantics;
mod sorted_input;
pub mod waksman;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod routed_tests;
#[cfg(test)]
mod grand_product_tests;

pub(crate) use emit::masked_order_statistic;
pub use search::SearchSortedSemantics;
pub use argmax::ArgmaxQueries;
pub use counts::SortCountQueries;
pub use table::ConstantTableReads;
pub use payload::payload_at_argsort_lane;
pub use sorted_input::SortedInput;
pub use semantics::{SortSemantics, COUNT_AT_MOST, COUNT_BELOW, INDICES, PAYLOAD, VALUES};

use std::collections::{BTreeSet, HashMap};

use crate::builder::IRBuilder;
use crate::ir_defs::IR;
use crate::types::{NumberType, StmtId, Value};

use self::cost::{Demand, Network, Pin, Plan, Run, UnitCosts};
use self::emit::{ElementMemory, Elements, Membership, OutputMemory, Payload};
use super::{Candidate, Construction, Context, EmitCx, Emitter, Group, Lane, OpKind, Requirement};

/// The sort family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    let mut all: Vec<&'static dyn Construction> = vec![
        &SortingNetwork,
        &RankSelection,
        &PermutationCheck,
        &PinnedSelection,
        &RoutedPermutation,
        &GrandProductPermutation,
        &sorted_input::SortedInput,
    ];
    all.extend(search::constructions());
    all.extend(arg_extreme::constructions());
    all
}

/// A sorting network over all keys (`emit::sort_by_network`), pruned to the
/// live outputs by dead-code elimination; runtime-rank outputs read the
/// network's outputs from a segment. One candidate per network that
/// computes the live outputs (`cost::network_plans`). No hint.
pub struct SortingNetwork;

/// Every live output as an inner product with the stable sort's rank
/// selectors, from one comparison per pair of elements
/// (`emit::rank_selectors`, `cost::rank_selector_plan`). No hint.
pub struct RankSelection;

/// One run of pinned keys over every rank: the permutation check
/// (`emit::pinned_run`); runtime-rank outputs read its outputs from a
/// segment (`cost::full_permutation`).
pub struct PermutationCheck;

/// The live constant ranks in pinned runs, counts or extreme pins, and the
/// runtime-rank outputs pinned one by one (`cost::selection_plans`).
pub struct PinnedSelection;

/// Every rank by a Waksman network with hinted switch bits whose routed
/// keys are checked increasing (`routed`, with its soundness argument;
/// priced by `cost::routed_plan`, offered where the target's unit costs
/// have routing costs: halo2-base).
pub struct RoutedPermutation;

impl Construction for RoutedPermutation {
    fn name(&self) -> &'static str {
        "sort: routed permutation"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        plans(group, cx, |demand, costs| cost::routed_plan(demand, costs).into_iter().collect())
    }
}

/// Every rank from hinted sorted keys whose multiset a grand product checks
/// at verifier challenges, checked increasing (`grand_product`, with its
/// soundness argument; priced by `cost::grand_product_plan`, offered where
/// the target has verifier challenges: halo2-base). Precondition: the
/// program-wide challenge (`lower::challenge`).
pub struct GrandProductPermutation;

impl Construction for GrandProductPermutation {
    fn name(&self) -> &'static str {
        "sort: grand-product permutation"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if !super::challenge::available(cx.target) {
            return Vec::new();
        }
        let mut candidates = plans(group, cx, |demand, costs| cost::grand_product_plan(demand, costs).into_iter().collect());
        for candidate in &mut candidates {
            candidate.pre.push(super::challenge::CHALLENGE);
        }
        candidates
    }
}

impl Construction for SortingNetwork {
    fn name(&self) -> &'static str {
        "sort: network"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        plans(group, cx, cost::network_plans)
    }
}

impl Construction for RankSelection {
    fn name(&self) -> &'static str {
        "sort: rank selectors"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        plans(group, cx, |demand, costs| vec![cost::rank_selector_plan(demand, costs)])
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
    let demand = demand(group, cx);
    let range = narrow_range(group, cx);
    offer(&demand, &cost::unit_costs(cx.target))
        .into_iter()
        .map(|(plan, cost)| {
            let mut candidate = plan_candidate(group, demand.clone(), plan, cost, range.map(|(lo, hi, _)| (lo, hi)));
            candidate.pre.extend(range.map(|(_, _, fact)| Requirement::Fact(fact)));
            candidate
        })
        .collect()
}

/// The range of the sort's elements when the layer establishes one
/// narrower than the int64 window (`Context::element_range`): the plans
/// then bound each value they take to be an element into it
/// (`emit::Elements::bound`), so that their comparisons are narrow.
fn narrow_range(group: &Group, cx: &Context) -> Option<(i64, i64, crate::lower::Fact)> {
    cx.element_range(&group.shared).filter(|&(lo, hi, _)| (hi as i128 - lo as i128) < 1i128 << 62)
}

/// A candidate that lowers every lane of `group` by `plan` (bounding the
/// values it takes to be elements into `range`, when given).
fn plan_candidate(group: &Group, demand: Demand, plan: Plan, cost: u64, range: Option<(i64, i64)>) -> Candidate {
    Candidate {
        pre: Vec::new(),
        post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
        cost,
        emitter: Box::new(SortEmitter {
            elements: group.shared.clone(),
            payloads: payload_arrays(group),
            demand,
            plan,
            range,
            sort: None,
        }),
    }
}

/// The arrays a sort carries as payloads (the own operands of its payload
/// lanes), each once, in lane order.
fn payload_arrays(group: &Group) -> Vec<Vec<StmtId>> {
    let mut arrays: Vec<Vec<StmtId>> = Vec::new();
    for lane in group.lanes.iter().filter(|lane| lane.output == PAYLOAD) {
        if !arrays.contains(&lane.own) {
            arrays.push(lane.own.clone());
        }
    }
    arrays
}

/// Whether the program compares the sorted value `lane` for equality with
/// every element: `Some(true)` when the value is each comparison's first
/// operand, `Some(false)` when it is the second (in every comparison).
fn compared_with_every_element(group: &Group, cx: &Context, lane: StmtId) -> Option<bool> {
    let order = |element: StmtId| {
        cx.users(lane).iter().find_map(|&user| {
            let stmt = cx.stmt(user);
            if !matches!(stmt.ir, IR::EqI | IR::EqF) {
                return None;
            }
            match stmt.arguments[..] {
                [x, y] if x == lane && y == element => Some(true),
                [x, y] if x == element && y == lane => Some(false),
                _ => None,
            }
        })
    };
    let orders: Vec<bool> = group.shared.iter().map(|&e| order(e)).collect::<Option<_>>()?;
    let first = *orders.first()?;
    orders.iter().all(|&o| o == first).then_some(first)
}

/// The live outputs of a sort.
fn demand(group: &Group, cx: &Context) -> Demand {
    let mut demand = Demand { n: group.shared.len(), ..Demand::default() };
    let arrays = payload_arrays(group);
    demand.payloads = vec![BTreeSet::new(); arrays.len()];
    for lane in group.lanes.iter().filter(|lane| lane.output == PAYLOAD) {
        let array = arrays.iter().position(|a| *a == lane.own).expect("listed");
        let rank = lane.position.expect("a payload lane has a constant position");
        demand.payloads[array].insert(rank as usize);
    }
    let n = group.shared.len();
    for lane in group.lanes.iter().filter(|lane| lane.output == VALUES || lane.output == INDICES) {
        match (lane.position, lane.output == INDICES) {
            (Some(r), false) => {
                demand.values.insert(r as usize);
                if r == 0 || r as usize + 1 == n {
                    if let Some(value_first) = compared_with_every_element(group, cx, lane.stmt) {
                        demand.masked_extremes.insert(r as usize, value_first);
                    }
                }
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
    /// The payload arrays (`payload_arrays`), as statements.
    payloads: Vec<Vec<StmtId>>,
    demand: Demand,
    plan: Plan,
    /// The elements' established range (`narrow_range`), when the plan
    /// relies on it.
    range: Option<(i64, i64)>,
    sort: Option<LoweredSort>,
}

impl Emitter for SortEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let elements = cx.values(&self.elements);
        let payloads: Vec<Payload> = self
            .payloads
            .iter()
            .map(|array| {
                let slots = cx.values(array);
                let dtype = if slots.iter().any(|v| matches!(v, Value::Float(_))) {
                    NumberType::Float
                } else {
                    NumberType::Integer
                };
                Payload { slots, dtype }
            })
            .collect();
        self.sort = Some(LoweredSort::new(b, &self.demand, &self.plan, &elements, payloads, self.range));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let sort = self.sort.as_mut().expect("started");
        if lane.output == COUNT_AT_MOST || lane.output == COUNT_BELOW {
            let e = cx.value(lane.own[0]);
            return counts::count_query(b, &sort.as_elements(), &e, lane.output == COUNT_AT_MOST);
        }
        if lane.output == PAYLOAD {
            let array = self.payloads.iter().position(|a| *a == lane.own).expect("listed");
            let rank = lane.position.expect("a payload lane has a constant position") as usize;
            return sort.payload_output(array, rank);
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
    /// The elements' established range (`narrow_range`).
    range: Option<(i64, i64)>,
    /// Values and indices by constant rank.
    values: HashMap<usize, Value>,
    indices: HashMap<usize, Value>,
    /// Whether every rank's outputs exist (a network or the full run), so
    /// runtime ranks read them instead of being pinned on their own.
    complete: bool,
    element_memory: Option<ElementMemory>,
    value_memory: Option<OutputMemory>,
    index_memory: Option<OutputMemory>,
    /// Each payload's slots by rank: carried through a network, or read at
    /// the hinted indices.
    payloads: Vec<HashMap<usize, Value>>,
}

impl LoweredSort {
    fn new(
        b: &mut IRBuilder,
        demand: &Demand,
        plan: &Plan,
        elements: &[Value],
        mut payloads: Vec<Payload>,
        range: Option<(i64, i64)>,
    ) -> LoweredSort {
        let dtype = if elements.iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        };
        let mut sort = LoweredSort {
            range,
            elements: elements.to_vec(),
            dtype,
            values: HashMap::new(),
            indices: HashMap::new(),
            complete: false,
            element_memory: None,
            value_memory: None,
            index_memory: None,
            payloads: Vec::new(),
        };
        let n = elements.len();
        match plan {
            Plan::Network { network: Network::PackedMergeExchange } => {
                let indices = packed::argsort_by_packed_keys(b, &sort.as_elements(), &mut payloads);
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::Network { network } => {
                let (values, indices) = emit::sort_by_network(b, &sort.as_elements(), *network, &mut payloads);
                sort.values = values.into_iter().enumerate().collect();
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::Routed | Plan::GrandProduct => {
                let with_values = !demand.values.is_empty() || demand.runtime_values > 0;
                let sort_by = if *plan == Plan::Routed { routed::sort_by_routing } else { grand_product::sort_by_grand_product };
                let (values, indices) = sort_by(b, &sort.as_elements(), with_values, &mut payloads);
                sort.values = values.into_iter().enumerate().collect();
                sort.indices = indices.into_iter().enumerate().collect();
                sort.complete = true;
            }
            Plan::RankSelectors => {
                let selectors = emit::rank_selectors(b, &sort.as_elements());
                let indices: Vec<Value> = (0..n).map(|k| b.ir_constant_int(k as i64)).collect();
                for (r, by_element) in selectors.iter().enumerate() {
                    let value = emit::select_by(b, by_element, elements, dtype);
                    let index = emit::select_by(b, by_element, &indices, NumberType::Integer);
                    sort.values.insert(r, value);
                    sort.indices.insert(r, index);
                }
                for payload in &mut payloads {
                    payload.slots = selectors.iter().map(|s| emit::select_by(b, s, &payload.slots, payload.dtype)).collect();
                }
                sort.complete = true;
            }
            Plan::Hinted { runs } => {
                for run in runs {
                    match run.pin {
                        Pin::Keys => sort.pin_keys(b, run),
                        Pin::Counts => sort.pin_by_counts(b, demand, run),
                        Pin::Extreme | Pin::MaskedExtreme => {
                            let largest = run.first + 1 == n && n > 1;
                            let membership = match (run.pin, demand.masked_extremes.get(&run.first)) {
                                (Pin::MaskedExtreme, Some(&value_first)) => Membership::EqualityBits { value_first },
                                _ => Membership::Product,
                            };
                            let value = emit::extreme_value(b, &sort.as_elements(), largest, membership);
                            sort.values.insert(run.first, value);
                        }
                    }
                }
                sort.complete = matches!(runs.as_slice(), [run] if run.pin == Pin::Keys && run.first == 0 && run.last + 1 == n);
                sort.read_payloads(b, demand, &payloads);
                return sort;
            }
        }
        sort.payloads = payloads.into_iter().map(|p| p.slots.into_iter().enumerate().collect()).collect();
        sort
    }

    /// Each payload's live ranks read at the hinted argsort indices (the
    /// plan's runs cover every payload rank, see `cost::Demand`). The index
    /// is pinned to the stable argsort and bounded to `[0, n)`, so the read
    /// is the payload entry of the element at that rank.
    fn read_payloads(&mut self, b: &mut IRBuilder, demand: &Demand, payloads: &[Payload]) {
        for (payload, ranks) in payloads.iter().zip(&demand.payloads) {
            let segment = crate::helpers::segment::alloc_and_write_values(b, &payload.slots, payload.dtype);
            let slots = ranks
                .iter()
                .map(|&r| {
                    let index = self.output(r, true);
                    (r, crate::helpers::segment::read_typed(b, segment, &index, payload.dtype))
                })
                .collect();
            self.payloads.push(slots);
        }
    }

    fn payload_output(&self, array: usize, rank: usize) -> Value {
        self.payloads[array]
            .get(&rank)
            .cloned()
            .unwrap_or_else(|| panic!("sort lowering: payload rank {rank} was not lowered"))
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
        Elements { values: &self.elements, dtype: self.dtype, range: self.range }
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

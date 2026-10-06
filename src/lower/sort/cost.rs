//! The cost rule of the sort lowering: which lowering each sort's live
//! outputs get, from unit costs measured on each compile target.
//!
//! Every lowering computes the same outputs (NumPy's stable sort), so the
//! choice only moves cost. It depends on the target because the two
//! backends price the operations very differently: a read at a hinted
//! index is a one-hot selection over every slot in halo2-base (about 20
//! cells per slot) but a native memory read in Noir (about 7 gates, plus 3
//! per slot to build the array once), while a compare-exchange costs about
//! 79 cells against 24 gates. The rule is the same for every program: the
//! plan with the smallest estimated cost for the live outputs.

use std::collections::BTreeSet;

use crate::lower::Target;

/// The unit costs measured on `target`.
pub(crate) fn unit_costs(target: Target) -> UnitCosts {
    match target {
        Target::Halo2Base => HALO2_BASE,
        Target::Noir => NOIR,
    }
}

/// Costs of the pieces the plans are made of, in the target's metric
/// (halo2-base: advice + lookup cells; Noir: `bb gates`), measured on the
/// release build (2026-10-06) from small kernels: a 100-element argsort and
/// sort under each plan, the 7 largest of 100, a median, a minimum and a
/// value at a runtime rank of 100 elements, minus the same kernel's input
/// handling (`return a`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnitCosts {
    /// One compare-exchange of values (Batcher's network on values, the
    /// adjacent-swap networks without indices): the comparison, and a
    /// select per output that is live or compared again (`value_selects`;
    /// it reuses the comparison's difference).
    compare: u32,
    compare_select: u32,
    /// One compare-exchange of `(value, index)` keys with the index
    /// tie-break (`IR::KeyLess`) and four selects.
    exchange_keys: u32,
    /// One compare-exchange of values that also moves the indices (the
    /// adjacent-swap argsort).
    exchange_tracked: u32,
    /// One compare-exchange of packed keys (`packed::argsort_by_packed_keys`),
    /// the packing of one element, and the unpacking of one output index.
    exchange_packed: u32,
    pack: u32,
    unpack: u32,
    /// Exchanging one payload's two slots on a comparator's result.
    exchange_payload: u32,
    /// A hinted key: its index bounded and its value read, a fixed part and
    /// a part per slot of the segment read.
    key_read_fixed: u32,
    key_read_per_slot: u32,
    /// Making `n` values readable at runtime addresses, per slot, once per
    /// segment (Noir builds an array; halo2-base reads by selection).
    segment_per_slot: u32,
    /// An asserted key comparison (consecutive keys of a run).
    key_link: u32,
    /// One term of a count of the keys below or above a key.
    key_count_term: u32,
    /// One element's terms in an order statistic's two counts (a
    /// comparison, an equality test and two additions).
    rank_count_term: u32,
    /// The minimum's (maximum's) pin: per element an asserted comparison
    /// (one range check of its gap) and a factor of the membership
    /// product; once, the hint's range check and the product's zero test.
    extreme_term: u32,
    extreme_fixed: u32,
    /// The extreme pin when the program's equality bits give the
    /// membership: per element the asserted bound only; once the hint's
    /// window check and the bits' sum tested non-zero. Measured on the row
    /// maxima of 0436 (halo2-base 41 cells per element, as much as the
    /// bound's 64-bit gap check; Noir one gate per row more than the
    /// product at five elements): it saves only the product's factors.
    masked_extreme_term: u32,
    masked_extreme_fixed: u32,
    /// The routed Waksman network (`routed`), `None` where it is not
    /// offered.
    routing: Option<RoutingCosts>,
}

/// Unit costs of the routed Waksman network (`routed`, `routed_plan`); the
/// packing and unpacking are `UnitCosts::pack` and `UnitCosts::unpack`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutingCosts {
    /// One switch: its hinted bit inside its bit gate and the two selects
    /// moving the packed keys.
    switch: u32,
    /// The two selects moving one more carried array (the sorted values, a
    /// payload) through a switch.
    carried: u32,
    /// The asserted order of two consecutive routed keys (one range check
    /// of their gap).
    link: u32,
}

/// halo2-base: an argsort of 100 by Batcher's network costs 84,600 cells
/// (78.5 per exchange with the `KeyLess` tie-break), 67,800 on packed keys
/// (55 per exchange, 4 per packing, 46 per unpacked index), a sort 54,900
/// (51 per exchange: a comparison of 43 and two selects of 4; a select is
/// left out when its output is neither live nor compared again, as in the
/// argsort of 3 carrying payloads of 0355 / 0358); a read at a hinted index
/// is a selection over the slots (`prove/halo2_base/synthesizer/memory.rs`):
/// 13 + 8.24 cells per slot measured for 5 to 100 slots, priced 20 + 8 per
/// slot; `KeyLess` is about 55 cells; an order
/// statistic of 100 about 4,700-6,500; the extreme pin about 42 cells per
/// element plus 38 (fitted on maxima of 5 and minima of 48 values in the
/// perf cases 0436 and 0314), against 47-51 per scan step, so a scan stays
/// cheaper below 10 values.
const HALO2_BASE: UnitCosts = UnitCosts {
    compare: 43,
    compare_select: 4,
    exchange_keys: 79,
    exchange_tracked: 63,
    exchange_packed: 55,
    pack: 4,
    unpack: 46,
    exchange_payload: 12,
    key_read_fixed: 20,
    key_read_per_slot: 8,
    segment_per_slot: 0,
    key_link: 55,
    key_count_term: 59,
    rank_count_term: 50,
    extreme_term: 42,
    extreme_fixed: 38,
    masked_extreme_term: 41,
    masked_extreme_fixed: 30,
    routing: Some(HALO2_BASE_ROUTING),
};

/// halo2-base: a switch is a bit gate holding its hint (4 cells) and two
/// selects of packed keys (the difference and two multiply-adds, 12
/// cells); a carried array adds a difference and two selects (12); an
/// asserted link is one range check of a gap of up to `65 + b` bits (40
/// cells at lookup width 8, 41 for the difference). Measured on argsorts of
/// 3 to 100 elements (`eval-redo/layer/sort-route.md`).
const HALO2_BASE_ROUTING: RoutingCosts = RoutingCosts { switch: 16, carried: 12, link: 42 };

/// Noir: an argsort of 100 by the permutation check costs 1,520 gates (a
/// native read per key, the array built once, an asserted `KeyLess` per
/// link), by Batcher's network 26,100 (24 per exchange); a sort's exchange
/// is 17; the 7 largest of 100 cost 1,000 (one count of 100 key terms); an
/// order statistic of 100 about 840; the extreme pin about 5 per element
/// plus 3 (a maximum of 3 values costs 15 gates less than its scan).
const NOIR: UnitCosts = UnitCosts {
    // A compare-exchange of 17 gates whether or not its selects are used
    // (not recalibrated per select).
    compare: 17,
    compare_select: 0,
    exchange_keys: 24,
    exchange_tracked: 20,
    exchange_packed: 22,
    pack: 0,
    unpack: 10,
    exchange_payload: 2,
    key_read_fixed: 7,
    key_read_per_slot: 0,
    segment_per_slot: 3,
    key_link: 6,
    key_count_term: 10,
    rank_count_term: 9,
    extreme_term: 5,
    extreme_fixed: 3,
    masked_extreme_term: 5,
    masked_extreme_fixed: 4,
    // Not offered: the permutation check's native reads are linear in `n`.
    routing: None,
};

/// The outputs of one sort that are still live after dead-code
/// elimination.
#[derive(Clone, Debug, Default)]
pub(super) struct Demand {
    pub n: usize,
    /// Constant ranks whose sorted value is live.
    pub values: BTreeSet<usize>,
    /// Constant ranks whose argsort index is live.
    pub indices: BTreeSet<usize>,
    /// Live outputs at runtime ranks (values, indices).
    pub runtime_values: usize,
    pub runtime_indices: usize,
    /// The live ranks of each payload the sort carries (`emit::Payload`).
    pub payloads: Vec<BTreeSet<usize>>,
    /// The extreme ranks (0, `n - 1`) whose value the program compares for
    /// equality with every element (`a.max(axis) == a`), with whether the
    /// value is the comparisons' first operand: their pin can take its
    /// membership from those equality bits (`emit::extreme_value`).
    pub masked_extremes: std::collections::BTreeMap<usize, bool>,
}

impl Demand {
    /// The demand a hinted plan meets: a payload's rank is read at that
    /// rank's argsort index, so the index is needed.
    pub(super) fn with_payload_indices(&self) -> Demand {
        let mut demand = self.clone();
        for ranks in &self.payloads {
            demand.indices.extend(ranks);
        }
        demand
    }

    /// One read of each live payload rank (and a segment per payload).
    fn payload_reads(&self, costs: &UnitCosts) -> u64 {
        self.payloads.iter().map(|ranks| costs.reads(self.n, ranks.len())).sum()
    }
}

/// A run of consecutive ranks lowered together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Run {
    pub first: usize,
    pub last: usize,
    pub pin: Pin,
}

/// How a run's outputs are pinned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pin {
    /// Hinted keys for every rank of the run (`emit::pinned_run`).
    Keys,
    /// Each live rank's value by its own counts (`emit::order_statistic`):
    /// only for runs whose live outputs are all values.
    Counts,
    /// The smallest (rank 0) or largest (rank `n - 1`) value alone, by
    /// asserted bounds and a membership product (`emit::extreme_value`).
    Extreme,
    /// The same, membership by the program's equality bits
    /// (`Demand::masked_extremes`).
    MaskedExtreme,
}

/// How a sort's live outputs are computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Plan {
    /// A sorting network over all keys (`emit::sort_by_network`), pruned to
    /// the live outputs by dead-code elimination; runtime-rank outputs read
    /// the network's outputs.
    Network { network: Network },
    /// Hinted outputs pinned in runs of ranks (`emit::pinned_run`); a run
    /// over every rank is the full permutation check, whose outputs the
    /// runtime-rank outputs then read. Without it, runtime-rank values are
    /// pinned by counts and runtime-rank indices by their key's rank.
    Hinted { runs: Vec<Run> },
    /// Packed keys routed into ascending order by a Waksman network whose
    /// switches the prover sets, checked increasing (`routed`); every
    /// rank's outputs, which runtime-rank outputs read.
    Routed,
}

/// A sorting network and how it orders equal values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Network {
    /// Batcher's merge exchange on values only (equal values are
    /// interchangeable; no indices are tracked).
    MergeExchange,
    /// Batcher's merge exchange on `(value, index)` keys (stable).
    MergeExchangeTieBreak,
    /// Batcher's merge exchange on packed keys `v·2^b + j` (stable); for
    /// sorts whose live outputs are indices only.
    PackedMergeExchange,
    /// The adjacent-swap network, stable by construction, moving indices.
    AdjacentSwap,
    /// The adjacent-swap network on values only (its first pass is the
    /// scan for the largest value).
    AdjacentSwapValues,
    /// The same network passing the other way (its first pass is the scan
    /// for the smallest value).
    ReverseAdjacentSwapValues,
}

impl Network {
    pub(super) fn pairs(self, n: usize) -> Vec<(usize, usize)> {
        match self {
            Network::MergeExchange | Network::MergeExchangeTieBreak | Network::PackedMergeExchange => {
                merge_exchange_pairs(n)
            }
            Network::AdjacentSwap | Network::AdjacentSwapValues => adjacent_swap_pairs(n),
            Network::ReverseAdjacentSwapValues => reverse_adjacent_swap_pairs(n),
        }
    }

    pub(super) fn tracks_indices(self) -> bool {
        matches!(self, Network::MergeExchangeTieBreak | Network::AdjacentSwap | Network::PackedMergeExchange)
    }

    /// Whether equal values keep their original order (what a payload or an
    /// argsort needs).
    pub(super) fn is_stable(self) -> bool {
        self.tracks_indices()
    }

    pub(super) fn tie_break(self) -> bool {
        matches!(self, Network::MergeExchangeTieBreak)
    }
}

impl UnitCosts {
    /// One read of a segment of `n` values at a bounded runtime address.
    fn read(&self, n: usize) -> u64 {
        u64::from(self.key_read_fixed) + n as u64 * u64::from(self.key_read_per_slot)
    }

    /// `count` reads of one segment of `n` values, with the segment (none
    /// when `count` is 0).
    fn reads(&self, n: usize, count: usize) -> u64 {
        if count == 0 {
            return 0;
        }
        n as u64 * u64::from(self.segment_per_slot) + count as u64 * self.read(n)
    }

    /// One compare-exchange of a network that tracks indices or packed
    /// keys (value-only exchanges are priced by `compare` and
    /// `compare_select`).
    fn tracked_comparator(&self, network: Network) -> u64 {
        u64::from(match network {
            Network::MergeExchangeTieBreak => self.exchange_keys,
            Network::PackedMergeExchange => self.exchange_packed,
            Network::AdjacentSwap => self.exchange_tracked,
            Network::MergeExchange | Network::AdjacentSwapValues | Network::ReverseAdjacentSwapValues => {
                unreachable!("a value-only network has no tracked comparator")
            }
        })
    }

    /// One value pinned by `count(a < v) <= r < count(a <= v)`.
    fn order_statistic(&self, n: usize) -> u64 {
        n as u64 * u64::from(self.rank_count_term)
    }

    /// The smallest or largest of `n` values.
    fn extreme(&self, n: usize) -> u64 {
        n as u64 * u64::from(self.extreme_term) + u64::from(self.extreme_fixed)
    }

    /// The smallest or largest of `n` values, membership by the program's
    /// equality bits.
    fn masked_extreme(&self, n: usize) -> u64 {
        n as u64 * u64::from(self.masked_extreme_term) + u64::from(self.masked_extreme_fixed)
    }

    /// The runtime-rank index whose key has exactly `r` keys below it,
    /// without its read (see `reads`).
    fn index_at_rank(&self, n: usize) -> u64 {
        n as u64 * u64::from(self.key_count_term)
    }

    /// The ranks `first..=last` as hinted keys, without their reads (see
    /// `reads`): the links between them and, at each end that is not an
    /// end of the sort, a count of the keys beyond it.
    fn key_run(&self, n: usize, first: usize, last: usize) -> u64 {
        let len = (last - first + 1) as u64;
        let boundaries = u64::from(first > 0) + u64::from(last + 1 < n);
        (len - 1) * u64::from(self.key_link) + boundaries * n as u64 * u64::from(self.key_count_term)
    }
}

/// The compare-exchange pairs `(i, j)`, `i < j`, of Batcher's
/// merge-exchange sorting network for `n` keys, in execution order (Knuth,
/// TAOCP vol. 3, 5.2.2, Algorithm M). Applying "put the smaller key at
/// `i`" to every pair sorts any `n` keys. It uses O(n log^2 n)
/// comparators: 97 for n = 20, 395 for n = 50, 1,077 for n = 100, where an
/// adjacent-swap network needs n(n-1)/2 (190, 1,225, 4,950).
pub(super) fn merge_exchange_pairs(n: usize) -> Vec<(usize, usize)> {
    if n < 2 {
        return Vec::new();
    }
    let t = usize::BITS - (n - 1).leading_zeros(); // ceil(log2 n)
    let mut pairs = Vec::new();
    let mut p = 1usize << (t - 1);
    while p > 0 {
        let (mut q, mut r, mut d) = (1usize << (t - 1), 0, p);
        loop {
            pairs.extend((0..n - d).filter(|i| i & p == r).map(|i| (i, i + d)));
            if q == p {
                break;
            }
            d = q - p;
            q >>= 1;
            r = p;
        }
        p >>= 1;
    }
    pairs
}

/// The compare-exchange pairs of the adjacent-swap (bubble) network: pass
/// `k` carries the largest remaining key to position `n - 1 - k`, so it
/// needs n(n-1)/2 comparators. Its comparators only exchange neighbours
/// that are strictly out of order, so equal keys never change their
/// relative order: the network is stable without any tie-break.
pub(super) fn adjacent_swap_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n.saturating_sub(1))
        .flat_map(|k| (0..n - 1 - k).map(|i| (i, i + 1)))
        .collect()
}

/// The adjacent-swap network passing towards the start: pass `k` carries
/// the smallest remaining key to position `k`.
pub(super) fn reverse_adjacent_swap_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n.saturating_sub(1))
        .flat_map(|k| (k..n - 1).rev().map(|i| (i, i + 1)))
        .collect()
}

/// How many value selects the comparators of `pairs` in the cone of `live`
/// need (see `cone_size`): a comparator's output is selected when it is a
/// value in `live_values` or a later comparator of the cone compares it;
/// the others are removed by dead-code elimination.
pub(super) fn value_selects(pairs: &[(usize, usize)], n: usize, live: &BTreeSet<usize>, live_values: &BTreeSet<usize>) -> usize {
    let mut in_cone = vec![false; n];
    for &position in live {
        in_cone[position] = true;
    }
    let mut value_used = vec![false; n];
    for &position in live_values {
        value_used[position] = true;
    }
    let mut count = 0;
    for &(i, j) in pairs.iter().rev() {
        if in_cone[i] || in_cone[j] {
            count += usize::from(value_used[i]) + usize::from(value_used[j]);
            in_cone[i] = true;
            in_cone[j] = true;
            value_used[i] = true;
            value_used[j] = true;
        }
    }
    count
}

/// How many of `pairs` the outputs at `live` positions depend on (the
/// comparators dead-code elimination keeps): walking the network
/// backwards, a comparator is needed when either of its positions is live
/// after it, and then both are live before it.
pub(super) fn cone_size(pairs: &[(usize, usize)], n: usize, live: &BTreeSet<usize>) -> usize {
    let mut needed = vec![false; n];
    for &position in live {
        needed[position] = true;
    }
    let mut count = 0;
    for &(i, j) in pairs.iter().rev() {
        if needed[i] || needed[j] {
            needed[i] = true;
            needed[j] = true;
            count += 1;
        }
    }
    count
}

/// The cheapest plan for `demand` under `costs`, with its estimated cost:
/// the first of the cheapest among the plans of the four constructions,
/// in their registration order (as the selection breaks ties).
#[cfg(test)]
pub(super) fn choose_plan(demand: &Demand, costs: &UnitCosts) -> (Plan, u64) {
    network_plans(demand, costs)
        .into_iter()
        .chain(full_permutation(demand, costs))
        .chain(selection_plans(demand, costs))
        .chain(routed_plan(demand, costs))
        .min_by_key(|(_, cost)| *cost)
        .expect("a network always applies")
}

/// Reads of the complete outputs at the runtime ranks: one segment per
/// kind of output read (values, indices).
fn runtime_reads_of_outputs(demand: &Demand, costs: &UnitCosts) -> u64 {
    costs.reads(demand.n, demand.runtime_values) + costs.reads(demand.n, demand.runtime_indices)
}

/// Every sorting network that computes the live outputs, with its cost
/// (the comparators in the cone of the live outputs, and the runtime reads).
pub(super) fn network_plans(demand: &Demand, costs: &UnitCosts) -> Vec<(Plan, u64)> {
    let n = demand.n;
    let runtime = demand.runtime_values + demand.runtime_indices;
    let mut live: BTreeSet<usize> = if runtime > 0 {
        (0..n).collect()
    } else {
        demand.values.union(&demand.indices).copied().collect()
    };
    for ranks in &demand.payloads {
        live.extend(ranks);
    }
    let needs_indices = !demand.indices.is_empty() || demand.runtime_indices > 0;
    let needs_values = !demand.values.is_empty() || demand.runtime_values > 0;
    let needs_stable = needs_indices || !demand.payloads.is_empty();
    let candidates: &[Network] = if needs_stable && !needs_values {
        &[Network::MergeExchangeTieBreak, Network::PackedMergeExchange, Network::AdjacentSwap]
    } else if needs_stable {
        &[Network::MergeExchangeTieBreak, Network::AdjacentSwap]
    } else {
        &[Network::MergeExchange, Network::AdjacentSwapValues, Network::ReverseAdjacentSwapValues]
    };
    let live_values: BTreeSet<usize> = if demand.runtime_values > 0 { (0..n).collect() } else { demand.values.clone() };
    let reads = runtime_reads_of_outputs(demand, costs);
    candidates
        .iter()
        .map(|&network| {
            let pairs = network.pairs(n);
            let comparators = cone_size(&pairs, n, &live) as u64;
            let packing = if network == Network::PackedMergeExchange {
                let unpacked = if demand.runtime_indices > 0 { n } else { demand.indices.len() };
                n as u64 * u64::from(costs.pack) + unpacked as u64 * u64::from(costs.unpack)
            } else {
                0
            };
            let carried: u64 = demand
                .payloads
                .iter()
                .map(|ranks| cone_size(&pairs, n, ranks) as u64 * u64::from(costs.exchange_payload))
                .sum();
            // The adjacent-swap network is stable without its indices: when
            // none is live, dead-code elimination drops their selects.
            let exchanges = if network.tracks_indices() && !(network == Network::AdjacentSwap && !needs_indices) {
                comparators * costs.tracked_comparator(network)
            } else {
                comparators * u64::from(costs.compare)
                    + value_selects(&pairs, n, &live, &live_values) as u64 * u64::from(costs.compare_select)
            };
            (Plan::Network { network }, exchanges + packing + carried + reads)
        })
        .collect()
}

/// The routed Waksman network (`routed`) when the target offers it and
/// there are at least two elements: every rank, the sorted values carried
/// when one is live, each payload carried; runtime-rank outputs read its
/// outputs.
pub(super) fn routed_plan(demand: &Demand, costs: &UnitCosts) -> Option<(Plan, u64)> {
    let routing = costs.routing?;
    let n = demand.n;
    if n < 2 {
        return None;
    }
    let switches = super::waksman::switch_count(n) as u64;
    let needs_values = !demand.values.is_empty() || demand.runtime_values > 0;
    let carried = u64::from(needs_values) + demand.payloads.len() as u64;
    let unpacked = if demand.runtime_indices > 0 { n } else { demand.indices.len() };
    let cost = n as u64 * u64::from(costs.pack)
        + switches * (u64::from(routing.switch) + carried * u64::from(routing.carried))
        + (n as u64 - 1) * u64::from(routing.link)
        + unpacked as u64 * u64::from(costs.unpack)
        + runtime_reads_of_outputs(demand, costs);
    Some((Plan::Routed, cost))
}

/// Every rank as one pinned run (the permutation check); runtime-rank
/// outputs read its outputs.
pub(super) fn full_permutation(demand: &Demand, costs: &UnitCosts) -> Option<(Plan, u64)> {
    let n = demand.n;
    let run = Run { first: 0, last: n - 1, pin: Pin::Keys };
    let cost = costs.key_run(n, 0, n - 1)
        + costs.reads(n, n)
        + runtime_reads_of_outputs(demand, costs)
        + demand.payload_reads(costs);
    Some((Plan::Hinted { runs: vec![run] }, cost))
}

/// The live constant ranks covered by runs (`cheapest_runs`), runtime-rank
/// values pinned by counts and runtime-rank indices by their key's rank,
/// with key runs allowed and (when no index is live) without them. Every
/// key (a run's rank, a runtime-rank index) reads one shared segment of the
/// elements.
pub(super) fn selection_plans(demand: &Demand, costs: &UnitCosts) -> Vec<(Plan, u64)> {
    let payload_reads = demand.payload_reads(costs);
    let demand = &demand.with_payload_indices();
    let n = demand.n;
    let per_output = demand.runtime_values as u64 * costs.order_statistic(n)
        + demand.runtime_indices as u64 * costs.index_at_rank(n);
    [true, false]
        .into_iter()
        .filter_map(|allow_keys| {
            let (runs, runs_cost) = cheapest_runs(demand, costs, allow_keys)?;
            let keys: usize = runs
                .iter()
                .filter(|run| run.pin == Pin::Keys)
                .map(|run| run.last - run.first + 1)
                .sum::<usize>()
                + demand.runtime_indices;
            // `runs_cost` has the read of each run's keys; add the segment
            // they share (once) and the reads of the runtime-rank indices.
            let segment = if keys > 0 { n as u64 * u64::from(costs.segment_per_slot) } else { 0 };
            let runtime_reads = demand.runtime_indices as u64 * costs.read(n);
            Some((Plan::Hinted { runs }, runs_cost + per_output + segment + runtime_reads + payload_reads))
        })
        .collect()
}

/// The cheapest cover of the live constant ranks by runs: a dynamic
/// program over the live ranks in order, where a run spans from one live
/// rank to a later one (including the ranks between) and costs
/// `UnitCosts::key_run` plus one read per rank (without the segment, which
/// the caller adds once), or, when every live output it covers is a value,
/// the cheaper of that and one count per live rank (one extreme pin for a
/// lone rank 0 or `n - 1`). Without `allow_keys` only counts and extremes
/// are allowed (`None` when an index is live).
fn cheapest_runs(demand: &Demand, costs: &UnitCosts, allow_keys: bool) -> Option<(Vec<Run>, u64)> {
    let n = demand.n;
    let live: Vec<usize> = demand.values.union(&demand.indices).copied().collect();
    let m = live.len();
    // best[i]: the cheapest cover of live[..i], and the run ending it.
    let mut best: Vec<(u64, Option<(usize, Run)>)> = vec![(0, None); m + 1];
    for end in 1..=m {
        best[end] = (u64::MAX, None);
        for start in 0..end {
            let (first, last) = (live[start], live[end - 1]);
            let values_only = live[start..end].iter().all(|r| !demand.indices.contains(r));
            let mut options: Vec<(u64, Pin)> = Vec::new();
            if allow_keys {
                options.push((costs.key_run(n, first, last) + (last - first + 1) as u64 * costs.read(n), Pin::Keys));
            }
            if values_only {
                options.push(((end - start) as u64 * costs.order_statistic(n), Pin::Counts));
                if first == last && (first == 0 || first + 1 == n) {
                    options.push((costs.extreme(n), Pin::Extreme));
                    if demand.masked_extremes.contains_key(&first) {
                        options.push((costs.masked_extreme(n), Pin::MaskedExtreme));
                    }
                }
            }
            let Some(&(cost, pin)) = options.iter().min_by_key(|(cost, _)| *cost) else {
                continue;
            };
            let total = best[start].0.saturating_add(cost);
            if total < best[end].0 {
                best[end] = (total, Some((start, Run { first, last, pin })));
            }
        }
    }
    if best[m].0 == u64::MAX {
        return None;
    }
    let mut runs = Vec::new();
    let mut end = m;
    while end > 0 {
        let (start, run) = best[end].1.expect("every prefix has a cover");
        runs.push(run);
        end = start;
    }
    runs.reverse();
    Some((runs, best[m].0))
}

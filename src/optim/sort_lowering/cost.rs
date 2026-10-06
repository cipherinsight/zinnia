//! The cost rule of the sort lowering: which lowering each sort's live
//! outputs get, from unit costs measured on each compile target.
//!
//! Every lowering computes the same outputs (NumPy's stable sort), so the
//! choice only moves cost. It depends on the target because the two
//! backends price the operations very differently: a dynamic read is a
//! one-hot selection over every slot in halo2-base (about 15 cells per
//! slot) but a native memory read in Noir (about 10 gates, plus 3 per slot
//! to build the array once), while comparisons cost about 43 cells against
//! about 5 gates. The rule is the same for every program: the plan with
//! the smallest estimated cost for the live outputs.

use std::collections::BTreeSet;

/// The backend a program is compiled for (`ZinniaConfig.backend`). Any IR
/// proves on any backend; the target only selects the cheaper lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortTarget {
    Halo2Base,
    Noir,
}

impl SortTarget {
    /// The target of a `ZinniaConfig.backend` name: `noir`, else halo2-base
    /// (the default backend, `halo2`).
    pub fn from_backend(backend: &str) -> SortTarget {
        if backend == "noir" {
            SortTarget::Noir
        } else {
            SortTarget::Halo2Base
        }
    }

    pub(super) fn costs(self) -> UnitCosts {
        match self {
            SortTarget::Halo2Base => HALO2_BASE,
            SortTarget::Noir => NOIR,
        }
    }
}

/// Costs of the operations the lowerings emit, in the target's metric
/// (halo2-base: advice + lookup cells; Noir: `bb gates`).
#[derive(Clone, Copy, Debug)]
pub(super) struct UnitCosts {
    /// `a < b` of two runtime values (a bit).
    compare: u32,
    /// `a == b` of two runtime values.
    equal: u32,
    /// A select between two runtime values.
    select: u32,
    /// `i < j` of two runtime indices in `[0, n)`.
    index_less: u32,
    /// `j < p` of a constant and a runtime index in `[0, n)`.
    constant_index_less: u32,
    /// A logical and / or of two bits.
    logic: u32,
    /// Adding a bit to a count.
    count_add: u32,
    /// `0 <= p < n` of a runtime index.
    bound: u32,
    /// Making a segment of `n` values readable at runtime addresses, per
    /// slot, once per segment (Noir builds an array; halo2-base nothing).
    segment_per_slot: u32,
    /// A dynamic read: a fixed part and a part per slot of the segment.
    read_fixed: u32,
    read_per_slot: u32,
}

/// Measured on the release build (`costs_halo2 --profile`, 2026-10-06):
/// `LtF` 43 (35 advice + 8 lookup), `EqF` 12, `LtIBounded` 14, selects 4,
/// `LogicalOr` 8, `LogicalAnd` 4; a dynamic read is an equality indicator
/// and an inner-product term per slot.
const HALO2_BASE: UnitCosts = UnitCosts {
    compare: 43,
    equal: 12,
    select: 4,
    index_less: 14,
    constant_index_less: 8,
    logic: 6,
    count_add: 4,
    bound: 8,
    segment_per_slot: 0,
    read_fixed: 4,
    read_per_slot: 15,
};

/// Measured with `bb gates` (UltraHonk): a comparison is one hinted bit and
/// a 64-bit gap check; a native read is about 10 gates, plus 3 per element
/// to build the array, once for all reads of a segment.
const NOIR: UnitCosts = UnitCosts {
    compare: 6,
    equal: 3,
    select: 1,
    index_less: 3,
    constant_index_less: 2,
    logic: 1,
    count_add: 1,
    bound: 3,
    segment_per_slot: 3,
    read_fixed: 10,
    read_per_slot: 0,
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
}

/// A run of consecutive ranks lowered together (see `emit::pinned_run`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Run {
    pub first: usize,
    pub last: usize,
    /// Each rank's value pinned by its own count (`emit::order_statistic`)
    /// instead of keys: only for runs whose live outputs are all values.
    pub by_count: bool,
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
}

/// A sorting network and how it orders equal values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Network {
    /// Batcher's merge exchange on values only (equal values are
    /// interchangeable; no indices are tracked).
    MergeExchange,
    /// Batcher's merge exchange on `(value, index)` keys (stable).
    MergeExchangeTieBreak,
    /// The adjacent-swap network, stable by construction.
    AdjacentSwap,
}

impl Network {
    pub(super) fn pairs(self, n: usize) -> Vec<(usize, usize)> {
        match self {
            Network::MergeExchange | Network::MergeExchangeTieBreak => merge_exchange_pairs(n),
            Network::AdjacentSwap => adjacent_swap_pairs(n),
        }
    }

    pub(super) fn tracks_indices(self) -> bool {
        !matches!(self, Network::MergeExchange)
    }

    pub(super) fn tie_break(self) -> bool {
        matches!(self, Network::MergeExchangeTieBreak)
    }
}

impl UnitCosts {
    /// One read of a segment of `n` values at a bounded runtime address.
    fn read(&self, n: usize) -> u64 {
        u64::from(self.bound + self.read_fixed) + n as u64 * u64::from(self.read_per_slot)
    }

    /// `count` reads of one segment of `n` values, with the segment (none
    /// when `count` is 0).
    fn reads(&self, n: usize, count: usize) -> u64 {
        if count == 0 {
            return 0;
        }
        n as u64 * u64::from(self.segment_per_slot) + count as u64 * self.read(n)
    }

    /// One compare-exchange of the network.
    fn comparator(&self, network: Network) -> u64 {
        let mut cost = self.compare + 2 * self.select;
        if network.tracks_indices() {
            cost += 2 * self.select;
        }
        if network.tie_break() {
            cost += self.equal + self.index_less + 2 * self.logic;
        }
        u64::from(cost)
    }

    /// `(v_i, p_i) < (v_j, p_j)` of two hinted keys, asserted.
    fn chain_link(&self) -> u64 {
        u64::from(self.compare + self.equal + self.index_less + 2 * self.logic)
    }

    /// One term of a count of the keys below (or above) a hinted key.
    fn key_count_term(&self) -> u64 {
        u64::from(self.compare + self.equal + self.constant_index_less + 2 * self.logic + self.count_add)
    }

    /// One value pinned by `count(a < v) <= r < count(a <= v)`.
    fn order_statistic(&self, n: usize) -> u64 {
        n as u64 * u64::from(self.compare + self.equal + 2 * self.count_add)
            + 2 * u64::from(self.compare)
    }

    /// The runtime-rank index whose key has exactly `r` keys below it,
    /// without its read (see `reads`).
    fn index_at_rank(&self, n: usize) -> u64 {
        n as u64 * self.key_count_term() + u64::from(self.equal)
    }

    /// The ranks `first..=last` as hinted keys, without their reads (see
    /// `reads`): the chain between them and, at each end that is not an end
    /// of the sort, a count of the keys beyond it.
    fn key_run(&self, n: usize, first: usize, last: usize) -> u64 {
        let len = (last - first + 1) as u64;
        let boundaries = u64::from(first > 0) + u64::from(last + 1 < n);
        (len - 1) * self.chain_link()
            + boundaries * (n as u64 * self.key_count_term() + u64::from(self.equal))
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

/// The cheapest plan for `demand` under `costs`, with its estimated cost.
pub(super) fn choose_plan(demand: &Demand, costs: &UnitCosts) -> (Plan, u64) {
    let candidates = [
        cheapest_network(demand, costs),
        full_permutation(demand, costs),
        cheapest_selection(demand, costs),
    ];
    candidates
        .into_iter()
        .flatten()
        .min_by_key(|(_, cost)| *cost)
        .expect("a network always applies")
}

/// Reads of the complete outputs at the runtime ranks: one segment per
/// kind of output read (values, indices).
fn runtime_reads_of_outputs(demand: &Demand, costs: &UnitCosts) -> u64 {
    costs.reads(demand.n, demand.runtime_values) + costs.reads(demand.n, demand.runtime_indices)
}

fn cheapest_network(demand: &Demand, costs: &UnitCosts) -> Option<(Plan, u64)> {
    let n = demand.n;
    let runtime = demand.runtime_values + demand.runtime_indices;
    let live: BTreeSet<usize> = if runtime > 0 {
        (0..n).collect()
    } else {
        demand.values.union(&demand.indices).copied().collect()
    };
    let needs_indices = !demand.indices.is_empty() || demand.runtime_indices > 0;
    let candidates: &[Network] = if needs_indices {
        &[Network::MergeExchangeTieBreak, Network::AdjacentSwap]
    } else {
        &[Network::MergeExchange]
    };
    let reads = runtime_reads_of_outputs(demand, costs);
    candidates
        .iter()
        .map(|&network| {
            let comparators = cone_size(&network.pairs(n), n, &live) as u64;
            (Plan::Network { network }, comparators * costs.comparator(network) + reads)
        })
        .min_by_key(|(_, cost)| *cost)
}

/// Every rank as one pinned run (the permutation check); runtime-rank
/// outputs read its outputs.
fn full_permutation(demand: &Demand, costs: &UnitCosts) -> Option<(Plan, u64)> {
    let n = demand.n;
    let run = Run { first: 0, last: n - 1, by_count: false };
    let cost = costs.key_run(n, 0, n - 1) + costs.reads(n, n) + runtime_reads_of_outputs(demand, costs);
    Some((Plan::Hinted { runs: vec![run] }, cost))
}

/// The live constant ranks covered by runs (`cheapest_runs`), runtime-rank
/// values pinned by counts and runtime-rank indices by their key's rank.
/// Every key (a run's rank, a runtime-rank index) reads one shared segment
/// of the elements.
fn cheapest_selection(demand: &Demand, costs: &UnitCosts) -> Option<(Plan, u64)> {
    let n = demand.n;
    let per_output = demand.runtime_values as u64 * costs.order_statistic(n)
        + demand.runtime_indices as u64 * costs.index_at_rank(n);
    [true, false]
        .into_iter()
        .filter_map(|allow_keys| {
            let (runs, runs_cost) = cheapest_runs(demand, costs, allow_keys)?;
            let keys: usize = runs
                .iter()
                .filter(|run| !run.by_count)
                .map(|run| run.last - run.first + 1)
                .sum::<usize>()
                + demand.runtime_indices;
            Some((Plan::Hinted { runs }, runs_cost + per_output + costs.reads(n, keys)))
        })
        .min_by_key(|(_, cost)| *cost)
}

/// The cheapest cover of the live constant ranks by runs: a dynamic
/// program over the live ranks in order, where a run spans from one live
/// rank to a later one (including the ranks between) and costs
/// `UnitCosts::key_run` plus one read per rank (without the segment, which
/// the caller adds once), or, when every live output it covers is a value,
/// the cheaper of that and one count per live rank. Without `allow_keys`
/// only counts are allowed (`None` when an index is live).
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
            let by_keys = allow_keys
                .then(|| costs.key_run(n, first, last) + (last - first + 1) as u64 * costs.read(n));
            let by_count = values_only.then(|| (end - start) as u64 * costs.order_statistic(n));
            let (cost, by_count) = match (by_keys, by_count) {
                (Some(k), Some(c)) if c < k => (c, true),
                (Some(k), _) => (k, false),
                (None, Some(c)) => (c, true),
                (None, None) => continue,
            };
            let total = best[start].0.saturating_add(cost);
            if total < best[end].0 {
                best[end] = (total, Some((start, Run { first, last, by_count })));
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

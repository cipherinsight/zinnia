//! P5 — `SmtTelemetry` for instrumenting the resolver pipeline.
//!
//! Lock-free counters (`AtomicUsize` / `AtomicU64`) shared across the layers
//! of a `LayeredResolver`. The intent is to see, on a worst-case benchmark,
//! where compile time goes:
//!
//! * Did range resolve (cheap)?
//! * Did SMT resolve (we paid Z3 to learn something)?
//! * Did SMT return unknown / timeout (we paid Z3 to learn nothing — bad)?
//! * What does the duration histogram look like? Is the 500 ms timeout
//!   genuinely the long-tail spend, or is it cheap-and-frequent queries?
//!
//! Wired in: `SmtResolver` and `RangeResolver` each hold an
//! `Arc<SmtTelemetry>`. A `LayeredResolver` shares one telemetry across the
//! layers via `with_telemetry`. Every counter is bumped before the resolver
//! returns, so the summary is consistent w.r.t. the "first Some wins" layered
//! semantics.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Number of histogram buckets for SMT query durations.
pub const NUM_DURATION_BUCKETS: usize = 8;

/// Bucket edges, log-spaced from 1 µs to 1 s. A query of duration `d` lands
/// in bucket `i` where `BUCKET_EDGES_NS[i-1] < d <= BUCKET_EDGES_NS[i]`.
/// Bucket 0: (0, 1µs], bucket 1: (1µs, 10µs], ... bucket 7: (>100ms, ∞).
const BUCKET_EDGES_NS: [u64; NUM_DURATION_BUCKETS - 1] = [
    1_000,           // 1µs
    10_000,          // 10µs
    100_000,         // 100µs
    1_000_000,       // 1ms
    10_000_000,      // 10ms
    100_000_000,     // 100ms
    1_000_000_000,   // 1s
];

/// All resolver-pipeline counters, lock-free. Accumulated across the
/// lifetime of a single compilation; reset implicitly when a fresh
/// `IRGenerator` / `IRGraph` builds a new telemetry instance.
#[derive(Debug, Default)]
pub struct SmtTelemetry {
    // -- Counters ------------------------------------------------------
    pub queries_total: AtomicUsize,
    /// Resolved by the static_val fast-path inside `SmtResolver` /
    /// `RangeResolver` (no walk, no Z3).
    pub queries_static_val_hit: AtomicUsize,
    /// Resolved by the range layer (interval collapsed to a point).
    pub queries_range_hit: AtomicUsize,
    /// Resolved by Z3 (returned `Some`).
    pub queries_smt_resolved: AtomicUsize,
    /// Z3 returned `unknown` / non-unique / `Sat` with an alternate model.
    /// "Paid Z3 to learn nothing."
    pub queries_smt_unknown: AtomicUsize,
    /// Z3 timeout. Counted alongside `queries_smt_unknown` (the resolver
    /// can't always distinguish "unknown" from "timeout" via the public
    /// API, but the duration histogram surfaces the tail).
    pub queries_smt_timeout: AtomicUsize,
    /// Resolved from the per-ptr cache (no walk, no Z3 — cheapest case
    /// after static_val).
    pub queries_cache_hit: AtomicUsize,
    /// `smt_enable=false` short-circuit path.
    pub queries_skipped_disabled: AtomicUsize,
    /// Formula-size budget exceeded (P5 commit 3 — only bumped when the
    /// reverse-reachability walk aborts early).
    pub queries_skipped_oversized: AtomicUsize,

    // -- Timing (nanos) -----------------------------------------------
    pub total_time_in_smt_ns: AtomicU64,
    pub total_time_in_range_ns: AtomicU64,

    // -- Duration histogram (SMT layer only) --------------------------
    pub smt_duration_buckets: [AtomicUsize; NUM_DURATION_BUCKETS],

    // -- Misc ---------------------------------------------------------
    /// Largest formula encountered, in IR statements (count of nodes
    /// visited by the reverse-reachability walk for one query).
    pub largest_formula_size: AtomicUsize,
    /// Cache occupancy when `summary()` is called. Snapshot, not running.
    pub cache_size_at_end: AtomicUsize,

    // -- AlwaysSatisfiedElimination counters (P4 round 1) -------------
    /// `assert(cond)` removed because constant-folding proved `cond` true
    /// — the pre-P4 path. Cheap (no resolver invocation).
    pub assertions_eliminated_const_fold: AtomicUsize,
    /// `assert(cond)` removed because the resolver (range / SMT)
    /// proved `cond` true. P4 round 1 — counts how many times we paid
    /// resolver cost and got a productive elimination.
    pub assertions_eliminated_resolver: AtomicUsize,
    /// `assert(cond)` whose `cond` was provably false at compile time.
    /// We turn these into compile-time errors; the counter is bumped
    /// before the error is raised (so it shows up in telemetry even on
    /// the failing path). P4 round 1.
    pub assertions_provably_false: AtomicUsize,

    // -- Recursive-chip bound discharge counters (P4 round 2) ---------
    /// Recursive chip call where the measure was a literal int — bound
    /// proved by `int_val()` without consulting the resolver.
    pub recursion_bound_static_val: AtomicUsize,
    /// Recursive chip call where `resolve_max(measure)` returned `Some(n)`.
    /// We tightened the per-call unroll cap to `min(n, recursion_limit)`.
    pub recursion_bound_resolver_proved: AtomicUsize,
    /// Recursive chip call where the heuristic could not pick a measure
    /// (no integer arg decreased across the call). Falls back to the
    /// hard `recursion_limit` budget. High counter values suggest the
    /// heuristic needs a user-side `# zinnia: recursion_measure=...`
    /// pragma escape hatch.
    pub recursion_no_measure_found: AtomicUsize,

    // -- Per-chokepoint telemetry (item #7 of smt-invocation-load-bearing) -
    /// Per-`SiteKind` invocation counts. Bumped by `require_static_int`
    /// and `probe_in_range` (the new dyn-index chokepoint), tagged by
    /// `SiteKind::short_name()`. The bool tracks whether the chokepoint
    /// reached the SMT layer (i.e. the global `queries_smt_resolved` or
    /// `queries_smt_unknown` counter was bumped while the chokepoint was
    /// active). The two maps together give attribution: which sites are
    /// hot, and which of those actually engage SMT.
    pub chokepoint_invocations: Mutex<HashMap<&'static str, u64>>,
    pub chokepoint_smt_engagements: Mutex<HashMap<&'static str, u64>>,
    pub chokepoint_resolved: Mutex<HashMap<&'static str, u64>>,
}

impl SmtTelemetry {
    /// A fresh, all-zero telemetry instance. Wrapped in `Arc` so multiple
    /// resolver layers can share it.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record an SMT-layer query duration into the histogram.
    pub fn record_smt_duration(&self, dur: Duration) {
        let ns = dur.as_nanos().min(u64::MAX as u128) as u64;
        self.total_time_in_smt_ns.fetch_add(ns, Ordering::Relaxed);
        let bucket = bucket_for_ns(ns);
        self.smt_duration_buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Record a range-layer query duration (no histogram, just total).
    pub fn record_range_duration(&self, dur: Duration) {
        let ns = dur.as_nanos().min(u64::MAX as u128) as u64;
        self.total_time_in_range_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Note the size (statement count) of the formula encoded for one
    /// query. Updates `largest_formula_size` via fetch_max.
    pub fn note_formula_size(&self, size: usize) {
        self.largest_formula_size.fetch_max(size, Ordering::Relaxed);
    }

    /// Snapshot a final cache occupancy.
    pub fn note_cache_size(&self, size: usize) {
        self.cache_size_at_end.store(size, Ordering::Relaxed);
    }

    /// Record one chokepoint invocation, tagged by `SiteKind::short_name()`.
    /// Item #7 of the smt-invocation-load-bearing card.
    pub fn record_chokepoint_invocation(&self, site_name: &'static str) {
        let mut map = self.chokepoint_invocations.lock().unwrap();
        *map.entry(site_name).or_insert(0) += 1;
    }

    /// Record that the chokepoint reached the SMT layer (the delta in
    /// `queries_smt_resolved + queries_smt_unknown` was > 0). Called by
    /// `require_static_int` / `probe_in_range` after they capture before /
    /// after counts around the resolver query.
    pub fn record_chokepoint_smt_engagement(&self, site_name: &'static str) {
        let mut map = self.chokepoint_smt_engagements.lock().unwrap();
        *map.entry(site_name).or_insert(0) += 1;
    }

    /// Record that the chokepoint successfully resolved the value (any
    /// layer — static_val / range / SMT). Bumped when the resolver
    /// returned `Some`.
    pub fn record_chokepoint_resolved(&self, site_name: &'static str) {
        let mut map = self.chokepoint_resolved.lock().unwrap();
        *map.entry(site_name).or_insert(0) += 1;
    }

    /// Human-readable dump for stderr at end of compilation.
    pub fn summary(&self) -> String {
        let total = self.queries_total.load(Ordering::Relaxed);
        let static_hit = self.queries_static_val_hit.load(Ordering::Relaxed);
        let range_hit = self.queries_range_hit.load(Ordering::Relaxed);
        let smt_ok = self.queries_smt_resolved.load(Ordering::Relaxed);
        let smt_unk = self.queries_smt_unknown.load(Ordering::Relaxed);
        let smt_timeout = self.queries_smt_timeout.load(Ordering::Relaxed);
        let cache_hit = self.queries_cache_hit.load(Ordering::Relaxed);
        let disabled = self.queries_skipped_disabled.load(Ordering::Relaxed);
        let oversized = self.queries_skipped_oversized.load(Ordering::Relaxed);
        let smt_ns = self.total_time_in_smt_ns.load(Ordering::Relaxed);
        let range_ns = self.total_time_in_range_ns.load(Ordering::Relaxed);
        let largest = self.largest_formula_size.load(Ordering::Relaxed);
        let cache_at_end = self.cache_size_at_end.load(Ordering::Relaxed);
        let asserts_const = self.assertions_eliminated_const_fold.load(Ordering::Relaxed);
        let asserts_resolver = self.assertions_eliminated_resolver.load(Ordering::Relaxed);
        let asserts_false = self.assertions_provably_false.load(Ordering::Relaxed);
        let rec_static = self.recursion_bound_static_val.load(Ordering::Relaxed);
        let rec_resolver = self.recursion_bound_resolver_proved.load(Ordering::Relaxed);
        let rec_no_measure = self.recursion_no_measure_found.load(Ordering::Relaxed);

        // Cache hit % (over total queries).
        let cache_pct = if total > 0 {
            (cache_hit as f64) * 100.0 / (total as f64)
        } else {
            0.0
        };

        let mut s = String::new();
        s.push_str("SmtTelemetry summary:\n");
        s.push_str(&format!("  queries_total                = {}\n", total));
        s.push_str(&format!("  queries_static_val_hit       = {}\n", static_hit));
        s.push_str(&format!("  queries_range_hit            = {}\n", range_hit));
        s.push_str(&format!("  queries_smt_resolved         = {}\n", smt_ok));
        s.push_str(&format!("  queries_smt_unknown          = {}\n", smt_unk));
        s.push_str(&format!("  queries_smt_timeout          = {}\n", smt_timeout));
        s.push_str(&format!("  queries_cache_hit            = {} ({:.1}%)\n",
                            cache_hit, cache_pct));
        s.push_str(&format!("  queries_skipped_disabled     = {}\n", disabled));
        s.push_str(&format!("  queries_skipped_oversized    = {}\n", oversized));
        s.push_str(&format!("  total_time_in_smt_ms         = {:.1}\n",
                            smt_ns as f64 / 1e6));
        s.push_str(&format!("  total_time_in_range_ms       = {:.1}\n",
                            range_ns as f64 / 1e6));
        s.push_str(&format!("  largest_formula_size         = {}\n", largest));
        s.push_str(&format!("  cache_size_at_end            = {}\n", cache_at_end));
        s.push_str(&format!("  assertions_eliminated_const  = {}\n", asserts_const));
        s.push_str(&format!("  assertions_eliminated_solver = {}\n", asserts_resolver));
        s.push_str(&format!("  assertions_provably_false    = {}\n", asserts_false));
        s.push_str(&format!("  recursion_bound_static_val   = {}\n", rec_static));
        s.push_str(&format!("  recursion_bound_resolver     = {}\n", rec_resolver));
        s.push_str(&format!("  recursion_no_measure_found   = {}\n", rec_no_measure));

        // Histogram.
        s.push_str("  smt_duration_histogram:\n");
        let labels = [
            "       0..=1µs",
            "    1µs..=10µs",
            "   10µs..=100µs",
            "  100µs..=1ms",
            "    1ms..=10ms",
            "   10ms..=100ms",
            "  100ms..=1s",
            "      >1s",
        ];
        for (i, label) in labels.iter().enumerate() {
            let n = self.smt_duration_buckets[i].load(Ordering::Relaxed);
            s.push_str(&format!("    {}: {}\n", label, n));
        }

        // Approximate p95 from histogram (count from the right until 5%
        // crossed). Useful for "is the tail blowing up?".
        let total_smt: usize =
            (0..NUM_DURATION_BUCKETS).map(|i| self.smt_duration_buckets[i].load(Ordering::Relaxed)).sum();
        if total_smt > 0 {
            let cutoff = (total_smt as f64 * 0.05).ceil() as usize;
            let mut acc = 0usize;
            let mut p95_bucket = 0usize;
            for i in (0..NUM_DURATION_BUCKETS).rev() {
                acc += self.smt_duration_buckets[i].load(Ordering::Relaxed);
                if acc >= cutoff {
                    p95_bucket = i;
                    break;
                }
            }
            s.push_str(&format!("  smt_p95_bucket               = {} ({})\n",
                                p95_bucket, labels[p95_bucket].trim()));
        }

        // Per-chokepoint breakdown.
        let invocations = self.chokepoint_invocations.lock().unwrap();
        let engagements = self.chokepoint_smt_engagements.lock().unwrap();
        let resolved = self.chokepoint_resolved.lock().unwrap();
        if !invocations.is_empty() {
            s.push_str("  chokepoint_breakdown:\n");
            let mut keys: Vec<_> = invocations.keys().collect();
            keys.sort();
            for k in keys {
                let n = invocations.get(k).copied().unwrap_or(0);
                let e = engagements.get(k).copied().unwrap_or(0);
                let r = resolved.get(k).copied().unwrap_or(0);
                s.push_str(&format!(
                    "    {:32}invocations={} resolved={} smt_engaged={}\n",
                    k, n, r, e
                ));
            }
        }

        s
    }
}

/// Map a nanosecond duration onto a histogram bucket index.
fn bucket_for_ns(ns: u64) -> usize {
    for (i, edge) in BUCKET_EDGES_NS.iter().enumerate() {
        if ns <= *edge {
            return i;
        }
    }
    NUM_DURATION_BUCKETS - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_edges_match_labels() {
        // Sanity: 0 ns and 1 ns land in bucket 0.
        assert_eq!(bucket_for_ns(0), 0);
        assert_eq!(bucket_for_ns(1), 0);
        assert_eq!(bucket_for_ns(1_000), 0);
        // Just above 1µs goes to bucket 1.
        assert_eq!(bucket_for_ns(1_001), 1);
        // 10ms goes into bucket 4.
        assert_eq!(bucket_for_ns(10_000_000), 4);
        // 1s lands in bucket 6 (the inclusive upper edge).
        assert_eq!(bucket_for_ns(1_000_000_000), 6);
        // >1s goes into the overflow bucket.
        assert_eq!(bucket_for_ns(2_000_000_000), 7);
    }

    #[test]
    fn record_smt_duration_increments_bucket_and_total() {
        let t = SmtTelemetry::new();
        t.record_smt_duration(Duration::from_micros(5));
        t.record_smt_duration(Duration::from_millis(50));
        // Bucket 1 (1..10µs) should have 1; bucket 5 (10..100ms) should
        // have 1.
        assert_eq!(t.smt_duration_buckets[1].load(Ordering::Relaxed), 1);
        assert_eq!(t.smt_duration_buckets[5].load(Ordering::Relaxed), 1);
        // Total SMT time > 50ms.
        assert!(t.total_time_in_smt_ns.load(Ordering::Relaxed) >= 50_000_000);
    }

    #[test]
    fn summary_renders_without_panic() {
        let t = SmtTelemetry::new();
        t.queries_total.fetch_add(10, Ordering::Relaxed);
        t.queries_smt_resolved.fetch_add(7, Ordering::Relaxed);
        t.queries_smt_unknown.fetch_add(3, Ordering::Relaxed);
        let s = t.summary();
        assert!(s.contains("queries_total"));
        assert!(s.contains("smt_duration_histogram"));
    }
}

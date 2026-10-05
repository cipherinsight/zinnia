//! The oracle statistics of the step-1 brief §3.2.6.
//!
//! The concern the whole design has to answer is solver scalability: how
//! many questions a construction asks, how many reach Z3, how large each
//! sliced formula is, and where the time goes. Nothing here is used by the
//! construction; it is what a report reads.

use std::collections::BTreeMap;

use super::questions::Kind;

/// Running minimum, maximum, sum and count of one measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    pub count: u64,
    pub total: u128,
    pub max: u128,
}

impl Tally {
    pub fn add(&mut self, value: u128) {
        self.count += 1;
        self.total += value;
        self.max = self.max.max(value);
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total as f64 / self.count as f64
        }
    }
}

/// Per-kind counts: how a question of this kind was answered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindStats {
    /// Questions asked, memo hits included.
    pub asked: u64,
    /// Questions the memo answered.
    pub memo_hits: u64,
    pub syntactic: u64,
    pub interval: u64,
    pub solver: u64,
}

/// What the solver did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SolverStats {
    pub queries: u64,
    pub sat: u64,
    pub unsat: u64,
    pub unknown: u64,
    pub millis: u128,
    /// Constraints asserted per query, after cone slicing.
    pub cone_constraints: Tally,
    /// Variables in the cone per query.
    pub cone_variables: Tally,
    /// Milliseconds per query.
    pub query_millis: Tally,
    /// Solvers built, and how often a cached one was reused.
    pub solvers_built: u64,
    pub solver_reuse: u64,
    /// The CANFAIL sweep (`sweep.rs`): its queries, their time, and the
    /// constraints its one solver asserted in total — linear in |Φ|, where
    /// the sliced path would have asserted the sum of the cones.
    pub sweep_queries: u64,
    pub sweep_millis: u128,
    pub sweep_asserted: u64,
    /// Constraints Φ holds in total, for the "the cone is smaller than Φ"
    /// comparison.
    pub phi_constraints: u64,
    pub phi_variables: u64,
}

/// Everything [`crate::phase1::Phase1Result`] reports.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub kinds: BTreeMap<&'static str, KindStats>,
    pub solver: SolverStats,
    /// Requirement and implementation nodes of G.
    pub requirements: u64,
    pub implementations: u64,
    /// Implementation nodes the dominance pruning of §4 dropped.
    pub pruned: u64,
    pub roots: u64,
    /// Rows the provider table holds.
    pub rows: u64,
    /// The coordinates of every batch a reading rule composed, and how many
    /// of them were constant (`../README.md` §4 step 4: "ask CONST per
    /// composed coordinate"). A literal of a grid template is constant by
    /// construction and asks nothing; a term coordinate is one CONST
    /// question, memoised on its term id.
    pub batch_coordinates: u64,
    pub constant_coordinates: u64,
    /// Terms phase 1 built over Φ's variables (`../README.md` §1.1): the
    /// arena's node count less phase 0's.
    pub terms_built: u64,
}

impl Stats {
    pub fn kind_mut(&mut self, kind: Kind) -> &mut KindStats {
        self.kinds.entry(kind.as_str()).or_default()
    }

    pub fn kind(&self, kind: Kind) -> KindStats {
        self.kinds.get(kind.as_str()).copied().unwrap_or_default()
    }

    /// Questions asked across every kind.
    pub fn questions(&self) -> u64 {
        self.kinds.values().map(|stats| stats.asked).sum()
    }

    /// The share of batch coordinates that were constant, which is what
    /// decides `READ_DIRECT` against the mux and the ROM.
    pub fn constant_share(&self) -> f64 {
        if self.batch_coordinates == 0 {
            return 1.0;
        }
        self.constant_coordinates as f64 / self.batch_coordinates as f64
    }

    /// The share of questions answered without the solver.
    pub fn without_solver(&self) -> f64 {
        let asked = self.questions();
        if asked == 0 {
            return 1.0;
        }
        let solver: u64 = self.kinds.values().map(|stats| stats.solver).sum();
        1.0 - solver as f64 / asked as f64
    }
}

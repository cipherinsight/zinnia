//! Discharge result cache.
//!
//! Keyed on `(target_ptr, obligation_hash, slice_hash)`, value is the
//! cached [`DischargeResult`]. The cache lives at the discharge layer so
//! repeated chokepoint queries (same target + obligation, even across
//! different invocations) amortise to a hash lookup.
//!
//! Cardinality bound today: per-compilation, one cache instance. Memory
//! is proportional to (number of chokepoints × number of distinct
//! obligations); no eviction yet — a future card adds an LRU policy if
//! profiling shows memory pressure.

use std::collections::HashMap;

use crate::types::StmtId;

/// Outcome of a discharge query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DischargeResult {
    /// Z3 proved the obligation holds.
    Proved,
    /// Z3 returned a counterexample / disproof.
    Disproved,
    /// Z3 timed out / returned `unknown`, or the formula exceeded the
    /// configured budget. Resolver treats this as "the contract layer
    /// abstains; let the next layer try."
    Unknown,
}

/// Lookup key for the cache. The discharge layer constructs this from
/// the (ptr, obligation, def-use slice) tuple; tests construct one
/// directly to exercise cache eq/hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DischargeKey {
    pub target: StmtId,
    pub obligation_hash: u64,
    pub slice_hash: u64,
}

impl DischargeKey {
    pub fn new(target: StmtId, obligation_hash: u64, slice_hash: u64) -> Self {
        Self {
            target,
            obligation_hash,
            slice_hash,
        }
    }
}

/// A simple `HashMap`-backed cache. Single-thread-owned by the
/// discharger; if the discharger ever moves behind an `Arc<Mutex<_>>`
/// the cache moves with it.
#[derive(Debug, Default)]
pub struct DischargeCache {
    inner: HashMap<DischargeKey, DischargeResult>,
}

impl DischargeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &DischargeKey) -> Option<DischargeResult> {
        self.inner.get(key).copied()
    }

    pub fn insert(&mut self, key: DischargeKey, value: DischargeResult) {
        self.inner.insert(key, value);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

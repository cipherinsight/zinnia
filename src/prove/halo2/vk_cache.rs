//! In-process verifying-key cache + telemetry for the halo2-ipa re-keygen
//! verify path (D' direction of
//! `compiler.halo2-cross-process-vk-serialization`).
//!
//! # Why this exists
//!
//! zcash/halo2 v0.3 has no public `VerifyingKey` serde. The chosen
//! cross-process verify path (D') ships the IR inside `ProofArtifact.vk_bytes`
//! and re-runs `keygen_vk` on the verify side to recover the VK. Keygen is
//! the expensive part of the halo2 pipeline (proportional to circuit-row
//! count and constraint-system complexity), so we cache by `ir_hash` to
//! amortize across repeated verifies of the same circuit shape — which is
//! the dominant pattern in (a) the differential fuzzer's prover-pool /
//! single-verifier topology, and (b) on-disk replay of stored artifacts.
//!
//! # Soundness note
//!
//! The cache is keyed on `ir_hash` (BLAKE2b-256 of the canonical IR JSON).
//! A wrong-IR proof presented to verify either:
//!
//! 1. Hashes to an `ir_hash` not in the cache → cache miss → re-keygen
//!    against the (also-wrong) IR → produces a VK that doesn't match the
//!    real prover's VK → Fiat-Shamir transcript diverges at the VK-bind
//!    step inside `verify_proof` → rejects.
//! 2. Hashes to an `ir_hash` already in the cache (collision against an
//!    earlier-verified circuit) → uses the cached VK → if the artifact's
//!    proof was actually produced for the cached VK's circuit, it would
//!    have shipped that circuit's IR, contradicting the assumption →
//!    if it ships a different IR, the artifact's `ir_hash` field is
//!    recomputed *before* `verify_proof` and rejected as `IrHashMismatch`.
//!    BLAKE2b-256 collision resistance is the binding here.
//!
//! So the cache is correctness-preserving under the standard halo2
//! Fiat-Shamir self-binding assumption; see the doc-comment on the verify
//! path in `mod.rs` for the load-bearing invariant.
//!
//! The module also owns a smaller prove-time IPA-parameter cache keyed only by
//! `k`. `ParamsIPA::<EqAffine>::new(k)` is shape/setup data independent of IR,
//! witness, precision, lookup width, and public inputs. Reusing it does not
//! change VK/proof-artifact semantics: `keygen_vk`, `keygen_pk`, and
//! `create_proof` still receive the same parameter value they would have
//! received from a fresh constructor call.
//!
//! Finally, the prove path owns an exact-shape proving-key cache. Its key is
//! the same conservative proof-shape key used for the VK cache:
//! `(ir_hash, k, precision_bits, lookup_bits)`. This is intentionally narrower
//! than the `k`-only IPA params cache because a proving key commits to the
//! circuit's fixed columns and constraint-system shape.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use halo2_proofs::halo2curves::bn256::G1Affine as EqAffine;
use halo2_proofs::plonk::{ProvingKey, VerifyingKey};
use halo2_proofs::poly::commitment::{Params, ParamsProver};
use halo2_proofs::poly::ipa::commitment::ParamsIPA;

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

/// Lock-free counters for the halo2 VK cache. Shaped like
/// `optim::telemetry::SmtTelemetry`: shared via `Arc`, atomics for hot
/// counters, snapshotted at end-of-run for human-readable summary.
///
/// Wired in:
/// - `halo2_vk_cache_telemetry()` returns the process-wide `Arc<Self>` (one
///   per process; the cache itself is process-wide too, so a shared
///   telemetry handle is the right granularity).
/// - `Halo2ProverBackend::verify` bumps `cache_hits` or `cache_misses` on
///   every lookup and adds to `keygen_time_ns_total` on every keygen.
/// - `Halo2ProverBackend` (and Python via a thin getter) can read
///   `summary()` for the differential fuzzer's sweep report and for the
///   end-of-compile diagnostic line, exactly as @Vera wired the
///   `SmtTelemetry` summary into her Phase-1 sink.
#[derive(Debug, Default)]
pub struct Halo2VkCacheTelemetry {
    /// Number of verify calls that hit the cache (no keygen ran).
    pub cache_hits: AtomicUsize,
    /// Number of verify calls that missed the cache and triggered keygen.
    pub cache_misses: AtomicUsize,
    /// Total time spent in `keygen_vk` on the verify side, in nanoseconds.
    /// Divide by `cache_misses` for a per-miss average.
    pub keygen_time_ns_total: AtomicU64,
    /// Current cache occupancy (count of distinct `ir_hash` entries) at the
    /// last snapshot. Refreshed by `record_cache_size` after every insert.
    /// Vera asked for this as a regression-alarm signal: a runaway
    /// `cache_size` against a bounded-grammar fuzzer would indicate
    /// non-determinism in IR export or in `keygen_vk`.
    pub cache_size: AtomicUsize,
}

impl Halo2VkCacheTelemetry {
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self, keygen_duration: Duration) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let ns = keygen_duration.as_nanos().min(u64::MAX as u128) as u64;
        self.keygen_time_ns_total.fetch_add(ns, Ordering::Relaxed);
    }

    pub fn record_cache_size(&self, size: usize) {
        self.cache_size.store(size, Ordering::Relaxed);
    }

    /// Snapshot the counters and render a multi-line summary. Format
    /// intentionally mirrors `SmtTelemetry::summary` so the fuzzer's sweep
    /// reporter can ingest both with the same parser.
    pub fn summary(&self) -> String {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let keygen_ns = self.keygen_time_ns_total.load(Ordering::Relaxed);
        let size = self.cache_size.load(Ordering::Relaxed);

        let hit_pct = if total > 0 {
            (hits as f64) * 100.0 / (total as f64)
        } else {
            0.0
        };
        let keygen_ms_total = keygen_ns as f64 / 1e6;
        let keygen_ms_avg = if misses > 0 {
            keygen_ms_total / (misses as f64)
        } else {
            0.0
        };

        let mut s = String::new();
        s.push_str("Halo2VkCacheTelemetry summary:\n");
        s.push_str(&format!("  halo2_vk_cache_hits          = {}\n", hits));
        s.push_str(&format!("  halo2_vk_cache_misses        = {}\n", misses));
        s.push_str(&format!(
            "  halo2_vk_cache_hit_pct       = {:.1}%\n",
            hit_pct
        ));
        s.push_str(&format!(
            "  halo2_vk_keygen_time_ms      = {:.1} (total) / {:.1} (per miss)\n",
            keygen_ms_total, keygen_ms_avg
        ));
        s.push_str(&format!("  halo2_vk_cache_size          = {}\n", size));
        s
    }

    /// Snapshot of the counters as a `(hits, misses, keygen_ms_total,
    /// cache_size)` tuple. Used by tests and by the Python-side telemetry
    /// dump.
    pub fn snapshot(&self) -> (usize, usize, f64, usize) {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let keygen_ms = self.keygen_time_ns_total.load(Ordering::Relaxed) as f64 / 1e6;
        let size = self.cache_size.load(Ordering::Relaxed);
        (hits, misses, keygen_ms, size)
    }
}

/// Return the process-wide `Halo2VkCacheTelemetry` handle. Lock-free reads
/// and writes; cheap to call on the hot path.
pub fn halo2_vk_cache_telemetry() -> &'static Arc<Halo2VkCacheTelemetry> {
    static TELEMETRY: OnceLock<Arc<Halo2VkCacheTelemetry>> = OnceLock::new();
    TELEMETRY.get_or_init(|| Arc::new(Halo2VkCacheTelemetry::default()))
}

// ---------------------------------------------------------------------------
// VK cache
// ---------------------------------------------------------------------------

/// One cache entry: the IPA params and the verifying key, both wrapped in
/// `Arc` so the cache can hand out cheap clones without giving up exclusive
/// ownership.
type CacheEntry = (Arc<ParamsIPA<EqAffine>>, Arc<VerifyingKey<EqAffine>>);

/// Cache key: `ir_hash` (hex BLAKE2b-256 of canonical IR JSON) + the three
/// `ProvingParams` components that affect circuit shape via the
/// synthesizer's `replay_into_region` / chip-config use:
///
/// - `k` — log2 of circuit rows; controls `ParamsIPA::<EqAffine>::new(k)` and
///   the domain inside the VK.
/// - `precision_bits` — fixed-point precision; threads into the float
///   chips' constraint shape.
/// - `lookup_bits` — lookup-table sizing for range checks; directly shapes
///   the constraint system seen by `keygen_vk`.
///
/// `ir_hash` is intentionally over IR only (matches @Sage's diagnostic-hash
/// framing) so a hash mismatch is unambiguously "wrong IR shipped"; the
/// params live alongside in the cache key so two artifacts with the same
/// IR but different params don't share a cache slot.
type CacheKey = (String, u32, u32, u32);

/// Return the process-wide VK cache (a `Mutex<HashMap>` for write
/// serialization). Reads also go through the mutex, which is fine: each
/// lookup is a single hashmap probe and the cache is small (bounded by
/// the population of distinct circuit shapes, which is bounded by the
/// fuzzer's grammar). If contention ever surfaces in sweep telemetry,
/// swap to `parking_lot::RwLock` or a sharded map without changing the
/// public API.
pub fn vk_cache() -> &'static Mutex<HashMap<CacheKey, CacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Look up the full cache key in the cache; on hit, return the cached
/// `(params, vk)` and bump `cache_hits`. On miss, return `None` — caller
/// is responsible for keygen + insert.
pub fn cache_get(
    ir_hash: &str,
    k: u32,
    precision_bits: u32,
    lookup_bits: u32,
) -> Option<CacheEntry> {
    let guard = vk_cache().lock().expect("vk cache mutex poisoned");
    let entry = guard
        .get(&(ir_hash.to_string(), k, precision_bits, lookup_bits))
        .cloned();
    if entry.is_some() {
        halo2_vk_cache_telemetry().record_cache_hit();
    }
    entry
}

fn insert_entry(
    ir_hash: &str,
    k: u32,
    precision_bits: u32,
    lookup_bits: u32,
    params: Arc<ParamsIPA<EqAffine>>,
    vk: Arc<VerifyingKey<EqAffine>>,
) -> usize {
    let new_size = {
        let mut guard = vk_cache().lock().expect("vk cache mutex poisoned");
        guard.insert(
            (ir_hash.to_string(), k, precision_bits, lookup_bits),
            (params, vk),
        );
        guard.len()
    };
    halo2_vk_cache_telemetry().record_cache_size(new_size);
    new_size
}

/// Insert a freshly-keygen'd `(params, vk)` into the cache after a verify-side
/// miss. Records the keygen duration and updates `cache_size`.
pub fn cache_insert_after_miss(
    ir_hash: &str,
    k: u32,
    precision_bits: u32,
    lookup_bits: u32,
    params: Arc<ParamsIPA<EqAffine>>,
    vk: Arc<VerifyingKey<EqAffine>>,
    keygen_duration: Duration,
) {
    insert_entry(ir_hash, k, precision_bits, lookup_bits, params, vk);
    let telemetry = halo2_vk_cache_telemetry();
    telemetry.record_cache_miss(keygen_duration);
}

/// Warm the cache with a VK already computed during prove(). This should not
/// count as a verify-side cache miss; miss telemetry tracks only verify calls
/// that had to run `keygen_vk`.
pub fn cache_warm(
    ir_hash: &str,
    k: u32,
    precision_bits: u32,
    lookup_bits: u32,
    params: Arc<ParamsIPA<EqAffine>>,
    vk: Arc<VerifyingKey<EqAffine>>,
) {
    insert_entry(ir_hash, k, precision_bits, lookup_bits, params, vk);
}

// ---------------------------------------------------------------------------
// Prove-time exact-shape proving key cache
// ---------------------------------------------------------------------------

/// Process-wide proving-key cache keyed by exact proof shape.
///
/// A `ProvingKey` is artifact-affecting setup data for one circuit shape, so
/// this cache deliberately reuses the full VK-cache key rather than attempting
/// a coarser structural fingerprint. The `ir_hash` is over canonical exported
/// IR JSON; precision and lookup bits are separate because they affect Halo2
/// constraint shape while keeping IR transport diagnostics unambiguous.
pub fn proving_key_cache() -> &'static Mutex<HashMap<CacheKey, Arc<ProvingKey<EqAffine>>>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<ProvingKey<EqAffine>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn proving_key_cache_get(
    ir_hash: &str,
    k: u32,
    precision_bits: u32,
    lookup_bits: u32,
) -> Option<Arc<ProvingKey<EqAffine>>> {
    proving_key_cache()
        .lock()
        .expect("proving key cache mutex poisoned")
        .get(&(ir_hash.to_string(), k, precision_bits, lookup_bits))
        .cloned()
}

pub fn proving_key_cache_insert(
    ir_hash: &str,
    k: u32,
    precision_bits: u32,
    lookup_bits: u32,
    pk: Arc<ProvingKey<EqAffine>>,
) {
    proving_key_cache()
        .lock()
        .expect("proving key cache mutex poisoned")
        .insert((ir_hash.to_string(), k, precision_bits, lookup_bits), pk);
}

// ---------------------------------------------------------------------------
// Prove-time IPA params cache
// ---------------------------------------------------------------------------

/// Process-wide cache for `ParamsIPA::<EqAffine>::new(k)`, keyed only by `k`.
///
/// Halo2 IPA params depend only on the polynomial commitment domain size, so a
/// `k`-keyed cache is semantics-neutral across circuits and witnesses. This is
/// intentionally separate from the VK cache: it does not imply anything about
/// IR identity, precision, lookup width, or artifact format.
///
/// If `ZINNIA_HALO2_IPA_PARAMS_DIR` is set, the first miss for a given `k`
/// attempts to load `halo2-ipa-eqaffine-k{k}.params` from that directory, and
/// writes newly generated params there for future processes. The on-disk bytes
/// use halo2's own `Params::write/read` encoding, and a loaded file is accepted
/// only if its embedded `k` matches the requested key.
pub fn ipa_params_cache() -> &'static Mutex<HashMap<u32, Arc<ParamsIPA<EqAffine>>>> {
    static CACHE: OnceLock<Mutex<HashMap<u32, Arc<ParamsIPA<EqAffine>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ipa_params_persistent_path(k: u32) -> Option<PathBuf> {
    let dir = std::env::var_os("ZINNIA_HALO2_IPA_PARAMS_DIR")?;
    Some(PathBuf::from(dir).join(format!("halo2-ipa-eqaffine-k{}.params", k)))
}

fn read_persistent_ipa_params(k: u32) -> io::Result<Option<ParamsIPA<EqAffine>>> {
    let Some(path) = ipa_params_persistent_path(k) else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let mut file = fs::File::open(&path)?;
    let params = ParamsIPA::<EqAffine>::read(&mut file)?;
    if params.k() != k {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IPA params file {:?} contains k={}, expected k={}",
                path,
                params.k(),
                k
            ),
        ));
    }
    Ok(Some(params))
}

fn write_persistent_ipa_params(k: u32, params: &ParamsIPA<EqAffine>) -> io::Result<()> {
    let Some(path) = ipa_params_persistent_path(k) else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!(
        "params.{}.{}.tmp",
        std::process::id(),
        format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
    ));
    {
        let mut file = fs::File::create(&tmp)?;
        params.write(&mut file)?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

/// Return cached IPA params for `k`, constructing them on the first use.
pub fn ipa_params_get_or_create(k: u32) -> Arc<ParamsIPA<EqAffine>> {
    if let Some(params) = ipa_params_cache()
        .lock()
        .expect("ipa params cache mutex poisoned")
        .get(&k)
        .cloned()
    {
        return params;
    }

    if let Ok(Some(params)) = read_persistent_ipa_params(k) {
        let params = Arc::new(params);
        let mut guard = ipa_params_cache()
            .lock()
            .expect("ipa params cache mutex poisoned");
        return Arc::clone(guard.entry(k).or_insert(params));
    }

    // `Params::new(k)` is the expensive operation this cache exists to avoid.
    // Build it outside the mutex so one cold miss does not block unrelated
    // cached-k lookups in other prover threads.
    let params = Arc::new(ParamsIPA::<EqAffine>::new(k));
    let _ = write_persistent_ipa_params(k, &params);
    let mut guard = ipa_params_cache()
        .lock()
        .expect("ipa params cache mutex poisoned");
    Arc::clone(guard.entry(k).or_insert(params))
}

#[cfg(test)]
pub(crate) fn ipa_params_cache_clear_for_test() {
    ipa_params_cache()
        .lock()
        .expect("ipa params cache mutex poisoned")
        .clear();
}

#[cfg(test)]
pub(crate) fn ipa_params_cache_size_for_test() -> usize {
    ipa_params_cache()
        .lock()
        .expect("ipa params cache mutex poisoned")
        .len()
}

#[cfg(test)]
pub(crate) fn proving_key_cache_clear_for_test() {
    proving_key_cache()
        .lock()
        .expect("proving key cache mutex poisoned")
        .clear();
}

#[cfg(test)]
pub(crate) fn proving_key_cache_size_for_test() -> usize {
    proving_key_cache()
        .lock()
        .expect("proving key cache mutex poisoned")
        .len()
}

#[cfg(test)]
pub(crate) fn cache_clear_for_test() {
    let mut guard = vk_cache().lock().expect("vk cache mutex poisoned");
    guard.clear();
    halo2_vk_cache_telemetry().record_cache_size(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_summary_renders_without_panic_on_empty_state() {
        let t = Halo2VkCacheTelemetry::default();
        let s = t.summary();
        assert!(s.contains("halo2_vk_cache_hits"));
        assert!(s.contains("halo2_vk_cache_misses"));
        assert!(s.contains("halo2_vk_keygen_time_ms"));
        assert!(s.contains("halo2_vk_cache_size"));
    }

    #[test]
    fn telemetry_records_hit_and_miss() {
        let t = Halo2VkCacheTelemetry::default();
        t.record_cache_hit();
        t.record_cache_hit();
        t.record_cache_miss(Duration::from_millis(42));
        let (hits, misses, keygen_ms, _size) = t.snapshot();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
        assert!(keygen_ms >= 42.0);
    }

    #[test]
    fn telemetry_hit_pct_handles_empty_total() {
        let t = Halo2VkCacheTelemetry::default();
        let s = t.summary();
        // No queries → hit% renders as 0.0%, not NaN.
        assert!(s.contains("0.0%"));
    }
}

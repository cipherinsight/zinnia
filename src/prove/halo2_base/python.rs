//! Python bindings for the separated halo2-base steps (see the `keys`
//! module).
//!
//! ```python
//! keys = halo2_base_keygen(compiled_ir, params_json)        # once, no witness
//! keys.save(directory)                                      # params, pk, vk, pinning
//! keys = halo2_base_load_keys(directory, "prover")          # or "verifier"
//! out = json.loads(halo2_base_prove(compiled_ir, keys, inputs_json, externals))
//! ok = json.loads(halo2_base_verify(keys, out["proof"], out["public_values"]))
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::keys::{keygen, keygen_or_next_k, prove_with_keys, verify_with_keys, Halo2BaseKeys, KeyRole, LoadTimings};
use super::{cell_profile, cell_stats, fp_from_hex, fp_to_hex, Halo2BaseProverBackend};
use crate::prove::traits::ProverBackend;
use crate::prove::types::ProvingParams;
use crate::CompiledIR;

/// Proving and verifying keys of one compiled circuit (opaque handle).
#[pyclass(name = "Halo2BaseKeys")]
pub struct PyHalo2BaseKeys {
    inner: Arc<Halo2BaseKeys>,
    /// Seconds spent in `ParamsIPA::new` + `keygen_vk` + `keygen_pk`, for
    /// keys generated in this process.
    keygen_seconds: Option<f64>,
    /// Seconds spent reading the key files, for keys read from a directory.
    load_timings: Option<LoadTimings>,
}

impl PyHalo2BaseKeys {
    fn generated(keys: Halo2BaseKeys, keygen_seconds: f64) -> Self {
        Self { inner: Arc::new(keys), keygen_seconds: Some(keygen_seconds), load_timings: None }
    }

    fn loaded(keys: Halo2BaseKeys, timings: LoadTimings) -> Self {
        Self { inner: Arc::new(keys), keygen_seconds: None, load_timings: Some(timings) }
    }
}

#[pymethods]
impl PyHalo2BaseKeys {
    #[getter]
    fn k(&self) -> u32 {
        self.inner.params().k
    }
    #[getter]
    fn lookup_bits(&self) -> u32 {
        self.inner.params().lookup_bits
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.inner.params().precision_bits
    }
    #[getter]
    fn ir_hash(&self) -> String {
        self.inner.pinning.ir_hash.clone()
    }
    /// Whether these keys can prove (generated, or read for a prover).
    #[getter]
    fn has_proving_key(&self) -> bool {
        self.inner.pk().is_ok()
    }
    #[getter]
    fn keygen_seconds(&self) -> Option<f64> {
        self.keygen_seconds
    }
    /// For keys read from a directory, the seconds spent reading each file:
    /// `{"params": s, "proving_key" | "verifying_key": s}`; `None` otherwise.
    #[getter]
    fn load_seconds(&self) -> Option<HashMap<&'static str, f64>> {
        let timings = self.load_timings?;
        let key = if self.has_proving_key() { "proving_key" } else { "verifying_key" };
        Some(HashMap::from([("params", timings.params_seconds), (key, timings.key_seconds)]))
    }
    /// The verifying key's transcript digest (halo2's `transcript_repr`: a
    /// hash of the domain, the constraint system, the fixed and permutation
    /// commitments), hex: equal digests mean the same circuit.
    #[getter]
    fn vk_digest(&self) -> String {
        use halo2_base::halo2_proofs::halo2curves::ff::PrimeField;
        let repr = self.inner.vk().transcript_repr().to_repr();
        repr.as_ref().iter().rev().map(|b| format!("{b:02x}")).collect()
    }
    /// The verifying key in halo2's raw-bytes encoding.
    fn vk_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.inner.vk_bytes())
    }
    /// Write the keys to `directory` (`params.bin`, `pk.bin`, `vk.bin`,
    /// `keys.json`); needs the proving key.
    fn save(&self, directory: PathBuf) -> PyResult<()> {
        self.inner.save(&directory).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Generate keys for `compiled_ir`. `params_json` (`{"k", "lookup_bits",
/// "precision_bits"}`) defaults to the backend's `estimate_params` (with a
/// retry at the next `k` if the layout fails, see `keygen_or_next_k`).
/// `keygen_seconds` covers key generation, not the estimate.
#[pyfunction]
#[pyo3(signature = (compiled_ir, params_json=None))]
fn halo2_base_keygen(compiled_ir: &CompiledIR, params_json: Option<&str>) -> PyResult<PyHalo2BaseKeys> {
    let params = params_or_estimate(compiled_ir, params_json)?;
    let started = Instant::now();
    let keys = match params_json {
        Some(_) => keygen(&compiled_ir.graph, &params),
        None => keygen_or_next_k(&compiled_ir.graph, &params),
    }
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyHalo2BaseKeys::generated(keys, started.elapsed().as_secs_f64()))
}

/// Read keys written by `Halo2BaseKeys.save`: `role` is `"prover"` (IPA
/// parameters and proving key) or `"verifier"` (IPA parameters and verifying
/// key). The handle's `load_seconds` times each read.
#[pyfunction]
fn halo2_base_load_keys(directory: PathBuf, role: &str) -> PyResult<PyHalo2BaseKeys> {
    let role = match role {
        "prover" => KeyRole::Prover,
        "verifier" => KeyRole::Verifier,
        other => return Err(PyValueError::new_err(format!("role must be 'prover' or 'verifier', not {other:?}"))),
    };
    let (keys, timings) = Halo2BaseKeys::load(&directory, role).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyHalo2BaseKeys::loaded(keys, timings))
}

/// The parameters `estimate_params` chooses (or `params_json`) and the
/// circuit's cell counts at that lookup width, without key generation.
/// Returns JSON `{"k", "lookup_bits", "precision_bits", "advice", "lookup",
/// "fixed", "instances"}`.
#[pyfunction]
#[pyo3(signature = (compiled_ir, params_json=None))]
fn halo2_base_cell_stats(compiled_ir: &CompiledIR, params_json: Option<&str>) -> PyResult<String> {
    let params = params_or_estimate(compiled_ir, params_json)?;
    let cells = cell_stats(&compiled_ir.graph, params.lookup_bits, params.precision_bits)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(serde_json::json!({
        "k": params.k,
        "lookup_bits": params.lookup_bits,
        "precision_bits": params.precision_bits,
        "advice": cells.advice,
        "lookup": cells.lookup,
        "fixed": cells.fixed,
        "instances": cells.instances,
    })
    .to_string())
}

/// Advice and lookup cells per IR statement at `lookup_bits` (default 8):
/// JSON list of `[statement id, advice, lookup]`. A measurement aid.
#[pyfunction]
#[pyo3(signature = (compiled_ir, lookup_bits=8))]
fn halo2_base_cell_profile(compiled_ir: &CompiledIR, lookup_bits: u32) -> PyResult<String> {
    let precision_bits = ProvingParams::default().precision_bits;
    let profile = cell_profile(&compiled_ir.graph, lookup_bits, precision_bits)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    serde_json::to_string(&profile).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn params_or_estimate(compiled_ir: &CompiledIR, params_json: Option<&str>) -> PyResult<ProvingParams> {
    match params_json {
        Some(pj) => serde_json::from_str(pj)
            .map_err(|e| PyValueError::new_err(format!("Params JSON parse error: {}", e))),
        None => Halo2BaseProverBackend
            .estimate_params(&compiled_ir.graph)
            .map_err(|e| PyRuntimeError::new_err(e.to_string())),
    }
}

/// Prove with existing keys. Returns JSON with `proof` (hex),
/// `public_values` (hex field elements, as in proof artifacts) and the
/// seconds of each phase: `parse_inputs_seconds` (reading `inputs_json`),
/// `preprocess_seconds` (the IR evaluation pass that resolves external
/// calls), `witness_seconds` (circuit synthesis), `thread_pool_seconds`
/// (starting the proving thread pool) and `prove_seconds` (`create_proof` +
/// transcript).
#[pyfunction]
fn halo2_base_prove(
    py: Python<'_>,
    compiled_ir: &CompiledIR,
    keys: &PyHalo2BaseKeys,
    inputs_json: &str,
    external_callables: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<String> {
    let (_, ir_hash) = compiled_ir.cached_artifact_inputs().map_err(PyRuntimeError::new_err)?;
    if *ir_hash != keys.inner.pinning.ir_hash {
        return Err(PyValueError::new_err("halo2-base keys were generated for a different circuit"));
    }
    let started = Instant::now();
    let inputs: crate::circuit_input::CircuitInputs = serde_json::from_str(inputs_json)
        .map_err(|e| PyValueError::new_err(format!("Witness JSON parse error: {}", e)))?;
    let parse_inputs_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let callback = crate::prove::preprocess::py_callback::PyExternalCallback::new(py, external_callables);
    let witness =
        crate::prove::preprocess::run_preprocess(&compiled_ir.graph, &inputs, keys.inner.params(), &callback)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let preprocess_seconds = started.elapsed().as_secs_f64();

    let proof = prove_with_keys(&compiled_ir.graph, &witness, &keys.inner)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(serde_json::json!({
        "proof": hex(&proof.proof),
        "public_values": proof.public_values.iter().map(fp_to_hex).collect::<Vec<_>>(),
        "parse_inputs_seconds": parse_inputs_seconds,
        "preprocess_seconds": preprocess_seconds,
        "witness_seconds": proof.witness_seconds,
        "thread_pool_seconds": proof.thread_pool_seconds,
        "prove_seconds": proof.prove_seconds,
        "k": keys.inner.params().k,
    })
    .to_string())
}

/// Verify a proof against the keys' verifying key. Returns JSON
/// `{"valid": bool, "error": str | null, "verify_seconds": float}`;
/// `verify_seconds` covers the transcript and `verify_proof`, not decoding
/// the arguments.
#[pyfunction]
fn halo2_base_verify(keys: &PyHalo2BaseKeys, proof_hex: &str, public_values: Vec<String>) -> PyResult<String> {
    let proof = unhex(proof_hex).map_err(PyValueError::new_err)?;
    let publics = public_values
        .iter()
        .map(|s| fp_from_hex(s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
    let started = Instant::now();
    let result = verify_with_keys(&keys.inner, &proof, &publics);
    let verify_seconds = started.elapsed().as_secs_f64();
    Ok(serde_json::json!({
        "valid": result.valid,
        "error": result.error,
        "verify_seconds": verify_seconds,
    })
    .to_string())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("odd-length hex string".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("invalid hex: {}", e)))
        .collect()
}

/// Register the classes and functions on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHalo2BaseKeys>()?;
    m.add_function(wrap_pyfunction!(halo2_base_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(halo2_base_load_keys, m)?)?;
    m.add_function(wrap_pyfunction!(halo2_base_cell_stats, m)?)?;
    m.add_function(wrap_pyfunction!(halo2_base_cell_profile, m)?)?;
    m.add_function(wrap_pyfunction!(halo2_base_prove, m)?)?;
    m.add_function(wrap_pyfunction!(halo2_base_verify, m)?)?;
    Ok(())
}

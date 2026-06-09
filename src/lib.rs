use pyo3::prelude::*;

pub mod ast;
pub mod backend;
pub mod builder;
pub mod circuit_input;
pub mod error;
pub mod helpers;
pub mod ir;
pub mod ir_ctx;
pub mod ir_defs;
pub mod ir_gen;
pub mod ops;
pub mod optim;
pub mod prove;
pub mod scope;
pub mod types;

use ir::IRGraph;
use ir_gen::{IRGenConfig, IRGenerator};
use optim::{
    AlwaysSatisfiedElimination, AssertSubsumption, AssertedEqualityConstantPropagation,
    ConstantFold, DeadCodeElimination, DoubleNotElimination, DuplicateCodeElimination,
    DynamicNDArrayMemoryLowering, DynamicNDArrayMetaAssertInjection, ExternalCallRemover, IRPass,
    MemoryTraceInjection, PatternMatchOptim, UnreadMemoryEliminator,
};

/// Opaque handle holding a compiled IR graph.
/// Python holds the reference; Rust owns the data.
#[pyclass]
pub struct CompiledIR {
    graph: IRGraph,
    /// Lazily-computed cache of the exported IR statements (`Vec<Value>`)
    /// and its BLAKE2b ir-hash. Both are needed on every prove call (the
    /// artifact carries them so verifiers can re-keygen against the same
    /// IR) and BOTH cost O(N) per call: for output-heavy kernels like
    /// 0387 (40 002 stmts) they were ~200 ms of pre-prove overhead per
    /// invocation. The IR is immutable after compile, so we cache once.
    artifact_cache: std::sync::OnceLock<(Vec<serde_json::Value>, String)>,
}

impl CompiledIR {
    /// Return the cached `(ir_stmts, ir_hash)`, computing on first call.
    /// Both are needed by the halo2-IPA prove path (carried in the proof
    /// artifact + used as the VK cache key).
    pub fn cached_artifact_inputs(&self) -> Result<&(Vec<serde_json::Value>, String), String> {
        let mut error: Option<String> = None;
        let result = self.artifact_cache.get_or_init(|| {
            let stmts = self.graph.export_stmts();
            match crate::prove::halo2::artifact_format::compute_ir_hash(&stmts) {
                Ok(hash) => (stmts, hash),
                Err(e) => {
                    error = Some(e);
                    (Vec::new(), String::new())
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        Ok(result)
    }
}

fn apply_pass(graph: IRGraph, pass_name: &str, mux_threshold: u32) -> Option<IRGraph> {
    Some(match pass_name {
        "ExternalCallRemover" => ExternalCallRemover.exec(graph),
        "DeadCodeElimination" => DeadCodeElimination.exec(graph),
        "DoubleNotElimination" => DoubleNotElimination.exec(graph),
        "AlwaysSatisfiedElimination" => AlwaysSatisfiedElimination.exec(graph),
        "AssertedEqualityConstantPropagation" => AssertedEqualityConstantPropagation.exec(graph),
        "UnreadMemoryEliminator" => UnreadMemoryEliminator.exec(graph),
        "ConstantFold" => ConstantFold.exec(graph),
        "DuplicateCodeElimination" => DuplicateCodeElimination.exec(graph),
        "AssertSubsumption" => AssertSubsumption.exec(graph),
        "DynamicNDArrayMemoryLowering" => {
            DynamicNDArrayMemoryLowering::new(mux_threshold).exec(graph)
        }
        "DynamicNDArrayMetaAssertInjection" => DynamicNDArrayMetaAssertInjection.exec(graph),
        "MemoryTraceInjection" => MemoryTraceInjection.exec(graph),
        "PatternMatchOptim" => PatternMatchOptim.exec(graph),
        _ => return None,
    })
}

fn compile_phase_trace_enabled() -> bool {
    std::env::var("ZINNIA_COMPILE_PHASE_TRACE")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            !matches!(s.as_str(), "0" | "false" | "off" | "no" | "")
        })
        .unwrap_or(false)
}

fn trace_compile_phase(phase: &str, started: std::time::Instant, count: Option<usize>) {
    match count {
        Some(count) => eprintln!(
            "[zinnia-compile-phase] phase={} elapsed_s={:.3} ir_count={}",
            phase,
            started.elapsed().as_secs_f64(),
            count
        ),
        None => eprintln!(
            "[zinnia-compile-phase] phase={} elapsed_s={:.3}",
            phase,
            started.elapsed().as_secs_f64()
        ),
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown Rust panic".to_string()
    }
}

fn run_pass_pipeline(
    graph: IRGraph,
    passes: &[&str],
    mux_threshold: u32,
    trace_phases: bool,
) -> IRGraph {
    // P5: optim passes rebuild the IRGraph from a fresh `IRBuilder`, which
    // discards both the `astgen_telemetry` slot and the active resolver.
    // We snapshot them before the loop and re-attach after each pass so
    // the end-of-compilation telemetry summary still has access. (The
    // resolver itself is per-phase per P1 option (b); we just preserve
    // the *handle* across passes within a single optim phase.)
    let astgen = graph.astgen_telemetry();
    let mut g = graph;
    for pass_name in passes {
        let pass_started = std::time::Instant::now();
        match apply_pass(g, pass_name, mux_threshold) {
            Some(mut result) => {
                if let Some(t) = astgen.as_ref() {
                    result.set_astgen_telemetry(std::sync::Arc::clone(t));
                }
                if trace_phases {
                    trace_compile_phase(pass_name, pass_started, Some(result.stmts.len()));
                }
                g = result;
            }
            None => panic!("Unknown optimization pass in pipeline: {}", pass_name),
        }
    }
    g
}

/// Compile a circuit: generate IR from AST and run all optimization passes.
/// Returns a CompiledIR handle (opaque to Python — no JSON round-trip).
#[pyfunction]
#[pyo3(signature = (ast_json, config_json, chips_json="{}".to_string(), externals_json="{}".to_string()))]
fn compile_circuit(
    ast_json: &str,
    config_json: &str,
    chips_json: String,
    externals_json: String,
) -> PyResult<CompiledIR> {
    // Parse config
    let config: serde_json::Value = serde_json::from_str(config_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Config JSON parse error: {}", e))
    })?;

    let loop_limit = config["loop_limit"].as_u64().unwrap_or(1000) as u32;
    let recursion_limit = config["recursion_limit"].as_u64().unwrap_or(16) as u32;
    // SMT resolver knobs. **Default: enabled** — `LayeredResolver::range_then_smt`,
    // which routes every `require_static_int` query through static_val → range
    // analysis → SMT in that order. P5 round 1 (commit 427a913) re-measured
    // the on-vs-off compile-time delta with a serial sweep and found 0.99×
    // aggregate, 1.05× worst case (within run-to-run noise). The earlier
    // P3 readings of 25× aggregate / 66× worst case (a52fe45) were a
    // process-pool contention artifact, not real SMT cost. The defensive
    // mitigations (100 ms query timeout, 4096-statement formula cap from
    // c75f3fb) bound the future worst case if a consumer reaches the SMT
    // layer with a hard formula. `ZINNIA_SMT_ENABLE=0` (or `smt_enable: false`
    // in config JSON) flips back to `StaticOnlyResolver` byte-for-byte —
    // the safety net.
    let smt_enable_env = std::env::var("ZINNIA_SMT_ENABLE").ok().map(|s| {
        let s = s.trim().to_ascii_lowercase();
        !matches!(s.as_str(), "0" | "false" | "off" | "no")
    });
    let smt_enable = smt_enable_env.unwrap_or_else(|| {
        config
            .get("smt_enable")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    });
    let smt_query_timeout_ms = std::env::var("ZINNIA_SMT_QUERY_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or_else(|| {
            config
                .get("smt_query_timeout_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(100)
        });
    let smt_max_formula_size = std::env::var("ZINNIA_SMT_MAX_FORMULA_SIZE")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or_else(|| {
            config
                .get("smt_max_formula_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(4096)
        });
    // P5: opt-in stderr dump of the SMT-pipeline telemetry summary at the
    // end of compilation. Useful for the worst-case profiling sweep.
    let smt_log_telemetry = std::env::var("ZINNIA_SMT_LOG_TELEMETRY")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            !matches!(s.as_str(), "0" | "false" | "off" | "no" | "")
        })
        .unwrap_or_else(|| {
            config
                .get("smt_log_telemetry")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        });
    let backend = config["backend"].as_str().unwrap_or("halo2");
    let enable_memory_consistency = config["enable_memory_consistency"]
        .as_bool()
        .unwrap_or(false);
    let enable_dynamic_ndarray_meta_assertions = config["enable_dynamic_ndarray_meta_assertions"]
        .as_bool()
        .unwrap_or(false);
    let mux_threshold = config
        .get("mux_threshold")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as u32;

    let optim = &config["optimization"];
    let shortcut_optimization = optim["shortcut_optimization"].as_bool().unwrap_or(true);
    let constant_fold = optim["constant_fold"].as_bool().unwrap_or(true);
    let dead_code_elimination = optim["dead_code_elimination"].as_bool().unwrap_or(true);
    let always_satisfied_elimination = optim["always_satisfied_elimination"]
        .as_bool()
        .unwrap_or(true);
    let duplicate_code_elimination = optim["duplicate_code_elimination"]
        .as_bool()
        .unwrap_or(true);
    let trace_phases = compile_phase_trace_enabled();

    // Parse chips and externals
    let chips: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&chips_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Chips JSON parse error: {}", e))
        })?;
    let externals: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&externals_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Externals JSON parse error: {}", e))
        })?;

    // Generate IR from AST
    let ir_gen_started = std::time::Instant::now();
    let ir_config = IRGenConfig {
        loop_limit,
        recursion_limit,
        smt_enable,
        smt_query_timeout_ms,
        smt_max_formula_size,
        smt_log_telemetry,
    };
    let base_graph = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        IRGenerator::generate_from_json_with_chips(ir_config.clone(), ast_json, &chips, &externals)
    })) {
        Ok(result) => result.map_err(pyo3::exceptions::PyValueError::new_err)?,
        Err(payload) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "IR generation failed: {}",
                panic_payload_to_string(payload)
            )));
        }
    };
    if trace_phases {
        trace_compile_phase("IRGen", ir_gen_started, Some(base_graph.stmts.len()));
    }

    // Build optimization pass pipeline
    let mut passes: Vec<&str> = Vec::new();
    if shortcut_optimization {
        passes.push("PatternMatchOptim");
        passes.push("DoubleNotElimination");
    }
    if constant_fold {
        // AssertedEqualityConstantPropagation must precede ConstantFold so
        // the constants it propagates downstream get folded the rest of the
        // way. Sound under the soundness invariant in its module docs (the
        // assert chain itself is preserved).
        passes.push("AssertedEqualityConstantPropagation");
        passes.push("ConstantFold");
    }
    if dead_code_elimination {
        passes.push("DeadCodeElimination");
    }
    if always_satisfied_elimination {
        passes.push("AlwaysSatisfiedElimination");
    }
    if duplicate_code_elimination {
        passes.push("DuplicateCodeElimination");
    }
    // Collapse `AssertU128Lt(stride_k + base, hi)` chains to the tightest
    // (max-stride) member: see `optim/assert_subsumption.rs`. Runs after
    // DCE so the redundant asserts are visible side-by-side.
    passes.push("AssertSubsumption");
    if backend == "halo2" && enable_memory_consistency {
        passes.push("DynamicNDArrayMemoryLowering");
        passes.push("DynamicNDArrayMetaAssertInjection");
        passes.push("MemoryTraceInjection");
    } else if enable_dynamic_ndarray_meta_assertions {
        passes.push("DynamicNDArrayMetaAssertInjection");
    }

    let optimized_graph = run_pass_pipeline(base_graph, &passes, mux_threshold, trace_phases);

    // P5: end-of-compilation telemetry dump (when enabled). We surface
    // both the AST→IR phase and the optim-phase summaries since each
    // phase has its own resolver / cache (P1 option (b)). The optim
    // phase resolver is rebuilt by every IRPass (each pass calls
    // `builder.export_ir_graph()` which initialises a fresh
    // StaticOnlyResolver), so by the time we reach this point the
    // optim-phase resolver_telemetry will most often be `None`. The
    // AST→IR snapshot is preserved across optim passes via
    // `IRGraph::astgen_telemetry`.
    if smt_log_telemetry {
        eprintln!(
            "[zinnia-smt] log_enabled=true smt_enable={} timeout_ms={}",
            smt_enable, smt_query_timeout_ms
        );
        match optimized_graph.astgen_telemetry() {
            Some(t) => eprintln!("[zinnia-smt] AST→IR phase {}", t.summary()),
            None => eprintln!("[zinnia-smt] AST→IR phase: no telemetry handle"),
        }
        match optimized_graph.resolver_telemetry() {
            Some(t) => {
                t.note_cache_size(0);
                eprintln!("[zinnia-smt] optim phase {}", t.summary());
            }
            None => eprintln!("[zinnia-smt] optim phase: no telemetry handle (default)"),
        }
    }

    Ok(CompiledIR {
        graph: optimized_graph,
        artifact_cache: std::sync::OnceLock::new(),
    })
}

/// Execute a compiled circuit: preprocess (resolve externals) then prove.
/// Takes a CompiledIR handle directly — no JSON deserialization.
#[pyfunction]
#[pyo3(signature = (compiled_ir, inputs_json, external_callables, backend="mock", params_json=None))]
fn prove_circuit(
    py: Python<'_>,
    compiled_ir: &CompiledIR,
    inputs_json: &str,
    external_callables: &Bound<'_, pyo3::types::PyDict>,
    backend: &str,
    params_json: Option<&str>,
) -> PyResult<String> {
    let ir_graph = &compiled_ir.graph;

    // Parse structured witness
    let witness_input: circuit_input::CircuitInputs =
        serde_json::from_str(inputs_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Witness JSON parse error: {}", e))
        })?;

    // Create prover backend + params
    let prover = prove::create_prover_backend(backend)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let params = if let Some(pj) = params_json {
        serde_json::from_str(pj).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Params JSON parse error: {}", e))
        })?
    } else {
        prover
            .estimate_params(ir_graph)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    };

    // Preprocess: resolve external calls, build resolved witness
    let py_callback =
        prove::preprocess::py_callback::PyExternalCallback::new(py, external_callables);
    let resolved_witness =
        prove::preprocess::run_preprocess(ir_graph, &witness_input, &params, &py_callback)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Pre-compute (or fetch cached) artifact inputs so the prove path
    // doesn't re-serialize / re-hash the IR on every call.
    let cached = compiled_ir
        .cached_artifact_inputs()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // Prove
    let artifact = prover
        .prove_with_cached_artifact(ir_graph, &resolved_witness, &params, &cached.0, &cached.1)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    serde_json::to_string(&artifact).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e))
    })
}

/// Verify a ZK proof.
/// Returns a JSON-serialized VerifyResult.
#[pyfunction]
fn verify_proof_artifact(proof_artifact_json: &str) -> PyResult<String> {
    let artifact: prove::ProofArtifact = serde_json::from_str(proof_artifact_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;

    let backend = prove::create_prover_backend(&artifact.backend)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let result = backend
        .verify(&artifact)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    serde_json::to_string(&result).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e))
    })
}

/// On-demand IR serialization: CompiledIR → JSON string (for debugging/persistence).
#[pyfunction]
fn export_ir_json(compiled_ir: &CompiledIR) -> PyResult<String> {
    let exported = compiled_ir.graph.export_stmts();
    serde_json::to_string(&exported).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e))
    })
}

/// Deserialize IR from JSON string → CompiledIR handle (for load-from-disk).
#[pyfunction]
fn import_ir_json(ir_json: &str) -> PyResult<CompiledIR> {
    let data: Vec<serde_json::Value> = serde_json::from_str(ir_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    let graph = IRGraph::import_stmts(&data).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(CompiledIR {
        graph,
        artifact_cache: std::sync::OnceLock::new(),
    })
}

/// Compute the Poseidon hash of a list of integers using the Rust kernel.
/// Returns the hash as a hex string of the full field element bytes (little-endian).
/// This ensures Python-side hash computation matches the prover exactly.
#[pyfunction]
fn poseidon_hash(values: Vec<i64>) -> String {
    use halo2_proofs::halo2curves::bn256::Fr as Fp;
    use halo2_proofs::halo2curves::ff::PrimeField;
    let fps: Vec<Fp> = values
        .iter()
        .map(|v| prove::kernel::i64_to_fp(*v))
        .collect();
    let result = prove::kernel::fp_poseidon(&fps);
    let bytes = result.to_repr();
    bytes
        .as_ref()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Read-and-reset the prove-layer counters. Returns a tuple
/// `(call_count, proved, disproved, unknown, z3_invocations, fastpath_hits)`.
/// Used by the acceptance/rejection sweep to attribute compile cost.
#[pyfunction]
fn get_and_reset_prove_counters() -> (u64, u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering;
    let v = (
        optim::prove::PROVE_CALL_COUNT.load(Ordering::Relaxed),
        optim::prove::PROVE_PROVED_COUNT.load(Ordering::Relaxed),
        optim::prove::PROVE_DISPROVED_COUNT.load(Ordering::Relaxed),
        optim::prove::PROVE_UNKNOWN_COUNT.load(Ordering::Relaxed),
        optim::prove::PROVE_Z3_INVOCATIONS.load(Ordering::Relaxed),
        optim::prove::PROVE_FASTPATH_HITS.load(Ordering::Relaxed),
    );
    optim::prove::reset_prove_counters();
    v
}

/// The Python module definition for zinnia._zinnia_core
#[pymodule]
fn _zinnia_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledIR>()?;
    m.add_function(wrap_pyfunction!(compile_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(prove_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(export_ir_json, m)?)?;
    m.add_function(wrap_pyfunction!(import_ir_json, m)?)?;
    m.add_function(wrap_pyfunction!(poseidon_hash, m)?)?;
    m.add_function(wrap_pyfunction!(get_and_reset_prove_counters, m)?)?;
    Ok(())
}

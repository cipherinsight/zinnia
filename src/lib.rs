use pyo3::prelude::*;

pub mod ast;
pub mod backend;
pub mod builder;
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
    AlwaysSatisfiedElimination, ConstantFold, DeadCodeElimination, DoubleNotElimination,
    DuplicateCodeElimination, DynamicNDArrayMemoryLowering, DynamicNDArrayMetaAssertInjection,
    ExternalCallRemover, IRPass, MemoryTraceInjection, PatternMatchOptim,
};

/// Opaque handle holding a compiled IR graph.
/// Python holds the reference; Rust owns the data.
#[pyclass]
pub struct CompiledIR {
    graph: IRGraph,
}

fn apply_pass(graph: IRGraph, pass_name: &str, mux_threshold: u32) -> Option<IRGraph> {
    Some(match pass_name {
        "ExternalCallRemover" => ExternalCallRemover.exec(graph),
        "DeadCodeElimination" => DeadCodeElimination.exec(graph),
        "DoubleNotElimination" => DoubleNotElimination.exec(graph),
        "AlwaysSatisfiedElimination" => AlwaysSatisfiedElimination.exec(graph),
        "ConstantFold" => ConstantFold.exec(graph),
        "DuplicateCodeElimination" => DuplicateCodeElimination.exec(graph),
        "DynamicNDArrayMemoryLowering" => DynamicNDArrayMemoryLowering::new(mux_threshold).exec(graph),
        "DynamicNDArrayMetaAssertInjection" => DynamicNDArrayMetaAssertInjection.exec(graph),
        "MemoryTraceInjection" => MemoryTraceInjection.exec(graph),
        "PatternMatchOptim" => PatternMatchOptim.exec(graph),
        _ => return None,
    })
}

fn run_pass_pipeline(graph: IRGraph, passes: &[&str], mux_threshold: u32) -> IRGraph {
    let mut g = graph;
    for pass_name in passes {
        match apply_pass(g, pass_name, mux_threshold) {
            Some(result) => g = result,
            None => panic!("Unknown optimization pass in pipeline: {}", pass_name),
        }
    }
    g
}

/// Compile a circuit: generate IR from AST and run all optimization passes.
/// Returns a CompiledIR handle (opaque to Python — no JSON round-trip).
#[pyfunction]
#[pyo3(signature = (ast_json, config_json, chips_json="{}".to_string(), externals_json="{}".to_string()))]
fn compile_circuit(ast_json: &str, config_json: &str, chips_json: String, externals_json: String) -> PyResult<CompiledIR> {
    // Parse config
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Config JSON parse error: {}", e)))?;

    let loop_limit = config["loop_limit"].as_u64().unwrap_or(1000) as u32;
    let recursion_limit = config["recursion_limit"].as_u64().unwrap_or(100) as u32;
    let backend = config["backend"].as_str().unwrap_or("halo2");
    let enable_memory_consistency = config["enable_memory_consistency"].as_bool().unwrap_or(false);
    let mux_threshold = config.get("mux_threshold")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as u32;

    let optim = &config["optimization"];
    let shortcut_optimization = optim["shortcut_optimization"].as_bool().unwrap_or(true);
    let constant_fold = optim["constant_fold"].as_bool().unwrap_or(true);
    let dead_code_elimination = optim["dead_code_elimination"].as_bool().unwrap_or(true);
    let always_satisfied_elimination = optim["always_satisfied_elimination"].as_bool().unwrap_or(true);
    let duplicate_code_elimination = optim["duplicate_code_elimination"].as_bool().unwrap_or(true);

    // Parse chips and externals
    let chips: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&chips_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Chips JSON parse error: {}", e)))?;
    let externals: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&externals_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Externals JSON parse error: {}", e)))?;

    // Generate IR from AST
    let ir_config = IRGenConfig {
        loop_limit,
        recursion_limit,
    };
    let base_graph = IRGenerator::generate_from_json_with_chips(ir_config.clone(), ast_json, &chips, &externals)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Build optimization pass pipeline
    let mut passes: Vec<&str> = Vec::new();
    if backend == "halo2" && enable_memory_consistency {
        passes.push("DynamicNDArrayMemoryLowering");
        passes.push("DynamicNDArrayMetaAssertInjection");
        passes.push("MemoryTraceInjection");
    }
    if shortcut_optimization {
        passes.push("PatternMatchOptim");
        passes.push("DoubleNotElimination");
    }
    if constant_fold {
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

    let optimized_graph = run_pass_pipeline(base_graph, &passes, mux_threshold);

    Ok(CompiledIR { graph: optimized_graph })
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

    // Parse initial witness
    let witness: prove::WitnessInput = serde_json::from_str(inputs_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Witness JSON parse error: {}", e)))?;

    // Create prover backend + params
    let prover = prove::create_prover_backend(backend)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let params = if let Some(pj) = params_json {
        serde_json::from_str(pj)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Params JSON parse error: {}", e)))?
    } else {
        prover.estimate_params(ir_graph)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    };

    // Preprocess: resolve external calls
    let py_callback = prove::preprocess::py_callback::PyExternalCallback::new(py, external_callables);
    let enriched_witness = prove::preprocess::run_preprocess(
        ir_graph, &witness, &params, &py_callback,
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Prove
    let artifact = prover.prove(ir_graph, &enriched_witness, &params)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    serde_json::to_string(&artifact)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Verify a ZK proof.
/// Returns a JSON-serialized VerifyResult.
#[pyfunction]
fn verify_proof_artifact(proof_artifact_json: &str) -> PyResult<String> {
    let artifact: prove::ProofArtifact = serde_json::from_str(proof_artifact_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;

    let backend = prove::create_prover_backend(&artifact.backend)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let result = backend.verify(&artifact)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// On-demand IR serialization: CompiledIR → JSON string (for debugging/persistence).
#[pyfunction]
fn export_ir_json(compiled_ir: &CompiledIR) -> PyResult<String> {
    let exported = compiled_ir.graph.export_stmts();
    serde_json::to_string(&exported)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Deserialize IR from JSON string → CompiledIR handle (for load-from-disk).
#[pyfunction]
fn import_ir_json(ir_json: &str) -> PyResult<CompiledIR> {
    let data: Vec<serde_json::Value> = serde_json::from_str(ir_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    let graph = IRGraph::import_stmts(&data)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(CompiledIR { graph })
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
    Ok(())
}

use pyo3::prelude::*;

pub mod ast;
pub mod backend;
pub mod builder;
pub mod dyn_ndarray;
pub mod error;
pub mod ir;
pub mod ir_ctx;
pub mod ir_defs;
pub mod ir_gen;
pub mod mock_exec;
pub mod ops;
pub mod optim;
pub mod scope;
pub mod types;

use ir::IRGraph;
use ir_gen::{IRGenConfig, IRGenerator};
use optim::{
    AlwaysSatisfiedElimination, ConstantFold, DeadCodeElimination, DoubleNotElimination,
    DuplicateCodeElimination, DynamicNDArrayMemoryLowering, DynamicNDArrayMetaAssertInjection,
    ExternalCallRemover, IRPass, MemoryTraceInjection, PatternMatchOptim,
};
use types::{DTDescriptorDict, ZinniaType};

/// A smoke-test function to verify the PyO3 bridge is working.
#[pyfunction]
fn hello() -> String {
    "Hello from zinnia_core (Rust)!".to_string()
}

/// Returns the version of the Rust core library.
#[pyfunction]
fn core_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Export an IR statement dict list from Rust, for interop testing.
#[pyfunction]
fn round_trip_ir_stmts(json_str: &str) -> PyResult<String> {
    let data: Vec<serde_json::Value> = serde_json::from_str(json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    let graph = IRGraph::import_stmts(&data)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let exported = graph.export_stmts();
    serde_json::to_string(&exported)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Round-trip a ZinniaType through DTDescriptor dict format.
#[pyfunction]
fn round_trip_dt_descriptor(json_str: &str) -> PyResult<String> {
    let dict: DTDescriptorDict = serde_json::from_str(json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    let zinnia_type = ZinniaType::from_dt_dict(&dict)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let exported = zinnia_type.to_dt_dict();
    serde_json::to_string(&exported)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Run an optimization pass on an IR graph (JSON in, JSON out).
#[pyfunction]
#[pyo3(signature = (pass_name, ir_stmts_json, mux_threshold=100))]
fn run_optimization_pass(
    pass_name: &str,
    ir_stmts_json: &str,
    mux_threshold: u32,
) -> PyResult<String> {
    let data: Vec<serde_json::Value> = serde_json::from_str(ir_stmts_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    let graph = IRGraph::import_stmts(&data)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let result_graph = match pass_name {
        "ExternalCallRemover" => ExternalCallRemover.exec(graph),
        "DeadCodeElimination" => DeadCodeElimination.exec(graph),
        "DoubleNotElimination" => DoubleNotElimination.exec(graph),
        "AlwaysSatisfiedElimination" => AlwaysSatisfiedElimination.exec(graph),
        "ConstantFold" => ConstantFold.exec(graph),
        "DuplicateCodeElimination" => DuplicateCodeElimination.exec(graph),
        "DynamicNDArrayMemoryLowering" => {
            DynamicNDArrayMemoryLowering::new(mux_threshold).exec(graph)
        }
        "DynamicNDArrayMetaAssertInjection" => DynamicNDArrayMetaAssertInjection.exec(graph),
        "MemoryTraceInjection" => MemoryTraceInjection.exec(graph),
        "PatternMatchOptim" => PatternMatchOptim.exec(graph),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown optimization pass: {}", pass_name
            )));
        }
    };

    let exported = result_graph.export_stmts();
    serde_json::to_string(&exported)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Generate IR from an AST JSON string. Returns IR statements as JSON.
#[pyfunction]
#[pyo3(signature = (ast_json, loop_limit=256, recursion_limit=16))]
fn generate_ir(ast_json: &str, loop_limit: u32, recursion_limit: u32) -> PyResult<String> {
    let config = IRGenConfig {
        loop_limit,
        recursion_limit,
    };
    let graph = IRGenerator::generate_from_json(config, ast_json)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let exported = graph.export_stmts();
    serde_json::to_string(&exported)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

fn apply_pass(graph: IRGraph, pass_name: &str, mux_threshold: u32) -> IRGraph {
    match pass_name {
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
        _ => graph,
    }
}

fn run_pass_pipeline(graph: IRGraph, passes: &[&str], mux_threshold: u32) -> IRGraph {
    let mut g = graph;
    for pass_name in passes {
        g = apply_pass(g, pass_name, mux_threshold);
    }
    g
}

/// Compile a circuit: generate IR from AST and run all optimization passes in one call.
/// Takes AST JSON, config JSON, chips JSON, and externals JSON.
/// Returns result JSON with zk_program_irs and preprocess_irs.
#[pyfunction]
#[pyo3(signature = (ast_json, config_json, chips_json="{}".to_string(), externals_json="{}".to_string()))]
fn compile_circuit(ast_json: &str, config_json: &str, chips_json: String, externals_json: String) -> PyResult<String> {
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

    // Build pass pipeline for zk_program
    let mut zk_passes: Vec<&str> = vec!["ExternalCallRemover"];
    if backend == "halo2" && enable_memory_consistency {
        zk_passes.push("DynamicNDArrayMemoryLowering");
        zk_passes.push("DynamicNDArrayMetaAssertInjection");
    }
    if shortcut_optimization {
        zk_passes.push("PatternMatchOptim");
        zk_passes.push("DoubleNotElimination");
    }
    if constant_fold {
        zk_passes.push("ConstantFold");
    }
    if dead_code_elimination {
        zk_passes.push("DeadCodeElimination");
    }
    if always_satisfied_elimination {
        zk_passes.push("AlwaysSatisfiedElimination");
    }
    if duplicate_code_elimination {
        zk_passes.push("DuplicateCodeElimination");
    }

    // Build pass pipeline for preprocess
    let mut preprocess_passes: Vec<&str> = Vec::new();
    if backend == "halo2" && enable_memory_consistency {
        preprocess_passes.push("MemoryTraceInjection");
    }
    if shortcut_optimization {
        preprocess_passes.push("PatternMatchOptim");
        preprocess_passes.push("DoubleNotElimination");
    }
    if constant_fold {
        preprocess_passes.push("ConstantFold");
    }
    if dead_code_elimination {
        preprocess_passes.push("DeadCodeElimination");
    }
    if always_satisfied_elimination {
        preprocess_passes.push("AlwaysSatisfiedElimination");
    }
    if duplicate_code_elimination {
        preprocess_passes.push("DuplicateCodeElimination");
    }

    // Run zk_program pipeline
    let zk_graph = run_pass_pipeline(base_graph, &zk_passes, mux_threshold);

    // Re-generate IR for the preprocess pipeline (IRGraph is not Clone)
    let preprocess_base = IRGenerator::generate_from_json_with_chips(ir_config, ast_json, &chips, &externals)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let preprocess_graph = run_pass_pipeline(preprocess_base, &preprocess_passes, mux_threshold);

    // Serialize results
    let zk_stmts = zk_graph.export_stmts();
    let preprocess_stmts = preprocess_graph.export_stmts();

    let result = serde_json::json!({
        "zk_program_irs": zk_stmts,
        "preprocess_irs": preprocess_stmts,
    });

    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Mock-execute a compiled circuit with concrete inputs.
/// Evaluates preprocess IR (with external callbacks) then main IR.
#[pyfunction]
#[pyo3(signature = (zk_program_ir_json, preprocess_ir_json, inputs_json, external_callables))]
fn mock_execute(
    py: Python<'_>,
    zk_program_ir_json: &str,
    preprocess_ir_json: &str,
    inputs_json: &str,
    external_callables: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<String> {
    mock_exec::run_mock_execute(py, zk_program_ir_json, preprocess_ir_json, inputs_json, external_callables)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// The Python module definition for zinnia._zinnia_core
#[pymodule]
fn _zinnia_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(round_trip_ir_stmts, m)?)?;
    m.add_function(wrap_pyfunction!(round_trip_dt_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimization_pass, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(mock_execute, m)?)?;
    Ok(())
}

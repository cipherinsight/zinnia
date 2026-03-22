mod always_satisfied_elimination;
mod constant_fold;
mod dead_code_elimination;
mod double_not_elimination;
mod duplicate_code_elimination;
mod dyn_ndarray_memory_lowering;
mod dyn_ndarray_meta_assert;
mod external_call_remover;
mod memory_trace_injection;
mod pattern_match_optim;

#[cfg(test)]
mod tests;

pub use always_satisfied_elimination::AlwaysSatisfiedElimination;
pub use constant_fold::ConstantFold;
pub use dead_code_elimination::DeadCodeElimination;
pub use double_not_elimination::DoubleNotElimination;
pub use duplicate_code_elimination::DuplicateCodeElimination;
pub use dyn_ndarray_memory_lowering::DynamicNDArrayMemoryLowering;
pub use dyn_ndarray_meta_assert::DynamicNDArrayMetaAssertInjection;
pub use external_call_remover::ExternalCallRemover;
pub use memory_trace_injection::MemoryTraceInjection;
pub use pattern_match_optim::PatternMatchOptim;

use crate::ir::IRGraph;

/// Trait matching Python `AbstractIRPass.exec(ir_graph) -> IRGraph`.
pub trait IRPass {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph;
}

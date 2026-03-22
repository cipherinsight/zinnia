mod passes;

pub use passes::*;

use crate::ir::IRGraph;

/// Trait matching Python `AbstractIRPass.exec(ir_graph) -> IRGraph`.
pub trait IRPass {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph;
}

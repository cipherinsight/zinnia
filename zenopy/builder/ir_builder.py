from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.ir.ir_ctx import IRContext
from zenopy.ir.ir_graph import IRGraph, IRGraphMetadata


class IRBuilder(AbsIRBuilderInterface):
    def __init__(self, ir_ctx: IRContext | None = None) -> None:
        super().__init__()
        self.stmts = []
        self._next_id = 0
        self.ir_ctx = ir_ctx

    def export_ir_graph(self) -> IRGraph:
        return IRGraph(self.stmts, IRGraphMetadata())

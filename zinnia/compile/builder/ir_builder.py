from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.ir.ir_graph import IRGraph


class IRBuilder(IRBuilderInterface):
    def __init__(self) -> None:
        super().__init__()
        self.stmts = []
        self._next_id = 0

    def export_ir_graph(self) -> IRGraph:
        return IRGraph(self.stmts)

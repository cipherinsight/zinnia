from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.compile.ir_graph import IRGraph


class IRBuilder(AbsIRBuilderInterface):
    def __init__(self) -> None:
        super().__init__()
        self.stmts = []
        self._next_id = 0

    def export_ir_graph(self) -> IRGraph:
        return IRGraph(self.stmts)

from pyzk.ir.ir_graph import IRGraph


class AbstractIRPass:
    def __init__(self):
        pass

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ...

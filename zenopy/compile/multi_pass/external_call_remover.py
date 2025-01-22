from zenopy.compile.ir_graph import IRGraph
from zenopy.compile.multi_pass.abstract_pass import AbstractIRPass
from zenopy.opdef.ir_op.ir_export_external_f import ExportExternalFIR
from zenopy.opdef.ir_op.ir_export_external_i import ExportExternalIIR
from zenopy.opdef.ir_op.ir_invoke_external import InvokeExternalIR


class ExternalCallRemoverIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        stmts = ir_graph.export_stmts()
        to_be_eliminated = []
        for i, stmt in enumerate(stmts):
            if isinstance(stmt.operator, InvokeExternalIR):
                to_be_eliminated.append(stmt.stmt_id)
            elif isinstance(stmt.operator, ExportExternalIIR):
                to_be_eliminated.append(stmt.stmt_id)
            elif isinstance(stmt.operator, ExportExternalFIR):
                to_be_eliminated.append(stmt.stmt_id)
        ir_graph.remove_stmt_bunch(to_be_eliminated)
        return ir_graph

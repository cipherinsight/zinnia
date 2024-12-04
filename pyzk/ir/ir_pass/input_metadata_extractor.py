from pyzk.ir.ir_graph import IRGraph
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.opdef.nocls.op_input import InputOp
from pyzk.util.prog_meta_data import ProgramMetadata, ProgramInputMetadata


class InputMetadataExtractorIRPass(AbstractIRPass):
    def __init__(self, prog_meta_data: ProgramMetadata):
        super().__init__()
        self.prog_meta_data = prog_meta_data

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        stmts = ir_graph.export_stmts()
        inputs = []
        for stmt in stmts:
            if isinstance(stmt.operator, InputOp):
                assert stmt.annotation is not None
                inputs.append(ProgramInputMetadata(stmt.annotation.typename, stmt.annotation.shape, stmt.annotation.public))
        self.prog_meta_data.set_program_inputs(inputs)
        return ir_graph

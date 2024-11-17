from typing import Dict

from pyzk.ir.ir_builder import IRBuilder, IRGraph
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.util.prog_meta_data import ProgramMetadata
from pyzk.inference.op_flattener import OperatorFlattener, OperatorFlattenInfo
from pyzk.util.op_name import OpName


class NDArrayFlattenerIRPass(AbstractIRPass):
    def __init__(self, prog_meta_data: ProgramMetadata):
        super().__init__()
        self.prog_meta_data = prog_meta_data

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_builder = IRBuilder()
        stmts = ir_graph.export_stmts()
        flattener = OperatorFlattener(ir_builder, self.prog_meta_data)
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        flatten_info: Dict[int, OperatorFlattenInfo] = {}
        for stmt in stmts:
            if stmt.op == OpName.Special.INPUT:
                flatten_info[stmt.stmt_id] = flattener.flatten_input(stmt)
        for stmt in topological_order:
            if stmt.op == OpName.Special.INPUT:
                continue
            referring_tos = in_links[stmt.stmt_id]
            arg_infos = []
            args = []
            for referring_to in referring_tos:
                args.append(ir_graph.retrieve_stmt_with_id(referring_to))
                arg_infos.append(flatten_info[referring_to])
            flatten_info[stmt.stmt_id] = flattener.flatten(stmt, args, arg_infos)

        ir_graph = ir_builder.export_ir_graph()
        ir_graph.metadata.ndarray_flattened = True
        return ir_graph

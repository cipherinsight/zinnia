import copy

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.builder.value import IntegerValue
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.multi_pass.abstract_pass import AbstractIRPass
from zinnia.opdef.ir_op.ir_assert import AssertIR


class AlwaysSatisfiedEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        builder = IRBuilderImpl()
        in_links, out_links = ir_graph.get_io_links()
        topological_order = ir_graph.get_topological_order(False)
        to_be_eliminated = []
        values_lookup = {}
        for stmt in topological_order:
            referring_tos = in_links[stmt.stmt_id]
            arg_values = [None for _ in referring_tos]
            for i, referring_to in enumerate(referring_tos):
                arg_values[i] = values_lookup[referring_to]
            values_lookup[stmt.stmt_id] = builder.create_ir(stmt.operator, arg_values, {}, None)
        for stmt in topological_order:
            if isinstance(stmt.operator, AssertIR):
                value: IntegerValue = values_lookup[in_links[stmt.stmt_id][0]]
                if value.val() is not None and value.val() != 0:
                    to_be_eliminated.append(stmt.stmt_id)
        ir_graph.remove_stmt_bunch(to_be_eliminated)
        return ir_graph


import copy
from typing import List

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass
from zinnia.compile.triplet import Value, BooleanValue
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.ir_def.defs.ir_logical_not import LogicalNotIR


class DoubleNotEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()
        self.lookup_not_original = {}

    def optimize_ir(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        if isinstance(ir_instance, LogicalNotIR):
            operand = ir_args[0]
            assert isinstance(operand, BooleanValue)
            if operand.ptr() in self.lookup_not_original:
                # This is a double negation, we can eliminate it
                return self.lookup_not_original[operand.ptr()]
            # Otherwise, we create a new logical not IR statement
            new_ir_stmt = ir_builder.ir_logical_not(operand)
            self.lookup_not_original[new_ir_stmt.ptr()] = operand
            return new_ir_stmt
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        self.lookup_not_original = {}
        ir_graph = copy.copy(ir_graph)
        ir_builder = IRBuilderImpl()
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        value_lookup_by_ptr = {}
        for stmt in topological_order:
            ir_args: List[Value] = [None for _ in in_links[stmt.stmt_id]]
            for i, old_ptr in enumerate(in_links[stmt.stmt_id]):
                ir_args[i] = value_lookup_by_ptr[old_ptr]
            value_lookup_by_ptr[stmt.stmt_id] = self.optimize_ir(ir_builder, stmt.ir_instance, ir_args)
        return ir_builder.export_ir_graph()

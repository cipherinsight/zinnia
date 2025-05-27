import copy
from typing import List

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.triplet import IntegerValue, FloatValue, Value, BooleanValue
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass


class ConstantFoldIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        ir_builder = IRBuilderImpl()
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        value_lookup_by_ptr = {}
        constant_int_ir, constant_float_ir = {}, {}
        constant_true, constant_false = ir_builder.ir_constant_bool(True), ir_builder.ir_constant_bool(False)
        for stmt in topological_order:
            ir_args: List[Value] = [None for _ in in_links[stmt.stmt_id]]
            for i, old_ptr in enumerate(in_links[stmt.stmt_id]):
                value = ir_args[i] = value_lookup_by_ptr[old_ptr]
                if isinstance(value, BooleanValue) and value.val() is not None:
                    if value.val():
                        ir_args[i] = constant_true
                    else:
                        ir_args[i] = constant_false
                elif isinstance(value, IntegerValue) and value.val() is not None:
                    if constant_int_ir.get(value.val(), None) is None:
                        constant_int_ir[value.val()] = ir_builder.ir_constant_int(value.val())
                    ir_args[i] = constant_int_ir[value.val()]
                elif isinstance(value, FloatValue) and value.val() is not None:
                    if constant_float_ir.get(value.val(), None) is None:
                        constant_float_ir[value.val()] = ir_builder.ir_constant_float(value.val())
                    ir_args[i] = constant_float_ir[value.val()]
            new_val = ir_builder.create_ir(stmt.ir_instance, ir_args, None)
            value_lookup_by_ptr[stmt.stmt_id] = new_val
        return ir_builder.export_ir_graph()

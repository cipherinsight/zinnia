import copy

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass
from zinnia.ir_def.defs.ir_dynamic_ndarray_get_item import DynamicNDArrayGetItemIR
from zinnia.ir_def.defs.ir_dynamic_ndarray_set_item import DynamicNDArraySetItemIR
from zinnia.ir_def.defs.ir_read_memory import ReadMemoryIR
from zinnia.ir_def.defs.ir_write_memory import WriteMemoryIR


class DynamicNDArrayMemoryLoweringIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        topological_order = ir_graph.get_topological_order(False)
        in_links, _ = ir_graph.get_io_links()
        values_lookup = {}
        ir_builder = IRBuilderImpl()

        for stmt in topological_order:
            ir_args = [values_lookup[arg] for arg in in_links[stmt.stmt_id]]

            if isinstance(stmt.ir_instance, DynamicNDArrayGetItemIR):
                lowered_val = ir_builder.create_ir(
                    ReadMemoryIR(stmt.ir_instance.segment_id),
                    [ir_args[0]],
                    None,
                )
                values_lookup[stmt.stmt_id] = lowered_val
                continue

            if isinstance(stmt.ir_instance, DynamicNDArraySetItemIR):
                lowered_val = ir_builder.create_ir(
                    WriteMemoryIR(stmt.ir_instance.segment_id),
                    [ir_args[0], ir_args[1]],
                    None,
                )
                values_lookup[stmt.stmt_id] = lowered_val
                continue

            values_lookup[stmt.stmt_id] = ir_builder.create_ir(stmt.ir_instance, ir_args, None)

        return ir_builder.export_ir_graph()

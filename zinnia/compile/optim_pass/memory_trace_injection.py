import copy

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass
from zinnia.ir_def.defs.ir_memory_trace_emit import MemoryTraceEmitIR
from zinnia.ir_def.defs.ir_memory_trace_seal import MemoryTraceSealIR
from zinnia.ir_def.defs.ir_read_memory import ReadMemoryIR
from zinnia.ir_def.defs.ir_write_memory import WriteMemoryIR


class MemoryTraceInjectionIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        topological_order = ir_graph.get_topological_order(False)
        in_links, _ = ir_graph.get_io_links()
        values_lookup = {}
        ir_builder = IRBuilderImpl()
        has_memory_access = False

        for stmt in topological_order:
            ir_args = [values_lookup[arg] for arg in in_links[stmt.stmt_id]]
            new_val = ir_builder.create_ir(stmt.ir_instance, ir_args, None)
            values_lookup[stmt.stmt_id] = new_val

            if isinstance(stmt.ir_instance, WriteMemoryIR):
                has_memory_access = True
                ir_builder.create_ir(
                    MemoryTraceEmitIR(stmt.ir_instance.segment_id, is_write=True),
                    [ir_args[0], ir_args[1]],
                    None,
                )
            elif isinstance(stmt.ir_instance, ReadMemoryIR):
                has_memory_access = True
                ir_builder.create_ir(
                    MemoryTraceEmitIR(stmt.ir_instance.segment_id, is_write=False),
                    [ir_args[0], new_val],
                    None,
                )

        if has_memory_access:
            ir_builder.create_ir(MemoryTraceSealIR(), [], None)

        return ir_builder.export_ir_graph()

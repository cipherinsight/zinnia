import copy

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass
from zinnia.compile.triplet import NoneValue
from zinnia.ir_def.defs.ir_allocate_memory import AllocateMemoryIR
from zinnia.ir_def.defs.ir_dynamic_ndarray_get_item import DynamicNDArrayGetItemIR
from zinnia.ir_def.defs.ir_dynamic_ndarray_set_item import DynamicNDArraySetItemIR
from zinnia.ir_def.defs.ir_read_memory import ReadMemoryIR
from zinnia.ir_def.defs.ir_write_memory import WriteMemoryIR


class DynamicNDArrayMemoryLoweringIRPass(AbstractIRPass):
    def __init__(self, mux_threshold: int = 100):
        super().__init__()
        if mux_threshold <= 0:
            raise ValueError("mux_threshold must be positive")
        self.mux_threshold = mux_threshold

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        topological_order = ir_graph.get_topological_order(False)
        in_links, _ = ir_graph.get_io_links()
        values_lookup = {}
        ir_builder = IRBuilderImpl()
        segment_size_lookup = {
            stmt.ir_instance.segment_id: stmt.ir_instance.size
            for stmt in topological_order
            if isinstance(stmt.ir_instance, AllocateMemoryIR)
        }
        dynamic_access_segments = {
            stmt.ir_instance.segment_id
            for stmt in topological_order
            if isinstance(stmt.ir_instance, (DynamicNDArrayGetItemIR, DynamicNDArraySetItemIR))
        }
        mux_segments = {
            seg_id
            for seg_id, seg_size in segment_size_lookup.items()
            if seg_id in dynamic_access_segments and seg_size < self.mux_threshold
        }
        mux_segment_cells = {}

        for stmt in topological_order:
            ir_args = [values_lookup[arg] for arg in in_links[stmt.stmt_id]]

            if isinstance(stmt.ir_instance, AllocateMemoryIR) and stmt.ir_instance.segment_id in mux_segments:
                init_value = ir_builder.ir_constant_int(stmt.ir_instance.init_value)
                mux_segment_cells[stmt.ir_instance.segment_id] = [init_value for _ in range(stmt.ir_instance.size)]
                values_lookup[stmt.stmt_id] = NoneValue()
                continue

            if isinstance(stmt.ir_instance, DynamicNDArrayGetItemIR):
                if stmt.ir_instance.segment_id in mux_segments and stmt.ir_instance.segment_id in mux_segment_cells:
                    idx_val = ir_args[0]
                    segment_cells = mux_segment_cells[stmt.ir_instance.segment_id]
                    lowered_val = segment_cells[0]
                    for i, cell_val in enumerate(segment_cells):
                        cond = ir_builder.ir_equal_i(idx_val, ir_builder.ir_constant_int(i))
                        lowered_val = ir_builder.ir_select_i(cond, cell_val, lowered_val)
                    values_lookup[stmt.stmt_id] = lowered_val
                    continue

                lowered_val = ir_builder.create_ir(
                    ReadMemoryIR(stmt.ir_instance.segment_id),
                    [ir_args[0]],
                    None,
                )
                values_lookup[stmt.stmt_id] = lowered_val
                continue

            if isinstance(stmt.ir_instance, DynamicNDArraySetItemIR):
                if stmt.ir_instance.segment_id in mux_segments and stmt.ir_instance.segment_id in mux_segment_cells:
                    idx_val = ir_args[0]
                    write_val = ir_args[1]
                    segment_cells = mux_segment_cells[stmt.ir_instance.segment_id]
                    next_cells = []
                    for i, cell_val in enumerate(segment_cells):
                        cond = ir_builder.ir_equal_i(idx_val, ir_builder.ir_constant_int(i))
                        next_cells.append(ir_builder.ir_select_i(cond, write_val, cell_val))
                    mux_segment_cells[stmt.ir_instance.segment_id] = next_cells
                    values_lookup[stmt.stmt_id] = NoneValue()
                    continue

                lowered_val = ir_builder.create_ir(
                    WriteMemoryIR(stmt.ir_instance.segment_id),
                    [ir_args[0], ir_args[1]],
                    None,
                )
                values_lookup[stmt.stmt_id] = lowered_val
                continue

            values_lookup[stmt.stmt_id] = ir_builder.create_ir(stmt.ir_instance, ir_args, None)

        return ir_builder.export_ir_graph()

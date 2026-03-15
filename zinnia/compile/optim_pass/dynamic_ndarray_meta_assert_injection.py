import copy

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass
from zinnia.ir_def.defs.ir_allocate_dynamic_ndarray_meta import AllocateDynamicNDArrayMetaIR
from zinnia.ir_def.defs.ir_assert_dynamic_ndarray_meta import AssertDynamicNDArrayMetaIR
from zinnia.ir_def.defs.ir_witness_dynamic_ndarray_meta import WitnessDynamicNDArrayMetaIR


class DynamicNDArrayMetaAssertInjectionIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        topological_order = ir_graph.get_topological_order(False)
        in_links, _ = ir_graph.get_io_links()
        values_lookup = {}
        meta_lookup = {}
        ir_builder = IRBuilderImpl()

        for stmt in topological_order:
            ir_args = [values_lookup[arg] for arg in in_links[stmt.stmt_id]]
            new_val = ir_builder.create_ir(stmt.ir_instance, ir_args, None)
            values_lookup[stmt.stmt_id] = new_val

            if isinstance(stmt.ir_instance, AllocateDynamicNDArrayMetaIR):
                meta_lookup[stmt.ir_instance.array_id] = (stmt.ir_instance.max_rank, stmt.ir_instance.max_length)
            elif isinstance(stmt.ir_instance, WitnessDynamicNDArrayMetaIR):
                if stmt.ir_instance.array_id not in meta_lookup:
                    raise ValueError(
                        f"WitnessDynamicNDArrayMetaIR references unknown array_id={stmt.ir_instance.array_id}"
                    )
                max_rank, max_length = meta_lookup[stmt.ir_instance.array_id]
                if max_rank != stmt.ir_instance.max_rank:
                    raise ValueError(
                        "WitnessDynamicNDArrayMetaIR max_rank mismatch with allocation metadata"
                    )
                ir_builder.create_ir(
                    AssertDynamicNDArrayMetaIR(stmt.ir_instance.array_id, max_rank, max_length),
                    ir_args,
                    None,
                )

        return ir_builder.export_ir_graph()

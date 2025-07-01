import copy

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass


class DuplicateCodeEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        stmts = ir_graph.export_stmts()
        to_be_replaced = {}
        duplicate_lookup = []  # We temporarily perform O(n^2) search for a quick impl. This can be improved with a hash table.
        for i, stmt in enumerate(stmts):
            existing_stmt_id = None
            for item in duplicate_lookup:
                if item[0] == stmt.ir_instance and item[1] == stmt.arguments:
                    existing_stmt_id = item[2]
                    break
            if existing_stmt_id is None:
                duplicate_lookup.append((stmt.ir_instance, stmt.arguments, stmt.stmt_id))
            else:
                to_be_replaced[stmt.stmt_id] = existing_stmt_id
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        values_lookup = {}
        ir_builder = IRBuilderImpl()
        for stmt in topological_order:
            if stmt.stmt_id in to_be_replaced.keys():
                values_lookup[stmt.stmt_id] = to_be_replaced[stmt.stmt_id]
                continue
            ir_args = [None for _ in in_links[stmt.stmt_id]]
            for i, arg in enumerate(in_links[stmt.stmt_id]):
                replacement = to_be_replaced.get(arg, None)
                if replacement is None:
                    ir_args[i] = values_lookup[arg]
                else:
                    ir_args[i] = values_lookup[replacement]
            values_lookup[stmt.stmt_id] = ir_builder.create_ir(stmt.ir_instance, ir_args, None)
        return ir_builder.export_ir_graph()

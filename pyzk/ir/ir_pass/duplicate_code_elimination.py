from pyzk.ir.ir_builder import IRGraph, IRBuilder
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass


class DuplicateCodeEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        stmts = ir_graph.export_stmts()
        to_be_replaced = {}
        duplicate_lookup = []  # We temporarily perform O(n^2) search for a quick impl
        for i, stmt in enumerate(stmts):
            existing_stmt_id = None
            for item in duplicate_lookup:
                if item[0] == stmt.operator and item[1] == stmt.arguments:
                    existing_stmt_id = item[2]
                    break
            if existing_stmt_id is None:
                duplicate_lookup.append((stmt.operator, stmt.arguments, stmt.stmt_id))
            else:
                to_be_replaced[stmt.stmt_id] = existing_stmt_id
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        old_ptr_to_new_ptr = {}
        ir_builder = IRBuilder()
        for stmt in topological_order:
            if stmt.stmt_id in to_be_replaced.keys():
                old_ptr_to_new_ptr[stmt.stmt_id] = to_be_replaced[stmt.stmt_id]
                continue
            args_as_new_ptrs = {}
            for key, arg in in_links[stmt.stmt_id]:
                replacement = to_be_replaced.get(arg, None)
                if replacement is None:
                    args_as_new_ptrs[key] = old_ptr_to_new_ptr[arg]
                else:
                    args_as_new_ptrs[key] = old_ptr_to_new_ptr[replacement]
            old_ptr_to_new_ptr[stmt.stmt_id] = ir_builder.create_similar(stmt, args_as_new_ptrs)
        return ir_builder.export_ir_graph()


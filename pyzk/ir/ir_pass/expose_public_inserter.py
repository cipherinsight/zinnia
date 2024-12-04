from pyzk.ir.ir_builder import IRGraph, IRBuilder
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass


class ExposePublicInserterIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        assert ir_graph.metadata.annotated
        ir_builder = IRBuilder()
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        old_ptr_to_new_ptr = {}
        for stmt in topological_order:
            referring_tos = in_links[stmt.stmt_id]
            args = {}
            for key, referring_to in referring_tos:
                args[key] = None
                if referring_to is not None:
                    args[key] = old_ptr_to_new_ptr[referring_to]
            old_ptr_to_new_ptr[stmt.stmt_id] = new_ptr = ir_builder.create_similar(stmt, args)
            if stmt.annotation is not None and stmt.annotation.public:
                # TODO:
                # ir_builder.create_expose_public(new_ptr)
                pass
        ir_graph = ir_builder.export_ir_graph()
        return ir_graph

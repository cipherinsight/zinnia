from pyzk.ir.ir_builder import IRGraph
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.opdef.nocls.op_assert import AssertOp


class AlwaysSatisfiedEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        in_links, out_links = ir_graph.get_io_links()
        topological_order = ir_graph.get_topological_order(False)
        to_be_eliminated = []
        inference_descriptors = {}
        for stmt in topological_order:
            referring_tos = in_links[stmt.stmt_id]
            arg_inference_descriptors = {}
            for key, referring_to in referring_tos:
                if referring_to is None:
                    arg_inference_descriptors[key] = None
                    continue
                arg_inference_descriptors[key] = inference_descriptors[referring_to]
            inference_descriptors[stmt.stmt_id] = stmt.operator.static_infer(None, arg_inference_descriptors)
        for stmt in topological_order:
            if isinstance(stmt.operator, AssertOp):
                inf_d = inference_descriptors[in_links[stmt.stmt_id][0][1]]
                if inf_d.get() is not None and inf_d.get() != 0:
                    to_be_eliminated.append(stmt.stmt_id)
        ir_graph.remove_stmt_bunch(to_be_eliminated)
        return ir_graph


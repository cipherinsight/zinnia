from typing import Dict

from pyzk.ir.ir_builder import IRBuilder, IRGraph
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.util.flatten_descriptor import FlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor


class NDArrayFlattenerIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_builder = IRBuilder()
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        flatten_descriptors: Dict[int, FlattenDescriptor] = {}
        inference_descriptors: Dict[int, InferenceDescriptor] = {}
        for stmt in topological_order:
            referring_tos = in_links[stmt.stmt_id]
            arg_flatten_descriptors = {}
            arg_inference_descriptors = {}
            for key, referring_to in referring_tos:
                if referring_to is None:
                    arg_flatten_descriptors[key] = None
                    arg_inference_descriptors[key] = None
                    continue
                arg_flatten_descriptors[key] = flatten_descriptors[referring_to]
                arg_inference_descriptors[key] = inference_descriptors[referring_to]
            flatten_descriptors[stmt.stmt_id] = stmt.operator.ir_flatten(ir_builder, arg_flatten_descriptors)
            inference_descriptors[stmt.stmt_id] = stmt.operator.static_infer(None, arg_inference_descriptors)
            flatten_descriptors[stmt.stmt_id].set_val(inference_descriptors[stmt.stmt_id].get())
        ir_graph = ir_builder.export_ir_graph()
        ir_graph.metadata.ndarray_flattened = True
        return ir_graph

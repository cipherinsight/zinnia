from pyzk.ir.ir_builder import IRGraph, IRBuilder
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.internal.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor


class ConstantFoldIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_builder = IRBuilder()
        constant_number_to_new_ptr = {}
        old_ptr_to_new_ptr = {}
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
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
            args_as_new_ptrs = {}
            for key, arg in in_links[stmt.stmt_id]:
                args_as_new_ptrs[key] = old_ptr_to_new_ptr[arg]
                inference_d: InferenceDescriptor = inference_descriptors[arg]
                if isinstance(inference_d, NumberInferenceDescriptor) and inference_d.get() is not None:
                    constant_val = inference_d.get()
                    if constant_number_to_new_ptr.get(constant_val, None) is None:
                        constant_number_to_new_ptr[constant_val] = ir_builder.create_constant(constant_val)
                    old_ptr_to_new_ptr[stmt.stmt_id] = args_as_new_ptrs[key] = constant_number_to_new_ptr[constant_val]
            old_ptr_to_new_ptr[stmt.stmt_id] = ir_builder.create_similar(stmt, args_as_new_ptrs)
        return ir_builder.export_ir_graph()

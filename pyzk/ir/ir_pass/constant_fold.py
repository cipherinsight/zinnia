from pyzk.ir.ir_builder import IRGraph, IRBuilder
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.inference.constant_fold import ConstantFold
from pyzk.util.op_name import OpName

class ConstantFoldIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_builder = IRBuilder()
        constant_number_to_new_ptr = {}
        new_ptr_to_constant_number = {}
        old_ptr_to_new_ptr = {}
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        for stmt in topological_order:
            if stmt.op == OpName.Special.CONSTANT:
                if constant_number_to_new_ptr.get(stmt.constant_value, None) is None:
                    constant_number_to_new_ptr[stmt.constant_value] = ir_builder.create_constant(stmt.constant_value)
                old_ptr_to_new_ptr[stmt.stmt_id] = constant_number_to_new_ptr[stmt.constant_value]
                new_ptr_to_constant_number[old_ptr_to_new_ptr[stmt.stmt_id]] = stmt.constant_value
                continue
            should_try_fold = True
            args_as_constants = []
            args_as_new_ptrs = []
            for arg in in_links[stmt.stmt_id]:
                args_as_new_ptrs.append(old_ptr_to_new_ptr[arg])
            for arg in args_as_new_ptrs:
                if new_ptr_to_constant_number.get(arg) is not None and should_try_fold:
                    args_as_constants.append(new_ptr_to_constant_number[arg])
                else:
                    should_try_fold = False
            if should_try_fold:
                constant_fold_value = ConstantFold.constant_fold(stmt.op, args_as_constants)
                if constant_fold_value is not None:
                    if constant_number_to_new_ptr.get(constant_fold_value, None) is None:
                        constant_number_to_new_ptr[constant_fold_value] = ir_builder.create_constant(constant_fold_value)
                    old_ptr_to_new_ptr[stmt.stmt_id] = constant_number_to_new_ptr[constant_fold_value]
                    new_ptr_to_constant_number[old_ptr_to_new_ptr[stmt.stmt_id]] = constant_fold_value
                    continue
            old_ptr_to_new_ptr[stmt.stmt_id] = ir_builder.create_similar(stmt, args_as_new_ptrs)
        return ir_builder.export_ir_graph()

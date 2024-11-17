from typing import Dict, List

from pyzk.inference.ir_inference import IRInferenceDescriptor


class IRContext:
    def __init__(self):
        self.name_to_ptr_stack = [{}]
        self.var_table = {}
        self.if_condition_stack = []
        self.for_condition_stack = []
        self.inference_table: Dict[int, IRInferenceDescriptor] = {}

    def block_enter(self):
        self.name_to_ptr_stack.append({})

    def block_leave(self):
        self.name_to_ptr_stack.pop()

    def assign_name_to_ptr(self, name: str, ptr: int):
        self.var_table[ptr] = name
        for i in reversed(range(len(self.name_to_ptr_stack))):
            if name in self.name_to_ptr_stack[i]:
                self.name_to_ptr_stack[i][name] = ptr
                return
        self.name_to_ptr_stack[-1][name] = ptr

    def lookup_ptr_by_name(self, name: str) -> int | None:
        for i in reversed(range(len(self.name_to_ptr_stack))):
            if name in self.name_to_ptr_stack[i]:
                return self.name_to_ptr_stack[i][name]
        return None

    def is_name_in_stack_scope_top(self, name: str) -> bool:
        assert self.lookup_ptr_by_name(name) is not None
        return name in self.name_to_ptr_stack[-1]

    def set_inference_descriptor(self, ptr: int, descriptor: IRInferenceDescriptor):
        self.inference_table[ptr] = descriptor

    def get_inference_descriptor(self, ptr: int) -> IRInferenceDescriptor:
        return self.inference_table.get(ptr, None)

    def is_inferred_datatype_equal(self, lhs: int, rhs: int) -> bool:
        lhs_descriptor = self.inference_table.get(lhs, None)
        rhs_descriptor = self.inference_table.get(rhs, None)
        if lhs_descriptor is None or rhs_descriptor is None:
            return False
        return lhs_descriptor.datatype_matches(rhs_descriptor)

    def get_inferred_datatype_name(self, ptr: int) -> str:
        descriptor = self.inference_table.get(ptr, None)
        assert descriptor is not None
        return descriptor.pretty_typename()

    def get_inferred_constant_value(self, ptr: int) -> int | List | None:
        descriptor: IRInferenceDescriptor = self.inference_table.get(ptr, None)
        if descriptor is None:
            return None
        inferred_value = descriptor.value
        if inferred_value is None:
            return None
        return inferred_value

    def if_block_enter(self, cond_expr: int):
        self.if_condition_stack.append(cond_expr)

    def if_block_leave(self):
        self.if_condition_stack.pop()

    def for_block_enter(self, constant_true: int, constant_false: int):
        self.for_condition_stack.append([True, True, constant_true, constant_false])

    def for_block_continue(self):
        assert len(self.for_condition_stack) > 0
        self.for_condition_stack[-1][0] = False

    def for_block_break(self):
        assert len(self.for_condition_stack) > 0
        self.for_condition_stack[-1][1] = False

    def for_block_reiter(self):
        assert len(self.for_condition_stack) > 0
        self.for_condition_stack[-1][0] = True

    def for_block_exists(self):
        return len(self.for_condition_stack) > 0

    def for_block_leave(self):
        self.for_condition_stack.pop()

    def get_condition_variables(self):
        variables = self.if_condition_stack.copy()
        for for_block in self.for_condition_stack:
            if for_block[0] and for_block[1]:
                variables.append(for_block[2])
            else:
                variables.append(for_block[3])
        return variables

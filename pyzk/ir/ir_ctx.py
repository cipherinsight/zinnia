from typing import Dict, List, Any, Tuple

from pyzk.builder.value import Value, IntegerValue


class IRContext:
    class _Scope:
        name_to_ptr_stack: List[Dict]
        var_table: Dict
        return_cond_vals: List[Tuple[Value, IntegerValue]]
        return_prevent_conditions = List[Value]
        branch_has_default_return_stack: List[bool]

        def __init__(self):
            self.name_to_ptr_stack = [{}]
            self.var_table = {}
            self.return_cond_vals = []
            self.return_prevent_conditions = []
            self.branch_has_default_return_stack = [False]

    def __init__(self):
        self.scopes_stack = [IRContext._Scope()]
        self.if_condition_stack = []
        self.for_condition_stack = []
        self.chip_registry = {}
        self.chip_recur_stack = []

    def block_enter(self):
        self.scopes_stack[-1].name_to_ptr_stack.append({})

    def block_leave(self):
        self.scopes_stack[-1].name_to_ptr_stack.pop()

    def get_block_depth(self):
        return len(self.scopes_stack[-1].name_to_ptr_stack) - 1

    def register_chip(self, key: str, chip: Any):
        self.chip_registry[key] = chip

    def lookup_chip(self, key: str) -> Any:
        if key not in self.chip_registry.keys():
            return None
        return self.chip_registry[key]

    def chip_enter(self, chip: Any):
        self.chip_recur_stack.append(chip)
        self.scopes_stack.append(IRContext._Scope())

    def chip_leave(self):
        self.chip_recur_stack.pop()
        self.scopes_stack.pop()

    def get_chip_depth(self):
        return len(self.chip_recur_stack)

    def get_current_chip(self):
        return self.chip_recur_stack[-1]

    def assign_name_to_ptr(self, name: str, ptr: Value):
        assert isinstance(ptr, Value)
        scope = self.scopes_stack[-1]
        scope.var_table[ptr] = name
        for i in reversed(range(len(scope.name_to_ptr_stack))):
            if name in scope.name_to_ptr_stack[i]:
                scope.name_to_ptr_stack[i][name] = ptr
                return
        scope.name_to_ptr_stack[-1][name] = ptr

    def lookup_ptr_by_name(self, name: str) -> Value | None:
        scope = self.scopes_stack[-1]
        for i in reversed(range(len(scope.name_to_ptr_stack))):
            if name in scope.name_to_ptr_stack[i]:
                return scope.name_to_ptr_stack[i][name]
        return None

    def is_name_in_stack_scope_top(self, name: str) -> bool:
        assert self.lookup_ptr_by_name(name) is not None
        scope = self.scopes_stack[-1]
        return name in scope.name_to_ptr_stack[-1]

    def if_block_enter(self, cond_expr: Value):
        self.if_condition_stack.append(cond_expr)
        scope = self.scopes_stack[-1]
        scope.branch_has_default_return_stack.append(False)

    def if_block_leave(self):
        self.if_condition_stack.pop()
        scope = self.scopes_stack[-1]
        scope.branch_has_default_return_stack.pop()

    def add_return_value(self, return_value: Value, return_condition: IntegerValue, return_prevent_condition: IntegerValue):
        scope = self.scopes_stack[-1]
        scope.branch_has_default_return_stack[-1] = True
        scope.return_cond_vals.append((return_value, return_condition))
        scope.return_prevent_conditions.append(return_prevent_condition)

    def get_return_values_and_conditions(self) -> List[Tuple[Value, IntegerValue]]:
        scope = self.scopes_stack[-1]
        return scope.return_cond_vals

    def get_branch_has_default_return(self) -> bool:
        return self.scopes_stack[-1].branch_has_default_return_stack[-1]

    def set_branch_has_default_return(self):
        self.scopes_stack[-1].branch_has_default_return_stack[-1] = True

    def for_block_enter(self):
        self.for_condition_stack.append(([], []))

    def for_block_continue(self, continue_conditions_not: IntegerValue):
        assert len(self.for_condition_stack) > 0
        self.for_condition_stack[-1][0].append(continue_conditions_not)

    def for_block_break(self, break_conditions_not: IntegerValue):
        assert len(self.for_condition_stack) > 0
        self.for_condition_stack[-1][1].append(break_conditions_not)

    def for_block_reiter(self):
        assert len(self.for_condition_stack) > 0
        self.for_condition_stack[-1] = [], self.for_condition_stack[-1][1]

    def for_block_exists(self):
        return len(self.for_condition_stack) > 0

    def for_block_leave(self):
        self.for_condition_stack.pop()

    def get_condition_variables(self) -> List[IntegerValue]:
        variables = self.if_condition_stack.copy()
        for for_block in self.for_condition_stack:
            variables += for_block[0] + for_block[1]
        for scope in self.scopes_stack:
            variables += scope.return_prevent_conditions.copy()
        return variables

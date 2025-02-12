from typing import Optional, Dict

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.scope import AbstractScope
from zinnia.compile.triplet.store import ValueStore
from zinnia.compile.type_sys import DTDescriptor


class LoopScope(AbstractScope):
    continue_condition: IntegerValue | None
    break_condition: IntegerValue | None
    return_guaranteed: bool
    loop_terminated_guaranteed: bool
    var_table: Dict[str, ValueStore]
    calculated_looping_condition: IntegerValue | None
    super_looping_condition: IntegerValue | None

    def __init__(self, ir_builder: IRBuilderInterface, super_scope: Optional['AbstractScope']):
        super().__init__(ir_builder, super_scope)
        self.continue_condition = None
        self.break_condition = None
        self.return_guaranteed = False
        self.loop_terminated_guaranteed = False
        self.var_table = {}
        self.calculated_looping_condition = None
        self.super_looping_condition = self.super_scope.get_looping_condition()

    def set(self, name: str, ptr: ValueStore):
        assert isinstance(ptr, ValueStore)
        if self.super_scope.exists(name):
            self.super_scope.set(name, ptr)
        else:
            self.var_table[name] = ptr

    def get(self, name: str) -> ValueStore:
        if name in self.var_table:
            return self.var_table[name]
        return self.super_scope.get(name)

    def exists(self, name: str) -> bool:
        if name in self.var_table:
            return True
        return self.super_scope.exists(name)

    def exists_in_this(self, name: str) -> bool:
        if name in self.var_table:
            return True
        return False

    def scope_enter(self, *args, **kwargs):
        pass

    def scope_leave(self, *args, **kwargs):
        pass

    def is_in_chip(self) -> bool:
        return self.super_scope.is_in_chip()

    def is_in_loop(self) -> bool:
        return True

    def get_branching_condition(self) -> IntegerValue | None:
        return self.super_scope.get_branching_condition()

    def get_looping_condition(self) -> IntegerValue | None:
        return self.calculated_looping_condition

    def get_breaking_condition(self) -> IntegerValue | None:
        return self.break_condition

    def get_returning_condition(self) -> IntegerValue | None:
        return self.super_scope.get_returning_condition()

    def get_return_dtype(self) -> DTDescriptor:
        return self.super_scope.get_return_dtype()

    def set_return_guarantee(self):
        self.return_guaranteed = True

    def set_terminated_guarantee(self):
        self.loop_terminated_guaranteed = True

    def register_return(self, value: Value, condition: IntegerValue):
        self.super_scope.register_return(value, condition)

    def is_return_guaranteed(self) -> bool:
        return self.return_guaranteed

    def is_terminated_guaranteed(self) -> bool:
        return self.loop_terminated_guaranteed

    def loop_continue(self, condition: IntegerValue):
        condition = self.ir_builder.ir_logical_not(condition)
        if self.continue_condition is None:
            self.continue_condition = condition
        else:
            self.continue_condition = self.ir_builder.ir_logical_and(self.continue_condition, condition)
        if self.break_condition is None:
            self.calculated_looping_condition = self.continue_condition
        else:
            self.calculated_looping_condition = self.ir_builder.ir_logical_and(self.continue_condition, self.break_condition)
        if self.super_looping_condition is not None:
            self.calculated_looping_condition = self.ir_builder.ir_logical_and(self.calculated_looping_condition, self.super_looping_condition)

    def loop_break(self, condition: IntegerValue):
        condition = self.ir_builder.ir_logical_not(condition)
        if self.break_condition is None:
            self.break_condition = condition
        else:
            self.break_condition = self.ir_builder.ir_logical_and(self.break_condition, condition)
        if self.continue_condition is None:
            self.calculated_looping_condition = self.break_condition
        else:
            self.calculated_looping_condition = self.ir_builder.ir_logical_and(self.continue_condition, self.break_condition)
        if self.super_looping_condition is not None:
            self.calculated_looping_condition = self.ir_builder.ir_logical_and(self.calculated_looping_condition, self.super_looping_condition)

    def loop_reiterate(self):
        self.continue_condition = None
        self.calculated_looping_condition = self.break_condition
        if self.super_looping_condition is not None:
            self.calculated_looping_condition = self.ir_builder.ir_logical_and(self.calculated_looping_condition, self.super_looping_condition)

    def lock_parent_variable_types(self) -> bool:
        if self.break_condition is None and self.continue_condition is None:
            return False
        if (self.break_condition is not None and (self.break_condition.val() is None or self.break_condition.val() == 0)) or (self.continue_condition is not None and (self.continue_condition.val() is None or self.continue_condition.val() == 0)):
            return True
        return False

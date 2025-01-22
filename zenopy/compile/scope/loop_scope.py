from typing import Optional, List, Dict

from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue
from zenopy.compile.scope import AbstractScope
from zenopy.internal.dt_descriptor import DTDescriptor


class LoopScope(AbstractScope):
    continue_condition: IntegerValue | None
    break_condition: IntegerValue | None
    var_table: Dict[str, Value]
    calculated_looping_condition: IntegerValue | None
    super_looping_condition: IntegerValue | None

    def __init__(self, ir_builder: AbsIRBuilderInterface, super_scope: Optional['AbstractScope']):
        super().__init__(ir_builder, super_scope)
        self.continue_condition = None
        self.break_condition = None
        self.var_table = {}
        self.calculated_looping_condition = None
        self.super_looping_condition = self.super_scope.get_looping_condition()

    def set(self, name: str, ptr: Value):
        assert isinstance(ptr, Value)
        if self.super_scope.exists(name):
            self.super_scope.set(name, ptr)
        else:
            self.var_table[name] = ptr

    def get(self, name: str) -> Value:
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

    def return_value(self, value: Value, condition: IntegerValue):
        self.super_scope.return_value(value, condition)

    def set_has_return(self):
        self.super_scope.set_has_return()

    def register_return(self, value: Value, condition: IntegerValue):
        self.super_scope.register_return(value, condition)

    def has_return_statement(self) -> bool:
        return self.super_scope.has_return_statement()

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

from typing import Optional, List, Tuple, Dict

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.scope.abs_scope import AbstractScope
from zinnia.compile.triplet.store import ValueStore
from zinnia.compile.type_sys import DTDescriptor


class ConditionalScope(AbstractScope):
    var_table: Dict[str, ValueStore]
    condition: IntegerValue
    return_guaranteed: bool
    loop_terminated_guaranteed: bool
    calculated_branching_condition: IntegerValue | None

    def __init__(self, ir_builder: IRBuilderInterface, super_scope: Optional['AbstractScope'], condition: IntegerValue):
        super().__init__(ir_builder, super_scope)
        self.var_table = {}
        self.return_guaranteed = False
        self.loop_terminated_guaranteed = False
        self.condition = condition
        super_branching_condition = self.super_scope.get_branching_condition()
        if super_branching_condition is not None:
            self.calculated_branching_condition = self.ir_builder.ir_logical_and(self.condition, super_branching_condition)
        else:
            self.calculated_branching_condition = self.condition

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

    def get_branching_condition(self) -> IntegerValue | None:
        return self.calculated_branching_condition

    def get_looping_condition(self) -> IntegerValue | None:
        return self.super_scope.get_looping_condition()

    def get_breaking_condition(self) -> IntegerValue | None:
        return self.super_scope.get_breaking_condition()

    def get_returning_condition(self) -> IntegerValue | None:
        return self.super_scope.get_returning_condition()

    def get_returns_with_conditions(self) -> List[Tuple[Value, IntegerValue]]:
        return self.super_scope.get_returns_with_conditions()

    def is_return_guaranteed(self) -> bool:
        return self.return_guaranteed

    def is_terminated_guaranteed(self) -> bool:
        return self.loop_terminated_guaranteed

    def set_return_guarantee(self):
        self.return_guaranteed = True

    def set_terminated_guarantee(self):
        self.loop_terminated_guaranteed = True

    def register_return(self, value: Value, condition: IntegerValue):
        self.super_scope.register_return(value, condition)

    def loop_break(self, condition: IntegerValue):
        return self.super_scope.loop_break(condition)

    def loop_continue(self, condition: IntegerValue):
        return self.super_scope.loop_continue(condition)

    def loop_reiterate(self):
        return self.super_scope.loop_reiterate()

    def is_in_chip(self) -> bool:
        return self.super_scope.is_in_chip()

    def is_in_loop(self) -> bool:
        return self.super_scope.is_in_loop()

    def get_return_dtype(self) -> DTDescriptor:
        return self.super_scope.get_return_dtype()

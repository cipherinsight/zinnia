from typing import Optional, List, Tuple

from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue
from zenopy.internal.dt_descriptor import DTDescriptor


class AbstractScope:
    super_scope: Optional['AbstractScope']
    ir_builder: AbsIRBuilderInterface

    def __init__(self, ir_builder: AbsIRBuilderInterface, super_scope: Optional['AbstractScope']):
        self.ir_builder = ir_builder
        self.super_scope = super_scope

    def scope_enter(self, *args, **kwargs):
        pass

    def scope_leave(self, *args, **kwargs):
        pass

    def set(self, name: str, ptr: Value):
        raise NotImplementedError()

    def get(self, name: str) -> Value:
        raise NotImplementedError()

    def exists(self, name: str) -> bool:
        raise NotImplementedError()

    def exists_in_this(self, name: str) -> bool:
        raise NotImplementedError()

    def set_has_return(self):
        raise NotImplementedError()

    def is_in_chip(self) -> bool:
        raise NotImplementedError()

    def is_in_loop(self) -> bool:
        raise NotImplementedError()

    def get_branching_condition(self) -> IntegerValue | None:
        raise NotImplementedError()

    def get_looping_condition(self) -> IntegerValue | None:
        raise NotImplementedError()

    def get_breaking_condition(self) -> IntegerValue | None:
        raise NotImplementedError()

    def get_returning_condition(self) -> IntegerValue | None:
        raise NotImplementedError()

    def return_value(self, value: Value, condition: IntegerValue):
        raise NotImplementedError()

    def register_return(self, value: Value, condition: IntegerValue):
        raise NotImplementedError()

    def has_return_statement(self) -> bool:
        raise NotImplementedError()

    def get_returns_with_conditions(self) -> List[Tuple[Value, IntegerValue]]:
        raise NotImplementedError()

    def get_return_dtype(self) -> DTDescriptor:
        raise NotImplementedError()

    def loop_break(self, condition: IntegerValue):
        raise NotImplementedError()

    def loop_continue(self, condition: IntegerValue):
        raise NotImplementedError()

    def loop_reiterate(self):
        raise NotImplementedError()

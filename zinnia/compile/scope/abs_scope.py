from typing import Optional, List, Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet.store import ValueStore
from zinnia.compile.triplet.value import Value, BooleanValue
from zinnia.compile.type_sys import DTDescriptor


class AbstractScope:
    super_scope: Optional['AbstractScope']
    ir_builder: IRBuilderInterface

    def __init__(self, ir_builder: IRBuilderInterface, super_scope: Optional['AbstractScope']):
        self.ir_builder = ir_builder
        self.super_scope = super_scope

    def scope_enter(self, *args, **kwargs):
        pass

    def scope_leave(self, *args, **kwargs):
        pass

    def set(self, name: str, ptr: ValueStore):
        raise NotImplementedError()

    def get(self, name: str) -> Value:
        raise NotImplementedError()

    def exists(self, name: str) -> bool:
        raise NotImplementedError()

    def exists_in_this(self, name: str) -> bool:
        raise NotImplementedError()

    def lock_parent_variable_types(self) -> bool:
        return False

    def set_return_guarantee(self):
        raise NotImplementedError()

    def set_terminated_guarantee(self):
        raise NotImplementedError()

    def is_in_chip(self) -> bool:
        raise NotImplementedError()

    def is_in_loop(self) -> bool:
        raise NotImplementedError()

    def get_branching_condition(self) -> BooleanValue | None:
        raise NotImplementedError()

    def get_looping_condition(self) -> BooleanValue | None:
        raise NotImplementedError()

    def get_breaking_condition(self) -> BooleanValue | None:
        raise NotImplementedError()

    def get_returning_condition(self) -> BooleanValue | None:
        raise NotImplementedError()

    def get_assertion_condition(self) -> BooleanValue | None:
        raise NotImplementedError()

    def register_return(self, value: Value, condition: BooleanValue):
        raise NotImplementedError()

    def is_return_guaranteed(self) -> bool:
        raise NotImplementedError()

    def is_terminated_guaranteed(self) -> bool:
        raise NotImplementedError()

    def get_returns_with_conditions(self) -> List[Tuple[Value, BooleanValue]]:
        raise NotImplementedError()

    def get_return_dtype(self) -> DTDescriptor:
        raise NotImplementedError()

    def loop_break(self, condition: BooleanValue):
        raise NotImplementedError()

    def loop_continue(self, condition: BooleanValue):
        raise NotImplementedError()

    def loop_reiterate(self):
        raise NotImplementedError()

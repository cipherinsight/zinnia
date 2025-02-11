from typing import Dict

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import IntegerValue
from zinnia.compile.scope.abs_scope import AbstractScope
from zinnia.compile.triplet.store import ValueStore


class MasterScope(AbstractScope):
    var_table: Dict[str, ValueStore]

    def __init__(self, ir_builder: IRBuilderInterface):
        super().__init__(ir_builder, None)
        self.var_table = {}

    def scope_enter(self, *args, **kwargs):
        pass

    def scope_leave(self, *args, **kwargs):
        pass

    def set(self, name: str, ptr: ValueStore):
        assert isinstance(ptr, ValueStore)
        self.var_table[name] = ptr

    def get(self, name: str) -> ValueStore:
        if name in self.var_table:
            return self.var_table[name]
        raise ValueError(f'Internal Error: Variable {name} not found in scope. Did you forget to check existence?')

    def exists(self, name: str) -> bool:
        if name in self.var_table:
            return True
        return False

    def exists_in_this(self, name: str) -> bool:
        if name in self.var_table:
            return True
        return False

    def is_in_chip(self) -> bool:
        return False

    def is_in_loop(self) -> bool:
        return False

    def get_branching_condition(self) -> IntegerValue | None:
        return None

    def get_looping_condition(self) -> IntegerValue | None:
        return None

    def get_returning_condition(self) -> IntegerValue | None:
        return None

    def get_breaking_condition(self) -> IntegerValue | None:
        return None

    def has_return_statement(self) -> bool:
        return False

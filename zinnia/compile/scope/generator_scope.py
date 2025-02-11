from typing import Optional, Dict

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import IntegerValue
from zinnia.compile.scope.abs_scope import AbstractScope
from zinnia.compile.triplet.store import ValueStore


class GeneratorScope(AbstractScope):
    var_table: Dict[str, ValueStore]

    def __init__(self, ir_builder: IRBuilderInterface, super_scope: Optional['AbstractScope']):
        super().__init__(ir_builder, super_scope)
        self.var_table = {}

    def set(self, name: str, ptr: ValueStore):
        assert isinstance(ptr, ValueStore)
        self.var_table[name] = ptr

    def get(self, name: str) -> ValueStore:
        if name in self.var_table:
            return self.var_table[name]
        return self.super_scope.get(name)

    def get_branching_condition(self) -> IntegerValue | None:
        return None

    def get_looping_condition(self) -> IntegerValue | None:
        return None

    def get_breaking_condition(self) -> IntegerValue | None:
        return None

    def get_returning_condition(self) -> IntegerValue | None:
        return None

    def exists(self, name: str) -> bool:
        if name in self.var_table:
            return True
        return self.super_scope.exists(name)

    def exists_in_this(self, name: str) -> bool:
        if name in self.var_table:
            return True
        return False

    def is_in_chip(self) -> bool:
        return self.super_scope.is_in_chip()

    def is_in_loop(self) -> bool:
        return self.super_scope.is_in_loop()

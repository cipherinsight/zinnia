from typing import Optional, Dict

from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value
from zenopy.compile.scope.abs_scope import AbstractScope


class GeneratorScope(AbstractScope):
    var_table: Dict[str, Value]

    def __init__(self, ir_builder: AbsIRBuilderInterface, super_scope: Optional['AbstractScope']):
        super().__init__(ir_builder, super_scope)
        self.var_table = {}

    def set(self, name: str, ptr: Value):
        assert isinstance(ptr, Value)
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

    def is_in_chip(self) -> bool:
        return self.super_scope.is_in_chip()

    def is_in_loop(self) -> bool:
        return self.super_scope.is_in_loop()

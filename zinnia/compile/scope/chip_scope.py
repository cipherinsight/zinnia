from typing import Optional, List, Tuple, Dict

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, BooleanValue
from zinnia.compile.scope.abs_scope import AbstractScope
from zinnia.compile.triplet.store import ValueStore
from zinnia.compile.type_sys import DTDescriptor


class ChipScope(AbstractScope):
    var_table: Dict[str, ValueStore]
    return_guaranteed: bool
    return_dtype: DTDescriptor
    returns_with_conditions: List[Tuple[Value, BooleanValue]]
    calculated_returning_condition: BooleanValue | None
    assertion_condition: BooleanValue | None

    def __init__(
            self,
            ir_builder: IRBuilderInterface,
            super_scope: Optional['AbstractScope'],
            return_dtype: DTDescriptor,
            assertion_condition: BooleanValue | None
    ):
        super().__init__(ir_builder, super_scope)
        self.var_table = {}
        self.return_guaranteed = False
        self.return_dtype = return_dtype
        self.returns_with_conditions = []
        self.calculated_returning_condition = None
        self.assertion_condition = assertion_condition

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
        return True

    def is_in_loop(self) -> bool:
        return False

    def get_branching_condition(self) -> BooleanValue | None:
        return None

    def get_looping_condition(self) -> BooleanValue | None:
        return None

    def get_breaking_condition(self) -> BooleanValue | None:
        return None

    def get_returning_condition(self) -> BooleanValue | None:
        return self.calculated_returning_condition

    def get_assertion_condition(self) -> BooleanValue | None:
        return self.assertion_condition

    def get_returns_with_conditions(self) -> List[Tuple[Value, BooleanValue]]:
        return self.returns_with_conditions

    def is_return_guaranteed(self) -> bool:
        return self.return_guaranteed

    def set_return_guarantee(self):
        self.return_guaranteed = True

    def set_terminated_guarantee(self):
        raise NotImplementedError("Unexpected `set_terminated_guarantee` call on a chip scope.")

    def is_terminated_guaranteed(self) -> bool:
        return False

    def register_return(self, value: Value, condition: BooleanValue):
        self.returns_with_conditions.append((value, condition))
        if self.calculated_returning_condition is None:
            self.calculated_returning_condition = self.ir_builder.ir_logical_not(condition)
        else:
            self.calculated_returning_condition = self.ir_builder.ir_logical_and(self.calculated_returning_condition, self.ir_builder.ir_logical_not(condition))

    def get_return_dtype(self) -> DTDescriptor:
        return self.return_dtype

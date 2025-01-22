from typing import Optional, List, Tuple, Dict

from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue
from zenopy.compile.scope.abs_scope import AbstractScope
from zenopy.internal.dt_descriptor import DTDescriptor


class ChipScope(AbstractScope):
    var_table: Dict[str, Value]
    has_return_stmt: bool
    return_dtype: DTDescriptor
    returns_with_conditions: List[Tuple[Value, IntegerValue]]
    calculated_returning_condition: IntegerValue | None

    def __init__(self, ir_builder: AbsIRBuilderInterface, super_scope: Optional['AbstractScope'], return_dtype: DTDescriptor):
        super().__init__(ir_builder, super_scope)
        self.var_table = {}
        self.has_return_stmt = False
        self.return_dtype = return_dtype
        self.returns_with_conditions = []
        self.calculated_returning_condition = None

    def scope_enter(self, *args, **kwargs):
        pass

    def scope_leave(self, *args, **kwargs):
        pass

    def set(self, name: str, ptr: Value):
        assert isinstance(ptr, Value)
        self.var_table[name] = ptr

    def get(self, name: str) -> Value:
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

    def get_branching_condition(self) -> IntegerValue | None:
        return None

    def get_looping_condition(self) -> IntegerValue | None:
        return None

    def get_breaking_condition(self) -> IntegerValue | None:
        return None

    def get_returning_condition(self) -> IntegerValue | None:
        return self.calculated_returning_condition

    def get_returns_with_conditions(self) -> List[Tuple[Value, IntegerValue]]:
        return self.returns_with_conditions

    def return_value(self, value: Value, condition: IntegerValue):
        self.has_return_stmt = True
        self.register_return(value, condition)

    def has_return_statement(self) -> bool:
        return self.has_return_stmt

    def set_has_return(self):
        self.has_return_stmt = True

    def register_return(self, value: Value, condition: IntegerValue):
        self.returns_with_conditions.append((value, condition))
        if self.calculated_returning_condition is None:
            self.calculated_returning_condition = self.ir_builder.ir_logical_not(condition)
        else:
            self.calculated_returning_condition = self.ir_builder.ir_logical_and(self.calculated_returning_condition, self.ir_builder.ir_logical_not(condition))

    def get_return_dtype(self) -> DTDescriptor:
        return self.return_dtype

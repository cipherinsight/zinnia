from typing import List, Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue
from zinnia.compile.scope import MasterScope, ChipScope, AbstractScope, LoopScope, ConditionalScope, GeneratorScope
from zinnia.compile.type_sys import DTDescriptor
from zinnia.compile.triplet.value_factory import ValueFactory


class IRContext:
    scopes: List[AbstractScope]
    ir_builder: IRBuilderInterface

    def __init__(self, ir_builder: IRBuilderInterface):
        self.scopes = [MasterScope(ir_builder)]
        self.ir_builder = ir_builder

    def set(self, key: str, val: Value):
        self.scopes[-1].set(key, val.into_value_store())

    def get(self, key: str) -> Value:
        exists_in_this = self.scopes[-1].exists_in_this(key)
        value_store = self.scopes[-1].get(key)
        return ValueFactory.from_value_store(value_store, not exists_in_this)

    def exists(self, key: str) -> bool:
        return self.scopes[-1].exists(key)

    def exists_in_top_scope(self, key: str) -> bool:
        return self.scopes[-1].exists_in_this(key)

    def chip_enter(self, return_dtype: DTDescriptor):
        new_scope = ChipScope(self.ir_builder, self.scopes[-1], return_dtype)
        new_scope.scope_enter()
        self.scopes.append(new_scope)

    def chip_leave(self) -> AbstractScope:
        assert isinstance(self.scopes[-1], ChipScope)
        self.scopes[-1].scope_leave()
        self.scopes, last_scope = self.scopes[:-1], self.scopes[-1]
        return last_scope

    def loop_enter(self):
        new_scope = LoopScope(self.ir_builder, self.scopes[-1])
        new_scope.scope_enter()
        self.scopes.append(new_scope)

    def loop_leave(self) -> AbstractScope:
        assert isinstance(self.scopes[-1], LoopScope)
        self.scopes[-1].scope_leave()
        self.scopes, last_scope = self.scopes[:-1], self.scopes[-1]
        return last_scope

    def loop_reiter(self):
        assert self.scopes[-1].is_in_loop()
        self.scopes[-1].loop_reiterate()

    def loop_break(self):
        assert self.scopes[-1].is_in_loop()
        self.scopes[-1].loop_break(self.get_condition_value())

    def loop_continue(self):
        assert self.scopes[-1].is_in_loop()
        self.scopes[-1].loop_continue(self.get_condition_value())

    def if_enter(self, condition: IntegerValue):
        new_scope = ConditionalScope(self.ir_builder, self.scopes[-1], condition)
        new_scope.scope_enter()
        self.scopes.append(new_scope)

    def if_leave(self) -> AbstractScope:
        assert isinstance(self.scopes[-1], ConditionalScope)
        self.scopes[-1].scope_leave()
        self.scopes, last_scope = self.scopes[:-1], self.scopes[-1]
        return last_scope

    def generator_enter(self):
        new_scope = GeneratorScope(self.ir_builder, self.scopes[-1])
        new_scope.scope_enter()
        self.scopes.append(new_scope)

    def generator_leave(self) -> AbstractScope:
        assert isinstance(self.scopes[-1], GeneratorScope)
        self.scopes[-1].scope_leave()
        self.scopes, last_scope = self.scopes[:-1], self.scopes[-1]
        return last_scope

    def is_in_chip(self) -> bool:
        return self.scopes[-1].is_in_chip()

    def is_in_loop(self) -> bool:
        return self.scopes[-1].is_in_loop()

    def get_condition_value(self) -> IntegerValue:
        condition_1 = self.scopes[-1].get_looping_condition()
        condition_2 = self.scopes[-1].get_branching_condition()
        condition_3 = self.scopes[-1].get_returning_condition()
        conditions = []
        if condition_1 is not None:
            conditions.append(condition_1)
        if condition_2 is not None:
            conditions.append(condition_2)
        if condition_3 is not None:
            conditions.append(condition_3)
        if len(conditions) == 0:
            return self.ir_builder.ir_constant_int(1)
        elif len(conditions) == 1:
            return conditions[0]
        elif len(conditions) == 2:
            return self.ir_builder.ir_logical_and(conditions[0], conditions[1])
        elif len(conditions) == 3:
            return self.ir_builder.ir_logical_and(conditions[0], self.ir_builder.ir_logical_and(conditions[1], conditions[2]))
        raise NotImplementedError()

    def get_break_condition_value(self):
        result = self.scopes[-1].get_breaking_condition()
        if result is None:
            return self.ir_builder.ir_constant_int(1)
        return result

    def get_return_condition_value(self):
        result = self.scopes[-1].get_returning_condition()
        if result is None:
            return self.ir_builder.ir_constant_int(1)
        return result

    def check_return_existence(self) -> bool:
        return self.scopes[-1].has_return_statement()

    def get_returns_with_conditions(self) -> List[Tuple[Value, IntegerValue]]:
        return self.scopes[-1].get_returns_with_conditions()

    def get_return_dtype(self):
        return self.scopes[-1].get_return_dtype()

    def return_value(self, value: Value):
        return self.scopes[-1].return_value(value, self.get_condition_value())

    def set_has_return(self):
        return self.scopes[-1].set_has_return()

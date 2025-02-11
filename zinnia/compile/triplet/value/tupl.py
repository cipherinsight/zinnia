import copy
from typing import Tuple, Union

from zinnia.compile.type_sys import DTDescriptor, TupleDTDescriptor
from zinnia.compile.triplet.value.value import Value
from zinnia.compile.triplet.store import CompositeTupleValueStore, ValueStore


class TupleValue(Value):
    def __init__(self, elements_type: Tuple[DTDescriptor, ...], values: Tuple[Value, ...]):
        super().__init__()
        assert len(elements_type) == len(values)
        for i, (dt, value) in enumerate(zip(elements_type, values)):
            assert dt == value.type(), f"Element {i} has type {value.type()} but expected {dt}"
        self._store = CompositeTupleValueStore(elements_type, tuple(v.into_value_store() for v in values))

    def type(self) -> DTDescriptor:
        return TupleDTDescriptor(self._store.elements_type)

    def values(self) -> Tuple[Value, ...]:
        from ..value_factory import ValueFactory
        return tuple(ValueFactory.from_value_store(v, self.type_locked()) for v in self._store.values)

    def types(self) -> Tuple[DTDescriptor, ...]:
        return self._store.elements_type

    def get(self, i: int) -> Value:
        assert 0 <= i < len(self._store.values)
        return self.values()[i]

    def set(self, i: int, value: Value):
        assert 0 <= i < len(self._store.values)
        assert self.types()[i] == value.type()
        self._store.values = self._store.values[:i] + (value.into_value_store(),) + self._store.values[i + 1:]

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.copy(self._store)
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.deepcopy(self._store)
        return new_instance

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['TupleValue', None]:
        if not isinstance(store, CompositeTupleValueStore):
            return None
        new_instance = cls.__new__(cls)
        new_instance._store = store
        new_instance.set_type_locked(type_locked)
        return new_instance

    def into_value_store(self) -> ValueStore:
        return self._store

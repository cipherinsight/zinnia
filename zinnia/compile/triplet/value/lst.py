import copy
from typing import List, Union

from zinnia.compile.type_sys import DTDescriptor, ListDTDescriptor
from zinnia.compile.triplet.value.value import Value
from zinnia.compile.triplet.store import CompositeListValueStore, ValueStore


class ListValue(Value):
    def __init__(self, elements_type: List[DTDescriptor], values: List[Value]):
        super().__init__()
        assert len(elements_type) == len(values)
        for i, (dt, value) in enumerate(zip(elements_type, values)):
            assert dt == value.type(), f"Element {i} has type {value.type()} but expected {dt}"
        self._store = CompositeListValueStore(elements_type, [v.into_value_store() for v in values])

    def type(self) -> DTDescriptor:
        return ListDTDescriptor(self._store.elements_type)

    def values(self) -> List[Value]:
        from ..value_factory import ValueFactory
        return [ValueFactory.from_value_store(v, self.type_locked()) for v in self._store.values]

    def types(self) -> List[DTDescriptor]:
        return self._store.elements_type

    def get(self, i: int) -> Value:
        assert 0 <= i < len(self._store.values)
        return self.values()[i]

    def set(self, i: int, value: Value):
        assert 0 <= i < len(self._store.values)
        assert self.types()[i] == value.type()
        self._store.values = self._store.values[:i] + [value.into_value_store()] + self._store.values[i + 1:]

    def assign(self, value: 'ListValue') -> 'ListValue':
        assert value.__class__ == self.__class__
        if self.type_locked():
            assert value.type() == self.type()
        self._store.assign(value._store)
        return self

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.copy(self._store)
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.deepcopy(self._store)
        return new_instance

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['ListValue', None]:
        if not isinstance(store, CompositeListValueStore):
            return None
        new_instance = cls.__new__(cls)
        new_instance._store = store
        new_instance.set_type_locked(type_locked)
        return new_instance

    def into_value_store(self) -> ValueStore:
        return self._store

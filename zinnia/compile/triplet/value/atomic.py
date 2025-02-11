from typing import Union

from zinnia.compile.triplet.value.value import Value
from zinnia.compile.triplet.store import ValueTriplet, ValueStore
from zinnia.compile.type_sys import DTDescriptor


class AtomicValue(Value):
    def __init__(self, triplet: ValueTriplet, type_locked: bool = False):
        super().__init__(type_locked)
        self._triplet = triplet

    def type(self) -> DTDescriptor:
        return self._triplet.get_t()

    def assign(self, value: 'AtomicValue') -> 'AtomicValue':
        assert value.__class__ == self.__class__
        if self.type_locked():
            assert value._triplet.get_s() == self._triplet.get_s()
        self._triplet.assign(value._triplet)
        return self

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['AtomicValue', None]:
        raise NotImplementedError()

    def into_value_store(self) -> ValueStore:
        return self._triplet

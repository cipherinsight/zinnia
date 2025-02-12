from typing import Union

from zinnia.compile.triplet.value.atomic import AtomicValue
from zinnia.compile.type_sys import NumberDTDescriptor
from zinnia.compile.triplet.store import ValueTriplet, ValueStore


class NumberValue(AtomicValue):
    def __init__(self, triplet: ValueTriplet):
        super().__init__(triplet)

    def val(self) -> int | float | None:
        return self._triplet.get_s()

    def ptr(self) -> int | None:
        return self._triplet.get_v()

    def type(self) -> NumberDTDescriptor:
        assert isinstance(self._triplet.get_t(), NumberDTDescriptor)
        return self._triplet.get_t()

    def assign(self, value: 'NumberValue', force: bool = False) -> 'NumberValue':
        if self.type_locked():
            assert force or value._triplet.get_s() == self._triplet.get_s()
        self._triplet.assign(value._triplet)
        return self

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['NumberValue', None]:
        raise NotImplementedError()

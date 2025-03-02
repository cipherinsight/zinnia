from typing import Union

from zinnia.compile.triplet.value.number import NumberValue
from zinnia.compile.type_sys import IntegerDTDescriptor, IntegerType
from zinnia.compile.triplet.store import ValueTriplet, ValueStore


class IntegerValue(NumberValue):
    def __init__(self, value: int | None, ptr: int | None, dt=IntegerDTDescriptor()):
        super().__init__(ValueTriplet(ptr, value, dt))

    def val(self) -> int | None:
        return super().val()

    def ptr(self) -> int | None:
        return super().ptr()

    def __copy__(self):
        return self.__class__(self._triplet.get_s(), self._triplet.get_v())

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['IntegerValue', None]:
        if not isinstance(store, ValueTriplet) or store.get_t() != IntegerType:
            return None
        value = IntegerValue(store.get_s(), store.get_v())
        value.set_type_locked(type_locked)
        return value

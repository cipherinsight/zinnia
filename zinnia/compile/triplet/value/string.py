from typing import Union

from zinnia.compile.triplet.value.atomic import AtomicValue
from zinnia.compile.type_sys import StringDTDescriptor, StringType
from zinnia.compile.triplet.store import ValueTriplet, ValueStore


class StringValue(AtomicValue):
    def __init__(self, value: str, ptr: int):
        super().__init__(ValueTriplet(ptr, value, StringDTDescriptor()))

    def val(self) -> str:
        return self._triplet.get_s()

    def ptr(self) -> int:
        return self._triplet.get_v()

    def __copy__(self):
        return self.__class__(self.val(), self.ptr())

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['StringValue', None]:
        if not isinstance(store, ValueTriplet) or store.get_t() != StringType:
            return None
        value = StringValue(store.get_s(), store.get_v())
        value.set_type_locked(type_locked)
        return value

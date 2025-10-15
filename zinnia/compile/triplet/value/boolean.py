from typing import Union, List

from z3 import z3

from zinnia.compile.triplet.value.integer import IntegerValue
from zinnia.compile.type_sys import BooleanDTDescriptor, BooleanType
from zinnia.compile.triplet.store import ValueTriplet, ValueStore


class BooleanValue(IntegerValue):
    def __init__(self, value: bool | None, ptr: int | None, z3e = None, rel: List | None = None):
        super().__init__(value, ptr, z3e, rel, BooleanDTDescriptor())
        self.z3_sym = z3.Bool(f'bool_{self.ptr}')
        if z3e is not None:
            self.z3_rel += [self.z3_sym == z3e]

    def val(self) -> bool | None:
        if super().val() is not None:
            return bool(super().val())
        return None

    def ptr(self) -> int | None:
        return super().ptr()

    def __copy__(self):
        return self.__class__(self._triplet.get_s(), self._triplet.get_v())

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['BooleanValue', None]:
        if not isinstance(store, ValueTriplet) or store.get_t() != BooleanType:
            return None
        value = BooleanValue(store.get_s(), store.get_v())
        value.set_type_locked(type_locked)
        return value

from typing import Union, List

from z3 import z3

from zinnia.compile.triplet.value.number import NumberValue
from zinnia.compile.type_sys import IntegerDTDescriptor, IntegerType
from zinnia.compile.triplet.store import ValueTriplet, ValueStore


class IntegerValue(NumberValue):
    def __init__(self, value: int | None, ptr: int | None, z3e = None, rel: List | None = None, dt=IntegerDTDescriptor()):
        super().__init__(ValueTriplet(ptr, value, dt))
        self.z3_sym = z3.Int(f'int_{int(ptr)}')
        self.z3_val = rel is None and z3e is None
        self.z3_expr = z3e
        self.z3_rel = [] if rel is None else rel[:-10]
        if z3e is not None and isinstance(dt, IntegerDTDescriptor):
            self.z3_rel += [self.z3_sym == z3e]

    def val(self) -> int | None:
        if super().val() is not None:
            return super().val()
        if not self.z3_val:
            self.z3_val = True
            resolution = self.smt_resolve_expr(self.z3_sym, self.z3_rel)
            if resolution is not None:
                self.val = int(resolution)
                return self.val
        return None

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

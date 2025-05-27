import copy
from typing import Any

from zinnia.compile.triplet.store.store import ValueStore
from zinnia.compile.type_sys import DTDescriptor


class ValueTriplet(ValueStore):
    def __init__(self, _v: Any, _s: Any, _t: DTDescriptor):
        self._v = _v  # value
        self._s = _s  # static inference value
        self._t = _t  # type

    def get_v(self) -> Any:
        return self._v

    def get_s(self) -> Any:
        return self._s

    def get_t(self) -> DTDescriptor:
        return self._t

    def set_v(self, _v: Any):
        self._v = _v

    def set_s(self, _s: Any):
        self._s = _s

    def set_t(self, _t: DTDescriptor):
        self._t = _t

    def __copy__(self):
        return self.__class__(
            copy.copy(self.get_v()),
            copy.copy(self.get_s()),
            copy.copy(self.get_t())
        )

    def __deepcopy__(self, memo):
        return self.__class__(
            copy.deepcopy(self.get_v(), memo),
            copy.deepcopy(self.get_s(), memo),
            copy.deepcopy(self.get_t(), memo)
        )

    def assign(self, value: 'ValueTriplet') -> 'ValueTriplet':
        self.set_v(value.get_v())
        self.set_s(value.get_s())
        self.set_t(value.get_t())
        return self

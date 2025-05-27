from typing import Union

from zinnia.compile.triplet.store import ValueTriplet, ValueStore
from zinnia.compile.triplet.value.atomic import AtomicValue
from zinnia.compile.type_sys import DTDescriptor, ClassDTDescriptor


class ClassValue(AtomicValue):
    def __init__(self, value: DTDescriptor):
        super().__init__(ValueTriplet(None, value, ClassDTDescriptor()))

    def val(self) -> DTDescriptor:
        return self._triplet.get_s()

    def __copy__(self):
        return self.__class__(self.val())

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['ClassValue', None]:
        if not isinstance(store, ValueTriplet) or not isinstance(store.get_t(), ClassDTDescriptor):
            return None
        value = ClassValue(store.get_s())
        value.set_type_locked(type_locked)
        return value

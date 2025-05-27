from typing import Union

from zinnia.compile.triplet.store import ValueTriplet, ValueStore
from zinnia.compile.triplet.value.atomic import AtomicValue
from zinnia.compile.type_sys import NoneDTDescriptor, NoneType


class NoneValue(AtomicValue):
    def __init__(self):
        super().__init__(ValueTriplet(None, None, NoneDTDescriptor()))

    def __copy__(self):
        return self.__class__()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['NoneValue', None]:
        if not isinstance(store, ValueTriplet) or store.get_t() != NoneType:
            return None
        value = NoneValue()
        value.set_type_locked(type_locked)
        return value

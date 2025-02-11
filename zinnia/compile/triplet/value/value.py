from typing import Union

from zinnia.compile.triplet.store import ValueStore
from zinnia.compile.type_sys import DTDescriptor


class Value:
    def __init__(self, type_locked: bool = False):
        self._type_locked = type_locked

    def type(self) -> DTDescriptor:
        raise NotImplementedError()

    def assign(self, value: 'Value') -> 'Value':
        raise NotImplementedError()

    def type_locked(self) -> bool:
        return self._type_locked

    def set_type_locked(self, value: bool = True):
        self._type_locked = value

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['Value', None]:
        raise NotImplementedError()

    def into_value_store(self) -> ValueStore:
        raise NotImplementedError()

import copy

from zinnia import Tuple
from zinnia.compile.triplet.store import ValueStore
from zinnia.compile.type_sys import DTDescriptor


class CompositeTupleValueStore(ValueStore):
    def __init__(self, elements_type: Tuple[DTDescriptor, ...], values: Tuple[ValueStore, ...]):
        self.elements_type = elements_type
        self.values = values

    def assign(self, other: 'CompositeTupleValueStore') -> 'CompositeTupleValueStore':
        assert other.__class__ == self.__class__
        self.elements_type = other.elements_type
        self.values = other.values
        return self

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.elements_type = copy.copy(self.elements_type)
        new_instance.values = copy.copy(self.values)
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.elements_type = copy.deepcopy(self.elements_type)
        new_instance.values = copy.deepcopy(self.values)
        return new_instance

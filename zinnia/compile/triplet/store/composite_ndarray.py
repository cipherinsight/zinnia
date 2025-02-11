import copy

from zinnia.compile.triplet.store.store import ValueStore
from zinnia.compile.type_sys import NDArrayDTDescriptor
from zinnia.internal.internal_ndarray import InternalNDArray


class CompositeNDArrayValueStore(ValueStore):
    def __init__(self, data_type: NDArrayDTDescriptor, ndarray: InternalNDArray):
        self.ndarray = ndarray
        self.data_type = data_type

    def assign(self, other: 'CompositeNDArrayValueStore') -> 'CompositeNDArrayValueStore':
        assert other.__class__ == self.__class__
        self.ndarray = other.ndarray
        self.data_type = other.data_type
        return self

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.ndarray = copy.copy(self.ndarray)
        new_instance.data_type = copy.copy(self.data_type)
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.ndarray = copy.deepcopy(self.ndarray)
        new_instance.data_type = copy.deepcopy(self.data_type)
        return new_instance

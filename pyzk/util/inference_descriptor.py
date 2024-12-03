import copy
from typing import Tuple, Any

from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, NumberDTDescriptor, TupleDTDescriptor, \
    NoneDTDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper


class InferenceDescriptor:
    def __init__(self, dt: DTDescriptor):
        self.dt = dt

    def __new__(cls, *args, **kwargs):
        if cls is InferenceDescriptor:
            raise TypeError(f"<InferenceDescriptor> must be subclassed.")
        return object.__new__(cls)

    def get(self) -> Any:
        raise NotImplementedError()

    def type(self) -> DTDescriptor:
        return self.dt

    def set(self, value: Any) -> 'InferenceDescriptor':
        raise NotImplementedError()


class NDArrayInferenceDescriptor(InferenceDescriptor):
    def __init__(self, shape: Tuple[int, ...], value: NDArrayHelper):
        super().__init__(NDArrayDTDescriptor(shape))
        self.value = value

    def get(self) -> NDArrayHelper:
        return self.value

    def set(self, value: NDArrayHelper) -> 'NDArrayInferenceDescriptor':
        self.value = value
        return self

    def shape(self):
        assert isinstance(self.dt, NDArrayDTDescriptor)
        return self.dt.shape

    @staticmethod
    def new_instance(src: 'NDArrayInferenceDescriptor', value: NDArrayHelper) -> 'NDArrayInferenceDescriptor':
        return copy.copy(src).set(value)


class NumberInferenceDescriptor(InferenceDescriptor):
    def __init__(self, value: int | None = None):
        super().__init__(NumberDTDescriptor())
        self.value = value

    def get(self) -> int | None:
        return self.value

    def set(self, value: int | None) -> 'NumberInferenceDescriptor':
        self.value = value
        return self


class NoneInferenceDescriptor(InferenceDescriptor):
    def __init__(self):
        super().__init__(NoneDTDescriptor())

    def get(self) -> int | None:
        raise NotImplementedError()

    def set(self, value: int | None) -> 'NoneInferenceDescriptor':
        raise NotImplementedError()


class TupleInferenceDescriptor(InferenceDescriptor):
    def __init__(self, length: int, value: Tuple):
        super().__init__(TupleDTDescriptor(length))
        assert len(value) == length
        self.value = value

    def length(self) -> int:
        assert isinstance(self.dt, TupleDTDescriptor)
        return self.dt.length

    def get(self) -> Tuple:
        return self.value

    def set(self, value: Tuple) -> 'TupleInferenceDescriptor':
        self.value = value
        return self

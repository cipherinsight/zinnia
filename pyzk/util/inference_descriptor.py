import copy
from typing import Tuple, Any

from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, NumberDTDescriptor, TupleDTDescriptor, \
    NoneDTDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper


NumberInferenceValue = int | None
NDArrayInferenceValue = NDArrayHelper
TupleInferenceValue = tuple
NoneInferenceValue = type(None)


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
    def __init__(self, shape: Tuple[int, ...], value: NDArrayInferenceValue):
        super().__init__(NDArrayDTDescriptor(shape))
        self.value = value

    def get(self) -> NDArrayInferenceValue:
        return self.value

    def set(self, value: NDArrayInferenceValue) -> 'NDArrayInferenceDescriptor':
        self.value = value
        return self

    def shape(self):
        assert isinstance(self.dt, NDArrayDTDescriptor)
        return self.dt.shape

    @staticmethod
    def new_instance(src: 'NDArrayInferenceDescriptor', value: NDArrayHelper) -> 'NDArrayInferenceDescriptor':
        return copy.copy(src).set(value)


class NumberInferenceDescriptor(InferenceDescriptor):
    def __init__(self, value: NumberInferenceValue):
        super().__init__(NumberDTDescriptor())
        self.value = value

    def get(self) -> NumberInferenceValue:
        return self.value

    def set(self, value: NumberInferenceValue) -> 'NumberInferenceDescriptor':
        self.value = value
        return self


class NoneInferenceDescriptor(InferenceDescriptor):
    def __init__(self):
        super().__init__(NoneDTDescriptor())

    def get(self) -> NoneInferenceValue:
        return None

    def set(self, value: NoneInferenceValue) -> 'NoneInferenceDescriptor':
        assert value is None
        return self


class TupleInferenceDescriptor(InferenceDescriptor):
    def __init__(self, length: int, value: TupleInferenceValue):
        super().__init__(TupleDTDescriptor(length))
        assert len(value) == length
        self.value = value

    def length(self) -> int:
        assert isinstance(self.dt, TupleDTDescriptor)
        return self.dt.length

    def get(self) -> TupleInferenceValue:
        return self.value

    def set(self, value: TupleInferenceValue) -> 'TupleInferenceDescriptor':
        self.value = value
        return self

from typing import Tuple, Any

from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, IntegerDTDescriptor, TupleDTDescriptor, \
    NoneDTDescriptor, ClassDTDescriptor, FloatDTDescriptor, NumberDTDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper


IntegerInferenceValue = int | None
FloatInferenceValue = float | None
NDArrayInferenceValue = NDArrayHelper
TupleInferenceValue = tuple
NoneInferenceValue = type(None)
ClassInferenceValue = DTDescriptor


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

    def copy_reset(self) -> 'InferenceDescriptor':
        raise NotImplementedError()


class NDArrayInferenceDescriptor(InferenceDescriptor):
    def __init__(self, shape: Tuple[int, ...], dtype: DTDescriptor, value: NDArrayInferenceValue):
        super().__init__(NDArrayDTDescriptor(shape, dtype))
        self.value = value

    def get(self) -> NDArrayInferenceValue:
        return self.value

    def set(self, value: NDArrayInferenceValue) -> 'NDArrayInferenceDescriptor':
        self.value = value
        return self

    def shape(self):
        assert isinstance(self.dt, NDArrayDTDescriptor)
        return self.dt.shape

    def dtype(self):
        assert isinstance(self.dt, NDArrayDTDescriptor)
        return self.dt.dtype

    def copy_reset(self) -> 'NDArrayInferenceDescriptor':
        return NDArrayInferenceDescriptor(self.shape(), self.dtype(), NDArrayInferenceValue.fill(self.shape(), lambda: None))


class NumberInferenceDescriptor(InferenceDescriptor):
    def __init__(self, dt: DTDescriptor):
        super().__init__(dt)

    def __new__(cls, *args, **kwargs):
        if cls is InferenceDescriptor:
            raise TypeError(f"<NumberInferenceDescriptor> must be subclassed.")
        return object.__new__(cls)

    def copy_reset(self) -> 'NumberInferenceDescriptor':
        raise NotImplementedError()


class IntegerInferenceDescriptor(NumberInferenceDescriptor):
    def __init__(self, value: IntegerInferenceValue):
        super().__init__(IntegerDTDescriptor())
        self.value = value

    def get(self) -> IntegerInferenceValue:
        return self.value

    def set(self, value: IntegerInferenceValue) -> 'IntegerInferenceDescriptor':
        self.value = value
        return self

    def copy_reset(self) -> 'IntegerInferenceDescriptor':
        return IntegerInferenceDescriptor(None)


class FloatInferenceDescriptor(NumberInferenceDescriptor):
    def __init__(self, value: FloatInferenceValue):
        super().__init__(FloatDTDescriptor())
        self.value = value

    def get(self) -> FloatInferenceValue:
        return self.value

    def set(self, value: FloatInferenceValue) -> 'FloatInferenceDescriptor':
        self.value = value
        return self

    def copy_reset(self) -> 'FloatInferenceDescriptor':
        return FloatInferenceDescriptor(None)


class NoneInferenceDescriptor(InferenceDescriptor):
    def __init__(self):
        super().__init__(NoneDTDescriptor())

    def get(self) -> NoneInferenceValue:
        return None

    def set(self, value: NoneInferenceValue) -> 'NoneInferenceDescriptor':
        assert value is None
        return self

    def copy_reset(self) -> 'NoneInferenceDescriptor':
        return NoneInferenceDescriptor()


class ClassInferenceDescriptor(InferenceDescriptor):
    def __init__(self, cls: DTDescriptor):
        super().__init__(ClassDTDescriptor())
        self.cls = cls

    def get(self) -> ClassInferenceValue:
        return self.cls

    def set(self, value: ClassInferenceValue) -> 'ClassInferenceDescriptor':
        assert isinstance(value, ClassInferenceValue)
        return self

    def copy_reset(self) -> 'ClassInferenceDescriptor':
        raise NotImplementedError()


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

    def copy_reset(self) -> 'TupleInferenceDescriptor':
        return TupleInferenceDescriptor(self.length(), tuple(None for _ in range(self.length())))

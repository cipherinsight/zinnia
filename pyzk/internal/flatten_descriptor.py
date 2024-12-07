from typing import Tuple, Any, Optional

from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, NumberDTDescriptor, TupleDTDescriptor, \
    NoneDTDescriptor
from pyzk.internal.inference_descriptor import NDArrayInferenceValue, NumberInferenceValue, NoneInferenceValue, \
    TupleInferenceValue
from pyzk.algo.ndarray_helper import NDArrayHelper


NumberFlattenValue = int
NDArrayFlattenValue = NDArrayHelper
TupleFlattenValue = tuple
NoneFlattenValue = type(None)


class FlattenDescriptor:
    def __init__(self, dt: DTDescriptor):
        self.dt = dt

    def __new__(cls, *args, **kwargs):
        if cls is FlattenDescriptor:
            raise TypeError(f"<FlattenDescriptor> must be subclassed.")
        return object.__new__(cls)

    def val(self) -> Any:
        raise NotImplementedError()

    def ptr(self) -> Any:
        raise NotImplementedError()

    def type(self) -> DTDescriptor:
        return self.dt

    def set_ptr(self, value: Any) -> 'FlattenDescriptor':
        raise NotImplementedError()

    def set_val(self, value: Any) -> 'FlattenDescriptor':
        raise NotImplementedError()


class NDArrayFlattenDescriptor(FlattenDescriptor):
    def __init__(self, shape: Tuple[int, ...], ptrs: NDArrayFlattenValue, value: Optional[NDArrayInferenceValue] = None):
        super().__init__(NDArrayDTDescriptor(shape))
        self.value = value
        self._ptrs = ptrs

    def val(self) -> NDArrayInferenceValue:
        return self.value

    def ptr(self) -> NDArrayFlattenValue:
        return self._ptrs

    def set_val(self, value: Optional[NDArrayInferenceValue] = None) -> 'NDArrayFlattenDescriptor':
        self.value = value
        return self

    def set_ptr(self, ptrs: NDArrayFlattenValue) -> 'NDArrayFlattenDescriptor':
        self._ptrs = ptrs
        return self

    def shape(self):
        assert isinstance(self.dt, NDArrayDTDescriptor)
        return self.dt.shape


class NumberFlattenDescriptor(FlattenDescriptor):
    def __init__(self, ptr: NumberFlattenValue, value: Optional[NumberInferenceValue] = None):
        super().__init__(NumberDTDescriptor())
        self.value = value
        self._ptr = ptr

    def val(self) -> NumberInferenceValue:
        return self.value

    def ptr(self) -> NumberFlattenValue:
        return self._ptr

    def set_val(self, value: Optional[NumberInferenceValue] = None) -> 'NumberFlattenDescriptor':
        self.value = value
        return self

    def set_ptr(self, ptr: NumberFlattenValue) -> 'NumberFlattenDescriptor':
        self._ptr = ptr
        return self


class NoneFlattenDescriptor(FlattenDescriptor):
    def __init__(self):
        super().__init__(NoneDTDescriptor())

    def val(self) -> NoneInferenceValue:
        return None

    def ptr(self) -> NoneFlattenValue:
        raise NotImplementedError()

    def set_val(self, value: NoneInferenceValue) -> 'NoneFlattenDescriptor':
        assert value is None
        return self

    def set_ptr(self, ptr: NoneFlattenValue) -> 'NoneFlattenDescriptor':
        raise NotImplementedError()


class TupleFlattenDescriptor(FlattenDescriptor):
    def __init__(self, length: int, ptrs: TupleFlattenValue, value: Optional[TupleInferenceValue] = None):
        super().__init__(TupleDTDescriptor(length))
        assert value is None or len(value) == length
        self.value = value
        self._ptrs = ptrs

    def length(self) -> int:
        assert isinstance(self.dt, TupleDTDescriptor)
        return self.dt.length

    def val(self) -> TupleInferenceValue:
        return self.value

    def ptr(self) -> TupleFlattenValue:
        return self._ptrs

    def set_val(self, value: Optional[TupleInferenceValue] = None) -> 'TupleFlattenDescriptor':
        self.value = value
        return self

    def set_ptr(self, ptrs: TupleFlattenValue) -> 'TupleFlattenDescriptor':
        self._ptrs = ptrs
        return self

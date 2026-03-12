import copy
from typing import List, Tuple, Union

from zinnia.compile.triplet.store import CompositeNDArrayValueStore, ValueStore
from zinnia.compile.triplet.value.ndarray import NDArrayValue
from zinnia.compile.triplet.value.integer import IntegerValue
from zinnia.compile.triplet.value.number import NumberValue
from zinnia.compile.type_sys import DynamicNDArrayDTDescriptor, NumberDTDescriptor, DTDescriptor
from zinnia.internal.internal_ndarray import InternalNDArray


class DynamicNDArrayValue(NDArrayValue):
    @staticmethod
    def _default_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(shape) == 0:
            return tuple()
        stride = 1
        out = []
        for dim in reversed(shape):
            out.append(stride)
            stride *= dim
        out.reverse()
        return tuple(out)

    def __init__(
        self,
        max_length: int,
        max_rank: int,
        dtype: NumberDTDescriptor,
        value: InternalNDArray,
        logical_shape: Tuple[int, ...] | None = None,
        logical_offset: int = 0,
        logical_strides: Tuple[int, ...] | None = None,
        runtime_logical_length: IntegerValue | None = None,
        runtime_rank: IntegerValue | None = None,
        runtime_shape_entries: List[IntegerValue] | None = None,
        runtime_stride_entries: List[IntegerValue] | None = None,
        runtime_offset: IntegerValue | None = None,
    ):
        super(NDArrayValue, self).__init__()
        self._store = CompositeNDArrayValueStore(
            DynamicNDArrayDTDescriptor(dtype, max_length, max_rank),
            value.unary(lambda x: x.into_value_store()),
        )
        if logical_shape is None:
            logical_shape = (max_length,)
        if logical_strides is None:
            logical_strides = self._default_strides(logical_shape)
        self._logical_shape = tuple(logical_shape)
        self._logical_offset = logical_offset
        self._logical_strides = tuple(logical_strides)
        self._runtime_logical_length = runtime_logical_length or IntegerValue(max_length, None)
        default_rank = len(logical_shape)
        self._runtime_rank = runtime_rank or IntegerValue(default_rank, None)

        if runtime_shape_entries is None:
            padded_shape = [1 for _ in range(max_rank)]
            start = max_rank - len(logical_shape)
            for i, dim in enumerate(logical_shape):
                if 0 <= start + i < max_rank:
                    if len(logical_shape) == 1 and i == len(logical_shape) - 1:
                        padded_shape[start + i] = self._runtime_logical_length.val(None) if self._runtime_logical_length.val(None) is not None else dim
                    else:
                        padded_shape[start + i] = dim
            runtime_shape_entries = [IntegerValue(v, None) for v in padded_shape]
            if len(logical_shape) == 1 and max_rank > 0:
                runtime_shape_entries[-1] = self._runtime_logical_length
        self._runtime_shape_entries = list(runtime_shape_entries)

        if runtime_stride_entries is None:
            padded_strides = [1 for _ in range(max_rank)]
            start = max_rank - len(logical_shape)
            for i, s in enumerate(logical_strides):
                if 0 <= start + i < max_rank:
                    padded_strides[start + i] = s
            runtime_stride_entries = [IntegerValue(v, None) for v in padded_strides]
        self._runtime_stride_entries = list(runtime_stride_entries)

        self._runtime_offset = runtime_offset or IntegerValue(logical_offset, None)

    def type(self) -> DTDescriptor:
        return self._store.data_type

    def max_length(self) -> int:
        dt = self._store.data_type
        assert isinstance(dt, DynamicNDArrayDTDescriptor)
        return dt.max_length

    def max_rank(self) -> int:
        dt = self._store.data_type
        assert isinstance(dt, DynamicNDArrayDTDescriptor)
        return dt.max_rank

    def logical_shape(self) -> Tuple[int, ...]:
        return self._logical_shape

    def logical_offset(self) -> int:
        return self._logical_offset

    def logical_strides(self) -> Tuple[int, ...]:
        return self._logical_strides

    def runtime_logical_length(self) -> IntegerValue:
        return self._runtime_logical_length

    def runtime_rank(self) -> IntegerValue:
        return self._runtime_rank

    def runtime_shape_entries(self) -> List[IntegerValue]:
        return list(self._runtime_shape_entries)

    def runtime_stride_entries(self) -> List[IntegerValue]:
        return list(self._runtime_stride_entries)

    def runtime_offset(self) -> IntegerValue:
        return self._runtime_offset

    @staticmethod
    def from_max_bounds_and_vector(
        max_length: int,
        max_rank: int,
        dtype: NumberDTDescriptor,
        values: List[NumberValue],
        logical_shape: Tuple[int, ...] | None = None,
        logical_offset: int = 0,
        logical_strides: Tuple[int, ...] | None = None,
        runtime_logical_length: IntegerValue | None = None,
        runtime_rank: IntegerValue | None = None,
        runtime_shape_entries: List[IntegerValue] | None = None,
        runtime_stride_entries: List[IntegerValue] | None = None,
        runtime_offset: IntegerValue | None = None,
    ) -> "DynamicNDArrayValue":
        return DynamicNDArrayValue(
            max_length,
            max_rank,
            dtype,
            InternalNDArray.from_1d_values_and_shape(values, (max_length,)),
            logical_shape=logical_shape,
            logical_offset=logical_offset,
            logical_strides=logical_strides,
            runtime_logical_length=runtime_logical_length,
            runtime_rank=runtime_rank,
            runtime_shape_entries=runtime_shape_entries,
            runtime_stride_entries=runtime_stride_entries,
            runtime_offset=runtime_offset,
        )

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union["DynamicNDArrayValue", None]:
        if not isinstance(store, CompositeNDArrayValueStore):
            return None
        if not isinstance(store.data_type, DynamicNDArrayDTDescriptor):
            return None
        new_instance = cls.__new__(cls)
        new_instance._store = store
        new_instance._logical_shape = (store.data_type.max_length,)
        new_instance._logical_offset = 0
        new_instance._logical_strides = cls._default_strides(new_instance._logical_shape)
        new_instance._runtime_logical_length = IntegerValue(store.data_type.max_length, None)
        new_instance._runtime_rank = IntegerValue(1, None)
        new_instance._runtime_shape_entries = [IntegerValue(1, None) for _ in range(store.data_type.max_rank)]
        if store.data_type.max_rank > 0:
            new_instance._runtime_shape_entries[-1] = new_instance._runtime_logical_length
        new_instance._runtime_stride_entries = [IntegerValue(1, None) for _ in range(store.data_type.max_rank)]
        new_instance._runtime_offset = IntegerValue(0, None)
        new_instance.set_type_locked(type_locked)
        return new_instance

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.copy(self._store)
        new_instance._logical_shape = self._logical_shape
        new_instance._logical_offset = self._logical_offset
        new_instance._logical_strides = self._logical_strides
        new_instance._runtime_logical_length = self._runtime_logical_length
        new_instance._runtime_rank = self._runtime_rank
        new_instance._runtime_shape_entries = self._runtime_shape_entries
        new_instance._runtime_stride_entries = self._runtime_stride_entries
        new_instance._runtime_offset = self._runtime_offset
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.deepcopy(self._store)
        new_instance._logical_shape = self._logical_shape
        new_instance._logical_offset = self._logical_offset
        new_instance._logical_strides = self._logical_strides
        new_instance._runtime_logical_length = self._runtime_logical_length
        new_instance._runtime_rank = self._runtime_rank
        new_instance._runtime_shape_entries = self._runtime_shape_entries
        new_instance._runtime_stride_entries = self._runtime_stride_entries
        new_instance._runtime_offset = self._runtime_offset
        return new_instance

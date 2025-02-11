import copy
from typing import Any, Tuple, Callable, List, Union

from zinnia.compile.triplet.store import CompositeNDArrayValueStore, ValueStore
from zinnia.compile.triplet.value.value import Value
from zinnia.compile.triplet.value.number import NumberValue
from zinnia.internal.internal_ndarray import InternalNDArray
from zinnia.compile.type_sys import NumberDTDescriptor, NDArrayDTDescriptor, DTDescriptor


class NDArrayValue(Value):
    def __init__(self, shape: Tuple[int, ...], dtype: NumberDTDescriptor, value: InternalNDArray):
        super().__init__()
        self._store = CompositeNDArrayValueStore(
            NDArrayDTDescriptor(shape, dtype),
            value.unary(lambda x: x.into_value_store())
        )

    def type(self) -> DTDescriptor:
        return self._store.data_type

    def dtype(self) -> NumberDTDescriptor:
        return self._store.data_type.dtype

    def shape(self) -> Tuple[int, ...]:
        return self._store.data_type.shape

    def get(self) -> InternalNDArray:
        from ..value_factory import ValueFactory
        return self._store.ndarray.unary(lambda x: ValueFactory.from_value_store(x))

    @staticmethod
    def from_number(value: 'NumberValue') -> 'NDArrayValue':
        return NDArrayValue((), value.type(), InternalNDArray((), value))

    @staticmethod
    def from_shape_and_vector(shape: Tuple[int, ...], dtype: NumberDTDescriptor, values: List[NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(shape, dtype, InternalNDArray.from_1d_values_and_shape(values, shape))

    @staticmethod
    def binary(lhs: 'NDArrayValue', rhs: 'NDArrayValue', dtype: NumberDTDescriptor, op: Callable[[NumberValue, NumberValue], NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(
            lhs.shape(), dtype,
            lhs.get()
            .binary(rhs.get(), op)
        )

    @staticmethod
    def fill(shape: Tuple[int, ...], dtype: NumberDTDescriptor, filler: Callable[[], Any]):
        return NDArrayValue(shape, dtype, InternalNDArray.fill(shape, filler))

    @staticmethod
    def binary_broadcast_compatible(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> bool:
        return InternalNDArray.binary_broadcast_compatible(lhs, rhs)

    @staticmethod
    def binary_broadcast(lhs: 'NDArrayValue', rhs: 'NDArrayValue') -> Tuple['NDArrayValue', 'NDArrayValue']:
        _lhs, _rhs = InternalNDArray.binary_broadcast(lhs.get(), rhs.get())
        return NDArrayValue(_lhs.shape, lhs.dtype(), _lhs), NDArrayValue(_rhs.shape, rhs.dtype(), _rhs)

    @staticmethod
    def matmul(
            lhs: 'NDArrayValue',
            rhs: 'NDArrayValue',
            dtype: NumberDTDescriptor,
            adder: Callable[[NumberValue, NumberValue], NumberValue],
            multiplier: Callable[[NumberValue, NumberValue], NumberValue],
            initializer: Callable[[], NumberValue]
    ) -> 'NDArrayValue':
        result = InternalNDArray.matmul(
            lhs.get(), rhs.get(), adder, multiplier, initializer
        )
        return NDArrayValue(result.shape, dtype, result)

    @staticmethod
    def matmul_compatible(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> bool:
        return InternalNDArray.matmul_shape_matches(lhs, rhs)

    @staticmethod
    def stack(dtype: NumberDTDescriptor, axis: int, values: List['NDArrayValue']) -> 'NDArrayValue':
        result = InternalNDArray.stack([x.get() for x in values], axis)
        return NDArrayValue(result.shape, dtype, result)

    @staticmethod
    def concatenate(dtype: NumberDTDescriptor, axis: int, values: List['NDArrayValue']) -> 'NDArrayValue':
        result = InternalNDArray.concatenate([x.get() for x in values], axis)
        return NDArrayValue(result.shape, dtype, result)

    def broadcast_to_compatible(self, shape: Tuple[int, ...]) -> bool:
        return InternalNDArray.directed_broadcast_compatible(self.shape(), shape)

    def broadcast_to(self, shape: Tuple[int, ...]) -> 'NDArrayValue':
        return NDArrayValue(shape, self.dtype(), InternalNDArray.directed_broadcast(self.get(), shape))

    def for_each(self, op: Callable[[Tuple[int, ...], NumberValue], NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(self.shape(), self.dtype(), self.get().for_each(op))

    def accumulate(
            self,
            axis: int,
            aggregator: Callable[[NumberValue, NumberValue, NumberValue, NumberValue],
            Tuple[NumberValue, NumberValue | None]],
            initial: Callable[[NumberValue], Tuple[NumberValue, NumberValue | None]],
            enpair: Callable[[NumberValue, int], Tuple[NumberValue, NumberValue | None]],
            depair: Callable[[NumberValue, NumberValue], NumberValue]
    ) -> 'NDArrayValue':
        result = self.get().accumulate(axis, aggregator, initial, enpair, depair)
        if isinstance(result, InternalNDArray):
            return NDArrayValue(result.shape, self.dtype(), result)
        return result

    def unary(self, dtype: NumberDTDescriptor, op: Callable[[NumberValue], NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(self.shape(), dtype, self.get().unary(op))

    def flattened_values(self) -> List[NumberValue]:
        return self.get().flatten()

    def tolist(self) -> List[NumberValue]:
        return self.get().tolist()

    def flatten(self) -> 'NDArrayValue':
        flattened_values = self.get().flatten()
        return NDArrayValue(
            (len(flattened_values),),
            self.dtype(),
            InternalNDArray((len(flattened_values),), flattened_values)
        )

    def transpose(self, axes: Tuple[int, ...]) -> 'NDArrayValue':
        internal_ndarray = self.get().transpose(axes)
        return NDArrayValue(
            internal_ndarray.shape,
            self.dtype(),
            internal_ndarray
        )

    def get_item(self, slicing_params: List[int | Tuple[int, int, int]]) -> Value:
        result = self.get().ndarray_get_item(slicing_params)
        if isinstance(result, InternalNDArray):
            return NDArrayValue(result.shape, self.dtype(), result)
        return result

    def set_item(self, slicing_params: List[int | Tuple[int, int, int]], val: Union['NDArrayValue', NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(self.shape(), self.dtype(), self.get().ndarray_set_item(
            slicing_params,
            val.get() if isinstance(val, NDArrayValue) else val,
            lambda x, y: x.assign(y)
        ))

    def assign(self, value: 'NDArrayValue') -> 'NDArrayValue':
        assert value.__class__ == self.__class__
        if self.type_locked():
            assert value.shape() == self.shape() and value.dtype() == self.dtype()
        self._store.assign(value._store)
        return self

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.copy(self._store)
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._store = copy.deepcopy(self._store)
        return new_instance

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['Value', None]:
        if not isinstance(store, CompositeNDArrayValueStore):
            return None
        new_instance = cls.__new__(cls)
        new_instance._store = store
        new_instance.set_type_locked(type_locked)
        return new_instance

    def into_value_store(self) -> ValueStore:
        return self._store

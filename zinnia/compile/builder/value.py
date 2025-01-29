import copy
from typing import Any, Tuple, Callable, List, Union

from zinnia.internal.internal_ndarray import InternalNDArray
from zinnia.compile.type_sys import DTDescriptor, NumberDTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, \
    NDArrayDTDescriptor, TupleDTDescriptor, ListDTDescriptor, NoneDTDescriptor, ClassDTDescriptor, StringDTDescriptor, \
    PoseidonHashedDTDescriptor


class Value:
    def __init__(self, dt: DTDescriptor):
        self._dt = dt

    def type(self) -> DTDescriptor:
        return self._dt

    def assign(self, value: 'Value') -> 'Value':
        raise NotImplementedError()

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        return self.__copy__()


class NumberValue(Value):
    def __init__(self, dt: NumberDTDescriptor, value: Any, ptr: int | None):
        super().__init__(dt)
        self._value = value
        self._ptr = ptr

    def val(self) -> int | float | None:
        return self._value

    def ptr(self) -> int | None:
        return self._ptr

    def type(self) -> NumberDTDescriptor:
        assert isinstance(self._dt, NumberDTDescriptor)
        return self._dt

    def assign(self, value: 'NumberValue') -> 'NumberValue':
        assert self.type() == value.type()
        self._value = value._value
        self._ptr = value._ptr
        return self

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()


class IntegerValue(NumberValue):
    def __init__(self, value: int | None, ptr: int | None):
        super().__init__(IntegerDTDescriptor(), value, ptr)

    def val(self) -> int | None:
        return super().val()

    def ptr(self) -> int | None:
        return super().ptr()

    def assign(self, value: 'IntegerValue') -> 'IntegerValue':
        return super().assign(value)

    def __copy__(self):
        return self.__class__(self._value, self._ptr)

    def __deepcopy__(self, memo):
        return self.__copy__()


class FloatValue(NumberValue):
    def __init__(self, value: float | None, ptr: int | None):
        super().__init__(FloatDTDescriptor(), value, ptr)

    def val(self) -> float | None:
        return super().val()

    def ptr(self) -> int | None:
        return super().ptr()

    def assign(self, value: 'IntegerValue') -> 'IntegerValue':
        return super().assign(value)

    def __copy__(self):
        return self.__class__(self._value, self._ptr)

    def __deepcopy__(self, memo):
        return self.__copy__()


class NDArrayValue(Value):
    def __init__(self, shape: Tuple[int, ...], dtype: NumberDTDescriptor, value: InternalNDArray):
        super().__init__(NDArrayDTDescriptor(shape, dtype))
        self._value = value

    def dtype(self) -> NumberDTDescriptor:
        assert isinstance(self._dt, NDArrayDTDescriptor)
        return self._dt.dtype

    def shape(self) -> Tuple[int, ...]:
        assert isinstance(self._dt, NDArrayDTDescriptor)
        return self._dt.shape

    def get(self) -> InternalNDArray:
        return self._value

    @staticmethod
    def from_number(value: 'NumberValue') -> 'NDArrayValue':
        return NDArrayValue((1,), value.type(), InternalNDArray((1,), [value]))

    @staticmethod
    def from_shape_and_vector(shape: Tuple[int, ...], dtype: NumberDTDescriptor, values: List[NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(shape, dtype, InternalNDArray.from_1d_values_and_shape(values, shape))

    @staticmethod
    def binary(lhs: 'NDArrayValue', rhs: 'NDArrayValue', dtype: NumberDTDescriptor, op: Callable[[NumberValue, NumberValue], NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(lhs.shape(), dtype, lhs.get().binary(rhs.get(), op))

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
        result = InternalNDArray.matmul(lhs.get(), rhs.get(), adder, multiplier, initializer)
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
        assert isinstance(value, NDArrayValue)
        self._dt = value._dt
        self._value = value._value
        return self

    def __copy__(self):
        return self.__class__(self.shape(), self.dtype(), self._value)

    def __deepcopy__(self, memo):
        new_instance = self.__class__(self.shape(), self.dtype(), self._value)
        memo[id(self)] = new_instance
        new_instance._value = copy.deepcopy(self._value, memo)
        return new_instance


class TupleValue(Value):
    def __init__(self, elements_type: Tuple[DTDescriptor, ...], values: Tuple[Value, ...]):
        super().__init__(TupleDTDescriptor(elements_type))
        assert len(elements_type) == len(values)
        for i, (dt, value) in enumerate(zip(elements_type, values)):
            assert dt == value.type(), f"Element {i} has type {value.type()} but expected {dt}"
        self._values = values

    def values(self) -> Tuple[Value, ...]:
        return self._values

    def types(self) -> Tuple[DTDescriptor, ...]:
        assert isinstance(self._dt, TupleDTDescriptor)
        return self._dt.elements_dtype

    def get(self, i: int) -> Value:
        assert 0 <= i < len(self._values)
        return self._values[i]

    def set(self, i: int, value: Value):
        assert 0 <= i < len(self._values)
        assert self._values[i].type() == value.type()
        self._values = self._values[:i] + (value,) + self._values[i + 1:]

    def assign(self, value: 'TupleValue') -> 'TupleValue':
        assert isinstance(value, TupleValue)
        self._dt = value._dt
        self._values = value._values
        return self

    def __copy__(self):
        return self.__class__(self.types(), self.values())

    def __deepcopy__(self, memo):
        new_instance = self.__class__(self.types(), self.values())
        memo[id(self)] = new_instance
        new_instance._values = copy.deepcopy(self.values(), memo)
        return new_instance


class ListValue(Value):
    def __init__(self, elements_type: List[DTDescriptor], values: List[Value]):
        super().__init__(ListDTDescriptor(elements_type))
        assert len(elements_type) == len(values)
        for i, (dt, value) in enumerate(zip(elements_type, values)):
            assert dt == value.type(), f"Element {i} has type {value.type()} but expected {dt}"
        self._values = values

    def values(self) -> List[Value]:
        return self._values

    def types(self) -> List[DTDescriptor]:
        assert isinstance(self._dt, ListDTDescriptor)
        return self._dt.elements_dtype

    def get(self, i: int) -> Value:
        assert 0 <= i < len(self._values)
        return self._values[i]

    def set(self, i: int, value: Value):
        assert 0 <= i < len(self._values)
        assert self._values[i].type() == value.type()
        self._values = self._values[:i] + [value] + self._values[i + 1:]

    def assign(self, value: 'ListValue') -> 'ListValue':
        assert isinstance(value, ListValue)
        self._dt = value._dt
        self._values = value._values
        return self

    def __copy__(self):
        return self.__class__(self.types(), self.values())

    def __deepcopy__(self, memo):
        new_instance = self.__class__(self.types(), self.values())
        memo[id(self)] = new_instance
        new_instance._values = copy.deepcopy(self.values(), memo)
        return new_instance


class NoneValue(Value):
    def __init__(self):
        super().__init__(NoneDTDescriptor())

    def assign(self, value: 'NoneValue') -> 'NoneValue':
        return self

    def __copy__(self):
        return self.__class__()


class ClassValue(Value):
    def __init__(self, value: DTDescriptor):
        super().__init__(ClassDTDescriptor())
        self._value = value

    def val(self) -> DTDescriptor:
        return self._value

    def assign(self, value: 'ClassValue') -> 'ClassValue':
        self._value = value._value
        return self

    def __copy__(self):
        return self.__class__(self.val())


class StringValue(Value):
    def __init__(self, value: str, ptr: int):
        super().__init__(StringDTDescriptor())
        self._value = value
        self._ptr = ptr

    def val(self) -> str:
        return self._value

    def ptr(self) -> int:
        return self._ptr

    def assign(self, value: 'StringValue') -> 'StringValue':
        self._value = value._value
        self._ptr = value._ptr
        return self

    def __copy__(self):
        return self.__class__(self.val(), self.ptr())

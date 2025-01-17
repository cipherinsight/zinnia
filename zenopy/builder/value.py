import copy
from typing import Any, Tuple, Callable, List, Union

from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, \
    NDArrayDTDescriptor, TupleDTDescriptor, ListDTDescriptor, NoneDTDescriptor, ClassDTDescriptor


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
        return self.__class__(self.type(), self._value, self._ptr)

    def __deepcopy__(self, memo):
        return self.__copy__()


class IntegerValue(NumberValue):
    def __init__(self, value: int | None, ptr: int | None):
        super().__init__(IntegerDTDescriptor(), value, ptr)

    def val(self) -> int | None:
        return super().val()

    def ptr(self) -> int | None:
        return super().ptr()

    def assign(self, value: 'IntegerValue') -> 'IntegerValue':
        return super().assign(value)


class FloatValue(NumberValue):
    def __init__(self, value: float | None, ptr: int | None):
        super().__init__(FloatDTDescriptor(), value, ptr)

    def val(self) -> float | None:
        return super().val()

    def ptr(self) -> int | None:
        return super().ptr()

    def assign(self, value: 'IntegerValue') -> 'IntegerValue':
        return super().assign(value)


class NDArrayValue(Value):
    def __init__(self, shape: Tuple[int, ...], dtype: NumberDTDescriptor, value: NDArrayValueWrapper):
        super().__init__(NDArrayDTDescriptor(shape, dtype))
        self._value = value

    def dtype(self) -> NumberDTDescriptor:
        assert isinstance(self._dt, NDArrayDTDescriptor)
        return self._dt.dtype

    def shape(self) -> Tuple[int, ...]:
        assert isinstance(self._dt, NDArrayDTDescriptor)
        return self._dt.shape

    def get(self) -> NDArrayValueWrapper:
        return self._value

    @staticmethod
    def broadcast(lhs: 'NDArrayValue', rhs: 'NDArrayValue') -> Tuple['NDArrayValue', 'NDArrayValue']:
        _lhs, _rhs = NDArrayValueWrapper.binary_broadcast(lhs.get(), rhs.get())
        return NDArrayValue(lhs.shape(), lhs.dtype(), _lhs), NDArrayValue(rhs.shape(), rhs.dtype(), _rhs)

    @staticmethod
    def from_number(value: 'NumberValue') -> 'NDArrayValue':
        return NDArrayValue((1,), value.type(), NDArrayValueWrapper((1,), [value]))

    @staticmethod
    def binary(lhs: 'NDArrayValue', rhs: 'NDArrayValue', dtype: NumberDTDescriptor, op: Callable[[NumberValue, NumberValue], NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(lhs.shape(), dtype, lhs.get().binary(rhs.get(), op))

    def unary(self, dtype: NumberDTDescriptor, op: Callable[[NumberValue], NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(self.shape(), dtype, self.get().unary(op))

    def flattened_values(self) -> List[NumberValue]:
        return self.get().flatten()

    def get_item(self, slicing_params: List[int | Tuple[int, int, int]]) -> Value:
        result = self.get().slice(slicing_params)
        if isinstance(result, NDArrayValueWrapper):
            return NDArrayValue(result.shape, self.dtype(), result)
        return result

    def set_item(self, slicing_params: List[int | Tuple[int, int, int]], val: Union['NDArrayValue', NumberValue]) -> 'NDArrayValue':
        return NDArrayValue(self.shape(), self.dtype(), self.get().slice_assign(
            slicing_params,
            val.get() if isinstance(val, NDArrayValue) else val,
            lambda x, y: x.assign(y)
        ))

    def assign(self, value: 'NDArrayValue') -> 'NDArrayValue':
        assert isinstance(value, NDArrayValue)
        self._dt = value._dt
        self._value = value._value
        return self

    def deep_assign(self, value: 'NDArrayValue') -> 'NDArrayValue':
        assert value.shape() == self.shape() and value.dtype() == self.dtype()
        self._value.binary(value.get(), lambda x, y: x.assign(y))
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

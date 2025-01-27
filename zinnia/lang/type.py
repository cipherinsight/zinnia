from typing import Any

from zinnia.lang.metatype import NDArrayMeta, PoseidonHashedMeta


Integer = int
Float = float
Tuple = tuple
List = list


class NDArray(metaclass=NDArrayMeta):
    shape: Tuple[int, ...]
    dtype: type

    def __init__(self, internal_object, dtype):
        self.__ndarray = internal_object
        self.shape = self.__ndarray.shape
        self.dtype = dtype

    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: type = None) -> 'NDArray':
        from zinnia.internal.internal_ndarray import InternalNDArray

        if dtype is None:
            dtype = Integer
        if dtype == Integer:
            return NDArray(InternalNDArray.fill(shape, lambda: 0), Integer)
        if dtype == Float:
            return NDArray(InternalNDArray.fill(shape, lambda: 0.0), Float)
        raise NotImplementedError()

    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: type = None) -> 'NDArray':
        from zinnia.internal.internal_ndarray import InternalNDArray

        if dtype is None:
            dtype = Integer
        if dtype == Integer:
            return NDArray(InternalNDArray.fill(shape, lambda: 1), Integer)
        if dtype == Float:
            return NDArray(InternalNDArray.fill(shape, lambda: 1.0), Float)
        raise NotImplementedError()

    @staticmethod
    def identity(n: int, dtype: type = None) -> 'NDArray':
        from zinnia.internal.internal_ndarray import InternalNDArray

        if dtype is None:
            dtype = Integer
        if dtype == Integer:
            ndarray = InternalNDArray.fill((n, n), lambda: 0)
            for i in range(n):
                ndarray.ndarray_set_item([i, i], InternalNDArray((1, ), [1]), lambda x, y: y)
            return NDArray(ndarray, Integer)
        if dtype == Float:
            ndarray = InternalNDArray.fill((n, n), lambda: 0.0)
            for i in range(n):
                ndarray.ndarray_set_item([i, i], InternalNDArray((1, ), [1.0]), lambda x, y: y)
            return NDArray(ndarray, Float)
        raise NotImplementedError()

    @staticmethod
    def eye(n: int, m: int, dtype: type = None) -> 'NDArray':
        from zinnia.internal.internal_ndarray import InternalNDArray

        if dtype is None:
            dtype = Integer
        if dtype == Integer:
            ndarray = InternalNDArray.fill((n, m), lambda: 0)
            for i in range(min(n, m)):
                ndarray.ndarray_set_item([i, i], InternalNDArray((1, ), [1]), lambda x, y: y)
            return NDArray(ndarray, Integer)
        if dtype == Float:
            ndarray = InternalNDArray.fill((n, m), lambda: 0.0)
            for i in range(min(n, m)):
                ndarray.ndarray_set_item([i, i], InternalNDArray((1, ), [1.0]), lambda x, y: y)
            return NDArray(ndarray, Float)
        raise NotImplementedError()

    @staticmethod
    def asarray(values: Any, dtype: type = None) -> 'NDArray':
        from zinnia.internal.internal_ndarray import InternalNDArray

        try:
            import numpy as np

            if isinstance(values, np.ndarray):
                values = values.tolist()
        except ImportError:
            pass
        if not InternalNDArray.is_nested_list_in_good_shape(values):
            raise ValueError('Cannot parse list into an NDArray: the shapes of inner lists are not consistent.')
        shape = InternalNDArray.get_nested_list_shape(values)
        ndarray = InternalNDArray(shape, values)
        if dtype is None:
            inferred_dtype = Integer
            for val in ndarray.flatten():
                if isinstance(val, float):
                    inferred_dtype = Float
                elif not isinstance(val, int):
                    raise ValueError('Cannot parse list into an NDArray: the values are not integers or floats.')
        else:
            inferred_dtype = dtype
        return NDArray(ndarray, inferred_dtype)

    def __setitem__(self, key, value):
        from zinnia.internal.internal_ndarray import InternalNDArray

        assert isinstance(self.__ndarray, InternalNDArray)
        slicing_params = self.__parse_slicing_params(key)
        get_item_result = self.__ndarray.ndarray_get_item(slicing_params)
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                value = value.tolist()
        except ImportError:
            pass
        if isinstance(value, list):
            value = NDArray.asarray(value)
        if isinstance(get_item_result, InternalNDArray):
            if not isinstance(value, NDArray):
                value = NDArray(InternalNDArray.fill(get_item_result.shape, lambda: value), self.dtype)
            if not InternalNDArray.directed_broadcast_compatible(value.shape, get_item_result.shape):
                raise ValueError(f'ValueError: could not broadcast input array from shape {value.shape} into shape {get_item_result.shape}')
            value_ndarray = InternalNDArray.directed_broadcast(value.__ndarray, get_item_result.shape)
            self.__ndarray = self.__ndarray.ndarray_set_item(slicing_params, value_ndarray, lambda x, y: y)
        else:
            if self.dtype == Integer:
                value = int(value)
            elif self.dtype == Float:
                value = float(value)
            self.__ndarray = self.__ndarray.ndarray_set_item(slicing_params, value, lambda x, y: y)
        return value

    def __getitem__(self, item):
        from zinnia.internal.internal_ndarray import InternalNDArray

        assert isinstance(self.__ndarray, InternalNDArray)
        slicing_params = self.__parse_slicing_params(item)
        result = self.__ndarray.ndarray_get_item(slicing_params)
        if isinstance(result, InternalNDArray):
            return NDArray(result, self.dtype)
        return result

    @staticmethod
    def __parse_slicing_params(key):
        slicing_params = []
        if isinstance(key, tuple):
            for elt in key:
                if isinstance(elt, int):
                    slicing_params.append(elt)
                elif isinstance(elt, slice):
                    slicing_params.append((elt.start, elt.stop, elt.step))
                    if elt.start is not None and not isinstance(elt.start, int):
                        raise ValueError(f'Invalid parameter got for slice start: {elt.start} ({type(elt.start)})')
                    if elt.stop is not None and not isinstance(elt.stop, int):
                        raise ValueError(f'Invalid parameter got for slice stop: {elt.stop} ({type(elt.stop)})')
                    if elt.step is not None and not isinstance(elt.step, int):
                        raise ValueError(f'Invalid parameter got for slice step: {elt.step} ({type(elt.step)})')
                else:
                    raise ValueError(f'Invalid subscript parameter got: {elt} ({type(elt)})')
        elif isinstance(key, int):
            slicing_params.append(key)
        elif isinstance(key, slice):
            slicing_params.append((key.start, key.stop, key.step))
            if key.start is not None and not isinstance(key.start, int):
                raise ValueError(f'Invalid parameter got for slice start: {key.start} ({type(key.start)})')
            if key.stop is not None and not isinstance(key.stop, int):
                raise ValueError(f'Invalid parameter got for slice stop: {key.stop} ({type(key.stop)})')
            if key.step is not None and not isinstance(key.step, int):
                raise ValueError(f'Invalid parameter got for slice step: {key.step} ({type(key.step)})')
        return slicing_params

    def __str__(self):
        from zinnia.internal.internal_ndarray import InternalNDArray

        assert isinstance(self.__ndarray, InternalNDArray)
        return str(self.__ndarray.values)


class PoseidonHashed(metaclass=PoseidonHashedMeta):
    def __init__(self, actual_value: Any, hash_value: int):
        self.actual_value = actual_value
        self.hash_value = hash_value

    def get_hash(self) -> int:
        return self.hash_value

    def get_value(self) -> Any:
        return self.actual_value

    def set_hash(self, hash_value: int):
        self.hash_value = hash_value

    def set_value(self, actual_value: Any):
        self.actual_value = actual_value

    def __str__(self):
        return f"PoseidonHashed(actual_value={self.actual_value}, hash_value={self.hash_value})"

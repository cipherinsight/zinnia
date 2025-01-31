import copy
from typing import Tuple, List, Callable, Any, Union


class InternalNDArray:
    shape: Tuple[int, ...]
    values: List

    def __init__(self, shape: Tuple[int, ...], values: List):
        self.shape = shape
        self.values = values
        assert InternalNDArray.check_list_shape_matches(shape, values)

    def __copy__(self) -> "InternalNDArray":
        return InternalNDArray(shape=self.shape, values=copy.copy(self.values))

    def __deepcopy__(self, memo) -> "InternalNDArray":
        new_instance = InternalNDArray(shape=self.shape, values=copy.copy(self.values))
        memo[id(self)] = new_instance
        new_instance.values = copy.deepcopy(self.values, memo)
        return new_instance

    def shape_matches(self, other: 'InternalNDArray') -> bool:
        return InternalNDArray._shape_matches(self.shape, other.shape)

    def ndarray_get_item(self, slicing: List[int | Tuple[int, int, int]]) -> Any:
        padded_slicing = slicing + [(None, None, None) for _ in range(len(self.shape) - len(slicing))]
        def _internal_helper(_depth: int, _slicing: List, _values: List):
            _slice = _slicing[_depth]
            if _depth == len(self.shape) - 1:
                if isinstance(_slice, int):
                    return _values[_slice]
                return _values[_slice[0]:_slice[1]:_slice[2]]
            if isinstance(_slice, int):
                return _internal_helper(_depth + 1, _slicing, _values[_slice])
            _values = _values[_slice[0]:_slice[1]:_slice[2]]
            return [_internal_helper(_depth + 1, _slicing, x) for x in _values]
        new_values = _internal_helper(0, padded_slicing, self.values)
        number_of_singles = sum(1 for x in padded_slicing if isinstance(x, int))
        if number_of_singles == len(self.shape):
            return new_values
        return InternalNDArray(shape=InternalNDArray.get_nested_list_shape(new_values), values=new_values)

    def ndarray_set_item(
        self,
        slicing_data: List[int | Tuple[int, int, int]],
        other: Union['InternalNDArray', Any],
        assign_func: Callable[[Any, Any], Any]
    ) -> 'InternalNDArray':
        id_value_mapping = dict()
        encoder_next_id = 0
        def _encode(x):
            nonlocal encoder_next_id, id_value_mapping
            id_value_mapping[encoder_next_id] = x
            encoder_next_id += 1
            return encoder_next_id - 1
        numbered_ndarray = self.unary(_encode)
        sliced_numbered_ndarray = numbered_ndarray.ndarray_get_item(slicing_data)
        old_value_new_value_mapping = dict()
        def _create_mapping(x, y):
            nonlocal old_value_new_value_mapping
            old_value_new_value_mapping[x] = y
            return x
        def _decode(x):
            if old_value_new_value_mapping.get(x, None) is not None:
                return assign_func(id_value_mapping[x], old_value_new_value_mapping[x])
            return id_value_mapping[x]
        if isinstance(sliced_numbered_ndarray, InternalNDArray):
            assert sliced_numbered_ndarray.shape_matches(other)
            sliced_numbered_ndarray.binary(other, _create_mapping)
        else:
            _create_mapping(sliced_numbered_ndarray, other)
        return numbered_ndarray.unary(_decode)

    def flatten(self) -> List[Any]:
        def _internal_helper(_depth: int, _values: List):
            if _depth == len(self.shape):
                return [_values]
            _result = []
            for _val in _values:
                _result += _internal_helper(_depth + 1, _val)
            return _result
        return _internal_helper(0, self.values)

    def tolist(self) -> List[Any]:
        return self.values

    @staticmethod
    def from_1d_values_and_shape(values_1dim: List[Any], shape: Tuple[int, ...]) -> 'InternalNDArray':
        def _internal_helper(_shape: Tuple[int, ...], _values: List):
            if len(_shape) == 1:
                return _values
            _partition_len = len(_values) // _shape[0]
            _results = []
            for i in range(_shape[0]):
                _results.append(_internal_helper(_shape[1:], _values[i * _partition_len:(i + 1) * _partition_len]))
            return _results
        parsed_values = _internal_helper(shape, values_1dim)
        return InternalNDArray(shape=shape, values=parsed_values)

    @staticmethod
    def binary_broadcast(lhs: 'InternalNDArray', rhs: 'InternalNDArray') -> Tuple['InternalNDArray', 'InternalNDArray']:
        shape_lhs, shape_rhs = lhs.shape, rhs.shape
        if len(shape_lhs) < len(shape_rhs):
            shape_lhs = tuple(1 for _ in range(len(shape_rhs) - len(shape_lhs))) + shape_lhs
        if len(shape_rhs) < len(shape_lhs):
            shape_rhs = tuple(1 for _ in range(len(shape_lhs) - len(shape_rhs))) + shape_rhs
        assert all([a == 1 or b == 1 or a == b for a, b in zip(shape_lhs, shape_rhs)])
        broadcast_shape = tuple(max(a, b) for a, b in zip(shape_lhs, shape_rhs))
        def _internal_helper(expected_shape: Tuple[int, ...], _shape: Tuple[int, ...], _operand: List):
            if len(_shape) == 1:
                if _shape[0] == 1:
                    return [_operand[0] for _ in range(expected_shape[0])]
                assert _shape[0] == expected_shape[0]
                return _operand
            else:
                if _shape[0] == 1:
                    return [_internal_helper(expected_shape[1:], _shape[1:], _operand[0]) for _ in range(expected_shape[0])]
                assert _shape[0] == expected_shape[0]
                return [_internal_helper(expected_shape[1:], _shape[1:], _operand[i]) for i in range(expected_shape[0])]
        def _pad_values(values, depth: int):
            if depth <= 0:
                return values
            return [_pad_values(values, depth - 1)]
        new_values_lhs = _internal_helper(broadcast_shape, shape_lhs, _pad_values(lhs.values, len(rhs.shape) - len(lhs.shape)))
        new_values_rhs = _internal_helper(broadcast_shape, shape_rhs, _pad_values(rhs.values, len(lhs.shape) - len(rhs.shape)))
        return InternalNDArray(broadcast_shape, new_values_lhs), InternalNDArray(broadcast_shape, new_values_rhs)

    @staticmethod
    def binary_broadcast_compatible(shape_lhs: Tuple[int, ...], shape_rhs: Tuple[int, ...]) -> bool:
        if len(shape_lhs) < len(shape_rhs):
            shape_lhs = tuple(1 for _ in range(len(shape_rhs) - len(shape_lhs))) + shape_lhs
        if len(shape_rhs) < len(shape_lhs):
            shape_rhs = tuple(1 for _ in range(len(shape_lhs) - len(shape_rhs))) + shape_rhs
        return all([a == 1 or b == 1 or a == b for a, b in zip(shape_lhs, shape_rhs)])

    @staticmethod
    def directed_broadcast(src: 'InternalNDArray', dst: Tuple[int, ...]) -> 'InternalNDArray':
        if src.shape == dst:
            return src
        def _internal_helper(_shape: Tuple[int, ...], _values: List):
            if len(_shape) == 1:
                return _values[0]
            return [_internal_helper(_shape[1:], _values) for _ in range(_shape[0])]
        new_values = _internal_helper(dst, src.values)
        return InternalNDArray(dst, new_values)

    @staticmethod
    def directed_broadcast_compatible(src: Tuple[int, ...], dst: Tuple[int, ...]) -> bool:
        if len(src) < len(dst):
            src = tuple(1 for _ in range(len(dst) - len(src))) + src
        if len(dst) < len(src):
            return False
        return all([a == 1 or a == b for a, b in zip(src, dst)])

    def binary(self, other: 'InternalNDArray', op: Callable[[Any, Any], Any]) -> 'InternalNDArray':
        assert InternalNDArray._shape_matches(self.shape, other.shape)
        def _internal_helper(_shape: Tuple[int, ...], _lhs: List, _rhs: List):
            if len(_shape) == 1:
                return [op(a, b) for a, b in zip(_lhs, _rhs)]
            return [_internal_helper(_shape[1:], a, b) for a, b in zip(_lhs, _rhs)]
        new_values = _internal_helper(self.shape, self.values, other.values)
        return InternalNDArray(shape=self.shape, values=new_values)

    def unary(self, op: Callable[[Any], Any]) -> 'InternalNDArray':
        def _internal_helper(_shape: Tuple[int, ...], _operand: List):
            if len(_shape) == 1:
                return [op(x) for x in _operand]
            return [_internal_helper(_shape[1:], x) for x in _operand]
        new_values = _internal_helper(self.shape, self.values)
        return InternalNDArray(shape=self.shape, values=new_values)

    def accumulate(
            self,
            axis: int,
            accumulator: Callable[[Any, Any, Any, Any], Tuple[Any, Any]],
            initial_generator: Callable[[Any], Tuple[Any, Any]],
            enpair_func: Callable[[Any, Any], Tuple[Any, Any]] = lambda x, _: (x, None),
            depair_func: Callable[[Any, Any], Any] = lambda x, _: x
    ) -> Any:
        assert 0 <= axis < len(self.shape) or axis == -1
        if axis == -1:
            flatten_values = self.flatten()
            result, result_i = initial_generator(flatten_values[0])
            for i, x in enumerate(flatten_values):
                a, b = enpair_func(x, i)
                result, result_i = accumulator(result, result_i, a, b)
            return depair_func(result, result_i)
        else:
            def _generate_initial_by_shape(_shape: Tuple[int, ...], _operand: List):
                if len(_shape) == 1:
                    return [initial_generator(_operand[i]) for i in range(_shape[0])]
                return [_generate_initial_by_shape(_shape[1:], _operand[i]) for i in range(_shape[0])]

            def _binary_accumulate(_shape: Tuple[int, ...], _lhs, _rhs, _rhs_i):
                if len(_shape) == 0:
                    _a, _b = enpair_func(_rhs, _rhs_i)
                    return accumulator(_lhs[0], _lhs[1], _a, _b)
                return [_binary_accumulate(_shape[1:], _a, _b, _rhs_i) for _a, _b in zip(_lhs, _rhs)]

            def _internal_helper(_shape: Tuple[int, ...], depth: int, _operand: List):
                if depth < axis:
                    return [_internal_helper(_shape[1:], depth + 1, x) for x in _operand]
                elif depth == axis:
                    if len(_shape) == 1:
                        result = initial_generator(_operand[0])
                        for i, x in enumerate(_operand):
                            a, b = enpair_func(x, i)
                            result = accumulator(result[0], result[1], a, b)
                        return result
                    result = _generate_initial_by_shape(_shape[1:], _operand[0])
                    for i, x in enumerate(_operand):
                        result = _binary_accumulate(_shape[1:], result, x, i)
                    return result

            def _parsing_helper(_shape: Tuple[int, ...], _operand: List):
                if len(_shape) == 0:
                    return depair_func(_operand[0], _operand[1])
                return [_parsing_helper(_shape[1:], _operand[i]) for i in range(_shape[0])]

            new_values = _internal_helper(self.shape, 0, self.values)
            new_shape = tuple(x for i, x in enumerate(self.shape) if i != axis)
            if len(new_shape) == 0:
                val, idx = new_values
                return _parsing_helper(val, idx)
            new_values = _parsing_helper(new_shape, new_values)
            return InternalNDArray(shape=new_shape, values=new_values)

    def transpose(self, axes: Tuple[int, ...]) -> 'InternalNDArray':
        assert tuple(sorted(axes)) == tuple(i for i in range(len(self.shape)))
        values = self.flatten()
        # Compute the new shape after transposition
        new_shape = tuple(self.shape[axis] for axis in axes)

        # Convert flat index to multi-index
        def unravel_index(index, shape):
            result = []
            for size in reversed(shape):
                result.append(index % size)
                index //= size
            return tuple(reversed(result))

        # Convert multi-index to flat index
        def ravel_index(indices, shape):
            flat_index = 0
            for i, s in zip(indices, shape):
                flat_index = flat_index * s + i
            return flat_index

        # Initialize the transposed array
        transposed = [None] * len(values)

        # Map values to their new positions
        for flat_idx in range(len(values)):
            original_idx = unravel_index(flat_idx, self.shape)
            new_idx = tuple(original_idx[axis] for axis in axes)
            new_flat_idx = ravel_index(new_idx, new_shape)
            transposed[new_flat_idx] = values[flat_idx]

        # Recursively reshape the flattened transposed array into nested lists
        def reshape(flat_list, shape):
            if len(shape) == 1:
                return flat_list
            size = shape[0]
            sub_size = len(flat_list) // size
            return [reshape(flat_list[i * sub_size:(i + 1) * sub_size], shape[1:]) for i in range(size)]

        return InternalNDArray(new_shape, reshape(transposed, new_shape))

    def for_each(self, func: Callable[[Tuple[int, ...], Any], Any]) -> 'InternalNDArray':
        def _internal_helper(_indices: Tuple[int, ...], _depth: int, _operand: List):
            if _depth == len(self.shape):
                return func(_indices, _operand)
            return [_internal_helper(_indices + (i, ), _depth + 1, val) for i, val in enumerate(_operand)]
        new_values = _internal_helper(tuple(), 0, self.values)
        return InternalNDArray(shape=self.shape, values=new_values)

    @staticmethod
    def fill(shape: Tuple[int, ...], fill_value: Callable[[], Any]) -> 'InternalNDArray':
        def _internal_helper(_shape: Tuple[int, ...]):
            if len(_shape) == 1:
                return [fill_value() for _ in range(_shape[0])]
            return [_internal_helper(_shape[1:]) for _ in range(_shape[0])]
        new_values = _internal_helper(shape)
        return InternalNDArray(shape=shape, values=new_values)

    @staticmethod
    def concatenate(args: List['InternalNDArray'], axis=0) -> 'InternalNDArray':
        if axis == -1:
            flatten_values = []
            for arg in args:
                flatten_values += arg.flatten()
            return InternalNDArray((len(flatten_values),), flatten_values)
        new_shape_value = 0
        for arg in args:
            new_shape_value += arg.shape[axis]
        new_shape = tuple(x if i != axis else new_shape_value for i, x in enumerate(args[0].shape))
        def _internal_helper(_axis: int, _values_lhs: List, _values_rhs: List):
            if _axis == axis:
                return _values_lhs + _values_rhs
            assert len(_values_lhs) == len(_values_rhs)
            return [_internal_helper(_axis + 1, x, y) for x, y in zip(_values_lhs, _values_rhs)]
        result = args[0].values
        for arg in args[1:]:
            result = _internal_helper(0, result, arg.values)
        return InternalNDArray(new_shape, result)

    @staticmethod
    def stack(args: List['InternalNDArray'], axis=0) -> 'InternalNDArray':
        new_shape = list(args[0].shape)
        new_shape.insert(axis, len(args))
        new_shape = tuple(new_shape)
        def _internal_helper(_axis: int, _values: List[List]):
            if _axis == axis:
                return _values
            return [_internal_helper(_axis + 1, [x[i] for x in _values]) for i in range(len(_values[0]))]
        new_values = _internal_helper(0, [arg.values for arg in args])
        return InternalNDArray(new_shape, new_values)

    @staticmethod
    def matmul_shape_matches(shape_lhs: Tuple[int, ...], shape_rhs: Tuple[int, ...]) -> bool:
        if len(shape_lhs) > 2 or len(shape_rhs) > 2:
            return False
        if len(shape_lhs) == 1:
            shape_lhs = (1, shape_lhs[0])
        if len(shape_rhs) == 1:
            shape_rhs = (shape_rhs[0], 1)
        return shape_lhs[1] == shape_rhs[0]

    @staticmethod
    def matmul(
            lhs: 'InternalNDArray', rhs: 'InternalNDArray',
            adder: Callable[[Any, Any], Any], multiplier: Callable[[Any, Any], Any],
            initializer: Callable[[], Any]
    ) -> 'InternalNDArray':
        assert InternalNDArray.matmul_shape_matches(lhs.shape, rhs.shape)
        lhs_values, rhs_values = lhs.values, rhs.values
        lhs_shape, rhs_shape = lhs.shape, rhs.shape
        if len(lhs_shape) == 1:
            lhs_shape = (1, lhs.shape[0])
            lhs_values = [lhs_values]
        if len(rhs_shape) == 1:
            rhs_shape = (rhs.shape[0], 1)
            rhs_values = [[x] for x in rhs_values]
        new_values = [[initializer() for j in range(rhs_shape[1])] for i in range(lhs_shape[0])]
        for i in range(lhs_shape[0]):
            for j in range(rhs_shape[1]):
                for k in range(lhs_shape[1]):
                    new_values[i][j] = adder(new_values[i][j], multiplier(lhs_values[i][k], rhs_values[k][j]))
        if len(lhs.shape) == 1:
            new_values = new_values[0]
            return InternalNDArray((len(new_values),), new_values)
        if len(rhs.shape) == 1:
            new_values = [x[0] for x in new_values]
            return InternalNDArray((len(new_values),), new_values)
        return InternalNDArray((lhs_shape[0], rhs_shape[1]), new_values)

    @staticmethod
    def check_list_shape_matches(shape: Tuple[int, ...], values: List) -> bool:
        if len(shape) == 0:
            return False
        if shape[0] <= 0:
            return False
        if len(shape) == 1:
            return len(values) == shape[0]
        if shape[0] != len(values):
            return False
        for i in range(shape[0]):
            if not InternalNDArray.check_list_shape_matches(shape[1:], values[i]):
                return False
        return True

    @staticmethod
    def _shape_matches(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> bool:
        return lhs == rhs

    @staticmethod
    def get_nested_list_shape(values: List) -> Tuple[int, ...]:
        def _internal_helper(_vals):
            if isinstance(_vals, List):
                tailing_shape = _internal_helper(_vals[0])
                for i in range(1, len(_vals)):
                    assert tailing_shape == _internal_helper(_vals[i])
                return (len(_vals), ) + tailing_shape
            return tuple()
        return _internal_helper(values)

    @staticmethod
    def is_nested_list_in_good_shape(values: List) -> bool:
        try:
            InternalNDArray.get_nested_list_shape(values)
            return True
        except AssertionError:
            return False

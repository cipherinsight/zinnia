import copy
from typing import Tuple, List, Callable, Any, Union


class InternalNDArray:
    shape: Tuple[int, ...]
    values: List | Any

    def __init__(self, shape: Tuple[int, ...], values: List | Any):
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
        if self.shape == ():
            assert len(slicing) == 0
            return self.values
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
        if self.shape == ():
            return [self.values]
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
    def directed_broadcast(src: 'InternalNDArray', dst: Tuple[int, ...]) -> 'InternalNDArray':
        if src.shape == dst:
            return src
        shape_src = src.shape
        if len(shape_src) < len(dst):
            shape_src = tuple(1 for _ in range(len(dst) - len(shape_src))) + shape_src
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
        new_values = _internal_helper(dst, shape_src, _pad_values(src.values, len(dst) - len(src.shape)))
        return InternalNDArray(dst, new_values)

    @staticmethod
    def directed_broadcast_compatible(src: Tuple[int, ...], dst: Tuple[int, ...]) -> bool:
        if src == ():
            return True
        if len(src) < len(dst):
            src = tuple(1 for _ in range(len(dst) - len(src))) + src
        if len(dst) < len(src):
            return False
        return all([a == 1 or a == b for a, b in zip(src, dst)])

    def binary(self, other: 'InternalNDArray', op: Callable[[Any, Any], Any]) -> 'InternalNDArray':
        assert InternalNDArray._shape_matches(self.shape, other.shape)
        if self.shape == ():
            return InternalNDArray(shape=(), values=op(self.values, other.values))
        def _internal_helper(_shape: Tuple[int, ...], _lhs: List, _rhs: List):
            if len(_shape) == 1:
                return [op(a, b) for a, b in zip(_lhs, _rhs)]
            return [_internal_helper(_shape[1:], a, b) for a, b in zip(_lhs, _rhs)]
        new_values = _internal_helper(self.shape, self.values, other.values)
        return InternalNDArray(shape=self.shape, values=new_values)

    def unary(self, op: Callable[[Any], Any]) -> 'InternalNDArray':
        if self.shape == ():
            return InternalNDArray(shape=(), values=op(self.values))
        def _internal_helper(_shape: Tuple[int, ...], _operand: List):
            if len(_shape) == 1:
                return [op(x) for x in _operand]
            return [_internal_helper(_shape[1:], x) for x in _operand]
        new_values = _internal_helper(self.shape, self.values)
        return InternalNDArray(shape=self.shape, values=new_values)

    @staticmethod
    def fill(shape: Tuple[int, ...], fill_value: Callable[[], Any]) -> 'InternalNDArray':
        if shape == ():
            return InternalNDArray(shape=(), values=fill_value())
        def _internal_helper(_shape: Tuple[int, ...]):
            if len(_shape) == 1:
                return [fill_value() for _ in range(_shape[0])]
            return [_internal_helper(_shape[1:]) for _ in range(_shape[0])]
        new_values = _internal_helper(shape)
        return InternalNDArray(shape=shape, values=new_values)

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
    def check_list_shape_matches(shape: Tuple[int, ...], values: List) -> bool:
        if shape == ():
            return True
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

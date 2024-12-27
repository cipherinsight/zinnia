import copy
from typing import Tuple, List, Callable, Any


class NDArrayHelper:
    shape: Tuple[int, ...]
    values: List

    def __init__(self, shape: Tuple[int, ...], values: List):
        self.shape = shape
        self.values = values
        assert NDArrayHelper._assert_shape_matches_value(shape, values)

    def __deepcopy__(self) -> "NDArrayHelper":
        return NDArrayHelper(shape=self.shape, values=copy.deepcopy(self.values))

    def shape_matches(self, other: 'NDArrayHelper') -> bool:
        return NDArrayHelper._shape_matches(self.shape, other.shape)

    def slice(self, slicing: List[Tuple[int, ...]]) -> Any:
        assert self.check_slicing(slicing) is None
        padded_slicing = slicing + [(None, None, None) for _ in range(len(self.shape) - len(slicing))]
        padded_slicing = [(x + (None, ) if len(x) == 2 else x) for x in padded_slicing]
        def _internal_helper(_depth: int, _slicing: List, _values: List):
            _slice = _slicing[_depth]
            if _depth == len(self.shape) - 1:
                if len(_slice) == 1:
                    return _values[_slice[0]]
                return _values[_slice[0]:_slice[1]:_slice[2]]
            if len(_slice) == 1:
                return _internal_helper(_depth + 1, _slicing, _values[_slice[0]])
            _values = _values[_slice[0]:_slice[1]:_slice[2]]
            return [_internal_helper(_depth + 1, _slicing, x) for x in _values]
        new_values = _internal_helper(0, padded_slicing, self.values)
        if not isinstance(new_values, List):
            return new_values
        return NDArrayHelper(shape=NDArrayHelper._get_shape_of(new_values), values=new_values)

    def slice_assign(self, slicing_data: List[List[Tuple[int, ...]]], other: Any) -> 'NDArrayHelper':
        assert self.check_slicing_assign(slicing_data, other) is None
        id_value_mapping = dict()
        encoder_next_id = 0
        def _encode(x):
            nonlocal encoder_next_id, id_value_mapping
            id_value_mapping[encoder_next_id] = x
            encoder_next_id += 1
            return encoder_next_id - 1
        numbered_ndarray = self.unary(_encode)
        sliced_numbered_ndarray = numbered_ndarray
        for slicing in slicing_data:
            sliced_numbered_ndarray = sliced_numbered_ndarray.slice(slicing)
        old_value_new_value_mapping = dict()
        def _create_mapping(x, y):
            nonlocal old_value_new_value_mapping
            old_value_new_value_mapping[x] = y
            return x
        if isinstance(sliced_numbered_ndarray, NDArrayHelper):
            assert sliced_numbered_ndarray.shape_matches(other)
            sliced_numbered_ndarray.binary(other, _create_mapping)
        else:
            _create_mapping(sliced_numbered_ndarray, other)
        def _decode(x):
            if old_value_new_value_mapping.get(x, None) is not None:
                return old_value_new_value_mapping[x]
            return id_value_mapping[x]
        return numbered_ndarray.unary(_decode)


    def check_slicing_assign(self, slicing_data: List[List[Tuple[int, ...]]], other: Any) -> str | None:
        assignee = self
        for slicing in slicing_data:
            if not isinstance(assignee, NDArrayHelper):
                return f"Invalid slicing assignment: too many slicing parameters on the assignee"
            result_1 = assignee.check_slicing(slicing)
            if result_1 is not None:
                return result_1
            assignee = assignee.slice(slicing)
        if (not isinstance(other, NDArrayHelper) and not isinstance(assignee, NDArrayHelper)) or assignee.shape_matches(other):
            return None
        return f"Invalid slicing assignment: shape on the lhs {assignee.shape if isinstance(assignee, NDArrayHelper) else '(1,)'} is not equal to shape on the rhs {other.shape if isinstance(other, NDArrayHelper) else '(1,)'}"

    def check_slicing(self, slicing: List[Tuple[int, ...]]) -> str | None:
        if len(self.shape) < len(slicing):
            return f"Too many slicing dimensions: {len(slicing)} dimensions requested but there is only {len(self.shape)} dimensions on the target"
        for i, s in enumerate(slicing):
            if len(s) == 1 and s[0] is None:
                return f"Invalid {i}-th slicing `None`"
            if len(s) == 1 and s[0] >= self.shape[i]:
                return f"The {i}-th slicing index out of range: {s} out of range {self.shape[i]}"
            if len(s) == 1 and s[0] < 0 and s[0] + self.shape[i] < 0:
                return f"The {i}-th slicing index out of range: {s} out of range {self.shape[i]}"
        return None

    def flatten(self) -> List[Any]:
        def _internal_helper(_depth: int, _values: List):
            if _depth == len(self.shape):
                return [_values]
            _result = []
            for _val in _values:
                _result += _internal_helper(_depth + 1, _val)
            return _result
        return _internal_helper(0, self.values)

    @staticmethod
    def from_1d_values_and_shape(values_1dim: List[Any], shape: Tuple[int, ...]) -> 'NDArrayHelper':
        def _internal_helper(_shape: Tuple[int, ...], _values: List):
            if len(_shape) == 1:
                return _values
            _partition_len = len(_values) // _shape[0]
            _results = []
            for i in range(_shape[0]):
                _results.append(_internal_helper(_shape[1:], _values[i * _partition_len:(i + 1) * _partition_len]))
            return _results
        parsed_values = _internal_helper(shape, values_1dim)
        return NDArrayHelper(shape=shape, values=parsed_values)

    @staticmethod
    def broadcast(lhs: 'NDArrayHelper', rhs: 'NDArrayHelper') -> Tuple['NDArrayHelper', 'NDArrayHelper']:
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
        return NDArrayHelper(broadcast_shape, new_values_lhs), NDArrayHelper(broadcast_shape, new_values_rhs)

    @staticmethod
    def broadcast_shape(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> Tuple[int, ...]:
        shape_lhs, shape_rhs = lhs, rhs
        if len(shape_lhs) < len(shape_rhs):
            shape_lhs = tuple(1 for _ in range(len(shape_rhs) - len(shape_lhs))) + shape_lhs
        if len(shape_rhs) < len(shape_lhs):
            shape_rhs = tuple(1 for _ in range(len(shape_lhs) - len(shape_rhs))) + shape_rhs
        assert all([a == 1 or b == 1 or a == b for a, b in zip(shape_lhs, shape_rhs)])
        broadcast_shape = tuple(max(a, b) for a, b in zip(shape_lhs, shape_rhs))
        return broadcast_shape

    @staticmethod
    def broadcast_compatible(shape_lhs: Tuple[int, ...], shape_rhs: Tuple[int, ...]) -> bool:
        if len(shape_lhs) < len(shape_rhs):
            shape_lhs = tuple(1 for _ in range(len(shape_rhs) - len(shape_lhs))) + shape_lhs
        if len(shape_rhs) < len(shape_lhs):
            shape_rhs = tuple(1 for _ in range(len(shape_lhs) - len(shape_rhs))) + shape_rhs
        return all([a == 1 or b == 1 or a == b for a, b in zip(shape_lhs, shape_rhs)])

    def binary(self, other: 'NDArrayHelper', op: Callable[[Any, Any], Any]) -> 'NDArrayHelper':
        assert NDArrayHelper._shape_matches(self.shape, other.shape)
        def _internal_helper(_shape: Tuple[int, ...], _lhs: List, _rhs: List):
            if len(_shape) == 1:
                return [op(a, b) for a, b in zip(_lhs, _rhs)]
            return [_internal_helper(_shape[1:], a, b) for a, b in zip(_lhs, _rhs)]
        new_values = _internal_helper(self.shape, self.values, other.values)
        return NDArrayHelper(shape=self.shape, values=new_values)

    def unary(self, op: Callable[[Any], Any]) -> 'NDArrayHelper':
        def _internal_helper(_shape: Tuple[int, ...], _operand: List):
            if len(_shape) == 1:
                return [op(x) for x in _operand]
            return [_internal_helper(_shape[1:], x) for x in _operand]
        new_values = _internal_helper(self.shape, self.values)
        return NDArrayHelper(shape=self.shape, values=new_values)

    def accumulate(
            self,
            axis: int,
            accumulator: Callable[[Any, int, Any, int], Tuple[Any, int]],
            initial_generator: Callable[[Any], Tuple[Any, int]],
            result_parser: Callable[[Any, int], Any] = lambda x, _: x
    ) -> Any:
        assert 0 <= axis < len(self.shape) or axis == -1
        if axis == -1:
            def _internal_helper(_shape: Tuple[int, ...], _operand: List):
                if len(_shape) == 1:
                    result, result_idx = initial_generator(_operand[0])
                    for i, x in enumerate(_operand):
                        result, result_idx = accumulator(result, result_idx, x, i)
                    return result, result_idx
                else:
                    result, result_idx = initial_generator(_operand[0])
                    for i, x in enumerate(_operand):
                        result, result_idx = accumulator(result, result_idx, _internal_helper(_shape[1:], x)[0], i)
                    return result, result_idx
            ans, ans_i = _internal_helper(self.shape, self.values)
            return result_parser(ans, ans_i)
        else:
            def _generate_initial_by_shape(_shape: Tuple[int, ...], _operand: List):
                if len(_shape) == 1:
                    return [initial_generator(_operand[i]) for i in range(_shape[0])]
                return [_generate_initial_by_shape(_shape[1:], _operand[i]) for i in range(_shape[0])]

            def _binary_accumulate(_shape: Tuple[int, ...], _lhs, _rhs, _rhs_i):
                if len(_shape) == 0:
                    return accumulator(_lhs[0], _lhs[1], _rhs, _rhs_i)
                return [_binary_accumulate(_shape[1:], a, b, _rhs_i) for a, b in zip(_lhs, _rhs)]

            def _internal_helper(_shape: Tuple[int, ...], depth: int, _operand: List):
                if depth < axis:
                    return [_internal_helper(_shape[1:], depth + 1, x) for x in _operand]
                elif depth == axis:
                    result = _generate_initial_by_shape(_shape[1:], _operand[0])
                    for i, x in enumerate(_operand):
                        result = _binary_accumulate(_shape[1:], result, x, i)
                    return result

            def _parsing_helper(_shape: Tuple[int, ...], _operand: List):
                if len(_shape) == 0:
                    return result_parser(_operand[0], _operand[1])
                return [_parsing_helper(_shape[1:], _operand[i]) for i in range(_shape[0])]

            new_values = _internal_helper(self.shape, 0, self.values)
            new_shape = tuple(x for i, x in enumerate(self.shape) if i != axis)
            if len(new_shape) == 0:
                val, idx = new_values
                return _parsing_helper(val, idx)
            new_values = _parsing_helper(new_shape, new_values)
            return NDArrayHelper(shape=new_shape, values=new_values)

    def for_each(self, func: Callable[[Tuple[int, ...], Any], Any]) -> 'NDArrayHelper':
        def _internal_helper(_indices: Tuple[int, ...], _depth: int, _operand: List):
            if _depth == len(self.shape):
                return func(_indices, _operand)
            return [_internal_helper(_indices + (i, ), _depth + 1, val) for i, val in enumerate(_operand)]
        new_values = _internal_helper(tuple(), 0, self.values)
        return NDArrayHelper(shape=self.shape, values=new_values)

    @staticmethod
    def fill(shape: Tuple[int, ...], fill_value: Callable[[], Any]) -> 'NDArrayHelper':
        def _internal_helper(_shape: Tuple[int, ...]):
            if len(_shape) == 1:
                return [fill_value() for _ in range(_shape[0])]
            return [_internal_helper(_shape[1:]) for _ in range(_shape[0])]
        new_values = _internal_helper(shape)
        return NDArrayHelper(shape=shape, values=new_values)

    @staticmethod
    def concat(*args) -> 'NDArrayHelper':
        assert all([isinstance(arg, NDArrayHelper) for arg in args])
        for i, arg in enumerate(args):
            assert i == 0 or arg == args[i - 1]
        return NDArrayHelper((len(args),) + args[0].shape, [x.values.copy() for x in args])

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
    def matmul_shape(shape_lhs: Tuple[int, ...], shape_rhs: Tuple[int, ...]) -> Tuple[int, ...]:
        assert NDArrayHelper.matmul_shape_matches(shape_lhs, shape_rhs)
        if len(shape_lhs) == 1:
            return (shape_rhs[0], )
        if len(shape_rhs) == 1:
            return (shape_lhs[1], )
        return shape_lhs[0], shape_rhs[1]

    @staticmethod
    def matmul(
            lhs: 'NDArrayHelper', rhs: 'NDArrayHelper',
            adder: Callable[[Any, Any], Any], multiplier: Callable[[Any, Any], Any],
            initializer: Callable[[], Any]
    ) -> 'NDArrayHelper':
        assert NDArrayHelper.matmul_shape_matches(lhs.shape, rhs.shape)
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
            return NDArrayHelper((len(new_values), ), new_values)
        if len(rhs.shape) == 1:
            new_values = [x[0] for x in new_values]
            return NDArrayHelper((len(new_values), ), new_values)
        return NDArrayHelper((lhs_shape[0], rhs_shape[1]), new_values)

    @staticmethod
    def _assert_shape_matches_value(shape: Tuple[int, ...], values: List) -> bool:
        if len(shape) == 0:
            return False
        if shape[0] <= 0:
            return False
        if len(shape) == 1:
            return len(values) == shape[0]
        if shape[0] != len(values):
            return False
        for i in range(shape[0]):
            if not NDArrayHelper._assert_shape_matches_value(shape[1:], values[i]):
                return False
        return True

    @staticmethod
    def _shape_matches(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> bool:
        return lhs == rhs

    @staticmethod
    def _get_shape_of(values: List) -> Tuple[int, ...]:
        def _internal_helper(_vals):
            if isinstance(_vals, List):
                tailing_shape = _internal_helper(_vals[0])
                for i in range(1, len(_vals)):
                    assert tailing_shape == _internal_helper(_vals[i])
                return (len(_vals), ) + tailing_shape
            return tuple()
        return _internal_helper(values)
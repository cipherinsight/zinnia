from collections.abc import Callable
from typing import List, Tuple

from pyzk.exception.contextual import TypeInferenceError
from pyzk.util.datatype_name import DataTypeName
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class IRInferenceDescriptor:
    typename: str
    public: bool
    value: int | NDArrayHelper | None

    def __init__(
        self,
        typename: str,
        value: int | NDArrayHelper | None = None,
        public: bool = False
    ):
        assert typename is not None and public is not None
        assert ((isinstance(value, int) or value is None) and typename == DataTypeName.NUMBER) or (isinstance(value, NDArrayHelper) and typename == DataTypeName.NDARRAY)
        self.typename = typename
        self.public = public
        self.value = value

    def datatype_matches(self, other):
        return self.typename == other.typename

    def pretty_typename(self):
        s = self.typename
        if self.typename == DataTypeName.NDARRAY and len(self.value.shape) > 0:
            s += '[' + ', '.join([str(x) for x in self.value.shape]) + ']'
        return s

    def is_number(self):
        return self.typename == DataTypeName.NUMBER

    def is_ndarray(self):
        return self.typename == DataTypeName.NDARRAY

    def get_shape(self):
        return self.value.shape if self.is_ndarray() else tuple()

    def has_inferred_value(self):
        return self.value is not None

    def __str__(self):
        return self.pretty_typename() + f' ({self.value})'

    @staticmethod
    def new(
        typename: str,
        shape: Tuple[int, ...],
        public: bool = False,
        value: int | List | None = None,
    ) -> 'IRInferenceDescriptor':
        if typename == DataTypeName.NDARRAY:
            return IRInferenceDescriptor(typename, NDArrayHelper.fill(shape, lambda: None) if value is None else value, public)
        return IRInferenceDescriptor(typename, value, public)


class IRInference:
    @staticmethod
    def do_ir_inference(
        op_name: str,
        args: List[IRInferenceDescriptor],
        constant_args: List[int] | None = None,
        slicing_args: List[int | Tuple[int, int, int]] = None,
        slicing_assign_args: List[List[int | Tuple[int, int, int]]] = None,
        source_pos_info: SourcePosInfo | None = None,
        constant_value: int | None = None
    ) -> IRInferenceDescriptor | None:
        method_name = 'infer_' + op_name
        method = getattr(IRInference, method_name)
        kwargs = {}
        if slicing_args is not None:
            kwargs['slicing_args'] = slicing_args
        if slicing_assign_args is not None:
            kwargs['slicing_assign_args'] = slicing_assign_args
        if constant_args is not None:
            kwargs['constant_args'] = constant_args
        if source_pos_info is not None:
            kwargs['source_pos_info'] = source_pos_info
        if constant_value is not None:
            kwargs['constant_value'] = constant_value
        return method(*args, **kwargs)

    @staticmethod
    def infer_add(
        lhs: IRInferenceDescriptor,
        rhs: IRInferenceDescriptor,
        source_pos_info: SourcePosInfo | None = None
    ) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x + y))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Cannot perform binary operation `add` on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. These two operands are not broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x + y))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x + y))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x + y))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_sub(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x - y))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Cannot perform binary operation `sub` on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. These two operands are not broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x - y))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x - y))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x - y))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_mul(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x * y))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Cannot perform binary operation `mul` on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. These two operands are not broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x * y))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x * y))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: x * y))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_div(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: None))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Cannot perform binary operation `div` on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. These two operands are not broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: None))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: None))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: None))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_mat_mul(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number():
            raise TypeInferenceError(source_pos_info, f'Invalid binary operator `mat_mul` on lhs operand {lhs.pretty_typename()}. Note that `mat_mul` does not support multiplication using scalars')
        if rhs.is_number():
            raise TypeInferenceError(source_pos_info, f'Invalid binary operator `mat_mul` on rhs operand {rhs.pretty_typename()}. Note that `mat_mul` does not support multiplication using scalars')
        if len(lhs.value.shape) > 2 or len(rhs.value.shape) > 2:
            raise TypeInferenceError(
                source_pos_info, f'Invalid binary operator `mat_mul` on ndarray operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To multiply two ndarray, their dimension number must be equal or less than 2')
        if not NDArrayHelper.matmul_shape_matches(lhs.value.shape, rhs.value.shape):
            raise TypeInferenceError(
                source_pos_info, f'Invalid binary operator `mat_mul` on ndarray operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To multiply two ndarray, the 2-nd dimension number of lhs operand must be equal to the 1-st dimension of the rhs operand')
        inferred_value = IRInference._infer_value_ndarray_ndarray_multiplication(lhs, rhs)
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=inferred_value)

    @staticmethod
    def infer_and(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: (1 if x != 0 else 0) * (1 if y != 0 else 0)))
        raise TypeInferenceError(source_pos_info, f'Invalid binary operator `and` on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. Logical `and` can only be applied on scalar numbers')

    @staticmethod
    def infer_or(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if ((1 if x != 0 else 0) + (1 if y != 0 else 0) > 0) else 0))
        raise TypeInferenceError(source_pos_info, f'Invalid binary operator `or` on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. Logical `or` can only be applied on scalar numbers')

    @staticmethod
    def infer_not(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if operand.is_number():
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=IRInference._infer_value_unary(operand, lambda x: 1 if x == 0 else 0))
        raise TypeInferenceError(source_pos_info, f'Invalid binary operator `not` on operands {operand.pretty_typename()}. Logical `not` can only be applied on scalar numbers')

    @staticmethod
    def infer_usub(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if operand.is_number():
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=IRInference._infer_value_unary(operand, lambda x: -x))
        if operand.is_ndarray():
            return IRInferenceDescriptor(DataTypeName.NDARRAY, value=IRInference._infer_value_unary(operand, lambda x: -x))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_is_true(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if not operand.is_number():
            raise TypeInferenceError(source_pos_info, f'Invalid unary operator `is_true` on operand {operand.pretty_typename()}. The `is_true` operator only support scalar value')
        return IRInferenceDescriptor(DataTypeName.NUMBER, value=IRInference._infer_value_unary(operand, lambda x: 1 if x != 0 else 0))

    @staticmethod
    def infer_is_false(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if not operand.is_number():
            raise TypeInferenceError(source_pos_info, f'Invalid unary operator `is_false` on operand {operand.pretty_typename()}. The `is_false` operator only support scalar value')
        return IRInferenceDescriptor(DataTypeName.NUMBER, value=IRInference._infer_value_unary(operand, lambda x: 1 if x == 0 else 0))

    @staticmethod
    def infer_gte(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x >= y else 0))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid binary operator `gte` (>=) on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To compare two ndarray operands, their shapes must be broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x >= y else 0))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x >= y else 0))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x >= y else 0))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_lte(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x <= y else 0))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid binary operator `lte` (<=) on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To compare two ndarray operands, their shapes must be broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x <= y else 0))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x <= y else 0))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x <= y else 0))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_gt(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x > y else 0))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid binary operator `gt` (>) on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To compare two ndarray operands, their shapes must be broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x > y else 0))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x > y else 0))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x > y else 0))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_lt(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x < y else 0))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid binary operator `lt` (<) on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To compare two ndarray operands, their shapes must be broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x < y else 0))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x < y else 0))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x < y else 0))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_eq(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x == y else 0))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid binary operator `gte` (>=) on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To compare two ndarray operands, their shapes must be broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x == y else 0))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x == y else 0))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x == y else 0))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_ne(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if lhs.is_number() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NUMBER, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x != y else 0))
        elif lhs.is_ndarray() and rhs.is_ndarray():
            if not NDArrayHelper.broadcast_compatible(lhs.value.shape, rhs.value.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid binary operator `gte` (>=) on operands {lhs.pretty_typename()} and {rhs.pretty_typename()}. To compare two ndarray operands, their shapes must be broadcast compatible')
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x != y else 0))
        elif lhs.is_number() and rhs.is_ndarray():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x != y else 0))
        elif lhs.is_ndarray() and rhs.is_number():
            return IRInferenceDescriptor(
                DataTypeName.NDARRAY, value=IRInference._infer_value_binary(lhs, rhs, lambda x, y: 1 if x != y else 0))
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design')

    @staticmethod
    def infer_NDArray_all_zeros(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        for arg in args:
            assert isinstance(arg, IRInferenceDescriptor)
        if len(args) != 0:
            raise TypeInferenceError(source_pos_info, 'Invalid number of operands on operator `NDArray.all_zeros`. All operands for this operator should be constants')
        if len(constant_args) == 0:
            raise TypeInferenceError(source_pos_info, 'Invalid usage on operator `NDArray.all_zeros`. The number of constant operands on this operator should be greater than 0')
        exists_0 = any([arg == 0 for arg in constant_args])
        if exists_0:
            raise TypeInferenceError(source_pos_info, f'Invalid size value on `NDArray.all_zeros`. The size values should be greater than 0')
        shape = tuple([arg for arg in constant_args])
        inferred_value = NDArrayHelper.fill(shape, lambda: 0)
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=inferred_value)

    @staticmethod
    def infer_NDArray_all_ones(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if len(args) != 0:
            raise AssertionError(source_pos_info, 'Invalid number of operands on operator `NDArray.all_ones`. All operands for this operator should be constants. Please check the transpiler design')
        if len(constant_args) == 0:
            raise TypeInferenceError(source_pos_info, 'Invalid usage on operator `NDArray.all_ones`. The number of constant operands on this operator should be greater than 0')
        exists_0 = any([arg <= 0 for arg in constant_args])
        if exists_0:
            raise TypeInferenceError(source_pos_info, f'Invalid size value on `NDArray.all_ones`. All size values in all axis should be greater than 0')
        shape = tuple([arg for arg in constant_args])
        inferred_value = NDArrayHelper.fill(shape, lambda: 1)
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=inferred_value)

    @staticmethod
    def infer_NDArray_identity(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if len(args) != 0:
            raise AssertionError(source_pos_info, 'Invalid number of operands on operator `NDArray.identity`. All operands for this operator should be constants. Please check the transpiler design')
        if len(constant_args) != 1:
            raise TypeInferenceError(source_pos_info, 'Invalid number of operands on operator `NDArray.identity`. This operator generates a 2-dimensional identity ndarray (matrix). There should be only 1 constant operand passed to this operator')
        if constant_args[0] <= 0:
            raise TypeInferenceError(source_pos_info, f'Invalid size value on `NDArray.identity`. The size value should be greater than 0')
        inferred_value = list([list([0 if j != i else 1 for j in range(constant_args[0])]) for i in range(constant_args[0])])
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=NDArrayHelper((constant_args[0], constant_args[0]), inferred_value))

    @staticmethod
    def infer_concat(*args, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        for arg in args:
            assert isinstance(arg, IRInferenceDescriptor)
        if len(args) == 0:
            raise TypeInferenceError(source_pos_info, 'Not enough elements provided. To concat ndarray, there should be at least 1 element provided')
        for i, arg in enumerate(args):
            if not arg.is_ndarray():
                raise TypeInferenceError(source_pos_info, f"Cannot concat ndarray: all elements provided to concat should be ndarray")
            if i - 1 >= 0 and not args[i].datatype_matches(args[i - 1]):
                raise TypeInferenceError(source_pos_info, f'Cannot concat ndarray: the datatype of the {i - 1}-th element ({args[i - 1].pretty_typename()}) does not match the datatype of the {i}-th element ({args[i].pretty_typename()})')
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=NDArrayHelper.concat(*[arg.value for arg in args]))

    @staticmethod
    def infer_new_list(*args, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        for arg in args:
            assert isinstance(arg, IRInferenceDescriptor)
        if all([arg.is_number() for arg in args]):
            return IRInferenceDescriptor(DataTypeName.NDARRAY, value=NDArrayHelper((len(args), ), [arg.value for arg in args]))
        elif all([arg.is_ndarray() for arg in args]):
            return IRInference.infer_concat(*args, source_pos_info)
        raise TypeInferenceError(source_pos_info, 'Cannot create new ndarray: datatype mismatches in the provided elements')

    @staticmethod
    def infer_list(*args, source_pos_info: SourcePosInfo | None = None):
        if len(args) != 1:
            raise TypeInferenceError(source_pos_info, 'The `list` operator accepts only one argument.')
        assert isinstance(args[0], IRInferenceDescriptor)
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=args[0].value)

    @staticmethod
    def infer_range(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if len(args) != 0:
            raise AssertionError(source_pos_info, 'Invalid number of operands on operator `range`. All operands for this operator should be constants. Please check the transpiler design')
        if len(constant_args) == 0:
            raise TypeInferenceError(source_pos_info, 'Cannot create a `range` with no parameters')
        if len(constant_args) > 3:
            raise TypeInferenceError(source_pos_info, 'Too many parameters for `range`. There are at most 3 parameters for `range`')
        range_args = []
        for c_a in constant_args:
            range_args.append(c_a)
        the_range = list(range(*range_args))
        if len(the_range) == 0:
            raise TypeInferenceError(source_pos_info, f'Cannot create an empty range. This range is inferred as `range({", ".join([str(x) for x in range_args])})`')
        return IRInferenceDescriptor(DataTypeName.NDARRAY,  value=NDArrayHelper((len(the_range),), the_range))

    @staticmethod
    def infer_NDArray_sum(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        for arg in args:
            assert isinstance(arg, IRInferenceDescriptor)
        if len(args) != 1:
            raise TypeInferenceError(source_pos_info, f'Invalid number of operands on operator `NDArray.sum`. A ndarray should be provided to this operator')
        target = args[0]
        if not target.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'A ndarray should be provided to this operator. Here, a {target.pretty_typename()} is provided')
        if len(constant_args) == 1:
            axis = constant_args[0]
            if axis >= len(target.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid axis for operator `NDArray.sum`. The axis number exceeds total number of dimensions of the ndarray')
            inferred_value = target.value.accumulate(axis, lambda x, y: x + y if x is not None and y is not None else None, lambda: 0)
            if not isinstance(inferred_value, NDArrayHelper):
                return IRInferenceDescriptor(DataTypeName.NUMBER, value=inferred_value)
            return IRInferenceDescriptor(DataTypeName.NDARRAY, value=inferred_value)
        elif len(constant_args) == 0:
            inferred_value = target.value.accumulate(-1, lambda x, y: x + y if x is not None and y is not None else None, lambda: 0)
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=inferred_value)
        raise TypeInferenceError(source_pos_info, f'Invalid number of operands on operator `NDArray.sum`. The number of operands should be either 1 or 0')

    @staticmethod
    def infer_NDArray_any(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        for arg in args:
            assert isinstance(arg, IRInferenceDescriptor)
        if len(args) != 1:
            raise TypeInferenceError(source_pos_info, f'Invalid number of operands on operator `NDArray.any`. A ndarray should be provided to this operator')
        target = args[0]
        if not target.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'A ndarray should be provided to this operator. Here, a {target.pretty_typename()} is provided')
        if len(constant_args) == 1:
            axis = constant_args[0]
            if axis >= len(target.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid axis for operator `NDArray.any`. The axis number exceeds total number of dimensions of the ndarray')
            inferred_value = target.value.accumulate(axis, lambda x, y: (1 if x != 0 or y != 0 else 0) if x is not None and y is not None else None , lambda: 0)
            if not isinstance(inferred_value, NDArrayHelper):
                return IRInferenceDescriptor(DataTypeName.NUMBER, value=inferred_value)
            return IRInferenceDescriptor(DataTypeName.NDARRAY, value=inferred_value)
        elif len(constant_args) == 0:
            inferred_value = target.value.accumulate(-1, lambda x, y: (1 if x != 0 or y != 0 else 0) if x is not None and y is not None else None, lambda: 0)
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=inferred_value)
        raise TypeInferenceError(source_pos_info, f'Invalid number of operands on operator `NDArray.any`. The number of operands should be either 1 or 0')

    @staticmethod
    def infer_NDArray_all(*args, constant_args: List[int], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        for arg in args:
            assert isinstance(arg, IRInferenceDescriptor)
        if len(args) != 1:
            raise TypeInferenceError(source_pos_info, f'Invalid number of operands on operator `NDArray.all`. A ndarray should be provided to this operator')
        target = args[0]
        if not target.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'A ndarray should be provided to this operator. Here, a {target.pretty_typename()} is provided')
        if len(constant_args) == 1:
            axis = constant_args[0]
            if axis >= len(target.shape):
                raise TypeInferenceError(
                    source_pos_info, f'Invalid axis for operator `NDArray.all`. The axis number exceeds total number of dimensions of the ndarray')
            inferred_value = target.value.accumulate(axis, lambda x, y: (1 if x != 0 and y != 0 else 0) if x is not None and y is not None else None, lambda: 1)
            if not isinstance(inferred_value, NDArrayHelper):
                return IRInferenceDescriptor(DataTypeName.NUMBER, value=inferred_value)
            return IRInferenceDescriptor(DataTypeName.NDARRAY, value=inferred_value)
        elif len(constant_args) == 0:
            inferred_value = target.value.accumulate(-1, lambda x, y: (1 if x != 0 and y != 0 else 0) if x is not None and y is not None else None, lambda: 1)
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=inferred_value)
        raise TypeInferenceError(source_pos_info, f'Invalid number of operands on operator `NDArray.all`. The number of operands should be either 1 or 0')

    @staticmethod
    def infer_assert(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> None:
        if not operand.is_number():
            raise TypeInferenceError(source_pos_info, f'Invalid operand on operator `assert`. Operator `assert` only accepts a scalar number')
        return None

    @staticmethod
    def infer_expose_public(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> None:
        return None

    @staticmethod
    def infer_len(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if not operand.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'Invalid operand on operator `len`. Operator `len` only accepts a ndarray')
        return IRInferenceDescriptor(DataTypeName.NUMBER, value=operand.value.shape[0])

    @staticmethod
    def infer_NDArray_shape(operand: IRInferenceDescriptor, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        if not operand.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'Invalid operand on operator `shape`. Operator `shape` only accepts a ndarray')
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=NDArrayHelper((len(operand.value.shape), ), list(operand.value.shape)))

    @staticmethod
    def infer_slicing(operand: IRInferenceDescriptor, slicing_args: List[int | Tuple[int, int, int]], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        slicing_args = slicing_args.copy()
        if not operand.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'`slicing` operator can only be carried on `NDArray` variables')
        check_result = operand.value.check_slicing(slicing_args)
        if check_result is not None:
            raise TypeInferenceError(source_pos_info, f'Cannot perform slicing: {check_result}')
        sliced_value = operand.value.slice(slicing_args)
        if not isinstance(sliced_value, NDArrayHelper):
            return IRInferenceDescriptor(DataTypeName.NUMBER, value=sliced_value)
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=sliced_value)

    @staticmethod
    def infer_slicing_assign(assignee: IRInferenceDescriptor, value: IRInferenceDescriptor, slicing_assign_args: List[List[int | Tuple[int, int, int]]], source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        slicing_args = slicing_assign_args.copy()
        if not assignee.is_ndarray():
            raise TypeInferenceError(source_pos_info, f'`slicing` operator can only be carried on `NDArray` variables')
        if not assignee.value.check_slicing_assign(slicing_args, value.value):
            raise TypeInferenceError(source_pos_info, f'Invalid assignment with slicing. We only support slicing assignment between ndarray with the same shapes')
        return IRInferenceDescriptor(DataTypeName.NDARRAY, value=assignee.value.slice_assign(slicing_args, value.value))

    @staticmethod
    def infer_input(*args, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        raise NotImplementedError('Please make inference for `input` outside the IRInference class. Please check the transpiler design')

    @staticmethod
    def infer_read_int(*args, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        return IRInferenceDescriptor(DataTypeName.NUMBER, value=None)

    @staticmethod
    def infer_constant(constant_value: int, source_pos_info: SourcePosInfo | None = None) -> IRInferenceDescriptor:
        return IRInferenceDescriptor(DataTypeName.NUMBER, value=constant_value)

    @staticmethod
    def _infer_value_binary(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor, func: Callable[[int, int], int | None]) -> int | NDArrayHelper | None:
        if lhs.is_number() and rhs.is_number():
            if not lhs.has_inferred_value() or not rhs.has_inferred_value():
                return None
            return func(lhs.value, rhs.value)
        lhs_ndarray, rhs_ndarray = lhs.value, rhs.value
        if lhs.is_number():
            lhs_ndarray = NDArrayHelper((1, ), [lhs.value])
        if rhs.is_number():
            rhs_ndarray = NDArrayHelper((1, ), [rhs.value])
        lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_ndarray, rhs_ndarray)
        return lhs_ndarray.binary(rhs_ndarray, lambda x, y: None if x is None or y is None else func(x, y))

    @staticmethod
    def _infer_value_ndarray_ndarray_multiplication(lhs: IRInferenceDescriptor, rhs: IRInferenceDescriptor) -> NDArrayHelper:
        assert lhs.is_ndarray() and rhs.is_ndarray()
        assert lhs.has_inferred_value() and rhs.has_inferred_value()
        assert NDArrayHelper.matmul_shape_matches(lhs.value.shape, rhs.value.shape)
        def _initializer():
            return 0
        def _adder(x, y):
            if x is None or y is None:
                return None
            return x + y
        def _multiplier(x, y):
            if x is None or y is None:
                return None
            return x * y
        return NDArrayHelper.matmul(lhs.value, rhs.value, _adder, _multiplier, _initializer)

    @staticmethod
    def _infer_value_unary(operand: IRInferenceDescriptor, func: Callable[[int], int | None]) -> int | NDArrayHelper | None:
        if operand.is_number():
            if not operand.has_inferred_value():
                return None
            return func(operand.value)
        return NDArrayHelper.unary(operand.value, lambda x: None if x is None else func(x))

    @staticmethod
    def _infer_value_new_ndarray_all_same_values(shape: Tuple[int, ...], value: int) -> List:
        return NDArrayHelper.fill(shape, lambda: value).values

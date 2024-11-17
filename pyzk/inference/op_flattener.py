from typing import List

from pyzk.ir.ir_builder import IRBuilder
from pyzk.ir.ir_stmt import IRStatement
from pyzk.util.datatype_name import DataTypeName
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.op_name import OpName
from pyzk.util.prog_meta_data import ProgramMetadata, ProgramInputMetadata


class OperatorFlattenInfo:
    def __init__(self, typename: str, value: NDArrayHelper | int):
        self.typename = typename
        self.value = value

    def get(self) -> NDArrayHelper | int:
        return self.value


class OperatorFlattener:
    def __init__(self, ir_builder: IRBuilder, prog_meta_data: ProgramMetadata):
        self._ir_builder = ir_builder
        self._prog_meta_data = prog_meta_data

    def flatten(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo | None:
        op_name = stmt.op
        assert op_name != OpName.Special.INPUT
        method_name = 'flatten_' + op_name
        method = getattr(self, method_name, None)
        if method is None:
            new_arg_ptrs = []
            for arg in info_args:
                if isinstance(arg.value, NDArrayHelper):
                    raise NotImplementedError(f'Operator flatten rule for {stmt.op} must be implemented during transpiler design.')
                new_arg_ptrs.append(arg.get())
            ptr = self._ir_builder.create_similar(stmt, new_arg_ptrs)
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        return method(stmt, args, info_args)

    def flatten_assert(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> None:
        new_arg_ptrs = []
        for arg in info_args:
            if isinstance(arg.value, NDArrayHelper):
                raise NotImplementedError(f'Operator flatten rule for {stmt.op} must be implemented during transpiler design.')
            new_arg_ptrs.append(arg.get())
        self._ir_builder.create_similar(stmt, new_arg_ptrs)
        return None

    def flatten_expose_public(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> None:
        assert len(args) == len(info_args) == 1
        # for ptr in info_args[0].get_all():
        #     self._ir_builder.create_expose_public(ptr)
        return None

    def flatten_add(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_add(lhs_info.get(), rhs_info.get())
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert NDArrayHelper.broadcast_compatible(lhs_info.get().shape, rhs_info.get().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_add(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), NDArrayHelper((1,), [rhs_info.get()]))
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_add(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1,), [lhs_info.get()]), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_add(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_sub(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_sub(lhs_info.get(), rhs_info.get())
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert NDArrayHelper.broadcast_compatible(lhs_info.get().shape, rhs_info.get().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_sub(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), NDArrayHelper((1,), [rhs_info.get()]))
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_sub(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1,), [lhs_info.get()]), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_sub(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_mul(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_mul(lhs_info.get(), rhs_info.get())
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert NDArrayHelper.broadcast_compatible(lhs_info.get().shape, rhs_info.get().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_mul(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), NDArrayHelper((1,), [rhs_info.get()]))
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_mul(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1,), [lhs_info.get()]), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_mul(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_div(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_div(lhs_info.get(), rhs_info.get())
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert NDArrayHelper.broadcast_compatible(lhs_info.get().shape, rhs_info.get().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_div(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), NDArrayHelper((1,), [rhs_info.get()]))
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_div(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1,), [lhs_info.get()]), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_div(x, y))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_mat_mul(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        assert lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY
        assert len(lhs_info.value.shape) <= 2 and len(rhs_info.value.shape) <= 2
        result = NDArrayHelper.matmul(lhs_info.get(), rhs_info.get(), lambda x, y: self._ir_builder.create_add(x, y), lambda x, y: self._ir_builder.create_mul(x, y), lambda: self._ir_builder.create_constant(0))
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def flatten_usub(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        info = info_args[0]
        if info.typename == DataTypeName.NUMBER:
            return OperatorFlattenInfo(DataTypeName.NUMBER,
                self._ir_builder.create_sub(self._ir_builder.create_constant(0), info.get())
            )
        elif info.typename == DataTypeName.NDARRAY:
            result = info.value.unary(lambda x: self._ir_builder.create_sub(self._ir_builder.create_constant(0), x))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        else:
            raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_ne(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_compare(stmt, args, info_args, OpName.Binary.NE)

    def flatten_eq(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_compare(stmt, args, info_args, OpName.Binary.EQ)

    def flatten_gte(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_compare(stmt, args, info_args, OpName.Binary.GTE)

    def flatten_lte(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_compare(stmt, args, info_args, OpName.Binary.LTE)

    def flatten_gt(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_compare(stmt, args, info_args, OpName.Binary.GT)

    def flatten_lt(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_compare(stmt, args, info_args, OpName.Binary.LT)

    def flatten_NDArray_all_zeros(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 0
        assert len(stmt.constant_args) > 0
        zero_ptr = self._ir_builder.create_constant(0)
        result = NDArrayHelper.fill(tuple(stmt.constant_args), lambda: zero_ptr)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def flatten_NDArray_all_ones(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 0
        assert len(stmt.constant_args) > 0
        one_ptr = self._ir_builder.create_constant(1)
        result = NDArrayHelper.fill(tuple(stmt.constant_args), lambda: one_ptr)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def flatten_NDArray_identity(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 0
        assert len(stmt.constant_args) == 1
        size_n = stmt.constant_args[0]
        zero_ptr = self._ir_builder.create_constant(0)
        one_ptr = self._ir_builder.create_constant(1)
        val_ptrs = [[one_ptr if i == j else zero_ptr for j in range(size_n)] for i in range(size_n)]
        return OperatorFlattenInfo(DataTypeName.NDARRAY, NDArrayHelper((size_n, size_n), val_ptrs))

    def flatten_concat(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) > 0
        result = NDArrayHelper.concat(*[arg.value for arg in info_args])
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def flatten_list(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        return OperatorFlattenInfo(DataTypeName.NDARRAY, info_args[0].get())

    def flatten_new_list(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) > 0
        if all([isinstance(arg.value, NDArrayHelper) for arg in info_args]):
            result = NDArrayHelper.concat(*[arg.value for arg in info_args])
        else:
            result = NDArrayHelper((len(info_args), ), [arg.value for arg in info_args])
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def flatten_len(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        return OperatorFlattenInfo(DataTypeName.NUMBER, info_args[0].get().shape[0])

    def flatten_NDArray_shape(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        return OperatorFlattenInfo(DataTypeName.NDARRAY, NDArrayHelper((len(info_args[0].get().shape), ), list(info_args[0].get().shape)))

    def flatten_range(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 0
        assert 0 < len(stmt.constant_args) <= 3
        range_args = []
        for c_a in stmt.constant_args:
            range_args.append(c_a)
        the_range = list(range(*range_args))
        val_ptrs = []
        for val in the_range:
            val_ptrs.append(self._ir_builder.create_constant(val))
        return OperatorFlattenInfo(DataTypeName.NDARRAY, NDArrayHelper((len(the_range),), val_ptrs))

    def flatten_NDArray_sum(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_NDArray_binary_operator_over_axis(stmt, args, info_args, OpName.Binary.ADD, 0)

    def flatten_NDArray_any(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_NDArray_binary_operator_over_axis(stmt, args, info_args, OpName.Binary.OR, 0)

    def flatten_NDArray_all(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_NDArray_binary_operator_over_axis(stmt, args, info_args, OpName.Binary.AND, 1)

    def flatten_slicing(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        assert stmt.constant_args is None or len(stmt.constant_args) == 0
        assert len(stmt.slicing_args) > 0
        assert info_args[0].typename == DataTypeName.NDARRAY and len(info_args[0].value.shape) > 0
        new_values = info_args[0].value.slice(stmt.slicing_args)
        if isinstance(new_values, NDArrayHelper):
            return OperatorFlattenInfo(DataTypeName.NDARRAY, new_values)
        return OperatorFlattenInfo(DataTypeName.NUMBER, new_values)

    def flatten_slicing_assign(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        assert stmt.constant_args is None or len(stmt.constant_args) == 0
        assert len(stmt.slicing_assign_args) > 0
        assignee_info, value_info = info_args[0], info_args[1]
        assert assignee_info.typename == DataTypeName.NDARRAY and len(assignee_info.get().shape) > 0
        result = assignee_info.get().slice_assign(stmt.slicing_assign_args, value_info.get())
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def flatten_input(self, stmt: IRStatement) -> OperatorFlattenInfo:
        assert stmt.op == OpName.Special.INPUT
        assert len(stmt.constant_args) == 1 and len(stmt.args) == 0
        assert stmt.annotation is not None
        input_idx = stmt.constant_args[0]
        if stmt.annotation.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_read_int(input_idx, 0)
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        assert len(stmt.annotation.shape) > 0
        input_metadata: ProgramInputMetadata = self._prog_meta_data.inputs[input_idx]
        _filler_next_id = 0
        def _filler_func():
            nonlocal _filler_next_id
            _filler_next_id += 1
            return self._ir_builder.create_read_int(input_idx, _filler_next_id - 1)
        result = NDArrayHelper.fill(input_metadata.shape, _filler_func)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, result)

    def _flatten_compare(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo], op_name: str) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_op(op_name, [lhs_info.get(), rhs_info.get()])
            return OperatorFlattenInfo(DataTypeName.NUMBER, ptr)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert NDArrayHelper.broadcast_compatible(lhs_info.get().shape, rhs_info.get().shape)
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_op(op_name, [x, y]))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(lhs_info.get(), NDArrayHelper((1,), [rhs_info.get()]))
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_op(op_name, [x, y]))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            lhs_ndarray, rhs_ndarray = NDArrayHelper.broadcast(NDArrayHelper((1,), [lhs_info.get()]), rhs_info.get())
            result = lhs_ndarray.binary(rhs_ndarray, lambda x, y: self._ir_builder.create_op(op_name, [x, y]))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def _flatten_NDArray_binary_operator_over_axis(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo], op_name: str, initial_value: int) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        assert len(stmt.constant_args) in [0, 1]
        info = info_args[0]
        assert info.typename == DataTypeName.NDARRAY and len(info.value.shape) > 0
        if len(stmt.constant_args) == 0:
            result = info.value.accumulate(-1, lambda x, y: self._ir_builder.create_op(op_name, [x, y]), lambda: self._ir_builder.create_constant(initial_value))
            return OperatorFlattenInfo(DataTypeName.NUMBER, result)
        elif len(stmt.constant_args) == 1:
            result = info.value.accumulate(-1, lambda x, y: self._ir_builder.create_op(op_name, [x, y]), lambda: self._ir_builder.create_constant(initial_value))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, result)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

from typing import List, Tuple

from pyzk.ir.ir_builder import IRBuilder
from pyzk.ir.ir_stmt import IRStatement
from pyzk.util.datatype_name import DataTypeName
from pyzk.util.op_name import OpName
from pyzk.util.prog_meta_data import ProgramMetadata, ProgramInputMetadata


class OperatorFlattenInfo:
    def __init__(self, typename: str, shape: Tuple[int, ...], values: List[int]):
        self.typename = typename
        self.shape = shape
        self.values = values
        assert isinstance(values, List)
        if typename == DataTypeName.NDARRAY:
            total_number_elements = 1
            for axis in shape:
                total_number_elements *= axis
            assert len(values) == total_number_elements
        elif typename == DataTypeName.NUMBER:
            assert len(values) == 1 and len(shape) == 0
        else:
            raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def get(self) -> int:
        return self.values[0]

    def get_all(self) -> List[int]:
        return self.values

    def is_flat(self):
        return self.typename == DataTypeName.NUMBER

    def get_by_slicing(self, slicing: List[int | Tuple[int, int]]) -> List[int] | int:
        assert 0 < len(slicing) <= len(self.shape)
        for i, sli in enumerate(slicing):
            if isinstance(sli, int):
                assert 0 <= sli < self.shape[i]
            elif isinstance(sli, tuple):
                assert len(sli) == 2
                lo, hi = sli[0], sli[1]
                lo = 0 if lo is None else lo
                hi = self.shape[i] if hi is None else hi
                lo = self.shape[i] + lo if lo < 0 else lo
                hi = self.shape[i] + hi if hi < 0 else hi
                assert 0 <= lo < hi <= self.shape[i]
            else:
                raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')
        def _arr_filler(shape_idx: int, shape: Tuple[int, ...], val_idx: int) -> Tuple[List | int, int]:
            if shape_idx == len(shape):
                return self.values[val_idx], val_idx + 1
            n = shape[shape_idx]
            result = []
            for _i in range(n):
                _arr, val_idx = _arr_filler(shape_idx + 1, shape, val_idx)
                result.append(_arr)
            return result, val_idx
        values_arr, _ = _arr_filler(0, self.shape, 0)
        def _slicing_helper(axis: int, arr: List | int):
            if axis >= len(slicing) or axis == len(self.shape):
                return arr
            if isinstance(slicing[axis], int):
                return _slicing_helper(axis + 1, arr[slicing[axis]])
            elif isinstance(slicing[axis], tuple):
                _lo, _hi = slicing[axis][0], slicing[axis][1]
                ans = []
                for elm in arr[_lo:_hi]:
                    ans.append(_slicing_helper(axis + 1, elm))
                return ans
        def _arr_flattener(arr: List[int]) -> List[int]:
            result = []
            for elt in arr:
                if isinstance(elt, int):
                    result.append(elt)
                else:
                    result.extend(_arr_flattener(elt))
            return result
        slicing_result = _slicing_helper(0, values_arr)
        if isinstance(slicing_result, int):
            return slicing_result
        return _arr_flattener(slicing_result)


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
                if not arg.is_flat():
                    raise NotImplementedError(f'Operator flatten rule for {stmt.op} must be implemented during transpiler design.')
                new_arg_ptrs.append(arg.get())
            ptr = self._ir_builder.create_similar(stmt, new_arg_ptrs)
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [ptr])
        return method(stmt, args, info_args)

    def flatten_assert(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> None:
        new_arg_ptrs = []
        for arg in info_args:
            if not arg.is_flat():
                raise NotImplementedError(f'Operator flatten rule for {stmt.op} must be implemented during transpiler design.')
            new_arg_ptrs.append(arg.get())
        self._ir_builder.create_similar(stmt, new_arg_ptrs)
        return None

    def flatten_expose_public(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> None:
        assert len(args) == len(info_args) == 1
        for ptr in info_args[0].get_all():
            self._ir_builder.create_expose_public(ptr)
        return None

    def flatten_add(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_add(lhs_info.get(), rhs_info.get())
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [ptr])
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert lhs_info.shape == rhs_info.shape
            ptrs = []
            lhs_val_list, rhs_val_list = lhs_info.get_all(), rhs_info.get_all()
            for lhs_v, rhs_v in zip(lhs_val_list, rhs_val_list):
                ptr = self._ir_builder.create_add(lhs_v, rhs_v)
                ptrs.append(ptr)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            val_list = lhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_add(val, rhs_info.get()))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            val_list = rhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_add(lhs_info.get(), val))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, rhs_info.shape, ptrs)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_sub(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_sub(lhs_info.get(), rhs_info.get())
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [ptr])
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert lhs_info.shape == rhs_info.shape
            ptrs = []
            lhs_val_list, rhs_val_list = lhs_info.get_all(), rhs_info.get_all()
            for lhs_v, rhs_v in zip(lhs_val_list, rhs_val_list):
                ptr = self._ir_builder.create_sub(lhs_v, rhs_v)
                ptrs.append(ptr)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            val_list = lhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_sub(val, rhs_info.get()))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            val_list = rhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_sub(lhs_info.get(), val))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, rhs_info.shape, ptrs)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_mat_mul(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        assert lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY
        assert len(lhs_info.shape) <= 2 and len(rhs_info.shape) <= 2
        lhs_shape = lhs_info.shape if len(lhs_info.shape) == 2 else (lhs_info.shape[0], 1)
        rhs_shape = rhs_info.shape if len(rhs_info.shape) == 2 else (rhs_info.shape[0], 1)
        assert lhs_shape[1] == rhs_shape[0]
        output_shape = (lhs_shape[0], rhs_shape[1])
        output_shape_trim = len(rhs_info.shape) == 1
        value_ptrs = []
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                sum_ptr = self._ir_builder.create_constant(0)
                for k in range(lhs_shape[1]):
                    value_cell_l = lhs_info.get_by_slicing([i, k]) if len(lhs_info.shape) == 2 else lhs_info.get_by_slicing([i])
                    value_cell_r = rhs_info.get_by_slicing([k, j]) if len(rhs_info.shape) == 2 else rhs_info.get_by_slicing([k])
                    mul_ptr = self._ir_builder.create_mul(value_cell_l, value_cell_r)
                    sum_ptr = self._ir_builder.create_add(sum_ptr, mul_ptr)
                value_ptrs.append(sum_ptr)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, (output_shape[0],) if output_shape_trim else output_shape, value_ptrs)

    def flatten_mul(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            val_list = lhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_mul(val, rhs_info.get()))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            val_list = rhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_mul(lhs_info.get(), val))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, rhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [
                self._ir_builder.create_mul(lhs_info.get(), rhs_info.get())
            ])
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            val_list = rhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_mul(lhs_info.get(), val))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, rhs_info.shape, ptrs)
        else:
            raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_div(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == rhs_info.typename == DataTypeName.NUMBER:
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [
                self._ir_builder.create_div(lhs_info.get(), rhs_info.get())
            ])
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            val_list = rhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_div(lhs_info.get(), val))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, rhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            val_list = lhs_info.get_all()
            ptrs = []
            for val in val_list:
                ptrs.append(self._ir_builder.create_div(val, rhs_info.get()))
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        else:
            raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def flatten_usub(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        info = info_args[0]
        if info.typename == DataTypeName.NUMBER:
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [
                self._ir_builder.create_sub(self._ir_builder.create_constant(0), info.get())
            ])
        elif info.typename == DataTypeName.NDARRAY:
            ptrs = []
            val_list = info.get_all()
            zero_ptr = self._ir_builder.create_constant(0)
            for v in val_list:
                ptr = self._ir_builder.create_sub(zero_ptr, v)
                ptrs.append(ptr)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, info.shape, ptrs)
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
        element_amount = 1
        for sh in stmt.constant_args:
            element_amount *= sh
        zero_ptr = self._ir_builder.create_constant(0)
        val_ptrs = list([zero_ptr for _ in range(element_amount)])
        return OperatorFlattenInfo(DataTypeName.NDARRAY, tuple(stmt.constant_args), val_ptrs)

    def flatten_NDArray_all_ones(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 0
        assert len(stmt.constant_args) > 0
        element_amount = 1
        for sh in stmt.constant_args:
            element_amount *= sh
        one_ptr = self._ir_builder.create_constant(1)
        val_ptrs = list([one_ptr for _ in range(element_amount)])
        return OperatorFlattenInfo(DataTypeName.NDARRAY, tuple(stmt.constant_args), val_ptrs)

    def flatten_NDArray_identity(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 0
        assert len(stmt.constant_args) == 1
        size_n = stmt.constant_args[0]
        zero_ptr = self._ir_builder.create_constant(0)
        one_ptr = self._ir_builder.create_constant(1)
        val_ptrs = []
        for i in range(size_n):
            for j in range(size_n):
                if i == j:
                    val_ptrs.append(one_ptr)
                else:
                    val_ptrs.append(zero_ptr)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, (size_n, size_n), val_ptrs)

    def flatten_concat(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) > 0
        val_ptrs = []
        for info in info_args:
            val_ptrs.extend(info.get_all())
        return OperatorFlattenInfo(DataTypeName.NDARRAY, (len(args),) + info_args[0].shape, val_ptrs)

    def flatten_list(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        return OperatorFlattenInfo(DataTypeName.NDARRAY, info_args[0].shape, info_args[0].get_all())

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
        return OperatorFlattenInfo(DataTypeName.NDARRAY, (len(the_range),), val_ptrs)

    def flatten_NDArray_sum(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_NDArray_binary_operator_over_axis(stmt, args, info_args, OpName.Binary.ADD)

    def flatten_NDArray_any(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_NDArray_binary_operator_over_axis(stmt, args, info_args, OpName.Binary.OR)

    def flatten_NDArray_all(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        return self._flatten_NDArray_binary_operator_over_axis(stmt, args, info_args, OpName.Binary.AND)

    def flatten_slicing(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        assert stmt.constant_args is None or len(stmt.constant_args) == 0
        assert len(stmt.slicing_args) > 0
        info = info_args[0]
        assert info.typename == DataTypeName.NDARRAY and len(info.shape) > 0
        new_shape = []
        for i, sh in enumerate(info.shape):
            if i < len(stmt.slicing_args):
                sli = stmt.slicing_args[i]
                if isinstance(sli, Tuple):
                    lo, hi = sli
                    lo = 0 if lo is None else lo
                    hi = info.shape[i] if hi is None else hi
                    lo = info.shape[i] + lo if lo < 0 else lo
                    hi = info.shape[i] + hi if hi < 0 else hi
                    assert 0 <= lo < hi <= info.shape[i]
                    new_shape.append(hi - lo)
            else:
                new_shape.append(info.shape[i])
        new_values = info.get_by_slicing(stmt.slicing_args)
        if isinstance(new_values, int):
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [new_values])
        return OperatorFlattenInfo(DataTypeName.NDARRAY, tuple(new_shape), new_values)

    def flatten_slicing_assign(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo]) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        assert stmt.constant_args is None or len(stmt.constant_args) == 0
        assert len(stmt.slicing_assign_args) > 0
        assignee_info, value_info = info_args[0], info_args[1]
        assert assignee_info.typename == DataTypeName.NDARRAY and len(assignee_info.shape) > 0
        slicing_args = []
        for sub_slicing_arg in stmt.slicing_assign_args:
            assert len(sub_slicing_arg) > 0
            _axis = -1
            for sli in sub_slicing_arg:
                if isinstance(sli, Tuple):
                    lo, hi = sli
                    while True:
                        _axis += 1
                        if _axis >= len(slicing_args):
                            lo = 0 if lo is None else lo
                            hi = assignee_info.shape[_axis] if hi is None else hi
                            lo = assignee_info.shape[_axis] + lo if lo < 0 else lo
                            hi = assignee_info.shape[_axis] + hi if hi < 0 else hi
                            slicing_args.append((lo, hi))
                        elif isinstance(slicing_args[_axis], int):
                            continue
                        else:
                            e_lo, e_hi = slicing_args[_axis]
                            lo = e_lo if lo is None else e_lo + lo
                            hi = e_hi if hi is None else e_lo + hi
                            lo = e_lo + lo if lo < 0 else lo
                            hi = e_hi + hi if hi < 0 else hi
                            assert e_lo <= lo < hi <= e_hi
                            slicing_args[_axis] = (lo, hi)
                        break
                elif isinstance(sli, int):
                    while True:
                        _axis += 1
                        if _axis >= len(slicing_args):
                            slicing_args.append(sli)
                        elif isinstance(slicing_args[_axis], int):
                            continue
                        else:
                            e_lo, e_hi = slicing_args[_axis]
                            assert sli < e_hi - e_lo
                            slicing_args[_axis] = e_lo + sli
                        break
                else:
                    raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

        new_values = assignee_info.get_all().copy()
        value_ptrs = value_info.get_all()
        def _update_values_list_helper(axis: int, assignee_idx: int, value_idx: int, should_assign: bool) -> Tuple[int, int]:
            if axis == len(assignee_info.shape):
                if should_assign:
                    new_values[assignee_idx] = value_ptrs[value_idx]
                    return assignee_idx + 1, value_idx + 1
                return assignee_idx + 1, value_idx
            for i in range(assignee_info.shape[axis]):
                _should_assign = should_assign
                if axis < len(slicing_args):
                    _sli = slicing_args[axis]
                    if isinstance(_sli, int):
                        _should_assign = _should_assign and _sli == i
                    else:
                        _lo, _hi = _sli[0], _sli[1]
                        assert 0 <= _lo < _hi <= assignee_info.shape[axis]
                        _should_assign = _should_assign and _lo <= i < _hi
                assignee_idx, value_idx = _update_values_list_helper(axis + 1, assignee_idx, value_idx, _should_assign)
            return assignee_idx, value_idx

        _update_values_list_helper(0, 0, 0, True)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, assignee_info.shape, new_values)

    def flatten_input(self, stmt: IRStatement) -> OperatorFlattenInfo:
        assert stmt.op == OpName.Special.INPUT
        assert len(stmt.constant_args) == 1 and len(stmt.args) == 0
        assert stmt.annotation is not None
        input_idx = stmt.constant_args[0]
        if stmt.annotation.typename == DataTypeName.NUMBER:
            ptr = self._ir_builder.create_read_int(input_idx, 0)
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [ptr])
        assert len(stmt.annotation.shape) > 0
        input_metadata: ProgramInputMetadata = self._prog_meta_data.inputs[input_idx]
        total_number_elements = 1
        for sh in input_metadata.shape:
            total_number_elements *= sh
        ptrs_list = []
        for i in range(total_number_elements):
            ptr = self._ir_builder.create_read_int(input_idx, i)
            ptrs_list.append(ptr)
        return OperatorFlattenInfo(DataTypeName.NDARRAY, input_metadata.shape, ptrs_list)

    def _flatten_compare(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo], op_name: str) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 2
        lhs_info, rhs_info = info_args[0], info_args[1]
        if lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NUMBER:
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [
                self._ir_builder.create_op(op_name, [lhs_info.get(), rhs_info.get()])
            ])
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NDARRAY:
            assert lhs_info.shape == rhs_info.shape
            ptrs = []
            lhs_val_list, rhs_val_list = lhs_info.get_all(), rhs_info.get_all()
            for lhs_v, rhs_v in zip(lhs_val_list, rhs_val_list):
                ptr = self._ir_builder.create_op(op_name, [lhs_v, rhs_v])
                ptrs.append(ptr)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NUMBER and rhs_info.typename == DataTypeName.NDARRAY:
            ptrs = []
            val_list = rhs_info.get_all()
            for v in zip(val_list):
                ptr = self._ir_builder.create_op(op_name, [lhs_info.get(), v])
                ptrs.append(ptr)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, rhs_info.shape, ptrs)
        elif lhs_info.typename == DataTypeName.NDARRAY and rhs_info.typename == DataTypeName.NUMBER:
            ptrs = []
            val_list = lhs_info.get_all()
            for v in zip(val_list):
                ptr = self._ir_builder.create_op(op_name, [v, rhs_info.get()])
                ptrs.append(ptr)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, lhs_info.shape, ptrs)
        else:
            raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

    def _flatten_NDArray_binary_operator_over_axis(self, stmt: IRStatement, args: List[IRStatement], info_args: List[OperatorFlattenInfo], op_name: str) -> OperatorFlattenInfo:
        assert len(args) == len(info_args) == 1
        assert len(stmt.constant_args) in [0, 1]
        info = info_args[0]
        assert info.typename == DataTypeName.NDARRAY and len(info.shape) > 0
        if len(stmt.constant_args) == 0:
            cumulative_ptr = self._ir_builder.create_constant(0)
            val_list = info.get_all()
            for v in val_list:
                cumulative_ptr = self._ir_builder.create_op(op_name, [cumulative_ptr, v])
            return OperatorFlattenInfo(DataTypeName.NUMBER, tuple(), [cumulative_ptr])
        elif len(stmt.constant_args) == 1:
            axis = stmt.constant_args[0]
            assert axis < len(info.shape)
            amount_elements = 1
            for sh in info.shape:
                amount_elements *= sh
            cum_window = amount_elements // info.shape[0]
            for i in range(1, axis + 1):
                cum_window = cum_window // info.shape[i]
            cum_step = info.shape[axis]
            values_base = info.get_all()
            new_values = []
            assert amount_elements % (cum_step * cum_window) == 0
            for i in range(amount_elements // (cum_step * cum_window)):
                pos = i * cum_step * cum_window
                val_ptrs = [values_base[pos + j] for j in range(cum_window)]
                for j in range(cum_step):
                    for k in range(cum_window):
                        val_ptrs[k] = self._ir_builder.create_op(op_name, [val_ptrs[k], values_base[pos + k + j * cum_window]])
                new_values.extend(val_ptrs)
            new_shape = []
            for i, sh in enumerate(info.shape):
                if i != axis:
                    new_shape.append(sh)
            return OperatorFlattenInfo(DataTypeName.NDARRAY, tuple(new_shape), new_values)
        raise NotImplementedError('Oops! Something not implemented. Please check the transpiler design.')

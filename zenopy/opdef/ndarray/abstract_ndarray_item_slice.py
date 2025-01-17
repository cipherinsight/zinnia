from typing import List, Tuple, Optional

from zenopy.debug.dbg_info import DebugInfo
from zenopy.debug.exception import TypeInferenceError
from zenopy.opdef.nocls.abstract_item_slice import AbstractItemSliceOp
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import TupleValue, IntegerValue


class AbstractNDArrayItemSlice(AbstractItemSliceOp):
    def __init__(self):
        super().__init__()

    def check_slicing_dimensions(self, sps: List[TupleValue | IntegerValue], shape: Tuple[int, ...], dbg: Optional[DebugInfo]):
        if len(sps) > len(shape):
            raise TypeInferenceError(dbg, f"Too many indices for array: array is {len(shape)}-dimensional, but {len(sps)} were indexed")

    def find_all_candidates(self, reducer: AbsIRBuilderInterface, _sps: List[TupleValue | IntegerValue], _shape: Tuple[int, ...], dbg: Optional[DebugInfo]):
        sp, dim = _sps[0], _shape[0]
        if len(_sps) == 1:
            if isinstance(sp, IntegerValue):
                if sp.val() is not None:
                    self.check_single_slicing_number(sp, dim, dbg)
                    return [[sp.val()]], [reducer.ir_constant_int(1)]
                self.insert_slicing_number_assertion(sp, dim, reducer)
                return [[i] for i in range(dim)], [reducer.ir_equal_i(reducer.ir_constant_int(i), sp) for i in range(dim)]
            elif isinstance(sp, TupleValue):
                start, stop, step = sp.values()
                start = start.val() if isinstance(start, IntegerValue) else None
                stop = stop.val() if isinstance(stop, IntegerValue) else None
                step = step.val() if isinstance(step, IntegerValue) else None
                return [[(start, stop, step)]], [reducer.ir_constant_int(1)]
            raise NotImplementedError()
        _candidates, _conditions = self.find_all_candidates(reducer, _sps[1:], _shape[1:], dbg)
        if isinstance(sp, IntegerValue):
            if sp.val() is not None:
                self.check_single_slicing_number(sp, dim, dbg)
                _candidates = [[sp.val()] + x for x in _candidates]
                return _candidates, _conditions
            self.insert_slicing_number_assertion(sp, dim, reducer)
            _new_candidates, _new_conditions = [], []
            for i in range(dim):
                _new_candidates.extend([[i] + x for x in _candidates])
                _new_conditions.extend([reducer.ir_logical_and(x, reducer.ir_equal_i(reducer.ir_constant_int(i), sp)) for x in _conditions])
            return _new_candidates, _new_conditions
        elif isinstance(sp, TupleValue):
            start, stop, step = sp.values()
            start = start.val() if isinstance(start, IntegerValue) else None
            stop = stop.val() if isinstance(stop, IntegerValue) else None
            step = step.val() if isinstance(step, IntegerValue) else None
            _candidates = [[(start, stop, step)] + x for x in _candidates]
            return _candidates, _conditions
        raise NotImplementedError()

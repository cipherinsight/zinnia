import copy
from typing import List

from zinnia.compile.builder.builder_impl import IRBuilderImpl
from zinnia.compile.triplet import IntegerValue, FloatValue, Value
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.optim_pass.abstract_pass import AbstractIRPass
from zinnia.compile.triplet.value.boolean import BooleanValue
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.ir_def.defs.ir_add_f import AddFIR
from zinnia.ir_def.defs.ir_add_i import AddIIR
from zinnia.ir_def.defs.ir_div_f import DivFIR
from zinnia.ir_def.defs.ir_div_i import DivIIR
from zinnia.ir_def.defs.ir_logical_and import LogicalAndIR
from zinnia.ir_def.defs.ir_logical_or import LogicalOrIR
from zinnia.ir_def.defs.ir_mul_f import MulFIR
from zinnia.ir_def.defs.ir_mul_i import MulIIR
from zinnia.ir_def.defs.ir_select_b import SelectBIR
from zinnia.ir_def.defs.ir_select_f import SelectFIR
from zinnia.ir_def.defs.ir_select_i import SelectIIR
from zinnia.ir_def.defs.ir_sub_f import SubFIR
from zinnia.ir_def.defs.ir_sub_i import SubIIR


class ShortcutOptimIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def optimize_for_LogicalAndIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if lhs.val() is not None and lhs.val() == False:
            return ir_builder.ir_constant_bool(False)
        if rhs.val() is not None and rhs.val() == False:
            return ir_builder.ir_constant_bool(False)
        if lhs.val() is not None and lhs.val() != False:
            return rhs
        if rhs.val() is not None and rhs.val() != False:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_bool(True if (lhs.val() != False and rhs.val() != False) else False)
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_LogicalOrIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if lhs.val() is not None and lhs.val() != False:
            return ir_builder.ir_constant_bool(True)
        if rhs.val() is not None and rhs.val() != False:
            return ir_builder.ir_constant_bool(True)
        if lhs.val() is not None and lhs.val() == False:
            return rhs
        if rhs.val() is not None and rhs.val() == False:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_bool(True if (lhs.val() != False or rhs.val() != False) else False)
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_SelectIIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        cond, tv, fv = ir_args[0], ir_args[1], ir_args[2]
        assert isinstance(cond, BooleanValue)
        assert isinstance(tv, IntegerValue) and isinstance(fv, IntegerValue)
        if isinstance(tv, BooleanValue) and isinstance(fv, BooleanValue):
            return self.optimize_for_SelectBIR(ir_builder, ir_instance, ir_args)
        if cond.val() is not None and cond.val() != 0:
            if tv.val() is not None:
                return ir_builder.ir_constant_int(tv.val())
            return tv
        if cond.val() is not None and cond.val() == 0:
            if fv.val() is not None:
                return ir_builder.ir_constant_int(fv.val())
            return fv
        if tv.ptr() == fv.ptr():
            return tv
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_SelectBIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        cond, tv, fv = ir_args[0], ir_args[1], ir_args[2]
        assert isinstance(cond, BooleanValue)
        assert isinstance(tv, BooleanValue) and isinstance(fv, BooleanValue)
        if cond.val() is not None and cond.val() != 0:
            if tv.val() is not None:
                return ir_builder.ir_constant_int(tv.val())
            return tv
        if cond.val() is not None and cond.val() == 0:
            if fv.val() is not None:
                return ir_builder.ir_constant_int(fv.val())
            return fv
        if tv.ptr() == fv.ptr():
            return tv
        # if fv.val() is not None and fv.val() == True:
        #     return ir_builder.ir_logical_or(ir_builder.ir_logical_not(cond), tv)
        if fv.val() is not None and fv.val() == False:
            return ir_builder.ir_logical_and(cond, tv)
        if tv.val() is not None and tv.val() == True:
            return ir_builder.ir_logical_or(cond, fv)
        # if tv.val() is not None and tv.val() == False:
        #     return ir_builder.ir_logical_and(ir_builder.ir_logical_not(cond), fv)
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_SelectFIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        cond, tv, fv = ir_args[0], ir_args[1], ir_args[2]
        assert isinstance(cond, BooleanValue)
        assert isinstance(tv, FloatValue) and isinstance(fv, FloatValue)
        if cond.val() is not None and cond.val() != 0:
            if tv.val() is not None:
                return ir_builder.ir_constant_float(tv.val())
            return tv
        if cond.val() is not None and cond.val() == 0:
            if fv.val() is not None:
                return ir_builder.ir_constant_float(fv.val())
            return fv
        if tv.ptr() == fv.ptr():
            return tv
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_AddIIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if lhs.val() is not None and lhs.val() == 0:
            return rhs
        if rhs.val() is not None and rhs.val() == 0:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_int(lhs.val() + rhs.val())
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_AddFIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue)
        if lhs.val() is not None and lhs.val() == 0.0:
            return rhs
        if rhs.val() is not None and rhs.val() == 0.0:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_float(lhs.val() + rhs.val())
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_SubIIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if rhs.val() is not None and rhs.val() == 0:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_int(lhs.val() - rhs.val())
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_SubFIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue)
        if rhs.val() is not None and rhs.val() == 0.0:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_float(lhs.val() - rhs.val())
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_MulIIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if lhs.val() is not None and lhs.val() == 1:
            return rhs
        if rhs.val() is not None and rhs.val() == 1:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_int(lhs.val() * rhs.val())
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_MulFIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue)
        if lhs.val() is not None and lhs.val() == 1.0:
            return rhs
        if rhs.val() is not None and rhs.val() == 1.0:
            return lhs
        if lhs.val() is not None and rhs.val() is not None:
            return ir_builder.ir_constant_float(lhs.val() * rhs.val())
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_DivIIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue)
        if rhs.val() is not None and rhs.val() == 1:
            return lhs
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_for_DivFIR(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        lhs, rhs = ir_args[0], ir_args[1]
        assert isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue)
        if rhs.val() is not None and rhs.val() == 1.0:
            return lhs
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def optimize_ir(self, ir_builder: IRBuilderImpl, ir_instance: AbstractIR, ir_args: List[Value]) -> Value:
        if isinstance(ir_instance, LogicalAndIR):
            return self.optimize_for_LogicalAndIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, LogicalOrIR):
            return self.optimize_for_LogicalOrIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, SelectBIR):
            return self.optimize_for_SelectBIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, SelectIIR):
            return self.optimize_for_SelectIIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, SelectFIR):
            return self.optimize_for_SelectFIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, AddIIR):
            return self.optimize_for_AddIIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, AddFIR):
            return self.optimize_for_AddFIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, SubIIR):
            return self.optimize_for_SubIIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, SubFIR):
            return self.optimize_for_SubFIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, MulIIR):
            return self.optimize_for_MulIIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, MulFIR):
            return self.optimize_for_MulFIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, DivIIR):
            return self.optimize_for_DivIIR(ir_builder, ir_instance, ir_args)
        if isinstance(ir_instance, DivFIR):
            return self.optimize_for_DivFIR(ir_builder, ir_instance, ir_args)
        return ir_builder.create_ir(ir_instance, ir_args, None)

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        ir_graph = copy.copy(ir_graph)
        ir_builder = IRBuilderImpl()
        topological_order = ir_graph.get_topological_order(False)
        in_links, out_links = ir_graph.get_io_links()
        value_lookup_by_ptr = {}
        for stmt in topological_order:
            ir_args: List[Value] = [None for _ in in_links[stmt.stmt_id]]
            for i, old_ptr in enumerate(in_links[stmt.stmt_id]):
                ir_args[i] = value_lookup_by_ptr[old_ptr]
            value_lookup_by_ptr[stmt.stmt_id] = self.optimize_ir(ir_builder, stmt.ir_instance, ir_args)
        return ir_builder.export_ir_graph()


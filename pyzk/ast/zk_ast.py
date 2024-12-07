from typing import List, Tuple, Optional, Dict

from pyzk.internal.dt_descriptor import DTDescriptor
from pyzk.debug.dbg_info import DebugInfo

class ASTComponent:
    def __init__(self, dbg_i: DebugInfo):
        self.dbg_i = dbg_i


class ASTStatement(ASTComponent):
    def __init__(self, dbg_i: DebugInfo):
        super().__init__(dbg_i)


class ASTAnnotation(ASTComponent):
    def __init__(self, dbg_i: DebugInfo, dt: DTDescriptor, public: bool = False):
        super().__init__(dbg_i)
        self.dt = dt
        self.public = public


class ASTProgramInput(ASTComponent):
    def __init__(
        self,
        dbg_i: DebugInfo,
        public: bool,
        name: str,
        annotation: ASTAnnotation
    ):
        super().__init__(dbg_i)
        self.public = public
        self.name = name
        self.annotation = annotation


class ASTChipInput(ASTComponent):
    def __init__(
        self,
        dbg_i: DebugInfo,
        name: str,
        annotation: ASTAnnotation
    ):
        super().__init__(dbg_i)
        self.name = name
        self.annotation = annotation


class ASTChip(ASTComponent):
    def __init__(
        self,
        dbg_i: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTChipInput],
        return_anno: ASTAnnotation,
    ):
        super().__init__(dbg_i)
        self.block = block
        self.inputs = inputs
        self.return_anno = return_anno


class ASTProgram(ASTComponent):
    def __init__(
        self,
        dbg_i: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTProgramInput],
        chips: Dict[str, ASTChip]
    ):
        super().__init__(dbg_i)
        self.block = block
        self.inputs = inputs
        self.chips = chips


class ASTExpression(ASTComponent):
    def __init__(self, dbg_i: DebugInfo):
        super().__init__(dbg_i)


class ASTAbstractOperator(ASTExpression):
    def __init__(self, dbg_i: DebugInfo, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg_i)
        self.args = args
        self.kwargs = kwargs


class ASTBinaryOperator(ASTAbstractOperator):
    def __init__(self, dbg_i: DebugInfo, op_cls: type, lhs: ASTExpression, rhs: ASTExpression):
        super().__init__(dbg_i, [lhs, rhs], {})
        self.operator = op_cls()


class ASTUnaryOperator(ASTAbstractOperator):
    def __init__(self, dbg_i: DebugInfo, op_cls: type, operand: ASTExpression):
        super().__init__(dbg_i, [operand], {})
        self.operator = op_cls()


class ASTNamedAttribute(ASTAbstractOperator):
    def __init__(self, dbg_i: DebugInfo, target: Optional[str], member: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg_i, args, kwargs)
        self.target = target
        self.member = member


class ASTExprAttribute(ASTAbstractOperator):
    def __init__(self, dbg_i: DebugInfo, target: ASTExpression, member: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg_i, args, kwargs)
        self.target = target
        self.member = member


class ASTConstant(ASTExpression):
    def __init__(self, dbg_i: DebugInfo, value: int):
        super().__init__(dbg_i)
        self.value = value


class ASTLoad(ASTExpression):
    def __init__(self, dbg_i: DebugInfo, name: str):
        super().__init__(dbg_i)
        self.name = name


class ASTSlicingData(ASTComponent):
    def __init__(self, dbg_i: DebugInfo, data: List[ASTExpression | Tuple[ASTExpression, ASTExpression, ASTExpression]]):
        super().__init__(dbg_i)
        self.data = data


class ASTSlicingAssignData(ASTComponent):
    def __init__(self, dbg_i: DebugInfo, data: List[ASTSlicingData]):
        super().__init__(dbg_i)
        self.data = data


class ASTSlicing(ASTExpression):
    def __init__(self, dbg_i: DebugInfo, val: ASTExpression, slicing: ASTSlicingData):
        super().__init__(dbg_i)
        self.val = val
        self.slicing = slicing


class ASTSquareBrackets(ASTExpression):
    def __init__(self, dbg_i: DebugInfo, dim_size: int, values: List[ASTExpression]):
        super().__init__(dbg_i)
        self.dim_size = dim_size
        self.values = values


class ASTParenthesis(ASTExpression):
    def __init__(self, dbg_i: DebugInfo, dim_size: int, values: List[ASTExpression]):
        super().__init__(dbg_i)
        self.dim_size = dim_size
        self.values = values


class ASTAssignStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo, assignee: str, value: ASTExpression, annotation: Optional[ASTAnnotation]):
        super().__init__(dbg_i)
        self.assignee = assignee
        self.value = value
        self.annotation = annotation


class ASTPassStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo):
        super().__init__(dbg_i)


class ASTBreakStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo):
        super().__init__(dbg_i)


class ASTContinueStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo):
        super().__init__(dbg_i)


class ASTSlicingAssignStatement(ASTAssignStatement):
    def __init__(self, dbg_i: DebugInfo, assignee: str, slicing: ASTSlicingAssignData, value: ASTExpression, annotation: Optional[ASTAnnotation]):
        super().__init__(dbg_i, assignee, value, annotation)
        self.slicing = slicing


class ASTForStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo):
        super().__init__(dbg_i)


class ASTForInStatement(ASTForStatement):
    def __init__(self, dbg_i: DebugInfo, assignee: str, iter_expr: ASTExpression, block: List[ASTStatement]):
        super().__init__(dbg_i)
        self.assignee = assignee
        self.iter_expr = iter_expr
        self.block = block


class ASTCondStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo, cond: ASTExpression, t_block: List[ASTStatement], f_block: List[ASTStatement]):
        super().__init__(dbg_i)
        self.cond = cond
        self.t_block = t_block
        self.f_block = f_block


class ASTAssertStatement(ASTForStatement):
    def __init__(self, dbg_i: DebugInfo, expr: ASTExpression):
        super().__init__(dbg_i)
        self.expr = expr


class ASTReturnStatement(ASTStatement):
    def __init__(self, dbg_i: DebugInfo, expr: ASTExpression | None):
        super().__init__(dbg_i)
        self.expr = expr


class ASTCallStatement(ASTAbstractOperator, ASTStatement):
    def __init__(self, dbg_i: DebugInfo, name: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg_i, args, kwargs)
        self.name = name

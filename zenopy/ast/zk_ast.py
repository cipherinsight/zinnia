from typing import List, Tuple, Optional, Dict

from zenopy.internal.dt_descriptor import DTDescriptor
from zenopy.debug.dbg_info import DebugInfo


class ASTComponent:
    def __init__(self, dbg: DebugInfo):
        self.dbg = dbg


class ASTStatement(ASTComponent):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTAnnotation(ASTComponent):
    def __init__(self, dbg: DebugInfo, dt: DTDescriptor, public: bool = False):
        super().__init__(dbg)
        self.dt = dt
        self.public = public


class ASTProgramInput(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        public: bool,
        name: str,
        annotation: ASTAnnotation
    ):
        super().__init__(dbg)
        self.public = public
        self.name = name
        self.annotation = annotation


class ASTChipInput(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        name: str,
        annotation: ASTAnnotation
    ):
        super().__init__(dbg)
        self.name = name
        self.annotation = annotation


class ASTChip(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTChipInput],
        return_anno: ASTAnnotation,
    ):
        super().__init__(dbg)
        self.block = block
        self.inputs = inputs
        self.return_anno = return_anno


class ASTProgram(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        block: List[ASTStatement],
        inputs: List[ASTProgramInput],
        chips: Dict[str, ASTChip]
    ):
        super().__init__(dbg)
        self.block = block
        self.inputs = inputs
        self.chips = chips


class ASTExpression(ASTComponent):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTAbstractOperator(ASTExpression):
    def __init__(self, dbg: DebugInfo, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg)
        self.args = args
        self.kwargs = kwargs


class ASTBinaryOperator(ASTAbstractOperator):
    class Op:
        ADD = "add"
        SUB = "sub"
        MUL = "mul"
        DIV = "div"
        MOD = "mod"
        POW = "pow"
        FLOOR_DIV = "floor_div"
        MAT_MUL = "mat_mul"
        EQ = "eq"
        NE = "ne"
        LT = "lt"
        LTE = "lte"
        GT = "gt"
        GTE = "gte"
        AND = "and"
        OR = "or"

    def __init__(self, dbg: DebugInfo, op_type: str, lhs: ASTExpression, rhs: ASTExpression):
        super().__init__(dbg, [lhs, rhs], {})
        self.operator = op_type
        self.lhs = lhs
        self.rhs = rhs


class ASTUnaryOperator(ASTAbstractOperator):
    class Op:
        USUB = "usub"
        UADD = "uadd"
        NOT = "not"

    def __init__(self, dbg: DebugInfo, op_type: str, operand: ASTExpression):
        super().__init__(dbg, [operand], {})
        self.operator = op_type
        self.operand = operand


class ASTNamedAttribute(ASTAbstractOperator):
    def __init__(self, dbg: DebugInfo, target: Optional[str], member: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg, args, kwargs)
        self.target = target
        self.member = member


class ASTExprAttribute(ASTAbstractOperator):
    def __init__(self, dbg: DebugInfo, target: ASTExpression, member: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg, args, kwargs)
        self.target = target
        self.member = member


class ASTSlice(ASTComponent):
    def __init__(self, dbg: DebugInfo, data: List[ASTExpression | Tuple[ASTExpression, ASTExpression, ASTExpression]]):
        super().__init__(dbg)
        self.data = data


class ASTAssignTarget(ASTComponent):
    def __init__(self, dbg: DebugInfo, star: bool = False):
        super().__init__(dbg)
        self.star = star


class ASTNameAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, name: str, star: bool = False):
        super().__init__(dbg, star=star)
        self.name = name


class ASTSubscriptAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, target: ASTExpression, slicing: ASTSlice, star: bool = False):
        super().__init__(dbg, star=star)
        self.target = target
        self.slicing = slicing


class ASTTupleAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, targets: Tuple[ASTAssignTarget, ...], star: bool = False):
        super().__init__(dbg, star=star)
        self.targets = targets


class ASTListAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, targets: List[ASTAssignTarget], star: bool = False):
        super().__init__(dbg, star=star)
        self.targets = targets


class ASTGenerator(ASTComponent):
    def __init__(self, dbg: DebugInfo, target: ASTAssignTarget, _iter: ASTExpression, ifs: List[ASTExpression]):
        super().__init__(dbg)
        self.target = target
        self.iter = _iter
        self.ifs = ifs


class ASTGeneratorExp(ASTExpression):
    class Kind:
        LIST = "list"
        TUPLE = "tuple"

    def __init__(self, dbg: DebugInfo, elt: ASTExpression, generators: List[ASTGenerator], kind: str):
        super().__init__(dbg)
        self.elt = elt
        self.generators = generators
        self.kind = kind


class ASTCondExp(ASTExpression):
    def __init__(self, dbg: DebugInfo, cond: ASTExpression, t_expr: ASTExpression, f_expr: ASTExpression):
        super().__init__(dbg)
        self.cond = cond
        self.t_expr = t_expr
        self.f_expr = f_expr


class ASTConstantInteger(ASTExpression):
    def __init__(self, dbg: DebugInfo, value: int):
        super().__init__(dbg)
        self.value = value


class ASTConstantFloat(ASTExpression):
    def __init__(self, dbg: DebugInfo, value: float):
        super().__init__(dbg)
        self.value = value


class ASTConstantNone(ASTExpression):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTConstantString(ASTExpression):
    def __init__(self, dbg: DebugInfo, value: str):
        super().__init__(dbg)
        self.value = value


class ASTLoad(ASTExpression):
    def __init__(self, dbg: DebugInfo, name: str):
        super().__init__(dbg)
        self.name = name


class ASTSlicing(ASTExpression):
    def __init__(self, dbg: DebugInfo, val: ASTExpression, slicing: ASTSlice):
        super().__init__(dbg)
        self.val = val
        self.slicing = slicing


class ASTSquareBrackets(ASTExpression):
    def __init__(self, dbg: DebugInfo, dim_size: int, values: List[ASTExpression]):
        super().__init__(dbg)
        self.dim_size = dim_size
        self.values = values


class ASTParenthesis(ASTExpression):
    def __init__(self, dbg: DebugInfo, dim_size: int, values: List[ASTExpression]):
        super().__init__(dbg)
        self.dim_size = dim_size
        self.values = values


class ASTAssignStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, targets: List[ASTAssignTarget], value: ASTExpression):
        super().__init__(dbg)
        self.targets = targets
        self.value = value


class ASTPassStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTBreakStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTContinueStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTForStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)


class ASTForInStatement(ASTForStatement):
    def __init__(self, dbg: DebugInfo, target: ASTAssignTarget, iter_expr: ASTExpression, block: List[ASTStatement]):
        super().__init__(dbg)
        self.target = target
        self.iter_expr = iter_expr
        self.block = block


class ASTCondStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, cond: ASTExpression, t_block: List[ASTStatement], f_block: List[ASTStatement]):
        super().__init__(dbg)
        self.cond = cond
        self.t_block = t_block
        self.f_block = f_block


class ASTAssertStatement(ASTForStatement):
    def __init__(self, dbg: DebugInfo, expr: ASTExpression):
        super().__init__(dbg)
        self.expr = expr


class ASTReturnStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, expr: ASTExpression | None):
        super().__init__(dbg)
        self.expr = expr


class ASTCallStatement(ASTAbstractOperator, ASTStatement):
    def __init__(self, dbg: DebugInfo, name: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg, args, kwargs)
        self.name = name


class ASTExprStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, expr: ASTExpression):
        super().__init__(dbg)
        self.expr = expr

from typing import List, Tuple, Optional

from pyzk.util.source_pos_info import SourcePosInfo

class ASTComponent:
    def __init__(self, source_pos_info: SourcePosInfo):
        self.source_pos_info = source_pos_info


class ASTStatement(ASTComponent):
    def __init__(self, source_pos_info: SourcePosInfo):
        super().__init__(source_pos_info)


class ASTAnnotation(ASTComponent):
    def __init__(self, source_pos_info: SourcePosInfo, typename: str, shape: Tuple[int, ...], public: bool, **kwargs):
        super().__init__(source_pos_info)
        self.typename = typename
        self.shape = shape
        self.public = public


class ASTProgramInput(ASTComponent):
    def __init__(
        self,
        source_pos_info: SourcePosInfo,
        public: bool,
        name: str,
        annotation: ASTAnnotation
    ):
        super().__init__(source_pos_info)
        self.public = public
        self.name = name
        self.annotation = annotation


class ASTProgram(ASTComponent):
    def __init__(
        self,
        source_pos_info: SourcePosInfo,
        block: List[ASTStatement],
        inputs: List[ASTProgramInput]
    ):
        super().__init__(source_pos_info)
        self.block = block
        self.inputs = inputs


class ASTExpression(ASTComponent):
    def __init__(self, source_pos_info: SourcePosInfo):
        super().__init__(source_pos_info)


class ASTOperator(ASTExpression):
    def __init__(self, source_pos_info: SourcePosInfo, op: str, args: List[ASTExpression]):
        super().__init__(source_pos_info)
        self.op = op
        self.args = args


class ASTConstant(ASTExpression):
    def __init__(self, source_pos_info: SourcePosInfo, value: int):
        super().__init__(source_pos_info)
        self.value = value


class ASTLoad(ASTExpression):
    def __init__(self, source_pos_info: SourcePosInfo, name: str):
        super().__init__(source_pos_info)
        self.name = name


class ASTSlicingData(ASTComponent):
    def __init__(self, source_pos_info: SourcePosInfo, data: List[ASTExpression | Tuple[ASTExpression, ASTExpression]]):
        super().__init__(source_pos_info)
        self.data = data


class ASTSlicingAssignData(ASTComponent):
    def __init__(self, source_pos_info: SourcePosInfo, data: List[ASTSlicingData]):
        super().__init__(source_pos_info)
        self.data = data


class ASTSlicing(ASTExpression):
    def __init__(self, source_pos_info: SourcePosInfo, val: ASTExpression, slicing: ASTSlicingData):
        super().__init__(source_pos_info)
        self.val = val
        self.slicing = slicing


class ASTCreateNDArray(ASTExpression):
    def __init__(self, source_pos_info: SourcePosInfo, dim_size: int, values: List[ASTExpression]):
        super().__init__(source_pos_info)
        self.dim_size = dim_size
        self.values = values


class ASTAssignStatement(ASTStatement):
    def __init__(self, source_pos_info: SourcePosInfo, assignee: str, value: ASTExpression, annotation: Optional[ASTAnnotation]):
        super().__init__(source_pos_info)
        self.assignee = assignee
        self.value = value
        self.annotation = annotation


class ASTPassStatement(ASTStatement):
    def __init__(self, source_pos_info: SourcePosInfo):
        super().__init__(source_pos_info)


class ASTBreakStatement(ASTStatement):
    def __init__(self, source_pos_info: SourcePosInfo):
        super().__init__(source_pos_info)


class ASTContinueStatement(ASTStatement):
    def __init__(self, source_pos_info: SourcePosInfo):
        super().__init__(source_pos_info)


class ASTSlicingAssignStatement(ASTAssignStatement):
    def __init__(self, source_pos_info: SourcePosInfo, assignee: str, slicing: ASTSlicingAssignData, value: ASTExpression, annotation: Optional[ASTAnnotation]):
        super().__init__(source_pos_info, assignee, value, annotation)
        self.slicing = slicing


class ASTForStatement(ASTStatement):
    def __init__(self, source_pos_info: SourcePosInfo):
        super().__init__(source_pos_info)


class ASTForInStatement(ASTForStatement):
    def __init__(self, source_pos_info: SourcePosInfo, assignee: str, iter_expr: ASTExpression, block: List[ASTStatement]):
        super().__init__(source_pos_info)
        self.assignee = assignee
        self.iter_expr = iter_expr
        self.block = block


class ASTCondStatement(ASTStatement):
    def __init__(self, source_pos_info: SourcePosInfo, cond: ASTExpression, t_block: List[ASTStatement], f_block: List[ASTStatement]):
        super().__init__(source_pos_info)
        self.cond = cond
        self.t_block = t_block
        self.f_block = f_block


class ASTAssertStatement(ASTForStatement):
    def __init__(self, source_pos_info: SourcePosInfo, expr: ASTExpression):
        super().__init__(source_pos_info)
        self.expr = expr

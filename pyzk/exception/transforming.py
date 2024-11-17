from pyzk.exception.base import InternalPyzkException
from pyzk.util.source_pos_info import SourcePosInfo


class ASTTransformingException(InternalPyzkException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidCircuitInputException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidCircuitStatementException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidProgramException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidAssignStatementException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidAnnotationException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidSlicingException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class InvalidForStatementException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class UnsupportedOperatorException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class UnsupportedConstantLiteralException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)


class UnsupportedLangFeatureException(ASTTransformingException):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(source_pos, msg, *args)

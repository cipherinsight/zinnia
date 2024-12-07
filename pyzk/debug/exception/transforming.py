from pyzk.debug.exception.base import InternalPyzkException
from pyzk.debug.dbg_info import DebugInfo


class ASTTransformingException(InternalPyzkException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidCircuitInputException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidChipInputException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidCircuitStatementException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidProgramException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidAssignStatementException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidAnnotationException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidSlicingException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InvalidForStatementException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class UnsupportedOperatorException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class UnsupportedConstantLiteralException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class UnsupportedLangFeatureException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class OperatorOrChipNotFoundException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class StatementNoEffectException(ASTTransformingException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)

from zenopy.debug.exception.base import InternalZenoPyException
from zenopy.debug.dbg_info import DebugInfo


class ContextualException(InternalZenoPyException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class VariableNotFoundError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class ConstantInferenceError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class StaticInferenceError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class TypeInferenceError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class OperatorCallError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class NoForElementsError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class NotInLoopError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class InterScopeError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class UnreachableStatementError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class ControlEndWithoutReturnError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class ReturnDatatypeMismatchError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class ChipArgumentsError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class TupleUnpackingError(ContextualException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)

from pyzk.exception.base import InternalPyzkException
from pyzk.util.source_pos_info import SourcePosInfo


class ContextualException(InternalPyzkException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class VariableNotFoundError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class ConstantInferenceError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class StaticInferenceError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class TypeInferenceError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class OperatorCallError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class NoForElementsError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class NotInLoopError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)


class InterScopeError(ContextualException):
    def __init__(self, spi: SourcePosInfo | None, msg: str, *args):
        super().__init__(spi, msg, *args)

from zenopy.debug.exception.base import InternalZenoPyException
from zenopy.debug.dbg_info import DebugInfo


class ZKExecutionException(InternalZenoPyException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class ZKCircuitParameterException(ZKExecutionException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)

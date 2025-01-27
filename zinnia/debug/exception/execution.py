from zinnia.debug.exception.base import InternalZinniaException
from zinnia.debug.dbg_info import DebugInfo


class ZKExecutionException(InternalZinniaException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)


class ZKCircuitParameterException(ZKExecutionException):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(dbg_i, msg, *args)

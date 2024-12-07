from pyzk.debug.dbg_info import DebugInfo


class InternalPyzkException(Exception):
    def __init__(self, dbg_i: DebugInfo | None, msg: str, *args):
        super().__init__(msg, *args)
        self.dbg_i = dbg_i
        self.msg = msg


class PyZKException(Exception):
    def __init__(self, prettified_error_message: str):
        super().__init__(prettified_error_message)

from pyzk.util.source_pos_info import SourcePosInfo


class InternalPyzkException(Exception):
    def __init__(self, source_pos: SourcePosInfo | None, msg: str, *args):
        super().__init__(msg, *args)
        self.source_pos = source_pos
        self.msg = msg


class PyZKException(Exception):
    def __init__(self, prettified_error_message: str):
        super().__init__(prettified_error_message)

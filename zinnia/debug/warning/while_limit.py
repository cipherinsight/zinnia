from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.prettifier import prettify_debug_info


class LoopLimitReachedWarning(Warning):
    def __init__(self, dbg: DebugInfo, limit: int):
        self.limit = limit
        self.dbg = dbg

    def __str__(self):
        return f"While loop reached limit {self.limit}. The program may not work as expected. Please consider increasing the loop limit or hard code the loop limit in the program.\n" + prettify_debug_info(self.dbg)

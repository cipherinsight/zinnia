from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.prettifier import prettify_debug_info


class RecursionLimitReachedWarning(Warning):
    def __init__(self, dbg: DebugInfo, limit: int):
        self.limit = limit
        self.dbg = dbg

    def __str__(self):
        return f"Recursion reached limit {self.limit}. The program may not work as expected. Please consider increasing the recursion limit or hard code the recursion limit in the program.\n" + prettify_debug_info(self.dbg)

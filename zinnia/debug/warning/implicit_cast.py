from zinnia.compile.type_sys import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.prettifier import prettify_debug_info


class ImplicitCastWarning(Warning):
    def __init__(self, dbg: DebugInfo, from_type: DTDescriptor, to_type: DTDescriptor):
        self.from_type = from_type
        self.to_type = to_type
        self.dbg = dbg

    def __str__(self):
        return f"Implicit cast from {self.from_type} to {self.to_type}.\n" + prettify_debug_info(self.dbg)

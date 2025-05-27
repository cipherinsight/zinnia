from typing import Callable

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor


class InternalExternalFuncObject:
    def __init__(self, name: str, _callable: Callable, return_dt: DTDescriptor):
        self.name = name
        self.callable = _callable
        self.return_dt = return_dt

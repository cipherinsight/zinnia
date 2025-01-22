from typing import Callable

from zenopy.internal.dt_descriptor import DTDescriptor


class ExternalFuncObj:
    def __init__(self, name: str, _callable: Callable, return_dt: DTDescriptor):
        self.name = name
        self.callable = _callable
        self.return_dt = return_dt

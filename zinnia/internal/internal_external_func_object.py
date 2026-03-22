from typing import Callable


class InternalExternalFuncObject:
    def __init__(self, name: str, _callable: Callable, return_dt: dict):
        self.name = name
        self.callable = _callable
        self.return_dt = return_dt

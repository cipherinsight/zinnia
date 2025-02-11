from typing import Dict

from zinnia.compile.triplet import Value, IntegerValue


class OpArgsContainer:
    def __init__(
            self,
            kwargs: Dict[str, Value],
            condition: IntegerValue = None
    ):
        self.kwargs = kwargs
        self.condition = condition

    def __getitem__(self, item):
        return self.kwargs[item]

    def __setitem__(self, key, value):
        raise NotImplementedError("OpArgsContainer is read-only")

    def __delitem__(self, key):
        raise NotImplementedError("OpArgsContainer is read-only")

    def get(self, key, default=None):
        return self.kwargs.get(key, default)

    def get_kwargs(self) -> Dict[str, Value]:
        return self.kwargs

    def get_condition(self) -> IntegerValue:
        return self.condition

    def has_condition(self) -> bool:
        return self.condition is not None


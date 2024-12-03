from typing import Tuple


class DTDescriptor(object):
    def __init__(self, typename: str):
        self.typename = typename

    def __new__(cls, *args, **kwargs):
        if cls is DTDescriptor:
            raise TypeError(f"<DTDescriptor> must be subclassed.")
        return object.__new__(cls)

    def __str__(self) -> str:
        return self.typename

    def __eq__(self, other) -> bool:
        return self.typename == other.typename


class NDArrayDTDescriptor(DTDescriptor):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__("NDArray")
        self.shape = shape

    def __str__(self) -> str:
        return f'{self.typename}[{",".join([str(x) for x in self.shape])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.shape == other.shape



class TupleDTDescriptor(DTDescriptor):
    def __init__(self, length: int):
        super().__init__("Tuple")
        self.length = length

    def __str__(self) -> str:
        return f'{self.typename}[{self.length}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.length == other.length


class NumberDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__("Number")


class NoneDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__("None")


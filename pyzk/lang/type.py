from dataclasses import field
from typing import Tuple

from pyzk.lang.metatype import NDArrayMeta

Number = int

class NDArray(metaclass=NDArrayMeta):
    shape: int

    def __init__(self):
        pass

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> 'NDArray':
        return NDArray()

    @staticmethod
    def ones(shape: Tuple[int, ...]) -> 'NDArray':
        return NDArray()

    @staticmethod
    def identity(n: int) -> 'NDArray':
        return NDArray()

    @staticmethod
    def eye(n: int, m: int) -> 'NDArray':
        return NDArray()

    def sum(self, axis: int = None) -> Number:
        raise NotImplementedError("Cannot perform action outside circuit method.")

    def all(self, axis: int = None) -> Number:
        raise NotImplementedError("Cannot perform action outside circuit method.")

    def any(self, axis: int = None) -> Number:
        raise NotImplementedError("Cannot perform action outside circuit method.")

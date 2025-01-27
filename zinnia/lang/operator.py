from typing import Any

from zinnia.lang.type import Float, Integer, NDArray, Tuple, List


def add(lhs: Any, rhs: Any) -> Any:
    pass

def sub(lhs: Any, rhs: Any) -> Any:
    pass

def mul(lhs: Any, rhs: Any) -> Any:
    pass

def div(lhs: Any, rhs: Any) -> Any:
    pass

def sin(x: Any) -> Any:
    pass

def cos(x: Any) -> Any:
    pass

def tan(x: Any) -> Any:
    pass

def sinh(x: Any) -> Any:
    pass

def cosh(x: Any) -> Any:
    pass

def tanh(x: Any) -> Any:
    pass

def exp(x: Any) -> Any:
    pass

def log(x: Any) -> Any:
    pass

def concatenate(arrays: Tuple, axis: Integer = 0) -> NDArray:
    pass

def stack(arrays: Tuple, axis: Integer = 0) -> NDArray:
    pass

from typing import Union
from zenopy.lang.type import *

def add(
    lhs: Integer | Float | NDArray,
    rhs: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def sub(
    lhs: Integer | Float | NDArray,
    rhs: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def mul(
    lhs: Integer | Float | NDArray,
    rhs: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def div(
    lhs: Integer | Float | NDArray,
    rhs: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def sin(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def cos(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def tan(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def sinh(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def cosh(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def tanh(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def exp(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def log(
    x: Integer | Float | NDArray
) -> Union[Integer | Float, NDArray]:
    pass

def concatenate(
    *args, axis: Integer = 0
) -> NDArray:
    pass

def stack(
    *args, axis: Integer = 0
) -> NDArray:
    pass

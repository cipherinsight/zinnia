from typing import Any

from zinnia.lang.type import Float, Integer, NDArray, Tuple, List
from zinnia.lang.poseidon_params import (
    POSEIDON_ALPHA,
    POSEIDON_BN254_FIELD_MODULUS,
    POSEIDON_FULL_ROUNDS,
    POSEIDON_PARTIAL_ROUNDS,
    POSEIDON_PRIME_BIT_LEN,
    POSEIDON_RATE,
    POSEIDON_SECURITY_LEVEL,
    POSEIDON_T,
)

try:
    import numpy as _np
except ImportError:
    _np = None

try:
    import poseidon as _poseidon
except ImportError:
    _poseidon = None


_POSEIDON_FIELD = POSEIDON_BN254_FIELD_MODULUS
_POSEIDON_MDS_MATRIX: list[list[int]] | None = None
_POSEIDON_ROUND_CONSTANTS: list[int] | None = None
_POSEIDON_HALF_FULL_ROUNDS = POSEIDON_FULL_ROUNDS // 2
_POSEIDON_CAPACITY_INIT = 1 << 64


def _init_poseidon_constants():
    global _POSEIDON_MDS_MATRIX, _POSEIDON_ROUND_CONSTANTS

    if _POSEIDON_MDS_MATRIX is not None and _POSEIDON_ROUND_CONSTANTS is not None:
        return
    if _poseidon is None:
        raise ImportError("`poseidon-hash` is required for pure-python `poseidon_hash` execution")

    def _to_int(value: Any) -> int:
        if isinstance(value, str):
            return int(value, 16)
        return int(value)

    _POSEIDON_MDS_MATRIX = [
        [_to_int(cell) % _POSEIDON_FIELD for cell in row]
        for row in _poseidon.matrix_254
    ]
    _POSEIDON_ROUND_CONSTANTS = [
        _to_int(constant) % _POSEIDON_FIELD
        for constant in _poseidon.round_constants_254
    ]


def _poseidon_permute(state: list[int]) -> list[int]:
    _init_poseidon_constants()
    assert _POSEIDON_MDS_MATRIX is not None
    assert _POSEIDON_ROUND_CONSTANTS is not None

    rc_index = 0

    def _apply_mds(cur_state: list[int]) -> list[int]:
        out: list[int] = []
        for row in _POSEIDON_MDS_MATRIX:
            acc = 0
            for coeff, value in zip(row, cur_state):
                acc = (acc + (coeff * value)) % _POSEIDON_FIELD
            out.append(acc)
        return out

    for _ in range(_POSEIDON_HALF_FULL_ROUNDS):
        for i in range(POSEIDON_T):
            state[i] = (state[i] + _POSEIDON_ROUND_CONSTANTS[rc_index]) % _POSEIDON_FIELD
            rc_index += 1
            state[i] = pow(state[i], POSEIDON_ALPHA, _POSEIDON_FIELD)
        state = _apply_mds(state)

    for _ in range(POSEIDON_PARTIAL_ROUNDS):
        for i in range(POSEIDON_T):
            state[i] = (state[i] + _POSEIDON_ROUND_CONSTANTS[rc_index]) % _POSEIDON_FIELD
            rc_index += 1
        state[0] = pow(state[0], POSEIDON_ALPHA, _POSEIDON_FIELD)
        state = _apply_mds(state)

    for _ in range(_POSEIDON_HALF_FULL_ROUNDS):
        for i in range(POSEIDON_T):
            state[i] = (state[i] + _POSEIDON_ROUND_CONSTANTS[rc_index]) % _POSEIDON_FIELD
            rc_index += 1
            state[i] = pow(state[i], POSEIDON_ALPHA, _POSEIDON_FIELD)
        state = _apply_mds(state)

    return state


def _flatten_zinnia_ndarray(value: NDArray) -> list[Any]:
    flattened: list[Any] = []

    def _walk(indices: tuple[int, ...]):
        if len(indices) == len(value.shape):
            flattened.append(value[indices])
            return
        for index in range(value.shape[len(indices)]):
            _walk(indices + (index,))

    _walk(())
    return flattened


def _normalize_scalar(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"poseidon_hash(float) only supports integer-valued floats, got {value}")
        return int(value)
    raise TypeError(f"poseidon_hash does not support scalar type `{type(value).__name__}`")


def _poseidon_hash_of_scalars(values: list[int]) -> int:
    field_values = [value % _POSEIDON_FIELD for value in values]

    state = [_POSEIDON_CAPACITY_INIT % _POSEIDON_FIELD] + [0] * (POSEIDON_T - 1)
    for start in range(0, len(field_values), POSEIDON_RATE):
        chunk = field_values[start:start + POSEIDON_RATE]
        for i, value in enumerate(chunk):
            state[i + 1] = (state[i + 1] + value) % _POSEIDON_FIELD
        if len(chunk) < POSEIDON_RATE:
            state[len(chunk) + 1] = (state[len(chunk) + 1] + 1) % _POSEIDON_FIELD
        state = _poseidon_permute(state)

    if len(field_values) % POSEIDON_RATE == 0:
        state[1] = (state[1] + 1) % _POSEIDON_FIELD
        state = _poseidon_permute(state)

    return state[1] % _POSEIDON_FIELD


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

def argmax(array: NDArray, axis: Integer = None) -> Any:
    pass

def argmin(array: NDArray, axis: Integer = None) -> Any:
    pass


def poseidon_hash(x: Any) -> Integer:
    if isinstance(x, (bool, int, float)):
        return _poseidon_hash_of_scalars([_normalize_scalar(x)])

    if _np is not None and isinstance(x, _np.ndarray):
        return _poseidon_hash_of_scalars([_normalize_scalar(value.item() if hasattr(value, "item") else value) for value in x.flatten()])

    if isinstance(x, NDArray):
        return _poseidon_hash_of_scalars([_normalize_scalar(value) for value in _flatten_zinnia_ndarray(x)])

    if isinstance(x, (tuple, list)):
        hashed_values = [poseidon_hash(value) for value in x]
        return _poseidon_hash_of_scalars(hashed_values)

    raise TypeError(f"poseidon_hash does not support value of type `{type(x).__name__}`")


def merkle_verify(leaf: Integer, root: Integer, siblings: Any, directions: Any) -> bool:
    pass

from typing import List, Any

from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.debug.exception import ZinniaException


def build_circuit_inputs(program_inputs: List[ZKProgramInput], args: tuple) -> dict:
    """Build a structured V2 witness from user arguments.

    Returns a dict ready for JSON serialization matching Rust's CircuitInputs:
    {
        "params": [
            {
                "name": "x",
                "is_public": true,
                "dtype": "Integer",
                "value": {"Int": 42}
            },
            ...
        ]
    }
    """
    if len(args) != len(program_inputs):
        raise ZinniaException(
            f"Expected {len(program_inputs)} arguments, got {len(args)}"
        )

    params = []
    for i, (pi, arg) in enumerate(zip(program_inputs, args)):
        params.append({
            "name": pi.name,
            "is_public": pi.kind == "Public",
            "dtype": pi.dt,
            "value": _build_input_node(arg, pi.dt, pi.name),
        })
    return {"params": params}


def _build_input_node(value: Any, dt, name: str) -> dict:
    """Recursively build an InputNode dict from a Python value."""
    variant, data = _get_type_variant(dt)

    if variant == "Integer":
        return {"Int": _coerce_integer(value, name)}

    elif variant == "Float":
        return {"Float": _coerce_float(value, name)}

    elif variant == "NDArray":
        shape = data["shape"]
        dtype = data["dtype"]
        flat = _flatten_ndarray(value, shape, name)
        elements = [_build_input_node(elem, dtype, name) for elem in flat]
        return {"Array": {"shape": shape, "elements": elements}}

    elif variant == "SymbolicNDArray":
        concrete_dt = _concretize_symbolic_ndarray(data, name)
        return _build_input_node(value, concrete_dt, name)

    elif variant == "List":
        elements_dt = data["elements"]
        if not isinstance(value, (list, tuple)):
            raise ZinniaException(f"Expected list for `{name}`, got {type(value).__name__}")
        if len(value) != len(elements_dt):
            raise ZinniaException(f"Expected list of length {len(elements_dt)} for `{name}`, got {len(value)}")
        elems = [_build_input_node(e, edt, name) for e, edt in zip(value, elements_dt)]
        return {"Sequence": elems}

    elif variant == "Tuple":
        elements_dt = data["elements"]
        if not isinstance(value, (list, tuple)):
            raise ZinniaException(f"Expected tuple for `{name}`, got {type(value).__name__}")
        if len(value) != len(elements_dt):
            raise ZinniaException(f"Expected tuple of length {len(elements_dt)} for `{name}`, got {len(value)}")
        elems = [_build_input_node(e, edt, name) for e, edt in zip(value, elements_dt)]
        return {"Sequence": elems}

    elif variant == "PoseidonHashed":
        from zinnia.lang.type import PoseidonHashed as PoseidonHashedType
        from zinnia.compile._bridge import poseidon_hash as _rust_poseidon_hash
        inner_dt = data["dtype"]
        if isinstance(value, PoseidonHashedType):
            actual = value.actual_value
        else:
            actual = value
        inner_node = _build_input_node(actual, inner_dt, name)
        # Compute hash from leaf scalars
        scalars = _collect_scalars(inner_node)
        hash_hex = _rust_poseidon_hash(scalars)
        return {"Hashed": {"inner": inner_node, "hash": hash_hex}}

    elif variant == "DynamicNDArray":
        dtype = data["dtype"]
        max_length = data["max_length"]
        max_rank = data.get("max_rank", 1)
        max_shape = data.get("max_shape")
        flat, runtime_shape = _flatten_bounded_dynarray_with_shape(
            value, max_length, max_rank, dtype, name, max_shape
        )
        elements = [_build_input_node(elem, dtype, name) for elem in flat]
        return {
            "Array": {
                "shape": [max_length],
                "elements": elements,
                "runtime_shape": runtime_shape,
            }
        }

    elif variant in ("Complex", "String"):
        raise ZinniaException(
            f"Cannot build witness for `{name}`: `{variant}` values are not "
            f"supported as circuit inputs."
        )

    else:
        raise ZinniaException(f"Unsupported type `{variant}` for `{name}`")


def _collect_scalars(node: dict) -> list:
    """Collect all scalar integer values from an InputNode tree (for hash computation)."""
    if "Int" in node:
        return [node["Int"]]
    elif "Float" in node:
        # Floats shouldn't appear in hash inputs typically, but handle it
        return [int(node["Float"])]
    elif "Bool" in node:
        return [int(node["Bool"])]
    elif "Sequence" in node:
        result = []
        for elem in node["Sequence"]:
            result.extend(_collect_scalars(elem))
        return result
    elif "Array" in node:
        result = []
        for elem in node["Array"]["elements"]:
            result.extend(_collect_scalars(elem))
        return result
    elif "Hashed" in node:
        return _collect_scalars(node["Hashed"]["inner"])
    return []


def _get_type_variant(dt):
    """Extract the type variant name and data from a ZinniaType serde dict."""
    if isinstance(dt, str):
        return dt, {}
    if isinstance(dt, dict) and len(dt) == 1:
        variant = next(iter(dt))
        return variant, dt[variant]
    raise ZinniaException(f"Invalid type descriptor: {dt}")


def _concretize_symbolic_ndarray(data: dict, name: str) -> dict:
    shape, unresolved = _concretize_symbolic_shape(data["shape"], data.get("bindings") or {})
    if unresolved:
        raise ZinniaException(
            f"Cannot build witness for `{name}` because symbolic shape "
            f"dimension(s) {', '.join(unresolved)} are unresolved."
        )
    return {"NDArray": {"shape": shape, "dtype": data["dtype"]}}


def _concretize_symbolic_shape(shape_spec: list, bindings: dict) -> tuple[list[int], list[str]]:
    shape = []
    unresolved = []
    for dim in shape_spec:
        if isinstance(dim, int):
            shape.append(dim)
        elif isinstance(dim, str) and dim in bindings:
            shape.append(int(bindings[dim]))
        else:
            unresolved.append(str(dim))
    return shape, unresolved


def _coerce_integer(value: Any, name: str) -> int:
    """Convert a value to an integer, with numpy support."""
    try:
        from zinnia.lang.type import PoseidonHashed as PoseidonHashedType
        if isinstance(value, PoseidonHashedType):
            return _coerce_integer(value.actual_value, name)
    except ImportError:
        pass
    if isinstance(value, float):
        raise ZinniaException(f"Input datatype mismatch for `{name}`. Expected Integer, got float.")
    try:
        import numpy as np
        if isinstance(value, np.floating):
            raise ZinniaException(f"Input datatype mismatch for `{name}`. Expected Integer, got {type(value).__name__}.")
        if isinstance(value, (np.integer, np.bool_)):
            return int(value)
        if isinstance(value, np.ndarray):
            raise ZinniaException(f"Input datatype mismatch for `{name}`. Expected scalar Integer, got ndarray.")
    except ImportError:
        pass
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise ZinniaException(f"Input datatype mismatch for `{name}`. Expected Integer, got {type(value).__name__}.")


def _coerce_float(value: Any, name: str) -> float:
    """Convert a value to a float, with numpy support."""
    try:
        import numpy as np
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return float(value)
    except ImportError:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    raise ZinniaException(f"Input datatype mismatch for `{name}`. Expected Float, got {type(value).__name__}.")


def _flatten_ndarray(value: Any, expected_shape: list, name: str) -> list:
    """Flatten an array-like value into a flat list, validating shape."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            actual_shape = list(value.shape)
            if actual_shape != expected_shape:
                raise ZinniaException(
                    f"Shape mismatch for `{name}`. Expected {expected_shape}, got {actual_shape}."
                )
            return value.flatten().tolist()
    except ImportError:
        pass

    # Handle nested Python lists. Compare the full rectangular shape (not
    # just the element count) so that ragged or mis-shaped nested lists are
    # rejected even when they happen to flatten to the right number of
    # elements (e.g. [[1,2,3],[4]] for an NDArray[Integer, 2, 2]).
    if isinstance(value, (list, tuple)):
        actual_shape = _nested_shape(value, name)
        if actual_shape != list(expected_shape):
            raise ZinniaException(
                f"Shape mismatch for `{name}`. Expected {list(expected_shape)}, got {actual_shape}."
            )

    flat = []
    _flatten_nested_list(value, flat)
    import math
    expected_count = math.prod(expected_shape)
    if len(flat) != expected_count:
        raise ZinniaException(
            f"Shape mismatch for `{name}`. Expected {expected_count} elements, got {len(flat)}."
        )
    return flat


def _flatten_bounded_dynarray_with_shape(
    value: Any, max_length: int, max_rank: int, dtype, name: str, max_shape: list[int] | None = None
) -> tuple[list, list[int]]:
    """Flatten a dynamic array input, recording its actual runtime shape."""
    def check_shape_bounds(actual_shape: list[int]) -> None:
        if max_shape is None:
            return
        if len(actual_shape) != len(max_shape):
            raise ZinniaException(
                f"Shape mismatch for `{name}`. Expected DynamicNDArray rank {len(max_shape)} with per-axis bounds {max_shape}, got shape {actual_shape}."
            )
        for axis, (actual, bound) in enumerate(zip(actual_shape, max_shape)):
            if actual > bound:
                raise ZinniaException(
                    f"Shape mismatch for `{name}`. Axis {axis} expected at most {bound}, got {actual}."
                )

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            actual_shape = list(value.shape)
            if len(actual_shape) == 0 or len(actual_shape) > max_rank:
                raise ZinniaException(
                    f"Shape mismatch for `{name}`. Expected DynamicNDArray rank 1..{max_rank} bounded by {max_length}, got shape {actual_shape}."
                )
            check_shape_bounds(actual_shape)
            return _pad_dynamic_array_payload(
                value.tolist(),
                actual_shape,
                max_length,
                dtype,
                name,
                max_shape,
            ), actual_shape
    except ImportError:
        pass

    runtime_shape = _nested_shape(value, name)
    if len(runtime_shape) == 0 or len(runtime_shape) > max_rank:
        raise ZinniaException(
            f"Shape mismatch for `{name}`. Expected DynamicNDArray rank 1..{max_rank} bounded by {max_length}, got shape {runtime_shape}."
        )
    check_shape_bounds(runtime_shape)
    flat = []
    _flatten_nested_list(value, flat)
    return _pad_dynamic_array_payload(
        value,
        runtime_shape,
        max_length,
        dtype,
        name,
        max_shape,
    ), runtime_shape


def _pad_dynamic_array_payload(
    value: Any,
    runtime_shape: list[int],
    max_length: int,
    dtype,
    name: str,
    max_shape: list[int] | None,
) -> list:
    if max_shape is None:
        flat = []
        _flatten_nested_list(value, flat)
        if len(flat) > max_length:
            raise ZinniaException(
                f"Shape mismatch for `{name}`. Expected at most {max_length} elements, got {len(flat)}."
            )
        return flat + [_zero_value_for_dtype(dtype)] * (max_length - len(flat))

    padded = [_zero_value_for_dtype(dtype)] * max_length
    strides = _row_major_strides(max_shape)

    def visit(node, coords):
        if len(coords) == len(runtime_shape):
            flat_idx = sum(coord * stride for coord, stride in zip(coords, strides))
            if flat_idx >= max_length:
                raise ZinniaException(
                    f"Shape mismatch for `{name}`. Flat index {flat_idx} exceeds max_length {max_length}."
                )
            padded[flat_idx] = node
            return
        for idx, child in enumerate(node):
            visit(child, coords + [idx])

    visit(value, [])
    return padded


def _row_major_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return strides


def _zero_value_for_dtype(dtype):
    variant, _ = _get_type_variant(dtype)
    if variant == "Float":
        return 0.0
    if variant == "Complex":
        return 0j
    return 0


def _flatten_nested_list(value: Any, flat: list):
    """Recursively flatten nested lists/tuples."""
    if isinstance(value, (list, tuple)):
        for elem in value:
            _flatten_nested_list(elem, flat)
    else:
        flat.append(value)


def _nested_shape(value: Any, name: str) -> list[int]:
    """Return the rectangular nested-list shape used for runtime metadata."""
    if not isinstance(value, (list, tuple)):
        return []
    length = len(value)
    if length == 0:
        return [0]
    child_shapes = [_nested_shape(elem, name) for elem in value]
    first = child_shapes[0]
    for child in child_shapes[1:]:
        if child != first:
            raise ZinniaException(
                f"Shape mismatch for `{name}`. Nested-list array inputs must be rectangular."
            )
    return [length] + first

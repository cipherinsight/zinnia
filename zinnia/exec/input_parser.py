from typing import List, Any

from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.api.zk_parsed_input import ZKParsedInput
from zinnia.debug.exception import ZinniaException


def parse_inputs(program_inputs: List[ZKProgramInput], args: tuple) -> list:
    """Flatten user arguments into input entries for the Rust mock executor.

    Returns a list of dicts: [{"key": "0_0", "kind": "Integer", "value": 42}, ...]
    """
    if len(args) != len(program_inputs):
        raise ZinniaException(
            f"Expected {len(program_inputs)} arguments, got {len(args)}"
        )

    entries = []
    for i, (pi, arg) in enumerate(zip(program_inputs, args)):
        _flatten_value(entries, arg, pi.dt, (0, i), pi.name)
    return entries


def parse_inputs_to_parsed_input(program_inputs: List[ZKProgramInput], args: tuple) -> ZKParsedInput:
    """Flatten user arguments into a ZKParsedInput object."""
    entries = parse_inputs(program_inputs, args)
    parsed_entries = []
    for e in entries:
        indices = tuple(int(x) for x in e["key"].split("_"))
        parsed_entries.append(ZKParsedInput.Entry(indices, e["kind"], e["value"]))
    return ZKParsedInput(parsed_entries)


def _get_type_variant(dt):
    """Extract the type variant name and data from a ZinniaType serde dict.

    Scalars: "Integer" → ("Integer", {})
    Compound: {"NDArray": {"shape": ...}} → ("NDArray", {"shape": ...})
    """
    if isinstance(dt, str):
        return dt, {}
    if isinstance(dt, dict) and len(dt) == 1:
        variant = next(iter(dt))
        return variant, dt[variant]
    raise ZinniaException(f"Invalid type descriptor: {dt}")


def _flatten_value(entries: list, value: Any, dt, indices: tuple, name: str):
    """Recursively flatten a value according to its type descriptor."""
    variant, data = _get_type_variant(dt)

    if variant == "Integer":
        int_val = _coerce_integer(value, name)
        key = "_".join(str(i) for i in indices)
        entries.append({"key": key, "kind": "Integer", "value": int_val})

    elif variant == "Float":
        float_val = _coerce_float(value, name)
        key = "_".join(str(i) for i in indices)
        entries.append({"key": key, "kind": "Float", "value": float_val})

    elif variant == "NDArray":
        shape = data["shape"]
        dtype = data["dtype"]
        flat = _flatten_ndarray(value, shape, name)
        for flat_idx, elem in enumerate(flat):
            _flatten_value(entries, elem, dtype, indices + (flat_idx,), name)

    elif variant == "List":
        elements_dt = data["elements"]
        if not isinstance(value, (list, tuple)):
            raise ZinniaException(f"Expected list for `{name}`, got {type(value).__name__}")
        if len(value) != len(elements_dt):
            raise ZinniaException(f"Expected list of length {len(elements_dt)} for `{name}`, got {len(value)}")
        for j, (elem, elem_dt) in enumerate(zip(value, elements_dt)):
            _flatten_value(entries, elem, elem_dt, indices + (j,), name)

    elif variant == "Tuple":
        elements_dt = data["elements"]
        if not isinstance(value, (list, tuple)):
            raise ZinniaException(f"Expected tuple for `{name}`, got {type(value).__name__}")
        if len(value) != len(elements_dt):
            raise ZinniaException(f"Expected tuple of length {len(elements_dt)} for `{name}`, got {len(value)}")
        for j, (elem, elem_dt) in enumerate(zip(value, elements_dt)):
            _flatten_value(entries, elem, elem_dt, indices + (j,), name)

    elif variant == "PoseidonHashed":
        from zinnia.lang.type import PoseidonHashed as PoseidonHashedType
        inner_dt = data["dtype"]
        if isinstance(value, PoseidonHashedType):
            _flatten_value(entries, value.actual_value, inner_dt, indices, name)
        else:
            _flatten_value(entries, value, inner_dt, indices, name)

    elif variant == "DynamicNDArray":
        dtype = data["dtype"]
        max_length = data["max_length"]
        flat = _flatten_ndarray(value, [max_length], name)
        for flat_idx, elem in enumerate(flat):
            _flatten_value(entries, elem, dtype, indices + (flat_idx,), name)

    else:
        raise ZinniaException(f"Unsupported type `{variant}` for `{name}`")


def _coerce_integer(value: Any, name: str) -> int:
    """Convert a value to an integer, with numpy support."""
    # Handle PoseidonHashed — extract actual value
    try:
        from zinnia.lang.type import PoseidonHashed as PoseidonHashedType
        if isinstance(value, PoseidonHashedType):
            return _coerce_integer(value.actual_value, name)
    except ImportError:
        pass
    # Check for float-like values that shouldn't be integers
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
            # Validate dtype
            if np.issubdtype(value.dtype, np.floating):
                pass  # float arrays are OK for both Integer and Float params
                # Note: Integer type check happens in _coerce_integer per element
            actual_shape = list(value.shape)
            if actual_shape != expected_shape:
                raise ZinniaException(
                    f"Shape mismatch for `{name}`. Expected {expected_shape}, got {actual_shape}."
                )
            return value.flatten().tolist()
    except ImportError:
        pass

    # Handle nested Python lists
    flat = []
    _flatten_nested_list(value, flat)
    # Verify total count matches shape product
    import math
    expected_count = math.prod(expected_shape)
    if len(flat) != expected_count:
        raise ZinniaException(
            f"Shape mismatch for `{name}`. Expected {expected_count} elements, got {len(flat)}."
        )
    return flat


def _flatten_nested_list(value: Any, flat: list):
    """Recursively flatten nested lists/tuples."""
    if isinstance(value, (list, tuple)):
        for elem in value:
            _flatten_nested_list(elem, flat)
    else:
        flat.append(value)

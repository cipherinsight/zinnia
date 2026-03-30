from typing import List, Any

from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.api.zk_parsed_input import ZKParsedInput
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

    elif variant == "Boolean":
        return {"Bool": bool(value)}

    elif variant == "NDArray":
        shape = data["shape"]
        dtype = data["dtype"]
        flat = _flatten_ndarray(value, shape, name)
        elements = [_build_input_node(elem, dtype, name) for elem in flat]
        return {"Array": {"shape": shape, "elements": elements}}

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
        flat = _flatten_ndarray(value, [max_length], name)
        elements = [_build_input_node(elem, dtype, name) for elem in flat]
        return {"Array": {"shape": [max_length], "elements": elements}}

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


# ---------------------------------------------------------------------------
# Legacy interface (kept for argparse compatibility)
# ---------------------------------------------------------------------------

def parse_inputs(program_inputs: List[ZKProgramInput], args: tuple) -> list:
    """Flatten user arguments into input entries (legacy flat format).

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
    """Extract the type variant name and data from a ZinniaType serde dict."""
    if isinstance(dt, str):
        return dt, {}
    if isinstance(dt, dict) and len(dt) == 1:
        variant = next(iter(dt))
        return variant, dt[variant]
    raise ZinniaException(f"Invalid type descriptor: {dt}")


def _flatten_value(entries: list, value: Any, dt, indices: tuple, name: str):
    """Recursively flatten a value according to its type descriptor (legacy)."""
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
        from zinnia.compile._bridge import poseidon_hash as _rust_poseidon_hash
        inner_dt = data["dtype"]
        if isinstance(value, PoseidonHashedType):
            actual = value.actual_value
        else:
            actual = value
        inner_entries_start = len(entries)
        _flatten_value(entries, actual, inner_dt, indices + (0,), name)
        scalar_values = [e["value"] for e in entries[inner_entries_start:]]
        hash_hex = _rust_poseidon_hash(scalar_values)
        hash_key = "_".join(str(i) for i in indices + (1,))
        entries.append({"key": hash_key, "kind": "Str", "value": hash_hex})

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

    # Handle nested Python lists
    flat = []
    _flatten_nested_list(value, flat)
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

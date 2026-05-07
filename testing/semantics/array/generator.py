from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .manifest import ArrayMode, ExpectedStatus, OperatorCase, ShapeCase, load_manifest


MANIFEST_PATH = Path(__file__).with_name("operators.yaml")


@dataclass(frozen=True)
class GeneratedCase:
    operator: OperatorCase
    shape: ShapeCase
    spelling: str
    mode: ArrayMode
    dtype: str
    expected: ExpectedStatus

    @property
    def effective_expected(self) -> ExpectedStatus:
        return self.shape.expected_for_mode(self.mode) or self.expected

    @property
    def id(self) -> str:
        return "::".join(
            [self.operator.name, self.spelling, self.mode.value, self.dtype, self.shape.id]
        )

    @property
    def rank(self) -> int | None:
        return self.shape.rank

    @property
    def effective_dtype(self) -> str:
        return self.shape.data.get("dtype_override", self.dtype)

    @property
    def family(self) -> str:
        return self.shape.data.get("family", self.operator.name)


def generate_cases(path: Path = MANIFEST_PATH) -> list[GeneratedCase]:
    out: list[GeneratedCase] = []
    for op in load_manifest(path):
        for spelling, mode, dtype, shape in product(op.spellings, op.modes, op.dtypes, op.shapes):
            if mode == ArrayMode.MIXED and op.kind not in {"binary"}:
                continue
            if shape.spellings is not None and spelling not in shape.spellings:
                continue
            if shape.modes is not None and mode not in shape.modes:
                continue
            out.append(
                GeneratedCase(
                    operator=op,
                    shape=shape,
                    spelling=spelling,
                    mode=mode,
                    dtype=dtype,
                    expected=op.expected,
                )
            )
    return out


def ndarray_for_shape(shape: list[int], dtype: str, salt: int = 0) -> np.ndarray:
    size = int(np.prod(shape)) if shape else 1
    if dtype == "bool":
        data = [((i + salt) % 2) == 0 for i in range(size)]
    elif dtype == "float":
        data = [float((i + 1 + salt) * 0.5) for i in range(size)]
    else:
        data = [((i + 1) * 7 + salt * 11) % 29 - 9 for i in range(size)]
    return np.asarray(data).reshape(tuple(shape))


def scalar_for_dtype(dtype: str, salt: int = 0) -> Any:
    if dtype == "bool":
        return salt % 2 == 0
    if dtype == "float":
        return float(salt + 2)
    return salt + 3


def oracle(case: GeneratedCase) -> Any:
    dispatch: dict[str, Callable[[GeneratedCase], Any]] = {
        "binary": _oracle_binary,
        "constructor": _oracle_constructor,
        "indexing": _oracle_indexing,
        "join": _oracle_join,
        "linspace": _oracle_linspace,
        "scenario": _oracle_scenario,
        "shape": _oracle_shape,
        "reduction": _oracle_reduction,
        "mutation": _oracle_mutation,
        "unary": _oracle_unary,
    }
    return dispatch[case.operator.kind](case)


def _shape_or_scalar(value: Any, dtype: str, salt: int) -> Any:
    if value == "scalar":
        return scalar_for_dtype(dtype, salt)
    return ndarray_for_shape(value, dtype, salt)


def _value_from_spec(
    shape: Any | None,
    dtype: str,
    salt: int,
    value: Any | None = None,
    expr: str | None = None,
) -> Any:
    if expr is not None:
        return _eval_oracle_expr(expr)
    if value is not None:
        return np.asarray(value, dtype=_numpy_dtype(dtype))
    if shape is None:
        raise ValueError("shape or value or expr must be provided")
    return _shape_or_scalar(shape, dtype, salt)


def _oracle_binary(case: GeneratedCase) -> Any:
    lhs, rhs = _binary_values(case)
    spelling = case.spelling
    if spelling in {"+", "np.add"}:
        return lhs + rhs
    if spelling in {"-", "np.subtract"}:
        return lhs - rhs
    if spelling in {"*", "np.multiply"}:
        return lhs * rhs
    if spelling == "/":
        return lhs / rhs
    if spelling in {"==", "np.equal"}:
        return lhs == rhs
    if spelling == "!=":
        return lhs != rhs
    if spelling in {"<", "np.less"}:
        return lhs < rhs
    if spelling == "<=":
        return lhs <= rhs
    if spelling == ">":
        return lhs > rhs
    if spelling == ">=":
        return lhs >= rhs
    raise NotImplementedError(spelling)


def _oracle_indexing(case: GeneratedCase) -> Any:
    source = _index_source("a", case.shape.data["index"])
    env = {
        "np": _ORACLE_NP,
        "a": _value_from_spec(
            case.shape.data.get("input"),
            case.effective_dtype,
            0,
            case.shape.data.get("input_value"),
            case.shape.data.get("input_expr"),
        ),
    }
    return eval(source, env)


def _oracle_constructor(case: GeneratedCase) -> Any:
    data = case.shape.data
    name = case.family
    if data.get("oracle_expr"):
        return _eval_oracle_expr(data["oracle_expr"])
    if data.get("source_expr"):
        return _eval_oracle_expr(data["source_expr"])
    if name == "asarray":
        dtype = _numpy_dtype(data.get("dtype"))
        return np.asarray(data["value"], dtype=dtype)
    if name == "arange":
        args = data["args"]
        dtype = _numpy_dtype(data.get("dtype"))
        return np.arange(*args, dtype=dtype)
    if name == "zeros":
        return np.zeros(tuple(data["input"]), dtype=_numpy_dtype(data.get("dtype", "int")))
    if name == "identity":
        return np.identity(data["n"], dtype=_numpy_dtype(data.get("dtype", "int")))
    raise NotImplementedError(name)


def _oracle_join(case: GeneratedCase) -> Any:
    arrays = [
        ndarray_for_shape(shape, dtype, salt)
        for salt, (shape, dtype) in enumerate(zip(case.shape.data["inputs"], case.shape.data["input_dtypes"]))
    ]
    axis = case.shape.data.get("axis")
    if case.family == "concatenate":
        return np.concatenate(arrays, axis=axis)
    if case.family == "stack":
        return np.stack(arrays, axis=axis if axis is not None else 0)
    raise NotImplementedError(case.family)


def _oracle_linspace(case: GeneratedCase) -> Any:
    data = case.shape.data
    start = data["start"]
    stop = data["stop"]
    num = data["num"]
    kwargs = {
        "dtype": _numpy_dtype(data.get("dtype")),
        "endpoint": data.get("endpoint", True),
    }
    if "axis" in data:
        kwargs["axis"] = data["axis"]
    return np.linspace(start, stop, num, **kwargs)


def _oracle_shape(case: GeneratedCase) -> Any:
    a = _value_from_spec(
        case.shape.data.get("input"),
        case.effective_dtype,
        0,
        case.shape.data.get("input_value"),
        case.shape.data.get("input_expr"),
    )
    spelling = case.spelling
    if case.family == "reshape":
        newshape = tuple(case.shape.data["newshape"])
        return a.reshape(newshape) if spelling == "method" else np.reshape(a, newshape)
    if case.family == "transpose":
        axes = case.shape.data.get("axes")
        if spelling == ".T":
            return a.T
        if spelling == "method":
            return a.transpose(tuple(axes)) if axes is not None else a.transpose()
        return np.transpose(a, axes=tuple(axes)) if axes is not None else np.transpose(a)
    if case.family == "expand_dims":
        return np.expand_dims(a, case.shape.data["axis"])
    if case.family == "squeeze":
        axis = case.shape.data.get("axis")
        return np.squeeze(a, axis=axis) if axis is not None else np.squeeze(a)
    raise NotImplementedError(case.family)


def _oracle_scenario(case: GeneratedCase) -> Any:
    data = case.shape.data
    env = _oracle_env(case)
    env.update({"np": _ORACLE_NP, "int": int, "float": float, "bool": bool})
    for line in data.get("setup", []):
        exec(line, env)
    if data.get("oracle_expr"):
        return _eval_oracle_expr(data["oracle_expr"], env)
    if data.get("source_expr"):
        return _eval_oracle_expr(data["source_expr"], env)
    raise NotImplementedError("scenario cases require oracle_expr or source_expr")


def _oracle_reduction(case: GeneratedCase) -> Any:
    a = _value_from_spec(
        case.shape.data.get("input"),
        case.effective_dtype,
        0,
        case.shape.data.get("input_value"),
        case.shape.data.get("input_expr"),
    )
    axis = case.shape.data.get("axis")
    spelling = case.spelling
    if spelling == "sum":
        return a.sum(axis=axis)
    if spelling == "max":
        return a.max(axis=axis)
    if spelling == "min":
        return a.min(axis=axis)
    if spelling == "any":
        return a.any(axis=axis)
    if spelling == "all":
        return a.all(axis=axis)
    if spelling == "argmax":
        return a.argmax(axis=axis)
    if spelling == "argmin":
        return a.argmin(axis=axis)
    raise NotImplementedError(spelling)


def _oracle_mutation(case: GeneratedCase) -> Any:
    a = _value_from_spec(
        case.shape.data.get("input"),
        case.effective_dtype,
        0,
        case.shape.data.get("input_value"),
        case.shape.data.get("input_expr"),
    ).copy()
    source = _setitem_source("a", case.shape.data["target"], _literal(case.shape.data["value"]))
    env = {"np": _ORACLE_NP, "a": a}
    exec(source, env)
    return env["a"]


def _oracle_unary(case: GeneratedCase) -> Any:
    data = case.shape.data
    if data.get("oracle_expr"):
        a = _eval_oracle_expr(data["oracle_expr"])
    elif data.get("input_expr"):
        a = _eval_oracle_expr(data["input_expr"])
    else:
        a = np.asarray(data["value"], dtype=_numpy_dtype(data.get("input_dtype")))
    if case.family == "astype":
        return a.astype(_numpy_dtype(data["to"]))
    if case.family == "tolist":
        return a.tolist()
    raise NotImplementedError(case.family)


def render_source(case: GeneratedCase, expected: Any) -> str:
    body = _render_operation(case)
    assertions = _render_case_assertions(case, expected)
    assertions.extend(_render_extra_assertions(case))
    signature = _render_signature(case)
    return "\n".join(
        [
            f"def semantic_case({signature}):" if signature else "def semantic_case():",
            *[f"    {line}" if line else "" for line in body],
            *[f"    {line}" if line else "" for line in assertions],
        ]
    )


def _render_case_assertions(case: GeneratedCase, expected: Any) -> list[str]:
    mode = case.shape.data.get("assertion_mode", "elementwise")
    if mode == "tolist":
        arr = np.asarray(expected)
        return [
            f"assert out.shape == {_literal(tuple(arr.shape))}",
            f"assert out.tolist() == {_literal(arr.tolist())}",
        ]
    return _render_assertions("out", expected)


def _render_operation(case: GeneratedCase) -> list[str]:
    kind = case.operator.kind
    if kind == "binary":
        lhs = _render_binary_value(case, "lhs", 0)
        rhs = _render_binary_value(case, "rhs", 1)
        op = _render_binary_expr("lhs", "rhs", case.spelling)
        return [lhs, rhs, f"out = {op}"]
    if kind == "constructor":
        return [f"out = {_render_constructor_expr(case)}"]
    if kind == "indexing":
        lines = [_render_case_value(case, "a", "input", 0)]
        lines.append(f"out = {_index_source('a', case.shape.data['index'])}")
        return lines
    if kind == "join":
        lines = []
        for idx, (shape, dtype) in enumerate(zip(case.shape.data["inputs"], case.shape.data["input_dtypes"])):
            lines.append(_render_value(f"a{idx}", shape, dtype, case.mode, idx))
        names = ", ".join(f"a{idx}" for idx in range(len(case.shape.data["inputs"])))
        axis = case.shape.data.get("axis")
        axis_arg = "" if axis is None else f", axis={axis}"
        lines.append(f"out = np.{case.family}([{names}]{axis_arg})")
        return lines
    if kind == "linspace":
        return [f"out = {_render_linspace_expr(case)}"]
    if kind == "scenario":
        lines = []
        if case.shape.data.get("setup"):
            lines.extend(case.shape.data["setup"])
        lines.append(f"out = {case.shape.data['source_expr']}")
        return lines
    if kind == "shape":
        lines = [_render_case_value(case, "a", "input", 0)]
        lines.append(f"out = {_render_shape_expr(case)}")
        return lines
    if kind == "reduction":
        lines = [_render_case_value(case, "a", "input", 0)]
        axis = case.shape.data.get("axis")
        axis_arg = "" if axis is None else f"axis={axis}"
        lines.append(f"out = a.{case.spelling}({axis_arg})")
        return lines
    if kind == "mutation":
        lines = [_render_case_value(case, "out", "input", 0)]
        lines.append(_setitem_source("out", case.shape.data["target"], _literal(case.shape.data["value"])))
        return lines
    if kind == "unary":
        data = case.shape.data
        if case.family == "tolist":
            if data.get("input_expr"):
                return [f"a = {data['input_expr']}", "out = a.tolist()"]
            return [
                f"a = np.asarray({_literal(data['value'])}, dtype={data.get('input_dtype', 'int')})",
                "out = a.tolist()",
            ]
        if data.get("input_expr"):
            input_line = f"a = {data['input_expr']}"
        else:
            input_line = f"a = np.asarray({_literal(data['value'])}, dtype={data.get('input_dtype', 'int')})"
        return [
            input_line,
            f"out = a.astype({data['to']})",
        ]
    raise NotImplementedError(kind)


def _render_constructor_expr(case: GeneratedCase) -> str:
    data = case.shape.data
    if data.get("source_expr"):
        return data["source_expr"]
    if case.family == "asarray":
        dtype = f", dtype={data['dtype']}" if data.get("dtype") else ""
        return f"np.asarray({_literal(data['value'])}{dtype})"
    if case.family == "arange":
        args = ", ".join(_literal(v) for v in data["args"])
        if data.get("dtype"):
            args = f"{args}, {data['dtype']}"
        return f"np.arange({args})"
    if case.family == "zeros":
        return f"np.zeros({_literal(tuple(data['input']))}, {data.get('dtype', 'int')})"
    if case.family == "identity":
        return f"np.identity({data['n']}, {data.get('dtype', 'int')})"
    raise NotImplementedError(case.family)


def _render_linspace_expr(case: GeneratedCase) -> str:
    data = case.shape.data
    args = [
        _literal(data["start"]),
        _literal(data["stop"]),
        _literal(data["num"]),
    ]
    kwargs = []
    if "axis" in data:
        kwargs.append(f"axis={data['axis']}")
    if "dtype" in data:
        kwargs.append(f"dtype={data['dtype']}")
    if "endpoint" in data:
        kwargs.append(f"endpoint={_literal(data['endpoint'])}")
    joined = ", ".join(args + kwargs)
    return f"np.linspace({joined})"


def _render_value(name: str, shape: Any, dtype: str, mode: ArrayMode, salt: int) -> str:
    if shape == "scalar":
        return f"{name} = {_literal(scalar_for_dtype(dtype, salt))}"
    arr = ndarray_for_shape(shape, dtype, salt)
    expr = f"np.asarray({_literal(arr.tolist())})"
    if dtype == "float":
        expr = f"np.asarray({_literal(arr.tolist())}, dtype=float)"
    elif dtype == "bool":
        expr = f"np.asarray({_literal(arr.tolist())}, dtype=bool)"
    elif dtype == "int":
        expr = f"np.asarray({_literal(arr.tolist())}, dtype=int)"
    if mode == ArrayMode.DYNAMIC or (mode == ArrayMode.MIXED and name in {"lhs", "a", "out"}):
        expr = f"np.promote_to_dynamic({expr})"
    return f"{name} = {expr}"


def _render_literal_value(name: str, value: Any, dtype: str, mode: ArrayMode) -> str:
    expr = f"np.asarray({_literal(value)}, dtype={dtype})"
    if mode == ArrayMode.DYNAMIC or (mode == ArrayMode.MIXED and name in {"lhs", "a", "out"}):
        expr = f"np.promote_to_dynamic({expr})"
    return f"{name} = {expr}"


def _render_case_value(case: GeneratedCase, target_name: str, data_prefix: str, salt: int) -> str:
    expr_key = f"{data_prefix}_expr"
    value_key = f"{data_prefix}_value"
    shape_key = data_prefix
    if case.shape.data.get(expr_key) is not None:
        expr = case.shape.data[expr_key]
        if case.mode == ArrayMode.DYNAMIC or (case.mode == ArrayMode.MIXED and target_name in {"lhs", "a", "out"}):
            expr = f"np.promote_to_dynamic({expr})"
        return f"{target_name} = {expr}"
    if case.shape.data.get(value_key) is not None:
        return _render_literal_value(target_name, case.shape.data[value_key], case.effective_dtype, case.mode)
    return _render_value(target_name, case.shape.data[shape_key], case.effective_dtype, case.mode, salt)


def _binary_values(case: GeneratedCase) -> tuple[Any, Any]:
    if case.spelling == "/":
        return np.asarray([10, 20, 30]), np.asarray([2, 5, 10])
    if "lhs_value" in case.shape.data or "rhs_value" in case.shape.data or "rhs_literal" in case.shape.data:
        lhs = np.asarray(
            case.shape.data["lhs_value"] if "lhs_value" in case.shape.data else case.shape.data["lhs"],
            dtype=_numpy_dtype(case.shape.data.get("lhs_dtype_override", case.effective_dtype)),
        )
        if "rhs_literal" in case.shape.data:
            rhs = case.shape.data["rhs_literal"]
        else:
            rhs = np.asarray(
                case.shape.data["rhs_value"] if "rhs_value" in case.shape.data else case.shape.data["rhs"],
                dtype=_numpy_dtype(case.shape.data.get("rhs_dtype_override", case.effective_dtype)),
            )
        return lhs, rhs
    return (
        _shape_or_scalar(case.shape.data["lhs"], case.dtype, 0),
        _shape_or_scalar(case.shape.data["rhs"], case.dtype, 1),
    )


def _render_binary_value(case: GeneratedCase, name: str, salt: int) -> str:
    if case.spelling != "/":
        value_key = f"{name}_value"
        literal_key = f"{name}_literal"
        dtype_key = f"{name}_dtype_override"
        if value_key in case.shape.data:
            return _render_literal_value(
                name,
                case.shape.data[value_key],
                case.shape.data.get(dtype_key, case.effective_dtype),
                case.mode,
            )
        if literal_key in case.shape.data:
            return f"{name} = {_literal(case.shape.data[literal_key])}"
        return _render_case_value(case, name, name, salt)
    arr = [10, 20, 30] if name == "lhs" else [2, 5, 10]
    expr = f"np.asarray({arr}, dtype=int)"
    if case.mode == ArrayMode.DYNAMIC or (case.mode == ArrayMode.MIXED and name == "lhs"):
        expr = f"np.promote_to_dynamic({expr})"
    return f"{name} = {expr}"


def _render_binary_expr(lhs: str, rhs: str, spelling: str) -> str:
    if spelling.startswith("np."):
        return f"{spelling}({lhs}, {rhs})"
    return f"{lhs} {spelling} {rhs}"


def _render_shape_expr(case: GeneratedCase) -> str:
    if case.family == "reshape":
        shape = tuple(case.shape.data["newshape"])
        if case.spelling == "method":
            return f"a.reshape({_literal(shape)})"
        return f"np.reshape(a, {_literal(shape)})"
    if case.family == "transpose":
        axes = case.shape.data.get("axes")
        if case.spelling == ".T":
            return "a.T"
        if case.spelling == "method":
            return "a.transpose()" if axes is None else f"a.transpose(axes={_literal(tuple(axes))})"
        return "np.transpose(a)" if axes is None else f"np.transpose(a, axes={_literal(tuple(axes))})"
    if case.family == "expand_dims":
        return f"np.expand_dims(a, {case.shape.data['axis']})"
    if case.family == "squeeze":
        axis = case.shape.data.get("axis")
        return "np.squeeze(a)" if axis is None else f"np.squeeze(a, axis={axis})"
    raise NotImplementedError(case.family)


def _render_assertions(var_name: str, expected: Any) -> list[str]:
    if isinstance(expected, np.ndarray):
        return _render_array_assertions(var_name, expected)
    if isinstance(expected, list):
        lines = [f"assert len({var_name}) == {len(expected)}"]
        for idx, item in enumerate(expected):
            lines.extend(_render_assertions(f"{var_name}[{idx}]", item))
        return lines
    if isinstance(expected, tuple):
        lines = [f"assert len({var_name}) == {len(expected)}"]
        for idx, item in enumerate(expected):
            lines.extend(_render_assertions(f"{var_name}[{idx}]", item))
        return lines
    return [f"assert {var_name} == {_literal(expected)}"]


def _render_array_assertions(var_name: str, expected: np.ndarray) -> list[str]:
    arr = np.asarray(expected)
    if arr.shape == ():
        return [f"assert {var_name} == {_literal(arr.item())}"]
    if var_name.startswith("out["):
        lines = [f"assert len({var_name}) == {len(arr)}"]
        for idx in np.ndindex(arr.shape):
            lines.append(f"assert {var_name}{_subscript(idx)} == {_literal(arr[idx].item())}")
        return lines
    lines = [
        f"assert len({var_name}.shape) == {arr.ndim}",
        f"assert {var_name}.shape == {_literal(tuple(arr.shape))}",
    ]
    for idx in np.ndindex(arr.shape):
        lines.append(f"assert {var_name}{_subscript(idx)} == {_literal(arr[idx].item())}")
    return lines


def _render_extra_assertions(case: GeneratedCase) -> list[str]:
    dtype = case.shape.data.get("assert_dtype")
    if dtype is None:
        return []
    return [f"assert out.dtype == {dtype}"]


def _render_signature(case: GeneratedCase) -> str:
    return ", ".join(case.shape.data.get("circuit_args", []))


def _index_source(name: str, index: str) -> str:
    return f"{name}{index}"


def _setitem_source(name: str, target: str, value: str) -> str:
    return f"{name}{target} = {value}"


def _subscript(idx: tuple[int, ...]) -> str:
    if len(idx) == 1:
        return f"[{idx[0]}]"
    return "[" + ", ".join(str(i) for i in idx) + "]"


def _literal(value: Any) -> str:
    if isinstance(value, np.generic):
        return _literal(value.item())
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    return repr(value)


def _numpy_dtype(dtype: str | None) -> Any:
    if dtype is None:
        return None
    return {"int": int, "float": float, "bool": bool}[dtype]


def _eval_oracle_expr(expr: str, env: dict[str, Any] | None = None) -> Any:
    base_env = {"np": np, "int": int, "float": float, "bool": bool}
    if env:
        base_env.update(env)
    return eval(expr, base_env)


def _oracle_env(case: GeneratedCase) -> dict[str, Any]:
    env: dict[str, Any] = {}
    arg_names = case.shape.data.get("arg_names", [])
    prove_args = case.shape.data.get("prove_args", [])
    for name, value in zip(arg_names, prove_args):
        env[name] = value
    return env


class _OracleNP:
    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    @staticmethod
    def promote_to_dynamic(value: Any) -> Any:
        return value


_ORACLE_NP = _OracleNP()

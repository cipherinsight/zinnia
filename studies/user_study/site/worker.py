"""Execute participant code only through compiled Zinnia IR, inside the sandbox."""
import ast
import json
import math
from pathlib import Path
import traceback

import numpy as np
from zinnia import ZKCircuit


def entry_function(tree):
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel"]
    if len(entries) != 1:
        raise ValueError("Provide a function named kernel using the starter template's arguments.")
    return entries[0]


def expose_result(tree):
    """Return rank, dimensions, and cells as ordered compiled public outputs.

    A constrained zero input keeps constant cells from being elided from the
    public output layout. It does not change any returned value. No participant
    expression is evaluated by Python to determine values or output shape.
    """
    fn = entry_function(tree)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
    def fresh(base):
        while base in names:
            base += "_"
        names.add(base)
        return base
    zero, actual, integer = (fresh(n) for n in ("study_output_zero", "study_output_array", "study_output_integer"))
    fn.args.args.append(ast.arg(arg=zero, annotation=ast.Name(id="Integer", ctx=ast.Load())))

    class Returns(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def visit_Return(self, node):
            body = ast.parse(f"{actual} = np.asarray(None)").body
            body[0].value.args[0] = node.value or ast.Constant(None)
            body += ast.parse(f"""
assert 1 <= {actual}.ndim <= 5
assert {actual}.size <= 4096
{integer} = {actual}.astype(int)
assert np.all({actual} == {integer})
return ({actual}.ndim + {zero}, np.array({actual}.shape) + {zero}, {integer} + {zero})
""").body
            return body

    body = ast.parse(f"assert {zero} == 0").body
    for node in fn.body:
        transformed = Returns().visit(node)
        body.extend(transformed if isinstance(transformed, list) else [transformed])
    fn.body, fn.returns = body, None
    return ast.unparse(ast.fix_missing_locations(tree))


def decode_result(program, proof):
    if proof.proof_bytes_hex != "mock_satisfied":
        raise ValueError("Compiled execution failed an assertion; return an integer-valued array with 1–5 dimensions.")
    if program.get_elided_publics():
        raise RuntimeError("Compiled output layout unexpectedly elided a value")
    values = dict(item.rsplit("=", 1) for item in proof.public_values)
    ordered = []
    for stmt in program.get_ir_stmts():
        kind = stmt["ir_instance"]["__class__"]
        if kind == "ExposePublicIIR":
            ordered.append(int(values.pop(f'public_i_{stmt["stmt_id"]}')))
        elif kind == "ExposePublicFIR":
            raise RuntimeError("Compiled output layout contains a noninteger value")
    if values or not ordered:
        raise RuntimeError("Invalid compiled output layout")
    rank = ordered[0]
    shape, cells = ordered[1:1+rank], ordered[1+rank:]
    if not 1 <= rank <= 5 or len(shape) != rank or any(d < 0 for d in shape) or math.prod(shape) != len(cells) or len(cells) > 4096:
        raise ValueError("Invalid array shape from compiled execution")
    return {"shape": shape, "values": cells}


def main():
    request = json.loads(Path("request.json").read_text())
    raw = Path("submission.py").read_text()
    results, stage = [], "compile"
    try:
        fn = entry_function(ast.parse(raw, filename="submission.py"))
        if [a.arg for a in fn.args.args] != request["arguments"]:
            raise ValueError("Keep the function name and arguments from the starter template.")
        compiled_by_shape = {}
        for case in request["cases"]:
            typed = ast.parse(raw)
            args = [np.asarray(case[name], dtype=np.int64) for name in request["arguments"]]
            for arg, value in zip(entry_function(typed).args.args, args):
                dims = " ".join(map(str, value.shape))
                arg.annotation = ast.parse(f'Integer[Array, "{dims}"]', mode="eval").body
            stage = "compile"
            # Preserve the original compilation gate before adding output metadata.
            shape_key = tuple(value.shape for value in args)
            if shape_key not in compiled_by_shape:
                ZKCircuit.from_source("kernel", ast.unparse(typed)).compile()
                compiled_by_shape[shape_key] = ZKCircuit.from_source("kernel", expose_result(typed)).compile()
            program = compiled_by_shape[shape_key]
            stage = "execute"
            proof = program.prove(*args, 0, backend="mock")
            results.append(decode_result(program, proof))
        payload = {"status": "executed", "outputs": results, "stdout": ""}
    except BaseException as exc:
        payload = {"status": "compile_error" if stage == "compile" else "runtime_error", "message": str(exc),
                   "stderr": traceback.format_exc()[-16000:], "stdout": ""}
    Path("result.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()

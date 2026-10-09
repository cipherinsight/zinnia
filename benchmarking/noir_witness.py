"""Transport constraint-free Noir witnesses to the pinned Barretenberg backend.

Noir retains unused ABI inputs in its witness map. bb derives its maximum
witness index from constraints/publics, ignoring private_parameters; with neither
it expects one unused slot, not the ABI's input count. This adapter is deliberately
restricted to a single function with no opcodes and no public values. It changes
no circuit, key, statement, or public input. All other artifacts pass untouched.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


def constraint_free(bytecode: Path) -> bool:
    import msgpack

    raw = gzip.decompress(base64.b64decode(json.loads(bytecode.read_text())["bytecode"]))
    if raw[0] != 3:  # pinned Noir's compact MessagePack format
        return False
    program = msgpack.unpackb(raw[1:], strict_map_key=False)
    if not isinstance(program, list) or len(program) != 2:
        return False
    functions, unconstrained = program
    if len(functions) != 1 or unconstrained:
        return False
    circuit = functions[0]
    return (len(circuit) == 7 and circuit[2] == [] and circuit[4] == []
            and circuit[5] == [])


def for_barretenberg(bytecode: Path, witness: Path) -> Path:
    """Keep executor witnesses for audit; write a separate bb transport if needed."""
    if not constraint_free(bytecode):
        return witness
    # Format 3, WitnessStack([StackItem(index=0, witness={0: Field(0)})]).
    raw = bytes.fromhex("03919192008100c420") + bytes(32)
    target = witness.with_name(witness.stem + "_bb.gz")
    target.write_bytes(gzip.compress(raw, mtime=0))
    return target

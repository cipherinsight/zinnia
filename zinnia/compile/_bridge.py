"""
Bridge module for calling into the Rust zinnia_core library.

This module provides a thin adapter layer between the Python compiler
and the Rust implementation.
"""

from zinnia._zinnia_core import (
    CompiledIR,
    compile_circuit,
    prove_circuit,
    verify_proof_artifact,
    export_ir_json,
    import_ir_json,
)

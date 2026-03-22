"""
Bridge module for calling into the Rust zinnia_core library.

This module provides a thin adapter layer between the Python compiler
and the Rust implementation.
"""

from zinnia._zinnia_core import (
    hello,
    core_version,
    generate_ir,
    compile_circuit,
    mock_execute,
    run_optimization_pass,
    round_trip_ir_stmts,
    round_trip_dt_descriptor,
)


def check_rust_backend() -> bool:
    """Verify the Rust backend is available and working."""
    try:
        result = hello()
        return "zinnia_core" in result
    except Exception:
        return False

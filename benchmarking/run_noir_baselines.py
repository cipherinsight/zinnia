"""Run the handwritten Noir baselines against Barretenberg UltraHonk.

Per case:
  1. Stage a temp Noir package: Nargo.toml + src/main.nr + Prover.toml.
  2. `nargo execute` to produce the ACIR bytecode (target/case*.json) and
     the witness (target/case*.gz).
  3. `bb write_vk` once (so subsequent proves can reuse it).
  4. `bb prove --scheme ultra_honk` -- timed -- repeated `repeats` times.
     First call is the cold/warm-up prove; subsequent calls are warm.
     Apples-to-apples comparison point for Zinnia's `prove_seconds`.
  5. `bb verify` once.

Usage:
    python3 run_noir_baselines.py [--repeats N] [--cases 0291,0334,...]

If `nargo` or `bb` is missing on PATH (or in the usual ~/.nargo/bin and
~/.bb locations) the script prints installation instructions and exits
non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

THIS_DIR = Path(__file__).parent
NOIR_FPC = THIS_DIR / "noir_fpc"
DEFAULT_NARGO = Path.home() / ".nargo/bin/nargo"
DEFAULT_BB = Path.home() / ".bb/bb"
Q = 4_294_967_296  # 2^32; mirror noir_fpc::Q.

ABANDONED_BASELINE_CASES: dict[str, str] = {
    "0298": "np.unique has runtime-dependent output width; Noir source was a skipped fixture specialization",
    "0383": "runtime N changes output length; active Noir source omitted Zinnia's length/output contract",
    "0388": "runtime patch_size changes output shape; Noir source was a skipped fixture specialization",
    "0389": "runtime h/w change output shape; Noir source was a skipped fixture specialization",
    "0390": "runtime patch_size changes output shape; Noir source was a skipped fixture specialization",
    "0467": "runtime window size changes nested output shape; active Noir source returned only fixture-flat payload",
    "0468": "runtime window size changes nested output shape; active Noir source returned only fixture-flat payload",
}

INSTALL_INSTRUCTIONS = """
Noir + Barretenberg are required. Install with:

  curl -L https://raw.githubusercontent.com/noir-lang/noirup/refs/heads/main/install \
    | bash
  source ~/.bashrc  # or restart shell
  noirup --version 1.0.0-beta.20

  mkdir -p ~/.bb
  curl -fsSL https://raw.githubusercontent.com/AztecProtocol/aztec-packages/next/barretenberg/bbup/bbup \
    -o ~/.bb/bbup
  chmod 755 ~/.bb/bbup
  ~/.bb/bbup --version 5.0.0-nightly.20260324

Then re-run this script. (Versions can be bumped; the bb<->noir mapping
lives at https://raw.githubusercontent.com/AztecProtocol/aztec-packages/next/barretenberg/bbup/bb-versions.json.)
"""


# ---------------------------------------------------------------------------
# Input builders -- Q32 quantization + Prover.toml emission.
# ---------------------------------------------------------------------------

# Reuse the halo2 INPUT_BUILDERS for fixture compatibility.
sys.path.insert(0, str(THIS_DIR))
from run_halo2_baselines import INPUT_BUILDERS  # noqa: E402


def q32(v: float) -> int:
    return int(round(v * Q))


def _flatten(xs):
    if isinstance(xs, list):
        out = []
        for x in xs:
            out.extend(_flatten(x))
        return out
    return [xs]


def _to_int(s) -> int:
    if isinstance(s, str):
        return int(s)
    if isinstance(s, bool):
        return int(s)
    if isinstance(s, int):
        return s
    raise TypeError(f"cannot coerce {s!r} to int")


# Per-case Prover.toml writers. The function receives the dict from
# INPUT_BUILDERS and the path to write Prover.toml. Each writer knows
# whether the kernel takes floats (apply q32) or integers (pass through).

def _toml_array(values: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def _write_floats(path: Path, *named: tuple[str, list[float]]):
    with open(path, "w") as f:
        for name, vs in named:
            f.write(f"{name} = {_toml_array([q32(v) for v in vs])}\n")


def _ints(d: dict, key: str) -> list[int]:
    return [_to_int(v) for v in d[key]]


def _floats(d: dict, key: str) -> list[float]:
    return [float(v) for v in d[key]]


def writer_0291(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")


def writer_0309(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


def writer_0323(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"degree = {_to_int(d['degree'])}\n")


def writer_pow_8x6(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"power = {_to_int(d['power'])}\n")


def writer_three_50(d: dict, path: Path):
    _write_floats(
        path,
        ("a", _floats(d, "a")),
        ("b", _floats(d, "b")),
        ("c", _floats(d, "c")),
    )


def writer_0336(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


def writer_0373(d: dict, path: Path):
    _write_floats(path, ("grades", _floats(d, "grades")))


def writer_0387(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


def writer_bin_8x6(d: dict, path: Path):
    a = _floats(d, "data")
    with open(path, "w") as f:
        f.write(f"data = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"bin_size = {_to_int(d['bin_size'])}\n")


def writer_0428(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


def writer_mask_10x5(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


def writer_norm_5x4(d: dict, path: Path):
    # Integer inputs in halo2; for Noir we pass plain decimal Field values
    # (the circuit promotes to Q32 internally by multiplying by Q).
    with open(path, "w") as f:
        f.write(f"x = {_toml_array(_ints(d, 'x'))}\n")


def writer_complex_mean(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


# ---------------------------------------------------------------------------
# Auto-added writers for the full 106-case roll-out.
# Each writer maps the halo2 INPUT_BUILDERS dict to a Prover.toml matching
# the per-case main.nr signature. Float arrays go through q32; integer
# arrays/scalars pass through as decimal Field strings.
# ---------------------------------------------------------------------------

def _emit(path: Path, lines: list[str]):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def writer_a_floats(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")))


def writer_a_ints(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")


def writer_im_ints(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"im = {_toml_array(_ints(d, 'im'))}\n")


def writer_im_floats(d: dict, path: Path):
    _write_floats(path, ("im", _floats(d, "im")))


def writer_x_ints(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"x = {_toml_array(_ints(d, 'x'))}\n")


def writer_x_floats(d: dict, path: Path):
    _write_floats(path, ("x", _floats(d, "x")))


def writer_reshape_ncol(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"ncol = {_to_int(d['ncol'])}\n")


def writer_reshape_nrow(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"nrow = {_to_int(d['nrow'])}\n")


def writer_0324(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"degree = {q32(float(d['degree']))}\n")


def writer_0325(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"number = {q32(float(d['number']))}\n")


def writer_0326(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"value = {q32(float(d['value']))}\n")


def writer_0345(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"col = {_to_int(d['col'])}\n")
        f.write(f"multiply_number = {_to_int(d['multiply_number'])}\n")


def writer_0346(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"row = {_to_int(d['row'])}\n")
        f.write(f"multiply_number = {_to_int(d['multiply_number'])}\n")


def writer_0347(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"row = {_to_int(d['row'])}\n")
        f.write(f"divide_number = {_to_int(d['divide_number'])}\n")


def writer_0344(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"mystr = {_to_int(d['mystr'])}\n")


def writer_0367(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"pairs = {_toml_array(_ints(d, 'pairs'))}\n")
        f.write(f"array_of_arrays = {_toml_array(_ints(d, 'array_of_arrays'))}\n")


def writer_perm_8x6(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"permutation = {_toml_array(_ints(d, 'permutation'))}\n")


def writer_perm_3d(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"permutation = {_toml_array(_ints(d, 'permutation'))}\n")


def writer_a_floats_b_ints(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")


def writer_ab_ints(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")


def writer_insert_scalar(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"pos = {_to_int(d['pos'])}\n")
        f.write(f"element = {_to_int(d['element'])}\n")


def writer_0364(d: dict, path: Path):
    a = _floats(d, "a")
    elem = _floats(d, "element")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"pos = {_to_int(d['pos'])}\n")
        f.write(f"element = {_toml_array([q32(v) for v in elem])}\n")


def writer_0353(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")


def writer_sort3d_a_floats_b_ints(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")


def writer_argsort_topn(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"n = {_to_int(d['n'])}\n")


def writer_0389(d: dict, path: Path):
    a = _floats(d, "a")
    claimed = _floats(d, "claimed")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"h = {_to_int(d['h'])}\n")
        f.write(f"w = {_to_int(d['w'])}\n")
        f.write(f"claimed = {_toml_array([q32(v) for v in claimed])}\n")


def writer_patch_claimed(d: dict, path: Path):
    a = _floats(d, "a")
    claimed = _floats(d, "claimed")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"patch_size = {_to_int(d['patch_size'])}\n")
        f.write(f"claimed = {_toml_array([q32(v) for v in claimed])}\n")


def writer_slice_low_high(d: dict, path: Path):
    a = _floats(d, "a")
    claimed = _floats(d, "claimed")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"low = {_to_int(d['low'])}\n")
        f.write(f"high = {_to_int(d['high'])}\n")
        f.write(f"claimed = {_toml_array([q32(v) for v in claimed])}\n")


def writer_0398(d: dict, path: Path):
    a_arr = _floats(d, "a_arr")
    claimed = _floats(d, "claimed")
    with open(path, "w") as f:
        f.write(f"a_arr = {_toml_array([q32(v) for v in a_arr])}\n")
        f.write(f"a_scalar = {q32(float(d['a_scalar']))}\n")
        f.write(f"b_scalar = {q32(float(d['b_scalar']))}\n")
        f.write(f"claimed = {_toml_array([q32(v) for v in claimed])}\n")


def writer_0399(d: dict, path: Path):
    a_arr = _floats(d, "a_arr")
    c_arr = _floats(d, "c_arr")
    claimed = _floats(d, "claimed")
    with open(path, "w") as f:
        f.write(f"a_arr = {_toml_array([q32(v) for v in a_arr])}\n")
        f.write(f"a_scalar = {q32(float(d['a_scalar']))}\n")
        f.write(f"b_scalar = {q32(float(d['b_scalar']))}\n")
        f.write(f"c_arr = {_toml_array([q32(v) for v in c_arr])}\n")
        f.write(f"claimed = {_toml_array([q32(v) for v in claimed])}\n")


def writer_empty(d: dict, path: Path):
    # Cases 0400, 0401, 0451 take no inputs.
    open(path, "w").close()


def writer_ravel_index(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"dims = {_toml_array(_ints(d, 'dims'))}\n")
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"index = {_toml_array(_ints(d, 'index'))}\n")
        f.write(f"claimed = {_to_int(d['claimed'])}\n")


def writer_xy_floats(d: dict, path: Path):
    _write_floats(path, ("x", _floats(d, "x")), ("y", _floats(d, "y")))


def writer_data_bin(d: dict, path: Path):
    a = _floats(d, "data")
    with open(path, "w") as f:
        f.write(f"data = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"bin_size = {_to_int(d['bin_size'])}\n")


def writer_ab_float_50(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")), ("b", _floats(d, "b")))


def writer_zero_rows_cols_scalar(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"zero_rows = {_to_int(d['zero_rows'])}\n")
        f.write(f"zero_cols = {_to_int(d['zero_cols'])}\n")


def writer_zero_rows_cols_arr(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"zero_rows = {_toml_array(_ints(d, 'zero_rows'))}\n")
        f.write(f"zero_cols = {_toml_array(_ints(d, 'zero_cols'))}\n")


def writer_0438(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"post = {_toml_array(_ints(d, 'post'))}\n")
        f.write(f"distance = {_toml_array(_ints(d, 'distance'))}\n")


def writer_X_floats(d: dict, path: Path):
    _write_floats(path, ("x", _floats(d, "X")))


def writer_X_floats_l2(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"X = {_toml_array([q32(float(v)) for v in d['X']])}\n")


def writer_0439(d: dict, path: Path):
    _write_floats(path, ("x", _floats(d, "X")))


def writer_0441(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"number = {_to_int(d['number'])}\n")


def writer_0442(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")


def writer_in1d_claimed(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")
        f.write(f"claimed_len = {_to_int(d['claimed_len'])}\n")
        f.write(f"claimed = {_toml_array(_ints(d, 'claimed'))}\n")


def writer_xy_dists(d: dict, path: Path):
    _write_floats(path, ("x_dists", _floats(d, "x_dists")),
                  ("y_dists", _floats(d, "y_dists")))


def writer_0450(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"second = {_toml_array(_ints(d, 'second'))}\n")
        f.write(f"third = {_toml_array(_ints(d, 'third'))}\n")


def writer_0453(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"x = {_toml_array(_ints(d, 'x'))}\n")


def writer_a_floats_na_ints(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"na = {_toml_array(_ints(d, 'na'))}\n")


def writer_dedup_claimed(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"claimed_len = {_to_int(d['claimed_len'])}\n")
        f.write(f"claimed = {_toml_array(_ints(d, 'claimed'))}\n")


def writer_window_3x3(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"a = {_toml_array(_ints(d, 'a'))}\n")
        f.write(f"size = {_toml_array(_ints(d, 'size'))}\n")
        f.write(f"claim_counts = {_toml_array(_ints(d, 'claim_counts'))}\n")
        f.write(f"claim_total = {_to_int(d['claim_total'])}\n")
        f.write(f"claim_flat = {_toml_array(_ints(d, 'claim_flat'))}\n")


def writer_0473(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"c = {_toml_array(_ints(d, 'c'))}\n")
        f.write(f"cnts = {_toml_array(_ints(d, 'cnts'))}\n")


def writer_xy_a_b(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"x = {_toml_array(_ints(d, 'x'))}\n")
        f.write(f"y = {_toml_array(_ints(d, 'y'))}\n")
        f.write(f"a = {_to_int(d['a'])}\n")
        f.write(f"b_val = {_to_int(d['b_val'])}\n")


def writer_xy_a_b_claimed(d: dict, path: Path):
    with open(path, "w") as f:
        f.write(f"x = {_toml_array(_ints(d, 'x'))}\n")
        f.write(f"y = {_toml_array(_ints(d, 'y'))}\n")
        f.write(f"a = {_to_int(d['a'])}\n")
        f.write(f"b_val = {_to_int(d['b_val'])}\n")
        f.write(f"claimed_len = {_to_int(d['claimed_len'])}\n")
        f.write(f"claimed = {_toml_array(_ints(d, 'claimed'))}\n")


def writer_0485(d: dict, path: Path):
    _write_floats(path, ("a", _floats(d, "a")), ("b", _floats(d, "b")))


def writer_0490(d: dict, path: Path):
    arr = [float(v) for v in _flatten(d["arr"])]
    with open(path, "w") as f:
        f.write(f"arr = {_toml_array([q32(v) for v in arr])}\n")
        f.write(f"n1 = {_toml_array(_ints(d, 'n1'))}\n")
        f.write(f"n2 = {_toml_array(_ints(d, 'n2'))}\n")


def writer_s12(d: dict, path: Path):
    _write_floats(path, ("s1", _floats(d, "s1")), ("s2", _floats(d, "s2")))


def writer_ab_float_int(d: dict, path: Path):
    a = _floats(d, "a")
    with open(path, "w") as f:
        f.write(f"a = {_toml_array([q32(v) for v in a])}\n")
        f.write(f"b = {_toml_array(_ints(d, 'b'))}\n")


WRITERS: dict[str, Callable[[dict, Path], None]] = {
    "0291": writer_0291,
    "0309": writer_0309,
    "0323": writer_0323,
    "0329": writer_pow_8x6,
    "0330": writer_pow_8x6,
    "0334": writer_three_50,
    "0335": writer_three_50,
    "0336": writer_0336,
    "0373": writer_0373,
    "0387": writer_0387,
    "0416": writer_bin_8x6,
    "0418": writer_bin_8x6,
    "0428": writer_0428,
    "0436": writer_mask_10x5,
    "0437": writer_mask_10x5,
    "0452": writer_norm_5x4,
    "0454": writer_norm_5x4,
    "0469": writer_complex_mean,
    "0470": writer_complex_mean,
    # --- new ---
    "0298": (lambda d, p: _emit(p, [
        f"a = {_toml_array([q32(v) for v in _floats(d, 'a')])}",
        f"n_unique = {_to_int(d['n_unique'])}",
        f"b = {_toml_array([q32(v) for v in _floats(d, 'b')])}",
    ])),
    "0301": writer_reshape_ncol,
    "0302": writer_reshape_nrow,
    "0303": writer_reshape_ncol,
    "0304": writer_reshape_ncol,
    "0310": writer_a_floats,
    "0311": writer_a_floats,
    "0312": writer_a_floats,
    "0313": writer_a_floats,
    "0314": writer_a_floats,
    "0317": writer_a_floats,
    "0318": writer_perm_8x6,
    "0319": writer_perm_3d,
    "0320": writer_a_floats,
    "0321": writer_a_floats,
    "0322": writer_a_ints,
    "0324": writer_0324,
    "0325": writer_0325,
    "0326": writer_0326,
    "0337": writer_a_floats,
    "0338": writer_a_floats,
    "0339": writer_a_floats,
    "0340": writer_x_ints,
    "0341": writer_x_ints,
    "0342": writer_x_ints,
    "0343": writer_x_ints,
    "0344": writer_0344,
    "0345": writer_0345,
    "0346": writer_0346,
    "0347": writer_0347,
    "0349": writer_a_floats,
    "0353": writer_0353,
    "0355": writer_sort3d_a_floats_b_ints,
    "0356": writer_sort3d_a_floats_b_ints,
    "0357": writer_sort3d_a_floats_b_ints,
    "0358": writer_sort3d_a_floats_b_ints,
    "0359": writer_a_floats,
    "0360": writer_a_floats,
    "0361": writer_a_floats,
    "0363": writer_insert_scalar,
    "0364": writer_0364,
    "0365": writer_insert_scalar,
    "0367": writer_0367,
    "0368": writer_a_ints,
    "0369": writer_a_ints,
    "0370": writer_a_ints,
    "0374": (lambda d, p: _emit(p, [
        f"grades = {_toml_array([q32(v) for v in _floats(d, 'grades')])}",
        f"eval = {_toml_array(_ints(d, 'eval'))}",
    ])),
    "0375": (lambda d, p: _emit(p, [
        f"grades = {_toml_array([q32(v) for v in _floats(d, 'grades')])}",
        f"threshold = {q32(float(d['threshold']))}",
    ])),
    "0381": writer_a_floats,
    "0382": writer_a_floats,
    "0383": writer_argsort_topn,
    "0385": writer_a_floats,
    "0388": (lambda d, p: _emit(p, [
        f"a = {_toml_array([q32(v) for v in _floats(d, 'a')])}",
        f"patch_size = {_to_int(d['patch_size'])}",
    ])),
    "0389": writer_0389,
    "0390": writer_patch_claimed,
    "0391": writer_slice_low_high,
    "0392": writer_slice_low_high,
    "0393": writer_slice_low_high,
    "0398": writer_0398,
    "0399": writer_0399,
    "0400": writer_empty,
    "0401": writer_empty,
    "0402": writer_ravel_index,
    "0403": writer_ravel_index,
    "0409": writer_xy_floats,
    "0412": writer_x_floats,
    "0413": writer_x_floats,
    "0414": writer_data_bin,
    "0415": writer_data_bin,
    "0417": writer_data_bin,
    "0419": writer_data_bin,
    "0422": writer_ab_float_50,
    "0429": writer_a_floats,
    "0430": writer_a_floats,
    "0431": writer_a_ints,
    "0433": writer_zero_rows_cols_scalar,
    "0434": writer_zero_rows_cols_arr,
    "0435": writer_a_ints,
    "0438": writer_0438,
    "0439": writer_0439 if False else (lambda d, p: _emit(p, [
        f"x = {_toml_array([q32(v) for v in _floats(d, 'x')])}",
    ])),
    "0441": writer_0441,
    "0442": writer_0442,
    "0443": writer_in1d_claimed,
    "0444": writer_in1d_claimed,
    "0448": writer_xy_dists,
    "0449": writer_xy_dists,
    "0450": writer_0450,
    "0451": writer_empty,
    "0453": writer_0453,
    "0459": writer_a_floats_na_ints,
    "0460": writer_a_floats_na_ints,
    "0462": writer_dedup_claimed,
    "0463": writer_dedup_claimed,
    "0467": writer_window_3x3,
    "0468": writer_window_3x3,
    "0472": writer_a_floats,
    "0473": writer_0473,
    "0478": writer_a_ints,
    "0480": writer_xy_a_b,
    "0481": writer_xy_a_b_claimed,
    "0485": writer_0485,
    "0490": writer_0490,
    "0491": writer_s12,
    "0492": writer_s12,
    "0500": writer_a_floats,
    "0501": writer_ab_float_int,
    "0502": writer_ab_float_int,
    "0503": writer_ab_float_int,
    "0504": writer_ab_float_int,
    "0505": writer_ab_float_int,
    "0507": writer_im_ints,
    "0508": writer_a_ints,
    "0509": writer_im_floats,
    "0510": writer_im_floats,
}

for _abandoned_case in ABANDONED_BASELINE_CASES:
    WRITERS.pop(_abandoned_case, None)


# ---------------------------------------------------------------------------
# Substitutions and skip rationale (see NOIR_BASELINES.md).
# ---------------------------------------------------------------------------

# Cases intentionally skipped (see per-case SKIPPED.md notes).
SKIPPED: dict[str, str] = {
    "0300": "np.percentile requires runtime floor + sort-and-gather; too costly",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case: str
    status: str = "pending"
    first_prove_s: float | None = None
    warm_prove_times_s: list = field(default_factory=list)
    prove_seconds_median: float | None = None
    nargo_execute_s: float | None = None
    verify_ok: bool | None = None
    verify_time_s: float | None = None
    gate_count: int | None = None
    proof_bytes: int | None = None
    vk_bytes: int | None = None
    error: str | None = None


def resolve_tools(nargo_arg: str | None, bb_arg: str | None) -> tuple[Path, Path]:
    nargo = Path(nargo_arg) if nargo_arg else None
    bb = Path(bb_arg) if bb_arg else None
    if nargo is None:
        which = shutil.which("nargo")
        nargo = Path(which) if which else DEFAULT_NARGO
    if bb is None:
        which = shutil.which("bb")
        bb = Path(which) if which else DEFAULT_BB
    if not nargo.exists():
        sys.stderr.write(f"nargo not found at {nargo}\n")
        sys.stderr.write(INSTALL_INSTRUCTIONS)
        sys.exit(2)
    if not bb.exists():
        sys.stderr.write(f"bb not found at {bb}\n")
        sys.stderr.write(INSTALL_INSTRUCTIONS)
        sys.exit(2)
    return nargo, bb


def stage_case(case: str, workdir: Path) -> Path:
    src = THIS_DIR / f"ds1000_case{case}"
    if not (src / "main.nr").exists():
        raise FileNotFoundError(src / "main.nr")
    pkg_dir = workdir / f"case{case}"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "src").mkdir(exist_ok=True)
    shutil.copy(src / "main.nr", pkg_dir / "src" / "main.nr")
    (pkg_dir / "Nargo.toml").write_text(
        f'[package]\nname = "case{case}"\ntype = "bin"\nauthors = [""]\n\n'
        f'[dependencies]\nnoir_fpc = {{ path = "{NOIR_FPC}" }}\n'
    )
    return pkg_dir


def run_case(case: str, repeats: int, nargo: Path, bb: Path,
             workroot: Path) -> CaseResult:
    cr = CaseResult(case=case)
    if case in SKIPPED:
        cr.status = f"skipped: {SKIPPED[case]}"
        return cr
    if case not in WRITERS:
        cr.status = "skipped: no writer registered"
        cr.error = "no writer"
        return cr
    if case not in INPUT_BUILDERS:
        cr.status = "skipped: no INPUT_BUILDER"
        cr.error = "no input builder"
        return cr

    try:
        pkg = stage_case(case, workroot)
    except FileNotFoundError as e:
        cr.status = "missing"
        cr.error = f"main.nr not found: {e}"
        return cr

    inp = INPUT_BUILDERS[case]()
    try:
        WRITERS[case](inp, pkg / "Prover.toml")
    except Exception as e:
        cr.status = "input_error"
        cr.error = f"writer: {type(e).__name__}: {e}"
        return cr

    # 1. nargo execute (witness generation).
    t0 = time.time()
    proc = subprocess.run(
        [str(nargo), "execute"],
        cwd=pkg, capture_output=True, text=True,
    )
    cr.nargo_execute_s = time.time() - t0
    if proc.returncode != 0:
        cr.status = "nargo_execute_failed"
        cr.error = (proc.stdout + proc.stderr)[-800:]
        return cr

    bytecode = pkg / "target" / f"case{case}.json"
    witness = pkg / "target" / f"case{case}.gz"
    if not bytecode.exists() or not witness.exists():
        cr.status = "missing_artifacts"
        cr.error = f"expected {bytecode}, {witness}"
        return cr

    target = pkg / "target"

    # 2. write_vk once.
    proc = subprocess.run(
        [str(bb), "write_vk",
         "-b", str(bytecode), "-o", str(target),
         "--scheme", "ultra_honk",
         "--verifier_target", "noir-recursive-no-zk"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        cr.status = "vk_failed"
        cr.error = (proc.stdout + proc.stderr)[-800:]
        return cr
    vk_path = target / "vk"
    if vk_path.exists():
        cr.vk_bytes = vk_path.stat().st_size

    # 3. gate count via `bb gates`.
    proc = subprocess.run(
        [str(bb), "gates",
         "-b", str(bytecode),
         "--scheme", "ultra_honk",
         "--verifier_target", "noir-recursive-no-zk"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        # Output is JSON-ish; try to extract acir_opcodes and circuit_size.
        try:
            blob = proc.stdout.strip()
            # Strip preamble lines like "Scheme is: ...".
            jstart = blob.find("{")
            if jstart >= 0:
                obj = json.loads(blob[jstart:])
                fns = obj.get("functions") or []
                if fns:
                    cr.gate_count = fns[0].get("circuit_size") or fns[0].get("acir_opcodes")
        except Exception:
            pass

    # 4. Warm + repeated prove. Each call is `bb prove ... -k <vk>` so we
    #    avoid recomputing the VK each iteration. The reported time below
    #    is wall-clock around the prove subprocess.
    prove_args = [
        str(bb), "prove",
        "-b", str(bytecode),
        "-w", str(witness),
        "-k", str(vk_path),
        "-o", str(target),
        "--scheme", "ultra_honk",
        "--verifier_target", "noir-recursive-no-zk",
    ]
    times = []
    for i in range(repeats):
        t0 = time.time()
        proc = subprocess.run(prove_args, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            cr.status = f"prove_failed_on_iter_{i}"
            cr.error = (proc.stdout + proc.stderr)[-800:]
            return cr
        times.append(dt)
    cr.first_prove_s = times[0] if times else None
    cr.warm_prove_times_s = times[1:] if len(times) > 1 else times[:]
    if cr.warm_prove_times_s:
        cr.prove_seconds_median = statistics.median(cr.warm_prove_times_s)
    elif times:
        cr.prove_seconds_median = times[0]

    proof_path = target / "proof"
    if proof_path.exists():
        cr.proof_bytes = proof_path.stat().st_size

    # 5. verify.
    pi_path = target / "public_inputs"
    if pi_path.exists():
        verify_args = [
            str(bb), "verify",
            "-k", str(vk_path), "-p", str(proof_path), "-i", str(pi_path),
            "--scheme", "ultra_honk",
            "--verifier_target", "noir-recursive-no-zk",
        ]
        t0 = time.time()
        proc = subprocess.run(verify_args, capture_output=True, text=True)
        cr.verify_time_s = time.time() - t0
        cr.verify_ok = proc.returncode == 0
        if not cr.verify_ok:
            cr.error = (proc.stdout + proc.stderr)[-400:]
            cr.status = "verify_failed"
            return cr
    else:
        cr.verify_ok = False
        cr.error = "no public_inputs file"

    cr.status = "ok"
    return cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3,
                    help="number of prove iterations per case (first is cold)")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids; default = all with main.nr")
    ap.add_argument("--out", default=str(THIS_DIR / "noir_baseline_results.json"))
    ap.add_argument("--nargo", default=None,
                    help="override path to `nargo` (default: PATH or ~/.nargo/bin/nargo)")
    ap.add_argument("--bb", default=None,
                    help="override path to `bb` (default: PATH or ~/.bb/bb)")
    ap.add_argument("--workdir", default=None,
                    help="staging directory (default: a tempdir; deleted on exit)")
    args = ap.parse_args()

    nargo, bb = resolve_tools(args.nargo, args.bb)
    print(f"nargo: {nargo}")
    print(f"bb:    {bb}")

    if args.cases and args.cases != "all":
        requested = [c.strip() for c in args.cases.split(",") if c.strip()]
        skipped = [c for c in requested if c in ABANDONED_BASELINE_CASES]
        for c in skipped:
            print(f"Skipping abandoned Noir baseline {c}: {ABANDONED_BASELINE_CASES[c]}")
        cases = [c for c in requested if c not in ABANDONED_BASELINE_CASES]
    else:
        # --cases all (or default): all cases with main.nr + a writer.
        cases = sorted(
            c for c in WRITERS
            if c not in ABANDONED_BASELINE_CASES
            if (THIS_DIR / f"ds1000_case{c}" / "main.nr").exists()
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.workdir:
        workroot = Path(args.workdir)
        workroot.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.mkdtemp(prefix="noir_baselines_")
        workroot = Path(tmp)
        cleanup = True

    results: list[CaseResult] = []
    try:
        for i, c in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] case {c} ", end="", flush=True)
            t0 = time.time()
            r = run_case(c, args.repeats, nargo, bb, workroot)
            wall = time.time() - t0
            results.append(r)
            if r.status == "ok":
                med = r.prove_seconds_median or 0.0
                print(f"prove={med*1000:.0f}ms first={r.first_prove_s*1000:.0f}ms "
                      f"gates={r.gate_count} verify={r.verify_ok} "
                      f"proof={r.proof_bytes}B (wall {wall:.1f}s)")
            else:
                print(f"{r.status} (wall {wall:.1f}s)"
                      + (f" -- {r.error[:200]}" if r.error else ""))
            out_path.write_text(json.dumps(
                {"results": [asdict(x) for x in results]},
                indent=2,
            ))
    finally:
        if cleanup:
            shutil.rmtree(workroot, ignore_errors=True)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

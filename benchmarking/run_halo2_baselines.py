"""Run the handwritten halo2 baselines against halo2-graph and report prove times.

Mirrors the legacy `benchmarker_halo2.py` flow (keygen → prove → verify) but
for the *handwritten* per-case `target.rs` files we just generated.

Usage:
    python3 run_halo2_baselines.py [--repeats N] [--cases 0309,0323,...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

HALO2_FOLDER = Path(os.environ.get("HALO2_FOLDER", "/home/zhantong/halo2-graph"))
THIS_DIR = Path(__file__).parent
K_DEFAULT = "16"
K_MIN_BISECT = 4
K_MAX_BISECT = 22
LOOKUP_BITS = "6"
RUSTUP_TOOLCHAIN_OVERRIDE = "nightly-2025-12-01"

ABANDONED_BASELINE_CASES: dict[str, str] = {
    "0298": "np.unique has runtime-dependent output width; target.rs specialized the fixture shape",
    "0383": "runtime N changes output length; target.rs asserted the fixture N",
    "0388": "runtime patch_size changes output shape; target.rs fixed the fixture patch_size",
    "0389": "runtime h/w change output shape; target.rs fixed the fixture h/w",
    "0390": "runtime patch_size changes output shape; target.rs fixed the fixture patch_size",
    "0391": "runtime slice width changes output contract; target.rs emitted only fixture width",
    "0392": "runtime slice height changes output contract; target.rs emitted only fixture height",
    "0393": "runtime clamped slice width changes output contract; target.rs emitted only fixture width",
    "0467": "runtime window size changes nested output shape; target.rs fixed size=(3,3)",
    "0468": "runtime window size changes nested output shape; target.rs fixed size=(3,3)",
}


# ---------------------------------------------------------------------------
# Per-case input builders
# ---------------------------------------------------------------------------

def build_input_0309() -> dict:
    # Apples-to-apples with Zinnia: no claim field. Inputs are 48 floats
    # row-major; the circuit exposes the computed argmax as public.
    a = [0.0] * 48
    for i in range(8):
        for j in range(6):
            a[i * 6 + j] = 0.10 * (i + 1) + 0.01 * (j + 1)
    a[3 * 6 + 2] = 99.0
    return {"a": a}


def build_input_0323() -> dict:
    # Single integer input; the circuit exposes sin(deg2rad(degree)).
    return {"degree": "60"}


def build_input_0329() -> dict:
    a = [1.0 + 0.1 * (i + j) for i in range(8) for j in range(6)]
    return {"a": a, "power": "3"}


def build_input_0334() -> dict:
    a = [0.30 + 0.01 * i for i in range(50)]
    b = [0.30 + 0.02 * i for i in range(50)]
    c = [0.30 + 0.03 * i for i in range(50)]
    return {"a": a, "b": b, "c": c}


def build_input_0336() -> dict:
    a = [1.0 + 0.1 * (i * 8 + j) for i in range(8) for j in range(8)]
    return {"a": a}


def build_input_0291() -> dict:
    # 100 integers (unused); shape result is 4 constants Zinnia dedupes to (2, 5).
    return {"a": [str(i) for i in range(100)]}


def build_input_0310() -> dict:
    a = [0.0] * 48
    for i in range(8):
        for j in range(6):
            a[i * 6 + j] = 0.10 * (i + 1) + 0.01 * (j + 1)
    a[5 * 6 + 3] = -99.0  # deliberate min ⇒ argmin = 33
    return {"a": a}


def build_input_0317() -> dict:
    a = [0.0] * 1000
    for i in range(10):
        for j in range(20):
            for k in range(5):
                a[i * 100 + j * 5 + k] = i + j * 0.01 + k * 0.0001
    return {"a": a}


def build_input_0335() -> dict:
    return {
        "a": [0.10 * i for i in range(50)],
        "b": [0.20 * i for i in range(50)],
        "c": [0.30 * i for i in range(50)],
    }


def build_input_0368() -> dict:
    # All rows equal ⇒ isclose(...).all() == True.
    return {"a": ["5", "7", "9", "5", "7", "9", "5", "7", "9"]}


def build_input_0409() -> dict:
    x = [0.10 * (i * 6 + j) for i in range(8) for j in range(6)]
    y = [0.01 * (i * 6 + j) for i in range(8) for j in range(6)]
    return {"x": x, "y": y}


def build_input_0428() -> dict:
    return {"a": [float(i) for i in range(30)]}


def build_input_0441() -> dict:
    return {"a": ["10", "20", "30", "40"], "number": "30"}


def build_input_0452() -> dict:
    # X[i][j] = i*4 + j + 1, all positive ⇒ L1 row-sum = X[i][0..3].
    return {"x": [str(i * 4 + j + 1) for i in range(5) for j in range(4)]}


def build_input_0500() -> dict:
    return {"a": [0.01 * i for i in range(54)]}


def build_input_0324() -> dict:
    # Kernel signature is `degree: Integer`; the prior float literal tripped
    # Zinnia's Integer type-check in the runner. Cairo / SP1 / halo2 baselines
    # also accept the integer form.
    return {"degree": "60"}


def build_input_0337() -> dict:
    return {"a": [1.0 + 0.1 * (i * 10 + j) for i in range(8) for j in range(10)]}


def build_input_0339() -> dict:
    return {"a": [1.0 + 0.1 * (i * 10 + j) for i in range(8) for j in range(10)]}


def build_input_0349() -> dict:
    return {"a": [0.10 * (i + 1) + 0.01 * (j + 1) for i in range(8) for j in range(6)]}


def build_input_0361() -> dict:
    return {"a": [1.0 + 0.1 * (i * 6 + j) for i in range(6) for j in range(6)]}


def build_input_0429() -> dict:
    return {"a": [float(i) for i in range(30)]}


def build_input_0431() -> dict:
    # Integer[8]; some values inside ±2σ, one clear outlier.
    return {"a": ["10", "12", "11", "13", "12", "11", "10", "99"]}


def build_input_0436() -> dict:
    # Float[10,5]; per-row max at a deterministic column for clarity.
    a = [0.0] * 50
    for i in range(10):
        for j in range(5):
            a[i * 5 + j] = 0.1 * (i + 1) + 0.01 * (j + 1)
    return {"a": a}


def build_input_0437() -> dict:
    return build_input_0436()


def build_input_0453() -> dict:
    return {"x": [str(i * 4 + j + 1) for i in range(5) for j in range(4)]}


def build_input_0454() -> dict:
    return {"x": [str(i * 4 + j + 1) for i in range(5) for j in range(4)]}


def build_input_0459() -> dict:
    return {"a": [0.1 * i for i in range(5)], "na": [str(10 + i) for i in range(5)]}


def build_input_0345() -> dict:
    return {"a": [0.1*(i*6+j+1) for i in range(8) for j in range(6)], "col": "3", "multiply_number": "2"}
def build_input_0346() -> dict:
    return {"a": [0.1*(i*6+j+1) for i in range(8) for j in range(6)], "row": "3", "multiply_number": "2"}
def build_input_0433() -> dict:
    return {"a": [str(i*4+j+1) for i in range(4) for j in range(4)], "zero_rows": "1", "zero_cols": "0"}


def build_input_0303() -> dict: return {"a": [0.01*i for i in range(23)], "ncol": "5"}
def build_input_0434() -> dict: return {"a": [str(i*4+j+1) for i in range(4) for j in range(4)], "zero_rows": ["1","3"], "zero_cols": ["0","2"]}


def build_input_0341() -> dict: return build_input_0340()
def build_input_0342() -> dict: return build_input_0340()
def build_input_0370() -> dict: return build_input_0368()
def build_input_0449() -> dict: return build_input_0448()
def build_input_0301() -> dict: return {"a": [0.01*i for i in range(20)], "ncol": "5"}
def build_input_0302() -> dict: return {"a": [0.01*i for i in range(20)], "nrow": "5"}


def build_input_0312() -> dict:
    a = [0.10*(i+1)+0.01*(j+1) for i in range(8) for j in range(6)]
    a[5 * 6 + 2] = 99.0
    return {"a": a}
def build_input_0400() -> dict: return {}
def build_input_0401() -> dict: return {}
def build_input_0413() -> dict: return {"x": [0.1 * i for i in range(10)]}
def build_input_0448() -> dict:
    return {"x_dists": [0.1*(i*4+j) for i in range(3) for j in range(4)],
            "y_dists": [0.01*(i*4+j) for i in range(3) for j in range(4)]}
def build_input_0451() -> dict: return {}


def build_input_0343() -> dict: return {"x": [str(i*6+j+1) for i in range(5) for j in range(6)]}
def build_input_0412() -> dict: return {"x": [(-1.0)**i * 0.1*(i+1) for i in range(10)]}
def build_input_0430() -> dict: return {"a": [float(i) for i in range(30)]}
def build_input_0435() -> dict: return {"a": [str(i*4+j+1) for i in range(4) for j in range(4)]}
def build_input_0438() -> dict: return {"post": ["1","2","3","4"], "distance": ["2","4","6","8"]}
def build_input_0480() -> dict: return {"x": ["1","2","3","2","3","4","5","6"], "y": ["1","2","3","2","3","4","5","6"], "a": "3", "b_val": "3"}


def build_input_0344() -> dict: return {"mystr": "12345"}
def build_input_0367() -> dict: return {"pairs": [str(i) for i in range(6)], "array_of_arrays": ["10", "20", "30"]}
def build_input_0340() -> dict: return {"x": [str(i * 6 + j + 1) for i in range(5) for j in range(6)]}
def build_input_0359() -> dict: return {"a": [1.0 + 0.1 * (i * 3 + j) for i in range(3) for j in range(3)]}
def build_input_0360() -> dict: return {"a": [1.0 + 0.1 * (i * 4 + j) for i in range(4) for j in range(4)]}
def build_input_0338() -> dict: return {"a": [1.0 + 0.1 * (i * 8 + j) for i in range(8) for j in range(8)]}
def build_input_0369() -> dict: return {"a": ["5", "7", "9", "5", "7", "9", "5", "7", "9"]}
def build_input_0325() -> dict: return {"number": "1"}  # kernel sig is `number: Integer`
def build_input_0326() -> dict: return {"value": 0.5}   # arcsin(0.5) = 30°
def build_input_0320() -> dict:
    a = [0.10 * (i + 1) + 0.01 * (j + 1) for i in range(5) for j in range(6)]
    a[2 * 6 + 3] = -99.0  # deliberate min ⇒ idx=15 ⇒ (2, 3)
    return {"a": a}
def build_input_0321() -> dict:
    a = [0.10 * (i + 1) + 0.01 * (j + 1) for i in range(5) for j in range(6)]
    a[3 * 6 + 1] = 99.0   # deliberate max ⇒ idx=19 ⇒ (3, 1)
    return {"a": a}
def build_input_0311() -> dict:
    a = [0.10 * (i + 1) + 0.01 * (j + 1) for i in range(8) for j in range(6)]
    a[5 * 6 + 2] = 99.0   # deliberate max
    return {"a": a}


# === Auto-added builders for 72 cases ===
def build_input_0298() -> dict:
    # a = [1.5, -0.4, 1.3, 1.5, 1.3]; vals=[-0.4,1.3,1.5], idx=[2,0,1,2,1], n_unique=3.
    return {
        "a": [1.5, -0.4, 1.3, 1.5, 1.3],
        "n_unique": "3",
        "b": [
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ],
    }


def build_input_0300() -> dict:
    # a = [1..20], p = 25; percentile linear-interp lo=4, frac=0.75 → 5.75.
    return {"a": [float(i + 1) for i in range(20)], "p": "25"}


def build_input_0304() -> dict:
    # A = [1..23], ncol = 5 → rows = 4, col_keep = 20.
    return {"a": [float(i + 1) for i in range(23)], "ncol": "5"}


def build_input_0313() -> dict:
    # Place argmax at (3, 2) ⇒ flat index 20 (deliberately interior).
    a = [0.0] * 48
    for i in range(8):
        for j in range(6):
            a[i * 6 + j] = 0.10 * (i + 1) + 0.01 * (j + 1)
    a[3 * 6 + 2] = 99.0
    return {"a": a}


def build_input_0314() -> dict:
    # First argmax at (3,2)=99.0 → replaced by min (0.11). Second argmax at (7,5)=0.86.
    a = [0.0] * 48
    for i in range(8):
        for j in range(6):
            a[i * 6 + j] = 0.10 * (i + 1) + 0.01 * (j + 1)
    a[3 * 6 + 2] = 99.0
    return {"a": a}


def build_input_0318() -> dict:
    # permutation = [2, 0, 4, 1, 5, 3]; a[i][j] = (i+1)*10 + (j+1).
    a = [float((i + 1) * 10 + (j + 1)) for i in range(8) for j in range(6)]
    return {"a": a, "permutation": ["2", "0", "4", "1", "5", "3"]}


def build_input_0319() -> dict:
    # permutation = [2, 0, 4, 1, 5, 3, 7, 6]; a[i][j][k] = 100*(i+1)+10*(j+1)+(k+1).
    a = [float(100 * (i + 1) + 10 * (j + 1) + (k + 1))
         for i in range(8) for j in range(6) for k in range(8)]
    return {"a": a, "permutation": ["2", "0", "4", "1", "5", "3", "7", "6"]}


def build_input_0322() -> dict:
    # 2x2 ints; unique min at a[1][0]=1 ⇒ argwhere = [(1,0)], length = 1.
    return {"a": ["5", "9", "1", "7"]}


def build_input_0330() -> dict:
    # a[i][j] = (i+1) + 0.1*(j+1); power = 3.
    a = [(i + 1) + 0.1 * (j + 1) for i in range(8) for j in range(6)]
    return {"a": a, "power": "3"}


def build_input_0347() -> dict:
    # row=3 (so row-1=2), divide_number=2; row 2 is [2,4,2,2,1,1] → product after /2 = 0.25.
    rows = [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [2.0, 4.0, 2.0, 2.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    a = [v for r in rows for v in r]
    return {"a": a, "row": "3", "divide_number": "2"}


def build_input_0353() -> dict:
    # A row 0 = [1,2,3] matches B[2]; A row 1 = [4,5,6] is kept.
    a = ["1", "2", "3", "4", "5", "6"]
    b = [
        "0", "0", "0",
        "1", "1", "1",
        "1", "2", "3",
        "2", "2", "2",
        "3", "3", "3",
        "7", "8", "9",
        "9", "9", "9",
    ]
    return {"a": a, "b": b}


def build_input_0355() -> dict:
    # a[:,j,k] = [3,1,2] for every (j,k) ⇒ argsort axis-0 = [1,2,0]; b[i][j][k]=100i+10j+k.
    a = []
    vals = [3.0, 1.0, 2.0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a.append(vals[i])
    b = [str(100 * i + 10 * j + k) for i in range(3) for j in range(3) for k in range(3)]
    return {"a": a, "b": b}
def build_input_0356() -> dict:
    a = [0.0] * 27
    b = [""] * 27
    raw = [[[0.1*i + 0.01*j + 0.001*k for k in range(3)] for j in range(3)] for i in range(3)]
    raw[0], raw[1] = raw[1], raw[0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a[i*9 + j*3 + k] = raw[i][j][k]
                b[i*9 + j*3 + k] = str(100*i + 10*j + k)
    return {"a": a, "b": b}

def build_input_0357() -> dict:
    a = [0.0] * 27
    b = [""] * 27
    raw = [[[0.1*i + 0.01*j + 0.001*k for k in range(3)] for j in range(3)] for i in range(3)]
    raw[0], raw[1] = raw[1], raw[0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a[i*9 + j*3 + k] = raw[i][j][k]
                b[i*9 + j*3 + k] = str(100*i + 10*j + k)
    return {"a": a, "b": b}

def build_input_0358() -> dict:
    a = [0.0] * 27
    b = [""] * 27
    layer_const = [3.0, 1.0, 2.0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a[i*9 + j*3 + k] = layer_const[i]
                b[i*9 + j*3 + k] = str(100*i + 10*j + k)
    return {"a": a, "b": b}

def build_input_0363() -> dict:
    return {"a": [(i + 1) / 10.0 for i in range(100)], "pos": "42", "element": "999"}

def build_input_0364() -> dict:
    return {
        "a": [float(i) + 0.01 * j for i in range(100) for j in range(10)],
        "pos": "5",
        "element": [999.0 + j for j in range(10)],
    }

def build_input_0365() -> dict:
    return {"a": [(i + 1) / 10.0 for i in range(100)], "pos": "42", "element": "999"}

def build_input_0373() -> dict:
    return {"grades": [(50 - i) * 1.5 for i in range(50)]}

def build_input_0374() -> dict:
    return {"grades": [float(i + 1) for i in range(50)], "eval": ["0", "10", "25", "50", "75"]}

def build_input_0375() -> dict:
    return {"grades": [float(i + 1) for i in range(50)], "threshold": 0.5}

def build_input_0381() -> dict:
    return {"a": [((i * 37 + 13) % 1000) + 0.5 * i for i in range(100)]}

def build_input_0382() -> dict:
    return {"a": [((i * 37 + 13) % 1000) + 0.5 * i for i in range(100)]}

def build_input_0383() -> dict:
    return {"a": [((i * 37 + 13) % 1000) + 0.5 * i for i in range(100)], "n": "10"}
def build_input_0385() -> dict:
    a = [i * 0.001 + j * 0.0001 for i in range(100) for j in range(200)]
    return {"a": a}


def build_input_0387() -> dict:
    a = [i * 0.001 + j * 0.0001 for i in range(100) for j in range(200)]
    return {"a": a}


def build_input_0388() -> dict:
    a = [i * 0.001 + j * 0.0001 for i in range(100) for j in range(200)]
    return {"a": a, "patch_size": "4"}


def build_input_0389() -> dict:
    H_OUT, W_OUT, NROWS, NCOLS = 6, 12, 3, 4
    a = [i * 12 + j * 4 + k for i in range(6) for j in range(3) for k in range(4)]
    a = [float(v) for v in a]
    claimed = []
    for out_i in range(H_OUT):
        for out_j in range(W_OUT):
            out_idx = out_i * W_OUT + out_j
            i0 = out_idx // 36
            i1 = (out_idx // 12) % 3
            i2 = (out_idx // 4) % 3
            i3 = out_idx % 4
            src = (i0 * NROWS + i2) * (NROWS * NCOLS) + i1 * NCOLS + i3
            claimed.append(a[src])
    return {"a": a, "h": "6", "w": "12", "claimed": claimed}


def build_input_0390() -> dict:
    H, W, PS = 100, 200, 10
    H_BLOCKS, W_BLOCKS = H // PS, W // PS
    a = [float(i * 200 + j) for i in range(H) for j in range(W)]
    claimed = []
    for p in range(H_BLOCKS * W_BLOCKS):
        block_col = p // H_BLOCKS
        block_row = p % H_BLOCKS
        for r in range(PS):
            for c in range(PS):
                claimed.append(float((block_row * PS + r) * 200 + (block_col * PS + c)))
    return {"a": a, "patch_size": "10", "claimed": claimed}


def build_input_0391() -> dict:
    ROWS, COLS = 20, 10
    low, high = 3, 7
    a = [float((i + 1) * 10 + (j + 1)) for i in range(ROWS) for j in range(COLS)]
    claimed = [float((i + 1) * 10 + (low + k + 1)) for i in range(ROWS) for k in range(high - low)]
    return {"a": a, "low": "3", "high": "7", "claimed": claimed}


def build_input_0392() -> dict:
    ROWS, COLS = 20, 10
    low, high = 5, 15
    a = [float((i + 1) * 10 + (j + 1)) for i in range(ROWS) for j in range(COLS)]
    claimed = [float((low + i + 1) * 10 + (j + 1)) for i in range(high - low) for j in range(COLS)]
    return {"a": a, "low": "5", "high": "15", "claimed": claimed}


def build_input_0393() -> dict:
    ROWS, COLS = 20, 10
    low, high_in = 3, 20
    high = min(high_in, COLS)
    a = [float((i + 1) * 10 + (j + 1)) for i in range(ROWS) for j in range(COLS)]
    claimed = [float((i + 1) * 10 + (low + k + 1)) for i in range(ROWS) for k in range(high - low)]
    return {"a": a, "low": "3", "high": "20", "claimed": claimed}


def build_input_0398() -> dict:
    A = [4.0, 2.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0]
    claimed = [2.0, 2.0, 1.5, 2.25, 2.125, 1.0625, 1.03125, 2.515625]
    # Kernel sig: (A: Float[Array, "8"], a: Float, b: Float). Keep `a_arr` /
    # `a_scalar` / `b_scalar` aliases so the handwritten halo2 baseline (which
    # imports this builder directly) still works.
    return {"A": A, "a": 0.5, "b": 0.5,
            "a_arr": A, "a_scalar": 0.5, "b_scalar": 0.5, "claimed": claimed}


def build_input_0399() -> dict:
    A = [4.0, 2.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0]
    claimed = [2.0, 2.0, 2.5, 3.75, 4.125, 3.9375, 4.53125, 6.234375]
    # Kernel sig: (A: Float[Array, "8"], a: Float, b: Float, c: Float). The
    # legacy `c_arr` was a per-cell tap but the kernel multiplies by a scalar
    # `c`, so collapse to the 0.5 used everywhere else.
    return {"A": A, "a": 0.5, "b": 0.5, "c": 0.5,
            "a_arr": A, "a_scalar": 0.5, "b_scalar": 0.5, "c_arr": [0.5] * 8,
            "claimed": claimed}


def build_input_0402() -> dict:
    dims = [8, 9, 8, 8, 8]
    index = [2, 3, 4, 5, 6]
    a = [k * 0.001 for k in range(8 * 9 * 8 * 8 * 8)]
    linear = index[0] + dims[0]*index[1] + dims[0]*dims[1]*index[2] + dims[0]*dims[1]*dims[2]*index[3] + dims[0]*dims[1]*dims[2]*dims[3]*index[4]
    return {"dims": [str(d) for d in dims], "a": a, "index": [str(i) for i in index], "claimed": str(linear)}


def build_input_0403() -> dict:
    dims = [8, 9, 8, 8, 8]
    index = [2, 3, 4, 5, 6]
    a = [k * 0.001 for k in range(8 * 9 * 8 * 8 * 8)]
    acc = index[0]
    for k in range(1, 5):
        acc = acc * dims[k] + index[k]
    return {"dims": [str(d) for d in dims], "a": a, "index": [str(i) for i in index], "claimed": str(acc)}
def build_input_0414() -> dict:
    return {"data": [float(i + 1) for i in range(8)], "bin_size": 2}


def build_input_0415() -> dict:
    return {"data": [1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0, 8.0], "bin_size": 2}


def build_input_0416() -> dict:
    return {"data": [(i + 1) * 10 + (j + 1) for i in range(8) for j in range(6)],
            "bin_size": 2}


def build_input_0417() -> dict:
    return {"data": [float(i + 1) for i in range(8)], "bin_size": 2}


def build_input_0418() -> dict:
    return {"data": [(i + 1) * 10 + (j + 1) for i in range(8) for j in range(6)],
            "bin_size": 2}


def build_input_0419() -> dict:
    return {"data": [(i + 1) * 10 + (j + 1) for i in range(8) for j in range(6)],
            "bin_size": 2}


def build_input_0422() -> dict:
    a = [float(i + 1) for i in range(50)]
    b = [0.0] * 50
    b[10] = 1.0
    return {"a": a, "b": b}


def build_input_0439() -> dict:
    return {"x": [0.01 * (i + 1) + 0.001 * (k + 1)
                  for i in range(10) for k in range(20)]}


def build_input_0442() -> dict:
    return {"a": list(range(20)), "b": [3, 7, 13]}


def build_input_0443() -> dict:
    claimed = [3, 7, 13] + [0] * 17
    return {"a": [str(i) for i in range(20)],
            "b": ["3", "7", "13"],
            "claimed_len": "3",
            "claimed": [str(v) for v in claimed]}


def build_input_0444() -> dict:
    kept = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    claimed = kept + [0] * (20 - len(kept))
    return {"a": [str(i) for i in range(20)],
            "b": ["3", "10", "17"],
            "claimed_len": str(len(kept)),
            "claimed": [str(v) for v in claimed]}


def build_input_0450() -> dict:
    a = [float(i * 100 + j * 10 + k)
         for i in range(7) for j in range(8) for k in range(9)]
    return {"a": a, "second": ["2", "5"], "third": ["3", "7"]}
def build_input_0460() -> dict:
    return {"a": [0.1, 0.2, 0.3, 0.4, 0.5], "na": ["2", "4", "6", "8", "10"]}


def build_input_0462() -> dict:
    a = [0, 0, 1, 1, 2, 3, 3, 0, 4, 4, 5, 0, 6, 7, 7, 8, 0, 9, 9, 0]
    kept = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    claimed = kept + [0] * (20 - len(kept))
    return {"a": [str(v) for v in a], "claimed_len": str(len(kept)), "claimed": [str(v) for v in claimed]}


def build_input_0463() -> dict:
    a = [0, 0, 1, 1, 1, 2, 2, 0, 1, 3, 3, 3, 0, 0, 5, 5, 4, 4, 0, 7]
    expected = [1, 2, 1, 3, 5, 4, 7]
    claimed = expected + [0] * (20 - len(expected))
    return {"a": [str(v) for v in a], "claimed_len": str(len(expected)), "claimed": [str(v) for v in claimed]}


def build_input_0467() -> dict:
    a = [i * 4 + j + 1 for i in range(4) for j in range(4)]
    counts = [4, 6, 6, 4, 6, 9, 9, 6, 6, 9, 9, 6, 4, 6, 6, 4]
    flat = [
        1, 2, 5, 6,
        1, 2, 3, 5, 6, 7,
        2, 3, 4, 6, 7, 8,
        3, 4, 7, 8,
        1, 2, 5, 6, 9, 10,
        1, 2, 3, 5, 6, 7, 9, 10, 11,
        2, 3, 4, 6, 7, 8, 10, 11, 12,
        3, 4, 7, 8, 11, 12,
        5, 6, 9, 10, 13, 14,
        5, 6, 7, 9, 10, 11, 13, 14, 15,
        6, 7, 8, 10, 11, 12, 14, 15, 16,
        7, 8, 11, 12, 15, 16,
        9, 10, 13, 14,
        9, 10, 11, 13, 14, 15,
        10, 11, 12, 14, 15, 16,
        11, 12, 15, 16,
    ]
    return {"a": [str(v) for v in a], "size": ["3", "3"],
            "claim_counts": [str(v) for v in counts], "claim_total": "100",
            "claim_flat": [str(v) for v in flat]}


def build_input_0468() -> dict:
    a = [i * 4 + j + 1 for i in range(4) for j in range(4)]
    counts = [4, 6, 6, 4, 6, 9, 9, 6, 6, 9, 9, 6, 4, 6, 6, 4]
    flat = [
        1, 2, 5, 6,
        1, 2, 5, 6, 9, 10,
        5, 6, 9, 10, 13, 14,
        9, 10, 13, 14,
        1, 2, 3, 5, 6, 7,
        1, 2, 3, 5, 6, 7, 9, 10, 11,
        5, 6, 7, 9, 10, 11, 13, 14, 15,
        9, 10, 11, 13, 14, 15,
        2, 3, 4, 6, 7, 8,
        2, 3, 4, 6, 7, 8, 10, 11, 12,
        6, 7, 8, 10, 11, 12, 14, 15, 16,
        10, 11, 12, 14, 15, 16,
        3, 4, 7, 8,
        3, 4, 7, 8, 11, 12,
        7, 8, 11, 12, 15, 16,
        11, 12, 15, 16,
    ]
    return {"a": [str(v) for v in a], "size": ["3", "3"],
            "claim_counts": [str(v) for v in counts], "claim_total": "100",
            "claim_flat": [str(v) for v in flat]}


def build_input_0469() -> dict:
    return {"a": [1.5, 2.5, 5.0]}


def build_input_0470() -> dict:
    return {"a": [1.5, 2.5, 5.0]}


def build_input_0472() -> dict:
    return {"a": [k * 0.01 for k in range(9072)]}


def build_input_0473() -> dict:
    return {"c": ["1", "2", "3", "4", "5", "6", "7", "8"], "cnts": ["1", "5", "7"]}


def build_input_0478() -> dict:
    a = [i * 4 + j + 1 for i in range(4) for j in range(4)]
    return {"a": [str(v) for v in a]}


def build_input_0481() -> dict:
    x = [1, 5, 5, 2, 5, 5, 3, 5, 4]
    y = [2, 7, 7, 3, 8, 7, 4, 7, 5]
    expected = [1, 2, 5, 7]
    claimed = expected + [0] * (9 - len(expected))
    return {"x": [str(v) for v in x], "y": [str(v) for v in y],
            "a": "5", "b_val": "7", "claimed_len": str(len(expected)),
            "claimed": [str(v) for v in claimed]}


def build_input_0485() -> dict:
    a = [(i * 18 + j * 3 + p) * 0.1 for i in range(5) for j in range(6) for p in range(3)]
    b = [(p * 3 + k + 1) * 0.5 for p in range(3) for k in range(3)]
    return {"a": a, "b": b}
def build_input_0490() -> dict:
    # Kernel sig: arr: Float[Array, "5 4 4"]; send nested rank-3 so the runner
    # produces a (5,4,4) tensor instead of a flat 80-element vector.
    arr = [[[float(((i * 16 + j * 4 + k) % 30)) for k in range(4)] for j in range(4)] for i in range(5)]
    return {"arr": arr, "n1": ["5", "10", "15", "0", "8"], "n2": ["20", "25", "22", "30", "18"]}


def build_input_0491() -> dict:
    return {"s1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "s2": [1.0, 2.5, 3.0, 4.1, 5.0, 6.000001]}


def build_input_0492() -> dict:
    return {"s1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "s2": [1.0, 2.5, 3.0, 4.1, 5.0, 6.000001]}


def build_input_0501() -> dict:
    a = [((i * 13 * 2 + j * 2 + k + 1) / 100.0)
         for i in range(13) for j in range(13) for k in range(2)]
    b = [(i + j) % 2 for i in range(13) for j in range(13)]
    return {"a": a, "b": b}


def build_input_0502() -> dict:
    a = [((i * 13 * 2 + j * 2 + k + 1) / 100.0)
         for i in range(13) for j in range(13) for k in range(2)]
    b = [(i + j + 1) % 2 for i in range(13) for j in range(13)]
    return {"a": a, "b": b}


def build_input_0503() -> dict:
    a = [((i * 13 * 5 + j * 5 + k + 1) / 100.0)
         for i in range(13) for j in range(13) for k in range(5)]
    b = [(i + j) % 5 for i in range(13) for j in range(13)]
    return {"a": a, "b": b}


def build_input_0504() -> dict:
    return build_input_0503()


def build_input_0505() -> dict:
    return build_input_0503()


def build_input_0507() -> dict:
    rows = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 0],
        [0, 0, 4, 5, 6, 0],
        [0, 0, 7, 8, 9, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    return {"im": [str(v) for row in rows for v in row]}


def build_input_0508() -> dict:
    a = [[0] * 10 for _ in range(10)]
    counter = 1
    for di in range(4):
        for dj in range(4):
            a[2 + di][3 + dj] = counter
            counter += 1
    return {"a": [str(v) for row in a for v in row]}


def build_input_0509() -> dict:
    im = [(i * 10 + j + 1) / 10.0 for i in range(10) for j in range(10)]
    for (i, j) in [(1, 2), (1, 6), (4, 2), (4, 6)]:
        im[i * 10 + j] = 0.0
    return {"im": im}


def build_input_0510() -> dict:
    im = [0.0] * 100
    counter = 1
    for di in range(4):
        for dj in range(4):
            im[(3 + di) * 10 + (2 + dj)] = counter / 10.0
            counter += 1
    return {"im": im}


INPUT_BUILDERS = {
    "0345": build_input_0345, "0346": build_input_0346, "0433": build_input_0433,
    "0303": build_input_0303, "0434": build_input_0434,
    "0341": build_input_0341, "0342": build_input_0342, "0370": build_input_0370, "0449": build_input_0449,
    "0301": build_input_0301, "0302": build_input_0302,
    "0312": build_input_0312, "0400": build_input_0400, "0401": build_input_0401,
    "0413": build_input_0413, "0448": build_input_0448, "0451": build_input_0451,
    "0343": build_input_0343, "0412": build_input_0412, "0430": build_input_0430,
    "0435": build_input_0435, "0438": build_input_0438, "0480": build_input_0480,
    "0344": build_input_0344, "0367": build_input_0367, "0340": build_input_0340,
    "0359": build_input_0359, "0360": build_input_0360, "0338": build_input_0338,
    "0369": build_input_0369, "0325": build_input_0325, "0326": build_input_0326,
    "0320": build_input_0320, "0321": build_input_0321, "0311": build_input_0311,
    "0309": build_input_0309,
    "0323": build_input_0323,
    "0329": build_input_0329,
    "0334": build_input_0334,
    "0336": build_input_0336,
    "0291": build_input_0291,
    "0310": build_input_0310,
    "0317": build_input_0317,
    "0335": build_input_0335,
    "0368": build_input_0368,
    "0409": build_input_0409,
    "0428": build_input_0428,
    "0441": build_input_0441,
    "0452": build_input_0452,
    "0500": build_input_0500,
    "0324": build_input_0324,
    "0337": build_input_0337,
    "0339": build_input_0339,
    "0349": build_input_0349,
    "0361": build_input_0361,
    "0429": build_input_0429,
    "0431": build_input_0431,
    "0436": build_input_0436,
    "0437": build_input_0437,
    "0453": build_input_0453,
    "0454": build_input_0454,
    "0459": build_input_0459,
    "0298": build_input_0298,
    "0300": build_input_0300,
    "0304": build_input_0304,
    "0313": build_input_0313,
    "0314": build_input_0314,
    "0318": build_input_0318,
    "0319": build_input_0319,
    "0322": build_input_0322,
    "0330": build_input_0330,
    "0347": build_input_0347,
    "0353": build_input_0353,
    "0355": build_input_0355,
    "0356": build_input_0356,
    "0357": build_input_0357,
    "0358": build_input_0358,
    "0363": build_input_0363,
    "0364": build_input_0364,
    "0365": build_input_0365,
    "0373": build_input_0373,
    "0374": build_input_0374,
    "0375": build_input_0375,
    "0381": build_input_0381,
    "0382": build_input_0382,
    "0383": build_input_0383,
    "0385": build_input_0385,
    "0387": build_input_0387,
    "0388": build_input_0388,
    "0389": build_input_0389,
    "0390": build_input_0390,
    "0391": build_input_0391,
    "0392": build_input_0392,
    "0393": build_input_0393,
    "0398": build_input_0398,
    "0399": build_input_0399,
    "0402": build_input_0402,
    "0403": build_input_0403,
    "0414": build_input_0414,
    "0415": build_input_0415,
    "0416": build_input_0416,
    "0417": build_input_0417,
    "0418": build_input_0418,
    "0419": build_input_0419,
    "0422": build_input_0422,
    "0439": build_input_0439,
    "0442": build_input_0442,
    "0443": build_input_0443,
    "0444": build_input_0444,
    "0450": build_input_0450,
    "0460": build_input_0460,
    "0462": build_input_0462,
    "0463": build_input_0463,
    "0467": build_input_0467,
    "0468": build_input_0468,
    "0469": build_input_0469,
    "0470": build_input_0470,
    "0472": build_input_0472,
    "0473": build_input_0473,
    "0478": build_input_0478,
    "0481": build_input_0481,
    "0485": build_input_0485,
    "0490": build_input_0490,
    "0491": build_input_0491,
    "0492": build_input_0492,
    "0501": build_input_0501,
    "0502": build_input_0502,
    "0503": build_input_0503,
    "0504": build_input_0504,
    "0505": build_input_0505,
    "0507": build_input_0507,
    "0508": build_input_0508,
    "0509": build_input_0509,
    "0510": build_input_0510,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case: str
    keygen_ok: bool = False
    min_k: int | None = None
    # Timed k-probes from the measured argmin walk: list of [k, prove_s].
    k_probes: list = field(default_factory=list)
    prove_times_s: list[float] = field(default_factory=list)
    snark_size_bytes: int | None = None
    verify_ok: bool | None = None
    verify_time_s: float | None = None
    advice_cells: int | None = None
    fixed_cells: int | None = None
    range_lookup_cells: int | None = None
    error: str | None = None
    install_failed: bool = False
    no_builder: bool = False

    def median_prove_s(self) -> float | None:
        if not self.prove_times_s:
            return None
        return statistics.median(self.prove_times_s)


def install_target(case: str) -> None:
    src = THIS_DIR / "datasets" / "ds1000" / f"ds1000_case{case}/target.rs"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = HALO2_FOLDER / "examples" / f"case{case}.rs"
    dst.write_text(src.read_text())


def install_input(case: str) -> None:
    inp = INPUT_BUILDERS[case]()
    out = HALO2_FOLDER / "data" / f"case{case}.in"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inp))


def cargo_run(case: str, sub: str, k: int, env: dict) -> tuple[int, str]:
    proc = subprocess.run(
        [
            "cargo", "run", "--release", "--example", f"case{case}", "--",
            "--name", f"case{case}",
            "-k", str(k),
            "--input", f"case{case}.in",
            sub,
        ],
        cwd=HALO2_FOLDER,
        capture_output=True, text=True, env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


PROVE_RE = re.compile(r"Proving time:\s*([\d\.]+)\s*(ms|s)")
VERIFY_RE = re.compile(r"Snark verified successfully in\s*([\d\.]+)(ms|s|µs|us)")
ADVICE_RE = re.compile(r"Gate Chip\s*\|\s*Phase\s*\d+:\s*(\d+)\s*advice cells")
FIXED_RE = re.compile(r"Total\s*(\d+)\s*fixed cells")
RANGE_RE = re.compile(r"Total range check advice cells to lookup per phase:\s*\[([^\]]+)\]")


def parse_prove_time(log: str) -> float | None:
    m = PROVE_RE.search(log)
    if not m:
        return None
    return float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)


def parse_verify_time(log: str) -> float | None:
    m = VERIFY_RE.search(log)
    if not m:
        return None
    unit = m.group(2)
    v = float(m.group(1))
    if unit == "ms":
        return v / 1000.0
    if unit == "s":
        return v
    return v / 1_000_000.0  # µs / us


def parse_cell_counts(log: str) -> tuple[int | None, int | None, int | None]:
    advice = sum(int(m.group(1)) for m in ADVICE_RE.finditer(log)) or None
    fm = FIXED_RE.search(log)
    fixed = int(fm.group(1)) if fm else None
    rm = RANGE_RE.search(log)
    rng = None
    if rm:
        rng = sum(int(x.strip()) for x in rm.group(1).split(",") if x.strip())
    return advice, fixed, rng


def is_k_too_small(log: str) -> bool:
    lower = log.lower()
    return ("notenoughrowsavailable" in lower
            or "not enough rows" in lower
            or "k is too small" in lower
            # halo2-graph scaffold asserts `LOOKUP_BITS < k` — fires when
            # bisection tries k<=LOOKUP_BITS. Treat as "k too small" so
            # bisection moves up to the next k.
            or "lookup_bits needs to be less than degree" in lower)


def run_case(case: str, repeats: int, k_fixed: int | None) -> CaseResult:
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "536870912"
    env["LOOKUP_BITS"] = LOOKUP_BITS
    if RUSTUP_TOOLCHAIN_OVERRIDE:
        env["RUSTUP_TOOLCHAIN"] = RUSTUP_TOOLCHAIN_OVERRIDE

    if case not in INPUT_BUILDERS:
        return CaseResult(case=case, no_builder=True, error="no INPUT_BUILDER")

    try:
        install_target(case)
        install_input(case)
    except Exception as e:
        return CaseResult(case=case, install_failed=True, error=f"install failed: {e}")

    # k bisection (or fixed k).
    k_candidates = [k_fixed] if k_fixed is not None else list(range(K_MIN_BISECT, K_MAX_BISECT + 1))
    min_k = None
    first_keygen_log = ""
    for k in k_candidates:
        rc, log = cargo_run(case, "keygen", k, env)
        if rc != 0:
            if is_k_too_small(log):
                continue
            return CaseResult(case=case, error=f"keygen failed: {log[-400:]}")
        # Try one prove at this k to confirm rows fit prover-side.
        rc, plog = cargo_run(case, "prove", k, env)
        if rc != 0:
            if is_k_too_small(plog):
                continue
            return CaseResult(case=case, keygen_ok=True, min_k=k,
                              error=f"prove failed: {plog[-400:]}")
        min_k = k
        first_keygen_log = log + "\n" + plog
        break

    if min_k is None:
        return CaseResult(case=case, error=f"no viable k in [{K_MIN_BISECT}..{K_MAX_BISECT}]")

    # Ground-truth k selection: prove time is U-shaped in k (small k packs
    # cells into many columns, i.e. many committed polynomials; large k pays
    # 2^k per polynomial). min_k is the smallest *viable* k, not the fastest.
    # Probe upward with real keygen+prove and keep the measured argmin. The
    # 3% margin is a stop-rule noise guard only.
    k_probes: list = []
    best_k = min_k
    best_t = parse_prove_time(first_keygen_log)
    best_log = first_keygen_log
    last_keygen_k = min_k
    if k_fixed is None and best_t is not None:
        k_probes.append([min_k, best_t])
        k = min_k + 1
        while k <= K_MAX_BISECT:
            rc, klog = cargo_run(case, "keygen", k, env)
            if rc != 0:
                break
            last_keygen_k = k
            rc, plog = cargo_run(case, "prove", k, env)
            if rc != 0:
                break
            t = parse_prove_time(plog)
            if t is None:
                break
            k_probes.append([k, t])
            if t < best_t * 0.97:
                best_k, best_t = k, t
                best_log = klog + "\n" + plog
                k += 1
            else:
                break
    min_k = best_k
    # The on-disk pk/pinning belong to the last keygen run; restore best_k's.
    if last_keygen_k != best_k:
        rc, log = cargo_run(case, "keygen", best_k, env)
        if rc != 0:
            return CaseResult(case=case, keygen_ok=True, min_k=best_k,
                              k_probes=k_probes,
                              error=f"re-keygen at best k failed: {log[-400:]}")

    advice, fixed, rng = parse_cell_counts(best_log)
    # First prove time:
    prove_times: list[float] = []
    if best_t is not None:
        prove_times.append(best_t)
    for _ in range(repeats - 1):
        rc, log = cargo_run(case, "prove", min_k, env)
        if rc != 0:
            break
        t = parse_prove_time(log)
        if t is not None:
            prove_times.append(t)

    snark = HALO2_FOLDER / "data" / f"case{case}.snark"
    size = snark.stat().st_size if snark.exists() else None

    rc, vlog = cargo_run(case, "verify", min_k, env)
    verify_ok = (rc == 0)
    verify_t = parse_verify_time(vlog) if verify_ok else None

    return CaseResult(case=case, keygen_ok=True, min_k=min_k,
                      k_probes=k_probes,
                      prove_times_s=prove_times, snark_size_bytes=size,
                      verify_ok=verify_ok, verify_time_s=verify_t,
                      advice_cells=advice, fixed_cells=fixed,
                      range_lookup_cells=rng,
                      error=None if verify_ok else f"verify failed: {vlog[-200:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids; default = all installed builders")
    ap.add_argument("--k", type=int, default=None,
                    help="fixed k; if omitted, bisect upward from 10")
    ap.add_argument("--out", default=str(THIS_DIR / "hw_halo2_results.json"))
    args = ap.parse_args()

    if not HALO2_FOLDER.exists():
        print(f"halo2-graph not found at {HALO2_FOLDER}.")
        print("Clone with:  git clone <repo> /home/zhantong/halo2-graph")
        sys.exit(2)

    if args.cases:
        requested = [c.strip() for c in args.cases.split(",") if c.strip()]
        skipped = [c for c in requested if c in ABANDONED_BASELINE_CASES]
        for c in skipped:
            print(f"Skipping abandoned Halo2 baseline {c}: {ABANDONED_BASELINE_CASES[c]}")
        cases = [c for c in requested if c not in ABANDONED_BASELINE_CASES]
    else:
        # All cases that have BOTH a target.rs and a registered builder.
        cases = sorted(
            c for c in INPUT_BUILDERS
            if c not in ABANDONED_BASELINE_CASES
            if (THIS_DIR / "datasets" / "ds1000" / f"ds1000_case{c}/target.rs").exists()
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for i, c in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] case {c} ", end="", flush=True)
        t0 = time.time()
        r = run_case(c, args.repeats, args.k)
        wall = time.time() - t0
        results.append(r)
        if r.median_prove_s() is not None:
            print(f"k={r.min_k} prove={r.median_prove_s()*1000:.0f}ms "
                  f"snark={r.snark_size_bytes}B verify={r.verify_ok} "
                  f"cells(adv={r.advice_cells},fix={r.fixed_cells},rng={r.range_lookup_cells}) "
                  f"(wall {wall:.1f}s)")
        elif r.error:
            print(f"ERROR: {r.error[:120]} (wall {wall:.1f}s)")
        # Write incrementally.
        out_path.write_text(json.dumps(
            {"results": [asdict(x) for x in results]},
            indent=2,
        ))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

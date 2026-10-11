"""Scaling series of the memory family (port of eval-redo/layer/layer-memory-data/scale.py,
paths as arguments; proofs added for the compiler's own choice).

Five kernels at growing sizes, compiled with the layer's choice ("auto") and with each
element-read construction forced (ZINNIA_FORCE_CONSTRUCTION; the wrap stays the layer's
choice). Every point records halo2-base cells (advice + lookup at the k estimate_params
picks, and that k) and Noir bb gates. With --time, the "auto" points are also proved on
halo2-base: keygen, 3 proofs (median; wall = witness + create_proof, and create_proof
alone), each verified, a tampered public value rejected, outputs compared with NumPy.

Usage (cwd = a Zinnia checkout):
  .venv/bin/python scale_memory.py --out OUT.jsonl [--jobs 4] [--kernels gather,rows,...]
      [--forces auto,chain,...] [--time] [--threads 8] [--mem 30G]
  scale_memory.py --child KERNEL SIZE FORCE [--time]     (one JSON line)
"""
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCES = {
    "auto": "",
    "chain": "gather: element by a select chain",
    "indicators": "gather: element by indicators",
    "bits": "memory: element by index bits",
    "table": "memory: element by table read",
}
SIZES = {
    "gather": [2, 4, 8, 16, 32, 64, 128, 256],
    "rows": [4, 8, 16, 32, 64, 128],
    "take": [2, 3, 5, 8, 16, 32],
    "planes": [3, 6, 9, 12, 16],
    "permute": [4, 8, 16, 32],
}


def kernel(name, n):
    """(source, arguments) of kernel `name` at size `n` (same sources and seeds as the local run)."""
    import numpy as np
    rng = np.random.default_rng(n)
    if name == "gather":  # 16 reads of a vector of n at private indices
        src = f'def kernel(x: Float[Array, "{n}"], idx: Integer[Array, "16"]):\n    return x[idx]\n'
        return src, [rng.normal(size=n).round(3).tolist(), rng.integers(0, n, 16).tolist()]
    if name == "rows":  # one private column of a 16 x n matrix (16 reads at one index)
        src = f'def kernel(x: Float[Array, "16 {n}"], j: Integer):\n    return x[:, j]\n'
        return src, [rng.normal(size=(16, n)).round(3).tolist(), int(rng.integers(0, n))]
    if name == "take":  # np.take_along_axis over an axis of n (0503's shape: 13 x 13)
        src = (f'def kernel(a: Float[Array, "13 13 {n}"], b: Integer[Array, "13 13"]):\n'
               '    return np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]\n')
        return src, [rng.normal(size=(13, 13, n)).round(3).tolist(), rng.integers(0, n, (13, 13)).tolist()]
    if name == "planes":  # 0450's gather: a[:, s.reshape(-1, 1), t] over (7, n, n)
        src = (f'def kernel(a: Float[Array, "7 {n} {n}"], second: Integer[Array, "2"], third: Integer[Array, "2"]):\n'
               '    return a[:, np.array(second).reshape(-1, 1), third]\n')
        return src, [rng.normal(size=(7, n, n)).round(3).tolist(), rng.integers(0, n, 2).tolist(), rng.integers(0, n, 2).tolist()]
    if name == "permute":  # 0319's gather by an inverse permutation, n planes of 6 x 8
        src = ('@requires(lambda a, permutation: is_permutation(permutation))\n'
               f'def kernel(a: Float[Array, "{n} 6 8"], permutation: Integer[Array, "{n}"]):\n'
               '    c = np.empty_like(permutation)\n'
               '    c[permutation] = np.arange(len(permutation))\n'
               '    return a[c, :, :]\n')
        return src, [rng.normal(size=(n, 6, 8)).round(3).tolist(), rng.permutation(n).tolist()]
    raise ValueError(name)


def reference(name, args):
    """NumPy's result of kernel `name` on `args`."""
    import numpy as np
    if name == "gather":
        return np.array(args[0])[np.array(args[1])]
    if name == "rows":
        return np.array(args[0])[:, args[1]]
    if name == "take":
        return np.take_along_axis(np.array(args[0]), np.array(args[1])[..., np.newaxis], axis=-1)[..., 0]
    if name == "planes":
        return np.array(args[0])[:, np.array(args[1]).reshape(-1, 1), np.array(args[2])]
    perm = np.array(args[1])
    c = np.empty_like(perm)
    c[perm] = np.arange(len(perm))
    return np.array(args[0])[c, :, :]


def child(name, n, force, timed):
    os.environ["ZINNIA_FORCE_CONSTRUCTION"] = FORCES[force]
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "benchmarking"))
    import numpy as np
    from zinnia import ZKCircuit
    from zinnia.config.zinnia_config import ZinniaConfig
    from zinnia._zinnia_core import halo2_base_cell_stats
    import noir_gates
    src, args = kernel(name, n)
    out = {"kernel": name, "n": n, "force": force}
    halo2 = ZKCircuit.from_source("kernel", src, config=ZinniaConfig(backend="halo2")).compile()
    st = json.loads(halo2_base_cell_stats(halo2._compiled_ir))
    out.update(k=st["k"], cells=st["advice"] + st["lookup"], poly_cost=st.get("poly_cost"),
               second_advice=st.get("second_advice"), challenges=st.get("challenges"))
    noir = ZKCircuit.from_source("kernel", src, config=ZinniaConfig(backend="noir")).compile()
    nr = noir.emit_structural(*args, backend="noir")
    out.update(noir_gates.gate_count(nr, "noir_fpc", noir_gates.NOIR_FPC))
    if timed:
        from benchmarking.harness.adapters.zinnia_common import describe_compiled, zinnia_publics
        from benchmarking.harness.outputs import field_from_le_hex, field_to_le_hex
        started = time.perf_counter()
        keys = halo2.setup()
        out["setup_s"] = round(time.perf_counter() - started, 3)
        walls, proofs = [], []
        for _ in range(3):
            started = time.perf_counter()
            proofs.append(halo2.prove_with_keys(keys, *args))
            walls.append(time.perf_counter() - started)
        out["prove_wall_s"] = [round(s, 3) for s in walls]
        out["prove_median_s"] = round(statistics.median(walls), 3)
        out["create_proof_s"] = [round(p["prove_seconds"], 3) for p in proofs]
        out["create_proof_median_s"] = round(statistics.median(p["prove_seconds"] for p in proofs), 3)
        out["verified"] = all(halo2.verify_with_keys(keys, p) for p in proofs)
        tampered = dict(proofs[0])
        first = field_from_le_hex(tampered["public_values"][0])
        tampered["public_values"] = [field_to_le_hex(first + 1), *tampered["public_values"][1:]]
        out["tamper_rejected"] = not halo2.verify_with_keys(keys, tampered)
        fields = [field_from_le_hex(v) for v in proofs[0]["public_values"]]
        got = np.array(zinnia_publics(fields, describe_compiled(halo2)).values, dtype=float).reshape(-1)
        out["max_abs_error"] = float(np.max(np.abs(got - reference(name, args).reshape(-1))))
    print(json.dumps(out))


def opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    if sys.argv[1] == "--child":
        return child(sys.argv[2], int(sys.argv[3]), sys.argv[4], "--time" in sys.argv)
    out = opt("--out", None)
    jobs = int(opt("--jobs", 4))
    names = opt("--kernels", ",".join(SIZES)).split(",")
    forces = opt("--forces", ",".join(FORCES)).split(",")
    timed = "--time" in sys.argv
    threads, mem = opt("--threads", "8"), opt("--mem", "30G")
    env = dict(os.environ, ZWT=os.getcwd(), RAYON_NUM_THREADS=threads, OMP_NUM_THREADS=threads,
               HARDWARE_CONCURRENCY=threads)
    specs = [(k, n, f) for k in names for n in SIZES[k] for f in forces]

    def run(spec):
        cmd = [sys.executable, str(HERE / "capped.py"), mem, "3600", sys.executable, __file__, "--child",
               *map(str, spec)] + (["--time"] if timed and spec[2] == "auto" else [])
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        for line in reversed(p.stdout.splitlines()):
            if line.startswith("{"):
                rec = json.loads(line)
                break
        else:
            rec = {"kernel": spec[0], "n": spec[1], "force": spec[2], "error": (p.stderr or p.stdout)[-300:]}
        peaks = [l for l in p.stderr.splitlines() if l.startswith("capped: peak_rss_bytes=")]
        rec.update(threads=int(threads), memory_cap=mem, exit=p.returncode,
                   peak_rss_bytes=int(peaks[-1].split("=")[1]) if peaks else None)
        return rec

    with open(out, "w") as fh, ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(run, specs):
            print(json.dumps(r), flush=True)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


main()

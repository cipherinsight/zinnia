"""Bounded, offline execution in a filesystem and process namespace."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
from . import content

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGET = "riscv32im-risc0-zkvm-elf"


def sysroot(toolchain="stable"):
    return Path(subprocess.check_output(["rustc", f"+{toolchain}", "--print", "sysroot"], text=True).strip()).resolve()


def rlib(directory, name):
    matches = sorted(Path(directory).glob(f"lib{name}-*.rlib"))
    if len(matches) != 1:
        raise RuntimeError(f"Build the prepared Rust dependencies first: expected one {name} library in {directory}")
    return matches[0].name


def rust_wrapper(task):
    params = ", ".join(f'&read_array::<Ix{rank}>(&case["{name}"])' for name, rank in zip(content.names(task), content.RANKS[task]))
    prefix = '#![no_main]\nrisc0_zkvm::guest::entry!(main);\n'
    finish = 'risc0_zkvm::guest::env::commit_slice(text.as_bytes());'
    return prefix + '''use ndarray::{Array, Array1, Array2, Array3, Array4, Array5, ArrayD, Axis, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn, s, array};
use std::io::Read;
include!("submission.rs");
fn read_array<D: Dimension>(value: &serde_json::Value) -> Array<i64,D> {
    let mut shape = Vec::new(); let mut current = value;
    while let Some(items) = current.as_array() { shape.push(items.len()); current = &items[0]; }
    fn flatten(value: &serde_json::Value, out: &mut Vec<i64>) {
        if let Some(items) = value.as_array() { for item in items { flatten(item, out); } }
        else { out.push(value.as_i64().expect("integer input")); }
    }
    let mut values = Vec::new(); flatten(value, &mut values);
    ArrayD::from_shape_vec(IxDyn(&shape),values).unwrap().into_dimensionality::<D>().unwrap()
}
fn main() {
    let mut input = String::new(); std::io::stdin().read_to_string(&mut input).unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&input).unwrap();
    let mut outputs = Vec::new();
    for case in cases {
        let result = kernel(''' + params + ''');
        outputs.push(serde_json::json!({"shape": result.shape(), "values": result.iter().copied().collect::<Vec<i64>>() }));
    }
    let text = serde_json::to_string(&serde_json::json!({"status":"executed", "outputs":outputs})).unwrap();
    ''' + finish + '\n}\n'


class Runner:
    def __init__(self, runtime, rust_mode="risc0", timeout=90):
        if rust_mode != "risc0":
            raise ValueError("Study execution requires RISC Zero; native execution is disabled")
        Path(runtime).mkdir(parents=True, exist_ok=True, mode=0o700)
        Path(runtime).chmod(0o700)
        self.root = Path(runtime) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.rust_mode, self.timeout = rust_mode, timeout
        self.python_packages = REPO / ".venv/lib/python3.14/site-packages"
        self.guest_deps = HERE / f"rust-guest/target/{TARGET}/release/deps"
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="study-worker")
        self.capacity = threading.BoundedSemaphore(64)
        self.health_lock = threading.Lock()
        self.health = None
        self.health_at = 0

    def readiness(self):
        with self.health_lock:
            if self.health is None or time.monotonic() - self.health_at > 300:
                self.health = self._readiness()
                self.health_at = time.monotonic()
            return self.health.copy()

    def _readiness(self):
        try:
            test = subprocess.run(["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64", "--unshare-all", "--die-with-parent", "--proc", "/proc", "--dev", "/dev", "--", "/usr/bin/true"], capture_output=True, text=True, timeout=5)
            sandbox = test.returncode == 0
            detail = test.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            sandbox, detail = False, str(exc)
        status = {"sandbox": sandbox, "sandbox_detail": detail, "rust_mode": self.rust_mode}
        samples = {"zinnia": "def kernel(x, y):\n    return x + y\n",
                   "rust": "fn kernel(x: &Array1<i64>, y: &Array1<i64>) -> Array1<i64> { x + y }"}
        for condition, source in samples.items():
            result = self.execute(condition, "practice", source, content.cases("practice")) if sandbox else {}
            status[condition] = result.get("status") == "passed"
            if sandbox and not status[condition]:
                status["sandbox_detail"] += f" {condition}: {result.get('message', result.get('status'))}"
        return status

    def sandbox(self, work, command, mounts=(), stdin=None):
        # No shell, inherited credentials, home directories, network, or study data.
        cmd = ["prlimit", "--as=6442450944", "--cpu=80", "--nproc=4096", "--nofile=256", "--fsize=33554432", "--",
               "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
               "--clearenv", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--bind", str(work), "/work", "--chdir", "/work",
               "--setenv", "PATH", "/opt/rust/bin:/usr/bin", "--setenv", "LANG", "C.UTF-8",
               "--setenv", "PYTHONPATH", "/opt/python:/opt/zinnia", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
               "--setenv", "LD_LIBRARY_PATH", "/opt/python/z3/lib:/opt/rust/lib",
               "--setenv", "RISC0_SERVER_PATH", "/opt/r0vm", "--setenv", "RAYON_NUM_THREADS", "2",
               "--setenv", "OPENBLAS_NUM_THREADS", "1"]
        for source, target in mounts:
            cmd += ["--ro-bind", str(source), target]
        cmd += ["--", *command]
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True)
            try:
                proc.communicate(stdin, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise TimeoutError(f"Execution exceeded {self.timeout} seconds")
            out.seek(0); err.seek(0)
            return proc.returncode, out.read(16000).decode(errors="replace"), err.read(24000).decode(errors="replace")

    def execute(self, condition, task, source, cases):
        start = time.monotonic()
        try:
            expected = [content.answer(task, case) for case in cases]
            with tempfile.TemporaryDirectory(dir=self.root, prefix="run-") as folder:
                work = Path(folder)
                if condition == "zinnia":
                    (work / "submission.py").write_text(source)
                    (work / "request.json").write_text(json.dumps({"arguments": content.names(task), "cases": cases}))
                    mounts = [(self.python_packages, "/opt/python"), (REPO / "zinnia", "/opt/zinnia/zinnia"), (HERE / "worker.py", "/opt/worker.py")]
                    code, stdout, stderr = self.sandbox(work, ["/usr/bin/python3.14", "-S", "/opt/worker.py"], mounts)
                else:
                    deps = self.guest_deps
                    rust = sysroot("risc0")
                    mounts = [(rust, "/opt/rust"), (deps, "/opt/deps")]
                    (work / "submission.rs").write_text(source)
                    (work / "main.rs").write_text(rust_wrapper(task))
                    command = ["/opt/rust/bin/rustc", "--edition=2021", "--crate-name", "study_submission", "main.rs", "-o", "/work/program", "-C", "opt-level=1", "-L", "dependency=/opt/deps"]
                    libraries = ["ndarray", "serde_json", "risc0_zkvm"]
                    for library in libraries:
                        command += ["--extern", f"{library}=/opt/deps/{rlib(deps, library)}"]
                    mounts.append((HERE / "rust-guest/target/release/deps", "/opt/host-deps"))
                    command += ["-L", "dependency=/opt/host-deps"]
                    command += ["--target", TARGET, "-C", "passes=lower-atomic", "-C", "link-arg=-Ttext=0x00200800", "-C", "link-arg=--fatal-warnings", "-C", "panic=abort", "--cfg", 'getrandom_backend="custom"']
                    code, stdout, stderr = self.sandbox(work, command, mounts)
                    if code:
                        return {"status": "compile_error", "message": "Rust compilation failed", "stdout": stdout, "stderr": stderr, "seconds": time.monotonic()-start}
                    vm = Path(shutil.which("r0vm") or Path.home()/".cargo/bin/r0vm").resolve()
                    mounts += [(vm, "/opt/r0vm"), (HERE / "rust-host/target/release/study-risc0-host", "/opt/host")]
                    command = ["/opt/host", "/work/program"]
                    code, stdout, stderr = self.sandbox(work, command, mounts, json.dumps(cases).encode())
                if code:
                    return {"status": "runtime_error", "message": "Execution failed", "stdout": stdout, "stderr": stderr, "seconds": time.monotonic()-start}
                if not (work / "result.json").exists() or (work / "result.json").stat().st_size > 256000:
                    raise RuntimeError("Runner did not produce a bounded result")
                result = json.loads((work / "result.json").read_text())
                result["stdout"] = (result.get("stdout", "") + stdout)[-16000:]
                result["stderr"] = (result.get("stderr", "") + stderr)[-24000:]
                if result.get("status") == "executed":
                    outputs = result.pop("outputs")
                    if len(outputs) != len(cases):
                        raise RuntimeError("Program returned an unexpected number of results")
                    result["cases"] = []
                    for i, (actual, wanted) in enumerate(zip(outputs, expected)):
                        arr = np.asarray(actual["values"])
                        shape = actual["shape"]
                        valid = list(np.asarray(wanted).shape) == shape and arr.size <= 4096
                        passed = valid and np.array_equal(arr.reshape(shape), np.asarray(wanted))
                        result["cases"].append({"index": i+1, "passed": bool(passed), "input": cases[i], "expected": wanted, "actual": arr.reshape(shape).tolist() if valid else actual})
                    result["status"] = "passed" if all(c["passed"] for c in result["cases"]) else "failed"
                result.update(seconds=round(time.monotonic()-start, 3), engine="zinnia-ir-mock" if condition == "zinnia" else "risc0-zkvm")
                return result
        except TimeoutError as exc:
            return {"status": "timeout", "message": str(exc), "seconds": round(time.monotonic()-start, 3)}
        except Exception as exc:
            return {"status": "infrastructure_error", "message": str(exc), "seconds": round(time.monotonic()-start, 3)}

    def schedule(self, store, job, cases=None):
        if not job:
            return
        if not self.capacity.acquire(blocking=False):
            store.update_run(job["id"], "complete", {"status": "infrastructure_error", "message": "Execution queue is full. Please retry."})
            return
        def run():
            try:
                store.update_run(job["id"], "running")
                inputs = cases if cases is not None else job["inputs"]
                store.update_run(job["id"], "complete", self.execute(job["condition"], job["task"], job["code"], inputs))
            except Exception as exc:
                store.update_run(job["id"], "complete", {"status": "infrastructure_error", "message": str(exc)})
            finally:
                self.capacity.release()
        self.pool.submit(run)

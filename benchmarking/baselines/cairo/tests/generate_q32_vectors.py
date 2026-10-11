"""Regenerate exact Cairo expectations using the unchanged Rust Q32 library."""
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="cairo-q32-oracle-") as tmp:
    lib = Path(tmp) / "libq32.rlib"
    exe = Path(tmp) / "oracle"
    subprocess.run(["rustc", "--edition=2021", "--crate-name=q32", "--crate-type=rlib",
                    str(HERE.parents[1] / "zkvm_common/q32/src/lib.rs"), "-O", "-o", str(lib)], check=True)
    subprocess.run(["rustc", "--edition=2021", str(HERE / "q32_oracle.rs"),
                    "--extern", f"q32={lib}", "-O", "-o", str(exe)], check=True)
    generated = subprocess.check_output([str(exe)], text=True)
    target = HERE.parent / "src/q32/rust_vectors.cairo"
    target.parent.mkdir(exist_ok=True)
    target.write_text(generated)

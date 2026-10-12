"""Copy a crate's src/ with every bare `pub` turned into `pub(crate)`, so rustc's dead_code lint
sees the whole crate as private and reports items unreachable from the #[no_mangle] PyInit root.
Optionally drop modules / pyfunctions (by name) from lib.rs."""
import re, sys, shutil
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
drop_mods = [m for m in sys.argv[3].split(",") if m] if len(sys.argv) > 3 else []
drop_fns = [m for m in sys.argv[4].split(",") if m] if len(sys.argv) > 4 else []
if out.exists(): shutil.rmtree(out)
shutil.copytree(src, out)
for m in drop_mods:
    shutil.rmtree(out / m, ignore_errors=True); (out / f"{m}.rs").unlink(missing_ok=True)
for f in out.rglob("*.rs"):
    t = f.read_text()
    t = re.sub(r"\bpub\b(?!\s*\()", "pub(crate)", t)
    if f.name == "lib.rs" and f.parent == out:
        for m in drop_mods:
            t = re.sub(rf"pub\(crate\) mod {m};\n", "", t)
        for fn in drop_fns:
            t = re.sub(rf"[^\n]*wrap_pyfunction!\({fn}, m\)[^\n]*\n", "", t)
            # remove the item: from preceding attrs to the matching closing brace
            i = t.index(f"fn {fn}(")
            s = t.rfind("\n\n", 0, i) + 2
            depth, j = 0, t.index("{", t.index(")", i))
            # find body start: first '{' after the signature's '->' or ')' at depth
            k = j
            while True:
                c = t[k]
                if c == "{": depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0: break
                k += 1
            t = t[:s] + t[k+1:]
    f.write_text(t)

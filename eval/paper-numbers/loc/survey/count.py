"""Code lines of Rust text: lines with something other than whitespace once comments are removed
(strings kept). Same definition as tokei's "code"."""
import sys, re
from pathlib import Path
sys.argv_saved = sys.argv; sys.argv = ["x", "/dev/null", "/dev/null"]
import importlib.util
spec = importlib.util.spec_from_file_location("st", "/home/zhantong/Projects/zinnia-project/zinnia-src/eval/paper-numbers/loc/strip_tests.py")
st = importlib.util.module_from_spec(spec)
src = Path(spec.origin).read_text().split("\ncompiled, test_files, removed = walk()")[0]
exec(compile(src, "st", "exec"), st.__dict__)
sys.argv = sys.argv_saved

def code_mask(t):
    """Per line: True when the line has code (comments blanked, literals kept)."""
    out, i, n = [], 0, len(t)
    while i < n:
        j = st.skip_trivia(t, i)
        if j != i:
            out.append(re.sub(r"[^\n]", " ", t[i:j])); i = j; continue
        k = st.skip_literal(t, i)
        if k is not None:
            out.append(re.sub(r"[^\n]", "x", t[i:k])); i = k; continue
        out.append(t[i]); i += 1
    return [bool(l.strip()) for l in "".join(out).split("\n")]

def count(t): return sum(code_mask(t))
if __name__ == "__main__":
    tot = 0
    for f in Path(sys.argv[1]).rglob("*.rs"):
        if any(p in f.parts for p in sys.argv[2:]): continue
        tot += count(f.read_text())
    print(tot)

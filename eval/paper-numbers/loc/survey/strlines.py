# code lines that lie wholly inside a multi-line string literal (embedded text such as Noir templates)
import sys, re
from pathlib import Path
from collections import Counter
from count import st
c = Counter()
for f in Path(sys.argv[1]).rglob("*.rs"):
    if "phase0" in f.parts or "phase1" in f.parts: continue
    t = f.read_text(); i = 0; n = len(t)
    while i < n:
        j = st.skip_trivia(t, i)
        if j != i: i = j; continue
        k = st.skip_literal(t, i)
        if k is not None:
            s = t[i:k]
            if s.count("\n") >= 1: c[str(f.relative_to(sys.argv[1]))] += s.count("\n") - 0
            i = k; continue
        i += 1
print(sum(c.values())); [print(v, k) for k, v in c.most_common(8)]

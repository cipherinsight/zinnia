"""Code lines of the items rustc's dead_code lint reports (from cargo --message-format=json).
Each reported item counts from its first attribute/doc line to its end; a field or variant counts
its own lines; inherent and trait impls whose self type is a dead struct/enum count too."""
import json, re, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from count import st, code_mask
msgs, crate = sys.argv[1], Path(sys.argv[2])
ranges = defaultdict(list)   # file -> [(start_line, end_line)] 0-based inclusive
dead_types = defaultdict(set)
kinds = defaultdict(int)
def extent(text, lines, L):
    off = sum(len(l) + 1 for l in lines[:L])
    i = off + (len(lines[L]) - len(lines[L].lstrip()))
    e = st.item_end(text, i)
    end = text.count("\n", 0, e)
    s = L
    while s > 0 and re.match(r"\s*(#\[|///|//!)", lines[s - 1]): s -= 1
    return s, end
for l in open(msgs):
    m = json.loads(l)
    if m.get("reason") != "compiler-message": continue
    msg = m["message"]
    if (msg.get("code") or {}).get("code") != "dead_code": continue
    kind = msg["message"].split()[0]
    for sp in msg["spans"]:
        if not sp["is_primary"]: continue
        f = sp["file_name"]; L = sp["line_start"] - 1
        text = (crate / f).read_text(); lines = text.split("\n")
        try:
            s, e = extent(text, lines, L)
        except Exception as ex:
            s, e = L, L
        ranges[f].append((s, e))
        kinds[kind] += 1
        if kind in ("struct", "enum"):
            dead_types[f].add(re.search(r"\x60(\w+)\x60", msg["message"]).group(1))
# impls of dead types (anywhere in the crate)
names = {n for v in dead_types.values() for n in v}
for f in crate.rglob("src/**/*.rs"):
    text = f.read_text(); lines = text.split("\n")
    for k, line in enumerate(lines):
        m = re.match(r"\s*impl\b(<[^>]*>)?\s*(?:[\w:<>, ]+\s+for\s+)?(\w+)\b", line)
        if m and m.group(2) in names:
            try:
                s, e = extent(text, lines, k)
                ranges[str(f.relative_to(crate))].append((s, e)); kinds["impl-of-dead-type"] += 1
            except Exception: pass
total = 0; per = {}
for f, rs in ranges.items():
    mask = code_mask((crate / f).read_text())
    hit = set()
    for s, e in rs: hit.update(range(s, e + 1))
    n = sum(1 for i in hit if i < len(mask) and mask[i]); per[f] = n; total += n
print("dead code lines:", total, dict(kinds))
for f, n in sorted(per.items(), key=lambda x: -x[1]): print(n, f)

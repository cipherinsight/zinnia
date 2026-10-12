"""Copy the compiler's Rust sources without their test code, for counting lines of code.

Usage: python strip_tests.py <src dir> <out dir>

Test code is what `cargo build` (no `--test`) never compiles:
  * a file whose module is declared only under a `#[cfg(test)]` attribute
    (`#[cfg(test)] mod tests;`), together with every module below it;
  * an item inside a compiled file that carries a `#[cfg(test)]` attribute
    (usually `#[cfg(test)] mod tests { ... }`, also test-only functions or `use` lines).
A cfg attribute counts as test-only when it requires `test` (`cfg(test)`, `cfg(all(test, ...))`),
not when it negates it (`cfg(not(test))`).
The module tree is walked from src/lib.rs; files it never reaches are reported and copied
unchanged (they are not compiled either, so they are listed separately).
"""
import re
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()


def skip_trivia(t, i):
    """Index of the first character at or after i that is not whitespace or a comment."""
    n = len(t)
    while i < n:
        if t[i].isspace():
            i += 1
        elif t.startswith("//", i):
            j = t.find("\n", i)
            i = n if j < 0 else j + 1
        elif t.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if t.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif t.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        else:
            break
    return i


def skip_literal(t, i):
    """If a string or char literal starts at i, the index after it; otherwise None."""
    m = re.match(r'b?r(#*)"', t[i:])
    if m:
        end = t.find('"' + m.group(1), i + m.end())
        return end + 1 + len(m.group(1))
    m = re.match(r'b?"', t[i:])
    if m:
        j = i + m.end()
        while t[j] != '"':
            j += 2 if t[j] == "\\" else 1
        return j + 1
    m = re.match(r"b?'(\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'])'", t[i:])
    if m:  # a char literal, not a lifetime
        return i + m.end()
    return None


ITEM = re.compile(r"(pub(\([^)]*\))?\s+)?((const|async|unsafe|extern(\s+\"[^\"]*\")?)\s+)*"
                  r"(fn|mod|impl|struct|enum|union|trait|use|const|static|type|macro_rules!)\b")


def item_end(t, i):
    """End of the item starting at i: its matching closing brace, or the `;` at depth 0.
    A field, variant or match arm (not a declaration item) also ends at a `,` at depth 0."""
    depth, n = 0, len(t)
    comma_ends = not ITEM.match(t, i)
    while i < n:
        j = skip_trivia(t, i)
        if j != i:
            i = j
            continue
        if (k := skip_literal(t, i)) is not None:
            i = k
            continue
        c = t[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:  # closer of the enclosing construct (field, variant, arm)
                return i
            depth -= 1
            if depth == 0 and c == "}":
                if (k := skip_trivia(t, i + 1)) < n and t[k] == ";":  # `use a::{..};`, `const X = S {..};`
                    return k + 1
                return i + 1
        elif depth == 0 and (c == ";" or (c == "," and comma_ends)):
            return i + 1
        i += 1
    raise ValueError("unterminated item")


ATTR = re.compile(r"#\[")
TEST_CFG = re.compile(r"#\[\s*cfg\s*\((?!\s*not\b)[^\]]*\btest\b")


def attribute_end(t, i):
    """End of the attribute `#[...]` starting at i."""
    depth = 0
    while True:
        if (k := skip_literal(t, i)) is not None:
            i = k
            continue
        if t[i] == "[":
            depth += 1
        elif t[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def scan(t):
    """Yield (start, end, attrs, item_text) for every attributed item at any nesting level,
    outermost first, skipping comments and literals."""
    i, n = 0, len(t)
    while i < n:
        j = skip_trivia(t, i)
        if j != i:
            i = j
            continue
        if (k := skip_literal(t, i)) is not None:
            i = k
            continue
        if t.startswith("#[", i):
            start, attrs = i, []
            while t.startswith("#[", i):
                e = attribute_end(t, i)
                attrs.append(t[i:e])
                i = skip_trivia(t, e)
            e = item_end(t, i)
            yield start, e, attrs, t[i:e]
            continue
        i += 1


def strip(text):
    """Remove test-only items; return (stripped text, test-only `mod name;` declarations)."""
    out, last, test_mods, plain_mods = [], 0, [], []
    for start, end, attrs, item in scan(text):
        if start < last:
            continue  # nested inside a removed item
        if any(TEST_CFG.match(a) for a in attrs):
            m = re.match(r"(pub(\([^)]*\))?\s+)?mod\s+(\w+)\s*;", item)
            if m:
                test_mods.append((m.group(3), path_attr(attrs)))
            # The item's doc comment (`///` lines right above it) goes with it.
            ls = text.rfind("\n", 0, start) + 1
            if ls >= last and not text[ls:start].strip():
                start = ls
                while start > last:
                    prev = text.rfind("\n", 0, start - 1) + 1
                    if prev >= last and re.fullmatch(r"[ \t]*///[^\n]*\n", text[prev:start]):
                        start = prev
                    else:
                        break
            out.append(text[last:start])
            last = end
    out.append(text[last:])
    return "".join(out), test_mods


def path_attr(attrs):
    for a in attrs:
        m = re.match(r'#\[\s*path\s*=\s*"([^"]+)"', a)
        if m:
            return m.group(1)
    return None


def code_only(t):
    """t with comments and string/char literals blanked (newlines kept); attributes' path
    strings are kept, since `#[path = "..."]` is read from them."""
    out, i, n = [], 0, len(t)
    while i < n:
        j = skip_trivia(t, i)
        if j != i:
            out.append(re.sub(r"[^\n]", " ", t[i:j]))
            i = j
            continue
        if (k := skip_literal(t, i)) is not None:
            keep = re.search(r"#\[\s*path\s*=\s*$", t[max(0, i - 40):i])
            out.append(t[i:k] if keep else re.sub(r"[^\n]", " ", t[i:k]))
            i = k
            continue
        out.append(t[i])
        i += 1
    return "".join(out)


def mod_decls(text):
    """All `mod name;` declarations of a (stripped) file, with their #[path] if any."""
    decls = []
    text = code_only(text)
    for m in re.finditer(r"((?:#\[[^\]]*\]\s*)*)(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", text):
        decls.append((m.group(2), path_attr(re.findall(r"#\[[^\]]*\]", m.group(1)))))
    return decls


def child_file(parent, name, path):
    """Rust 2018 module file resolution."""
    if parent.name in ("lib.rs", "mod.rs", "main.rs"):
        base = parent.parent
    else:
        base = parent.parent / parent.stem
    if path:
        return (parent.parent / path).resolve()
    for cand in (base / f"{name}.rs", base / name / "mod.rs"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{parent}: mod {name}")


def walk():
    compiled, test_files, removed_lines = {}, set(), {}
    todo = [SRC / "lib.rs"]
    while todo:
        f = todo.pop()
        text = f.read_text()
        if re.match(r"\s*(//[^\n]*\n\s*)*#!\[\s*cfg\s*\(\s*test", text):  # whole file is test-only
            test_files.add(f)
            continue
        stripped, test_mods = strip(text)
        compiled[f] = stripped
        removed_lines[f] = text.count("\n") - stripped.count("\n")
        for name, path in mod_decls(stripped):
            todo.append(child_file(f, name, path))
        for name, path in test_mods:
            stack = [child_file(f, name, path)]
            while stack:  # the whole subtree of a test-only module
                g = stack.pop()
                test_files.add(g)
                stack += [child_file(g, n, p) for n, p in mod_decls(g.read_text())]
    return compiled, test_files, removed_lines


compiled, test_files, removed = walk()
every = set(SRC.rglob("*.rs"))
orphans = every - set(compiled) - test_files
for f, text in compiled.items():
    dest = OUT / f.relative_to(SRC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
for f in orphans:
    dest = OUT / f.relative_to(SRC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f.read_text())
print(f"compiled files: {len(compiled)}; test-only files dropped: {len(test_files)}; "
      f"files with inline test items stripped: {sum(1 for v in removed.values() if v)} "
      f"({sum(removed.values())} physical lines); unreachable files (copied): {len(orphans)}")
for f in sorted(orphans):
    print("  unreachable:", f.relative_to(SRC))
(OUT.parent / (OUT.name + ".test-files.txt")).write_text(
    "".join(f"{f.relative_to(SRC)}\n" for f in sorted(test_files)))

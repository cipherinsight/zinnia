"""`ops.py` names the same operators as `ops.rs` (`acceptance.md` §2.1).

`ops.py`'s docstring claims to mirror `src/phase0/ops.rs` table for table.
That claim is only checkable if something compares the two name sets: a name
added on one side and not the other is otherwise found the first time a
circuit uses it, as an unexplained harness gap.

The Rust side is read out of the source: every table is a `match member`
over string literals, so the literals of one function's body *are* its
names. That is a text scan, but the alternative — a generated JSON — is a
third copy that can go stale on its own.

```sh
.venv/bin/python -m pytest testing/phase0/test_ops_mirror.py -q
```
"""

from __future__ import annotations

import re

from testing.phase0 import REPO_ROOT
from testing.phase0 import ops

OPS_RS = REPO_ROOT / "src" / "phase0" / "ops.rs"

#: `fn <name>(member: &str)` opens a table; the next one closes it.
_FUNCTION = re.compile(r"^(?:pub )?fn (\w+)\(member: &str\)", re.MULTILINE)
_LITERAL = re.compile(r'"([^"]+)"')


def rust_tables() -> dict[str, set[str]]:
    """The string literals of each `match member` table of `ops.rs`."""
    text = OPS_RS.read_text(encoding="utf-8")
    bounds = [(m.group(1), m.start()) for m in _FUNCTION.finditer(text)]
    tables: dict[str, set[str]] = {}
    for index, (name, start) in enumerate(bounds):
        end = bounds[index + 1][1] if index + 1 < len(bounds) else len(text)
        tables[name] = set(_LITERAL.findall(text[start:end]))
    return tables


def numpy_names() -> set[str]:
    """Every `np.*` name `ops.rs` dispatches, the dtype objects included."""
    tables = rust_tables()
    return (
        tables["numpy_elementwise"]
        | tables["numpy_shape"]
        | tables["numpy_other"]
        | tables["dtype_object"]
    )


def test_the_numpy_tables_name_the_same_operators() -> None:
    rust = numpy_names()
    python = set(ops.NUMPY) | set(ops.DTYPE_OBJECTS)
    assert rust == python, (
        f"only in ops.rs: {sorted(rust - python)}; "
        f"only in ops.py: {sorted(python - rust)}"
    )


def test_the_math_tables_name_the_same_operators() -> None:
    rust = rust_tables()["math"]
    assert rust == set(ops.MATH), (
        f"only in ops.rs: {sorted(rust - set(ops.MATH))}; "
        f"only in ops.py: {sorted(set(ops.MATH) - rust)}"
    )


def test_the_builtin_tables_name_the_same_operators() -> None:
    rust = rust_tables()["builtin"]
    assert rust == set(ops.BUILTINS), (
        f"only in ops.rs: {sorted(rust - set(ops.BUILTINS))}; "
        f"only in ops.py: {sorted(set(ops.BUILTINS) - rust)}"
    )


def test_the_list_method_tables_name_the_same_operators() -> None:
    rust = rust_tables()["list_method"]
    assert rust == set(ops.LIST_METHODS), (
        f"only in ops.rs: {sorted(rust - set(ops.LIST_METHODS))}; "
        f"only in ops.py: {sorted(set(ops.LIST_METHODS) - rust)}"
    )

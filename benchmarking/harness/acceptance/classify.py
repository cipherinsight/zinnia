"""Reason classes for programs the compiler rejects, with the construct named.

Every rejection gets a class and a short name of the construct or size that
caused it. The classes are the ones the support account reports:

- `out_of_scope`: outside the stated scope (imports other than numpy and
  math), decided before compiling by `kernels.out_of_scope_imports`, or
  reported so by the compiler;
- `unsupported_construct`: the program uses a language feature or library
  function Zinnia does not implement in any program (a `dict`, `is`, a
  function used as a value, `np.gcd`, ...);
- `no_provable_bound`: a size the circuit needs fixed (a loop's trip count, an
  array's length, a `while` loop's termination) depends on a private input
  and the compiler finds no maximum for it. Circuits have a fixed size, so
  the program needs a bound (an annotation or a `requires` precondition);
- `compiler_limitation`: Zinnia implements the construct, but not in the
  form or combination this program uses (an operation on an array whose
  shape is only known at runtime, an operator precondition it can neither
  prove nor check in the circuit, ...);
- `solver_budget`: rejected for `no_provable_bound` at the default SMT
  budget and accepted with a larger one (decided by re-running, `validate`);
- `resource_limit`: the compile exceeded the time or memory limit (decided by
  the runner, see `resource_rejection`);
- `compiler_error`: an internal error (panic, failed invariant, crash). Each
  is a compiler bug to fix, never a property of the program;
- `unclassified`: no rule matched; the raw message is kept so the rules can
  be refined. Never folded into another class.

The rules match the compiler's own message vocabulary, never program names;
the first matching rule wins, so specific rules come before generic ones and
internal errors before construct names (an internal error can mention an
operation the compiler otherwise supports).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OUT_OF_SCOPE = "out_of_scope"
UNSUPPORTED = "unsupported_construct"
NO_BOUND = "no_provable_bound"
LIMITATION = "compiler_limitation"
RESOURCE = "resource_limit"
INTERNAL = "compiler_error"
UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Rejection:
    """Why a program is rejected: the class and the construct it names."""

    reason_class: str
    construct: str


@dataclass(frozen=True)
class Rule:
    reason_class: str
    pattern: re.Pattern
    #: The construct's name; `{1}`, `{2}`, ... are the pattern's groups.
    construct: str

    def match(self, message: str) -> Rejection | None:
        found = self.pattern.search(message)
        if found is None:
            return None
        groups = [found.group(0), *(g or "" for g in found.groups())]
        return Rejection(self.reason_class, self.construct.format(*groups))


def _rule(reason_class: str, pattern: str, construct: str) -> Rule:
    return Rule(reason_class, re.compile(pattern, re.I | re.S), construct)


RULES: tuple[Rule, ...] = (
    _rule(OUT_OF_SCOPE, r"Only supported NumPy imports are allowed",
          "import of a module other than numpy and math"),

    # Sizes without a provable maximum.
    _rule(NO_BOUND, r"no provable maximum: the `while` condition was not proven false",
          "while-loop trip count"),
    _rule(NO_BOUND, r"no provable maximum: the proven bound on the number of iterations",
          "range() trip count (proven bound too large to unroll)"),
    _rule(NO_BOUND, r"no provable maximum: cannot bound the number of iterations of range\(\)",
          "range() trip count"),
    _rule(NO_BOUND, r"shape element at axis \d+ must be a compile-time constant",
          "array shape from a runtime value"),
    _rule(NO_BOUND, r"range (?:start|stop|step) must be a compile-time constant",
          "np.arange length from a runtime value"),
    _rule(NO_BOUND, r"(\w+) `?num`? must be a compile-time constant", "np.{1} num from a runtime value"),
    _rule(NO_BOUND, r"split sections must be a compile-time constant",
          "np.split sections from a runtime value"),
    _rule(NO_BOUND, r"reshape target dimension must be a compile-time constant",
          "reshape target from a runtime value"),
    _rule(NO_BOUND, r"slice (?:start|stop|step) must be a compile-time constant",
          "slice bound from a runtime value"),
    _rule(NO_BOUND, r"(np\.\w+): pad width is not statically bounded", "{1} width from a runtime value"),
    _rule(NO_BOUND, r"(np\.\w+): x entry value is not statically bounded",
          "{1} output length from runtime values"),
    _rule(NO_BOUND, r"sequence repetition `seq \* n` with a runtime count",
          "list repetition with a runtime count"),
    _rule(NO_BOUND, r"could not be proved in-bounds|must be resolved to an exact value or finite "
                    r"bound|expected statically bounded",
          "size or index without a provable bound"),

    # Forms of supported constructs the compiler does not handle yet.
    _rule(LIMITATION, r"op `(\w+)` requires precondition .*could not be lowered",
          "precondition of `{1}` neither proved nor checkable in the circuit"),
    _rule(LIMITATION, r"op `(\w+)` requires precondition",
          "precondition of `{1}` not provable"),
    _rule(LIMITATION, r"reshape is not supported on an array with a runtime shape",
          "reshape of a runtime-shaped array"),
    _rule(LIMITATION, r"this operation is not supported on an array with a runtime shape",
          "operation on a runtime-shaped array"),
    _rule(LIMITATION, r"(\w+): DynamicNDArray input is deferred", "np.{1} of a runtime-shaped array"),
    _rule(LIMITATION, r"list\(\) of an array with a runtime shape", "list() of a runtime-shaped array"),
    _rule(LIMITATION, r"(?:DynamicNDArray )?([\w.]+) is not yet supported for bounded-dynamic arrays",
          "{1} of a runtime-shaped array"),
    _rule(LIMITATION, r"(np\.\w+): unsupported operand type DynamicNDArray",
          "{1} of a runtime-shaped array"),
    _rule(LIMITATION, r"histogram: first slice supports compile-time-data histogram only",
          "np.histogram of runtime data"),
    _rule(LIMITATION, r"array-valued indices .*require compile-time-constant index values",
          "index array with runtime values"),
    _rule(LIMITATION, r"(np\.\w+): only static integer obj values", "{1} at a runtime position"),
    _rule(LIMITATION, r"(np\.\w+): this bounded slice supports one scalar obj",
          "{1} at several positions"),
    _rule(LIMITATION, r"filtered comprehensions currently require scalar elements",
          "filtered comprehension over arrays"),
    _rule(LIMITATION, r"Chip return inference does not support partial returns",
          "chip that returns on some paths only"),
    _rule(LIMITATION, r"range\(\) with a runtime argument is only supported as the iterable",
          "range() with a runtime bound used as a value"),
    _rule(LIMITATION, r"slice assignment into a complex array with a runtime index",
          "complex slice write with a runtime bound"),
    _rule(LIMITATION, r"broadcasting into a slice with runtime bounds",
          "broadcast into a slice with a runtime bound"),
    _rule(LIMITATION, r"concatenate/stack: unsupported element type",
          "np.concatenate/np.stack of nested lists"),
    _rule(LIMITATION, r"Reading `\w+` is not supported here: a name that refers to this array on "
                      r"some paths and to another array on others",
          "array name bound to different arrays on runtime paths, then written"),
    _rule(LIMITATION, r"(np\.linalg\.\w+): keyword arguments are not supported",
          "{1} with keyword arguments"),
    _rule(LIMITATION, r"(np\.linalg\.\w+): implicit result-returning linear algebra is deferred",
          "{1} (result-returning linear algebra)"),
    _rule(LIMITATION, r"RecursionLimitExceededError: recursion limit \((\d+)\) exceeded",
          "recursion deeper than the inlining limit ({1})"),
    _rule(LIMITATION, r"of a complex array with a runtime shape is not supported",
          "complex array with a runtime shape"),

    # NumPy's shape errors: the datasets' programs run under NumPy, so the
    # compiler computed a shape NumPy does not.
    _rule(INTERNAL, r"operands could not be broadcast together with shapes",
          "broadcast error on shapes NumPy accepts"),

    # Internal errors: compiler bugs, whatever operation they mention.
    _rule(INTERNAL, r"panicked|index out of bounds: the len|capacity overflow|IRBuilder::"
                    r"|InternalZinniaException|compiler process exited|pop index out of range"
                    r"|Variable `[^`]*` not found|Cannot promote value of type"
                    r"|assertion `left == right` failed|called `Option::unwrap\(\)`"
                    r"|Type mismatch: expected",
          "internal error"),

    # Constructs Zinnia does not implement.
    _rule(UNSUPPORTED, r"(np\.linalg\.\w+) is not implemented", "{1}"),
    _rule(UNSUPPORTED, r"(np\.\w+(?:\.\w+)?) is not implemented", "{1}"),
    _rule(UNSUPPORTED, r"Set literals are not supported", "set"),
    _rule(UNSUPPORTED, r"Dict literals are not supported", "dict"),
    _rule(UNSUPPORTED, r"`is` / `is not` compare object identity", "`is` / `is not`"),
    _rule(UNSUPPORTED, r"Unsupported assignment target `(\w+)`", "assignment to an {1}"),
    _rule(UNSUPPORTED, r"functions as values are not supported", "function used as a value"),
    _rule(UNSUPPORTED, r"mutating this list through `\w+` is not supported",
          "list mutated through a second name (aliasing)"),
    _rule(UNSUPPORTED, r"Named attribute `numpy\.(\w+)` not yet implemented",
          "numpy.{1} (module imported as `numpy`)"),
    _rule(UNSUPPORTED, r"Unknown function `(\w+)`", "call of `{1}` (no builtin, chip or numpy "
                                                     "function of that name in scope)"),
    _rule(UNSUPPORTED, r"Circuit input must be annotated", "unannotated entry parameter"),
    _rule(UNSUPPORTED, r"Lambda|lambda expressions are not supported", "lambda"),
    _rule(UNSUPPORTED, r"strings? (?:are|is) not supported|String values are not supported",
          "string"),
)


def classify_compile_error(message: str) -> Rejection:
    """The class and construct of a compile-time rejection."""
    for rule in RULES:
        rejection = rule.match(message)
        if rejection is not None:
            return rejection
    return Rejection(UNCLASSIFIED, "")


def resource_rejection(detail: str, timeout_s: float, memory_max: str | None) -> Rejection:
    """A compile the runner stopped (`timeout after Ns` / `out of memory after Ns`)."""
    if detail.startswith("timeout"):
        return Rejection(RESOURCE, f"compile time over {timeout_s:.0f} s")
    return Rejection(RESOURCE, f"compile memory over {memory_max or 'the limit'}")

//! Rendering a term: the pretty form, the SMT-LIB 2 text and the JSON DAG.
//!
//! Three printers over the same arena, kept together because they answer the
//! same question — what a consumer outside phase 0 sees of a term — and apart
//! from [`super`] because that module is the term language itself
//! (`definition.md` §3.1) and this one is only how it is written down.
//!
//! | Printer | For | Cost |
//! |---|---|---|
//! | [`TermArena::pretty`] | humans, tests, the `*_pretty` fields of the export | the *tree*, cut off at [`PRETTY_BUDGET`] |
//! | [`TermArena::to_smtlib`] | the consistency harness (`acceptance.md` §3) | linear: one `define-fun` per node |
//! | [`TermArena::to_json`] | the Python harnesses, which rebuild the DAG | linear: one object per node |

use super::{BoolId, Node, TermArena, TermId};

// ---------------------------------------------------------------------------
// Pretty printer
// ---------------------------------------------------------------------------

/// How many characters a rendered term may take before it is cut short.
///
/// The pretty form is a *tree*: a shared sub-term is printed once per
/// occurrence. That is what makes it readable, and also what makes it
/// exponential in a DAG that shares heavily — the accumulator of an unrolled
/// loop, whose `k`-th value mentions the `(k-1)`-th twice. The JSON export
/// prints one of these per descriptor, so the budget is what keeps the record
/// of a large program a record rather than an out-of-memory. Nothing a
/// consumer computes with is lost: [`TermArena::to_json`] exports the DAG
/// itself and [`TermArena::to_smtlib`] names every node once.
const PRETTY_BUDGET: usize = 4096;

/// One step of the pretty printer: a node still to render, or text to append
/// once the nodes before it are done.
enum Render {
    Node(u32),
    Text(&'static str),
    Owned(String),
}

/// `(<a> <op> <b>)`. The open parenthesis is written now; the rest is queued.
fn queue_infix(out: &mut String, stack: &mut Vec<Render>, a: u32, op: &'static str, b: u32) {
    out.push('(');
    stack.push(Render::Text(")"));
    stack.push(Render::Node(b));
    stack.push(Render::Text(op));
    stack.push(Render::Node(a));
}

/// `<name>(<a>, <b>)`, with `name` carrying its own open parenthesis.
fn queue_call(out: &mut String, stack: &mut Vec<Render>, name: &'static str, a: u32, b: u32) {
    out.push_str(name);
    stack.push(Render::Text(")"));
    stack.push(Render::Node(b));
    stack.push(Render::Text(", "));
    stack.push(Render::Node(a));
}

impl TermArena {
    /// Render the tree below `root` into `out`, stopping at
    /// [`PRETTY_BUDGET`] characters.
    ///
    /// Iterative rather than recursive: the DAG below an unrolled loop is as
    /// deep as the loop is long, and phase 0 must not overflow the stack on a
    /// program it has just accepted.
    fn render(&self, root: u32, out: &mut String) {
        let mut stack = vec![Render::Node(root)];
        while let Some(step) = stack.pop() {
            if out.len() >= PRETTY_BUDGET {
                out.push('…');
                return;
            }
            match step {
                Render::Text(text) => out.push_str(text),
                Render::Owned(text) => out.push_str(&text),
                Render::Node(i) => self.render_node(i, out, &mut stack),
            }
        }
    }

    /// Write one node's own text and queue the sub-terms it contains.
    fn render_node(&self, i: u32, out: &mut String, stack: &mut Vec<Render>) {
        match self.nodes[i as usize] {
            Node::Const(n) => out.push_str(&n.to_string()),
            Node::Var(v) | Node::BVar(v) => out.push_str(self.var_name(v)),
            Node::True => out.push_str("true"),
            Node::False => out.push_str("false"),
            Node::Add(a, b) => queue_infix(out, stack, a.0, " + ", b.0),
            Node::Sub(a, b) => queue_infix(out, stack, a.0, " - ", b.0),
            Node::Eq(a, b) => queue_infix(out, stack, a.0, " = ", b.0),
            Node::Lt(a, b) => queue_infix(out, stack, a.0, " < ", b.0),
            Node::Le(a, b) => queue_infix(out, stack, a.0, " <= ", b.0),
            Node::And(a, b) => queue_infix(out, stack, a.0, " && ", b.0),
            Node::Or(a, b) => queue_infix(out, stack, a.0, " || ", b.0),
            Node::Min(a, b) => queue_call(out, stack, "min(", a.0, b.0),
            Node::Max(a, b) => queue_call(out, stack, "max(", a.0, b.0),
            Node::Mul(k, t) => {
                out.push_str(&format!("({} * ", k));
                stack.push(Render::Text(")"));
                stack.push(Render::Node(t.0));
            }
            Node::Div(t, k) => {
                out.push('(');
                stack.push(Render::Owned(format!(" div {})", k)));
                stack.push(Render::Node(t.0));
            }
            Node::Mod(t, k) => {
                out.push('(');
                stack.push(Render::Owned(format!(" mod {})", k)));
                stack.push(Render::Node(t.0));
            }
            Node::Not(b) => {
                out.push('!');
                stack.push(Render::Node(b.0));
            }
            Node::Ite(c, a, b) => {
                out.push_str("ite(");
                stack.push(Render::Text(")"));
                stack.push(Render::Node(b.0));
                stack.push(Render::Text(", "));
                stack.push(Render::Node(a.0));
                stack.push(Render::Text(", "));
                stack.push(Render::Node(c.0));
            }
        }
    }

    /// Stable rendering of an integer term. Shared sub-terms are printed
    /// once per occurrence, so this is for humans, tests and the JSON
    /// `*_pretty` fields; [`TermArena::to_json`] carries the DAG itself. A
    /// rendering past [`PRETTY_BUDGET`] characters ends in `…`.
    pub fn pretty(&self, t: TermId) -> String {
        let mut out = String::new();
        self.render(t.0, &mut out);
        out
    }

    /// Stable rendering of a Boolean term.
    pub fn pretty_bool(&self, b: BoolId) -> String {
        let mut out = String::new();
        self.render(b.0, &mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// SMT-LIB 2 printer
// ---------------------------------------------------------------------------

impl TermArena {
    /// SMT-LIB 2 text: one `declare-const` per variable, one `define-fun`
    /// per arena node, then one `assert` per formula in `asserts`.
    ///
    /// Naming the nodes keeps the output linear in the DAG instead of the
    /// tree, which matters once loops are unrolled.
    pub fn to_smtlib(&self, asserts: &[(BoolId, String)]) -> String {
        let mut out = String::new();
        out.push_str("(set-logic ALL)\n");
        for (i, decl) in self.vars.iter().enumerate() {
            out.push_str(&format!(
                "(declare-const {} {}) ; var {}, sort {}\n",
                decl.name,
                decl.sort.smt(),
                i,
                decl.sort.as_str()
            ));
        }
        for i in 0..self.nodes.len() {
            let (sort, body) = self.smt_node(i);
            out.push_str(&format!("(define-fun t{} () {} {})\n", i, sort, body));
        }
        for (b, why) in asserts {
            out.push_str(&format!("(assert t{}) ; {}\n", b.0, why));
        }
        out
    }

    fn smt_node(&self, i: usize) -> (&'static str, String) {
        let s = |id: u32| format!("t{}", id);
        match self.nodes[i] {
            Node::Const(n) => ("Int", smt_int_literal(n)),
            Node::Var(v) => ("Int", self.var_name(v).to_string()),
            Node::Add(a, b) => ("Int", format!("(+ {} {})", s(a.0), s(b.0))),
            Node::Sub(a, b) => ("Int", format!("(- {} {})", s(a.0), s(b.0))),
            Node::Mul(k, t) => ("Int", format!("(* {} {})", smt_int_literal(k), s(t.0))),
            Node::Div(t, k) => ("Int", smt_floor_div(&s(t.0), k)),
            Node::Mod(t, k) => ("Int", smt_floor_mod(&s(t.0), k)),
            // SMT-LIB has no min/max; both are one `ite`.
            Node::Min(a, b) => ("Int", format!("(ite (<= {0} {1}) {0} {1})", s(a.0), s(b.0))),
            Node::Max(a, b) => ("Int", format!("(ite (<= {0} {1}) {1} {0})", s(a.0), s(b.0))),
            Node::Ite(c, a, b) => ("Int", format!("(ite {} {} {})", s(c.0), s(a.0), s(b.0))),
            Node::True => ("Bool", "true".to_string()),
            Node::False => ("Bool", "false".to_string()),
            Node::BVar(v) => ("Bool", self.var_name(v).to_string()),
            Node::Eq(a, b) => ("Bool", format!("(= {} {})", s(a.0), s(b.0))),
            Node::Lt(a, b) => ("Bool", format!("(< {} {})", s(a.0), s(b.0))),
            Node::Le(a, b) => ("Bool", format!("(<= {} {})", s(a.0), s(b.0))),
            Node::Not(b) => ("Bool", format!("(not {})", s(b.0))),
            Node::And(a, b) => ("Bool", format!("(and {} {})", s(a.0), s(b.0))),
            Node::Or(a, b) => ("Bool", format!("(or {} {})", s(a.0), s(b.0))),
        }
    }
}

/// SMT-LIB 2 has no negative numerals: `-3` is `(- 3)`.
fn smt_int_literal(n: i128) -> String {
    if n < 0 {
        format!("(- {})", -n)
    } else {
        n.to_string()
    }
}

/// Python's floor division rendered in SMT-LIB 2, whose `div` is
/// *Euclidean*: `(div a b)` is the `q` with `a = b*q + r` and `0 <= r < |b|`.
///
/// For `k > 0` Euclidean and floor division agree. For `k < 0` they differ
/// (Euclidean rounds toward `+inf`, Python toward `-inf`); with `m = -k > 0`,
/// `floor(a / k) = floor(-a / m) = (div (- a) m)`, because the divisor of
/// that Euclidean division is positive. `k` is always a non-zero constant
/// here, so no zero case can arise.
fn smt_floor_div(child: &str, k: i128) -> String {
    if k > 0 {
        format!("(div {} {})", child, k)
    } else {
        format!("(div (- {}) {})", child, -k)
    }
}

/// Python's `%` in SMT-LIB 2. For `k > 0` it is SMT-LIB's `mod`. For `k < 0`
/// with `m = -k`, `a mod k = a - k*floor(a/k) = a + m*(div (- a) m)`, using
/// the same rewrite as [`smt_floor_div`].
fn smt_floor_mod(child: &str, k: i128) -> String {
    if k > 0 {
        format!("(mod {} {})", child, k)
    } else {
        format!("(+ {} (* {} (div (- {}) {})))", child, -k, child, -k)
    }
}

// ---------------------------------------------------------------------------
// JSON export of the DAG
// ---------------------------------------------------------------------------

impl TermArena {
    /// The term DAG as JSON, so the Python harnesses of `acceptance.md` §2.3
    /// can rebuild it: `{"vars": [...], "nodes": [...]}` with `nodes[i].id ==
    /// i` and every argument a smaller id. Integer constants are strings so
    /// that values outside the JSON-safe range survive exactly.
    pub fn to_json(&self) -> serde_json::Value {
        let vars: Vec<serde_json::Value> = self
            .vars
            .iter()
            .enumerate()
            .map(|(i, d)| serde_json::json!({"id": i, "name": d.name, "sort": d.sort.as_str()}))
            .collect();
        let nodes: Vec<serde_json::Value> =
            (0..self.nodes.len()).map(|i| self.node_json(i)).collect();
        serde_json::json!({"vars": vars, "nodes": nodes})
    }

    fn node_json(&self, i: usize) -> serde_json::Value {
        let int = |op: &str, args: Vec<u32>| serde_json::json!({"id": i, "sort": "int", "op": op, "args": args});
        let boolean = |op: &str, args: Vec<u32>| serde_json::json!({"id": i, "sort": "bool", "op": op, "args": args});
        match self.nodes[i] {
            Node::Const(n) => {
                serde_json::json!({"id": i, "sort": "int", "op": "const", "value": n.to_string()})
            }
            Node::Var(v) => serde_json::json!({"id": i, "sort": "int", "op": "var", "var": v.0}),
            Node::Add(a, b) => int("add", vec![a.0, b.0]),
            Node::Sub(a, b) => int("sub", vec![a.0, b.0]),
            Node::Mul(k, t) => {
                serde_json::json!({"id": i, "sort": "int", "op": "mul", "k": k.to_string(), "args": [t.0]})
            }
            Node::Div(t, k) => {
                serde_json::json!({"id": i, "sort": "int", "op": "div", "k": k.to_string(), "args": [t.0]})
            }
            Node::Mod(t, k) => {
                serde_json::json!({"id": i, "sort": "int", "op": "mod", "k": k.to_string(), "args": [t.0]})
            }
            Node::Min(a, b) => int("min", vec![a.0, b.0]),
            Node::Max(a, b) => int("max", vec![a.0, b.0]),
            Node::Ite(c, a, b) => int("ite", vec![c.0, a.0, b.0]),
            Node::True => boolean("true", vec![]),
            Node::False => boolean("false", vec![]),
            Node::BVar(v) => serde_json::json!({"id": i, "sort": "bool", "op": "bvar", "var": v.0}),
            Node::Eq(a, b) => boolean("eq", vec![a.0, b.0]),
            Node::Lt(a, b) => boolean("lt", vec![a.0, b.0]),
            Node::Le(a, b) => boolean("le", vec![a.0, b.0]),
            Node::Not(b) => boolean("not", vec![b.0]),
            Node::And(a, b) => boolean("and", vec![a.0, b.0]),
            Node::Or(a, b) => boolean("or", vec![a.0, b.0]),
        }
    }
}

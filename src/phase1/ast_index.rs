//! Key → AST node: the inverse of phase 0's key assignment.
//!
//! `revision/phase1/README.md` §1 reads P through Δ's keys: "every key of Δ
//! names the AST node its value came from … phase 1 reads the operator and
//! its constant parameters from that node. Δ does not repeat them." Phase 0
//! assigns those keys with [`crate::phase0::desc::KeyCursor`] while it walks;
//! this module walks the same tree in the same order and records what each
//! key names.
//!
//! # What a key determines
//!
//! A [`ValueKey`] is (statement path, iteration counters, call depth,
//! expression index). The *AST node* depends on three of the four: the
//! iteration counters tell two unrollings of one node apart, and the node is
//! the same in every iteration. They are not dropped entirely, because the
//! expression counter restarts when a loop begins ([`KeyCursor::set_iteration`]):
//! the `range(…)` of a `for` is numbered in the statement's own counter and
//! the guard of a `while` in the per-iteration one, both at the statement's
//! path. So the index is keyed by (statement path, calls, *loop depth*,
//! expression index), and the depth is the length of the iteration vector.
//!
//! # What is recorded
//!
//! Not the AST node itself but the summary [`AstNode`] of what
//! `rules/README.md` §2 says a rule reads from it: the operator, the callee
//! and the constant parameters of the call. A borrowed node would make the
//! index self-referential, because a chip body is parsed from its JSON here
//! and not owned by the circuit.
//!
//! # When a key does not resolve
//!
//! The walk covers the statement and expression forms phase 0 keys; a form
//! it does not follow leaves its subtree unindexed, and a rule that finds no
//! node simply does not match, which phase 1 reports as a missing provider
//! rather than as a wrong one. Every rule also checks the node against the
//! producer phase 0 recorded, so a node that disagrees with Δ never matches
//! either.

use std::collections::HashMap;

use crate::ast::{
    ASTAnnotation, ASTChip, ASTCircuit, ASTNode, ASTSlice,
};
use crate::phase0::desc::{Arm, CallSite, PathStep, ValueKey};

/// The part of a [`ValueKey`] that determines the AST node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AstKey {
    pub stmt_path: Vec<PathStep>,
    pub calls: Vec<CallSite>,
    /// The number of enclosing unrolled loops, which is what tells the
    /// statement's own expression numbering from the per-iteration one.
    pub depth: usize,
    pub expr_index: u32,
}

impl AstKey {
    pub fn of(key: &ValueKey) -> AstKey {
        AstKey {
            stmt_path: key.stmt_path.clone(),
            calls: key.calls.clone(),
            depth: key.iters.len(),
            expr_index: key.expr_index,
        }
    }
}

/// The compile-time value of one argument of a call, as far as a rule reads
/// it (`rules/README.md` §2: "the AST node gives the operator and its
/// constant parameters").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constant {
    Int(i64),
    Bool(bool),
    /// A tuple or list of integer literals: an `axes` permutation, a
    /// `reshape` target, an `axis` tuple of a reduction.
    Ints(Vec<i64>),
    /// An argument that is no compile-time constant, or none the rules read.
    Other,
}

impl Constant {
    pub fn int(&self) -> Option<i64> {
        match self {
            Constant::Int(value) => Some(*value),
            Constant::Bool(value) => Some(*value as i64),
            _ => None,
        }
    }

    /// The argument as a list of axes: a tuple of literals, or one literal.
    pub fn ints(&self) -> Option<Vec<i64>> {
        match self {
            Constant::Ints(values) => Some(values.clone()),
            Constant::Int(value) => Some(vec![*value]),
            _ => None,
        }
    }

    pub fn boolean(&self) -> Option<bool> {
        match self {
            Constant::Bool(value) => Some(*value),
            Constant::Int(value) => Some(*value != 0),
            _ => None,
        }
    }
}

/// One entry of a subscript, as `mapping/make_ref.md` normalises it: what
/// the entry is, and the constant parameters it carries. Which value an
/// expression entry denotes — an integer scalar, an index array, a Boolean
/// mask — is Δ's to say, so the entry only records that there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscriptEntry {
    /// An expression entry, with the expression index of the value it
    /// evaluated to: `a[i]`'s index, `a[I]`'s index array, `a[m]`'s mask.
    Single { expr_index: u32 },
    /// `lo:hi:step`, with the literal step and the expression index of each
    /// endpoint that was written (an omitted endpoint is Python's default,
    /// `make_ref.md` P1).
    Slice { lo: Option<u32>, hi: Option<u32>, step: Option<i64> },
    NewAxis,
    Ellipsis,
}

impl SubscriptEntry {
    /// The expression index of the value an entry reads, when it has one.
    pub fn expr_index(&self) -> Option<u32> {
        match self {
            SubscriptEntry::Single { expr_index } => Some(*expr_index),
            _ => None,
        }
    }
}

/// What a rule reads from an AST node (`rules/README.md` §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstNode {
    /// An integer or Boolean literal.
    IntLiteral(i64),
    BoolLiteral(bool),
    /// A Float literal. The value is not carried: the row it selects is one
    /// `fop⟨constant_float⟩` whatever the number, and phase 3 reads the
    /// number off the AST node itself.
    FloatLiteral,
    /// A literal of a dtype the library states no encoding for.
    OpaqueLiteral,
    /// A use of a name.
    Name(String),
    /// `a op b`; `operator` is the front end's spelling (`add`, `floor_div`,
    /// `lt`, …).
    Binary { operator: String },
    /// `op a`: `usub`, `not`, `invert`, `uadd`.
    Unary { operator: String },
    /// `f(...)`, `np.f(...)`, `x.f(...)`: the module or receiver name when
    /// the front end wrote one, the member, and the literal value of each
    /// positional argument that is one.
    Call {
        target: Option<String>,
        member: String,
        arguments: Vec<Constant>,
        keywords: Vec<(String, Constant)>,
    },
    /// `<expr>.m(...)`.
    MethodCall {
        member: String,
        arguments: Vec<Constant>,
        keywords: Vec<(String, Constant)>,
    },
    /// `v[...]`, with its entries.
    Subscript { entries: Vec<SubscriptEntry> },
    /// `a if c else b`.
    Conditional,
    /// A list or tuple display.
    Composite { tuple: bool, elements: usize },
    /// A comprehension.
    Comprehension,
    /// A node phase 0 keys and no rule of step 1 reads.
    Other(&'static str),
}

impl AstNode {
    /// The operator of a binary or unary node.
    pub fn operator(&self) -> Option<&str> {
        match self {
            AstNode::Binary { operator } | AstNode::Unary { operator } => Some(operator),
            _ => None,
        }
    }

    /// The member of a call, whatever its form.
    pub fn member(&self) -> Option<&str> {
        match self {
            AstNode::Call { member, .. } | AstNode::MethodCall { member, .. } => Some(member),
            _ => None,
        }
    }

    /// The positional arguments of a call.
    pub fn arguments(&self) -> &[Constant] {
        match self {
            AstNode::Call { arguments, .. } | AstNode::MethodCall { arguments, .. } => arguments,
            _ => &[],
        }
    }

    /// One positional argument of a call.
    pub fn argument(&self, at: usize) -> Option<&Constant> {
        self.arguments().get(at)
    }

    /// One keyword argument of a call.
    pub fn keyword(&self, name: &str) -> Option<&Constant> {
        let keywords = match self {
            AstNode::Call { keywords, .. } | AstNode::MethodCall { keywords, .. } => keywords,
            _ => return None,
        };
        keywords
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }

    /// A parameter written either positionally or by keyword, which is how
    /// `axis` and `keepdims` arrive.
    pub fn parameter(&self, at: usize, name: &str) -> Option<&Constant> {
        self.keyword(name).or_else(|| self.argument(at))
    }

    /// The entries of a subscript.
    pub fn entries(&self) -> Option<&[SubscriptEntry]> {
        match self {
            AstNode::Subscript { entries } => Some(entries),
            _ => None,
        }
    }
}

/// A circuit input, as `INPUT_SCALAR` and `INPUT_STORE` read it.
#[derive(Debug, Clone)]
pub struct InputNode {
    pub name: String,
    pub annotation: ASTAnnotation,
}

/// Key → AST node for one circuit.
#[derive(Debug, Default)]
pub struct AstIndex {
    nodes: HashMap<AstKey, AstNode>,
    /// The nodes of the *intermediates* a value's key names
    /// (`../phase0/definition.md` §3.3), by (node, step).
    ///
    /// An intermediate reads the same node as the value it belongs to —
    /// that is what lets a rule read the operator's constant parameters off
    /// it — except where the intermediate is an operation of its own that
    /// the source never wrote. The one such case is the comparison a
    /// division records for `defcond_div` (§5.2): step 1 of a `div`,
    /// `floor_div` or `mod` node is the test that its divisor is not zero.
    steps: HashMap<(AstKey, u32), AstNode>,
    inputs: Vec<InputNode>,
}

impl AstIndex {
    /// Walk the circuit and its chips the way phase 0 walked them.
    pub fn build(circuit: &ASTCircuit, chips: &HashMap<String, serde_json::Value>) -> AstIndex {
        let mut index = AstIndex {
            nodes: HashMap::new(),
            steps: HashMap::new(),
            inputs: circuit
                .inputs
                .iter()
                .map(|input| InputNode {
                    name: input.name.clone(),
                    annotation: input.annotation.clone(),
                })
                .collect(),
        };
        let parsed = parse_chips(chips);
        let mut walk = Walk {
            index: &mut index,
            chips: &parsed,
            path: Vec::new(),
            calls: Vec::new(),
            depth: 0,
            next_expr: 0,
            recursion: 0,
        };
        walk.block(&circuit.block);
        index
    }

    /// The node the key names, when the walk reached it.
    ///
    /// An intermediate names the node of the value it belongs to, unless the
    /// walk recorded a node for that step of it (see [`AstIndex::steps`]).
    pub fn node(&self, key: &ValueKey) -> Option<&AstNode> {
        let at = AstKey::of(key);
        if key.step > 0 {
            if let Some(node) = self.steps.get(&(at.clone(), key.step)) {
                return Some(node);
            }
        }
        self.nodes.get(&at)
    }

    /// The parameter a circuit input's key names.
    ///
    /// `crate::phase0::build_input_descriptors` keys input `k` with an empty
    /// statement path and the expression index `k`, and no expression key
    /// can collide with it, because an expression always lies inside a
    /// statement.
    ///
    /// A step index names one element of a composite parameter
    /// (`../phase0/definition.md` §3.3), which the front end binds as part of
    /// that parameter: the element's node is the parameter's, so its
    /// annotation says whether the group of cells is private and whether it
    /// is hashed. The element's own dtype comes from Δ, not from here.
    pub fn input(&self, key: &ValueKey) -> Option<&InputNode> {
        if !key.stmt_path.is_empty() || !key.calls.is_empty() || !key.iters.is_empty() {
            return None;
        }
        self.inputs.get(key.expr_index as usize)
    }

    pub fn inputs(&self) -> &[InputNode] {
        &self.inputs
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

fn parse_chips(chips: &HashMap<String, serde_json::Value>) -> HashMap<String, ASTChip> {
    chips
        .iter()
        .filter_map(|(name, raw)| {
            let node: ASTNode = serde_json::from_value(raw["chip_ast"].clone()).ok()?;
            match node {
                ASTNode::ASTChip(chip) => Some((name.clone(), chip)),
                _ => None,
            }
        })
        .collect()
}

/// The traversal, mirroring `crate::phase0::walk`.
struct Walk<'a> {
    index: &'a mut AstIndex,
    chips: &'a HashMap<String, ASTChip>,
    path: Vec<PathStep>,
    calls: Vec<CallSite>,
    depth: usize,
    next_expr: u32,
    recursion: u32,
}

impl Walk<'_> {
    /// `Walker::block`: one statement path component per block.
    fn block(&mut self, statements: &[ASTNode]) {
        let mark = self.next_expr;
        self.path.push(PathStep::Stmt(0));
        for (at, statement) in statements.iter().enumerate() {
            if let Some(slot @ PathStep::Stmt(_)) = self.path.last_mut() {
                *slot = PathStep::Stmt(at as u32);
            }
            self.next_expr = 0;
            self.statement(statement);
        }
        self.path.pop();
        self.next_expr = mark;
    }

    fn statement(&mut self, node: &ASTNode) {
        match node {
            ASTNode::ASTAssignStatement(n) => {
                self.expr(&n.value);
                for target in &n.targets {
                    self.target(target);
                }
            }
            ASTNode::ASTCondStatement(n) => {
                self.expr(&n.cond);
                self.arm(Arm::Then, |walk| walk.block(&n.t_block));
                self.arm(Arm::Else, |walk| walk.block(&n.f_block));
            }
            ASTNode::ASTForInStatement(n) => {
                // `Walker::for_plan`: a `range` iterable is recognised
                // rather than evaluated, and the call node still takes a key.
                self.expr(&n.iter_expr);
                self.in_loop(|walk| {
                    walk.target(&n.target);
                    walk.block(&n.block);
                });
                self.block(&n.orelse);
            }
            ASTNode::ASTWhileStatement(n) => {
                self.in_loop(|walk| {
                    walk.expr(&n.test_expr);
                    walk.block(&n.block);
                });
                self.block(&n.orelse);
            }
            ASTNode::ASTReturnStatement(n) => {
                if let Some(expr) = &n.expr {
                    self.expr(expr);
                }
            }
            ASTNode::ASTAssertStatement(n) => {
                self.expr(&n.expr);
            }
            ASTNode::ASTExprStatement(n) => {
                self.expr(&n.expr);
            }
            _ => {}
        }
    }

    /// An assignment target: only the forms that take keys of their own.
    fn target(&mut self, node: &ASTNode) {
        match node {
            ASTNode::ASTSubscriptAssignTarget(n) => {
                // `phase0/walk/subscript.rs`, `setitem`: the *write* key is
                // taken first and names the region `a[idx]` as it was read;
                // the target and the slice entries are evaluated inside it,
                // and one further key holds the new version. The write key
                // is the one `SETITEM` reads the subscript off, so the node
                // is recorded under it, as a subscript expression's is.
                let write = self.next_expr;
                self.next_expr += 1;
                self.expr(&n.target);
                let entries = self.slice(&n.slicing);
                self.record(write, AstNode::Subscript { entries });
                // The new version.
                self.next_expr += 1;
            }
            ASTNode::ASTTupleAssignTarget(n) => {
                for target in &n.targets {
                    self.target(target);
                }
            }
            ASTNode::ASTListAssignTarget(n) => {
                for target in &n.targets {
                    self.target(target);
                }
            }
            _ => {}
        }
    }

    fn arm(&mut self, which: Arm, body: impl FnOnce(&mut Self)) {
        let mark = self.next_expr;
        self.path.push(PathStep::Arm(which));
        self.next_expr = 0;
        body(self);
        self.path.pop();
        self.next_expr = mark;
    }

    /// `KeyCursor::enter_loop` plus the `set_iteration` of every iteration:
    /// the AST node is the same in each, so one pass indexes them all.
    fn in_loop(&mut self, body: impl FnOnce(&mut Self)) {
        let mark = self.next_expr;
        self.depth += 1;
        self.next_expr = 0;
        body(self);
        self.depth -= 1;
        self.next_expr = mark;
    }

    /// One expression node, in preorder: its own key first, then the
    /// children in the order the walker evaluates them. The index the node
    /// was given is returned, because a subscript entry is named by the
    /// index of the expression it evaluated (`SubscriptEntry`).
    fn expr(&mut self, node: &ASTNode) -> u32 {
        let expr_index = self.next_expr;
        self.next_expr += 1;
        self.record(expr_index, summarise(node));
        match node {
            ASTNode::ASTBinaryOperator(n) => {
                // The divisor test of `defcond_div` is intermediate 0 of the
                // division, and it is a comparison, not the division itself
                // (`../phase0/definition.md` §5.2, §3.3).
                if matches!(n.operator.as_str(), "div" | "floor_div" | "mod") {
                    self.record_step(
                        expr_index,
                        1,
                        AstNode::Binary { operator: "ne".to_string() },
                    );
                }
                self.expr(&n.lhs);
                self.expr(&n.rhs);
            }
            ASTNode::ASTUnaryOperator(n) => {
                self.expr(&n.operand);
            }
            ASTNode::ASTSubscriptExp(n) => {
                self.expr(&n.val);
                let entries = self.slice(&n.slicing);
                // The entries are known only after the walk, because each
                // names the index of the expression it evaluated.
                self.record(expr_index, AstNode::Subscript { entries });
            }
            ASTNode::ASTNamedAttribute(n) => {
                self.arguments(&n.args, &n.kwargs);
                if n.target.is_none() {
                    self.inline(&n.member, expr_index);
                }
            }
            ASTNode::ASTExprAttribute(n) => {
                self.expr(&n.target);
                self.arguments(&n.args, &n.kwargs);
            }
            ASTNode::ASTSquareBrackets(n) => self.elements(&n.values),
            ASTNode::ASTParenthesis(n) => self.elements(&n.values),
            ASTNode::ASTCondExp(n) => {
                self.expr(&n.cond);
                self.arm(Arm::Then, |walk| {
                    walk.expr(&n.t_expr);
                });
                self.arm(Arm::Else, |walk| {
                    walk.expr(&n.f_expr);
                });
            }
            ASTNode::ASTGeneratorExp(n) => self.generators(n, 0),
            ASTNode::ASTJoinedStr(n) => self.elements(&n.values),
            ASTNode::ASTFormattedValue(n) => {
                self.expr(&n.value);
            }
            ASTNode::ASTStarredExpr(n) => {
                self.expr(&n.value);
            }
            _ => {}
        }
        expr_index
    }

    /// The elements of a display, with a starred element spread in place.
    fn elements(&mut self, values: &[ASTNode]) {
        for value in values {
            match value {
                ASTNode::ASTStarredExpr(starred) => self.expr(&starred.value),
                other => self.expr(other),
            };
        }
    }

    /// `Walker::call_args`: the positional arguments in order, then the
    /// keyword arguments by sorted name.
    fn arguments(&mut self, args: &[ASTNode], kwargs: &HashMap<String, ASTNode>) {
        for arg in args {
            match arg {
                ASTNode::ASTStarredExpr(starred) => self.expr(&starred.value),
                other => self.expr(other),
            };
        }
        let mut names: Vec<&String> = kwargs.keys().collect();
        names.sort();
        for name in names {
            self.expr(&kwargs[name]);
        }
    }

    /// `Walker::inline_chip`: the body is a block under a call component of
    /// the key, with the expression counter restarted.
    fn inline(&mut self, member: &str, site: u32) {
        let Some(chip) = self.chips.get(member).cloned() else {
            return;
        };
        if self.recursion >= 16 {
            return;
        }
        self.recursion += 1;
        let mark = self.next_expr;
        self.calls.push(CallSite { callee: member.to_string(), site });
        self.next_expr = 0;
        self.block(&chip.block);
        self.calls.pop();
        self.next_expr = mark;
        self.recursion -= 1;
    }

    /// `Walker::expand_generators`: the iterable, then one unrolled pass
    /// over the filters and the next generator.
    fn generators(&mut self, node: &crate::ast::ASTGeneratorExp, at: usize) {
        let Some(generator) = node.generators.get(at) else {
            self.expr(&node.elt);
            return;
        };
        self.expr(&generator.iter_expr);
        self.in_loop(|walk| {
            walk.target(&generator.target);
            for condition in &generator.ifs {
                walk.expr(condition);
            }
            walk.generators(node, at + 1);
        });
    }

    /// `Walker::slice_indices`: a range entry evaluates its endpoints, a
    /// single entry is one expression, `...` and `np.newaxis` take no key.
    ///
    /// The entries are returned with the expression index of every value
    /// they evaluated, which is what names that value's key in Δ.
    fn slice(&mut self, slicing: &ASTSlice) -> Vec<SubscriptEntry> {
        let mut out = Vec::with_capacity(slicing.data.len());
        for entry in &slicing.data {
            if let Some(triple) = entry.as_array() {
                let mut endpoints: [Option<u32>; 3] = [None; 3];
                let mut step: Option<i64> = None;
                for (at, endpoint) in triple.iter().take(3).enumerate() {
                    if endpoint.is_null() {
                        continue;
                    }
                    let Ok(node) = serde_json::from_value::<ASTNode>(endpoint.clone()) else {
                        continue;
                    };
                    if matches!(node, ASTNode::ASTConstantNone(_)) {
                        // The front end writes an omitted endpoint as a
                        // `None` node, which the walker evaluates like any
                        // other expression and then reads as "omitted".
                        self.expr(&node);
                        continue;
                    }
                    let index = self.expr(&node);
                    if at == 2 {
                        step = literal_of(&node).int();
                    } else {
                        endpoints[at] = Some(index);
                    }
                }
                out.push(SubscriptEntry::Slice {
                    lo: endpoints[0],
                    hi: endpoints[1],
                    step,
                });
                continue;
            }
            let class = entry.get("__class__").and_then(|value| value.as_str());
            match class {
                Some("ASTSliceNewAxis") => out.push(SubscriptEntry::NewAxis),
                Some("ASTSliceEllipsis") => out.push(SubscriptEntry::Ellipsis),
                _ => {
                    if let Ok(node) = serde_json::from_value::<ASTNode>(entry.clone()) {
                        let index = self.expr(&node);
                        out.push(SubscriptEntry::Single { expr_index: index });
                    }
                }
            }
        }
        out
    }

    /// The node one *intermediate* of the expression at `expr_index` reads.
    fn record_step(&mut self, expr_index: u32, step: u32, node: AstNode) {
        let key = AstKey {
            stmt_path: self.path.clone(),
            calls: self.calls.clone(),
            depth: self.depth,
            expr_index,
        };
        self.index.steps.insert((key, step), node);
    }

    fn record(&mut self, expr_index: u32, node: AstNode) {
        let key = AstKey {
            stmt_path: self.path.clone(),
            calls: self.calls.clone(),
            depth: self.depth,
            expr_index,
        };
        self.index.nodes.insert(key, node);
    }
}

/// The summary of one node.
fn summarise(node: &ASTNode) -> AstNode {
    match node {
        ASTNode::ASTConstantInteger(n) => AstNode::IntLiteral(n.value),
        ASTNode::ASTConstantBoolean(n) => AstNode::BoolLiteral(n.value),
        ASTNode::ASTConstantFloat(_) => AstNode::FloatLiteral,
        ASTNode::ASTConstantComplex(_)
        | ASTNode::ASTConstantString(_)
        | ASTNode::ASTConstantNone(_) => AstNode::OpaqueLiteral,
        ASTNode::ASTLoad(n) => AstNode::Name(n.name.clone()),
        ASTNode::ASTBinaryOperator(n) => AstNode::Binary { operator: n.operator.clone() },
        ASTNode::ASTUnaryOperator(n) => AstNode::Unary { operator: n.operator.clone() },
        ASTNode::ASTNamedAttribute(n) => AstNode::Call {
            target: n.target.clone(),
            member: n.member.clone(),
            arguments: n.args.iter().map(literal_of).collect(),
            keywords: keywords_of(&n.kwargs),
        },
        ASTNode::ASTExprAttribute(n) => AstNode::MethodCall {
            member: n.member.clone(),
            arguments: n.args.iter().map(literal_of).collect(),
            keywords: keywords_of(&n.kwargs),
        },
        ASTNode::ASTSubscriptExp(_) => AstNode::Subscript { entries: Vec::new() },
        ASTNode::ASTCondExp(_) => AstNode::Conditional,
        ASTNode::ASTSquareBrackets(n) => {
            AstNode::Composite { tuple: false, elements: n.values.len() }
        }
        ASTNode::ASTParenthesis(n) => {
            AstNode::Composite { tuple: true, elements: n.values.len() }
        }
        ASTNode::ASTGeneratorExp(_) => AstNode::Comprehension,
        other => AstNode::Other(class_name(other)),
    }
}

/// The class of a node phase 1 has no summary for, for a readable report.
fn class_name(node: &ASTNode) -> &'static str {
    match node {
        ASTNode::ASTStarredExpr(_) => "starred",
        ASTNode::ASTJoinedStr(_) | ASTNode::ASTFormattedValue(_) => "f-string",
        ASTNode::ASTNameAssignTarget(_)
        | ASTNode::ASTSubscriptAssignTarget(_)
        | ASTNode::ASTTupleAssignTarget(_)
        | ASTNode::ASTListAssignTarget(_) => "assign target",
        _ => "other",
    }
}

fn literal_of(node: &ASTNode) -> Constant {
    match node {
        ASTNode::ASTConstantInteger(n) => Constant::Int(n.value),
        ASTNode::ASTConstantBoolean(n) => Constant::Bool(n.value),
        ASTNode::ASTParenthesis(n) => ints_of(&n.values),
        ASTNode::ASTSquareBrackets(n) => ints_of(&n.values),
        ASTNode::ASTUnaryOperator(n) if n.operator == "usub" => match literal_of(&n.operand) {
            Constant::Int(value) => Constant::Int(-value),
            _ => Constant::Other,
        },
        _ => Constant::Other,
    }
}

/// A display of integer literals, which is how a permutation, a reshape
/// target and an axis tuple are written.
fn ints_of(values: &[ASTNode]) -> Constant {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        match literal_of(value) {
            Constant::Int(n) => out.push(n),
            _ => return Constant::Other,
        }
    }
    Constant::Ints(out)
}

fn keywords_of(kwargs: &HashMap<String, ASTNode>) -> Vec<(String, Constant)> {
    let mut names: Vec<&String> = kwargs.keys().collect();
    names.sort();
    names
        .into_iter()
        .map(|name| (name.clone(), literal_of(&kwargs[name])))
        .collect()
}



//! The structural cap of `S-While-Bounded` (`definition.md` §6.2).
//!
//! §6.2 unrolls a `while` to a cap `C ∈ S`, "the structural cap when
//! recognised, else `loop_limit`". The structural cap is the baseline's
//! (`structural_while_unroll_cap`, `src/ir_gen/visitors.rs`), and phase 0
//! reproduces it so that the two unroll the same number of times: a guard
//! `v > 0` on a name aliased from the innermost enclosing `for` variable,
//! whose first write in the body is `v = v - 1`, is capped by that
//! variable's current value.
//!
//! Recognising it is pure syntax over the AST — no descriptor, no term, no
//! $\Phi$ — so it is all here, behind [`WhileCap`], and the walker holds one
//! field instead of two. [`super::loops`] keeps the rules of §6.2.

use std::collections::{BTreeSet, HashMap};

use crate::ast::{ASTNode, ASTWhileStatement};

/// One enclosing `for` frame (`IRGenerator`'s `ForLoopFrame`,
/// `src/ir_gen/mod.rs`).
#[derive(Debug, Clone)]
struct ForFrame {
    /// The loop variable's name, empty when the target is not a plain name.
    var_name: String,
    /// One past the largest element value, or 0 when the elements are not
    /// non-negative constants.
    max_bound: u32,
    /// One past the value bound in the iteration now running.
    current_bound: Option<u32>,
}

/// The enclosing `for` frames and the induction aliases they license.
///
/// The walker opens a frame at every `for` ([`WhileCap::enter_for`]), notes
/// each iteration's element value ([`WhileCap::note_element`]) and each
/// `x = i` alias of the loop variable ([`WhileCap::note_alias`]), and asks
/// for a `while`'s cap with [`WhileCap::cap_for`].
#[derive(Debug, Default)]
pub(crate) struct WhileCap {
    frames: Vec<ForFrame>,
    /// Names that alias an enclosing `for` variable: alias -> loop variable.
    aliases: HashMap<String, String>,
}

impl WhileCap {
    /// Open a `for` frame. `values` is the compile-time value of each
    /// element, or `None` where it has none; a frame is only useful when the
    /// target is a plain name bound to non-negative constants, which is the
    /// baseline's `for_frame_from_elements`.
    pub fn enter_for(&mut self, target: &ASTNode, values: &[Option<i128>]) {
        let ASTNode::ASTNameAssignTarget(t) = target else {
            self.frames.push(ForFrame {
                var_name: String::new(),
                max_bound: 0,
                current_bound: None,
            });
            return;
        };
        self.clear_aliases_of(&t.name);
        self.frames.push(ForFrame {
            var_name: t.name.clone(),
            max_bound: one_past_max(values),
            current_bound: None,
        });
    }

    pub fn exit_for(&mut self) {
        if let Some(frame) = self.frames.pop() {
            self.clear_aliases_of(&frame.var_name);
        }
    }

    /// The compile-time value bound in the iteration about to run, or `None`
    /// when it has none.
    pub fn note_element(&mut self, value: Option<i128>) {
        let bound = value
            .filter(|n| *n >= 0)
            .and_then(|n| u32::try_from(n).ok())
            .map(|n| n.saturating_add(1));
        if let Some(frame) = self.frames.last_mut() {
            frame.current_bound = bound;
        }
    }

    /// Record `x = i` where `i` is the innermost `for` variable, which is
    /// what makes `x` an induction alias. Any other assignment to `x`
    /// removes the alias.
    pub fn note_alias(&mut self, target: &ASTNode, value: &ASTNode) {
        let ASTNode::ASTNameAssignTarget(t) = target else {
            return;
        };
        let origin = match (value, self.frames.last()) {
            (ASTNode::ASTLoad(load), Some(frame)) if load.name == frame.var_name => {
                Some(frame.var_name.clone())
            }
            _ => None,
        };
        match origin {
            Some(origin) => self.aliases.insert(t.name.clone(), origin),
            None => self.aliases.remove(&t.name),
        };
    }

    /// The structural cap of one `while`, or `None` when the loop does not
    /// match the pattern and `loop_limit` applies instead.
    pub fn cap_for(&self, n: &ASTWhileStatement) -> Option<u32> {
        let var = guard_positive_var(&n.test_expr)?;
        let origin = self.aliases.get(var)?;
        let frame = self.frames.last()?;
        if origin != &frame.var_name || frame.max_bound == 0 {
            return None;
        }
        if !body_has_first_write_unit_decrement(&n.block, var) {
            return None;
        }
        frame
            .current_bound
            .filter(|bound| *bound > 0)
            .or(Some(frame.max_bound))
    }

    fn clear_aliases_of(&mut self, origin: &str) {
        self.aliases.retain(|_, value| value != origin);
    }
}

/// One past the largest element value, or 0 unless every element is a
/// non-negative constant.
fn one_past_max(values: &[Option<i128>]) -> u32 {
    let mut max_seen: Option<i128> = None;
    for value in values {
        match value {
            Some(n) if *n >= 0 => max_seen = Some(max_seen.map_or(*n, |cur| cur.max(*n))),
            _ => return 0,
        }
    }
    match max_seen {
        Some(n) => u32::try_from(n).map_or(0, |n| n.saturating_add(1)),
        None => 0,
    }
}

/// `v > 0` or `0 < v`, possibly under an `and` (the baseline's
/// `guard_positive_var`).
fn guard_positive_var(node: &ASTNode) -> Option<&str> {
    let ASTNode::ASTBinaryOperator(bin) = node else {
        return None;
    };
    match bin.operator.as_str() {
        "and" => guard_positive_var(&bin.lhs).or_else(|| guard_positive_var(&bin.rhs)),
        "gt" => match (&*bin.lhs, &*bin.rhs) {
            (ASTNode::ASTLoad(load), ASTNode::ASTConstantInteger(z)) if z.value == 0 => {
                Some(load.name.as_str())
            }
            _ => None,
        },
        "lt" => match (&*bin.lhs, &*bin.rhs) {
            (ASTNode::ASTConstantInteger(z), ASTNode::ASTLoad(load)) if z.value == 0 => {
                Some(load.name.as_str())
            }
            _ => None,
        },
        _ => None,
    }
}

/// Whether the first write to `var` in the body is `var = var - 1`.
fn body_has_first_write_unit_decrement(block: &[ASTNode], var: &str) -> bool {
    for stmt in block {
        if is_unit_decrement(stmt, var) {
            return true;
        }
        let mut names = BTreeSet::new();
        assigned_names(stmt, &mut names);
        if names.contains(var) {
            return false;
        }
    }
    false
}

fn is_unit_decrement(stmt: &ASTNode, var: &str) -> bool {
    let ASTNode::ASTAssignStatement(assign) = stmt else {
        return false;
    };
    if assign.targets.len() != 1 || assigned_name(&assign.targets[0]) != Some(var) {
        return false;
    }
    let ASTNode::ASTBinaryOperator(bin) = &*assign.value else {
        return false;
    };
    bin.operator == "sub"
        && matches!(
            (&*bin.lhs, &*bin.rhs),
            (ASTNode::ASTLoad(load), ASTNode::ASTConstantInteger(one))
                if load.name == var && one.value == 1
        )
}

fn assigned_name(target: &ASTNode) -> Option<&str> {
    match target {
        ASTNode::ASTNameAssignTarget(t) => Some(t.name.as_str()),
        _ => None,
    }
}

/// Every name a statement can assign, including inside nested blocks
/// (the baseline's `collect_assigned_names_from_stmt`).
fn assigned_names(stmt: &ASTNode, out: &mut BTreeSet<String>) {
    let targets = |targets: &[ASTNode], out: &mut BTreeSet<String>| {
        for target in targets {
            assigned_names_of_target(target, out);
        }
    };
    match stmt {
        ASTNode::ASTAssignStatement(n) => targets(&n.targets, out),
        ASTNode::ASTAugAssignStatement(n) => targets(&n.targets, out),
        ASTNode::ASTCondStatement(n) => {
            for inner in n.t_block.iter().chain(n.f_block.iter()) {
                assigned_names(inner, out);
            }
        }
        ASTNode::ASTForInStatement(n) => {
            assigned_names_of_target(&n.target, out);
            for inner in n.block.iter().chain(n.orelse.iter()) {
                assigned_names(inner, out);
            }
        }
        ASTNode::ASTWhileStatement(n) => {
            for inner in n.block.iter().chain(n.orelse.iter()) {
                assigned_names(inner, out);
            }
        }
        _ => {}
    }
}

fn assigned_names_of_target(target: &ASTNode, out: &mut BTreeSet<String>) {
    match target {
        ASTNode::ASTNameAssignTarget(t) => {
            out.insert(t.name.clone());
        }
        ASTNode::ASTTupleAssignTarget(t) => {
            for inner in &t.targets {
                assigned_names_of_target(inner, out);
            }
        }
        ASTNode::ASTListAssignTarget(t) => {
            for inner in &t.targets {
                assigned_names_of_target(inner, out);
            }
        }
        ASTNode::ASTSubscriptAssignTarget(t) => {
            if let ASTNode::ASTLoad(load) = &*t.target {
                out.insert(load.name.clone());
            }
        }
        _ => {}
    }
}

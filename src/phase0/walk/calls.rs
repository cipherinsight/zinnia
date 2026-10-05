//! Calls: `S-Call` (§5.10), `E-External` (§5.11), and the dispatch from a
//! name to the shape class that handles it (`coverage.md` §5).
//!
//! The dispatch mirrors `IRGenerator::visit_named_attr` and
//! `visit_expr_attr` (`src/ir_gen/named_attr/mod.rs`): arguments are
//! evaluated first, left to right, then the keyword arguments; a bare name is
//! a chip before it is a builtin; a dotted name is `np`, `math`, `copy`, or a
//! method on a value in scope. What each class then does with the arguments
//! lives in [`super::arrays`] and [`super::values`].

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ast::{ASTChip, ASTExprAttribute, ASTNamedAttribute, ASTNode};
use crate::phase0::desc::{Desc, ValueKey};
use crate::phase0::ops::{self, Class};
use crate::phase0::rules::{e_external, Rule};
use crate::phase0::Reject;
use crate::types::ZinniaType;

use super::loops::Element;
use super::stmts::merge_returns;
use super::{Flow, Walker};

/// The evaluated arguments of one call.
///
/// Keyword arguments are held in a `BTreeMap` so that they are evaluated and
/// reported in a fixed order; the baseline's `HashMap` leaves the order
/// unspecified, which a record keyed by the traversal cannot afford
/// (`acceptance.md` §2.2).
///
/// Each argument carries the key of the expression that produced it, so that
/// a chip parameter can be bound to the value the caller passed
/// (`implementation.md` §5a): a chip parameter takes no key of its own. The
/// elements a starred argument spreads share the key of the starred
/// expression, which is the value they came out of.
pub(crate) struct Args {
    pub pos: Vec<Desc>,
    pub kw: BTreeMap<String, Desc>,
    keys: Vec<Option<ValueKey>>,
    kw_keys: BTreeMap<String, Option<ValueKey>>,
    /// The text of every argument the source wrote as a string literal, by
    /// keyword name and by position.
    ///
    /// A string carries no term and no shape (§3.2), so its descriptor is the
    /// opaque `String` scalar and the text is lost there. One operator needs
    /// it: `np.unravel_index` / `np.ravel_multi_index` compute a coordinate
    /// from the flat index, and `order=` says in which direction (§5.4).
    strings: BTreeMap<String, String>,
}

impl Args {
    /// The arguments of a call, with the key of each positional one.
    fn new(pos: Vec<Element>, kw: BTreeMap<String, Element>) -> Self {
        let (descs, keys) = pos.into_iter().map(|e| (e.desc, e.key)).unzip();
        let mut values = BTreeMap::new();
        let mut kw_keys = BTreeMap::new();
        for (name, element) in kw {
            values.insert(name.clone(), element.desc);
            kw_keys.insert(name, element.key);
        }
        Args {
            pos: descs,
            kw: values,
            keys,
            kw_keys,
            strings: BTreeMap::new(),
        }
    }

    /// The text of the argument at `index`, or of the keyword spelled
    /// `name`, when the source wrote it as a string literal.
    pub(crate) fn string(&self, index: usize, name: &str) -> Option<&str> {
        self.strings
            .get(&index.to_string())
            .or_else(|| self.strings.get(name))
            .map(String::as_str)
    }

    /// Put the receiver of a method call in front, keeping the keys in step
    /// with the descriptors.
    fn with_receiver(&mut self, receiver: Desc, key: Option<ValueKey>) {
        self.pos.insert(0, receiver);
        self.keys.insert(0, key);
        // The positional string literals are numbered by position, so they
        // shift with it.
        self.strings = self
            .strings
            .iter()
            .map(|(at, text)| match at.parse::<usize>() {
                Ok(index) => ((index + 1).to_string(), text.clone()),
                Err(_) => (at.clone(), text.clone()),
            })
            .collect();
    }

    /// The keys of every argument that has one, in evaluation order: the
    /// operands of a rule that reads the call's arguments.
    pub fn operand_keys(&self) -> Vec<ValueKey> {
        self.keys
            .iter()
            .chain(self.kw_keys.values())
            .flatten()
            .cloned()
            .collect()
    }

    /// The key of the argument at `index`, or of the keyword spelled `name`:
    /// the value a chip parameter is bound to, and the value a mask's count
    /// is keyed by (§5.5).
    pub(crate) fn key(&self, index: usize, name: &str) -> Option<ValueKey> {
        match self.keys.get(index) {
            Some(key) => key.clone(),
            None => self.kw_keys.get(name).cloned().flatten(),
        }
    }

    /// The argument at position `index`, or under `name` if the call spelled
    /// it as a keyword. This is the one place a numpy call's arguments are
    /// looked up.
    pub fn get(&self, index: usize, name: &str) -> Option<&Desc> {
        self.pos.get(index).or_else(|| self.kw.get(name))
    }

    /// A keyword-only argument.
    pub fn kwarg(&self, name: &str) -> Option<&Desc> {
        self.kw.get(name)
    }

    /// The argument a rule cannot do without.
    pub fn require(&self, rule: Rule, index: usize, name: &str) -> Result<&Desc, Reject> {
        self.get(index, name).ok_or_else(|| Reject::Premise {
            rule,
            detail: format!("without its `{}` argument", name),
        })
    }

    /// Every positional argument from `index` on, which is how the varargs
    /// spellings `a.reshape(2, 3)` and `np.transpose(a, 1, 0)` arrive.
    pub fn rest(&self, index: usize) -> &[Desc] {
        self.pos.get(index..).unwrap_or(&[])
    }
}

impl Walker<'_> {
    /// `f(...)`, `x.m(...)` and `np.f(...)`: a call whose callee is a name
    /// (`ASTNamedAttribute`).
    pub fn named_attribute(
        &mut self,
        n: &ASTNamedAttribute,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        // `np.linalg` is not a value; it only ever appears as the target of
        // an `ASTExprAttribute`, which handles it before evaluating the
        // target (`src/ir_gen/named_attr/mod.rs`).
        if n.target.as_deref() == Some("np") && n.member == "linalg" {
            return Err(Reject::Unsupported {
                construct: "`np.linalg` used as a value".to_string(),
            });
        }
        let args = self.call_args(&n.args, &n.kwargs)?;
        match n.target.as_deref() {
            None => self.bare_call(&n.member, args, key),
            Some("np") | Some("numpy") | Some("zinnia") => {
                self.dispatch(ops::numpy(&n.member), args, key, &n.member, "np")
            }
            Some("math") | Some("cmath") => {
                self.dispatch(ops::math(&n.member), args, key, &n.member, "math")
            }
            // `copy.copy` / `copy.deepcopy` are the identity on shapes.
            Some("copy") => self.dispatch(Some(Class::Copy), args, key, &n.member, "copy"),
            Some("DynamicNDArray") => self.dispatch(
                ops::numpy(&n.member),
                args,
                key,
                &n.member,
                "DynamicNDArray",
            ),
            Some(target) => self.method_call(target, &n.member, args, key),
        }
    }

    /// `<expr>.m(...)` (`ASTExprAttribute`): the target is evaluated first,
    /// then the arguments.
    pub fn expr_attribute(&mut self, n: &ASTExprAttribute, key: &ValueKey) -> Result<Desc, Reject> {
        if let ASTNode::ASTNamedAttribute(inner) = &*n.target {
            if inner.target.as_deref() == Some("np") && inner.member == "linalg" {
                let args = self.call_args(&n.args, &n.kwargs)?;
                return self.dispatch(ops::linalg(&n.member), args, key, &n.member, "np.linalg");
            }
        }
        let (receiver, receiver_key) = self.expr_keyed(&n.target)?;
        let mut args = self.call_args(&n.args, &n.kwargs)?;
        args.with_receiver(receiver, Some(receiver_key));
        self.dispatch(ops::method(&n.member), args, key, &n.member, "value")
    }

    /// A bare name: a chip first, then a builtin, then an external
    /// (`src/ir_gen/named_attr/mod.rs`'s dispatch order).
    fn bare_call(&mut self, member: &str, args: Args, key: &ValueKey) -> Result<Desc, Reject> {
        if self.is_chip(member) {
            return self.inline_chip(member, args, key);
        }
        if let Some(class) = ops::builtin(member) {
            return self.dispatch(Some(class), args, key, member, "builtin");
        }
        if let Some(return_dt) = self.external_return_dt(member) {
            return self.call_external(member, return_dt?, key);
        }
        Err(Reject::UnsupportedOperator {
            name: member.to_string(),
        })
    }

    /// `x.m(...)` with `x` a name in scope: the receiver becomes argument 0,
    /// which is how the baseline's method tables are written.
    fn method_call(
        &mut self,
        target: &str,
        member: &str,
        mut args: Args,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let receiver =
            self.frame()
                .get(target)
                .cloned()
                .ok_or_else(|| Reject::UnsupportedOperator {
                    name: format!("{}.{}", target, member),
                })?;
        let receiver_key = self.frame().key_of(target);
        if let Some(Class::ListMethod(method)) = ops::method(member) {
            return self.list_method(target, receiver, receiver_key, method, args, key);
        }
        // The receiver is a name, not an expression the walker evaluated, so
        // nothing has pushed it into this value's frame. It is the value the
        // method reads, and without this the rule's operand list is empty and
        // phase 1 cannot find the array a reduction ran over
        // (`../../../phase1/rules/README.md` §10 item 10).
        if let Some(receiver_key) = &receiver_key {
            self.an.operand(receiver_key);
        }
        args.with_receiver(receiver, receiver_key);
        self.dispatch(ops::method(member), args, key, member, target)
    }

    /// Apply a shape class, or reject with the name the histogram reports.
    fn dispatch(
        &mut self,
        class: Option<Class>,
        args: Args,
        key: &ValueKey,
        member: &str,
        namespace: &str,
    ) -> Result<Desc, Reject> {
        let class = class.ok_or_else(|| Reject::UnsupportedOperator {
            name: format!("{}.{}", namespace, member),
        })?;
        self.apply_class(class, &args, key, member)
    }

    /// Evaluate a call's arguments: positional first, left to right, with
    /// starred arguments flattened, then the keyword arguments by name.
    pub fn call_args(
        &mut self,
        args: &[ASTNode],
        kwargs: &std::collections::HashMap<String, ASTNode>,
    ) -> Result<Args, Reject> {
        let mut pos = Vec::with_capacity(args.len());
        let mut strings = BTreeMap::new();
        for arg in args {
            match arg {
                ASTNode::ASTStarredExpr(starred) => {
                    let (inner, source) = self.expr_keyed(&starred.value)?;
                    self.spread(&inner, Some(&source), &mut pos)?;
                }
                other => {
                    if let ASTNode::ASTConstantString(text) = other {
                        strings.insert(pos.len().to_string(), text.value.clone());
                    }
                    let (desc, key) = self.expr_keyed(other)?;
                    pos.push(Element::new(desc, Some(key)));
                }
            }
        }
        let mut names: Vec<&String> = kwargs.keys().collect();
        names.sort();
        let mut kw = BTreeMap::new();
        for name in names {
            if let ASTNode::ASTConstantString(text) = &kwargs[name] {
                strings.insert(name.clone(), text.value.clone());
            }
            let (desc, key) = self.expr_keyed(&kwargs[name])?;
            kw.insert(name.clone(), Element::new(desc, Some(key)));
        }
        let mut built = Args::new(pos, kw);
        built.strings = strings;
        Ok(built)
    }

    // ── `S-Call`: chip inlining (§5.10) ───────────────────────────────

    /// Inline a chip: the body runs in a fresh environment, the arguments
    /// bind positionally then by keyword then by default, and several returns
    /// merge by `S-If` (§5.10). `recursion_limit` is a hard depth bound.
    fn inline_chip(&mut self, name: &str, args: Args, key: &ValueKey) -> Result<Desc, Reject> {
        let (chip, return_dt) = self.chip_entry(name)?;
        self.ctx().applied(Rule::SCall);
        self.push_frame()?;
        let mark = self.an.keys.enter_call(name, key.expr_index);
        let bound = self.bind_chip_inputs(&chip, &args);
        let walked = bound.and_then(|()| self.block(&chip.block));
        self.an.keys.exit_call(mark);
        let frame = self.pop_frame();
        let flow = walked?;
        // A body that returns on some paths and runs off the end on others
        // has no value on the rest, and §5.10's merge of the returns it *did*
        // take would claim one. The baseline says the same, as "Chip return
        // annotation requires every path to return".
        if flow != Flow::Terminated && !frame.returns.is_empty() {
            return Err(Reject::Unsupported {
                construct: format!(
                    "chip `{}` returns on some paths and runs off the end on others, \
                     so it has no value on the rest",
                    name
                ),
            });
        }
        let merged = merge_returns(self, frame.returns, key, "chip")?;
        // The values the body returned are the operands of the call's value,
        // and the merge rules it applied are the call's own (§5a): the rules
        // land in this node's frame, and the operands are pushed here.
        for operand in &merged.operands {
            self.an.operand(operand);
        }
        match (merged.desc, return_dt) {
            // A chip with no `return` yields `None`, whatever its
            // annotation; a chip with one keeps the descriptor its body
            // produced, which is more precise than the annotation.
            (Desc::None, Some(dt)) => {
                // Under `φ`, not `true`: `e_external` records ψ and can add
                // checks and variables, and a chip called inside a branch is
                // guarded like any other value (§2). `call_external` below
                // passes the same thing for the same rule.
                let path = self.path();
                let Walker { an, symbols, .. } = self;
                e_external(&mut an.ctx(path), name, &dt, key, symbols)
            }
            (result, _) => Ok(result),
        }
    }

    /// Bind the chip's parameters to the argument values (§5.10).
    ///
    /// A parameter takes no key and no entry of its own: its binding carries
    /// the key of the argument expression in the caller, or of the default
    /// expression evaluated here, so that a use of the parameter inside the
    /// body aliases the value that was passed (`implementation.md` §5a).
    fn bind_chip_inputs(&mut self, chip: &ASTChip, args: &Args) -> Result<(), Reject> {
        for (index, input) in chip.inputs.iter().enumerate() {
            let supplied = args
                .pos
                .get(index)
                .cloned()
                .or_else(|| args.kw.get(&input.name).cloned());
            let (value, key) = match supplied {
                Some(value) => (value, args.key(index, &input.name)),
                None => match &input.default {
                    Some(default) => {
                        let (desc, key) = self.expr_keyed(default)?;
                        (desc, Some(key))
                    }
                    None => continue,
                },
            };
            self.frame_mut().set(&input.name, value, key);
        }
        Ok(())
    }

    /// The parsed chip, shared between call sites.
    fn chip_entry(&mut self, name: &str) -> Result<(Rc<ASTChip>, Option<ZinniaType>), Reject> {
        let entry = self
            .chip(name)
            .ok_or_else(|| Reject::UnsupportedOperator {
                name: name.to_string(),
            })??;
        Ok((entry.chip.clone(), entry.return_dt.clone()))
    }

    /// `E-External` (§5.11): the result descriptor comes from the declared
    /// return type and the arguments have no shape effect.
    fn call_external(
        &mut self,
        name: &str,
        return_dt: ZinniaType,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let path = self.path();
        let Walker { an, symbols, .. } = self;
        e_external(&mut an.ctx(path), name, &return_dt, key, symbols)
    }
}

//! The expression judgment: `⟨Env, φ, Φ⟩ ⊢ e ⇓ ⟨D, Φ'⟩` (`definition.md` §4).
//!
//! One handler per expression variant of `ASTNode`, mirroring the expression
//! arms of `IRGenerator::visit`. Operators dispatch on the descriptor kind,
//! exactly as the baseline dispatches on the `Value` kind
//! (`src/helpers/value_ops.rs`): a `Composite` is a Python list or tuple, so
//! `+` concatenates and `*` repeats, while an `Array` is an ndarray, so both
//! are elementwise (`coverage.md` §5.1). An operand pair with an array on
//! either side is always elementwise, which is the baseline's "NumPy
//! dominance" rule.

use crate::ast::{
    ASTBinaryOperator, ASTCondExp, ASTGenerator, ASTGeneratorExp, ASTNode, ASTUnaryOperator,
};
use crate::phase0::desc::{
    Arm, ArrayDesc, CompositeDesc, CompositeKind, Desc, Dtype, Origin, ScalarDesc, ScalarTerm,
    ValueKey,
};
use crate::phase0::ops::{Domain, MapOut, ZipOp};
use crate::phase0::phi::VarOrigin;
use crate::phase0::rules::{
    e_arith, e_cmp, e_composite, e_composite_binop, e_const, e_const_bool, e_const_opaque, e_map,
    e_matmul, e_stack, e_zip, merge_desc, require_array, CompositeBinop, VarRange,
};
use crate::phase0::term::{BoolId, Sort};
use crate::phase0::Reject;

use super::loops::Element;
use super::{node_name, Walker};
use crate::phase0::rules::Rule;

impl Walker<'_> {
    /// Evaluate one expression node, taking the next value key in preorder.
    pub fn expr(&mut self, node: &ASTNode) -> Result<Desc, Reject> {
        Ok(self.expr_keyed(node)?.0)
    }

    /// Evaluate one expression node and return its key with its value.
    ///
    /// The callers that bind, write or return the value need its key: it is
    /// the operand the producer of the new value names (`implementation.md`
    /// §5a).
    pub fn expr_keyed(&mut self, node: &ASTNode) -> Result<(Desc, ValueKey), Reject> {
        self.spend(1)?;
        let key = self.an.keys.next_expr();
        let desc = self.expr_at(node, &key)?;
        Ok((desc, key))
    }

    /// Evaluate an expression whose key the caller already took, and record
    /// its entry in $\Delta$.
    ///
    /// The value is an operand of whatever value encloses it, and the rules
    /// that run while it is evaluated are its own: [`Walker::in_value`] opens
    /// a frame that collects both (`implementation.md` §5a).
    pub fn expr_at(&mut self, node: &ASTNode, key: &ValueKey) -> Result<Desc, Reject> {
        self.an.operand(key);
        let (desc, frame) = self.in_value(key, |w| w.evaluate(node, key))?;
        let producer = frame.into_producer(&desc);
        // The key names this occurrence of the expression, which is executed
        // under the path condition in force; a rule's own output already
        // carries it, an alias would otherwise carry the ψ of the value it
        // names (`implementation.md` §5a, defect 14).
        let desc = super::created_at(desc, self.path());
        self.record(key.clone(), None, &desc, producer);
        Ok(desc)
    }

    /// The expression judgment itself: one arm per node.
    fn evaluate(&mut self, node: &ASTNode, key: &ValueKey) -> Result<Desc, Reject> {
        match node {
            ASTNode::ASTConstantInteger(n) => Ok(e_const(&mut self.ctx(), n.value)),
            ASTNode::ASTConstantBoolean(n) => Ok(e_const_bool(&mut self.ctx(), n.value)),
            ASTNode::ASTConstantFloat(_) => Ok(e_const_opaque(&mut self.ctx(), Dtype::Float)),
            ASTNode::ASTConstantComplex(_) => Ok(e_const_opaque(&mut self.ctx(), Dtype::Complex)),
            ASTNode::ASTConstantString(_) => Ok(e_const_opaque(&mut self.ctx(), Dtype::String)),
            ASTNode::ASTConstantNone(_) => Ok(Desc::None),
            // A name is the value it was bound to, under a new key: the
            // producer is an alias (`implementation.md` §5a). A dtype or
            // module name has no key and no entry, and is opaque.
            ASTNode::ASTLoad(n) => match self.frame().binding(&n.name) {
                Some(binding) => {
                    let (desc, bound) = (binding.desc.clone(), binding.key.clone());
                    if let Some(bound) = bound {
                        self.an.alias(&bound);
                    }
                    Ok(desc)
                }
                None => Err(self.frame().missing(&n.name)),
            },
            ASTNode::ASTBinaryOperator(n) => self.binary(n, key),
            ASTNode::ASTUnaryOperator(n) => self.unary(n, key),
            ASTNode::ASTSubscriptExp(n) => {
                let value = self.expr(&n.val)?;
                let indices = self.slice_indices(&n.slicing)?;
                self.subscript(&value, &indices, key)
            }
            ASTNode::ASTNamedAttribute(n) => self.named_attribute(n, key),
            ASTNode::ASTExprAttribute(n) => self.expr_attribute(n, key),
            ASTNode::ASTSquareBrackets(n) => self.composite(&n.values, CompositeKind::List),
            ASTNode::ASTParenthesis(n) => self.composite(&n.values, CompositeKind::Tuple),
            ASTNode::ASTGeneratorExp(n) => self.comprehension(n, key),
            ASTNode::ASTCondExp(n) => self.cond_exp(n, key),
            // f-strings have no shape effect; the baseline builds a string
            // value from them (`coverage.md` §3).
            ASTNode::ASTJoinedStr(n) => {
                for value in &n.values {
                    self.expr(value)?;
                }
                Ok(e_const_opaque(&mut self.ctx(), Dtype::String))
            }
            ASTNode::ASTFormattedValue(n) => {
                self.expr(&n.value)?;
                Ok(e_const_opaque(&mut self.ctx(), Dtype::String))
            }
            other => Err(Reject::Unsupported {
                construct: format!("{} in expression position", node_name(other)),
            }),
        }
    }

    // ── Composites (`E-Composite`, §5.11) ─────────────────────────────

    /// `[e_1, …, e_k]` and `(e_1, …, e_k)`, with starred elements of known
    /// length flattened into place.
    fn composite(&mut self, values: &[ASTNode], kind: CompositeKind) -> Result<Desc, Reject> {
        let mut elems = Vec::with_capacity(values.len());
        for value in values {
            match value {
                ASTNode::ASTStarredExpr(starred) => {
                    let (inner, source) = self.expr_keyed(&starred.value)?;
                    let mut spread = Vec::new();
                    self.spread(&inner, Some(&source), &mut spread)?;
                    elems.extend(spread.into_iter().map(|element| element.desc));
                }
                other => elems.push(self.expr(other)?),
            }
        }
        self.ctx().applied(Rule::EComposite);
        Ok(e_composite(kind, elems))
    }

    /// Flatten a starred element into a composite literal or an argument
    /// list. An array spreads into its rank-1 sub-descriptors, as
    /// `S-For-Elems` iterates it.
    pub fn spread(
        &mut self,
        inner: &Desc,
        source: Option<&ValueKey>,
        out: &mut Vec<Element>,
    ) -> Result<(), Reject> {
        match inner {
            Desc::Composite(c) => out.extend(
                c.elems
                    .iter()
                    .map(|desc| Element::new(desc.clone(), source.cloned())),
            ),
            Desc::Array(_) => {
                let elements = self.iterable_elements(inner, source, "starred expression")?;
                out.extend(elements);
            }
            other => out.push(Element::new(other.clone(), source.cloned())),
        }
        Ok(())
    }

    // ── Operators ─────────────────────────────────────────────────────

    fn binary(&mut self, n: &ASTBinaryOperator, key: &ValueKey) -> Result<Desc, Reject> {
        let lhs = self.expr(&n.lhs)?;
        let rhs = self.expr(&n.rhs)?;
        match n.operator.as_str() {
            // `x in v` scans a static container; the result is one Boolean
            // and no shape changes (`value_ops.rs`'s `apply_membership_op`).
            "in" | "not_in" => {
                let tt = self.ctx().tt();
                Ok(self.ctx().data_scalar(
                    Rule::ECmp,
                    Dtype::Bool,
                    "member",
                    key,
                    VarRange::default(),
                    tt,
                ))
            }
            "mat_mul" => {
                let a = self.as_array(&lhs, Rule::EMatmul)?;
                let b = self.as_array(&rhs, Rule::EMatmul)?;
                e_matmul(&mut self.ctx(), &a, &b)
            }
            operator => {
                let op = ZipOp::from_ast(operator).ok_or_else(|| Reject::UnsupportedOperator {
                    name: operator.to_string(),
                })?;
                self.apply_zip(op, &lhs, &rhs, key)
            }
        }
    }

    /// `a op b` for every elementwise binary operator: `E-Zip` (§5.3) when an
    /// array is involved, the composite rules of §5.11 when only composites
    /// are, and `E-Arith` / `E-Cmp` (§5.2) on two scalars.
    pub fn apply_zip(
        &mut self,
        op: ZipOp,
        lhs: &Desc,
        rhs: &Desc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        if lhs.as_array().is_some() || rhs.as_array().is_some() {
            return self.array_zip(op, lhs, rhs);
        }
        if lhs.as_composite().is_some() || rhs.as_composite().is_some() {
            return self.composite_zip(op, lhs, rhs, key);
        }
        self.scalar_zip(op, lhs, rhs, key)
    }

    /// `E-Zip` (§5.3) with both operands seen as arrays.
    fn array_zip(&mut self, op: ZipOp, lhs: &Desc, rhs: &Desc) -> Result<Desc, Reject> {
        let a = self.as_array(lhs, Rule::EZip)?;
        let b = self.as_array(rhs, Rule::EZip)?;
        let out = op
            .out_dtype(a.dtype, b.dtype)
            .ok_or_else(|| Reject::Premise {
                rule: Rule::EZip,
                detail: format!("on {} and {}", a.dtype.as_str(), b.dtype.as_str()),
            })?;
        let (da, db) = (Desc::Array(a), Desc::Array(b));
        e_zip(&mut self.ctx(), &da, &db, out)
    }

    /// The composite forms of `E-Composite-Binop` (§5.11) and the elementwise
    /// fallback the baseline applies to two lists of equal length.
    fn composite_zip(
        &mut self,
        op: ZipOp,
        lhs: &Desc,
        rhs: &Desc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let (left, right) = (lhs.as_composite().cloned(), rhs.as_composite().cloned());
        if let Some(sequence) = self.sequence_binop(op, lhs, rhs, &left, &right, key) {
            return sequence;
        }
        match (left, right) {
            // Python sequence comparison collapses to one Boolean (§5.11).
            // Two *numeric* composites are ndarrays as far as the baseline is
            // concerned, so they compare elementwise instead
            // (`src/helpers/value_ops.rs`, the `is_numeric_composite` arm).
            (Some(v), Some(w))
                if op.cmp().is_some()
                    && !(is_numeric(&Desc::Composite(v.clone()))
                        && is_numeric(&Desc::Composite(w.clone()))) =>
            {
                e_composite_binop(&mut self.ctx(), CompositeBinop::Equal(&w), &v, key, "cmp")
            }
            (Some(v), Some(w)) if v.elems.len() == w.elems.len() => {
                let mut elems = Vec::with_capacity(v.elems.len());
                for (a, b) in v.elems.iter().zip(w.elems.iter()) {
                    elems.push(self.apply_zip(op, a, b, key)?);
                }
                Ok(e_composite(CompositeKind::List, elems))
            }
            // Different lengths broadcast as arrays, which is what the
            // baseline does after materialising both sides.
            (Some(_), Some(_)) => self.array_zip(op, lhs, rhs),
            (Some(v), None) => {
                let mut elems = Vec::with_capacity(v.elems.len());
                for a in &v.elems {
                    elems.push(self.apply_zip(op, a, rhs, key)?);
                }
                Ok(e_composite(CompositeKind::List, elems))
            }
            (None, Some(w)) => {
                let mut elems = Vec::with_capacity(w.elems.len());
                for b in &w.elems {
                    elems.push(self.apply_zip(op, lhs, b, key)?);
                }
                Ok(e_composite(CompositeKind::List, elems))
            }
            (None, None) => unreachable!("one of the two operands is a composite"),
        }
    }

    /// The two operators §5.11 gives a sequence meaning: `+` concatenates two
    /// composites and `*` by a constant repeats one. `None` when neither
    /// applies, which sends the caller to the elementwise arms.
    fn sequence_binop(
        &mut self,
        op: ZipOp,
        lhs: &Desc,
        rhs: &Desc,
        left: &Option<CompositeDesc>,
        right: &Option<CompositeDesc>,
        key: &ValueKey,
    ) -> Option<Result<Desc, Reject>> {
        if op == ZipOp::Add {
            if let (Some(v), Some(w)) = (left, right) {
                return Some(e_composite_binop(
                    &mut self.ctx(),
                    CompositeBinop::Concat(w),
                    v,
                    key,
                    "concat",
                ));
            }
        }
        if op != ZipOp::Mul {
            return None;
        }
        let v = left.as_ref().xor(right.as_ref())?;
        // `[x] * n` with a runtime `n` is a value-sized composite, and §3.3
        // makes a composite's length a compile-time constant. The baseline
        // keeps one element and reads the rest as `None` (`coverage.md` §9).
        let Some(n) = self.constant_repeat(lhs, rhs) else {
            return Some(Err(Reject::ValueSizedWithoutMaximum {
                construct: "`v * n` with a runtime count: a composite's length is a \
                            compile-time constant (§3.3), so no declared maximum makes \
                            this one a length — build an array instead (`np.zeros(n)`), \
                            whose length may be value-sized (§5.9)"
                    .to_string(),
            }));
        };
        Some(e_composite_binop(
            &mut self.ctx(),
            CompositeBinop::Repeat(n),
            v,
            key,
            "repeat",
        ))
    }

    /// The constant repetition count of `v * n` or `n * v` (§5.11).
    fn constant_repeat(&self, lhs: &Desc, rhs: &Desc) -> Option<i64> {
        let scalar = match (lhs.as_composite(), rhs.as_composite()) {
            (Some(_), None) => rhs.as_scalar()?,
            (None, Some(_)) => lhs.as_scalar()?,
            _ => return None,
        };
        let term = scalar.term.as_int()?;
        i64::try_from(self.an.terms.constant_value(term)?).ok()
    }

    /// `E-Arith` and `E-Cmp` (§5.2), plus the Boolean connectives, on two
    /// scalars.
    fn scalar_zip(
        &mut self,
        op: ZipOp,
        lhs: &Desc,
        rhs: &Desc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let (a, b) = match (lhs.as_scalar(), rhs.as_scalar()) {
            (Some(a), Some(b)) => (a.clone(), b.clone()),
            _ => {
                return Err(Reject::Unsupported {
                    construct: format!(
                        "binary operator on a {} and a {}",
                        lhs.kind_str(),
                        rhs.kind_str()
                    ),
                })
            }
        };
        if op == ZipOp::Div {
            // Python's `/` is real division. It never contributes a length,
            // so the descriptor carries no term (§3.2).
            let dtype = if a.dtype == Dtype::Complex || b.dtype == Dtype::Complex {
                Dtype::Complex
            } else {
                Dtype::Float
            };
            return Ok(e_const_opaque(&mut self.ctx(), dtype));
        }
        if let Some(arith) = op.arith() {
            return e_arith(&mut self.ctx(), arith, &a, &b, key, "arith");
        }
        if let Some(cmp) = op.cmp() {
            return Ok(e_cmp(&mut self.ctx(), cmp, &a, &b, key, "cmp"));
        }
        if let Some(desc) = self.shape_extremum(op, &a, &b) {
            return Ok(desc);
        }
        self.scalar_connective(op, &a, &b, key)
    }

    /// `min(a, b)` and `max(a, b)` on two shape scalars: §3.1 has `min` and
    /// `max` as terms, so the result stays a shape scalar. This is what keeps
    /// the trip count of `for i in zip(x, y)` — which the front end desugars
    /// to `range(min(len(x), len(y)))` (`coverage.md` §2) — a constant.
    fn shape_extremum(&mut self, op: ZipOp, a: &ScalarDesc, b: &ScalarDesc) -> Option<Desc> {
        if !matches!(op, ZipOp::Min | ZipOp::Max) {
            return None;
        }
        if a.origin != Origin::Shape || b.origin != Origin::Shape {
            return None;
        }
        let (ta, tb) = (a.term.as_int()?, b.term.as_int()?);
        let term = match op {
            ZipOp::Min => self.an.terms.min(ta, tb),
            _ => self.an.terms.max(ta, tb),
        };
        let dtype = Dtype::promote(a.dtype, b.dtype)?;
        let tt = self.ctx().tt();
        Some(self.ctx().scalar(
            Rule::EArithShape,
            dtype,
            ScalarTerm::Int(term),
            Origin::Shape,
            tt,
        ))
    }

    /// `and`, `or`, `xor` and the bitwise operators on two scalars.
    ///
    /// On Booleans of shape origin the result is the Boolean term itself, so
    /// that a compound `while` guard or `if` condition stays a shape term and
    /// phase 1 can decide it. Everything else is a fresh data scalar: §3.2
    /// drops the arithmetic relation as soon as element contents enter.
    fn scalar_connective(
        &mut self,
        op: ZipOp,
        a: &ScalarDesc,
        b: &ScalarDesc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        if matches!(op, ZipOp::And | ZipOp::Or | ZipOp::Xor) {
            let (ta, origin_a) = self.bool_term(a, key, "cond");
            let (tb, origin_b) = self.bool_term(b, key, "cond");
            if Origin::join(origin_a, origin_b) == Origin::Shape {
                let term = self.connective_term(op, ta, tb);
                let tt = self.ctx().tt();
                return Ok(self.ctx().scalar(
                    Rule::ECmp,
                    Dtype::Bool,
                    ScalarTerm::Bool(term),
                    Origin::Shape,
                    tt,
                ));
            }
            let tt = self.ctx().tt();
            return Ok(self.ctx().data_scalar(
                Rule::ECmp,
                Dtype::Bool,
                "logic",
                key,
                VarRange::default(),
                tt,
            ));
        }
        let dtype = Dtype::promote(a.dtype, b.dtype).unwrap_or(Dtype::I64);
        let range = VarRange::of_dtype(&mut self.an.terms, dtype);
        let tt = self.ctx().tt();
        Ok(self
            .ctx()
            .data_scalar(Rule::EArithData, dtype, "bits", key, range, tt))
    }

    fn connective_term(&mut self, op: ZipOp, a: BoolId, b: BoolId) -> BoolId {
        match op {
            ZipOp::And => self.an.terms.and(a, b),
            ZipOp::Or => self.an.terms.or(a, b),
            _ => {
                let both = self.an.terms.and(a, b);
                let either = self.an.terms.or(a, b);
                let not_both = self.an.terms.not(both);
                self.an.terms.and(either, not_both)
            }
        }
    }

    fn unary(&mut self, n: &ASTUnaryOperator, key: &ValueKey) -> Result<Desc, Reject> {
        let (operand, operand_key) = self.expr_keyed(&n.operand)?;
        match n.operator.as_str() {
            // `+e` is the operand itself, under this node's key.
            "uadd" => {
                self.an.alias(&operand_key);
                Ok(operand)
            }
            "usub" => self.negate(&operand, key),
            "invert" => self.apply_map(MapOut::Same, Domain::All, &operand, key, "invert"),
            "not" => self.logical_not(&operand, key),
            other => Err(Reject::UnsupportedOperator {
                name: format!("unary {}", other),
            }),
        }
    }

    /// `-e`. On a shape scalar the result is the shape term `-t`, which is
    /// what keeps `range(n - 1, -1, -1)` and `a[-k]` exact (§5.2's
    /// `E-Arith-Shape` with the constant factor `-1`).
    fn negate(&mut self, operand: &Desc, key: &ValueKey) -> Result<Desc, Reject> {
        if let Desc::Scalar(s) = operand {
            if s.origin == Origin::Shape {
                if let ScalarTerm::Int(t) = s.term {
                    let dtype = s.dtype;
                    let negated = self.an.terms.mul_const(-1, t);
                    let term = if dtype.wraps() {
                        self.an.terms.wrap64(negated)
                    } else {
                        negated
                    };
                    let tt = self.ctx().tt();
                    return Ok(self.ctx().scalar(
                        Rule::EArithShape,
                        dtype,
                        ScalarTerm::Int(term),
                        Origin::Shape,
                        tt,
                    ));
                }
            }
        }
        self.apply_map(MapOut::Same, Domain::All, operand, key, "neg")
    }

    /// `not e`: the negated Boolean term when the operand has shape origin,
    /// so that `while not done` stays decidable, and `E-Map` otherwise.
    fn logical_not(&mut self, operand: &Desc, key: &ValueKey) -> Result<Desc, Reject> {
        if let Desc::Scalar(s) = operand {
            let s = s.clone();
            let (term, origin) = self.bool_term(&s, key, "cond");
            if origin == Origin::Shape {
                let negated = self.an.terms.not(term);
                let tt = self.ctx().tt();
                return Ok(self.ctx().scalar(
                    Rule::ECmp,
                    Dtype::Bool,
                    ScalarTerm::Bool(negated),
                    Origin::Shape,
                    tt,
                ));
            }
        }
        self.apply_map(MapOut::Bool, Domain::All, operand, key, "not")
    }

    /// `E-Map` (§5.3) on an array, the scalar form of the same operator, or
    /// the elementwise walk of a composite.
    pub fn apply_map(
        &mut self,
        out: MapOut,
        domain: Domain,
        operand: &Desc,
        key: &ValueKey,
        base: &str,
    ) -> Result<Desc, Reject> {
        match operand {
            Desc::Array(a) => {
                let dtype = map_dtype(out, a.dtype);
                let defcond = self.domain_condition(domain, operand, key);
                e_map(&mut self.ctx(), operand, dtype, defcond)
            }
            Desc::Scalar(s) => {
                let dtype = map_dtype(out, s.dtype);
                let range = VarRange::of_dtype(&mut self.an.terms, dtype);
                let defcond = self.domain_condition(domain, operand, key);
                Ok(self
                    .ctx()
                    .data_scalar(Rule::EMap, dtype, base, key, range, defcond))
            }
            Desc::Composite(c) => {
                let elems: Vec<Desc> = c.elems.clone();
                let kind = c.kind;
                let mut mapped = Vec::with_capacity(elems.len());
                for element in &elems {
                    mapped.push(self.apply_map(out, domain, element, key, base)?);
                }
                Ok(e_composite(kind, mapped))
            }
            other => Err(Reject::Unsupported {
                construct: format!("elementwise unary on a {}", other.kind_str()),
            }),
        }
    }

    /// `defcond_f`, the domain condition of an elementwise unary (§5.3).
    ///
    /// A scalar of shape origin carries a term built from the parameters in
    /// $S$ (§3.2), so the condition is that term's and phase 1 can decide it.
    /// Everything else — an array, whose elements $\Phi$ does not model, and
    /// a scalar with no term — gets an unconstrained Boolean, exactly as
    /// `E-Take`'s `inrange` does (§5.4).
    fn domain_condition(&mut self, domain: Domain, operand: &Desc, key: &ValueKey) -> BoolId {
        if domain == Domain::All {
            return self.an.terms.tt();
        }
        let term = operand
            .as_scalar()
            .filter(|s| s.origin == Origin::Shape)
            .and_then(|s| s.term.as_int());
        let Some(t) = term else {
            return self.ctx().definedness("in_domain", key);
        };
        match domain {
            Domain::All => self.an.terms.tt(),
            Domain::NonNegative => {
                let zero = self.an.terms.constant(0);
                self.an.terms.le(zero, t)
            }
            Domain::Positive => {
                let zero = self.an.terms.constant(0);
                self.an.terms.lt(zero, t)
            }
            Domain::UnitInterval => {
                let low = self.an.terms.constant(-1);
                let high = self.an.terms.constant(1);
                let above = self.an.terms.le(low, t);
                let below = self.an.terms.le(t, high);
                self.an.terms.and(above, below)
            }
        }
    }

    // ── `E-CondExp` and `E-Comp` (§5.11) ──────────────────────────────

    /// `a if c else b`: both arms evaluated under their own path condition
    /// and merged exactly as in `S-If` (§5.11, §6.1).
    ///
    /// Each arm takes its keys inside its own [`Arm`] component of the
    /// statement path, exactly as the two arms of an `if` statement do, so
    /// the arms cannot collide and the rest of the statement is keyed the
    /// same whether or not a statically decided condition skipped an arm.
    fn cond_exp(&mut self, n: &ASTCondExp, key: &ValueKey) -> Result<Desc, Reject> {
        let cond = self.expr(&n.cond)?;
        let b = self.condition(&cond, key)?;
        let not_b = self.an.terms.not(b);
        match self.cond_exp_arms(n, b, not_b)? {
            (Some(taken), None) | (None, Some(taken)) => Ok(taken),
            (Some(then_desc), Some(else_desc)) => {
                merge_desc(&mut self.ctx(), b, &then_desc, &else_desc, key, "condexp")
            }
            (None, None) => unreachable!("a condition is true, false, or neither"),
        }
    }

    /// The two arms of a conditional expression. A statically decided
    /// condition evaluates only the arm that runs, as `visit_cond` does for
    /// a statement (`coverage.md` §3).
    #[allow(clippy::type_complexity)]
    fn cond_exp_arms(
        &mut self,
        n: &ASTCondExp,
        b: BoolId,
        not_b: BoolId,
    ) -> Result<(Option<Desc>, Option<Desc>), Reject> {
        let mut then_desc = None;
        let mut else_desc = None;
        if !self.an.terms.is_false(b) {
            then_desc = Some(self.in_arm(Arm::Then, |w| w.under_branch(b, |w| w.expr(&n.t_expr)))?);
        }
        if !self.an.terms.is_true(b) {
            else_desc =
                Some(self.in_arm(Arm::Else, |w| w.under_branch(not_b, |w| w.expr(&n.f_expr)))?);
        }
        Ok((then_desc, else_desc))
    }

    /// Run `body` inside one arm of the key cursor's statement path.
    fn in_arm<T>(
        &mut self,
        which: Arm,
        body: impl FnOnce(&mut Self) -> Result<T, Reject>,
    ) -> Result<T, Reject> {
        let mark = self.an.keys.enter_arm(which);
        let out = body(self);
        self.an.keys.exit_arm(mark);
        out
    }

    /// `E-Comp` (§5.11): `[e for x in v if c]`, expanded statically over `v`
    /// as in `S-For-Elems`.
    ///
    /// Without a data-dependent `if` the result is a composite of the element
    /// descriptors. With one the length is unknown at compile time, so the
    /// result is an array whose length is a count variable — Python
    /// semantics; the baseline keeps every element (`coverage.md` §9).
    fn comprehension(&mut self, n: &ASTGeneratorExp, key: &ValueKey) -> Result<Desc, Reject> {
        let mut elems = Vec::new();
        let mut filtered = false;
        self.expand_generators(&n.generators, 0, &n.elt, &mut elems, &mut filtered)?;
        let kind = if n.kind == "list" {
            CompositeKind::List
        } else {
            CompositeKind::Tuple
        };
        self.ctx().applied(Rule::EComp);
        if !filtered {
            return Ok(e_composite(kind, elems));
        }
        self.counted_composite(elems, key)
    }

    /// The generator nest of `E-Comp`, expanded left to right.
    fn expand_generators(
        &mut self,
        generators: &[ASTGenerator],
        index: usize,
        elt: &ASTNode,
        out: &mut Vec<Desc>,
        filtered: &mut bool,
    ) -> Result<(), Reject> {
        let Some(generator) = generators.get(index) else {
            out.push(self.expr(elt)?);
            return Ok(());
        };
        let (iterable, source) = self.expr_keyed(&generator.iter_expr)?;
        let elements = self.iterable_elements(&iterable, Some(&source), "comprehension")?;
        let mark = self.an.keys.enter_loop();
        for (iteration, element) in elements.iter().enumerate() {
            self.an.keys.set_iteration(iteration as u32);
            self.assign_target(&generator.target, element.desc.clone(), element.key.as_ref())?;
            match self.generator_filter(&generator.ifs)? {
                Filter::Excluded => continue,
                Filter::Dynamic => *filtered = true,
                Filter::Kept => {}
            }
            self.expand_generators(generators, index + 1, elt, out, filtered)?;
        }
        self.an.keys.exit_loop(mark);
        Ok(())
    }

    /// Evaluate a generator's `if` clauses. A statically false one drops the
    /// element; a data-dependent one makes the result's length a count.
    fn generator_filter(&mut self, ifs: &[ASTNode]) -> Result<Filter, Reject> {
        let mut verdict = Filter::Kept;
        for condition in ifs {
            let key = self.an.keys.next_expr();
            let desc = self.expr_at(condition, &key)?;
            let term = self.condition(&desc, &key)?;
            if self.an.terms.is_false(term) {
                return Ok(Filter::Excluded);
            }
            if !self.an.terms.is_true(term) {
                verdict = Filter::Dynamic;
            }
        }
        Ok(verdict)
    }

    /// The result of a filtered comprehension: an array of the elements'
    /// common shape whose leading extent is a fresh count `0 ≤ c ≤ k`
    /// (§5.5's `Count`, keyed by the comprehension itself).
    fn counted_composite(&mut self, elems: Vec<Desc>, key: &ValueKey) -> Result<Desc, Reject> {
        if elems.is_empty() {
            return Ok(e_composite(CompositeKind::List, elems));
        }
        let stacked = self.stack_descs(&elems, Rule::EComp)?;
        let array = require_array(Rule::EComp, &stacked)?.clone();
        let upper = array.dims[0];
        let count = self
            .an
            .phi
            .count_var(&mut self.an.terms, key, upper, "c_comp");
        let mut dims = array.dims.clone();
        dims[0] = self.an.terms.var(count);
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EComp, array.dtype, dims, tt))
    }

    // ── Descriptor coercions ──────────────────────────────────────────

    /// The array view of a descriptor: an array is itself, a scalar is rank
    /// 0, and a composite is `E-Stack` over its elements, which is what
    /// `np.asarray` does and what the baseline's `to_static_array` boundary
    /// does before an elementwise operation (`coverage.md` §5.1).
    pub fn as_array(&mut self, d: &Desc, rule: Rule) -> Result<ArrayDesc, Reject> {
        match d {
            Desc::Array(a) => Ok(a.clone()),
            Desc::Scalar(s) => Ok(ArrayDesc {
                dtype: s.dtype,
                dims: Vec::new(),
                defcond: s.defcond,
                psi: s.psi,
            }),
            Desc::Composite(c) => {
                let elems = c.elems.clone();
                let stacked = self.stack_descs(&elems, rule)?;
                Ok(require_array(rule, &stacked)?.clone())
            }
            other => Err(Reject::Premise {
                rule,
                detail: format!("applied to a {}", other.kind_str()),
            }),
        }
    }

    /// `E-Stack` over a list of descriptors of the same shape: the
    /// composite-to-array conversion, and `np.array([a, b])`.
    pub fn stack_descs(&mut self, elems: &[Desc], rule: Rule) -> Result<Desc, Reject> {
        if elems.is_empty() {
            let zero = self.an.terms.constant(0);
            let tt = self.ctx().tt();
            return Ok(self.ctx().array(rule, Dtype::I64, vec![zero], tt));
        }
        let mut parts = Vec::with_capacity(elems.len());
        for element in elems {
            parts.push(self.as_array(element, rule)?);
        }
        let dtype = parts
            .iter()
            .map(|p| p.dtype)
            .reduce(|a, b| Dtype::promote(a, b).unwrap_or(a))
            .unwrap_or(Dtype::I64);
        let refs: Vec<&ArrayDesc> = parts.iter().collect();
        let stacked = e_stack(&mut self.ctx(), &refs, 0)?;
        let mut array = require_array(rule, &stacked)?.clone();
        array.dtype = dtype;
        Ok(Desc::Array(array))
    }

    /// `to_scalar_bool`: the Boolean term a value contributes as a condition
    /// (`src/helpers/value_ops.rs`).
    pub fn condition(&mut self, d: &Desc, key: &ValueKey) -> Result<BoolId, Reject> {
        match d {
            Desc::Scalar(s) => {
                let s = s.clone();
                Ok(self.bool_term(&s, key, "cond").0)
            }
            // An array or composite condition is the conjunction of its
            // elements, whose values Φ does not model, so it is one fresh
            // Boolean variable.
            Desc::Array(_) | Desc::Composite(_) => Ok(self.fresh_bool(key, "cond")),
            other => Err(Reject::Unsupported {
                construct: format!("a {} used as a condition", other.kind_str()),
            }),
        }
    }

    /// The Boolean term of a scalar, with its origin: a Boolean scalar
    /// carries one already, an integer scalar of shape origin contributes
    /// `t ≠ 0`, and anything else is a fresh data Boolean (§3.2).
    fn bool_term(&mut self, s: &ScalarDesc, key: &ValueKey, base: &str) -> (BoolId, Origin) {
        if let ScalarTerm::Bool(b) = s.term {
            return (b, s.origin);
        }
        if s.origin == Origin::Shape {
            if let ScalarTerm::Int(t) = s.term {
                let zero = self.an.terms.constant(0);
                let is_zero = self.an.terms.eq(t, zero);
                return (self.an.terms.not(is_zero), Origin::Shape);
            }
        }
        (self.fresh_bool(key, base), Origin::Data)
    }

    /// A fresh unconstrained data Boolean of `Φ` (§3.4): the value at `key`
    /// read as a truth value, which is what a branch condition and a mask
    /// atom are.
    pub fn fresh_bool(&mut self, key: &ValueKey, base: &str) -> BoolId {
        let var = self.an.phi.fresh_var(
            &mut self.an.terms,
            base,
            Sort::Bool,
            VarOrigin::DataBool { value: key.clone() },
            None,
            None,
        );
        self.an.terms.bool_var(var)
    }
}

/// Whether a generator's `if` clauses keep the element.
enum Filter {
    Kept,
    Excluded,
    Dynamic,
}

/// The number of leaf descriptors in a value, which is what merging it
/// costs ([`Walker::spend`]).
pub(super) fn leaves(d: &Desc) -> u64 {
    match d {
        Desc::Composite(c) => c.elems.iter().map(leaves).sum::<u64>().max(1),
        _ => 1,
    }
}

/// Whether every leaf of a descriptor is a number or an array of numbers.
///
/// The baseline splits Python sequence semantics from ndarray semantics on
/// exactly this test (`src/helpers/value_ops.rs`'s `is_numeric_composite`).
fn is_numeric(d: &Desc) -> bool {
    match d {
        Desc::Array(_) => true,
        Desc::Scalar(s) => !matches!(s.dtype, Dtype::String | Dtype::None),
        Desc::Composite(c) => c.elems.iter().all(is_numeric),
        _ => false,
    }
}

/// The result dtype of an elementwise unary operator.
fn map_dtype(out: MapOut, input: Dtype) -> Dtype {
    match out {
        MapOut::Same => input,
        MapOut::Float => Dtype::Float,
        MapOut::Bool => Dtype::Bool,
        MapOut::RealPart => match input {
            Dtype::Complex => Dtype::Float,
            other => other,
        },
    }
}

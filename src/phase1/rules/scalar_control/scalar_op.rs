//! `SCALAR_OP` (`revision/phase1/rules/scalar_control/scalar_op.md`).
//!
//! Computes one scalar value of Δ from the wires of its scalar operands:
//! arithmetic, comparison, min and max, the Boolean connectives, unary minus
//! and abs, casts between integer dtypes and Bool, and an integer or Boolean
//! literal. `Represent(v, Wire[τ_v])` and `Fetch(v, [()], g)` are one
//! requirement, and for a shape-origin `v` the row is also that of `v`'s term
//! (`rules/README.md` §3.2).
//!
//! Arithmetic follows the dtype: `Idx`, a Python `int`, is exact; a
//! fixed-width dtype wraps at `2^{w_τ}`. A shape-origin value carries the
//! term phase 0 built, so its emitter is exactly `term_to_wires.md`'s row on
//! that term's top node — which is what the rule file's "Exact rows" says.
//! A data-origin value carries a fresh variable instead, so its row comes
//! from the operator the AST node names.

use crate::phase0::desc::{Dtype, Origin, ScalarTerm, ValueKey};
use crate::phase0::interval::Interval;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::{bits_for, width_of, Builder};
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::graph::Target;
use crate::phase1::interface::{Batch, Family, Interface};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ScalarOp;

impl Rule for ScalarOp {
    fn id(&self) -> RuleId {
        RuleId::ScalarOp
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Target::Value(key) = &request.target else {
            return Ok(Vec::new());
        };
        let Some(entry) = builder.entry(key) else {
            return Ok(Vec::new());
        };
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return Ok(Vec::new());
        };
        // source_pattern: E-Const, E-Arith-Shape, E-Arith-Data,
        // E-MinMax-Shape (recorded as E-Arith-Shape) and E-Cmp on scalars.
        if !rules.iter().any(|rule| {
            matches!(
                rule,
                Phase0Rule::EConst
                    | Phase0Rule::EArithShape
                    | Phase0Rule::EArithData
                    | Phase0Rule::ECmp
            )
        }) {
            return Ok(Vec::new());
        }
        let Some(scalar) = entry.desc.as_scalar() else {
            return Ok(Vec::new());
        };
        // P2: both operands are scalars of an admitted dtype. Complex and
        // String have no wire encoding the library states (Open 1); Float
        // has the backend's, and its rows are `fop⟨name⟩`s.
        let (dtype, origin, term, psi) = (scalar.dtype, scalar.origin, scalar.term, scalar.psi);
        if !super::super::sources::has_encoding(dtype) {
            return Ok(Vec::new());
        }
        let operands: Vec<ValueKey> = operands.clone();

        // P1 (variant): CONST t_v under ψ_v gives `/const`, in which the
        // value is substituted at every use and no port is raised.
        let mut probe = Candidate::new(RuleId::ScalarOp, Some("/const"), request.epoch);
        let under = builder.under(psi);
        let constant = match term {
            ScalarTerm::Int(t) => {
                builder.ask_for(&mut probe, Question::Const(t), request.epoch, under)
            }
            ScalarTerm::Bool(b) => {
                builder.ask_for(&mut probe, Question::ConstBool(b), request.epoch, under)
            }
            ScalarTerm::Opaque => Verdict::NotConstant,
        };
        if let Verdict::Constant(_) | Verdict::ConstantBool(_) = constant {
            let mut candidate = probe.clone().output(
                request.target.clone(),
                Interface::Wire(dtype),
                request.condition,
            );
            if let Some(value) = constant.value() {
                candidate = candidate.parameter("n", value);
            }
            candidate.matched.push(key.clone());
            return Ok(vec![candidate]);
        }

        // `/ravel`: `np.ravel_multi_index(index, dims, order)` with a
        // run-time `dims`, whose operands are two arrays and not scalars.
        if let Some(out) = self.ravel(builder, request, key, &operands, dtype, &probe)? {
            return Ok(out);
        }

        // `/op`: one port per scalar operand whose value is not a literal.
        let mut candidate = Candidate::new(RuleId::ScalarOp, Some("/op"), request.epoch);
        candidate.matched.push(key.clone());
        candidate.answers = probe.answers.clone();
        let mut intervals: Vec<Interval> = Vec::new();
        // The dtype the row computes in: the operands', not the result's.
        // `x < y` on two Float scalars is a Bool whose row is `fop⟨lt_f⟩`.
        let mut operand_dtype = dtype;
        for operand in &operands {
            let Some(operand_scalar) = builder.scalar(operand) else {
                continue;
            };
            let this = operand_scalar.dtype;
            let operand_term = operand_scalar.term;
            if !super::super::sources::has_encoding(this) {
                return Ok(Vec::new());
            }
            operand_dtype = Dtype::promote(operand_dtype, this).unwrap_or(operand_dtype);
            intervals.push(match operand_term {
                ScalarTerm::Int(t) => builder.interval(t),
                ScalarTerm::Bool(_) => Interval::new(Some(0), Some(1)),
                ScalarTerm::Opaque => Interval::UNBOUNDED,
            });
            // A literal operand is wired in as the constant of the row.
            if let ScalarTerm::Int(t) = operand_term {
                if builder.terms.constant_value(t).is_some() {
                    continue;
                }
            }
            // Pre: ψ_v ⇒ ψ_{s_i}, because v's operands were computed on v's
            // path. The premise is checked, not assumed (§4).
            let psi = builder.psi(operand);
            let condition =
                builder.child_under_g(&mut candidate, request, psi, &operand.encode())?;
            let row = builder.child(
                request,
                Target::Value(operand.clone()),
                Family::Wire,
                condition,
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }

        // emitter: a shape-origin value is its term's top node; a data-origin
        // value is the operator's row.
        let primitives = match (origin, term) {
            (Origin::Shape, ScalarTerm::Int(t)) => super::node_gadget(builder, t.index() as u32),
            (Origin::Shape, ScalarTerm::Bool(b)) => super::node_gadget(builder, b.index() as u32),
            _ => {
                let operator = builder
                    .node(key)
                    .and_then(|node| operator_of(node))
                    .unwrap_or_default();
                match data_rows(builder, &operator, dtype, operand_dtype, &intervals, term) {
                    Some(rows) => rows,
                    // An operator `scalar_op.md` states no row for: the node
                    // is not offered, and the report names the operator.
                    None => {
                        builder.missing.push(format!(
                            "SCALAR_OP has no row for the operator `{}` on {}",
                            operator,
                            operand_dtype.as_str()
                        ));
                        return Ok(Vec::new());
                    }
                }
            }
        };
        // P3 (width): every width of the chosen row is admitted.
        if !super::widths_admitted(builder, &primitives) {
            return Ok(Vec::new());
        }
        for (primitive, times) in primitives {
            candidate = candidate.emit(primitive, times);
        }
        candidate = candidate.output(
            request.target.clone(),
            Interface::Wire(dtype),
            request.condition,
        );
        Ok(vec![candidate])
    }
}

impl ScalarOp {
    /// `/ravel` (`scalar_op.md`): the flat index of one coordinate tuple,
    /// `np.ravel_multi_index(index, dims, order)` with a run-time `dims`.
    ///
    /// Where every extent of `dims` is a compile-time constant, phase 0
    /// writes the whole arithmetic into the term and the value is an
    /// ordinary shape scalar. Where `dims` is a run-time array, every stride
    /// is a product of run-time terms, which §3.1 has none, so phase 0
    /// records an `E-Arith-Data` over the two arrays with a fresh data
    /// variable and the in-range condition as an unconstrained Boolean
    /// (`../../phase0/definition.md` §5.4, `E-Ravel`'s second form). The row
    /// is then the arithmetic itself: `Σ_k index_k · stride_k`, the strides
    /// the running products of the `dims` wires in the order the call names.
    ///
    /// `None` when the value is no `ravel_multi_index`; `Some(vec![])` when
    /// it is one the rule cannot fill, so that the generic `/op` path is not
    /// tried on operands it would skip.
    fn ravel(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        operands: &[ValueKey],
        dtype: Dtype,
        probe: &Candidate,
    ) -> Result<Option<Vec<Candidate>>, Reject> {
        let operator = builder.node(key).and_then(operator_of).unwrap_or_default();
        if operator != "ravel_multi_index" {
            return Ok(None);
        }
        let arrays: Vec<ValueKey> = operands
            .iter()
            .filter(|operand| builder.array(operand).is_some())
            .cloned()
            .collect();
        // The call's own order: the coordinate tuple first, the extents
        // second. A spelling with either missing is no node of this rule.
        let [index, dims] = arrays.as_slice() else {
            return Ok(Some(Vec::new()));
        };
        let (index, dims) = (index.clone(), dims.clone());
        let mut candidate = Candidate::new(RuleId::ScalarOp, Some("/ravel"), request.epoch);
        candidate.matched = vec![key.clone(), index.clone(), dims.clone()];
        candidate.answers = probe.answers.clone();
        let under = builder.under(request.condition);
        // P4: both arrays have rank 1 and one CONST extent r, the number of
        // axes the flat index runs over. The bound `0 ≤ index_k < dims_k` is
        // the value's own defcond, a check of Φ and `ENFORCE_CHECK`'s.
        let mut extent_of = |builder: &mut Builder<'_>,
                             candidate: &mut Candidate,
                             value: &ValueKey|
         -> Option<i128> {
            let dims = builder.array(value)?.dims.clone();
            let [only] = dims.as_slice() else { return None };
            match builder.ask_for(candidate, Question::Const(*only), request.epoch, under) {
                Verdict::Constant(value) => Some(value),
                _ => None,
            }
        };
        let (Some(axes), Some(extents)) = (
            extent_of(builder, &mut candidate, &index),
            extent_of(builder, &mut candidate, &dims),
        ) else {
            return Ok(Some(Vec::new()));
        };
        if axes < 1 || axes != extents {
            return Ok(Some(Vec::new()));
        }
        // `coords` and `extents`: one entry per axis of each array.
        let batch = Batch::grid(&[axes]);
        for array in [&index, &dims] {
            let row = builder.fetch_port(&mut candidate, request, array, &batch)?;
            let Some(first) = row.entries.first() else {
                return Ok(Some(Vec::new()));
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        // emitter: r − 2 products for the running strides — the outermost is
        // the literal 1 and the next is a `dims` wire — and r − 1 for the
        // terms of the sum, whose last factor is that 1; the sum is a `lin`.
        // The count is the same in C and F order, which differ only in which
        // end the running product starts at.
        let products = (axes - 2).max(0) + (axes - 1).max(0);
        Ok(Some(vec![candidate
            .emit(Primitive::Mul, products)
            .parameter("r", axes)
            .output(request.target.clone(), Interface::Wire(dtype), request.condition)]))
    }
}

/// The operator an AST node names, in the front end's spelling.
fn operator_of(node: &crate::phase1::ast_index::AstNode) -> Option<String> {
    use crate::phase1::ast_index::AstNode;
    match node {
        AstNode::Binary { operator } | AstNode::Unary { operator } => Some(operator.clone()),
        AstNode::Call { member, .. } | AstNode::MethodCall { member, .. } => Some(member.clone()),
        AstNode::IntLiteral(_) | AstNode::BoolLiteral(_) | AstNode::FloatLiteral => {
            Some("const".to_string())
        }
        _ => None,
    }
}

/// The rows of `scalar_op.md` §emitter for a data-origin value, whose term
/// is a fresh variable and so carries no structure to read the gadget off.
///
/// `intervals` are the operands' intervals in evaluation order; `term` is the
/// result's, for the Boolean rows.
fn data_rows(
    builder: &Builder<'_>,
    operator: &str,
    dtype: Dtype,
    operand_dtype: Dtype,
    intervals: &[Interval],
    term: ScalarTerm,
) -> Option<Vec<(Primitive, i128)>> {
    // A Float operand takes the float rows, whatever the result's dtype: a
    // comparison of two Float scalars is a Bool computed by `fop⟨lt_f⟩`.
    if super::super::sources::is_float(operand_dtype) {
        return float_rows(operator, dtype);
    }
    let width = builder.omega.round(builder.dtype_width(dtype));
    let x = intervals.first().copied().unwrap_or(Interval::UNBOUNDED);
    let y = intervals.get(1).copied().unwrap_or(Interval::UNBOUNDED);
    let comparison = builder.omega.round(width_of(super::difference(x, y)));
    let wraps = dtype == Dtype::I64;
    match operator {
        // Comparisons and connectives: the exact rows, whatever the dtype.
        "lt" | "gt" => Some(vec![(Primitive::Cmp { width: comparison }, 1)]),
        "lte" | "gte" => Some(vec![(Primitive::Cmp { width: comparison }, 1)]),
        "eq" | "ne" => Some(if matches!(term, ScalarTerm::Bool(_)) && dtype == Dtype::Bool {
            vec![(Primitive::Xor, 1)]
        } else {
            vec![(Primitive::Eq, 1)]
        }),
        "and" => Some(vec![(Primitive::And, 1)]),
        "or" => Some(vec![(Primitive::Or, 1)]),
        "bit_xor" if dtype == Dtype::Bool => Some(vec![(Primitive::Xor, 1)]),
        "not" | "invert" if dtype == Dtype::Bool => Some(vec![(Primitive::Not, 1)]),
        "min" | "max" => Some(vec![
            (Primitive::Cmp { width: comparison }, 1),
            (Primitive::Select, 1),
        ]),
        // `bool(x)` for an integer x is `not eq(x, 0)`; `int(b)` is the same
        // wire.
        "bool" => Some(vec![(Primitive::Eq, 1), (Primitive::Not, 1)]),
        "int" => Some(Vec::new()),
        // Arithmetic: exact on Idx, wrapping on a fixed-width dtype.
        "add" | "sub" if !wraps => Some(Vec::new()),
        "add" | "sub" => Some(vec![(wrap(builder, width, sum(x, y)), 1)]),
        // `k·x` is a `lin`, so a literal factor costs only the wrap; two
        // wires need the `mul` first.
        "mul" if !wraps => Some(match literal_factor(x, y) {
            Some(_) => Vec::new(),
            None => vec![(Primitive::Mul, 1)],
        }),
        "mul" => Some(match literal_factor(x, y) {
            Some((k, other)) => vec![(wrap(builder, width, scale(k, other)), 1)],
            None => vec![
                (Primitive::Mul, 1),
                (Primitive::Wrap { width, carry_width: builder.omega.round(width) }, 1),
            ],
        }),
        "usub" if !wraps => Some(Vec::new()),
        "usub" => Some(vec![(Primitive::Eq, 1)]),
        "abs" => {
            let magnitude = builder.omega.round(width_of(x));
            let mut rows = vec![
                (Primitive::Cmp { width: magnitude }, 1),
                (Primitive::Mul, 1),
            ];
            if wraps {
                rows.push((Primitive::Eq, 1));
            }
            Some(rows)
        }
        // `x // y` and `x % y` with a literal divisor are `divmod_k`; with a
        // wire divisor they are the data-division row of the rule file.
        "floor_div" | "mod" => match constant_of(y) {
            Some(k) if k != 0 => Some(vec![(
                Primitive::DivModConst {
                    quotient_width: builder.omega.round(width_of(x)),
                    divisor_width: builder.omega.round(bits_for(k.abs().max(1))),
                    power_of_two: k.abs().count_ones() == 1,
                },
                1,
            )]),
            _ => Some(vec![(Primitive::DivModWire { width }, 1), (Primitive::Mul, 4)]),
        },
        "shl" => Some(vec![(
            wrap(
                builder,
                width,
                match constant_of(y) {
                    Some(k) if (0..64).contains(&k) => scale(1i128 << k, x),
                    _ => Interval::UNBOUNDED,
                },
            ),
            1,
        )]),
        "shr" => Some(vec![(
            Primitive::DivModConst {
                quotient_width: builder.omega.round(width_of(x)),
                divisor_width: width,
                power_of_two: true,
            },
            1,
        )]),
        "bit_and" | "bit_or" | "bit_xor" => Some(vec![
            (Primitive::Range { width }, 2),
            (Primitive::And, width as i128),
        ]),
        // `x ** k` is square-and-multiply on a literal, non-negative k; an
        // exponent that is a wire has no row (Open 2).
        "pow" => match constant_of(y) {
            Some(k) if k >= 0 => Some(vec![(Primitive::Mul, bits_for(k.max(1)) as i128)]),
            _ => None,
        },
        // An operator the rule file states no row for.
        _ => None,
    }
}

/// The Float rows of `scalar_op.md` §emitter, in the `fop⟨name⟩`s of
/// `rules/README.md` §7.
///
/// The scalar table is the elementwise one at one position
/// (`elementwise::float_gadget`), with the two rows that only a scalar has:
/// a Float literal is one `constant_float` cell put on a wire, and an
/// integer scalar read as a float is one `float_cast`.
fn float_rows(operator: &str, dtype: Dtype) -> Option<Vec<(Primitive, i128)>> {
    match operator {
        // A literal: `CONSTANT`'s scalar row (§2.3), one `constant_float`.
        "const" => Some(vec![(Primitive::Float(FloatOp::Constant), 1)]),
        // `float(x)` on an integer scalar; on a float it is the same wire,
        // which `float_gadget` states.
        "float" | "float64" if dtype == Dtype::Float => {
            Some(vec![(Primitive::Float(FloatOp::FromInt), 1)])
        }
        other => crate::phase1::rules::elementwise::float_gadget(other, Some(dtype)),
    }
}

/// `wrap_w(t)` of `term_to_wires.md`: `range_w(r)` on the low bits plus the
/// carry check `range_{W(q − lo_q)}(q)`, `q` the quotient of `t + 2^{w−1}` by
/// `2^w`. The carry width follows from the interval of the unwrapped result.
fn wrap(builder: &Builder<'_>, width: u32, raw: Interval) -> Primitive {
    let modulus = 1i128 << width.min(126);
    let offset = modulus / 2;
    let quotient = |t: i128| crate::phase0::term::floor_div(t + offset, modulus);
    let carry = match (raw.lo, raw.hi) {
        (Some(lo), Some(hi)) => bits_for(quotient(hi) - quotient(lo) + 1),
        // Without an interval the carry needs the whole width.
        _ => width.max(1),
    };
    Primitive::Wrap { width, carry_width: builder.omega.round(carry) }
}

/// `I_{a + b}`.
fn sum(a: Interval, b: Interval) -> Interval {
    Interval::new(
        a.lo.zip(b.lo).map(|(x, y)| x + y),
        a.hi.zip(b.hi).map(|(x, y)| x + y),
    )
}

/// `I_{k·t}`; the endpoints swap for a negative `k`.
fn scale(k: i128, t: Interval) -> Interval {
    let (lo, hi) = if k >= 0 { (t.lo, t.hi) } else { (t.hi, t.lo) };
    Interval::new(lo.map(|n| n * k), hi.map(|n| n * k))
}

/// The literal factor of a product, with the other operand's interval.
fn literal_factor(x: Interval, y: Interval) -> Option<(i128, Interval)> {
    match (constant_of(x), constant_of(y)) {
        (Some(k), _) => Some((k, y)),
        (_, Some(k)) => Some((k, x)),
        _ => None,
    }
}

/// The single value an interval pins, when it pins one.
fn constant_of(range: Interval) -> Option<i128> {
    match (range.lo, range.hi) {
        (Some(lo), Some(hi)) if lo == hi => Some(lo),
        _ => None,
    }
}

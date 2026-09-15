//! Inputs and constants: `definition.md` §5.1.
//!
//! `E-Input-Fixed`, `E-Input-VarLen`, `E-Input-Symbolic`,
//! `E-Input-Composite`, `E-Input-Hashed` and `E-Const`, plus what §5.1 needs
//! from the circuit's contract: [`ContractSymbols`], the name -> variable
//! table the annotations and the `scalar_requires` facts share.

use std::collections::HashMap;

use super::Rule;
use super::{RuleCtx, VarRange};
use crate::optim::predicates::formula::{
    ArithOp as CArith, BoolOp as CBool, CmpOp as CCmp, ContractTerm, ContractVar,
};
use crate::phase0::desc::{
    CompositeDesc, CompositeKind, Desc, Dtype, Origin, ScalarTerm, ValueKey,
};
use crate::phase0::phi::{ConstraintKind, Phi, VarOrigin};
use crate::phase0::term::{BoolId, Sort, TermArena, TermId, VarId};
use crate::phase0::Reject;
use crate::types::{ShapeDim, ZinniaType};

/// `E-Const` (§5.1): an integer literal.
///
/// Its dtype is `Idx`, an unbounded Python integer, and its origin is
/// `shape` — a literal is built from the parameters in $S$.
pub fn e_const(ctx: &mut RuleCtx, n: i64) -> Desc {
    let t = ctx.terms.constant(n as i128);
    let tt = ctx.tt();
    ctx.scalar(
        Rule::EConst,
        Dtype::Idx,
        ScalarTerm::Int(t),
        Origin::Shape,
        tt,
    )
}

/// A Boolean literal, the Boolean-sorted companion of [`e_const`].
pub fn e_const_bool(ctx: &mut RuleCtx, b: bool) -> Desc {
    let v = ctx.terms.bool_const(b);
    let tt = ctx.tt();
    ctx.scalar(
        Rule::EConst,
        Dtype::Bool,
        ScalarTerm::Bool(v),
        Origin::Shape,
        tt,
    )
}

/// A literal of a dtype that carries no term: `Float`, `Complex`, `String`.
pub fn e_const_opaque(ctx: &mut RuleCtx, dtype: Dtype) -> Desc {
    let tt = ctx.tt();
    ctx.scalar(Rule::EConst, dtype, ScalarTerm::Opaque, Origin::Shape, tt)
}

/// The name -> variable table for everything §5.1 shares by name: the shape
/// symbols of `SymbolicNDArray` annotations ("variable `n_s` shared by name
/// across inputs"), the scalar inputs, and the syntactic bounds the
/// `scalar_requires` facts state for either.
///
/// Phase 0 makes no solver call, so the bounds are whatever a purely
/// syntactic scan of the facts finds: comparisons of a name against an
/// integer literal. Anything else only loosens the range, which is sound.
#[derive(Debug, Default)]
pub struct ContractSymbols {
    /// Shape symbols, by symbol name.
    vars: HashMap<String, VarId>,
    /// Scalar inputs, by parameter name. A name that is both is a shape
    /// symbol first (see [`scalar_input`]).
    scalars: HashMap<String, VarId>,
    /// Inclusive bounds harvested from `scalar_requires`, by name.
    lo: HashMap<String, i128>,
    hi: HashMap<String, i128>,
}

impl ContractSymbols {
    /// Read `lo`/`hi` bounds off the `scalar_requires` facts of the circuit.
    pub fn harvest(&mut self, facts: &[ContractTerm]) {
        for fact in facts {
            self.harvest_one(fact);
        }
    }

    fn harvest_one(&mut self, fact: &ContractTerm) {
        match fact {
            ContractTerm::BoolComb {
                op: CBool::And,
                operands,
            } => operands.iter().for_each(|f| self.harvest_one(f)),
            ContractTerm::Cmp { op, lhs, rhs } => {
                if let (Some(sym), Some(k)) = (symbol_of(lhs), literal_of(rhs)) {
                    self.note(&sym, *op, k);
                } else if let (Some(k), Some(sym)) = (literal_of(lhs), symbol_of(rhs)) {
                    self.note(&sym, mirror(*op), k);
                }
            }
            _ => {}
        }
    }

    /// Record one `symbol <op> k` fact as an inclusive bound.
    fn note(&mut self, symbol: &str, op: CCmp, k: i128) {
        let tighten = |map: &mut HashMap<String, i128>, value: i128, keep_max: bool| {
            let entry = map.entry(symbol.to_string()).or_insert(value);
            *entry = if keep_max {
                (*entry).max(value)
            } else {
                (*entry).min(value)
            };
        };
        match op {
            CCmp::Le => tighten(&mut self.hi, k, false),
            CCmp::Lt => tighten(&mut self.hi, k - 1, false),
            CCmp::Ge => tighten(&mut self.lo, k, true),
            CCmp::Gt => tighten(&mut self.lo, k + 1, true),
            CCmp::Eq => {
                tighten(&mut self.lo, k, true);
                tighten(&mut self.hi, k, false);
            }
            CCmp::Ne => {}
        }
    }

    /// The variable of a shape symbol, declaring it on first use. `upper` is
    /// the bound from the annotation (`Bounded { upper }`), intersected with
    /// the harvested one.
    /// Declared before the walk as well as during it, so it takes $\Phi$ and
    /// the arena rather than a [`RuleCtx`]: a shape symbol is not a value and
    /// has no key to be the subject of (`implementation.md` §5a).
    pub fn declare(
        &mut self,
        terms: &mut TermArena,
        phi: &mut Phi,
        symbol: &str,
        upper: Option<i128>,
    ) -> VarId {
        if let Some(&v) = self.vars.get(symbol) {
            return v;
        }
        // A dimension is non-negative; a fact may raise the lower bound.
        let lo = self.lo.get(symbol).copied().unwrap_or(0).max(0);
        let hi = match (upper, self.hi.get(symbol).copied()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let lo_term = terms.constant(lo);
        let hi_term = hi.map(|n| terms.constant(n));
        let v = phi.fresh_var(
            terms,
            symbol,
            Sort::Idx,
            VarOrigin::ShapeSymbol {
                symbol: symbol.to_string(),
            },
            Some(lo_term),
            hi_term,
        );
        self.vars.insert(symbol.to_string(), v);
        v
    }

    pub fn get(&self, symbol: &str) -> Option<VarId> {
        self.vars.get(symbol).copied()
    }

    /// The bounds the facts state for one name, as inclusive literals.
    pub fn bounds(&self, name: &str) -> (Option<i128>, Option<i128>) {
        (self.lo.get(name).copied(), self.hi.get(name).copied())
    }

    /// Remember the variable of a scalar input, so that a fact naming that
    /// input becomes a constraint on the same variable its descriptor uses.
    fn note_scalar(&mut self, name: &str, var: VarId) {
        self.scalars.insert(name.to_string(), var);
    }

    /// The variable of a scalar input, for [`add_scalar_requires`].
    pub fn scalar(&self, name: &str) -> Option<VarId> {
        self.scalars.get(name).copied()
    }
}

fn symbol_of(t: &ContractTerm) -> Option<String> {
    match t {
        ContractTerm::Var(ContractVar::ShapeParam(s))
        | ContractTerm::Var(ContractVar::Input(s)) => Some(s.clone()),
        _ => None,
    }
}

fn literal_of(t: &ContractTerm) -> Option<i128> {
    match t {
        ContractTerm::LitInt(n) => Some(*n as i128),
        _ => None,
    }
}

/// `a <op> b` becomes `b <mirror(op)> a`.
fn mirror(op: CCmp) -> CCmp {
    match op {
        CCmp::Lt => CCmp::Gt,
        CCmp::Le => CCmp::Ge,
        CCmp::Gt => CCmp::Lt,
        CCmp::Ge => CCmp::Le,
        CCmp::Eq => CCmp::Eq,
        CCmp::Ne => CCmp::Ne,
    }
}

/// `E-Input-*` (§5.1): the descriptor of a circuit input from its
/// annotation.
///
/// - `NDArray` is `E-Input-Fixed`.
/// - `SymbolicNDArray` is `E-Input-Symbolic`: a `Const` axis is a numeral, a
///   `Symbol` or `Bounded` axis is the variable of that name, and an exact
///   binding makes it a constant again.
/// - `DynamicNDArray` with a `max_shape` is `E-Input-VarLen`, one variable
///   per bounded axis; of rank 1 without a `max_shape` it is the same with
///   `N_x = max_length`; of higher rank without one it has a runtime rank
///   and is rejected (§5.1, last paragraph).
/// - `List` / `Tuple` is `E-Input-Composite`, `PoseidonHashed` is
///   `E-Input-Hashed`.
pub fn e_input(
    ctx: &mut RuleCtx,
    name: &str,
    key: &ValueKey,
    dt: &ZinniaType,
    symbols: &mut ContractSymbols,
) -> Result<Desc, Reject> {
    match dt {
        ZinniaType::NDArray { shape, dtype } => {
            let dims = shape
                .iter()
                .map(|&n| ctx.terms.constant(n as i128))
                .collect();
            let tt = ctx.tt();
            Ok(ctx.array(Rule::EInputFixed, Dtype::from_number_type(*dtype), dims, tt))
        }
        ZinniaType::SymbolicNDArray {
            shape,
            dtype,
            bindings,
        } => {
            let mut dims = Vec::with_capacity(shape.len());
            for dim in shape {
                dims.push(symbolic_dim(ctx, dim, bindings, symbols));
            }
            let tt = ctx.tt();
            Ok(ctx.array(
                Rule::EInputSymbolic,
                Dtype::from_number_type(*dtype),
                dims,
                tt,
            ))
        }
        ZinniaType::DynamicNDArray {
            dtype,
            max_length,
            max_rank,
            max_shape,
        } => {
            let bounds: Vec<usize> = match max_shape {
                Some(s) => s.clone(),
                None if *max_rank == 1 => vec![*max_length],
                None => {
                    return Err(Reject::RuntimeRankInput {
                        input: name.to_string(),
                    })
                }
            };
            let dims = bounds
                .iter()
                .enumerate()
                .map(|(axis, &n)| var_len_axis(ctx, name, axis, n))
                .collect();
            let tt = ctx.tt();
            Ok(ctx.array(
                Rule::EInputVarLen,
                Dtype::from_number_type(*dtype),
                dims,
                tt,
            ))
        }
        ZinniaType::List { elements } | ZinniaType::Tuple { elements } => {
            let kind = if matches!(dt, ZinniaType::List { .. }) {
                CompositeKind::List
            } else {
                CompositeKind::Tuple
            };
            let mut elems = Vec::with_capacity(elements.len());
            for (i, element) in elements.iter().enumerate() {
                let sub = format!("{}[{}]", name, i);
                // Each element takes the key of its position, so that two
                // scalar elements of one input do not claim the same value
                // ([`ValueKey::component`]).
                let element_key = key.component(i as u32);
                elems.push(e_input(ctx, &sub, &element_key, element, symbols)?);
            }
            Ok(Desc::Composite(CompositeDesc { kind, elems }))
        }
        ZinniaType::PoseidonHashed { dtype } => e_input(ctx, name, key, dtype, symbols),
        ZinniaType::Integer | ZinniaType::Float | ZinniaType::Boolean | ZinniaType::Complex => {
            Ok(scalar_input(ctx, name, key, dt, symbols))
        }
        ZinniaType::String => Ok(e_const_opaque(ctx, Dtype::String)),
        ZinniaType::None => Ok(Desc::None),
        ZinniaType::Class => Ok(Desc::Class(dt.clone())),
    }
}

/// One axis of `E-Input-Symbolic`.
fn symbolic_dim(
    ctx: &mut RuleCtx,
    dim: &ShapeDim,
    bindings: &std::collections::BTreeMap<String, usize>,
    symbols: &mut ContractSymbols,
) -> TermId {
    let (symbol, upper) = match dim {
        ShapeDim::Const(n) => return ctx.terms.constant(*n as i128),
        ShapeDim::Symbol(s) => (s.clone(), None),
        ShapeDim::Bounded { symbol, upper } => (symbol.clone(), Some(*upper as i128)),
    };
    // An exact binding makes the axis a constant (§5.1).
    if let Some(&n) = bindings.get(&symbol) {
        return ctx.terms.constant(n as i128);
    }
    let v = symbols.declare(ctx.terms, ctx.phi, &symbol, upper);
    ctx.terms.var(v)
}

/// One bounded axis of `E-Input-VarLen`: `new var n_x, range 0 <= n_x <= N_x`.
fn var_len_axis(ctx: &mut RuleCtx, input: &str, axis: usize, bound: usize) -> TermId {
    let zero = ctx.terms.constant(0);
    let hi = ctx.terms.constant(bound as i128);
    let base = format!("n_{}_{}", input, axis);
    let v = ctx.phi.fresh_var(
        ctx.terms,
        &base,
        Sort::Idx,
        VarOrigin::InputLength {
            input: input.to_string(),
            axis,
        },
        Some(zero),
        Some(hi),
    );
    ctx.terms.var(v)
}

/// A scalar input. Its value is a run-time quantity, so it is a data scalar
/// — unless its name is already a declared shape symbol, in which case the
/// two are the same quantity and the descriptor is that symbol's variable
/// with origin `shape`.
///
/// Its range is the admitted domain of §3.4, not just the dtype range: a
/// `@requires(lambda n: 0 <= n <= 1024)` bound is part of $S$ ("the admitted
/// domain of each input", §1), and it is what lets `E-ValueSized` (§5.9) and
/// `S-For-Range-Bounded` read a maximum off a term that mentions the input.
fn scalar_input(
    ctx: &mut RuleCtx,
    name: &str,
    key: &ValueKey,
    dt: &ZinniaType,
    symbols: &mut ContractSymbols,
) -> Desc {
    let dtype = match dt {
        ZinniaType::Boolean => Dtype::Bool,
        ZinniaType::Float => Dtype::Float,
        ZinniaType::Complex => Dtype::Complex,
        _ => Dtype::I64,
    };
    if let Some(v) = symbols.get(name) {
        let t = ctx.terms.var(v);
        let tt = ctx.tt();
        return ctx.scalar(
            Rule::EInputSymbolic,
            Dtype::Idx,
            ScalarTerm::Int(t),
            Origin::Shape,
            tt,
        );
    }
    let range = admitted_domain(ctx, symbols, name, dtype);
    let tt = ctx.tt();
    let term = ctx.fresh_data_term(dtype, name, key, range);
    if let Some(v) = term.as_int().and_then(|t| ctx.terms.var_of(t)) {
        symbols.note_scalar(name, v);
    }
    ctx.scalar(Rule::EInputFixed, dtype, term, Origin::Data, tt)
}

/// The range of a scalar input's variable: the dtype range of §3.4,
/// tightened by the bounds the `scalar_requires` facts state for that name.
///
/// Only a syntactic bound counts, as everywhere else in this file, and only
/// one that tightens: a fact that is wider than the dtype changes nothing.
fn admitted_domain(
    ctx: &mut RuleCtx,
    symbols: &ContractSymbols,
    name: &str,
    dtype: Dtype,
) -> VarRange {
    let (dtype_lo, dtype_hi) = dtype.value_range();
    let (fact_lo, fact_hi) = symbols.bounds(name);
    let lo = match (dtype_lo, fact_lo) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    let hi = match (dtype_hi, fact_hi) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    VarRange {
        lo: lo.map(|n| ctx.terms.constant(n)),
        hi: hi.map(|n| ctx.terms.constant(n)),
    }
}

// ---------------------------------------------------------------------------
// scalar_requires facts
// ---------------------------------------------------------------------------

/// A translated contract sub-term.
enum Translated {
    Int(TermId),
    Bool(BoolId),
}

/// Assert the circuit's `scalar_requires` facts on $\Phi$ (`coverage.md` §4:
/// "constraints on input scalars and shape symbols, added to $\Phi$").
///
/// A fact that does not fit the term language of §3.1 — a predicate
/// application, a float literal, a product of two non-constants, a division
/// by a variable — is skipped. Dropping a precondition only loosens $\Phi$,
/// which is sound: G1 says a successful run satisfies $\Phi$, and fewer
/// constraints can only make that easier.
pub fn add_scalar_requires(
    terms: &mut TermArena,
    phi: &mut Phi,
    facts: &[ContractTerm],
    symbols: &ContractSymbols,
) {
    for (i, fact) in facts.iter().enumerate() {
        if let Some(Translated::Bool(b)) = translate(terms, fact, symbols) {
            phi.add_fact(
                terms,
                b,
                ConstraintKind::Precondition,
                &format!("scalar_requires[{}]", i),
            );
        }
    }
}

fn translate(terms: &mut TermArena, t: &ContractTerm, symbols: &ContractSymbols) -> Option<Translated> {
    match t {
        ContractTerm::LitInt(n) => Some(Translated::Int(terms.constant(*n as i128))),
        ContractTerm::LitBool(b) => Some(Translated::Bool(terms.bool_const(*b))),
        ContractTerm::Var(v) => {
            let name = match v {
                ContractVar::ShapeParam(s) | ContractVar::Input(s) => s,
                _ => return None,
            };
            // A name is a shape symbol first and a scalar input second, as
            // in `scalar_input`; either way the fact lands on the variable
            // the input's descriptor uses.
            let var = symbols.get(name).or_else(|| symbols.scalar(name))?;
            Some(Translated::Int(terms.var(var)))
        }
        ContractTerm::Arith { op, lhs, rhs } => translate_arith(terms, *op, lhs, rhs, symbols),
        ContractTerm::Cmp { op, lhs, rhs } => {
            let a = int_of(translate(terms, lhs, symbols)?)?;
            let b = int_of(translate(terms, rhs, symbols)?)?;
            Some(Translated::Bool(compare(terms, *op, a, b)))
        }
        ContractTerm::Not(inner) => match translate(terms, inner, symbols)? {
            Translated::Bool(b) => Some(Translated::Bool(terms.not(b))),
            Translated::Int(_) => None,
        },
        ContractTerm::BoolComb { op, operands } => {
            let mut parts = Vec::with_capacity(operands.len());
            for operand in operands {
                match translate(terms, operand, symbols)? {
                    Translated::Bool(b) => parts.push(b),
                    Translated::Int(_) => return None,
                }
            }
            Some(Translated::Bool(match op {
                CBool::And => terms.all(&parts),
                CBool::Or => terms.any(&parts),
            }))
        }
        // Structural predicates have no shape effect in phase 0
        // (`coverage.md` §4); float literals are outside §3.1.
        ContractTerm::PredicateApp { .. } | ContractTerm::LitFloat(_) => None,
    }
}

fn translate_arith(
    terms: &mut TermArena,
    op: CArith,
    lhs: &ContractTerm,
    rhs: &ContractTerm,
    symbols: &ContractSymbols,
) -> Option<Translated> {
    let a = int_of(translate(terms, lhs, symbols)?)?;
    let b = int_of(translate(terms, rhs, symbols)?)?;
    let (ka, kb) = (terms.constant_value(a), terms.constant_value(b));
    let out = match op {
        CArith::Add => terms.add(a, b),
        CArith::Sub => terms.sub(a, b),
        // §3.1 allows `k * t` only.
        CArith::Mul => match (ka, kb) {
            (Some(k), _) => terms.mul_const(k, b),
            (_, Some(k)) => terms.mul_const(k, a),
            _ => return None,
        },
        CArith::FloorDiv | CArith::Div => terms.div_const(a, kb?)?,
        CArith::Mod => terms.mod_const(a, kb?)?,
        CArith::Pow => return None,
    };
    Some(Translated::Int(out))
}

fn compare(terms: &mut TermArena, op: CCmp, a: TermId, b: TermId) -> BoolId {
    match op {
        CCmp::Eq => terms.eq(a, b),
        CCmp::Ne => {
            let e = terms.eq(a, b);
            terms.not(e)
        }
        CCmp::Lt => terms.lt(a, b),
        CCmp::Le => terms.le(a, b),
        CCmp::Gt => terms.lt(b, a),
        CCmp::Ge => terms.le(b, a),
    }
}

fn int_of(t: Translated) -> Option<TermId> {
    match t {
        Translated::Int(i) => Some(i),
        Translated::Bool(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::tests::Fixture;
    use crate::types::NumberType;

    fn key_of(f: &mut Fixture) -> ValueKey {
        f.key()
    }

    /// §5.1: `E-Input-Fixed` takes the annotation's extents, `E-Const` is a
    /// shape scalar.
    #[test]
    fn e_input_fixed_takes_the_declared_shape() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let key = key_of(&mut f);
        let dt = ZinniaType::NDArray {
            shape: vec![2, 5],
            dtype: NumberType::Integer,
        };
        let built = e_input(&mut f.ctx(), "x", &key, &dt, &mut symbols).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(2), Some(5)]);
        assert_eq!(f.array_of(&built).dtype, Dtype::I64);
    }

    /// §5.1: `E-Input-VarLen` introduces `n_x` with `0 ≤ n_x ≤ N_x`.
    #[test]
    fn e_input_varlen_introduces_a_bounded_length() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let key = key_of(&mut f);
        let dt = ZinniaType::DynamicNDArray {
            dtype: NumberType::Integer,
            max_length: 64,
            max_rank: 1,
            max_shape: None,
        };
        let built = e_input(&mut f.ctx(), "x", &key, &dt, &mut symbols).unwrap();
        assert_eq!(f.dims_of(&built), vec![None], "the extent is a variable");
        let var = f
            .analysis
            .terms
            .var_of(f.array_of(&built).dims[0])
            .expect("one variable");
        let info = f.analysis.phi.var_info(var);
        assert_eq!(f.analysis.terms.constant_value(info.hi.unwrap()), Some(64));
    }

    /// §5.1, last paragraph: a `DynamicNDArray` of runtime rank rejects.
    #[test]
    fn a_runtime_rank_input_rejects() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let key = key_of(&mut f);
        let dt = ZinniaType::DynamicNDArray {
            dtype: NumberType::Integer,
            max_length: 64,
            max_rank: 2,
            max_shape: None,
        };
        let outcome = e_input(&mut f.ctx(), "x", &key, &dt, &mut symbols);
        assert!(
            matches!(outcome, Err(Reject::RuntimeRankInput { .. })),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.1: a `SymbolicNDArray` symbol is one variable shared by name
    /// across inputs; an exact binding makes the axis a constant again.
    #[test]
    fn a_shape_symbol_is_shared_by_name() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let dt = ZinniaType::SymbolicNDArray {
            shape: vec![ShapeDim::Symbol("n".to_string())],
            dtype: NumberType::Integer,
            bindings: Default::default(),
        };
        let key = key_of(&mut f);
        let first = e_input(&mut f.ctx(), "x", &key, &dt, &mut symbols).unwrap();
        let key = key_of(&mut f);
        let second = e_input(&mut f.ctx(), "y", &key, &dt, &mut symbols).unwrap();
        assert_eq!(f.array_of(&first).dims[0], f.array_of(&second).dims[0]);

        let mut bindings = std::collections::BTreeMap::new();
        bindings.insert("n".to_string(), 7usize);
        let bound = ZinniaType::SymbolicNDArray {
            shape: vec![ShapeDim::Symbol("n".to_string())],
            dtype: NumberType::Integer,
            bindings,
        };
        let key = key_of(&mut f);
        let exact = e_input(&mut f.ctx(), "z", &key, &bound, &mut symbols).unwrap();
        assert_eq!(f.dims_of(&exact), vec![Some(7)]);
    }

    /// §5.1: a `List` / `Tuple` input is a composite, and each element takes
    /// the key of its position (`ValueKey::component`), so two scalar
    /// elements do not share a variable provenance.
    #[test]
    fn e_input_composite_keys_its_elements_apart() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let key = key_of(&mut f);
        let dt = ZinniaType::Tuple {
            elements: vec![ZinniaType::Integer, ZinniaType::Integer],
        };
        let built = e_input(&mut f.ctx(), "p", &key, &dt, &mut symbols).unwrap();
        assert_eq!(built.as_composite().unwrap().elems.len(), 2);
        let keys: Vec<String> = f
            .analysis
            .phi
            .vars()
            .iter()
            .filter_map(|v| match &v.origin {
                VarOrigin::DataScalar { value } => Some(value.encode()),
                _ => None,
            })
            .collect();
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0], keys[1]);
    }

    /// `implementation.md` §8: a scalar input's range is its admitted
    /// domain, the dtype range tightened by the `scalar_requires` bounds.
    #[test]
    fn a_scalar_inputs_range_is_its_admitted_domain() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let facts = vec![ContractTerm::Cmp {
            op: CCmp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Input("n".to_string()))),
            rhs: Box::new(ContractTerm::LitInt(32)),
        }];
        symbols.harvest(&facts);
        let key = key_of(&mut f);
        let built = e_input(&mut f.ctx(), "n", &key, &ZinniaType::Integer, &mut symbols).unwrap();
        let term = f.scalar_of(&built).term.as_int().unwrap();
        let var = f.analysis.terms.var_of(term).expect("a fresh variable");
        let info = f.analysis.phi.var_info(var);
        assert_eq!(f.analysis.terms.constant_value(info.hi.unwrap()), Some(32));
    }
}

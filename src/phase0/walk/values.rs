//! The shape classes that act on values rather than on array shapes: the
//! builtins of `coverage.md` §5.7 and §5.8, `E-Property` and `S-ListMethod`
//! (§5.11), and the dtype casts of §5.1.
//!
//! [`Walker::apply_class`] is the single entry point from the dispatch in
//! [`super::calls`]; the array classes continue into [`super::arrays`].

use crate::phase0::desc::{
    ArrayDesc, CompositeDesc, CompositeKind, Desc, Dtype, Origin, ScalarTerm, ValueKey,
};
use crate::phase0::ops::{Class, Domain, ListMethod, MapOut, Prop, ZipOp};
use crate::phase0::rules::{
    e_composite, e_composite_index, e_const, e_const_opaque, e_len, e_property, e_reduce_axis,
    e_reduce_scalar, e_transpose, merge_desc, wrap_index, Property, ReduceKind, VarRange,
};
use crate::phase0::source::Producer;
use crate::phase0::Reject;

use super::calls::Args;
use super::ranges::RangePlan;
use super::Walker;
use crate::phase0::rules::Rule;

impl Walker<'_> {
    /// Apply one shape class of [`crate::phase0::ops`].
    pub fn apply_class(
        &mut self,
        class: Class,
        args: &Args,
        key: &ValueKey,
        member: &str,
    ) -> Result<Desc, Reject> {
        match class {
            Class::Map(out, domain) => {
                let operand = args.require(Rule::EMap, 0, "x")?.clone();
                self.apply_map(out, domain, &operand, key, member)
            }
            Class::Zip(op) => {
                let a = args.require(Rule::EZip, 0, "x1")?.clone();
                let b = args.require(Rule::EZip, 1, "x2")?.clone();
                self.apply_zip(op, &a, &b, key)
            }
            Class::Property(prop) => self.property(prop, args, key),
            Class::Cast(dtype) => self.cast(dtype, args, key),
            Class::DtypeClass(dtype) => Ok(Desc::Class(dtype.to_zinnia_type())),
            Class::FloatConstant => Ok(e_const_opaque(&mut self.ctx(), Dtype::Float)),
            Class::Len => {
                let operand = args.require(Rule::ELen, 0, "obj")?.clone();
                e_len(&mut self.ctx(), &operand)
            }
            Class::Range => self.builtin_range(args),
            Class::Enumerate => self.builtin_enumerate(args),
            Class::ListCtor => self.builtin_sequence(args, CompositeKind::List),
            Class::TupleCtor => self.builtin_sequence(args, CompositeKind::Tuple),
            Class::BuiltinSum => self.builtin_sum(args, key),
            Class::BuiltinMinMax { max } => self.builtin_minmax(args, key, max),
            Class::BuiltinPow => {
                let a = args.require(Rule::EArithData, 0, "base")?.clone();
                let b = args.require(Rule::EArithData, 1, "exp")?.clone();
                self.apply_zip(ZipOp::Pow, &a, &b, key)
            }
            Class::Complex => Ok(e_const_opaque(&mut self.ctx(), Dtype::Complex)),
            Class::NoShapeEffect => Ok(Desc::None),
            Class::Copy | Class::PromoteToDynamic => {
                let value = args.require(Rule::ECopy, 0, "a")?.clone();
                self.ctx().applied(Rule::ECopy);
                Ok(value)
            }
            Class::ListMethod(_) => Err(Reject::Unsupported {
                construct: format!("`{}` on a value that is not a bound name", member),
            }),
            other => self.apply_array_class(other, args, key, member),
        }
    }

    // ── `E-Property` (§5.11) ──────────────────────────────────────────

    fn property(&mut self, prop: Prop, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let receiver = args.require(Rule::EProperty, 0, "self")?.clone();
        match prop {
            Prop::Shape => self.array_property(&receiver, Property::Shape, key),
            Prop::Ndim => self.array_property(&receiver, Property::Ndim, key),
            Prop::Size => self.array_property(&receiver, Property::Size, key),
            Prop::Dtype => self.array_property(&receiver, Property::Dtype, key),
            Prop::Transposed => {
                let a = self.as_array(&receiver, Rule::ETranspose)?;
                e_transpose(&mut self.ctx(), &a, None)
            }
            Prop::Flat => self.ravel(&receiver, key),
            Prop::RealPart => self.apply_map(MapOut::RealPart, Domain::All, &receiver, key, "real"),
            Prop::ImagPart => self.apply_map(MapOut::RealPart, Domain::All, &receiver, key, "imag"),
        }
    }

    fn array_property(
        &mut self,
        receiver: &Desc,
        property: Property,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let a = self.as_array(receiver, Rule::EProperty)?;
        e_property(&mut self.ctx(), &a, property, key)
    }

    // ── Casts (§5.1) ──────────────────────────────────────────────────

    /// `int(x)`, `float(x)`, `bool(x)`, `x.astype(dt)` and the `np.<dtype>`
    /// calls: the shape is unchanged and only the dtype moves.
    ///
    /// `dtype` is `None` for `astype`, which reads its target from the
    /// argument instead ([`crate::phase0::ops::Class::Cast`]).
    fn cast(&mut self, dtype: Option<Dtype>, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let Some(operand) = args.get(0, "x").cloned() else {
            // `int()` with no argument is zero (`builtins.rs`).
            return Ok(e_const(&mut self.ctx(), 0));
        };
        let target = match dtype {
            Some(named) => named,
            None => match args.get(1, "dtype") {
                Some(given) => dtype_of_value(given)?.unwrap_or(Dtype::I64),
                None => Dtype::I64,
            },
        };
        self.cast_desc(&operand, target, key)
    }

    fn cast_desc(&mut self, d: &Desc, target: Dtype, key: &ValueKey) -> Result<Desc, Reject> {
        match d {
            Desc::Array(a) => {
                let dims = a.dims.clone();
                let tt = self.ctx().tt();
                Ok(self.ctx().array(Rule::EMap, target, dims, tt))
            }
            Desc::Scalar(s) => {
                // An integer cast of a shape scalar keeps the shape term, so
                // `int(len(a))` is still a length (§3.2).
                if s.origin == Origin::Shape && target.carries_term() {
                    if let ScalarTerm::Int(t) = s.term {
                        let tt = self.ctx().tt();
                        return Ok(self.ctx().scalar(
                            Rule::EMap,
                            target,
                            ScalarTerm::Int(t),
                            Origin::Shape,
                            tt,
                        ));
                    }
                }
                let range = VarRange::of_dtype(&mut self.an.terms, target);
                let tt = self.ctx().tt();
                Ok(self
                    .ctx()
                    .data_scalar(Rule::EMap, target, "cast", key, range, tt))
            }
            Desc::Composite(c) => {
                let (kind, elems) = (c.kind, c.elems.clone());
                let mut out = Vec::with_capacity(elems.len());
                for element in &elems {
                    out.push(self.cast_desc(element, target, key)?);
                }
                Ok(e_composite(kind, out))
            }
            other => Ok(other.clone()),
        }
    }

    // ── Builtins (`coverage.md` §5.7, §5.8) ───────────────────────────

    /// `range(stop)`, `range(start, stop)`, `range(start, stop, step)` as a
    /// *value*: a composite of integer constants, as the baseline's
    /// `builtin_range` builds it.
    ///
    /// Only the constant form is a value. A `range` whose trip count is a
    /// term stands for a list phase 0 cannot write down — its length is not a
    /// compile-time constant, which §3.3 requires of a composite — so it is
    /// rejected here and handled as a loop by `S-For-Range-Bounded`
    /// ([`super::loops`]). The baseline turns either non-constant form into
    /// zero iterations silently (`coverage.md` §9).
    fn builtin_range(&mut self, args: &Args) -> Result<Desc, Reject> {
        let (start, step, count) = match self.range_plan(args)? {
            RangePlan::Constant { start, step, count } => (start, step, count),
            RangePlan::Bounded { trip, .. } => {
                return Err(Reject::NonConstantRangeBound {
                    detail: format!(
                        "a `range` of `{}` elements is a list whose length is not a \
                         compile-time constant (§3.3); phase 0 unrolls it only as the \
                         iterable of a `for`",
                        self.an.terms.pretty(trip)
                    ),
                })
            }
        };
        self.spend(count as u64)?;
        let first = self.an.terms.constant_value(start).unwrap_or(0);
        let elems = (0..count as i128)
            .map(|j| e_const_i128(self, first + j * step))
            .collect();
        Ok(e_composite(CompositeKind::List, elems))
    }

    /// `enumerate(v)`: `S-For-Elems` with an index scalar (`coverage.md` §3).
    fn builtin_enumerate(&mut self, args: &Args) -> Result<Desc, Reject> {
        let iterable = args.require(Rule::SForElems, 0, "iterable")?.clone();
        let elements = self.iterable_elements(&iterable, None, "enumerate")?;
        let mut pairs = Vec::with_capacity(elements.len());
        for (index, element) in elements.into_iter().enumerate() {
            let position = e_const_i128(self, index as i128);
            pairs.push(e_composite(
                CompositeKind::Tuple,
                vec![position, element.desc],
            ));
        }
        Ok(e_composite(CompositeKind::List, pairs))
    }

    /// `list(v)` and `tuple(v)`: the same elements under the other kind.
    ///
    /// A value that is not a sequence comes back unchanged, which is what
    /// `builtins.rs` does rather than raising.
    fn builtin_sequence(&mut self, args: &Args, kind: CompositeKind) -> Result<Desc, Reject> {
        let Some(value) = args.get(0, "iterable").cloned() else {
            return Ok(e_composite(kind, Vec::new()));
        };
        match value {
            Desc::Composite(_) | Desc::Array(_) => {
                let elements = self.iterable_elements(&value, None, "list()/tuple()")?;
                let elems = elements.into_iter().map(|e| e.desc).collect();
                Ok(e_composite(kind, elems))
            }
            other => {
                // Not a sequence: the value comes back unchanged, which is
                // the identity `E-Copy` names (§8).
                self.ctx().applied(Rule::ECopy);
                Ok(other)
            }
        }
    }

    /// `sum(v)`: a reduction over a flat sequence, and an axis-0 reduction
    /// over a sequence of sequences, as `builtins.rs` splits the two cases.
    fn builtin_sum(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let value = args.require(Rule::EReduceScalar, 0, "iterable")?.clone();
        let reduced = self.reduce_outer(&value, ReduceKind::Sum, key, "sum")?;
        match args.get(1, "start") {
            Some(start) => {
                let start = start.clone();
                self.apply_zip(ZipOp::Add, &start, &reduced, key)
            }
            None => Ok(reduced),
        }
    }

    /// `min(a, b, …)` and `max(v)`: several arguments select between scalars,
    /// one argument reduces (`builtins.rs`).
    fn builtin_minmax(&mut self, args: &Args, key: &ValueKey, max: bool) -> Result<Desc, Reject> {
        let kind = if max {
            ReduceKind::Max
        } else {
            ReduceKind::Min
        };
        if args.pos.len() >= 2 {
            let mut result = args.pos[0].clone();
            for candidate in &args.pos[1..] {
                let op = if max { ZipOp::Max } else { ZipOp::Min };
                result = self.apply_zip(op, &result, candidate, key)?;
            }
            return Ok(result);
        }
        let value = args.require(Rule::EReduceScalar, 0, "iterable")?.clone();
        self.reduce_outer(&value, kind, key, if max { "max" } else { "min" })
    }

    /// The reduction a builtin performs: over every element when the value is
    /// flat, along axis 0 when its elements are themselves sequences.
    fn reduce_outer(
        &mut self,
        value: &Desc,
        kind: ReduceKind,
        key: &ValueKey,
        base: &str,
    ) -> Result<Desc, Reject> {
        let a = self.as_array(value, Rule::EReduceScalar)?;
        if a.rank() >= 2 {
            return e_reduce_axis(&mut self.ctx(), &a, kind, &[0], false);
        }
        e_reduce_scalar(&mut self.ctx(), &a, kind, key, base)
    }

    // ── `S-ListMethod` (§5.11) ────────────────────────────────────────

    /// The list methods: each changes the composite's length by a fixed
    /// amount and rebinds the name (§5.11).
    ///
    /// A method that changes the length is only meaningful where the path
    /// condition folds to `true`; see [`Walker::require_static_length`].
    pub(super) fn list_method(
        &mut self,
        name: &str,
        receiver: Desc,
        receiver_key: Option<ValueKey>,
        method: ListMethod,
        args: Args,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        // `.copy()` of an array is that array: the baseline's list-method
        // table has a `StaticArray` branch for exactly this
        // (`src/ir_gen/list_methods.rs`).
        if method == ListMethod::Copy && receiver.as_array().is_some() {
            self.ctx().applied(Rule::ECopy);
            return Ok(receiver);
        }
        if method.changes_length() {
            self.require_static_length(name, &format!("`{}`", method_name(method)))?;
        }
        let kind = receiver
            .as_composite()
            .map_or(CompositeKind::List, |c| c.kind);
        let mut elems: Vec<Desc> = self
            .iterable_elements(&receiver, receiver_key.as_ref(), "a list method")?
            .into_iter()
            .map(|element| element.desc)
            .collect();
        // `count` and `index` read the elements rather than changing them, so
        // they return a data scalar and leave the binding alone (§5.11).
        if matches!(method, ListMethod::Count | ListMethod::Index) {
            let range = VarRange::of_dtype(&mut self.an.terms, Dtype::Idx);
            let tt = self.ctx().tt();
            return Ok(self.ctx().data_scalar(
                Rule::SListMethod,
                Dtype::Idx,
                "count",
                key,
                range,
                tt,
            ));
        }
        if method == ListMethod::Copy {
            self.ctx().applied(Rule::ECopy);
            return Ok(e_composite(kind, elems));
        }
        let result = self.mutate_list(method, &mut elems, &args, key)?;
        // The receiver's new version takes one further key, after the call's
        // arguments, exactly as an assignment's binding does (§5a rule 3);
        // §5a before step 7 gave this rebinding no key at all.
        let version = self.an.keys.next_expr();
        // The new version is made here, so it takes the ψ of this state (§2).
        let updated = super::created_at(e_composite(kind, elems), self.path());
        let mut operands: Vec<ValueKey> = receiver_key.into_iter().collect();
        operands.extend(args.operand_keys());
        let producer = Producer::Applied {
            rules: vec![Rule::SListMethod],
            operands,
            condition: None,
        };
        self.record(version.clone(), None, &updated, producer);
        self.frame_mut().set(name, updated, Some(version));
        Ok(result)
    }

    /// §3.3: a composite's length is a compile-time constant, so an
    /// operation that changes it may not run under a condition.
    ///
    /// The path condition of a statement inside a guarded iteration
    /// (`S-For-Dyn`, `S-For-Range-Bounded`, `S-While-Bounded`), inside a
    /// data-dependent `if`, or after a conditional `break` does not fold to
    /// `true`, and the length would then depend on a runtime value — which a
    /// composite cannot express. A `while i < 10` guard *does* fold, once per
    /// iteration, so an unrolled counting loop is unaffected.
    ///
    /// The message names the list and the condition, because the fix is
    /// either to make the condition static or to build an array, whose length
    /// may be value-sized (§5.9).
    pub(super) fn require_static_length(
        &mut self,
        name: &str,
        operation: &str,
    ) -> Result<(), Reject> {
        let path = self.path();
        if self.an.terms.is_true(path) {
            return Ok(());
        }
        Err(Reject::ValueSizedWithoutMaximum {
            construct: format!(
                "{} changes the length of `{}` under the condition `{}`, so the length \
                 would depend on a runtime value; §3.3 makes a composite's length a \
                 compile-time constant — build an array instead (`np.zeros(n)`), whose \
                 length may be value-sized (§5.9)",
                operation,
                name,
                cut(self.an.terms.pretty_bool(path))
            ),
        })
    }

    /// The list methods that change the length, and what each returns
    /// (§5.11's one-line length rules).
    fn mutate_list(
        &mut self,
        method: ListMethod,
        elems: &mut Vec<Desc>,
        args: &Args,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        match method {
            ListMethod::Append => elems.push(args.require(Rule::SListMethod, 0, "object")?.clone()),
            ListMethod::Extend => {
                let other = args.require(Rule::SListMethod, 0, "iterable")?.clone();
                let extra = self.iterable_elements(&other, None, "extend")?;
                elems.extend(extra.into_iter().map(|element| element.desc));
            }
            ListMethod::Insert => {
                let at = self.list_position(args, 0, elems.len())?;
                elems.insert(at, args.require(Rule::SListMethod, 1, "object")?.clone());
            }
            ListMethod::Pop => return self.list_pop(elems, args, key),
            ListMethod::Remove => return self.list_remove(elems, args, key),
            ListMethod::Clear => elems.clear(),
            ListMethod::Reverse => elems.reverse(),
            ListMethod::Copy | ListMethod::Count | ListMethod::Index => {
                unreachable!("the caller handled the methods that do not mutate")
            }
        }
        Ok(Desc::None)
    }

    /// `pop()` and `pop(i)`: length −1, with `defcond = (-k ≤ t_i < k)` for a
    /// runtime index (§5.11).
    fn list_pop(
        &mut self,
        elems: &mut Vec<Desc>,
        args: &Args,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        if elems.is_empty() {
            return Err(Reject::Unsupported {
                construct: "`pop` on an empty list".to_string(),
            });
        }
        let length = elems.len();
        // The popped element is a value the rule returns rather than builds,
        // so it names its rule itself (§5a, I1).
        self.ctx().applied(Rule::ECompositeIndex);
        match args.get(0, "index") {
            None => Ok(elems.remove(length - 1)),
            Some(index) => {
                let index = index.clone();
                let popped = match self.constant_of(&index) {
                    Some(n) => {
                        let at = if n < 0 { n + length as i128 } else { n };
                        let at = usize::try_from(at).map_err(|_| Reject::Unsupported {
                            construct: format!("`pop` index {} out of range", n),
                        })?;
                        if at >= length {
                            return Err(Reject::Unsupported {
                                construct: format!("`pop` index {} out of range", n),
                            });
                        }
                        elems.remove(at)
                    }
                    None => {
                        // A runtime index removes an unknown element: the
                        // result is `E-Composite-Index` over the candidates
                        // (§5.11), and the elements after it shift down.
                        let merged = self.merge_elements(elems, &index, key)?;
                        let term = self.scalar_term(&index, "a list index")?;
                        let size = self.an.terms.constant(length as i128);
                        let at = wrap_index(&mut self.an.terms, term, size);
                        self.shift_elements_down(elems, at, key)?;
                        merged
                    }
                };
                Ok(popped)
            }
        }
    }

    /// `remove(v)`: the first element equal to `v` goes, and the elements
    /// after it shift down (§5.11).
    ///
    /// Which element that is depends on the values, which is why §5.11 leaves
    /// it open — but a composite whose elements carry integer terms says it
    /// exactly: the position is the nested `ite` over the element
    /// comparisons, and `defcond` is that the position is inside the list,
    /// which is Python's `ValueError: list.remove(x): x not in list`. A
    /// composite whose elements carry no term (an array, a `Float`) cannot
    /// say which element matched, and is rejected rather than guessed at.
    fn list_remove(
        &mut self,
        elems: &mut Vec<Desc>,
        args: &Args,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        if elems.is_empty() {
            return Err(Reject::Unsupported {
                construct: "`remove` on an empty list".to_string(),
            });
        }
        let value = args.require(Rule::SListMethod, 0, "value")?.clone();
        let at = self.first_match_position(elems, &value)?;
        let length = self.an.terms.constant(elems.len() as i128);
        let present = self.an.terms.lt(at, length);
        self.ctx()
            .check(Rule::SListMethod.detail("remove"), present);
        self.shift_elements_down(elems, at, key)?;
        Ok(Desc::None)
    }

    /// The position of the first element equal to `value`, or the length when
    /// there is none: `ite(e_0 = v, 0, ite(e_1 = v, 1, …, n))`.
    fn first_match_position(
        &mut self,
        elems: &[Desc],
        value: &Desc,
    ) -> Result<crate::phase0::term::TermId, Reject> {
        let wanted = self.scalar_term(value, "`remove`")?;
        let mut position = self.an.terms.constant(elems.len() as i128);
        for (at, element) in elems.iter().enumerate().rev() {
            let term = self.scalar_term(element, "an element of a list `remove` reads")?;
            let matches = self.an.terms.eq(term, wanted);
            let here = self.an.terms.constant(at as i128);
            position = self.an.terms.ite(matches, here, position);
        }
        Ok(position)
    }

    /// The integer term of a scalar `S-ListMethod` has to compare.
    fn scalar_term(
        &mut self,
        desc: &Desc,
        what: &str,
    ) -> Result<crate::phase0::term::TermId, Reject> {
        desc.as_scalar()
            .and_then(|s| s.term.as_int())
            .ok_or_else(|| Reject::Premise {
                rule: Rule::SListMethod,
                detail: format!("{} has no integer term", what),
            })
    }

    /// Drop the element at the term `at` and shift the rest down: element `k`
    /// of the result is `ite(at ≤ k, old[k+1], old[k])` (§6.1's merge, taken
    /// element by element).
    ///
    /// A constant `at` folds away, so `pop(0)` on a three-element list still
    /// gives the last two elements unchanged.
    fn shift_elements_down(
        &mut self,
        elems: &mut Vec<Desc>,
        at: crate::phase0::term::TermId,
        key: &ValueKey,
    ) -> Result<(), Reject> {
        let mut shifted = Vec::with_capacity(elems.len().saturating_sub(1));
        for k in 0..elems.len() - 1 {
            let here = self.an.terms.constant(k as i128);
            let moved = self.an.terms.le(at, here);
            let base = format!("shift_{}", k);
            shifted.push(merge_desc(
                &mut self.ctx(),
                moved,
                &elems[k + 1],
                &elems[k],
                key,
                &base,
            )?);
        }
        *elems = shifted;
        Ok(())
    }

    /// The merge of every element of a composite, for a runtime index
    /// (§5.11's `E-Composite-Index` with a runtime `i`).
    fn merge_elements(
        &mut self,
        elems: &[Desc],
        index: &Desc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let scalar = index
            .as_scalar()
            .cloned()
            .ok_or_else(|| Reject::Unsupported {
                construct: "`pop` with a non-scalar index".to_string(),
            })?;
        let composite = CompositeDesc {
            kind: CompositeKind::List,
            elems: elems.to_vec(),
        };
        e_composite_index(&mut self.ctx(), &composite, &scalar, key, "pop")
    }

    /// The insertion position of `insert(i, x)`, clamped into `[0, len]` as
    /// Python clamps it.
    fn list_position(&mut self, args: &Args, index: usize, length: usize) -> Result<usize, Reject> {
        let argument = args.require(Rule::SListMethod, index, "index")?.clone();
        let value = self
            .constant_of(&argument)
            .ok_or_else(|| Reject::Unsupported {
                construct: "`insert` with a runtime index".to_string(),
            })?;
        let shifted = if value < 0 {
            value + length as i128
        } else {
            value
        };
        Ok(shifted.clamp(0, length as i128) as usize)
    }

    /// `a.tolist()`: the nested list Python returns (`coverage.md` §5.3,
    /// "`tolist` becomes a Composite").
    ///
    /// A rank-`r` array becomes `r` levels of `List`, the innermost holding
    /// one data scalar per element; a rank-0 array is one scalar. Every
    /// extent has to be a literal: §3.3 makes a composite's length a
    /// compile-time constant, so an extent that is only a term is rejected.
    ///
    /// The elements share the call's value key. Their variables are data
    /// scalars, which no shape term reads, so the key is their provenance and
    /// not something $\theta(u)$ has to resolve one by one
    /// (`acceptance.md` §2.3).
    pub(super) fn tolist(&mut self, value: &Desc, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.as_array(value, Rule::EToList)?;
        let mut extents = Vec::with_capacity(a.rank());
        for (axis, &d) in a.dims.iter().enumerate() {
            let extent = self.an.terms.constant_value(d).ok_or_else(|| {
                Reject::ValueSizedWithoutMaximum {
                    construct: format!(
                        "`tolist` of an array whose axis {} has the runtime extent `{}`: \
                             §3.3 makes a composite's length a compile-time constant",
                        axis,
                        self.an.terms.pretty(d)
                    ),
                }
            })?;
            extents.push(extent.max(0) as u64);
        }
        let elements = extents.iter().try_fold(1u64, |a, &b| a.checked_mul(b));
        self.spend(elements.unwrap_or(u64::MAX))?;
        self.nested_list(a.dtype, &extents, key)
    }

    /// One level of [`Walker::tolist`]'s result: a list per remaining axis,
    /// a data scalar at the leaves.
    fn nested_list(
        &mut self,
        dtype: Dtype,
        extents: &[u64],
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let Some((&count, inner)) = extents.split_first() else {
            let range = VarRange::of_dtype(&mut self.an.terms, dtype);
            let tt = self.ctx().tt();
            return Ok(self
                .ctx()
                .data_scalar(Rule::EToList, dtype, "item", key, range, tt));
        };
        let mut elems = Vec::with_capacity(count as usize);
        for _ in 0..count {
            elems.push(self.nested_list(dtype, inner, key)?);
        }
        Ok(e_composite(CompositeKind::List, elems))
    }

    /// `ravel`, `flatten`, `.flat`: rank 1 of the same size
    /// (`coverage.md` §5.3).
    pub(super) fn ravel(&mut self, value: &Desc, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.as_array(value, Rule::EReshape)?;
        let size = self.size_of(&a, key, "flat");
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EReshape, a.dtype, vec![size], tt))
    }

    /// `size(a)` as a term, or the `σ` variable of §3.4 when the product is
    /// not one.
    pub(super) fn size_of(
        &mut self,
        a: &ArrayDesc,
        key: &ValueKey,
        base: &str,
    ) -> crate::phase0::term::TermId {
        crate::phase0::rules::size_term(&mut self.ctx(), &a.dims, key, base)
    }
}

/// The source spelling of a list method, for a rejection message.
fn method_name(method: ListMethod) -> &'static str {
    match method {
        ListMethod::Append => "append",
        ListMethod::Extend => "extend",
        ListMethod::Insert => "insert",
        ListMethod::Pop => "pop",
        ListMethod::Remove => "remove",
        ListMethod::Clear => "clear",
        ListMethod::Reverse => "reverse",
        ListMethod::Copy => "copy",
        ListMethod::Count => "count",
        ListMethod::Index => "index",
    }
}

/// Shorten a pretty-printed condition so that a rejection message stays
/// readable; the full term is in the export either way.
fn cut(text: String) -> String {
    const LIMIT: usize = 160;
    match text.char_indices().nth(LIMIT) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text,
    }
}

/// An integer constant descriptor, for a value that does not fit `i64`.
fn e_const_i128(w: &mut Walker, value: i128) -> Desc {
    let term = w.an.terms.constant(value);
    let tt = w.ctx().tt();
    w.ctx().scalar(
        Rule::EConst,
        Dtype::Idx,
        ScalarTerm::Int(term),
        Origin::Shape,
        tt,
    )
}

/// The [`Dtype`] a `dtype=` keyword or an `astype` argument names.
///
/// This is the one place the annotation and keyword spellings — `bool`,
/// `np.bool_`, `int`, `np.int64`, `float`, `np.float64`, `complex`,
/// `a.dtype` — become a `Dtype`. Every one of them reaches the walker as
/// `Desc::Class` of a [`ZinniaType`]: the name table of [`super::GLOBAL_DTYPES`]
/// binds the bare names, [`crate::phase0::ops::Class::DtypeClass`] the
/// `np.*` ones, and `E-Property`'s `Dtype` case `a.dtype`.
///
/// `None` means "not given": an absent keyword or an explicit `dtype=None`,
/// both of which leave the caller's fallback in place. Anything else is a
/// dtype phase 0 cannot read, and the call is rejected rather than silently
/// given the fallback — a wrong dtype turns a mask into an index array
/// (`implementation.md`, "Consistency findings" 3).
fn dtype_of_value(d: &Desc) -> Result<Option<Dtype>, Reject> {
    match d {
        Desc::None => Ok(None),
        Desc::Class(class) => match Dtype::from_zinnia_type(class) {
            Some(dtype) => Ok(Some(dtype)),
            None => Err(Reject::Unsupported {
                construct: format!("`dtype={}`, which is not an element dtype", class),
            }),
        },
        other => Err(Reject::Unsupported {
            construct: format!(
                "a `dtype=` given as a {}; phase 0 reads a dtype object, not a value",
                other.kind_str()
            ),
        }),
    }
}

/// The dtype a `dtype=` keyword selects, defaulting to `fallback`.
///
/// `dt=` is the spelling `DynamicNDArray.zeros` uses (`coverage.md` §5.4).
pub(super) fn dtype_kwarg(args: &Args, fallback: Dtype) -> Result<Dtype, Reject> {
    let Some(given) = args.kwarg("dtype").or_else(|| args.kwarg("dt")) else {
        return Ok(fallback);
    };
    Ok(dtype_of_value(given)?.unwrap_or(fallback))
}

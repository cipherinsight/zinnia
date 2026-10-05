//! The array shape classes: the transforms of `definition.md` §5.6, the
//! reductions of §5.7 and the constructors of §5.8 (`coverage.md` §5.2 to
//! §5.4).
//!
//! Every handler is the same three steps: read the arguments through
//! [`Args`], turn the constant ones into axes or extents, and call the
//! transfer rule. The gather and linear-algebra classes continue in
//! [`super::gather`].

use crate::phase0::desc::{ArrayDesc, Desc, Dtype, ScalarDesc, ValueKey};
use crate::phase0::ops::{Class, StackKind};
use crate::phase0::rules::{
    broadcast_dims, e_concat, e_cumsum, e_expand_dims, e_moveaxis, e_reduce_axis, e_reduce_scalar,
    e_repeat, e_reshape, e_reverse, e_sort, e_squeeze, e_stack, e_transpose, e_unique, ReduceKind,
    ReshapeExtent, VarRange,
};
use crate::phase0::term::{BoolId, TermId};
use crate::phase0::Reject;

use super::calls::Args;
use super::Walker;
use crate::phase0::rules::Rule;

impl Walker<'_> {
    /// The array half of [`Walker::apply_class`].
    pub(super) fn apply_array_class(
        &mut self,
        class: Class,
        args: &Args,
        key: &ValueKey,
        member: &str,
    ) -> Result<Desc, Reject> {
        match class {
            Class::Reduce(kind) => self.reduce(kind, args, key, member),
            Class::Cumulative => self.cumulative(args, key),
            Class::Percentile => self.percentile(args, key, member),
            Class::LinalgNorm => {
                let tt = self.ctx().tt();
                self.scalar_statistic(args, key, member, tt)
            }
            Class::Reshape => self.reshape(args, key),
            Class::Ravel => {
                let value = args.require(Rule::EReshape, 0, "a")?.clone();
                self.ravel(&value, key)
            }
            Class::ToList => {
                let value = args.require(Rule::EToList, 0, "a")?.clone();
                self.tolist(&value, key)
            }
            Class::Transpose => self.transpose(args),
            Class::Swapaxes => self.swapaxes(args),
            Class::Moveaxis => {
                let a = self.arg_array(args, Rule::EMoveaxis, 0, "a")?;
                let src = self.require_axis(args, 1, "source")?;
                let dst = self.require_axis(args, 2, "destination")?;
                e_moveaxis(&mut self.ctx(), &a, src, dst)
            }
            Class::Squeeze => {
                let a = self.arg_array(args, Rule::ESqueeze, 0, "a")?;
                let axis = self.optional_axis(args, 1, "axis")?;
                e_squeeze(&mut self.ctx(), &a, axis)
            }
            Class::ExpandDims => {
                let a = self.arg_array(args, Rule::ENewAxis, 0, "a")?;
                let axis = self.require_axis(args, 1, "axis")?;
                e_expand_dims(&mut self.ctx(), &a, axis)
            }
            Class::BroadcastTo => self.broadcast_to(args),
            Class::AtLeast(rank) => self.at_least(args, rank),
            Class::Reverse => {
                let a = self.arg_array(args, Rule::EReverse, 0, "m")?;
                e_reverse(&mut self.ctx(), &a)
            }
            Class::Rot90 => self.rot90(args),
            Class::Concat => self.concatenate(args, key),
            Class::Stack => self.stack(args),
            Class::StackFamily(kind) => self.stack_family(args, kind),
            Class::Diag => self.diag(args),
            Class::Delete => self.delete(args, key),
            Class::Insert => self.insert(args, key),
            Class::Pad => self.pad(args),
            Class::Repeat => self.repeat(args, key),
            Class::Tile => self.tile(args),
            Class::Diff => self.diff(args),
            Class::Sort { index_result } => self.sort(args, index_result, member),
            Class::Unique => self.unique(args, key),
            Class::Zeros | Class::Full => self.zeros(args, key, class == Class::Full),
            Class::Like => self.like(args),
            Class::Identity => self.identity(args),
            Class::Arange => self.arange(args, key),
            Class::Linspace => self.linspace(args),
            Class::Asarray => self.asarray(args),
            Class::Indices => self.indices(args),
            other => self.apply_gather_class(other, args, key, member),
        }
    }

    /// `np.percentile` and `np.linalg.norm`: the whole array reduces to one
    /// real number, whatever its rank (`coverage.md` §5.2, §5.6).
    fn scalar_statistic(
        &mut self,
        args: &Args,
        key: &ValueKey,
        member: &str,
        defcond: BoolId,
    ) -> Result<Desc, Reject> {
        self.arg_array(args, Rule::EReduceScalar, 0, "a")?;
        let range = VarRange::of_dtype(&mut self.an.terms, Dtype::Float);
        Ok(self.ctx().data_scalar(
            Rule::EReduceScalar,
            Dtype::Float,
            member,
            key,
            range,
            defcond,
        ))
    }

    /// `np.percentile(a, q)`: a statistic with a domain.
    ///
    /// NumPy raises `ValueError: Percentiles must be in the range [0, 100]`,
    /// so `0 ≤ q ≤ 100` is the definedness condition. A `q` that carries a
    /// term states it exactly; a real or array `q` carries none (§3.2), and
    /// the condition is then an unconstrained Boolean of the `E-Take` kind
    /// (§5.4).
    fn percentile(&mut self, args: &Args, key: &ValueKey, member: &str) -> Result<Desc, Reject> {
        let q = args.require(Rule::EReduceScalar, 1, "q")?.clone();
        let domain = match q.as_scalar().and_then(|s| s.term.as_int()) {
            Some(t) => {
                let zero = self.an.terms.constant(0);
                let hundred = self.an.terms.constant(100);
                let low = self.an.terms.le(zero, t);
                let high = self.an.terms.le(t, hundred);
                self.an.terms.and(low, high)
            }
            None => self.ctx().definedness("in_percentile", key),
        };
        self.scalar_statistic(args, key, member, domain)
    }

    // ── Argument helpers ──────────────────────────────────────────────

    /// The argument at `index` as an array (§5.6's operand).
    pub(super) fn arg_array(
        &mut self,
        args: &Args,
        rule: Rule,
        index: usize,
        name: &str,
    ) -> Result<ArrayDesc, Reject> {
        let value = args.require(rule, index, name)?.clone();
        let (array, _) = self.as_array_keyed(&value, args.key(index, name), rule)?;
        Ok(array)
    }

    /// The parts of a `concatenate` / `stack` call: one composite argument,
    /// or the arrays given positionally.
    fn arg_parts(&mut self, args: &Args, rule: Rule) -> Result<Vec<ArrayDesc>, Reject> {
        let first = args.require(rule, 0, "arrays")?.clone();
        let values = match first.as_composite() {
            Some(c) => c.elems.clone(),
            None => vec![first],
        };
        let mut parts = Vec::with_capacity(values.len());
        for value in &values {
            parts.push(self.as_array(value, rule)?);
        }
        Ok(parts)
    }

    /// A constant axis argument the rule cannot do without.
    fn require_axis(&mut self, args: &Args, index: usize, name: &str) -> Result<i64, Reject> {
        self.optional_axis(args, index, name)?
            .ok_or_else(|| Reject::Unsupported {
                construct: format!("a call without its constant `{}`", name),
            })
    }

    /// A constant axis argument, absent or `None` when the call omits it.
    fn optional_axis(
        &mut self,
        args: &Args,
        index: usize,
        name: &str,
    ) -> Result<Option<i64>, Reject> {
        match args.get(index, name) {
            None | Some(Desc::None) => Ok(None),
            Some(value) => {
                let constant = self.constant_of(value).ok_or_else(|| Reject::Unsupported {
                    construct: format!("a non-constant `{}`", name),
                })?;
                Ok(Some(constant as i64))
            }
        }
    }

    /// An `axis` argument that may be an integer or a tuple of integers
    /// (`coverage.md` §5.2). `None` means "every axis".
    fn axis_set(&mut self, args: &Args, index: usize) -> Result<Option<Vec<i64>>, Reject> {
        match args.get(index, "axis") {
            None | Some(Desc::None) => Ok(None),
            Some(Desc::Composite(c)) => {
                let elems = c.elems.clone();
                let mut axes = Vec::with_capacity(elems.len());
                for element in &elems {
                    axes.push(
                        self.constant_of(element)
                            .ok_or_else(|| Reject::Unsupported {
                                construct: "a non-constant axis in an axis tuple".to_string(),
                            })? as i64,
                    );
                }
                Ok(Some(axes))
            }
            Some(value) => {
                let constant = self.constant_of(value).ok_or_else(|| Reject::Unsupported {
                    construct: "a non-constant `axis`".to_string(),
                })?;
                Ok(Some(vec![constant as i64]))
            }
        }
    }

    /// Whether a keyword flag such as `keepdims` is set.
    fn flag(&self, args: &Args, name: &str) -> bool {
        args.kwarg(name)
            .and_then(|d| self.constant_of(d))
            .map(|n| n != 0)
            .unwrap_or(false)
    }

    /// The scalars of a shape argument: one extent, or a sequence of them
    /// ([`super::entries`]).
    pub(super) fn shape_scalars(&mut self, value: &Desc) -> Result<Vec<ScalarDesc>, Reject> {
        super::entries(value)
            .iter()
            .map(|entry| {
                entry
                    .as_scalar()
                    .cloned()
                    .ok_or_else(|| Reject::Unsupported {
                        construct: format!("a shape entry given as a {}", entry.kind_str()),
                    })
            })
            .collect()
    }

    /// The constant integers of a shape-like argument.
    fn constant_ints(&mut self, value: &Desc) -> Result<Vec<i128>, Reject> {
        let scalars = self.shape_scalars(value)?;
        scalars
            .iter()
            .map(|s| {
                s.term
                    .as_int()
                    .and_then(|t| self.an.terms.constant_value(t))
                    .ok_or_else(|| Reject::Unsupported {
                        construct: "a non-constant entry where a constant is required".to_string(),
                    })
            })
            .collect()
    }

    // ── Reductions (§5.2, §5.7) ───────────────────────────────────────

    /// `E-Reduce-Scalar` (§5.2) and `E-Reduce-Axis` (§5.7).
    ///
    /// A `dtype=` keyword is not read here: the baseline ignores it for `sum`
    /// and `prod` and rejects it for `var` and `std`, so honouring it would
    /// be a dtype divergence with no shape consequence and no baseline
    /// behaviour to be consistent with.
    fn reduce(
        &mut self,
        kind: ReduceKind,
        args: &Args,
        key: &ValueKey,
        base: &str,
    ) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EReduceScalar, 0, "a")?;
        let axes = self.axis_set(args, 1)?;
        let keepdims = self.flag(args, "keepdims");
        match axes {
            None if !keepdims => e_reduce_scalar(&mut self.ctx(), &a, kind, key, base),
            None => {
                let all: Vec<i64> = (0..a.rank() as i64).collect();
                e_reduce_axis(&mut self.ctx(), &a, kind, &all, true)
            }
            Some(axes) => e_reduce_axis(&mut self.ctx(), &a, kind, &axes, keepdims),
        }
    }

    /// `E-Sort` (§5.7): `np.sort(a, axis=k)` and `np.argsort(a, axis=k)`.
    ///
    /// An omitted `axis` sorts the last one and an integer `axis` sorts that
    /// one, both leaving the dimensions alone. `axis=None` flattens first,
    /// which the baseline implements only where the flattening is the
    /// identity — a rank-1 operand — and rejects on rank ≥ 2
    /// (`helpers/static_array/sorting.rs::try_apply_sort`). Phase 0 rejects
    /// the same form, so that no program it accepts has a shape the baseline
    /// will not build.
    fn sort(&mut self, args: &Args, index_result: bool, member: &str) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ESort, 0, "a")?;
        if matches!(args.get(1, "axis"), Some(Desc::None)) && a.rank() > 1 {
            return Err(Reject::Unsupported {
                construct: format!("np.{} with axis=None on a rank-{} array", member, a.rank()),
            });
        }
        e_sort(&mut self.ctx(), &a, index_result)
    }

    /// `cumsum` / `cumprod`: the dimensions are unchanged along an axis, and
    /// the array is flattened first when `axis` is omitted (§5.7).
    fn cumulative(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let value = args.require(Rule::ECumsum, 0, "a")?.clone();
        let a = self.as_array(&value, Rule::ECumsum)?;
        match self.optional_axis(args, 1, "axis")? {
            Some(_) => e_cumsum(&mut self.ctx(), &a),
            None if a.dims.len() == 1 => e_cumsum(&mut self.ctx(), &a),
            // NumPy scans the flattened array: `E-Reshape` to `[size(a)]`,
            // then `E-Cumsum` over it; Δ records both rules.
            None => {
                let flat = self.ravel(&value, key)?;
                let flat = self.as_array(&flat, Rule::ECumsum)?;
                e_cumsum(&mut self.ctx(), &flat)
            }
        }
    }

    // ── Shape transforms (§5.6) ───────────────────────────────────────

    /// `E-Reshape` (§5.6). The target is one tuple argument or several
    /// integer arguments, and may contain one `-1`.
    fn reshape(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EReshape, 0, "a")?;
        let rest = args.rest(1);
        let entries: Vec<Desc> = match rest {
            [] => match args.kwarg("shape") {
                Some(shape) => super::entries(shape),
                None => {
                    return Err(Reject::ReshapeNotSupported {
                        detail: "reshape without a target shape".to_string(),
                    })
                }
            },
            [only] => super::entries(only),
            many => many.to_vec(),
        };
        let mut extents = Vec::with_capacity(entries.len());
        for entry in &entries {
            extents.push(self.reshape_extent(entry)?);
        }
        e_reshape(&mut self.ctx(), &a, &extents, key)
    }

    /// One entry of a reshape target: a literal, the single `-1`, or a
    /// shape-derived term such as `a.shape[0]`.
    fn reshape_extent(&mut self, entry: &Desc) -> Result<ReshapeExtent, Reject> {
        let scalar = entry
            .as_scalar()
            .ok_or_else(|| Reject::ReshapeNotSupported {
                detail: format!("a reshape extent given as a {}", entry.kind_str()),
            })?;
        let term = scalar
            .term
            .as_int()
            .ok_or_else(|| Reject::ReshapeNotSupported {
                detail: "a reshape extent with no term".to_string(),
            })?;
        match self.an.terms.constant_value(term) {
            Some(-1) => Ok(ReshapeExtent::Infer),
            Some(n) => Ok(ReshapeExtent::Const(n as i64)),
            None => Ok(ReshapeExtent::Term(term)),
        }
    }

    fn transpose(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ETranspose, 0, "a")?;
        let rest = args.rest(1);
        let axes: Option<Vec<i64>> = match (rest, args.kwarg("axes")) {
            ([], None) => None,
            ([], Some(value)) => Some(
                self.constant_ints(value)?
                    .iter()
                    .map(|&n| n as i64)
                    .collect(),
            ),
            ([only], _) => Some(
                self.constant_ints(only)?
                    .iter()
                    .map(|&n| n as i64)
                    .collect(),
            ),
            (many, _) => {
                let mut axes = Vec::with_capacity(many.len());
                for entry in many {
                    axes.extend(self.constant_ints(entry)?.iter().map(|&n| n as i64));
                }
                Some(axes)
            }
        };
        e_transpose(&mut self.ctx(), &a, axes.as_deref())
    }

    fn swapaxes(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ETranspose, 0, "a")?;
        let first = self.require_axis(args, 1, "axis1")?;
        let second = self.require_axis(args, 2, "axis2")?;
        let rank = a.rank();
        let mut axes: Vec<i64> = (0..rank as i64).collect();
        let (i, j) = (
            crate::phase0::rules::helpers::normalise_axis(Rule::EMoveaxis, first, rank)?,
            crate::phase0::rules::helpers::normalise_axis(Rule::EMoveaxis, second, rank)?,
        );
        axes.swap(i, j);
        e_transpose(&mut self.ctx(), &a, Some(&axes))
    }

    /// `broadcast_to(a, shape)`: the result has the target shape and the
    /// broadcasting condition of §5.3 is the definedness condition.
    fn broadcast_to(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EBroadcast, 0, "array")?;
        let shape = args.require(Rule::EBroadcast, 1, "shape")?.clone();
        let scalars = self.shape_scalars(&shape)?;
        let mut dims = Vec::with_capacity(scalars.len());
        for extent in &scalars {
            dims.push(extent.term.as_int().ok_or_else(|| Reject::Unsupported {
                construct: "a broadcast extent with no term".to_string(),
            })?);
        }
        let (_, defcond) = broadcast_dims(&mut self.an.terms, &[&a.dims, &dims]);
        Ok(self.ctx().array(Rule::EBroadcast, a.dtype, dims, defcond))
    }

    /// `atleast_1d` / `2d` / `3d`: unit axes are added until the rank
    /// reaches the target (`coverage.md` §5.3).
    fn at_least(&mut self, args: &Args, rank: usize) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ENewAxis, 0, "a")?;
        let one = self.an.terms.constant(1);
        let mut dims = a.dims.clone();
        // NumPy appends the third axis and prepends the others.
        while dims.len() < rank {
            if rank == 3 && dims.len() == 2 {
                dims.push(one);
            } else {
                dims.insert(0, one);
            }
        }
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::ENewAxis, a.dtype, dims, tt))
    }

    /// `rot90(m, k)`: an odd quarter turn swaps the two axes.
    fn rot90(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ETranspose, 0, "m")?;
        let k = self.optional_axis(args, 1, "k")?.unwrap_or(1).rem_euclid(4);
        if a.rank() < 2 {
            return Err(Reject::Premise {
                rule: Rule::ETranspose,
                detail: "as rot90 of an array of rank below 2".to_string(),
            });
        }
        if k % 2 == 0 {
            return e_reverse(&mut self.ctx(), &a);
        }
        let mut axes: Vec<i64> = (0..a.rank() as i64).collect();
        axes.swap(0, 1);
        e_transpose(&mut self.ctx(), &a, Some(&axes))
    }

    /// `E-Concat` (§5.6). `axis=None` is NumPy's flattening form: every
    /// operand is raveled first and the results are joined into one rank-1
    /// array of `Σ size`, whatever the operands' ranks. An *absent* `axis` is
    /// axis 0, so the two spellings have to be told apart.
    fn concatenate(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let parts = self.arg_parts(args, Rule::EConcat)?;
        let axis = match args.get(1, "axis") {
            None => Some(0),
            Some(Desc::None) => None,
            Some(value) => {
                let value = value.clone();
                Some(
                    self.constant_of(&value)
                        .ok_or_else(|| Reject::Unsupported {
                            construct: "a non-constant `axis`".to_string(),
                        })? as i64,
                )
            }
        };
        let Some(axis) = axis else {
            return self.concatenate_flat(&parts, key);
        };
        let refs: Vec<&ArrayDesc> = parts.iter().collect();
        e_concat(&mut self.ctx(), &refs, axis)
    }

    /// `np.concatenate(..., axis=None)`: rank 1 of length `Σ size(part)`.
    fn concatenate_flat(&mut self, parts: &[ArrayDesc], key: &ValueKey) -> Result<Desc, Reject> {
        let first = parts.first().ok_or_else(|| Reject::Unsupported {
            construct: "E-Concat of an empty list".to_string(),
        })?;
        let dtype = parts
            .iter()
            .map(|p| p.dtype)
            .reduce(|a, b| Dtype::promote(a, b).unwrap_or(a))
            .unwrap_or(first.dtype);
        let mut total = self.an.terms.constant(0);
        for part in parts {
            let size = self.size_of(part, key, "sigma_concat");
            total = self.an.terms.add(total, size);
        }
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EConcat, dtype, vec![total], tt))
    }

    fn stack(&mut self, args: &Args) -> Result<Desc, Reject> {
        let parts = self.arg_parts(args, Rule::EStack)?;
        let axis = self.optional_axis(args, 1, "axis")?.unwrap_or(0);
        let refs: Vec<&ArrayDesc> = parts.iter().collect();
        e_stack(&mut self.ctx(), &refs, axis)
    }

    /// `vstack`, `hstack`, `dstack`, `column_stack` (`coverage.md` §5.3):
    /// each promotes its operands to a rank and concatenates on one axis.
    ///
    /// `dstack` promotes with NumPy's `atleast_3d`, which is not "prepend 1s":
    /// it sends `(n,)` to `(1, n, 1)` and `(m, n)` to `(m, n, 1)`, appending
    /// the new *last* axis and prepending only for a rank-1 operand.
    fn stack_family(&mut self, args: &Args, kind: StackKind) -> Result<Desc, Reject> {
        let parts = self.arg_parts(args, Rule::EConcat)?;
        let max_rank = parts.iter().map(|p| p.rank()).max().unwrap_or(0);
        let (target_rank, axis, promotion) = match kind {
            StackKind::Vertical => (2, 0, Promotion::Prepend),
            StackKind::Horizontal if max_rank >= 2 => (2, 1, Promotion::Prepend),
            StackKind::Horizontal => (1, 0, Promotion::Prepend),
            StackKind::Depth => (3, 2, Promotion::AtLeast3d),
            StackKind::Column => (2, 1, Promotion::Append),
        };
        let one = self.an.terms.constant(1);
        let promoted: Vec<ArrayDesc> = parts
            .iter()
            .map(|p| promotion.apply(p, target_rank, one))
            .collect();
        let refs: Vec<&ArrayDesc> = promoted.iter().collect();
        e_concat(&mut self.ctx(), &refs, axis)
    }

    /// `np.diag`: a rank-1 array becomes a square matrix, a rank-2 one its
    /// diagonal (`coverage.md` §5.3).
    fn diag(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EDiag, 0, "v")?;
        let tt = self.ctx().tt();
        match a.rank() {
            1 => {
                let d = a.dims[0];
                Ok(self.ctx().array(Rule::EDiag, a.dtype, vec![d, d], tt))
            }
            2 => {
                let d = self.an.terms.min(a.dims[0], a.dims[1]);
                Ok(self.ctx().array(Rule::EDiag, a.dtype, vec![d], tt))
            }
            other => Err(Reject::Premise {
                rule: Rule::EDiag,
                detail: format!("of a rank-{} array", other),
            }),
        }
    }

    /// `E-Unique` (§5.7), with the `return_inverse` flag the baseline
    /// supports: the inverse has one entry per element of the input.
    fn unique(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EUnique, 0, "ar")?;
        let unique = e_unique(&mut self.ctx(), &a, key, "c_unique")?;
        if !self.flag(args, "return_inverse") {
            return Ok(unique);
        }
        let size = self.size_of(&a, key, "sigma_unique");
        let tt = self.ctx().tt();
        let inverse = self.ctx().array(Rule::EUnique, Dtype::Idx, vec![size], tt);
        Ok(crate::phase0::rules::e_composite(
            crate::phase0::desc::CompositeKind::Tuple,
            vec![unique, inverse],
        ))
    }

    /// `np.delete(arr, obj, axis)`: the axis shrinks by the number of removed
    /// positions (`coverage.md` §5.3).
    ///
    /// A position has to name an existing element, `-d ≤ obj < d`, which is
    /// the same condition `E-Index` records (§5.4).
    fn delete(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EDelete, 0, "arr")?;
        let obj = args.require(Rule::EDelete, 1, "obj")?.clone();
        let removed = self.position_count(&obj)?;
        let at = self.position_axis(args, 2, a.rank())?;
        let defcond = self.positions_in_range(&obj, a.dims[at], PositionBound::Element, key)?;
        let mut dims = a.dims.clone();
        let count = self.an.terms.constant(removed);
        let shorter = self.an.terms.sub(dims[at], count);
        let zero = self.an.terms.constant(0);
        dims[at] = self.an.terms.max(zero, shorter);
        Ok(self.ctx().array(Rule::EDelete, a.dtype, dims, defcond))
    }

    /// `np.insert(arr, obj, values, axis)`: the axis grows by the number of
    /// inserted positions (`coverage.md` §5.3).
    ///
    /// A position names a gap rather than an element, so `d` itself is legal
    /// and the condition is `-d ≤ obj ≤ d`; NumPy raises `IndexError`
    /// otherwise.
    fn insert(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EInsert, 0, "arr")?;
        let obj = args.require(Rule::EInsert, 1, "obj")?.clone();
        let inserted = self.position_count(&obj)?;
        let at = self.position_axis(args, 3, a.rank())?;
        let defcond = self.positions_in_range(&obj, a.dims[at], PositionBound::Gap, key)?;
        let mut dims = a.dims.clone();
        let count = self.an.terms.constant(inserted);
        dims[at] = self.an.terms.add(dims[at], count);
        Ok(self.ctx().array(Rule::EInsert, a.dtype, dims, defcond))
    }

    /// The condition that every position a `delete` / `insert` `obj` names
    /// lies in the axis.
    ///
    /// A scalar position carries a term, so the condition is that term's; a
    /// position read out of an array is an element value, which $\Phi$ does
    /// not model, so it is an unconstrained Boolean of the `E-Take` kind
    /// (§5.4).
    fn positions_in_range(
        &mut self,
        obj: &Desc,
        extent: TermId,
        bound: PositionBound,
        key: &ValueKey,
    ) -> Result<BoolId, Reject> {
        match obj {
            Desc::Scalar(s) => match s.term.as_int() {
                Some(t) => Ok(bound.condition(&mut self.an.terms, t, extent)),
                None => Ok(self.ctx().definedness("in_position", key)),
            },
            Desc::Composite(c) => {
                let elems = c.elems.clone();
                let mut conditions = Vec::with_capacity(elems.len());
                for element in &elems {
                    conditions.push(self.positions_in_range(element, extent, bound, key)?);
                }
                Ok(self.an.terms.all(&conditions))
            }
            Desc::Array(_) => Ok(self.ctx().definedness("in_position", key)),
            other => Err(Reject::Unsupported {
                construct: format!("delete/insert with a {} position", other.kind_str()),
            }),
        }
    }

    /// How many positions a `delete` / `insert` `obj` names. Only the count
    /// matters for the shape, so a runtime position is fine; a runtime
    /// *number* of positions is not.
    fn position_count(&mut self, obj: &Desc) -> Result<i128, Reject> {
        match obj {
            Desc::Scalar(_) => Ok(1),
            Desc::Composite(c) => Ok(c.elems.len() as i128),
            Desc::Array(a) => a
                .dims
                .first()
                .and_then(|&d| self.an.terms.constant_value(d))
                .ok_or_else(|| Reject::Unsupported {
                    construct: "delete/insert with a runtime number of positions".to_string(),
                }),
            other => Err(Reject::Unsupported {
                construct: format!("delete/insert with a {} position", other.kind_str()),
            }),
        }
    }

    /// The axis of a `delete` / `insert`. NumPy flattens when it is omitted,
    /// which for a rank-1 array is axis 0.
    fn position_axis(&mut self, args: &Args, index: usize, rank: usize) -> Result<usize, Reject> {
        match self.optional_axis(args, index, "axis")? {
            Some(axis) => {
                crate::phase0::rules::helpers::normalise_axis(Rule::EReduceAxis, axis, rank)
            }
            None if rank == 1 => Ok(0),
            None => Err(Reject::Unsupported {
                construct: "delete/insert without an axis on an array of rank above 1".to_string(),
            }),
        }
    }

    /// `np.pad(arr, pad_width)`: every axis grows by its two constant widths
    /// (`coverage.md` §5.3; a non-constant width is a baseline rejection).
    fn pad(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EPad, 0, "array")?;
        let widths = args.require(Rule::EPad, 1, "pad_width")?.clone();
        let per_axis = self.pad_widths(&widths, a.rank())?;
        let mut dims = a.dims.clone();
        for (axis, (before, after)) in per_axis.iter().enumerate() {
            let extra = self.an.terms.constant(before + after);
            dims[axis] = self.an.terms.add(dims[axis], extra);
        }
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EPad, a.dtype, dims, tt))
    }

    /// `pad_width` as `n`, `(before, after)`, or one pair per axis.
    fn pad_widths(&mut self, widths: &Desc, rank: usize) -> Result<Vec<(i128, i128)>, Reject> {
        if let Some(c) = widths.as_composite() {
            if c.elems.iter().all(|e| e.as_composite().is_some()) {
                let elems = c.elems.clone();
                let mut out = Vec::with_capacity(elems.len());
                for pair in &elems {
                    let values = self.constant_ints(pair)?;
                    match values[..] {
                        [before, after] => out.push((before, after)),
                        [both] => out.push((both, both)),
                        _ => {
                            return Err(Reject::Unsupported {
                                construct: "a pad width that is not a pair".to_string(),
                            })
                        }
                    }
                }
                return Ok(out);
            }
        }
        let values = self.constant_ints(widths)?;
        let pair = match values[..] {
            [both] => (both, both),
            [before, after] => (before, after),
            _ => {
                return Err(Reject::Unsupported {
                    construct: "a pad width that is not a pair".to_string(),
                })
            }
        };
        Ok(vec![pair; rank])
    }

    /// `E-Repeat` (§5.8).
    fn repeat(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ERepeat, 0, "a")?;
        let repeats = args.require(Rule::ERepeat, 1, "repeats")?.clone();
        let count = repeats
            .as_scalar()
            .cloned()
            .ok_or_else(|| Reject::Unsupported {
                construct: "np.repeat with a per-element repeat count".to_string(),
            })?;
        let axis = self.optional_axis(args, 2, "axis")?;
        let declared = self.declared_maximum(args);
        e_repeat(&mut self.ctx(), &a, &count, axis, declared, key, "l_repeat")
    }

    /// `np.tile(a, reps)`: each axis is multiplied by its constant repetition
    /// count (`coverage.md` §5.3).
    fn tile(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ETile, 0, "A")?;
        let reps = args.require(Rule::ETile, 1, "reps")?.clone();
        let counts = self.constant_ints(&reps)?;
        let mut dims = a.dims.clone();
        let one = self.an.terms.constant(1);
        while dims.len() < counts.len() {
            dims.insert(0, one);
        }
        let offset = dims.len() - counts.len();
        for (index, &count) in counts.iter().enumerate() {
            dims[offset + index] = self.an.terms.mul_const(count, dims[offset + index]);
        }
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::ETile, a.dtype, dims, tt))
    }

    /// `np.diff(x, n)`: the last axis loses `n` entries (`coverage.md` §5.3).
    fn diff(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EDiff, 0, "x")?;
        let order = self.optional_axis(args, 1, "n")?.unwrap_or(1).max(0);
        let last = a.rank().checked_sub(1).ok_or_else(|| Reject::Premise {
            rule: Rule::EDiff,
            detail: "of a rank-0 array".to_string(),
        })?;
        let mut dims = a.dims.clone();
        let taken = self.an.terms.constant(order as i128);
        let shorter = self.an.terms.sub(dims[last], taken);
        let zero = self.an.terms.constant(0);
        dims[last] = self.an.terms.max(zero, shorter);
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EDiff, a.dtype, dims, tt))
    }
}

/// What a `delete` / `insert` position may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositionBound {
    /// An existing element: `-d ≤ i < d`, as `E-Index` (§5.4).
    Element,
    /// A gap between elements, `d` included: `-d ≤ i ≤ d`.
    Gap,
}

impl PositionBound {
    fn condition(
        self,
        terms: &mut crate::phase0::term::TermArena,
        index: TermId,
        extent: TermId,
    ) -> BoolId {
        let neg = terms.mul_const(-1, extent);
        let low = terms.le(neg, index);
        let high = match self {
            PositionBound::Element => terms.lt(index, extent),
            PositionBound::Gap => terms.le(index, extent),
        };
        terms.and(low, high)
    }
}

/// How a member of the `vstack` family raises an operand to its target rank
/// (`coverage.md` §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Promotion {
    /// Unit axes at the front: `vstack`, `hstack`.
    Prepend,
    /// Unit axes at the back: `column_stack`.
    Append,
    /// NumPy's `atleast_3d`, which `dstack` applies: `(n,)` becomes
    /// `(1, n, 1)` and `(m, n)` becomes `(m, n, 1)`.
    AtLeast3d,
}

impl Promotion {
    fn apply(self, a: &ArrayDesc, target: usize, one: TermId) -> ArrayDesc {
        let mut dims = a.dims.clone();
        match self {
            Promotion::Prepend => {
                while dims.len() < target {
                    dims.insert(0, one);
                }
            }
            Promotion::Append => {
                while dims.len() < target {
                    dims.push(one);
                }
            }
            Promotion::AtLeast3d => {
                if dims.len() < target {
                    if dims.len() == 1 {
                        dims.insert(0, one);
                    }
                    while dims.len() < target {
                        dims.push(one);
                    }
                }
            }
        }
        ArrayDesc {
            dtype: a.dtype,
            dims,
            defcond: a.defcond,
            psi: a.psi,
        }
    }
}

//! The gather and linear-algebra classes: `definition.md` §5.4's `E-Take`,
//! §5.5's `E-Nonzero` and `E-Filter`, and §5.7's `E-Matmul`, together with
//! the statistics of `coverage.md` §5.6.

use crate::phase0::desc::{
    ArrayDesc, CompositeKind, Desc, Dtype, Origin, ScalarTerm, ValueKey,
};
use crate::phase0::ops::{Class, ZipOp};
use crate::phase0::rules::{
    e_composite, e_filter_axis0, e_matmul, e_nonzero, e_take, e_value_sized, e_where3,
    helpers::normalise_axis,
    VarRange,
};
use crate::phase0::term::TermId;
use crate::phase0::Reject;

use super::calls::Args;
use super::Walker;
use crate::phase0::rules::Rule;

impl Walker<'_> {
    /// The gather half of [`Walker::apply_class`].
    pub(super) fn apply_gather_class(
        &mut self,
        class: Class,
        args: &Args,
        key: &ValueKey,
        member: &str,
    ) -> Result<Desc, Reject> {
        match class {
            Class::Take => self.take(args, key),
            Class::TakeAlongAxis => self.take_along_axis(args, key),
            Class::Where => self.where_(args),
            Class::Nonzero => self.nonzero(args, key, false),
            Class::Argwhere => self.argwhere(args, key),
            Class::Flatnonzero => self.nonzero(args, key, true),
            Class::In1d => self.in1d(args),
            Class::Bincount => self.bincount(args, key),
            Class::UnravelIndex => self.unravel_index(args, key),
            Class::RavelMultiIndex => self.ravel_multi_index(args, key),
            Class::Filter => self.filter(args, key),
            Class::Matmul => self.matmul(args, key),
            Class::Tensordot => self.tensordot(args),
            Class::Einsum => self.einsum(args),
            Class::Corrcoef => self.corrcoef(args),
            Class::Correlate => self.correlate(args),
            Class::Outer => self.outer(args, key),
            Class::Clip => self.clip(args),
            Class::IsClose => self.isclose(args, key),
            Class::Predicate => {
                let tt = self.ctx().tt();
                Ok(self.ctx().data_scalar(
                    Rule::ECmp,
                    Dtype::Bool,
                    member,
                    key,
                    VarRange::default(),
                    tt,
                ))
            }
            other => Err(Reject::UnsupportedOperator {
                name: format!("{:?} ({})", other, member),
            }),
        }
    }

    /// `E-Take` (§5.4). With `axis = None` NumPy flattens first, so the
    /// result is the index array's shape.
    fn take(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ETake, 0, "a")?;
        let indices = self.arg_array(args, Rule::ETake, 1, "indices")?;
        let axis = match args.get(2, "axis") {
            None | Some(Desc::None) => None,
            Some(value) => {
                let value = value.clone();
                Some(
                    self.constant_of(&value)
                        .ok_or_else(|| Reject::Unsupported {
                            construct: "np.take with a non-constant axis".to_string(),
                        })? as i64,
                )
            }
        };
        let (source, axis) = match axis {
            Some(axis) => (a.clone(), normalise_axis(Rule::ETake, axis, a.rank())?),
            None => (self.flattened(&a, key)?, 0),
        };
        e_take(&mut self.ctx(), &source, &indices, axis, key, "in_take")
    }

    /// `np.take_along_axis(a, indices, axis)`: the result has the index
    /// array's shape (`coverage.md` §5.5).
    fn take_along_axis(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ETake, 0, "arr")?;
        let indices = self.arg_array(args, Rule::ETake, 1, "indices")?;
        // As in `E-Take` (§5.4): Φ does not model the index elements, so
        // `inrange` is an unconstrained Boolean and CANFAIL answers "may
        // fail".
        let beta = self.ctx().definedness("in_take_along", key);
        let dims = indices.dims.clone();
        Ok(self.ctx().array(Rule::ETake, a.dtype, dims, beta))
    }

    /// `np.where`: the one-argument form is the mask itself, as the baseline
    /// returns it (`coverage.md` §5.5); the three-argument form is `E-Where3`
    /// (§5.5).
    fn where_(&mut self, args: &Args) -> Result<Desc, Reject> {
        let condition = args.require(Rule::EWhere3, 0, "condition")?.clone();
        let Some(a) = args.get(1, "x").cloned() else {
            // One argument: the mask itself, so that `a[np.where(m)]` is the
            // boolean-mask read of `E-Filter` (§7). The descriptor is
            // unchanged, which is the identity `E-Copy` names (§8).
            self.ctx().applied(Rule::ECopy);
            return Ok(condition);
        };
        let b = args.require(Rule::EWhere3, 2, "y")?.clone();
        let out = self.zip_dtype(&a, &b)?;
        e_where3(&mut self.ctx(), &condition, &a, &b, out)
    }

    /// `E-Nonzero` (§5.5). For a mask of rank `r > 1`, NumPy returns `r`
    /// index arrays; they share the one count, because `Count` is keyed by
    /// the mask (§5.5).
    fn nonzero(&mut self, args: &Args, key: &ValueKey, flat: bool) -> Result<Desc, Reject> {
        let m = self.arg_array(args, Rule::ENonzero, 0, "a")?;
        let mask = self.mask_key(args, 0, "a", key);
        let single = e_nonzero(&mut self.ctx(), &m, &mask, "c_nonzero")?;
        if flat || m.rank() <= 1 {
            return Ok(if flat {
                single
            } else {
                e_composite(CompositeKind::Tuple, vec![single])
            });
        }
        let elems = vec![single; m.rank()];
        Ok(e_composite(CompositeKind::Tuple, elems))
    }

    /// `np.argwhere(a)`: one row per selected element, one column per axis.
    fn argwhere(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let m = self.arg_array(args, Rule::ENonzero, 0, "a")?;
        let rank = m.rank();
        let mask = self.mask_key(args, 0, "a", key);
        let single = e_nonzero(&mut self.ctx(), &m, &mask, "c_argwhere")?;
        let count = single
            .as_array()
            .and_then(|a| a.dims.first().copied())
            .ok_or_else(|| Reject::Unsupported {
                construct: "np.argwhere on a value with no count".to_string(),
            })?;
        let columns = self.an.terms.constant(rank as i128);
        let tt = self.ctx().tt();
        Ok(self
            .ctx()
            .array(Rule::ENonzero, Dtype::Idx, vec![count, columns], tt))
    }

    /// `np.in1d(ar1, ar2)`: a Boolean mask with the shape of the first
    /// operand. The baseline declares the result Integer; phase 0 follows
    /// NumPy so that the mask is recognisable as one (`coverage.md` §9).
    fn in1d(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EMap, 0, "ar1")?;
        let dims = a.dims.clone();
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EMap, Dtype::Bool, dims, tt))
    }

    /// `np.bincount(x, minlength=…)`: a value-sized length (§5.9), declared
    /// by `minlength` when the source gives one.
    fn bincount(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let x = self.arg_array(args, Rule::EValueSized, 0, "x")?;
        let declared = self.declared_maximum(args);
        let extent = *x.dims.first().ok_or_else(|| Reject::Premise {
            rule: Rule::EValueSized,
            detail: "as np.bincount of a rank-0 array".to_string(),
        })?;
        let sized = e_value_sized(
            &mut self.ctx(),
            extent,
            declared,
            key,
            "l_bincount",
            "bincount",
        )?;
        let dtype = if args.kwarg("weights").is_some() {
            Dtype::Float
        } else {
            Dtype::I64
        };
        Ok(self
            .ctx()
            .array(Rule::EValueSized, dtype, vec![sized.length], sized.defcond))
    }

    /// `np.unravel_index(i, shape)`: one coordinate per axis, as arithmetic
    /// on the flat index (§5.4).
    ///
    /// Coordinate `k` is `(i div stride_k) mod dims_k`, with `stride_k` the
    /// product of the extents NumPy varies faster than axis `k` — the axes
    /// after it in C order, before it in F order. Every extent is a
    /// compile-time constant here, so both strides and both divisors are
    /// constants and the coordinate is a term of §3.1; a shape whose extents
    /// are not constants is rejected, since §3.1 has no division by a
    /// runtime value. The `div` is dropped where the stride is 1 and the
    /// `mod` where `stride_k * dims_k` is the whole size, because `i < size`
    /// already bounds the quotient there.
    ///
    /// The coordinates are elements of the tuple and take no key of their
    /// own: each is a term over the flat index's variable, so phase 1 names
    /// it by that term (§3.3, "Elements of a composite").
    fn unravel_index(&mut self, args: &Args, _key: &ValueKey) -> Result<Desc, Reject> {
        let shape = args.require(Rule::EArithData, 1, "shape")?.clone();
        let extents = self.constant_extents(&shape, "np.unravel_index")?;
        let flat = args.require(Rule::EArithData, 0, "indices")?.clone();
        let flat = flat
            .as_scalar()
            .and_then(|s| s.term.as_int())
            .ok_or_else(|| Reject::Unsupported {
                construct: "np.unravel_index of an index that is not an integer scalar"
                    .to_string(),
            })?;
        let size: i128 = extents.iter().product();
        // NumPy raises for a flat index outside `[0, size)`, and outside it
        // the arithmetic below is not the coordinate either, so the identity
        // this rule records holds exactly where the operation is defined.
        let total = self.an.terms.constant(size);
        let defcond = self.within(flat, total);
        // The rule tag and the effect line are the operation's, not each
        // coordinate's, so they are recorded once here (§4's rule format).
        self.ctx().applied(Rule::EArithData);
        self.ctx().check(Rule::EArithData, defcond);
        let fortran = matches!(args.string(2, "order"), Some("F") | Some("f"));
        let mut coordinates = Vec::with_capacity(extents.len());
        for (axis, &extent) in extents.iter().enumerate() {
            let stride: i128 = if fortran {
                extents[..axis].iter().product()
            } else {
                extents[axis + 1..].iter().product()
            };
            let mut t = flat;
            if stride != 1 {
                t = self
                    .an
                    .terms
                    .div_const(t, stride)
                    .ok_or_else(|| Reject::Unsupported {
                        construct: "np.unravel_index with a zero stride".to_string(),
                    })?;
            }
            if stride * extent != size {
                t = self
                    .an
                    .terms
                    .mod_const(t, extent)
                    .ok_or_else(|| Reject::Unsupported {
                        construct: "np.unravel_index with a zero extent".to_string(),
                    })?;
            }
            coordinates.push(self.ctx().element_scalar(
                Dtype::Idx,
                ScalarTerm::Int(t),
                Origin::Data,
                defcond,
            ));
        }
        Ok(e_composite(CompositeKind::Tuple, coordinates))
    }

    /// `np.ravel_multi_index(multi_index, dims)`: the flat index
    /// `Σ_k index_k · stride_k` (§5.4).
    ///
    /// The inverse of [`Walker::unravel_index`] and linear in the same
    /// place: the strides are products of the extents, so the sum is a term
    /// of §3.1 once `dims` is a tuple of constants. NumPy raises when a
    /// coordinate lies outside its own extent (`ValueError: invalid entry in
    /// coordinates array`), which is then `0 ≤ index_k < dims_k` per axis and
    /// a term too, since the coordinates carry terms.
    ///
    /// Two forms keep the older record. A `multi_index` of rank 2 holds one
    /// coordinate tuple per column and yields one flat index per column, an
    /// array and no scalar arithmetic. A `dims` spelled as a run-time array
    /// gives no extents, and a stride would be a product of run-time terms,
    /// which §3.1 has none; the result is a fresh data variable and the
    /// coordinate condition an unconstrained Boolean, as for `E-Take` (§5.4).
    fn ravel_multi_index(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let shape = args.require(Rule::EArithData, 1, "dims")?.clone();
        let multi = args.require(Rule::EArithData, 0, "multi_index")?.clone();
        // A per-element multi-index yields one flat index per element.
        if let Some(a) = multi.as_array() {
            if a.rank() == 2 {
                let count = a.dims[1];
                let in_dims = self.ctx().definedness("in_ravel_multi_index", key);
                return Ok(self
                    .ctx()
                    .array(Rule::ETake, Dtype::Idx, vec![count], in_dims));
            }
        }
        let Ok(extents) = self.constant_extents(&shape, "np.ravel_multi_index") else {
            return self.ravel_multi_index_opaque(&shape, key);
        };
        let entries = super::entries(&multi);
        if entries.len() != extents.len() {
            return Err(Reject::Premise {
                rule: Rule::EArithData,
                detail: format!(
                    "with {} coordinates for a shape of {} axes",
                    entries.len(),
                    extents.len()
                ),
            });
        }
        let fortran = matches!(args.string(2, "order"), Some("F") | Some("f"));
        let mut flat = self.an.terms.constant(0);
        let mut conditions = Vec::with_capacity(entries.len());
        for (axis, entry) in entries.iter().enumerate() {
            let coordinate = entry
                .as_scalar()
                .and_then(|s| s.term.as_int())
                .ok_or_else(|| Reject::Unsupported {
                    construct: "np.ravel_multi_index of a coordinate that is not an \
                                integer scalar"
                        .to_string(),
                })?;
            let extent = self.an.terms.constant(extents[axis]);
            conditions.push(self.within(coordinate, extent));
            let stride: i128 = if fortran {
                extents[..axis].iter().product()
            } else {
                extents[axis + 1..].iter().product()
            };
            let scaled = self.an.terms.mul_const(stride, coordinate);
            flat = self.an.terms.add(flat, scaled);
        }
        let defcond = self.an.terms.all(&conditions);
        Ok(self.ctx().scalar(
            Rule::EArithData,
            Dtype::Idx,
            ScalarTerm::Int(flat),
            Origin::Data,
            defcond,
        ))
    }

    /// The fallback of [`Walker::ravel_multi_index`] for a `dims` whose
    /// extents are not compile-time constants: a fresh data variable, ranged
    /// by the product when that product is a term, with the coordinate
    /// condition as the unconstrained Boolean of `E-Take` (§5.4).
    fn ravel_multi_index_opaque(
        &mut self,
        shape: &Desc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let extents = self.shape_terms(shape)?;
        let known: Option<Vec<TermId>> = extents.into_iter().collect();
        let size = known.and_then(|e| {
            crate::phase0::rules::product_term(&mut self.an.terms, &e)
        });
        let zero = self.an.terms.constant(0);
        let range = match size {
            Some(size) => VarRange::structural(&mut self.an.terms, zero, size),
            None => VarRange::default(),
        };
        let in_dims = self.ctx().definedness("in_ravel_multi_index", key);
        Ok(self.ctx().data_scalar(
            Rule::EArithData,
            Dtype::Idx,
            "flat_index",
            key,
            range,
            in_dims,
        ))
    }

    /// `a.filter(mask)`: the compaction of `E-Filter-1D` (§5.5).
    fn filter(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EFilter1D, 0, "self")?;
        let mask = self.arg_array(args, Rule::EFilter1D, 1, "mask")?;
        let mask_key = self.mask_key(args, 1, "mask", key);
        e_filter_axis0(&mut self.ctx(), &a, &mask, &mask_key, "c_filter")
    }

    /// The key `Count` is keyed by for a mask passed as an argument: the
    /// value that argument names (§5.5), or `key` — the value being built —
    /// when the argument has no key of its own, which leaves the count
    /// unshared rather than shared with the wrong mask.
    fn mask_key(&self, args: &Args, index: usize, name: &str, key: &ValueKey) -> ValueKey {
        match args.key(index, name) {
            Some(argument) => self.an.defining_key(&argument),
            None => key.clone(),
        }
    }

    /// `E-Matmul` (§5.7), also `np.dot`. Two rank-1 operands contract to a
    /// scalar, which the rule reports as a rank-0 array.
    fn matmul(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EMatmul, 0, "a")?;
        let b = self.arg_array(args, Rule::EMatmul, 1, "b")?;
        let product = e_matmul(&mut self.ctx(), &a, &b)?;
        Ok(self.demote_rank0(product, key, "dot"))
    }

    /// `np.tensordot(a, b, axes)`: the contracted axes disappear from both
    /// operands and the rest concatenate (`coverage.md` §5.6).
    fn tensordot(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EMatmul, 0, "a")?;
        let b = self.arg_array(args, Rule::EMatmul, 1, "b")?;
        let axes = args.require(Rule::EMatmul, 2, "axes")?.clone();
        let (left, right) = self.contracted_axes(&axes, a.rank(), b.rank())?;
        let mut dims: Vec<TermId> = a
            .dims
            .iter()
            .enumerate()
            .filter(|(k, _)| !left.contains(k))
            .map(|(_, &d)| d)
            .collect();
        dims.extend(
            b.dims
                .iter()
                .enumerate()
                .filter(|(k, _)| !right.contains(k))
                .map(|(_, &d)| d),
        );
        let mut conditions = Vec::with_capacity(left.len());
        for (&i, &j) in left.iter().zip(right.iter()) {
            conditions.push(self.an.terms.eq(a.dims[i], b.dims[j]));
        }
        let defcond = self.an.terms.all(&conditions);
        Ok(self.ctx().array(Rule::EMatmul, Dtype::Float, dims, defcond))
    }

    /// The axis pair of a `tensordot`: an integer `n` contracts the last `n`
    /// of the left operand with the first `n` of the right.
    #[allow(clippy::type_complexity)]
    fn contracted_axes(
        &mut self,
        axes: &Desc,
        left_rank: usize,
        right_rank: usize,
    ) -> Result<(Vec<usize>, Vec<usize>), Reject> {
        if let Some(n) = self.constant_of(axes) {
            let n = usize::try_from(n).map_err(|_| Reject::Unsupported {
                construct: "np.tensordot with a negative axis count".to_string(),
            })?;
            let left = (left_rank - n..left_rank).collect();
            let right = (0..n).collect();
            return Ok((left, right));
        }
        let pair = axes.as_composite().ok_or_else(|| Reject::Unsupported {
            construct: "np.tensordot with a non-constant `axes`".to_string(),
        })?;
        let elems = pair.elems.clone();
        if elems.len() != 2 {
            return Err(Reject::Unsupported {
                construct: "np.tensordot with an `axes` that is not a pair".to_string(),
            });
        }
        let left = self.axis_indices(&elems[0], left_rank)?;
        let right = self.axis_indices(&elems[1], right_rank)?;
        Ok((left, right))
    }

    fn axis_indices(&mut self, value: &Desc, rank: usize) -> Result<Vec<usize>, Reject> {
        super::entries(value)
            .iter()
            .map(|e| {
                let axis = self.constant_of(e).ok_or_else(|| Reject::Unsupported {
                    construct: "a non-constant contracted axis".to_string(),
                })?;
                normalise_axis(Rule::EMatmul, axis as i64, rank)
            })
            .collect()
    }

    /// `np.einsum('ii->i', a)`: the diagonal of a square matrix, which is the
    /// only subscript the baseline accepts (`coverage.md` §5.6).
    fn einsum(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EDiag, 1, "operand")?;
        if a.rank() != 2 {
            return Err(Reject::Premise {
                rule: Rule::EDiag,
                detail: format!("as np.einsum on a rank-{} operand", a.rank()),
            });
        }
        let d = self.an.terms.min(a.dims[0], a.dims[1]);
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EDiag, a.dtype, vec![d], tt))
    }

    /// `np.corrcoef(x, y)`: a two-by-two matrix (`coverage.md` §5.6).
    fn corrcoef(&mut self, args: &Args) -> Result<Desc, Reject> {
        let _ = self.arg_array(args, Rule::ECorrcoef, 0, "x")?;
        let two = self.an.terms.constant(2);
        let tt = self.ctx().tt();
        Ok(self
            .ctx()
            .array(Rule::ECorrcoef, Dtype::Float, vec![two, two], tt))
    }

    /// `np.correlate(a, v)` in `'valid'` mode (`coverage.md` §5.6).
    ///
    /// NumPy swaps the operands when `v` is the longer one, so the length is
    /// `max(n, m) - min(n, m) + 1`, not `n - m + 1`. The baseline swaps them
    /// too (`np_correlate_ds1000`, `src/ir_gen/named_attr/numpy_funcs.rs`)
    /// and supports no other mode, so the mode argument — a string, which
    /// §3.2 gives no term — need not be read.
    fn correlate(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::ECorrelate, 0, "a")?;
        let v = self.arg_array(args, Rule::ECorrelate, 1, "v")?;
        let (da, dv) = (a.dims[0], v.dims[0]);
        let longer = self.an.terms.max(da, dv);
        let shorter = self.an.terms.min(da, dv);
        let difference = self.an.terms.sub(longer, shorter);
        let one = self.an.terms.constant(1);
        let length = self.an.terms.add(difference, one);
        let tt = self.ctx().tt();
        Ok(self
            .ctx()
            .array(Rule::ECorrelate, Dtype::Float, vec![length], tt))
    }

    /// `np.<binop>.outer(a, b)`: both operands flattened, result rank 2
    /// (`coverage.md` §5.3).
    fn outer(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EOuter, 0, "a")?;
        let b = self.arg_array(args, Rule::EOuter, 1, "b")?;
        let rows = self.size_of(&a, key, "sigma_outer_a");
        let columns = self.size_of(&b, key, "sigma_outer_b");
        let dtype = Dtype::promote(a.dtype, b.dtype).unwrap_or(Dtype::Float);
        let tt = self.ctx().tt();
        Ok(self
            .ctx()
            .array(Rule::EOuter, dtype, vec![rows, columns], tt))
    }

    /// `np.clip(a, a_min, a_max)`: an `E-Zip` over the three operands.
    fn clip(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = args.require(Rule::EZip, 0, "a")?.clone();
        let lo = args.require(Rule::EZip, 1, "a_min")?.clone();
        let hi = args.require(Rule::EZip, 2, "a_max")?.clone();
        let dtype = self.zip_dtype(&a, &lo)?;
        e_where3(&mut self.ctx(), &a, &lo, &hi, dtype)
    }

    /// `np.isclose(a, b)`: an `E-Zip` with a Boolean result.
    fn isclose(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = args.require(Rule::EZip, 0, "a")?.clone();
        let b = args.require(Rule::EZip, 1, "b")?.clone();
        self.apply_zip(ZipOp::Eq, &a, &b, key)
    }

    // ── Shared helpers ────────────────────────────────────────────────

    /// The promoted dtype of two operands, for a rule that needs one.
    fn zip_dtype(&mut self, a: &Desc, b: &Desc) -> Result<Dtype, Reject> {
        let da = a.dtype().unwrap_or(Dtype::Float);
        let db = b.dtype().unwrap_or(Dtype::Float);
        Dtype::promote(da, db).ok_or_else(|| Reject::Premise {
            rule: Rule::EZip,
            detail: format!("on {} and {}", da.as_str(), db.as_str()),
        })
    }

    /// The rank-1 view of an array, for the operators that flatten first.
    fn flattened(&mut self, a: &ArrayDesc, key: &ValueKey) -> Result<ArrayDesc, Reject> {
        let size = self.size_of(a, key, "sigma_flat");
        Ok(ArrayDesc {
            dtype: a.dtype,
            dims: vec![size],
            defcond: a.defcond,
            psi: a.psi,
        })
    }

    /// `0 ≤ t < d`, the range NumPy's `'raise'` mode demands of a flat index
    /// and of a coordinate. It is not [`in_bounds`], which admits Python's
    /// negative indices: `np.unravel_index(-1, shape)` raises, and the
    /// arithmetic of §5.4 is the coordinate only on the non-negative side.
    fn within(&mut self, t: TermId, d: TermId) -> crate::phase0::term::BoolId {
        let zero = self.an.terms.constant(0);
        let low = self.an.terms.le(zero, t);
        let high = self.an.terms.lt(t, d);
        self.an.terms.and(low, high)
    }

    /// The extents of a shape argument as compile-time constants, one per
    /// axis, for the two operators that turn a shape into arithmetic (§5.4).
    ///
    /// A shape spelled as a run-time array, or one whose extents are terms
    /// rather than numerals, gives no constants, and §3.1 has no division by
    /// a runtime value; the rejection names the operator and the axis.
    fn constant_extents(&mut self, shape: &Desc, member: &str) -> Result<Vec<i128>, Reject> {
        let extents = self.shape_terms(shape)?;
        let mut out = Vec::with_capacity(extents.len());
        for (axis, extent) in extents.iter().enumerate() {
            let value = extent
                .and_then(|t| self.an.terms.constant_value(t))
                .ok_or_else(|| Reject::Unsupported {
                    construct: format!(
                        "{} with a shape whose axis {} is not a compile-time constant",
                        member, axis
                    ),
                })?;
            if value <= 0 {
                return Err(Reject::Unsupported {
                    construct: format!(
                        "{} with a shape whose axis {} has the extent {}",
                        member, axis, value
                    ),
                });
            }
            out.push(value);
        }
        if out.is_empty() {
            return Err(Reject::Unsupported {
                construct: format!("{} with an empty shape", member),
            });
        }
        Ok(out)
    }

    /// The extent terms of a shape argument, one entry per axis.
    ///
    /// A shape spelled as an array — `np.ravel_multi_index(idx, dims)` with
    /// `dims` computed at run time — gives the number of axes but no extents,
    /// so those entries are `None`.
    fn shape_terms(&mut self, shape: &Desc) -> Result<Vec<Option<TermId>>, Reject> {
        if let Some(a) = shape.as_array() {
            let rank = a
                .dims
                .first()
                .and_then(|&d| self.an.terms.constant_value(d))
                .ok_or_else(|| Reject::Unsupported {
                    construct: "a shape array of runtime length".to_string(),
                })?;
            return Ok(vec![None; rank.max(0) as usize]);
        }
        Ok(super::entries(shape)
            .iter()
            .map(|e| e.as_scalar().and_then(|s| s.term.as_int()))
            .collect())
    }

    /// A rank-0 array is a scalar; the descriptor kinds differ, so the
    /// conversion is explicit (`np.dot` of two vectors).
    fn demote_rank0(&mut self, d: Desc, key: &ValueKey, base: &str) -> Desc {
        match &d {
            Desc::Array(a) if a.rank() == 0 => {
                let dtype = a.dtype;
                let range = VarRange::of_dtype(&mut self.an.terms, dtype);
                let tt = self.ctx().tt();
                self.ctx()
                    .data_scalar(Rule::EMatmul, dtype, base, key, range, tt)
            }
            _ => d,
        }
    }
}

//! The gather and linear-algebra classes: `definition.md` §5.4's `E-Take`,
//! §5.5's `E-Nonzero` and `E-Filter`, and §5.7's `E-Matmul`, together with
//! the statistics of `coverage.md` §5.6.

use crate::phase0::desc::{ArrayDesc, CompositeKind, Desc, Dtype, ValueKey};
use crate::phase0::ops::{Class, ZipOp};
use crate::phase0::rules::{
    e_composite, e_filter_axis0, e_matmul, e_nonzero, e_take, e_value_sized, e_where3,
    helpers::normalise_axis, product_term, VarRange,
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
        let single = e_nonzero(&mut self.ctx(), &m, key, "c_nonzero")?;
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
        let single = e_nonzero(&mut self.ctx(), &m, key, "c_argwhere")?;
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

    /// `np.unravel_index(i, shape)`: one index scalar per axis
    /// (`coverage.md` §5.5). An extent phase 0 knows gives the coordinate its
    /// structural range `[0, d)`; one read out of an array does not.
    fn unravel_index(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let shape = args.require(Rule::ETake, 1, "shape")?.clone();
        let extents = self.shape_terms(&shape)?;
        let mut coordinates = Vec::with_capacity(extents.len());
        for (axis, extent) in extents.iter().enumerate() {
            let range = match extent {
                Some(extent) => {
                    let zero = self.an.terms.constant(0);
                    crate::phase0::rules::VarRange::structural(&mut self.an.terms, zero, *extent)
                }
                None => VarRange::of_dtype(&mut self.an.terms, Dtype::Idx),
            };
            let tt = self.ctx().tt();
            let name = format!("coord{}", axis);
            coordinates.push(self.ctx().data_scalar(
                Rule::ETake,
                Dtype::Idx,
                &name,
                key,
                range,
                tt,
            ));
        }
        Ok(e_composite(CompositeKind::Tuple, coordinates))
    }

    /// `np.ravel_multi_index(multi_index, dims)`: a flat index whose range is
    /// the product of the extents.
    ///
    /// NumPy raises when a coordinate lies outside its own extent
    /// (`ValueError: invalid entry in coordinates array`). That is a
    /// condition on element values, which $\Phi$ does not model, so it is an
    /// unconstrained Boolean of the `E-Take` kind (§5.4) rather than a term.
    fn ravel_multi_index(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let shape = args.require(Rule::ETake, 1, "dims")?.clone();
        let extents = self.shape_terms(&shape)?;
        let known: Option<Vec<TermId>> = extents.iter().copied().collect();
        let size = known.and_then(|e| product_term(&mut self.an.terms, &e));
        let multi = args.require(Rule::ETake, 0, "multi_index")?.clone();
        let zero = self.an.terms.constant(0);
        let range = match size {
            Some(size) => {
                crate::phase0::rules::VarRange::structural(&mut self.an.terms, zero, size)
            }
            None => VarRange::default(),
        };
        let in_dims = self.ctx().definedness("in_ravel_multi_index", key);
        // A per-element multi-index yields one flat index per element.
        let count = match multi.as_array() {
            Some(a) if a.rank() == 2 => Some(a.dims[1]),
            _ => None,
        };
        match count {
            Some(count) => Ok(self
                .ctx()
                .array(Rule::ETake, Dtype::Idx, vec![count], in_dims)),
            None => Ok(self.ctx().data_scalar(
                Rule::ETake,
                Dtype::Idx,
                "flat_index",
                key,
                range,
                in_dims,
            )),
        }
    }

    /// `a.filter(mask)`: the compaction of `E-Filter-1D` (§5.5).
    fn filter(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EFilter1D, 0, "self")?;
        let mask = self.arg_array(args, Rule::EFilter1D, 1, "mask")?;
        e_filter_axis0(&mut self.ctx(), &a, &mask, key, "c_filter")
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

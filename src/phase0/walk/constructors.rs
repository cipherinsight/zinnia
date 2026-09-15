//! The constructors of `definition.md` §5.8 and the value-sized lengths of
//! §5.9 (`coverage.md` §5.4).
//!
//! `zeros`, `ones`, `empty`, `full`, the `*_like` family, `identity` / `eye`,
//! `arange`, `linspace`, `array` / `asarray` and `indices`. Every one of them
//! reads its shape argument through [`Walker::shape_scalars`] and its element
//! dtype through [`dtype_kwarg`]; an axis whose extent is a data scalar is
//! value-sized, and [`crate::phase0::rules::e_value_sized`] is where the
//! declared maximum of §5.9 comes in. The transforms of §5.6 and the
//! reductions of §5.7 are in [`super::arrays`].

use crate::phase0::desc::{ArrayDesc, Desc, Dtype, ScalarDesc, ValueKey};
use crate::phase0::rules::e_zeros;
use crate::phase0::term::TermId;
use crate::phase0::Reject;

use super::calls::Args;
use super::values::dtype_kwarg;
use super::Walker;
use crate::phase0::rules::Rule;

impl Walker<'_> {
    /// `E-Zeros` (§5.8) and `np.full`, whose fill value only sets the dtype.
    pub(super) fn zeros(
        &mut self,
        args: &Args,
        key: &ValueKey,
        full: bool,
    ) -> Result<Desc, Reject> {
        let shape = args.require(Rule::EZeros, 0, "shape")?.clone();
        let scalars = self.shape_scalars(&shape)?;
        let fallback = if full {
            args.get(1, "fill_value")
                .and_then(|d| d.dtype())
                .unwrap_or(Dtype::Float)
        } else {
            Dtype::Float
        };
        let dtype = dtype_kwarg(args, fallback)?;
        let declared = self.declared_maximum(args);
        let refs: Vec<&ScalarDesc> = scalars.iter().collect();
        e_zeros(&mut self.ctx(), &refs, dtype, declared, key, "l_zeros")
    }

    /// The `*_like` constructors: the prototype's shape, and its dtype unless
    /// the call overrides it.
    pub(super) fn like(&mut self, args: &Args) -> Result<Desc, Reject> {
        let a = self.arg_array(args, Rule::EZeros, 0, "a")?;
        let dtype = dtype_kwarg(args, a.dtype)?;
        let dims = a.dims.clone();
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EZeros, dtype, dims, tt))
    }

    /// `np.identity(n)` and `np.eye(n)`: a square matrix. The baseline drops
    /// `eye`'s `M` and `k`; phase 0 reads `M` when it is given, which only
    /// makes the record more precise.
    pub(super) fn identity(&mut self, args: &Args) -> Result<Desc, Reject> {
        let n = args.require(Rule::EZeros, 0, "n")?.clone();
        let rows = self.extent_of(&n)?;
        let columns = match args.get(1, "M") {
            None | Some(Desc::None) => rows,
            Some(m) => {
                let m = m.clone();
                self.extent_of(&m)?
            }
        };
        let dtype = dtype_kwarg(args, Dtype::Float)?;
        let tt = self.ctx().tt();
        Ok(self
            .ctx()
            .array(Rule::EZeros, dtype, vec![rows, columns], tt))
    }

    /// One extent of a constructor's shape, clamped at zero as `E-Arange`
    /// clamps it (§5.8).
    pub(super) fn extent_of(&mut self, value: &Desc) -> Result<TermId, Reject> {
        let scalar = value.as_scalar().ok_or_else(|| Reject::Unsupported {
            construct: format!("an extent given as a {}", value.kind_str()),
        })?;
        let term = scalar.term.as_int().ok_or_else(|| Reject::Unsupported {
            construct: "an extent with no term".to_string(),
        })?;
        let zero = self.an.terms.constant(0);
        Ok(self.an.terms.max(zero, term))
    }

    /// `E-Arange` (§5.8), in its one-, two- and three-argument forms.
    pub(super) fn arange(&mut self, args: &Args, key: &ValueKey) -> Result<Desc, Reject> {
        let declared = self.declared_maximum(args);
        let scalars: Vec<ScalarDesc> = args
            .pos
            .iter()
            .filter_map(|d| d.as_scalar().cloned())
            .collect();
        let built = match scalars.len() {
            1 => crate::phase0::rules::e_arange(
                &mut self.ctx(),
                &scalars[0],
                declared,
                key,
                "l_arange",
            ),
            2 | 3 => {
                let step = match scalars.get(2) {
                    None => 1,
                    Some(s) => s
                        .term
                        .as_int()
                        .and_then(|t| self.an.terms.constant_value(t))
                        .ok_or_else(|| Reject::NonConstantStep {
                            detail: "arange with a non-constant step".to_string(),
                        })? as i64,
                };
                crate::phase0::rules::e_arange_range(
                    &mut self.ctx(),
                    &scalars[0],
                    &scalars[1],
                    step,
                )
            }
            other => Err(Reject::Unsupported {
                construct: format!("np.arange with {} arguments", other),
            }),
        }?;
        // `dtype=` only moves the element dtype; the length rule above is the
        // same either way.
        let dtype = dtype_kwarg(args, Dtype::Idx)?;
        match built.as_array() {
            Some(a) if a.dtype != dtype => Ok(Desc::Array(ArrayDesc { dtype, ..a.clone() })),
            _ => Ok(built),
        }
    }

    /// `np.linspace(start, stop, num)`: `num` entries, a real dtype.
    pub(super) fn linspace(&mut self, args: &Args) -> Result<Desc, Reject> {
        let count = match args.get(2, "num") {
            None | Some(Desc::None) => 50,
            Some(value) => {
                let value = value.clone();
                self.constant_of(&value)
                    .ok_or_else(|| Reject::Unsupported {
                        construct: "np.linspace with a non-constant `num`".to_string(),
                    })?
            }
        };
        let dtype = dtype_kwarg(args, Dtype::Float)?;
        let length = self.an.terms.constant(count.max(0));
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EZeros, dtype, vec![length], tt))
    }

    /// `np.array` / `np.asarray`: the composite becomes an array (§5.8).
    pub(super) fn asarray(&mut self, args: &Args) -> Result<Desc, Reject> {
        let value = args.require(Rule::EStack, 0, "a")?.clone();
        let a = self.as_array(&value, Rule::EStack)?;
        let dtype = dtype_kwarg(args, a.dtype)?;
        if dtype == a.dtype {
            // The identity on descriptors, which `E-Copy` names (§8).
            self.ctx().applied(Rule::ECopy);
            return Ok(Desc::Array(a));
        }
        let dims = a.dims.clone();
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EMap, dtype, dims, tt))
    }

    /// `np.indices(dimensions)`: shape `(len(dimensions), *dimensions)`
    /// (`coverage.md` §5.4).
    pub(super) fn indices(&mut self, args: &Args) -> Result<Desc, Reject> {
        let shape = args.require(Rule::EZeros, 0, "dimensions")?.clone();
        let scalars = self.shape_scalars(&shape)?;
        let mut dims = Vec::with_capacity(scalars.len() + 1);
        dims.push(self.an.terms.constant(scalars.len() as i128));
        for extent in &scalars {
            let value = Desc::Scalar(extent.clone());
            dims.push(self.extent_of(&value)?);
        }
        let tt = self.ctx().tt();
        Ok(self.ctx().array(Rule::EZeros, Dtype::Idx, dims, tt))
    }

    /// The declared maximum of a value-sized construct (§5.9), read from the
    /// `size` or `minlength` keyword the source spells it with.
    pub(super) fn declared_maximum(&mut self, args: &Args) -> Option<i64> {
        for name in ["size", "minlength", "max_length"] {
            if let Some(value) = args.kwarg(name) {
                if let Some(constant) = self.constant_of(value) {
                    return i64::try_from(constant).ok();
                }
            }
        }
        None
    }
}

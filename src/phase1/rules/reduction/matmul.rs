//! `MATMUL` (`revision/phase1/rules/reduction/matmul.md`).
//!
//! Computes a contraction. For `c = a @ b` with a of shape `α ++ [M, K]` and
//! b of shape `β ++ [K′, N]`, every element `c[𝐛, i, j]` is
//! `Σ_{k<K} a[…, i, k]·b[…, k, j]`. The rule fetches both operands at every
//! coordinate a product needs, enumerates the K products of every output
//! position as one fiber of its own, and folds them with the sum of
//! `FOLD_GROUPS`.
//!
//! Step 2 writes the form the corpus uses: the two operands of rank 1 or 2,
//! with the batch axes of the general contraction left to the variable-length
//! step (`README.md` Open, "BATCH undecided on a runtime batch axis").

use crate::phase0::desc::ValueKey;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Capacity, Coordinate, Family, Interface, Layout, LayoutKind, Live, RangeAxis,
    StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Matmul;

impl Rule for Matmul {
    fn id(&self) -> RuleId {
        RuleId::Matmul
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire, Family::Store, Family::Masked, Family::Packed]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(key) = request.target.as_value() else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some((left, right, kind)) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };
        let (Some(left_array), Some(right_array), Some(result)) = (
            builder.array(&left).cloned(),
            builder.array(&right).cloned(),
            builder.array(&key).cloned(),
        ) else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "MATMUL", result.dtype) {
            return Ok(Vec::new());
        }
        // The families this node's bundles answer: one wire for a rank-0
        // result, the box `Store` otherwise, and the `Masked` and `Packed`
        // of a rank-1 one over the same wires (§3.4).
        let rank = result.rank();
        let answered = match rank {
            0 => request.family == Family::Wire,
            1 => matches!(
                request.family,
                Family::Store | Family::Masked | Family::Packed
            ),
            _ => request.family == Family::Store,
        };
        if !answered {
            return Ok(Vec::new());
        }
        if kind == Kind::Correlate {
            return self.correlate(builder, request, &key, &left, &right, &result);
        }
        // The contraction pairs a's last axis with b's first. A right operand
        // of rank 1 is promoted to [K, 1] and a left one to [1, K]; a left
        // operand of rank ≥ 2 carries its leading axes into the result as
        // free row axes, which is what `a @ b` on a stack of matrices and
        // `tensordot` with one contracted pair both mean.
        if right_array.rank() > 2 {
            return Ok(Vec::new());
        }
        if !self.contracts_last_with_first(builder, &key, left_array.rank()) {
            return Ok(Vec::new());
        }
        let contracted: TermId = *left_array.dims.last().unwrap();
        let row_axes: Vec<TermId> = left_array.dims[..left_array.rank() - 1].to_vec();
        let columns: Option<TermId> = (right_array.rank() == 2).then(|| right_array.dims[1]);
        if rank != row_axes.len() + usize::from(columns.is_some()) {
            return Ok(Vec::new());
        }

        let mut probe = Candidate::new(RuleId::Matmul, None, request.epoch);
        let under = builder.under(request.condition);
        // The capacities: K, M and N by CONST, else MAX, with the
        // later-epoch variants of §6.2.
        let extents: Vec<TermId> = std::iter::once(contracted)
            .chain(row_axes.iter().copied())
            .chain(columns)
            .collect();
        let mut out = Vec::new();
        for capacities in builder.capacity_box(&mut probe, &extents, request.epoch, under) {
            let contracted_capacity = capacities.axes[0];
            let row_capacities: Vec<Capacity> =
                capacities.axes[1..1 + row_axes.len()].to_vec();
            let column_capacity = columns.map(|_| capacities.axes[1 + row_axes.len()]);
            let rows: Capacity = row_capacities.iter().product();

            let mut candidate = Candidate::new(RuleId::Matmul, None, capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = vec![key.clone(), left.clone(), right.clone()];
            // `left` and `right`: the range batches over each operand's own
            // coordinates, every product's position included.
            let left_batch = grid(
                &row_capacities
                    .iter()
                    .map(|&width| Some(width))
                    .chain(std::iter::once(Some(contracted_capacity)))
                    .collect::<Vec<_>>(),
            );
            let right_batch = grid(&[Some(contracted_capacity), column_capacity]);
            let left_row = builder.fetch_port(&mut candidate, request, &left, &left_batch)?;
            let right_row = builder.fetch_port(&mut candidate, request, &right, &right_batch)?;
            let (Some(left_port), Some(right_port)) =
                (left_row.entries.first().cloned(), right_row.entries.first().cloned())
            else {
                continue;
            };
            for port in [&left_port, &right_port] {
                candidate.inputs.push(port.requirement);
                candidate.epoch = candidate.epoch.max(port.epoch);
            }
            // `k`: the contracted extent's wire, for the fiber mask.
            let capacity = builder.literal(contracted_capacity);
            let masked = !builder.prove_eq(
                &mut candidate,
                contracted,
                capacity,
                capacities.epoch,
                under,
            );
            if masked {
                let row = builder.child(
                    request,
                    Target::Term(contracted),
                    Family::Wire,
                    request.condition,
                )?;
                if let Some(first) = row.entries.first() {
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                }
            }

            // emitter: the fiber mask, then the contraction itself — one
            // `mul` per product and one wrap per output position for an
            // integer result, and one `fop⟨product_sum_f⟩` per output
            // position for a Float one, which is the backend's own
            // product-sum aggregate over the K pairs (`src/ir_defs.rs`,
            // `ProductSumF`) and needs no wrap.
            let positions = rows * column_capacity.unwrap_or(1);
            if masked {
                candidate = candidate
                    .emit(Primitive::Therm { capacity: contracted_capacity }, 1)
                    .emit(Primitive::Mul, rows * contracted_capacity);
            }
            if super::super::sources::is_float(result.dtype) {
                candidate =
                    candidate.emit(Primitive::Float(FloatOp::ProductSum), positions);
            } else {
                let width = builder.omega.round(builder.dtype_width(result.dtype));
                candidate = candidate.emit(Primitive::Mul, positions * contracted_capacity);
                if width > 0 {
                    candidate = candidate.emit(
                        Primitive::Wrap {
                            width,
                            carry_width: builder
                                .omega
                                .round(width + bits_of_capacity(contracted_capacity) + 1),
                        },
                        positions,
                    );
                }
            }
            let grid: Vec<Capacity> =
                row_capacities.iter().copied().chain(column_capacity).collect();
            candidate =
                self.outputs(candidate, request, &key, result.dtype, rank, positions, &grid);
            candidate = candidate
                .parameter("K", contracted_capacity)
                .parameter("Q", positions);
            out.push(candidate);
        }
        Ok(out)
    }
}

impl Matmul {
    /// `source_pattern`: the producer names E-Matmul or E-Correlate, and the
    /// operands are its two arrays.
    fn matches(
        &self,
        builder: &Builder<'_>,
        key: &ValueKey,
    ) -> Option<(ValueKey, ValueKey, Kind)> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        let kind = if rules.iter().any(|rule| matches!(rule, Phase0Rule::EMatmul)) {
            Kind::Matmul
        } else if rules.iter().any(|rule| matches!(rule, Phase0Rule::ECorrelate)) {
            Kind::Correlate
        } else {
            return None;
        };
        let arrays: Vec<ValueKey> = operands
            .iter()
            .filter(|operand| {
                builder.array(operand).map(|array| array.rank() > 0).unwrap_or(false)
            })
            .cloned()
            .collect();
        match arrays.as_slice() {
            [left, right] => Some((left.clone(), right.clone(), kind)),
            _ => None,
        }
    }

    /// Whether the contraction pairs the left operand's last axis with the
    /// right operand's first (`matmul.md`, `source_pattern`).
    ///
    /// `@`, `matmul` and `dot` pair them by definition. `tensordot` names the
    /// pair in its `axes` argument: an integer `n` contracts the last `n`
    /// axes of a with the first `n` of b, which is this pairing only for
    /// `n = 1`; a pair of axis lists names them directly.
    fn contracts_last_with_first(
        &self,
        builder: &Builder<'_>,
        key: &ValueKey,
        left_rank: usize,
    ) -> bool {
        let Some(node) = builder.node(key) else {
            return true;
        };
        if node.member() != Some("tensordot") {
            return true;
        }
        let Some(axes) = node.parameter(2, "axes").and_then(|value| value.ints()) else {
            // `tensordot` with no `axes` contracts the last two axes of a
            // with the first two of b, which this form does not write.
            return false;
        };
        match axes.as_slice() {
            [1] => true,
            [left, right] => {
                let left = if *left < 0 { left + left_rank as i64 } else { *left };
                left == left_rank as i64 - 1 && *right == 0
            }
            _ => false,
        }
    }

    /// The `correlate` form of `matmul.md`: `c[j] = Σ_{k<W} long[j + k] ·
    /// short[k]` in NumPy's `'valid'` mode, with `W = min(n, m)` and
    /// `P = |n − m| + 1` the two constant lengths decide. Both operands are
    /// fetched whole at constant coordinates, so the rule asks no MAX and
    /// emits no fiber mask.
    #[allow(clippy::too_many_arguments)]
    fn correlate(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        left: &ValueKey,
        right: &ValueKey,
        result: &crate::phase0::desc::ArrayDesc,
    ) -> Result<Vec<Candidate>, Reject> {
        if result.rank() != 1 {
            return Ok(Vec::new());
        }
        let mut candidate = Candidate::new(RuleId::Matmul, Some("correlate"), request.epoch);
        candidate.matched = vec![key.clone(), left.clone(), right.clone()];
        let under = builder.under(request.condition);
        // CONST(n), CONST(m): which operand is the longer is a compile-time
        // fact here and a run-time one otherwise, which the rule file leaves
        // open.
        let mut lengths = Vec::with_capacity(2);
        for operand in [left, right] {
            let Some(dims) = builder.array(operand).map(|array| array.dims.clone()) else {
                return Ok(Vec::new());
            };
            let [extent] = dims[..] else {
                return Ok(Vec::new());
            };
            match builder.ask_for(
                &mut candidate,
                Question::Const(extent),
                request.epoch,
                under,
            ) {
                Verdict::Constant(value) if value >= 0 => lengths.push(value),
                _ => return Ok(Vec::new()),
            }
        }
        let (n, m) = (lengths[0], lengths[1]);
        let window = n.min(m);
        let positions = (n - m).abs() + 1;
        if window == 0 {
            return Ok(Vec::new());
        }
        for (operand, length) in [(left, n), (right, m)] {
            let batch = grid(&[Some(length)]);
            let row = builder.fetch_port(&mut candidate, request, operand, &batch)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        // emitter: the contraction, as in the matrix form — one
        // `fop⟨product_sum_f⟩` per output position for a Float result and one
        // `mul` per product with one wrap per position for an integer one.
        if super::super::sources::is_float(result.dtype) {
            candidate = candidate.emit(Primitive::Float(FloatOp::ProductSum), positions);
        } else {
            let width = builder.omega.round(builder.dtype_width(result.dtype));
            candidate = candidate.emit(Primitive::Mul, positions * window);
            if width > 0 {
                candidate = candidate.emit(
                    Primitive::Wrap {
                        width,
                        carry_width: builder.omega.round(width + bits_of_capacity(window) + 1),
                    },
                    positions,
                );
            }
        }
        candidate = self.outputs(
            candidate,
            request,
            key,
            result.dtype,
            1,
            positions,
            &[positions],
        );
        candidate = candidate.parameter("W", window).parameter("P", positions);
        Ok(vec![candidate])
    }

    /// The bundles of the contraction's output, as `FOLD_GROUPS` writes them.
    #[allow(clippy::too_many_arguments)]
    fn outputs(
        &self,
        candidate: Candidate,
        request: &RowRequest,
        key: &ValueKey,
        dtype: crate::phase0::desc::Dtype,
        rank: usize,
        positions: Capacity,
        grid: &[Capacity],
    ) -> Candidate {
        if rank == 0 {
            return candidate.output(
                request.target.clone(),
                Interface::Wire(dtype),
                request.condition,
            );
        }
        let grid: Vec<Capacity> = grid.to_vec();
        let mut out = candidate.output(
            request.target.clone(),
            Interface::Store(StoreIface {
                owner: key.clone(),
                dtype,
                capacity: positions,
                lambda: Layout::dense(LayoutKind::Box(grid), key.clone()),
            }),
            request.condition,
        );
        if rank == 1 {
            out = out
                .output(
                    request.target.clone(),
                    Interface::Masked {
                        dtype,
                        capacity: positions,
                        lambda: Layout::dense(
                            LayoutKind::Ranked { capacity: positions, ranks: None },
                            key.clone(),
                        ),
                    },
                    request.condition,
                )
                .output(
                    request.target.clone(),
                    Interface::Packed {
                        dtype,
                        capacity: positions,
                        lambda: Layout::dense(LayoutKind::Compact(positions), key.clone()),
                        canon: false,
                    },
                    request.condition,
                );
        }
        out
    }
}

/// Which contraction the producer names (`matmul.md`, `source_pattern`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// E-Matmul: `@`, `matmul`, `dot`, `tensordot` with one contracted pair.
    Matmul,
    /// E-Correlate with constant operand lengths.
    Correlate,
}

/// The range batch over a capacity box, skipping the axes a promoted rank-1
/// operand does not have.
fn grid(capacities: &[Option<Capacity>]) -> Batch {
    Batch::Range {
        axes: capacities
            .iter()
            .flatten()
            .map(|&width| RangeAxis {
                base: Coordinate::Literal(0),
                stride: 1,
                width,
                broadcast: None,
            })
            .collect(),
        live: Live::InRange,
    }
}

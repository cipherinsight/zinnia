//! Envelope type for dynamic ndarrays — Tier 2 in `ROADMAP/03-type-system.md`.
//!
//! An [`Envelope`] is the compile-time bound on a (possibly dynamic-shape)
//! ndarray's shape. It is purely shape-level — no payload, no dtype, no IR
//! state. Concretely, an envelope is a fixed-rank vector of [`Dim`]s, each
//! of which carries:
//!
//! - a [`DimVar`] — a symbolic identity, drawn from a global per-compilation
//!   [`DimTable`]. Two dims with the same `DimVar` (or that have been unified
//!   in the table) are *known* to be equal at runtime.
//! - a numeric range `min..=max` — the smallest and largest values the dim
//!   can take at runtime. `min == max` means the dim is statically known.
//!
//! This module is intentionally standalone:
//! - No `Value` dependency.
//! - No `IRBuilder` dependency.
//! - No interaction with the existing `(max_length, max_rank)` envelope on
//!   `DynamicNDArrayData` (that migration happens in Phase 1).
//!
//! The Tier 2 broadcast rule lives in [`broadcast_envelopes`]. It's the
//! envelope-level analogue of `helpers::shape_arith::broadcast_shapes` but
//! enriched with `DimVar` unification.

// ────────────────────────────────────────────────────────────────────────
// DimVar — symbolic identity for a dimension
// ────────────────────────────────────────────────────────────────────────

/// A symbolic identifier for a dimension. Allocated by [`DimTable::fresh`].
/// Two dims sharing the same root in the union-find table represent the
/// same runtime length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DimVar(pub u32);

// ────────────────────────────────────────────────────────────────────────
// DimTable — global union-find over DimVars
// ────────────────────────────────────────────────────────────────────────

/// Global union-find table for dimension variables. Lives once per
/// compilation (typically owned by the `IRGenerator`). All envelopes in
/// the same compilation share this table — there is no per-envelope
/// scoping.
#[derive(Clone, Debug, Default)]
pub struct DimTable {
    parent: Vec<u32>,
}

impl DimTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a fresh dim var with no equalities.
    pub fn fresh(&mut self) -> DimVar {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        DimVar(id)
    }

    /// Path-compressed find.
    pub fn find(&mut self, v: DimVar) -> DimVar {
        let mut cur = v.0;
        loop {
            let parent = self.parent[cur as usize];
            if parent == cur {
                return DimVar(cur);
            }
            // Halving: point cur to its grandparent.
            let grand = self.parent[parent as usize];
            self.parent[cur as usize] = grand;
            cur = grand;
        }
    }

    /// Unify two dim vars in the same equivalence class.
    pub fn unify(&mut self, a: DimVar, b: DimVar) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            // Pick the lower id as the canonical root for determinism.
            let (root, child) = if ra.0 < rb.0 { (ra, rb) } else { (rb, ra) };
            self.parent[child.0 as usize] = root.0;
        }
    }

    /// Are two dim vars currently in the same class?
    pub fn equal(&mut self, a: DimVar, b: DimVar) -> bool {
        self.find(a) == self.find(b)
    }

    /// Number of allocated dim vars (mostly useful in tests).
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────
// Dim — one axis of an envelope
// ────────────────────────────────────────────────────────────────────────

/// A single dimension of an [`Envelope`]. Always carries a symbolic
/// identity (`var`) plus a numeric `min..=max` range.
///
/// A dim with `min == max` is statically known. Use [`Dim::is_static`] to
/// detect this case rather than pattern-matching, so callers don't have to
/// care that "Static" and "Dynamic" share the same struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim {
    pub var: DimVar,
    pub min: usize,
    pub max: usize,
}

impl Dim {
    /// A dim known statically to be exactly `n` elements.
    pub fn new_static(table: &mut DimTable, n: usize) -> Self {
        Dim {
            var: table.fresh(),
            min: n,
            max: n,
        }
    }

    /// A dim with runtime length in `[min, max]`. Asserts `min <= max`.
    pub fn new_dynamic(table: &mut DimTable, min: usize, max: usize) -> Self {
        assert!(min <= max, "Dim::new_dynamic: min ({}) > max ({})", min, max);
        Dim {
            var: table.fresh(),
            min,
            max,
        }
    }

    /// `Some(n)` iff the dim is statically known to be exactly `n`.
    pub fn is_static(&self) -> Option<usize> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Envelope — fixed-rank shape bound
// ────────────────────────────────────────────────────────────────────────

/// Compile-time shape envelope for a (possibly dynamic) ndarray. Rank is
/// fixed at envelope construction time.
///
/// `total_bound` is the cross-axis upper bound on the total element count
/// (the `T` from ROADMAP/04-lazy-views.md §2). It may be tighter than the
/// product of per-axis maxima when cross-axis constraints are known (e.g.,
/// an input declared with `max_length = 100` reshaped to 2-D keeps
/// `total_bound = 100` even though each axis could individually be up to
/// 100).
#[derive(Clone, Debug)]
pub struct Envelope {
    pub dims: Vec<Dim>,
    pub total_bound: usize,
}

impl Default for Envelope {
    fn default() -> Self {
        Envelope {
            dims: Vec::new(),
            total_bound: 1,
        }
    }
}

impl Envelope {
    /// Construct an envelope from dims. `total_bound` defaults to the
    /// product of per-axis maxima (the loosest valid bound).
    pub fn new(dims: Vec<Dim>) -> Self {
        let total_bound = if dims.is_empty() {
            1
        } else {
            dims.iter().map(|d| d.max).product()
        };
        Envelope { dims, total_bound }
    }

    /// Construct an envelope with an explicit cross-axis total bound that
    /// is tighter than the per-axis product.
    pub fn new_with_bound(dims: Vec<Dim>, total_bound: usize) -> Self {
        Envelope { dims, total_bound }
    }

    /// 0-D scalar envelope (rank 0, total_bound = 1).
    pub fn scalar() -> Self {
        Envelope {
            dims: Vec::new(),
            total_bound: 1,
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Worst-case total element count, respecting the cross-axis
    /// `total_bound`. Returns `min(∏ dim.max, total_bound)`.
    pub fn max_total(&self) -> usize {
        let product: usize = if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().map(|d| d.max).product()
        };
        product.min(self.total_bound)
    }

    /// Best-case total element count (product of `min`).
    pub fn min_total(&self) -> usize {
        self.dims.iter().map(|d| d.min).product()
    }

    /// `Some(static_shape)` iff every dim is statically known.
    pub fn is_fully_static(&self) -> Option<Vec<usize>> {
        self.dims.iter().map(|d| d.is_static()).collect()
    }

    /// True if rank is 0.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Build an envelope from a known static shape, allocating one fresh
    /// dim var per axis. `total_bound` is set to the product of the shape
    /// (exact, since all dims are static).
    pub fn from_static_shape(table: &mut DimTable, shape: &[usize]) -> Self {
        let dims: Vec<Dim> = shape.iter().map(|&n| Dim::new_static(table, n)).collect();
        let total_bound = if shape.is_empty() { 1 } else { shape.iter().product() };
        Envelope { dims, total_bound }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Tier 2 broadcasting
// ────────────────────────────────────────────────────────────────────────

/// Broadcast two envelopes per NumPy rules, with `DimVar` unification on
/// runtime-equal axes. The output envelope has rank `max(a.rank, b.rank)`,
/// dims right-aligned, and bound intersection on overlapping dims.
///
/// **Cases per axis (right-aligned):**
/// - Only `a` provides a dim → output is `a.dim`.
/// - Only `b` provides a dim → output is `b.dim`.
/// - Both static `1` → output is fresh static `1`.
/// - One static `1` → output is the *other* dim (1 broadcasts).
/// - Both static, equal value, neither is `1` → output is that static.
///   Vars are *not* unified (the two dims happen to have the same value
///   but may come from independent sources).
/// - Both static, different values, neither is `1` → `Err`.
/// - At least one is not statically `1` and not statically equal → require
///   runtime equality. Unify the two vars and intersect bounds.
///   - If the bound intersection is empty (`max(a.min, b.min) > min(a.max, b.max)`),
///     `Err`.
///   - Otherwise the output dim takes the intersected bounds and the unified
///     var as its identity.
///
/// **Caveat (case 6 from the design discussion):** if a dim has `min == 1`
/// but `max > 1`, we can't decide statically whether it should broadcast or
/// require runtime equality. We *always* require runtime equality in that
/// case, losing the implicit dynamic broadcast. Users who want broadcast
/// from a dynamic 1 must call `np.broadcast_to` explicitly.
pub fn broadcast_envelopes(
    table: &mut DimTable,
    a: &Envelope,
    b: &Envelope,
) -> Result<Envelope, String> {
    let rank = a.rank().max(b.rank());
    let mut out_dims: Vec<Dim> = Vec::with_capacity(rank);

    // Track a-only and b-only broadcast factors for total_bound computation.
    // a-only factor: product of output dims at positions where b has size 1
    //                (or b doesn't extend to that rank).
    // b-only factor: symmetric.
    let mut a_only_factor: usize = 1;
    let mut b_only_factor: usize = 1;
    let mut has_shared_axis = false;

    for i in 0..rank {
        // Right-aligned indexing: axis `i` from the back of each operand.
        let a_axis = if i < a.rank() { Some(a.rank() - 1 - i) } else { None };
        let b_axis = if i < b.rank() { Some(b.rank() - 1 - i) } else { None };

        let dim = match (a_axis, b_axis) {
            (Some(ai), None) => {
                // a extends beyond b's rank: this is an a-only axis.
                // b implicitly has size 1 here.
                b_only_factor *= a.dims[ai].max;
                a.dims[ai]
            }
            (None, Some(bi)) => {
                // b extends beyond a's rank: b-only axis.
                a_only_factor *= b.dims[bi].max;
                b.dims[bi]
            }
            (Some(ai), Some(bi)) => {
                let x = a.dims[ai];
                let y = b.dims[bi];
                // Classify: if one is static 1, the other is a-only or b-only.
                if x.is_static() == Some(1) && y.is_static() != Some(1) {
                    // a has 1 here → b-only axis (a broadcasts)
                    a_only_factor *= y.max;
                } else if y.is_static() == Some(1) && x.is_static() != Some(1) {
                    // b has 1 here → a-only axis (b broadcasts)
                    b_only_factor *= x.max;
                } else {
                    // Shared axis (both non-1, or both 1)
                    has_shared_axis = true;
                }
                broadcast_one_dim(table, x, y, i)?
            }
            (None, None) => unreachable!(),
        };
        out_dims.push(dim);
    }

    out_dims.reverse();

    // Compute output total_bound per §3.3/§3.4:
    // - If no broadcast (a_only_factor == 1 && b_only_factor == 1): same-shape.
    //   T_out = min(T_a, T_b).
    // - If broadcast with a-only or b-only axes:
    //   T_out = min(T_a * a_only_factor, T_b * b_only_factor).
    //   a_only_factor is the product of b-only dims (which multiply a's elements).
    //   b_only_factor is the product of a-only dims (which multiply b's elements).
    //   (Naming: a_only_factor means "factor applied to T_a" — the dims that
    //   are new to a, coming from b's broadcast axes.)
    let _ = has_shared_axis;
    let t_a_contrib = a.total_bound.saturating_mul(a_only_factor);
    let t_b_contrib = b.total_bound.saturating_mul(b_only_factor);
    let total_bound = t_a_contrib.min(t_b_contrib);

    Ok(Envelope::new_with_bound(out_dims, total_bound))
}

/// Apply the per-axis broadcast rule. `axis_from_back` is purely for the
/// error message.
fn broadcast_one_dim(
    table: &mut DimTable,
    x: Dim,
    y: Dim,
    axis_from_back: usize,
) -> Result<Dim, String> {
    let xs = x.is_static();
    let ys = y.is_static();

    // Both statically 1 → fresh static 1.
    if xs == Some(1) && ys == Some(1) {
        return Ok(Dim::new_static(table, 1));
    }
    // One statically 1 → return the other (broadcast the 1).
    if xs == Some(1) {
        return Ok(y);
    }
    if ys == Some(1) {
        return Ok(x);
    }
    // Both statically known and equal → that value, no unification.
    if let (Some(xv), Some(yv)) = (xs, ys) {
        if xv == yv {
            return Ok(x);
        }
        return Err(format!(
            "broadcast: incompatible static dims at axis -{}: {} vs {}",
            axis_from_back + 1,
            xv,
            yv
        ));
    }
    // Already unified → return one of them (intersect bounds defensively).
    if table.equal(x.var, y.var) {
        let new_min = x.min.max(y.min);
        let new_max = x.max.min(y.max);
        if new_min > new_max {
            return Err(format!(
                "broadcast: empty bound intersection at axis -{}: ({}..={}) vs ({}..={})",
                axis_from_back + 1,
                x.min,
                x.max,
                y.min,
                y.max
            ));
        }
        return Ok(Dim {
            var: table.find(x.var),
            min: new_min,
            max: new_max,
        });
    }
    // Need runtime equality: intersect bounds, unify vars.
    let new_min = x.min.max(y.min);
    let new_max = x.max.min(y.max);
    if new_min > new_max {
        return Err(format!(
            "broadcast: empty bound intersection at axis -{}: ({}..={}) vs ({}..={})",
            axis_from_back + 1,
            x.min,
            x.max,
            y.min,
            y.max
        ));
    }
    table.unify(x.var, y.var);
    Ok(Dim {
        var: table.find(x.var),
        min: new_min,
        max: new_max,
    })
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn static_envelope(table: &mut DimTable, shape: &[usize]) -> Envelope {
        let dims = shape.iter().map(|&n| Dim::new_static(table, n)).collect();
        Envelope::new(dims)
    }

    fn dynamic_envelope(table: &mut DimTable, bounds: &[(usize, usize)]) -> Envelope {
        let dims = bounds
            .iter()
            .map(|&(lo, hi)| Dim::new_dynamic(table, lo, hi))
            .collect();
        Envelope::new(dims)
    }

    // ── DimTable ────────────────────────────────────────────────────

    #[test]
    fn dim_table_fresh_distinct() {
        let mut t = DimTable::new();
        let a = t.fresh();
        let b = t.fresh();
        assert_ne!(a, b);
        assert!(!t.equal(a, b));
    }

    #[test]
    fn dim_table_unify_makes_equal() {
        let mut t = DimTable::new();
        let a = t.fresh();
        let b = t.fresh();
        t.unify(a, b);
        assert!(t.equal(a, b));
    }

    #[test]
    fn dim_table_unify_transitive() {
        let mut t = DimTable::new();
        let a = t.fresh();
        let b = t.fresh();
        let c = t.fresh();
        t.unify(a, b);
        t.unify(b, c);
        assert!(t.equal(a, c));
    }

    // ── Dim helpers ─────────────────────────────────────────────────

    #[test]
    fn dim_static_is_static() {
        let mut t = DimTable::new();
        let d = Dim::new_static(&mut t, 7);
        assert_eq!(d.is_static(), Some(7));
        assert_eq!(d.min, 7);
        assert_eq!(d.max, 7);
    }

    #[test]
    fn dim_dynamic_is_not_static() {
        let mut t = DimTable::new();
        let d = Dim::new_dynamic(&mut t, 0, 100);
        assert_eq!(d.is_static(), None);
    }

    #[test]
    fn dim_dynamic_collapsed_range_is_static() {
        let mut t = DimTable::new();
        let d = Dim::new_dynamic(&mut t, 5, 5);
        assert_eq!(d.is_static(), Some(5));
    }

    // ── Envelope basics ─────────────────────────────────────────────

    #[test]
    fn envelope_scalar() {
        let e = Envelope::scalar();
        assert_eq!(e.rank(), 0);
        assert!(e.is_scalar());
        assert_eq!(e.max_total(), 1);
        assert_eq!(e.is_fully_static(), Some(vec![]));
    }

    #[test]
    fn envelope_fully_static_detects() {
        let mut t = DimTable::new();
        let e = static_envelope(&mut t, &[3, 4]);
        assert_eq!(e.is_fully_static(), Some(vec![3, 4]));
        assert_eq!(e.max_total(), 12);
        assert_eq!(e.min_total(), 12);
    }

    #[test]
    fn envelope_partially_dynamic_not_fully_static() {
        let mut t = DimTable::new();
        let e = Envelope::new(vec![
            Dim::new_static(&mut t, 3),
            Dim::new_dynamic(&mut t, 0, 100),
        ]);
        assert_eq!(e.is_fully_static(), None);
        assert_eq!(e.max_total(), 300);
        assert_eq!(e.min_total(), 0);
    }

    // ── broadcast_envelopes ─────────────────────────────────────────

    #[test]
    fn broadcast_static_same_shape() {
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[3, 4]);
        let b = static_envelope(&mut t, &[3, 4]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.is_fully_static(), Some(vec![3, 4]));
    }

    #[test]
    fn broadcast_static_outer_product() {
        // (3, 1) and (1, 4) -> (3, 4)
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[3, 1]);
        let b = static_envelope(&mut t, &[1, 4]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.is_fully_static(), Some(vec![3, 4]));
    }

    #[test]
    fn broadcast_lower_rank_left_pads() {
        // (4,) against (3, 4) -> (3, 4)
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[4]);
        let b = static_envelope(&mut t, &[3, 4]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.is_fully_static(), Some(vec![3, 4]));
    }

    #[test]
    fn broadcast_static_incompatible_errors() {
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[3]);
        let b = static_envelope(&mut t, &[4]);
        assert!(broadcast_envelopes(&mut t, &a, &b).is_err());
    }

    #[test]
    fn broadcast_dynamic_intersects_bounds() {
        // (Dyn 0..=10) and (Dyn 5..=20) -> (Dyn 5..=10)
        let mut t = DimTable::new();
        let a = dynamic_envelope(&mut t, &[(0, 10)]);
        let b = dynamic_envelope(&mut t, &[(5, 20)]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.rank(), 1);
        assert_eq!(out.dims[0].min, 5);
        assert_eq!(out.dims[0].max, 10);
        assert_eq!(out.dims[0].is_static(), None);
    }

    #[test]
    fn broadcast_dynamic_unifies_vars() {
        let mut t = DimTable::new();
        let a = dynamic_envelope(&mut t, &[(0, 100)]);
        let b = dynamic_envelope(&mut t, &[(0, 100)]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        // After broadcasting, the input dim vars should be unified.
        assert!(t.equal(a.dims[0].var, b.dims[0].var));
        // The output var is in the same class as both inputs.
        assert!(t.equal(out.dims[0].var, a.dims[0].var));
    }

    #[test]
    fn broadcast_dynamic_disjoint_bounds_errors() {
        // (Dyn 0..=4) and (Dyn 10..=20) → no intersection.
        let mut t = DimTable::new();
        let a = dynamic_envelope(&mut t, &[(0, 4)]);
        let b = dynamic_envelope(&mut t, &[(10, 20)]);
        assert!(broadcast_envelopes(&mut t, &a, &b).is_err());
    }

    #[test]
    fn broadcast_static_against_dynamic_collapses_to_static() {
        // Static 5 vs Dynamic 0..=10 → output bounds = (5, 5) → static 5.
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[5]);
        let b = dynamic_envelope(&mut t, &[(0, 10)]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.dims[0].is_static(), Some(5));
    }

    #[test]
    fn broadcast_static_against_dynamic_out_of_range_errors() {
        // Static 5 vs Dynamic 6..=10 → no intersection.
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[5]);
        let b = dynamic_envelope(&mut t, &[(6, 10)]);
        assert!(broadcast_envelopes(&mut t, &a, &b).is_err());
    }

    #[test]
    fn broadcast_static_one_against_anything() {
        // Static 1 vs Static 7 → Static 7 (broadcast the 1).
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[1]);
        let b = static_envelope(&mut t, &[7]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.dims[0].is_static(), Some(7));
    }

    #[test]
    fn broadcast_static_one_against_dynamic() {
        // Static 1 vs Dynamic 5..=10 → Dynamic 5..=10.
        let mut t = DimTable::new();
        let a = static_envelope(&mut t, &[1]);
        let b = dynamic_envelope(&mut t, &[(5, 10)]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.dims[0].min, 5);
        assert_eq!(out.dims[0].max, 10);
    }

    #[test]
    fn broadcast_three_dim_mixed() {
        // (Static 2, Static 1, Dyn 0..=8) and (Static 1, Static 4, Dyn 0..=8)
        //   -> (Static 2, Static 4, Dyn 0..=8)
        let mut t = DimTable::new();
        let a = Envelope::new(vec![
            Dim::new_static(&mut t, 2),
            Dim::new_static(&mut t, 1),
            Dim::new_dynamic(&mut t, 0, 8),
        ]);
        let b = Envelope::new(vec![
            Dim::new_static(&mut t, 1),
            Dim::new_static(&mut t, 4),
            Dim::new_dynamic(&mut t, 0, 8),
        ]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.rank(), 3);
        assert_eq!(out.dims[0].is_static(), Some(2));
        assert_eq!(out.dims[1].is_static(), Some(4));
        assert_eq!(out.dims[2].is_static(), None);
        assert_eq!(out.dims[2].min, 0);
        assert_eq!(out.dims[2].max, 8);
    }

    #[test]
    fn broadcast_scalar_against_anything() {
        let mut t = DimTable::new();
        let a = Envelope::scalar();
        let b = static_envelope(&mut t, &[3, 4]);
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.is_fully_static(), Some(vec![3, 4]));
    }

    // ── total_bound ─────────────────────────────────────────────────

    #[test]
    fn total_bound_static_envelope() {
        let mut t = DimTable::new();
        let e = Envelope::from_static_shape(&mut t, &[3, 4]);
        assert_eq!(e.total_bound, 12);
        assert_eq!(e.max_total(), 12);
    }

    #[test]
    fn total_bound_scalar() {
        let e = Envelope::scalar();
        assert_eq!(e.total_bound, 1);
        assert_eq!(e.max_total(), 1);
    }

    #[test]
    fn total_bound_tighter_than_product() {
        // Simulate a reshaped input: per-axis max is 100 each, but
        // total_bound is 100 (from the original max_length).
        let mut t = DimTable::new();
        let e = Envelope::new_with_bound(
            vec![
                Dim::new_dynamic(&mut t, 0, 100),
                Dim::new_dynamic(&mut t, 0, 100),
            ],
            100,
        );
        // Per-axis product would be 10,000, but total_bound caps it.
        assert_eq!(e.max_total(), 100);
        assert_eq!(e.total_bound, 100);
    }

    #[test]
    fn total_bound_broadcast_same_shape() {
        // Same-shape broadcast: T_out = min(T_a, T_b).
        let mut t = DimTable::new();
        let a = Envelope::new_with_bound(
            vec![Dim::new_dynamic(&mut t, 0, 100)],
            80,
        );
        let b = Envelope::new_with_bound(
            vec![Dim::new_dynamic(&mut t, 0, 100)],
            60,
        );
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        // No a-only or b-only axes, so T_out = min(80, 60) = 60.
        assert_eq!(out.total_bound, 60);
        assert_eq!(out.max_total(), 60);
    }

    #[test]
    fn total_bound_broadcast_with_static_one() {
        // [D, 1] + [1, 3] → [D, 3]. b-only factor = D_max, a-only factor = 3.
        // T_out = min(T_a * 3, T_b * D_max) = min(100 * 3, 3 * 100) = 300.
        let mut t = DimTable::new();
        let a = Envelope::new_with_bound(
            vec![
                Dim::new_dynamic(&mut t, 0, 100),
                Dim::new_static(&mut t, 1),
            ],
            100,
        );
        let b = Envelope::new_with_bound(
            vec![
                Dim::new_static(&mut t, 1),
                Dim::new_static(&mut t, 3),
            ],
            3,
        );
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.rank(), 2);
        // a-only factor (from b's perspective): the static 3 from b
        // b-only factor (from a's perspective): D_max = 100 from a
        // T_out = min(T_a * a_only_factor, T_b * b_only_factor)
        //       = min(100 * 3, 3 * 100) = 300
        assert_eq!(out.total_bound, 300);
    }

    #[test]
    fn total_bound_broadcast_rank_expansion() {
        // [D] + [3, D'] → [3, D_unified]. b-only factor = 1, a-only factor = 3.
        // axis 1 (from right): D vs D' → shared (unified)
        // axis 0 (from right): only b → b contributes static 3
        // a_only_factor = 3 (b's axis 0 is added to a), b_only_factor = 1
        // T_out = min(T_a * 3, T_b * 1) = min(300, 30) = 30
        let mut t = DimTable::new();
        let a = Envelope::new_with_bound(
            vec![Dim::new_dynamic(&mut t, 0, 100)],
            100,
        );
        let b = Envelope::new_with_bound(
            vec![
                Dim::new_static(&mut t, 3),
                Dim::new_dynamic(&mut t, 0, 100),
            ],
            30,
        );
        let out = broadcast_envelopes(&mut t, &a, &b).unwrap();
        assert_eq!(out.rank(), 2);
        assert_eq!(out.total_bound, 30);
    }

    #[test]
    fn total_bound_preserved_through_new() {
        // Envelope::new() sets total_bound = product of dim.max.
        let mut t = DimTable::new();
        let e = Envelope::new(vec![
            Dim::new_static(&mut t, 3),
            Dim::new_static(&mut t, 4),
        ]);
        assert_eq!(e.total_bound, 12);
        assert_eq!(e.max_total(), 12);
    }
}

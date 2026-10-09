//! Narrow an already-positive division using checked operand ranges.
use super::*;

impl NoirEmitter<'_> {
    pub(super) fn bounded_positive_division(&mut self, numerator: &str, divisor: &str) -> Option<String> {
        let (nl, nh) = self.known_range(numerator)?;
        let (dl, dh) = self.known_range(divisor)?;
        if nl < 0 || nh >= 1i128 << 63 || dl < 0 || dh < 1 || dh >= 1i128 << 63 { return None }
        // The calling path already requires a nonzero divisor. Both bounded
        // operands are integers, so 0 <= q <= nh and 0 <= r, d-r-1 < dh.
        // Keep widths aligned with Noir's existing 14-bit range tables.
        let aligned = |n: i128| ((128 - (n as u128).leading_zeros()).max(1).div_ceil(14) * 14).min(64);
        let qb = aligned(nh);
        let rb = aligned(dh - 1);
        if qb == 64 && rb == 64 { return None }
        self.used_bounded_division_helper = true;
        let out = self.bind(&format!("zinnia_bounded_div::<{qb}, {rb}>({numerator}, {divisor})"));
        self.proven_range.insert(out.clone(), (0, nh));
        Some(out)
    }
}

pub(super) const HELPERS: &str = r#"
unconstrained fn zinnia_bounded_div_hint(n: Field, d: Field) -> (Field, Field) {
    let n = n as u128;
    let d = d as u128;
    if d == 0 { (0, 0) } else { ((n / d) as Field, (n % d) as Field) }
}
fn zinnia_bounded_div<let QB: u32, let RB: u32>(n: Field, d: Field) -> Field {
    let (q, r) = unsafe { zinnia_bounded_div_hint(n, d) };
    assert(q * d + r == n);
    q.assert_max_bit_size::<QB>();
    r.assert_max_bit_size::<RB>();
    (d - r - 1).assert_max_bit_size::<RB>();
    q
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parity_emitter_bounded_division_requires_checked_ranges() {
        let p = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &p);
        em.interval_of.insert("d".into(), (0, 5));
        let n = em.constant_int(1i64 << 32).unwrap();
        assert!(em.bounded_positive_division(&n, "d").is_none());
        em.derived_range.insert("d".into(), (0, 5));
        let q = em.bounded_positive_division(&n, "d").unwrap();
        assert!(em.body.contains("zinnia_bounded_div::<42, 14>"));
        assert_eq!(em.known_range(&q), Some((0, 1i128 << 32)));
        em.derived_range.insert("d".into(), (-1, 5));
        assert!(em.bounded_positive_division(&n, "d").is_none());
    }

    #[test]
    fn parity_emitter_boolean_sums_gain_only_guaranteed_ranges() {
        let p = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &p);
        em.set_bound("unproven", 1);
        assert_eq!(em.known_range("unproven"), None);
        let a = em.bind_flag("if x == 0 { 1 } else { 0 }");
        let b = em.bind_flag("if y == 0 { 1 } else { 0 }");
        let sum = em.add_i(&a, &b).unwrap();
        assert_eq!(em.known_range(&sum), Some((0, 2)));
    }
}

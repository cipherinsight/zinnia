//! Relational bound a < max(...,a,...)+c for c > 0.
use super::*;

pub(super) fn plan(ir: &IRGraph) -> HashSet<u32> {
    use crate::ir_defs::IR;
    let mut maxima = HashMap::<u32, (u32, u32)>::new();
    let mut positive_add = HashMap::<u32, u32>::new();
    let mut lower = HashMap::<u32, Vec<u32>>::new();
    // Explicit unconditional comparisons are relational facts too. Hinted
    // extrema lower to these checks rather than a long max-select chain.
    for s in &ir.stmts {
        let (IR::Assert, [cmp]) = (&s.ir, s.arguments.as_slice()) else { continue };
        let c = &ir.stmts[*cmp as usize];
        let [a, b] = c.arguments.as_slice() else { continue };
        let pair = match c.ir {
            IR::LtI | IR::LteI | IR::LtF | IR::LteF => Some((*b, *a)),
            IR::GtI | IR::GteI | IR::GtF | IR::GteF => Some((*a, *b)),
            _ => None,
        };
        if let Some((upper, smaller)) = pair { lower.entry(upper).or_default().push(smaller); }
    }
    let constant = |i: u32| match &ir.stmts[i as usize].ir {
        IR::ConstantInt { value } => Some(*value as i128),
        IR::ConstantFloat { value } => Some(crate::prove::field::quantize(*value, Q_BITS)),
        _ => None,
    };
    let mut result = HashSet::new();
    for s in &ir.stmts {
        match (&s.ir, s.arguments.as_slice()) {
            (IR::SelectI | IR::SelectF, [cmp, t, f]) => {
                let c = &ir.stmts[*cmp as usize];
                let [a, b] = c.arguments.as_slice() else { continue };
                let max = match c.ir {
                    IR::GtI | IR::GteI | IR::GtF | IR::GteF => t == a && f == b,
                    IR::LtI | IR::LteI | IR::LtF | IR::LteF => t == b && f == a,
                    _ => false,
                };
                if max { maxima.insert(s.stmt_id, (*t, *f)); }
            }
            (IR::AddF | IR::AddI, [a, b]) => {
                let root = if constant(*a).is_some_and(|c| c > 0) { Some(*b) }
                    else if constant(*b).is_some_and(|c| c > 0) { Some(*a) } else { None };
                if let Some(root) = root { positive_add.insert(s.stmt_id, root); }
            }
            (IR::DivF, [a, b]) => {
                let Some(root) = positive_add.get(b) else { continue };
                let mut stack = vec![*root];
                let mut visited = HashSet::new();
                while let Some(x) = stack.pop() {
                    if !visited.insert(x) { continue }
                    if x == *a { result.insert(s.stmt_id); break }
                    if let Some((left, right)) = maxima.get(&x) { stack.extend([*left, *right]); }
                    if let Some(bounds) = lower.get(&x) { stack.extend(bounds.iter().copied()); }
                }
            }
            _ => (),
        }
    }
    result
}

impl NoirEmitter<'_> {
    pub(super) fn fraction_division(&mut self, a: &str, b: &str) -> Option<String> {
        if self.integral_float.contains_key(b) { return None }
        let strict = self.current_stmt.is_some_and(|id| self.fraction_plan.contains(&id));
        if !strict {
            if !self.nonzero.contains(b) { return None }
            let mut pending = vec![b];
            let mut visited = HashSet::new();
            let mut contains = false;
            while let Some(x) = pending.pop() {
                if !visited.insert(x) { continue }
                if x == a { contains = true; break }
                if let Some((left, right)) = self.sum_of.get(x) {
                    if [left, right].into_iter().all(|v| self.known_range(v).is_some_and(|(lo, _)| lo >= 0)) {
                        pending.extend([left.as_str(), right.as_str()]);
                    }
                }
            }
            if !contains { return None }
        }
        let (al, ah) = self.known_range(a)?;
        let (bl, bh) = self.known_range(b)?;
        // The relational facts imply the original signed64 quotient window.
        // Preserve the original sign_abs64 magnitude domain on broad sums.
        if al < 0 || ah >= 1i128 << 64 || bl < 0 { return None }
        let old_positive_division = self.nonzero.contains(b) && self.known_non_negative(b);
        if bh >= 1i128 << 64 && !old_positive_division {
            let _ = writeln!(self.body, "    {b}.assert_max_bit_size::<64>();");
        }
        self.used_fraction_helper = true;
        let width = if strict { 32 } else { 33 };
        let out = self.bind(&format!("zinnia_fraction_div::<{width}>({a}, {b})"));
        self.proven_range.insert(out.clone(), (0, (1i128 << Q_BITS) - strict as i128));
        Some(out)
    }

}

pub(super) const HELPERS: &str = r#"
unconstrained fn zinnia_fraction_hint(a: Field, b: Field) -> (Field, Field) {
    let n = (a * 4294967296) as u128;
    let d = b as u128;
    ((n / d) as Field, (n % d) as Field)
}
fn zinnia_fraction_div<let W: u32>(a: Field, b: Field) -> Field {
    let (q, r) = unsafe { zinnia_fraction_hint(a, b) };
    assert(q * b + r == a * 4294967296);
    r.assert_max_bit_size::<64>();
    (b - r - 1).assert_max_bit_size::<64>();
    q.assert_max_bit_size::<W>();
    q
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::IRStatement, ir_defs::IR, types::ValueId};

    #[test]
    fn parity_emitter_fraction_needs_strict_max_bound_and_checked_ranges() {
        let specs = vec![
            (IR::ReadFloat { path: InputPath::new("x", vec![]), is_public: false }, vec![]),
            (IR::AbsF, vec![0]),
            (IR::ConstantFloat { value: 1.0 }, vec![]),
            (IR::GtF, vec![1, 2]),
            (IR::SelectF, vec![3, 1, 2]),
            (IR::ConstantFloat { value: 4.0 }, vec![]),
            (IR::AddF, vec![4, 5]),
            (IR::DivF, vec![1, 6]),
            (IR::DivF, vec![1, 4]),
        ];
        let ir = IRGraph::new(specs.into_iter().enumerate().map(|(i, (ir, args))| {
            let values = args.iter().map(|a| ValueId(*a as u64)).collect();
            IRStatement::new(i as u32, ValueId(i as u64), ir, args, values, None)
        }).collect());
        assert_eq!(plan(&ir), HashSet::from([7]));
        // A hinted maximum is certified by unconditional a <= maximum
        // assertions; the planner must not rely on the hint itself.
        let mut asserted_ir = IRGraph::new(ir.stmts.clone());
        asserted_ir.stmts[4].ir = IR::ReadFloat { path: InputPath::new("maximum", vec![]), is_public: false };
        asserted_ir.stmts[4].arguments.clear();
        asserted_ir.stmts[4].arg_values.clear();
        assert!(plan(&asserted_ir).is_empty());
        asserted_ir.stmts.push(IRStatement::new(9, ValueId(9), IR::LteF, vec![1, 4], vec![ValueId(1), ValueId(4)], None));
        asserted_ir.stmts.push(IRStatement::new(10, ValueId(10), IR::Assert, vec![9], vec![ValueId(9)], None));
        assert_eq!(plan(&asserted_ir), HashSet::from([7]));
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &prov);
        em.fraction_plan = plan(&ir);
        em.current_stmt = Some(7);
        assert!(em.fraction_division("a", "b").is_none());
        em.proven_range.insert("a".into(), (0, (1i128 << 63) - 1));
        em.derived_range.insert("b".into(), (4, (1i128 << 63) + 4));
        let out = em.fraction_division("a", "b").unwrap();
        assert_eq!(em.known_range(&out), Some((0, (1i128 << 32) - 1)));
        em.derived_range.insert("b".into(), (4, 1i128 << 64));
        assert!(em.fraction_division("a", "b").is_some());
        assert!(em.body.contains("b.assert_max_bit_size::<64>()"));
        em.current_stmt = None;
        em.sum_of.insert("b".into(), ("a".into(), "c".into()));
        em.proven_range.insert("c".into(), (0, 5));
        em.nonzero.insert("b".into());
        assert!(em.fraction_division("a", "b").is_some());
        assert!(em.body.contains("zinnia_fraction_div::<33>(a, b)"));
        // Historical positive-divisor division admits magnitudes above2^64
        // when the two remainder windows fit; retain that acceptance domain.
        em.proven_range.insert("b".into(), (4, (1i128 << 65) - 2));
        let checks = em.body.matches("b.assert_max_bit_size::<64>()").count();
        assert!(em.fraction_division("a", "b").is_some());
        assert_eq!(em.body.matches("b.assert_max_bit_size::<64>()").count(), checks);
        em.integral_float.insert("b".into(), "integer_b".into());
        assert!(em.fraction_division("a", "b").is_none());
    }
}

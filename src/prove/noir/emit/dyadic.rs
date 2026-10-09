//! Shared Euclidean decompositions for reciprocal-power-of-two products.
//! Planning is purely structural; emission additionally requires checked root
//! bounds, which imply every replaced intermediate signed quotient window.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Product {
    root: crate::types::ValueId,
    shift: u32,
    negative: bool,
    max_shift: u32,
    min_shift: u32,
}

pub(super) fn plan(ir: &IRGraph) -> HashMap<u32, Product> {
    use crate::ir_defs::IR;
    let mut candidates: HashMap<u32, Product> = HashMap::new();
    for stmt in &ir.stmts {
        let (IR::MulF, [a, b]) = (&stmt.ir, stmt.arguments.as_slice()) else { continue };
        let constant = |id: u32| match ir.stmts.get(id as usize).map(|s| &s.ir) {
            Some(IR::ConstantFloat { value }) => Some(crate::prove::field::quantize(*value, Q_BITS)),
            _ => None,
        };
        let Some((x, raw)) = constant(*a).map(|c| (*b, c)).or_else(|| constant(*b).map(|c| (*a, c))) else { continue };
        let abs = raw.unsigned_abs();
        if !abs.is_power_of_two() || abs >= 1u128 << Q_BITS { continue }
        let shift = Q_BITS - abs.trailing_zeros();
        let mut product = Product {
            root: ir.stmts[x as usize].value_id, shift, negative: raw < 0,
            max_shift: shift, min_shift: shift,
        };
        // Positive outer floors compose. A negative outer floor does not.
        if raw > 0 {
            if let Some(inner) = candidates.get(&x) {
                product.root = inner.root;
                product.shift += inner.shift;
                product.negative = inner.negative;
            }
        }
        if product.shift <= 8 { candidates.insert(stmt.stmt_id, product); }
    }
    let mut groups: HashMap<crate::types::ValueId, (usize, u32, u32)> = HashMap::new();
    for p in candidates.values() {
        let g = groups.entry(p.root).or_insert((0, 0, 32));
        g.0 += 1;
        g.1 = g.1.max(p.shift);
        g.2 = g.2.min(p.shift);
    }
    candidates.retain(|_, p| {
        let (count, max, min) = groups[&p.root];
        p.max_shift = max;
        p.min_shift = min;
        count > 1
    });
    candidates
}

impl NoirEmitter<'_> {
    pub(super) fn shared_dyadic_product(&mut self) -> Option<String> {
        let plan = *self.dyadic_plan.get(&self.current_stmt?)?;
        let root = self.var_of_vid.get(&plan.root)?.clone();
        let (lo, hi) = self.known_range(&root)?;
        // These bounds prevent field wrap and make every intermediate
        // +/- floor(root/2^k) fit signed64, including negative rounding.
        let limit = 1i128 << (62 + plan.min_shift.min(2));
        if lo < -limit || hi >= limit { return None }
        let (high, bits) = if let Some(pair) = self.dyadic_splits.get(&root) {
            pair.clone()
        } else {
            self.used_dyadic_helper = true;
            let pair = self.divmod_call(format!("zinnia_dyadic_split::<{}>({root})", plan.max_shift));
            self.record_signed64(&pair.0);
            self.dyadic_splits.insert(root, pair.clone());
            pair
        };
        let mut terms = vec![format!("{high} * {}", 1u64 << (plan.max_shift - plan.shift))];
        for bit in plan.shift..plan.max_shift {
            terms.push(format!("{bits}[{bit}] * {}", 1u64 << (bit - plan.shift)));
        }
        let positive = if terms.len() == 1 && plan.shift == plan.max_shift { high }
            else { self.bind(&terms.join(" + ")) };
        let result = if plan.negative {
            let mut nz = format!("{bits}[0]");
            for bit in 1..plan.shift {
                nz = self.bind(&format!("{nz} + {bits}[{bit}] - {nz} * {bits}[{bit}]"));
            }
            self.bind(&format!("0 - {positive} - {nz}"))
        } else { positive };
        self.record_signed64(&result);
        Some(result)
    }
}

pub(super) const HELPERS: &str = r#"
unconstrained fn zinnia_dyadic_hint<let K: u32>(x: Field) -> (Field, [Field; K]) {
    let offset: u128 = 1 << 96;
    let shifted = (x + offset as Field) as u128;
    let high = (shifted >> (K as u128)) as Field - (offset >> (K as u128)) as Field;
    let mut bits: [Field; K] = [0; K];
    for i in 0..K { bits[i] = ((shifted >> (i as u128)) & 1) as Field; }
    (high, bits)
}
fn zinnia_dyadic_split<let K: u32>(x: Field) -> (Field, [Field; K]) {
    let (high, bits) = unsafe { zinnia_dyadic_hint::<K>(x) };
    let window = high + 9223372036854775808;
    window.assert_max_bit_size::<64>();
    let mut remainder: Field = 0;
    for i in 0..K {
        assert(bits[i] * (bits[i] - 1) == 0);
        remainder = remainder + bits[i] * 2.pow_32(i as Field);
    }
    assert(x == high * 2.pow_32(K as Field) + remainder);
    (high, bits)
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::IRStatement, ir_defs::IR, types::ValueId};

    #[test]
    fn parity_emitter_shared_dyadic_keeps_negative_remainder_and_bounds() {
        let specs = vec![
            (IR::ReadFloat { path: InputPath::new("x", vec![]), is_public: false }, vec![]),
            (IR::ConstantFloat { value: 0.25 }, vec![]),
            (IR::ConstantFloat { value: -0.25 }, vec![]),
            (IR::ConstantFloat { value: 0.5 }, vec![]),
            (IR::MulF, vec![0, 1]),
            (IR::MulF, vec![0, 2]),
            (IR::MulF, vec![4, 3]),
            (IR::MulF, vec![5, 3]),
        ];
        let ir = IRGraph::new(specs.into_iter().enumerate().map(|(i, (ir, args))| {
            let values = args.iter().map(|a| ValueId(*a as u64)).collect();
            IRStatement::new(i as u32, ValueId(i as u64), ir, args, values, None)
        }).collect());
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &prov);
        em.dyadic_plan = plan(&ir);
        em.var_of_vid.insert(ValueId(0), "x".into());
        em.current_stmt = Some(4);
        assert!(em.shared_dyadic_product().is_none());
        em.derived_range.insert("x".into(), (-(1i128 << 64), (1i128 << 64) - 1));
        for id in 4..8 {
            em.current_stmt = Some(id);
            assert!(em.shared_dyadic_product().is_some());
        }
        assert_eq!(em.body.matches("zinnia_dyadic_split::<3>(x)").count(), 1);
        assert_eq!(em.body.matches(": Field = 0 - ").count(), 2);
        assert!(em.body.contains("[2]"));
        assert!(!em.body.contains("qmul"));
    }
}

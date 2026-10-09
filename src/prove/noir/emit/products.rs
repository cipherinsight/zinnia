//! Share raw bilinear terms across separately rounded scaled products.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Product {
    left: crate::types::ValueId,
    right: crate::types::ValueId,
    scale: i128,
}

pub(super) fn plan(ir: &IRGraph) -> HashMap<u32, Product> {
    use crate::ir_defs::IR;
    let mut scaled = HashMap::<u32, (u32, i128)>::new();
    let mut products = HashMap::new();
    let mut counts = HashMap::<(crate::types::ValueId, crate::types::ValueId), usize>::new();
    for s in &ir.stmts {
        let (IR::MulF | IR::MulI, [a, b]) = (&s.ir, s.arguments.as_slice()) else { continue };
        let integral = |i: u32| match ir.stmts[i as usize].ir {
            IR::ConstantInt { value } if matches!(s.ir, IR::MulI) => Some(value as i128),
            IR::ConstantFloat { value } if matches!(s.ir, IR::MulF)
                && value.fract() == 0.0 && value.abs() < (1u64 << 31) as f64 => Some(value as i128),
            _ => None,
        };
        if let Some((x, n)) = integral(*a).map(|n| (*b, n)).or_else(|| integral(*b).map(|n| (*a, n))) {
            let (root, previous) = scaled.get(&x).copied().unwrap_or((x, 1));
            if let Some(scale) = previous.checked_mul(n) { scaled.insert(s.stmt_id, (root, scale)); }
            continue;
        }
        if matches!(s.ir, IR::MulI) { continue }
        let (left, ls) = scaled.get(a).copied().unwrap_or((*a, 1));
        let (right, rs) = scaled.get(b).copied().unwrap_or((*b, 1));
        let Some(scale) = ls.checked_mul(rs) else { continue };
        // Constants already fold naturally in Noir; only shared nonlinear
        // input products need an explicit common binding.
        if matches!(ir.stmts[left as usize].ir, IR::ConstantFloat { .. } | IR::ConstantInt { .. })
            || matches!(ir.stmts[right as usize].ir, IR::ConstantFloat { .. } | IR::ConstantInt { .. }) { continue }
        let lv = ir.stmts[left as usize].value_id;
        let rv = ir.stmts[right as usize].value_id;
        let (left, right) = if lv.0 <= rv.0 { (lv, rv) } else { (rv, lv) };
        *counts.entry((left, right)).or_default() += 1;
        products.insert(s.stmt_id, Product { left, right, scale });
    }
    products.retain(|_, p| counts[&(p.left, p.right)] > 1);
    products
}

impl NoirEmitter<'_> {
    pub(super) fn shared_raw_product(&mut self) -> Option<String> {
        let p = *self.product_plan.get(&self.current_stmt?)?;
        let key = (p.left, p.right);
        let raw = if let Some(raw) = self.raw_products.get(&key) { raw.clone() }
            else {
                let a = self.var_of_vid.get(&p.left)?.clone();
                let b = self.var_of_vid.get(&p.right)?.clone();
                let raw = self.bind(&format!("{a} * {b}"));
                self.raw_products.insert(key, raw.clone());
                raw
            };
        self.use_lib("qscale_down_raw");
        let term = if p.scale == 1 { raw } else { format!("{} * {raw}", signed_to_field_decimal(p.scale)) };
        Some(self.bind(&format!("qscale_down_raw({term})")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::IRStatement, ir_defs::IR, types::ValueId};

    #[test]
    fn parity_emitter_scaled_products_share_raw_but_keep_both_roundings() {
        let specs = vec![
            (IR::ReadFloat { path: InputPath::new("x", vec![]), is_public: false }, vec![]),
            (IR::ConstantFloat { value: 3.0 }, vec![]),
            (IR::MulF, vec![0, 1]),
            (IR::MulF, vec![2, 0]),
            (IR::MulF, vec![0, 0]),
        ];
        let ir = IRGraph::new(specs.into_iter().enumerate().map(|(i, (ir, args))| {
            let values = args.iter().map(|a| ValueId(*a as u64)).collect();
            IRStatement::new(i as u32, ValueId(i as u64), ir, args, values, None)
        }).collect());
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &prov);
        em.product_plan = plan(&ir);
        assert_eq!(em.product_plan.len(), 2);
        em.var_of_vid.insert(ValueId(0), "x".into());
        em.current_stmt = Some(3);
        assert!(em.shared_raw_product().is_some());
        em.current_stmt = Some(4);
        assert!(em.shared_raw_product().is_some());
        assert_eq!(em.body.matches("x * x").count(), 1);
        assert_eq!(em.body.matches("qscale_down_raw(").count(), 2);
        assert!(em.body.contains("qscale_down_raw(3 *"));
        // Frontend exact-integer scaling lowers integral float products to
        // MulI(raw, ConstantInt); this must preserve the same raw factor.
        let mut integer_scaled = IRGraph::new(ir.stmts.clone());
        integer_scaled.stmts[1].ir = IR::ConstantInt { value: 3 };
        integer_scaled.stmts[2].ir = IR::MulI;
        let integer_plan = plan(&integer_scaled);
        assert_eq!(integer_plan.len(), 2);
        assert_eq!(integer_plan[&3].left, ValueId(0));
        assert_eq!(integer_plan[&3].right, ValueId(0));
        assert_eq!(integer_plan[&3].scale, 3);
        assert_eq!(integer_plan[&4].scale, 1);
    }
}

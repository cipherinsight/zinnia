//! Programs for the memory family's tests and the backends' tests of the
//! memories the layer emits.

/// Two arrays the program writes at runtime positions and reads back, as a
/// version tree each (`crate::ir_array`), lowered with `constructions` on
/// `target` (the RAM constructions put both memories in one halo2-base
/// trace). Array `a` (`n` cells `x_k`) gets writes `v_t` at `a_t`, the
/// second predicated by `p`; array `b` (`m` cells `y_k`) one write `w` at
/// `c`; then reads of `a` at `r0`, `r1` and of `b` at `s` (exposed). Every
/// position is checked by the program (`discharge_index_in_range`).
pub(crate) fn two_memories(n: usize, m: usize, target: crate::lower::Target, constructions: &[&dyn crate::lower::Construction]) -> crate::ir::IRGraph {
    use crate::builder::IRBuilder;
    use crate::circuit_input::InputPath;
    use crate::ir_array::{AccessAxis, ArrayAccess, ArrayElem};
    let mut b = IRBuilder::new();
    let mut input = |b: &mut IRBuilder, name: &str| b.ir_read_integer(InputPath::new(name, vec![]), false);
    let flat = |len: usize| ArrayAccess::new(len as u32, ArrayElem::Int, 0, vec![AccessAxis { len: len as u32, stride: 1 }]);
    let checked = |b: &mut IRBuilder, x: &crate::types::Value, len: usize| {
        crate::optim::resolver::discharge_index_in_range(b, x, 0, len as i64, "test")
    };
    let xs: Vec<_> = (0..n).map(|k| input(&mut b, &format!("x{k}"))).collect();
    let ys: Vec<_> = (0..m).map(|k| input(&mut b, &format!("y{k}"))).collect();
    let yes = b.ir_constant_bool(true);
    let p = input(&mut b, "p");
    let mut a = b.ir_array_init(&xs, ArrayElem::Int);
    for (t, predicate) in [(0, &yes), (1, &p)] {
        let (at, v) = (input(&mut b, &format!("a{t}")), input(&mut b, &format!("v{t}")));
        let at = checked(&mut b, &at, n);
        a = b.ir_array_write(&a, flat(n), &[at], &v, predicate);
    }
    let mut bv = b.ir_array_init(&ys, ArrayElem::Int);
    let (c, w) = (input(&mut b, "c"), input(&mut b, "w"));
    let c = checked(&mut b, &c, m);
    bv = b.ir_array_write(&bv, flat(m), &[c], &w, &yes);
    for r in ["r0", "r1"] {
        let at = input(&mut b, r);
        let at = checked(&mut b, &at, n);
        let read = b.ir_array_read(&a, flat(n), &[at]);
        b.ir_expose_public_i(&read);
    }
    let s = input(&mut b, "s");
    let s = checked(&mut b, &s, m);
    let read = b.ir_array_read(&bv, flat(m), &[s]);
    b.ir_expose_public_i(&read);
    let graph = b.export_ir_graph();
    let lowered = crate::lower::emit::lower(&graph, target, constructions).expect("a lowering");
    crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, lowered)
}


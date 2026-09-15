//! The two worked examples of `definition.md` §8.
//!
//! §8.1 is reproduced by calling the transfer rules by hand in sequence,
//! because the point of that section is the rule sequence itself: the test
//! asserts each descriptor and each constraint the table shows.
//!
//! §8.2 is about control flow — `S-For-Unrolled`, `S-If`, the merge of §6.1
//! — which is [`crate::phase0::walk`]'s work, so it is built as the AST the
//! front end would export and run through [`crate::phase0::analyze`]. A
//! hand-threaded second walker here could drift from `walk/` without
//! failing anything.

use serde_json::Value;

use super::fixtures::*;
use super::Fixture;
use crate::phase0::desc::{Desc, Dtype, Origin};
use crate::phase0::phi::VarOrigin;
use crate::phase0::rules::{e_filter_1d, e_reduce_scalar, e_slice, e_zip, ReduceKind};

/// `definition.md` §8.1, the program of README 2.3:
///
/// ```python
/// s = x[mask][:K]
/// p = sum32(s)
/// q = sum32((s - p) * (s - p))
/// ```
///
/// $S$ gives `N` and `K`, so both are numerals (§1: "the only numerals in
/// its output come from $S$").
#[test]
fn example_8_1_masked_prefix() {
    const N: i64 = 8;
    const K: i64 = 3;
    let mut f = Fixture::new();
    let x = f.input(Dtype::I64, &[N]);
    let mask = f.input(Dtype::Bool, &[N]);

    // x[mask]: E-Filter-1D. defcond is N = N, which simplifies to true.
    let mask_key = f.key();
    let filtered = e_filter_1d(&mut f.ctx(), &x, &mask, &mask_key, "c").unwrap();
    let filtered = f.array_of(&filtered);
    assert_eq!(f.pretty(filtered.dims[0]), "c");

    // [:K]: E-Slice. L = norm(0, c) = 0, U = norm(K, c) = max(0, min(c, K)).
    let k = f.literal(K);
    let s = e_slice(&mut f.ctx(), &filtered, 0, None, Some(&k), 1).unwrap();
    let s = f.array_of(&s);
    // §8.1 writes this length as min(c, K); phase 0 keeps the outer max(0, ·)
    // because it applies no range-aware simplification (see the Deviations
    // section of implementation.md). The two are equal under 0 <= c.
    assert_eq!(f.pretty(s.dims[0]), "max(0, min(c, 3))");

    // p = sum32(s): E-Reduce-Scalar, a data scalar.
    let p_key = f.key();
    let p = e_reduce_scalar(&mut f.ctx(), &s, ReduceKind::Sum, &p_key, "v_p").unwrap();
    let p = f.scalar_of(&p);
    assert_eq!(p.origin, Origin::Data);

    // s - p and (s - p) * (s - p): E-Zip with scalar broadcasting.
    let s_desc = Desc::Array(s.clone());
    let diff = e_zip(&mut f.ctx(), &s_desc, &Desc::Scalar(p), Dtype::I64).unwrap();
    let square = e_zip(&mut f.ctx(), &diff, &diff, Dtype::I64).unwrap();
    let square_arr = f.array_of(&square);
    assert_eq!(
        square_arr.dims, s.dims,
        "scalar broadcasting keeps the extent"
    );

    // q = sum32(...): another data variable.
    let q_key = f.key();
    let q = e_reduce_scalar(&mut f.ctx(), &square_arr, ReduceKind::Sum, &q_key, "v_q").unwrap();
    assert_eq!(f.scalar_of(&q).origin, Origin::Data);

    assert_phi_of_example_8_1(&f, N);
}

/// "$\Phi$ ends with one variable $c$ and one constraint $0 \le c \le N$"
/// (§8.1). The data variables $v_p$ and $v_q$ carry only their dtype range,
/// which is what lets the section leave them out of that sentence.
fn assert_phi_of_example_8_1(f: &Fixture, n: i64) {
    let counts: Vec<_> = f
        .analysis
        .phi
        .vars()
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v.origin, VarOrigin::Count { .. }))
        .collect();
    assert_eq!(counts.len(), 1);
    let (c_id, c_info) = counts[0];
    assert_eq!(
        f.analysis
            .terms
            .var_name(crate::phase0::term::VarId(c_id as u32)),
        "c"
    );
    assert_eq!(f.pretty(c_info.lo.unwrap()), "0");
    assert_eq!(f.pretty(c_info.hi.unwrap()), n.to_string());
    // Every defcond in this program simplifies to true.
    assert!(f.checks().is_empty(), "unexpected checks: {:?}", f.checks());
}

/// `definition.md` §8.2, the control-flow program of archive §11:
///
/// ```python
/// acc = i32(0)
/// for t in range(T):
///     if enabled[t]:
///         B = X[t, :, None] + W[None, :]
///         C = B[:, 1::2]
///         r = sum32(C, axis=1)
///         f = r[mask[t]]
///         s = f[lo[t]:hi[t]]
///         p = sum32(s)
///         if p > threshold[t]:
///             d = s - p
///             q = sum32(d * d)
///             acc = acc + q
///         else:
///             acc = acc + p
/// return acc
/// ```
///
/// Unlike §8.1, this example is about *control flow*, and the walker is what
/// implements it (`S-For-Unrolled`, `S-If`, the merge of §6.1). So the
/// program is built as the AST the front end would export and run through
/// [`analyze`]; a second, hand-threaded walker here could drift from
/// `walk/` without failing.
///
/// `T`, `N` and `M` come from $S$, so they are numerals. `T` is kept small:
/// the section's claims are per iteration, and two iterations already show
/// that the count variables do not merge (§6.3).
const T82: usize = 2;
const N82: usize = 4;
const M82: usize = 5;

/// The one departure from §8.2's declarations: the annotation language has
/// no Boolean array — the front end aliases `Boolean` to `Integer`
/// (`coverage.md` §2) — so `mask` is declared integer and the program takes
/// `mask[t] > 0`, which is the Boolean array of the same shape by `E-Zip`.
fn program_8_2() -> (Vec<Value>, Vec<Value>) {
    let inputs = vec![
        array_input("X", &[T82, N82]),
        array_input("W", &[M82]),
        array_input("mask", &[T82, N82]),
        array_input("enabled", &[T82]),
        array_input("lo", &[T82]),
        array_input("hi", &[T82]),
        array_input("threshold", &[T82]),
    ];
    let full = || slice_range(None, None, None);
    let at_t = || load("t");
    // B = X[t, :, None] + W[None, :]
    let x_col = subscript_at(load("X"), vec![at_t(), full(), new_axis()]);
    let w_row = subscript_at(load("W"), vec![new_axis(), full()]);
    let b = assign(name_target("B"), binop("add", x_col, w_row));
    // C = B[:, 1::2]
    let c = assign(
        name_target("C"),
        subscript_at(
            load("B"),
            vec![full(), slice_range(Some(int(1)), None, Some(int(2)))],
        ),
    );
    let r = assign(
        name_target("r"),
        call_kw(Some("np"), "sum", vec![load("C")], vec![("axis", int(1))]),
    );
    // f = r[mask[t] > 0]
    let mask_row = binop("gt", subscript(load("mask"), at_t()), int(0));
    let filtered = assign(name_target("f"), subscript(load("r"), mask_row));
    // s = f[lo[t]:hi[t]]
    let s = assign(
        name_target("s"),
        subscript_at(
            load("f"),
            vec![slice_range(
                Some(subscript(load("lo"), at_t())),
                Some(subscript(load("hi"), at_t())),
                None,
            )],
        ),
    );
    let p = assign(name_target("p"), call(Some("np"), "sum", vec![load("s")]));
    let then_block = vec![
        assign(name_target("d"), binop("sub", load("s"), load("p"))),
        assign(
            name_target("q"),
            call(Some("np"), "sum", vec![binop("mul", load("d"), load("d"))]),
        ),
        assign(name_target("acc"), binop("add", load("acc"), load("q"))),
    ];
    let else_block = vec![assign(
        name_target("acc"),
        binop("add", load("acc"), load("p")),
    )];
    let inner = cond(
        binop("gt", load("p"), subscript(load("threshold"), at_t())),
        then_block,
        else_block,
    );
    let body = vec![cond(
        subscript(load("enabled"), at_t()),
        vec![b, c, r, filtered, s, p, inner],
        vec![],
    )];
    let block = vec![
        assign(name_target("acc"), call(None, "int", vec![int(0)])),
        for_in("t", call(None, "range", vec![int(T82 as i64)]), body),
        ret(load("acc")),
    ];
    (inputs, block)
}

/// The closing paragraph of §8.2: one count variable per iteration, each
/// ranged `0 ≤ c_t ≤ N`; a data variable per iteration for `v^lo_t`,
/// `v^hi_t`, `β^e_t`, `β^p_t` and `v^p_t`; and "in this program every
/// defcond simplifies to true", so `Φ` holds no check at all.
#[test]
fn example_8_2_control_flow_program() {
    let (inputs, block) = program_8_2();
    let result = run(inputs, block).expect("§8.2 is inside §5 and §6");

    let counts: Vec<&crate::phase0::phi::VarInfo> = result
        .phi
        .vars()
        .iter()
        .filter(|v| matches!(v.origin, VarOrigin::Count { .. }))
        .collect();
    assert_eq!(counts.len(), T82, "one count variable per iteration");
    for info in counts {
        assert_eq!(result.terms.constant_value(info.lo.unwrap()), Some(0));
        assert_eq!(
            result.terms.constant_value(info.hi.unwrap()),
            Some(N82 as i128),
            "0 <= c_t <= N"
        );
    }

    // `v^lo_t`, `v^hi_t`, `v^p_t` and the merge variables of `acc` are data
    // scalars; `β^e_t` and `β^p_t` are data Booleans. Two of each per
    // iteration is the section's list, before the merges.
    let data_bools = result
        .phi
        .vars()
        .iter()
        .filter(|v| matches!(v.origin, VarOrigin::DataBool { .. }))
        .count();
    assert!(
        data_bools >= 2 * T82,
        "at least β^e_t and β^p_t per iteration, got {}",
        data_bools
    );

    assert!(
        check_provenances(&result).is_empty(),
        "every defcond of §8.2 simplifies to true; got {:?}",
        check_provenances(&result)
    );

    // The delivered `acc` is the merged data scalar of the outer `if`.
    let delivered = delivered(&result).as_scalar().expect("a scalar");
    assert_eq!(delivered.origin, Origin::Data);
}

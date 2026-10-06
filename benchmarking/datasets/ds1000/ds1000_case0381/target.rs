//! DS-1000 case 0381, halo2-base baseline.
//! Kernel: `np.argsort(a)[::-1][:len(a)]` on `Float[100]`: the ascending
//! argsort reversed (`[:len(a)]` keeps all 100).
//! A sorting network: Batcher's merge exchange (Knuth, TAOCP 5.2.2, Algorithm
//! M), 1,077 compare-exchanges for 100 keys (an adjacent-swap network needs
//! 4,950). Element `i` travels as the single key `128 * a_i + 2^70 + i`; with
//! `a_i` bounded to the window the keys are distinct integers in `[0, 2^71)`
//! ordered by `(a_i, i)`, so each compare-exchange is one 71-bit comparison
//! and the result is the stable order: equal values keep their index order,
//! NumPy's `kind="stable"` (NumPy's default introsort leaves the order of
//! equal values unspecified); reversed, equal values come out in decreasing
//! index order. The index of a sorted key is its remainder mod
//! 128, split off by `floor_div_const` (quotient pinned to the window).

use crate::prelude::*;

const N: usize = 100;
/// Shifts `128 * a_i` (`a_i` in the window) into `[0, 2^71)`.
const KEY_SHIFT: i128 = 1 << 70;
const KEY_BITS: usize = 71;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a); // each a_i in the window, so each key is an integer
    let mut order = argsort(&q, ctx, &a);
    order.reverse();
    out.ints(&order);
    Ok(())
}

/// The stable ascending argsort of `a`.
fn argsort(q: &Q32, ctx: &mut Ctx, a: &[Cell]) -> Vec<Cell> {
    let mut keys: Vec<Cell> = a
        .iter()
        .enumerate()
        .map(|(i, x)| {
            q.gate.mul_add(ctx, *x, Constant(fe(128)), Constant(fe(KEY_SHIFT + i as i128)))
        })
        .collect();
    for (i, j) in merge_exchange(a.len()) {
        compare_exchange(q, ctx, &mut keys, i, j);
    }
    keys.iter()
        .map(|key| {
            let shifted = q.sub(ctx, *key, Constant(fe(KEY_SHIFT)));
            q.floor_div_const(ctx, shifted, 128).1
        })
        .collect()
}

/// Puts the smaller of `keys[i]`, `keys[j]` (`i < j`) at `i`, the larger at `j`.
fn compare_exchange(q: &Q32, ctx: &mut Ctx, keys: &mut [Cell], i: usize, j: usize) {
    let (lo, hi) = (keys[i], keys[j]);
    let swap = q.range.is_less_than(ctx, hi, lo, KEY_BITS);
    let diff = q.sub(ctx, hi, lo);
    keys[i] = q.gate.mul_add(ctx, swap, diff, lo);
    keys[j] = q.gate.sub_mul(ctx, hi, swap, diff);
}

/// Batcher's merge exchange for `n` keys: the compare-exchange pairs
/// `(i, j)`, `i < j`, in execution order.
fn merge_exchange(n: usize) -> Vec<(usize, usize)> {
    let t = usize::BITS - (n - 1).leading_zeros(); // ceil(log2 n)
    let mut pairs = Vec::new();
    let mut p = 1usize << (t - 1);
    while p > 0 {
        let (mut q, mut r, mut d) = (1usize << (t - 1), 0, p);
        loop {
            pairs.extend((0..n - d).filter(|i| i & p == r).map(|i| (i, i + d)));
            if q == p {
                break;
            }
            d = q - p;
            q >>= 1;
            r = p;
        }
        p >>= 1;
    }
    pairs
}

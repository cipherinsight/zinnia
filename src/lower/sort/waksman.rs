//! The arbitrary-size Waksman permutation network (Waksman 1968; Beauquier
//! and Darrot 2002 for sizes that are not powers of two).
//!
//! A *switch* takes two wires `(x, y)` and outputs `(x, y)` (straight) or
//! `(y, x)` (crossed). The network on `n` wires is built recursively:
//!
//! * `n < 2`: no switch; `n = 2`: one switch.
//! * `n >= 3`, `h = floor(n / 2)`: an input switch on each pair
//!   `(2p, 2p + 1)`, `p < h`, sends its top output to the upper subnetwork
//!   (size `h`) and its bottom output to the lower one (size `n - h`); for
//!   odd `n` the last wire goes straight to the lower subnetwork. An output
//!   switch on each pair `p < h` takes output `p` of both subnetworks and
//!   yields outputs `(2p, 2p + 1)`; for odd `n` output `n - 1` is the lower
//!   subnetwork's last output, and for even `n` the last output switch is
//!   left out (fixed straight).
//!
//! It has `W(n) = W(floor(n/2)) + W(ceil(n/2)) + n - 1` switches (573 for
//! `n = 100`, 9,217 for 1,024), and **every setting of its switches realizes
//! a permutation** (each switch is a bijection on its two wires), while every
//! permutation is realized by some setting (`route`, the looping algorithm).
//! A sorting relation then needs no comparator network: the prover supplies
//! the settings, and the circuit checks that the routed keys increase.
//!
//! The settings of one permutation are not always unique (for `n = 4` the
//! network has 5 switches, 32 settings, 24 permutations): the output of the
//! network is pinned by the checks that follow it, the switch bits are not.

/// The number of switches of the network on `n` wires.
pub fn switch_count(n: usize) -> usize {
    match n {
        0 | 1 => 0,
        2 => 1,
        _ => switch_count(n / 2) + switch_count(n - n / 2) + n - 1,
    }
}

/// The network applied to `wires`: `switch(x, y)` is called once per switch,
/// in the canonical order (input switches by pair, the upper subnetwork, the
/// lower subnetwork, output switches by pair), and returns `(top, bottom)`.
pub fn apply<T>(wires: Vec<T>, switch: &mut dyn FnMut(T, T) -> (T, T)) -> Vec<T> {
    let n = wires.len();
    if n < 2 {
        return wires;
    }
    let mut wires = wires.into_iter();
    if n == 2 {
        let (x, y) = (wires.next().expect("two"), wires.next().expect("two"));
        let (top, bottom) = switch(x, y);
        return vec![top, bottom];
    }
    let h = n / 2;
    let odd = n % 2 == 1;
    let mut upper = Vec::with_capacity(h);
    let mut lower = Vec::with_capacity(n - h);
    for _ in 0..h {
        let (x, y) = (wires.next().expect("pair"), wires.next().expect("pair"));
        let (top, bottom) = switch(x, y);
        upper.push(top);
        lower.push(bottom);
    }
    if odd {
        lower.push(wires.next().expect("the unpaired wire"));
    }
    let upper = apply(upper, switch);
    let mut lower = apply(lower, switch).into_iter();
    let mut out = Vec::with_capacity(n);
    for (p, u) in upper.into_iter().enumerate() {
        let l = lower.next().expect("as many as the upper outputs");
        if !odd && p + 1 == h {
            out.push(u);
            out.push(l);
        } else {
            let (top, bottom) = switch(u, l);
            out.push(top);
            out.push(bottom);
        }
    }
    if odd {
        out.push(lower.next().expect("the unpaired output"));
    }
    out
}

/// Switch settings (`true` = crossed), in `apply`'s order, under which input
/// wire `i` comes out at position `dest[i]`; `dest` must be a permutation.
pub fn route(dest: &[usize]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(switch_count(dest.len()));
    route_into(dest, &mut bits);
    bits
}

fn route_into(dest: &[usize], bits: &mut Vec<bool>) {
    let n = dest.len();
    if n < 2 {
        return;
    }
    if n == 2 {
        bits.push(dest[0] == 1);
        return;
    }
    let h = n / 2;
    let odd = n % 2 == 1;
    let mut src = vec![0; n];
    for (i, &d) in dest.iter().enumerate() {
        src[d] = i;
    }
    // Two-colour the inputs: `true` = through the lower subnetwork. The two
    // inputs of an input pair, and the two inputs bound for an output pair,
    // take different subnetworks. Each input has at most one partner of each
    // kind, so the constraints form paths and even cycles. Forced: for odd
    // `n` the unpaired input and the input bound for the unpaired output
    // (the two ends of the one path, an even number of edges apart); for
    // even `n` the input bound for output `n - 1` (the fixed last switch).
    let mut lower: Vec<Option<bool>> = vec![None; n];
    let in_partner = |i: usize| (i < 2 * h).then_some(i ^ 1);
    let out_partner = |i: usize| (dest[i] < 2 * h).then(|| src[dest[i] ^ 1]);
    let forced: Vec<usize> = if odd { vec![n - 1, src[n - 1]] } else { vec![src[n - 1]] };
    let seeds = forced.into_iter().map(|i| (i, true)).chain((0..n).map(|i| (i, false)));
    for (seed, colour) in seeds {
        if lower[seed].is_some() {
            continue;
        }
        let mut stack = vec![(seed, colour)];
        while let Some((i, c)) = stack.pop() {
            match lower[i] {
                Some(existing) => assert_eq!(existing, c, "Waksman routing: inconsistent colouring"),
                None => {
                    lower[i] = Some(c);
                    for partner in [in_partner(i), out_partner(i)].into_iter().flatten() {
                        if lower[partner].is_none() {
                            stack.push((partner, !c));
                        }
                    }
                }
            }
        }
    }
    let lower: Vec<bool> = lower.into_iter().map(|c| c.expect("every input coloured")).collect();
    let mut dest_upper = Vec::with_capacity(h);
    let mut dest_lower = Vec::with_capacity(n - h);
    for p in 0..h {
        let crossed = lower[2 * p];
        bits.push(crossed);
        let (top, bottom) = if crossed { (2 * p + 1, 2 * p) } else { (2 * p, 2 * p + 1) };
        dest_upper.push(dest[top] / 2);
        dest_lower.push(dest[bottom] / 2);
    }
    if odd {
        dest_lower.push(dest[n - 1] / 2);
    }
    route_into(&dest_upper, bits);
    route_into(&dest_lower, bits);
    for p in 0..h {
        if !odd && p + 1 == h {
            continue;
        }
        // Straight when output 2p comes from the upper subnetwork.
        bits.push(lower[src[2 * p]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routed(dest: &[usize]) -> Vec<usize> {
        let bits = route(dest);
        assert_eq!(bits.len(), switch_count(dest.len()));
        let mut next = bits.into_iter();
        let out = apply((0..dest.len()).collect(), &mut |x, y| {
            if next.next().expect("a bit per switch") { (y, x) } else { (x, y) }
        });
        assert!(next.next().is_none(), "every bit used");
        out
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![Vec::new()];
        }
        let mut all = Vec::new();
        for p in permutations(n - 1) {
            for k in 0..n {
                let mut q = p.clone();
                q.insert(k, n - 1);
                all.push(q);
            }
        }
        all
    }

    #[test]
    fn switch_counts() {
        assert_eq!(switch_count(100), 573);
        assert_eq!(switch_count(1024), 9217);
        for n in 2..300 {
            let mut count = 0;
            let _ = apply((0..n).collect::<Vec<_>>(), &mut |x, y| {
                count += 1;
                (x, y)
            });
            assert_eq!(count, switch_count(n), "n = {n}");
        }
    }

    #[test]
    fn every_permutation_is_routed_exhaustively_up_to_8() {
        for n in 0..=8 {
            for dest in permutations(n) {
                let out = routed(&dest);
                for (i, &d) in dest.iter().enumerate() {
                    assert_eq!(out[d], i, "dest {dest:?}");
                }
            }
        }
    }

    #[test]
    fn random_permutations_are_routed_up_to_600() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for n in (9..600).step_by(7) {
            for _ in 0..4 {
                let mut dest: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = (rand() % (i as u64 + 1)) as usize;
                    dest.swap(i, j);
                }
                let out = routed(&dest);
                for (i, &d) in dest.iter().enumerate() {
                    assert_eq!(out[d], i);
                }
            }
        }
    }

    #[test]
    fn every_setting_is_a_permutation() {
        for n in 2..=7 {
            let w = switch_count(n);
            for setting in 0u32..(1 << w) {
                let mut k = 0;
                let mut out = apply((0..n).collect::<Vec<_>>(), &mut |x, y| {
                    let crossed = setting >> k & 1 == 1;
                    k += 1;
                    if crossed { (y, x) } else { (x, y) }
                });
                out.sort();
                assert_eq!(out, (0..n).collect::<Vec<_>>());
            }
        }
    }
}

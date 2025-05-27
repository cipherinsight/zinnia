//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);

use alloy_sol_types::SolType;
use fibonacci_lib::PublicValuesStruct;
use light_poseidon::{Poseidon, PoseidonBytesHasher, parameters::bn254_x5};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use ethereum_types::U512;

// source start
fn extended_gcd(a: U512, b: U512, p: U512) -> (U512, U512, U512) {
    if a == U512::from(0) {
        (b, U512::from(0), U512::from(1))
    } else {
        let (g, y, x) = extended_gcd(b % a, a, p);
        // println!("Backtrack! g={:?}, y={:?}, x={:?}, a={:?}, b={:?}", g, y, x, a, b);
        // println!("step 1!! {:?}", (b / a) * y);
        (g, (x + (p - ((((b / a) % p) * y) % p)) % p) % p, y)
    }
}

fn modular_inverse(a: U512, p: U512) -> U512 {
    let (gcd, x, _) = extended_gcd(a % p, p, p);
    assert!(gcd == U512::from(1));
    return (x % p + p) % p;
}

pub fn main() {
    // Read an input to the program.
    //
    // Behind the scenes, this compiles down to a custom system call which handles reading inputs
    // from the prover.

    // u256 max = 115792089237316195423570985008687907853269984665640564039457584007913129639935
    // x1       = 995203441582195749578291179787384436505546430278305826713579947235728471134
    // y1       = 5472060717959818805561601436314318772137091100104008585924551046643952123905
    // x2       = 5299619240641551281634865583518297030282874472190772894086521144482721001553
    // y2       = 16950150798460657717958625567821834550301663161624707787222815936182638968203
    // x3       = 14805543388578810117460687107379140748822348273316260688573060998934016770136
    // y3       = 13589798946988221969763682225123791336245855044059976312385135587934609470572

    let x1 = U512([sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>()]);
    let y1 = U512([sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>()]);
    let x2 = U512([sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>()]);
    let y2 = U512([sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>()]);
    let x3 = U512([sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>()]);
    let y3 = U512([sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>(), sp1_zkvm::io::read::<u64>()]);

    let P = U512::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    let a = U512::from(168700);
    let neg_a: U512 = P - a;
    let d = U512::from(168696);
    let one = U512::from(1);

    // point 1 should on the curve
    let x1_square = (x1 * x1) % P;
    let y1_square = (y1 * y1) % P;
    let left= (((a * x1_square) % P) + y1_square) % P;
    let right = one + (d * ((x1_square * y1_square) % P)) % P;
    assert_eq!(left % P, right % P);

    // point 2 should on the curve
    let x2_square = (x2 * x2) % P;
    let y2_square = (y2 * y2) % P;
    let left= (((a * x2_square) % P) + y2_square) % P;
    let right = one + (d * ((x2_square * y2_square) % P)) % P;
    assert_eq!(left % P, right % P);

    // point 3 should on the curve
    let x3_square = (x3 * x3) % P;
    let y3_square = (y3 * y3) % P;
    let left= (((a * x3_square) % P) + y3_square) % P;
    let right = one + (d * ((x3_square * y3_square) % P)) % P;
    assert_eq!(left % P, right % P);

    // add p1 and p2 together
    let beta = (x1 * y2) % P;
    let gamma = (y1 * x2) % P;
    let delta = (((((neg_a * x1) % P) + y1) % P) * ((x2 + y2) % P)) % P;
    let tau = (beta * gamma) % P;
    let tmp = (one + ((d * tau) % P)) % P;
    let x4 = ((beta + gamma) % P) * modular_inverse(tmp, P);
    let y4 = ((((delta + ((a * beta) % P)) % P) + (P - gamma) % P) % P) * modular_inverse((one + (P - ((d * tau) % P)) % P) % P, P);

    // verify equality
    assert_eq!(x4 % P, x3);
    assert_eq!(y4 % P, y3);

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
